"""SparseRouter: 稀疏连接 router + 竞争 Hebbian 学习。

核心机制:
  1. forward 中 connect_mask 让 expert 只看到输入的一个子空间
     → 天然专精（不是被 router 分配的, 是连接结构决定的）
  2. 梯度只流向 connect_mask=1 的维度
     → Hebbian 强化是自动的、稀疏的 (selected → strengthen on support)
  3. hebbian_step(): 未被选中的 expert 连接衰减 + 侧抑制 + 连接修剪
     → 用进废退, 主动推离
  4. expand(): 新 expert 继承源 expert 的连接分布 (结构化试探)
     → 继承部分连接, 随机扰动, 可能桥接不同子空间
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.router import Router

_logger = logging.getLogger(__name__)


class SparseRouter(Router):
    """Router with per-expert sparse connectivity + Hebbian learning.

    继承 Router 全部功能 (forward/top-k/bias/prune/expand/restore),
    增加:
      connect_mask: (n_total, d_model) buffer, 1=连接, 0=断开
      inactive_steps: (n_total,) buffer, 未被选中的连续步数
      hebbian_step(): 训练循环中调用, 衰减/抑制/修剪
      expand(): 覆盖父类, 新 expert 使用结构化稀疏初始化
      remove_row(): 覆盖父类, 同步移除 connect_mask/inactive_steps 行
      restore(): 覆盖父类, 恢复时重新随机稀疏
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,
        top_k: int = 2,
        sparsity: float = 0.3,
        hebbian_decay: float = 0.001,
        inhibition_threshold: float = 0.8,
        inhibition_beta: float = 0.01,
        prune_threshold: float = 0.01,
        prune_inactive: int = 1000,
        growth_prob: float = 0.05,
        growth_interval: int = 50,
        exploration_rate: float = 0.05,     # v2.9: ε-greedy 探索概率
        exploration_decay: float = 0.999,    # 探索衰减 (每步)
        adaptive_threshold: float = 0.15,    # v2.9: 自适应激活阈值
    ):
        """
        Args:
            sparsity: 初始连接稀疏度 (0.3 = 30% 维度连接)
            hebbian_decay: 未被选中时连接衰减速率
            inhibition_threshold: 侧抑制触发的余弦重叠阈值
            inhibition_beta: 侧抑制步长
            prune_threshold: 连接权重低于此值 + 长期不活跃 → 从 mask 剪除
            prune_inactive: 不活跃步数阈值
            growth_prob: 连接生长概率 (每个断开维度每 growth_interval 步)
            growth_interval: 连接生长间隔 (步)
        """
        super().__init__(d_model, n_total, top_k)
        self.sparsity = sparsity
        self.hebbian_decay = hebbian_decay
        self.inhibition_threshold = inhibition_threshold
        self.inhibition_beta = inhibition_beta
        self.prune_threshold = prune_threshold
        self.prune_inactive = prune_inactive
        self.growth_prob = growth_prob
        self.growth_interval = growth_interval
        self.exploration_rate = exploration_rate         # v2.9
        self.exploration_decay = exploration_decay       # v2.9
        self.adaptive_threshold = adaptive_threshold     # v2.9

        # 连接掩码: (n_total, d_model), 1=连接
        self.register_buffer("connect_mask", torch.ones(n_total, d_model))
        # 未被选中的连续步数: (n_total,)
        self.register_buffer("inactive_steps", torch.zeros(n_total))
        # 初始化稀疏掩码
        self._init_sparse_masks()

    @property
    def d_model(self) -> int:
        return self.gate_weight.shape[1]

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_sparse_masks(self) -> None:
        """为每个 routed expert 随机生成稀疏连接掩码 (非重叠划分)。

        单向导通约束: 每个 input 维度最多被一个 expert 连接,
        避免多 expert 竞争同一维度导致的反馈循环。
        """
        d = self.d_model
        n_routed = self.n_total - 1
        # 每个 expert 连接 sparsity * d 个维度 (非重叠约束下)
        k_per_expert = min(int(d * self.sparsity), d // n_routed)
        k_per_expert = max(1, k_per_expert)
        
        # 随机划分维度给各 expert (无重叠)
        perm = torch.randperm(d, device=self.connect_mask.device)
        for idx, j in enumerate(range(1, self.n_total)):
            start = idx * k_per_expert
            end = min(start + k_per_expert, d)
            mask = torch.zeros(d, device=self.connect_mask.device)
            mask[perm[start:end]] = 1.0
            self.connect_mask.data[j] = mask
        
        # shared expert 连接未被 routed experts 占用的维度
        routed_mask = self.connect_mask[1:].sum(dim=0).clamp(0, 1)
        self.connect_mask.data[0] = 1.0 - routed_mask
        # 确保 shared 至少有一些连接
        if self.connect_mask[0].sum() < d * 0.1:
            n_extra = max(1, int(d * 0.1))
            free_dims = torch.where(self.connect_mask[0] == 0)[0]
            if len(free_dims) > 0:
                extra = free_dims[torch.randperm(len(free_dims))[:n_extra]]
                self.connect_mask.data[0, extra] = 1.0

    # ------------------------------------------------------------------
    # forward (覆盖: 应用 connect_mask)
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """前向传播。与 Router.forward 相同, 但 gate_weight 先经 connect_mask 过滤。"""
        masked_w = self.gate_weight * self.connect_mask
        logits_clean = F.linear(x, masked_w)  # (B, S, n_total)
        logits_clean = logits_clean + self.prune_mask

        z_loss = torch.logsumexp(logits_clean, dim=-1).pow(2).mean()

        logits = logits_clean + self.bias

        if self.training:
            noise = torch.randn_like(logits) * F.softplus(self.noise_std)
            finite = torch.isfinite(logits_clean)
            logits = logits + torch.where(finite, noise, torch.zeros_like(noise))

        routed_logits = logits[:, :, 1:]
        top_k_gate, top_k_indices = torch.topk(routed_logits, self.top_k, dim=-1)
        top_k_indices = top_k_indices + 1

        # v2.9: 自适应激活 — 第二个专家的概率低于阈值则屏蔽
        if self.top_k >= 2:
            raw_gate = F.softmax(top_k_gate, dim=-1)
            # raw_gate[..., 1] 是第二个专家的 softmax 权重
            mask = (raw_gate[..., 1] >= self.adaptive_threshold).float().unsqueeze(-1)
            # 不确定时保留第二个专家, 确定时只留第一个
            top_k_gate = raw_gate * torch.cat([torch.ones_like(raw_gate[..., :1]), mask], dim=-1)
        else:
            top_k_gate = F.softmax(top_k_gate, dim=-1)

        return top_k_gate, top_k_indices, z_loss, routed_logits

    # ------------------------------------------------------------------
    # expand (覆盖: 新 expert 结构化稀疏初始化)
    # ------------------------------------------------------------------

    def expand(self, source_idx: int, perturbation_std: float = 0.02,
               alpha: float | None = None) -> None:
        """新 expert 分裂: 继承源 expert 的连接分布 (结构化试探)。

        流程:
          1. 调用父类 expand (标准 gate_weight 行追加)
          2. 从源 expert 的权重分布采样支持集 S
             p_i ∝ |w_src[i]| / max|w_src| → 重要维度更可能被继承
             m_i ~ Bernoulli(alpha · p_i)
          3. 新 expert 的 gate_weight 只在 S 上保留值
          4. connect_mask 记录 S (用于后续梯度过滤)
          5. inactive_steps 追加一行

        Args:
            alpha: 稀疏度覆盖 (None → 使用 self.sparsity)
        """
        alpha = alpha if alpha is not None else self.sparsity
        w_src = self.gate_weight.data[source_idx]  # (d,)
        d = self.d_model

        # 单向导通: 新 expert 只能连接未被 ANY routed expert 占用的维度
        # (包括源 expert 的维度, 因为源已经"拥有"了那些维度)
        other_mask = self.connect_mask[1:].sum(dim=0).clamp(0, 1)
        available = (other_mask == 0)
        
        # 在可用维度中按源 expert 权重分布采样
        p = w_src.abs() / (w_src.abs().max() + 1e-8)
        p = p * available.float()  # 只采样可用维度
        
        n_available = int(available.sum().item())
        n_growth = max(2, int(d * alpha))
        n_growth = min(n_growth, n_available)
        
        if n_growth > 0 and p.sum() > 0:
            # 按权重分布采样
            p_norm = p / p.sum()
            growth_idx = torch.multinomial(p_norm, n_growth, replacement=False)
            m = torch.zeros(d, device=self.connect_mask.device)
            m[growth_idx] = 1.0
        elif n_available >= 2:
            # 随机采样
            avail_idx = torch.where(available)[0]
            chosen = avail_idx[torch.randperm(n_available)[:n_growth]]
            m = torch.zeros(d, device=self.connect_mask.device)
            m[chosen] = 1.0
        else:
            m = available.float()

        # 调用父类 expand (gate_weight/bias/prune_mask/noise_std 追加行)
        super().expand(source_idx, perturbation_std)

        new_idx = self.n_total - 1
        # 新 expert 的 gate_weight 只在支持集上保留值 (结构化试探)
        self.gate_weight.data[new_idx] *= m
        # connect_mask 追加新行
        self.connect_mask.data = torch.cat(
            [self.connect_mask.data, m.unsqueeze(0)], dim=0,
        )
        # inactive_steps 追加
        self.inactive_steps = torch.cat(
            [self.inactive_steps,
             torch.zeros(1, device=self.inactive_steps.device)],
        )

    def expand_bridge(self, src_a: int, src_b: int,
                      perturbation_std: float = 0.02,
                      alpha: float | None = None) -> None:
        """桥接生长: 新 expert 连接两个源 expert 的连接子集 (跨域)。

        支持集 = S_A ∪ S_B (两个源 expert 的支持集并集采样),
        用于覆盖已有 expert 之间的盲区。
        """
        alpha = alpha if alpha is not None else self.sparsity
        w_a = self.gate_weight.data[src_a]
        w_b = self.gate_weight.data[src_b]

        # 从两个源的连接分布采样
        p_a = w_a.abs() / (w_a.abs().max() + 1e-8)
        p_b = w_b.abs() / (w_b.abs().max() + 1e-8)
        p_bridge = torch.maximum(p_a, p_b)
        m = torch.bernoulli(alpha * p_bridge).to(self.connect_mask.device)
        if m.sum() < 2:
            m = (torch.rand(self.d_model, device=m.device) < alpha).float()

        # 用 src_a 做父类 expand
        super().expand(src_a, perturbation_std)

        new_idx = self.n_total - 1
        self.gate_weight.data[new_idx] *= m
        self.connect_mask.data = torch.cat(
            [self.connect_mask.data, m.unsqueeze(0)], dim=0,
        )
        self.inactive_steps = torch.cat(
            [self.inactive_steps,
             torch.zeros(1, device=self.inactive_steps.device)],
        )

    # ------------------------------------------------------------------
    # remove_row / restore (覆盖: 同步新 buffer)
    # ------------------------------------------------------------------

    def remove_row(self, idx: int) -> None:
        """物理删除一行, 同步 connect_mask/inactive_steps。"""
        super().remove_row(idx)
        keep = [i for i in range(self.connect_mask.shape[0]) if i != idx]
        self.connect_mask.data = self.connect_mask.data[keep].clone()
        self.inactive_steps = self.inactive_steps[keep].clone()

    def restore(self, idx: int) -> None:
        """恢复一行, 重新随机稀疏。"""
        super().restore(idx)
        d = self.d_model
        k = max(1, int(self.sparsity * d))
        mask_idx = torch.randperm(d, device=self.connect_mask.device)[:k]
        mask = torch.zeros(d, device=self.connect_mask.device)
        mask[mask_idx] = 1.0
        self.connect_mask.data[idx] = mask
        self.inactive_steps[idx] = 0.0

    # ------------------------------------------------------------------
    # Hebbian 训练步 (核心算法)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def hebbian_step(self, loads: list[float], step: int) -> None:
        """Hebbian 更新: 衰减未选中连接 + 侧抑制 + 连接修剪。

        在训练循环中调用 (trainer._train_step 之后)。
        Hebbian 强化已由反向传播自动完成 (梯度只流向 connect_mask=1 的维度)。

        Args:
            loads: (n_routed,) 各 routed expert 的负载 (load EMA)
            step: 当前训练步数
        """
        n = self.n_total
        decay = self.hebbian_decay

        # 1. 衰减: 未被选中的 expert 连接衰减
        for j in range(1, n):
            load = loads[j - 1] if j - 1 < len(loads) else 0.0
            if load < 1e-4:
                self.gate_weight.data[j] *= (1.0 - decay)
                self.inactive_steps[j] += 1
            else:
                self.inactive_steps[j] = 0

        # 2. 侧抑制: 余弦重叠超阈值时推离 (向量化, 只考虑活 expert)
        if step % 10 == 0:
            # 找出活 expert (prune_mask 有限)
            live_mask = torch.isfinite(self.prune_mask[1:])  # (n_routed,)
            live_idx = live_mask.nonzero(as_tuple=True)[0]
            if len(live_idx) >= 2:
                w_live = self.gate_weight.data[1:][live_idx]  # (n_live, d)
                norms = w_live.norm(dim=1, keepdim=True).clamp(min=1e-8)
                w_norm = w_live / norms
                sim = w_norm @ w_norm.T  # (n_live, n_live)
                overlap = sim.abs() - torch.eye(sim.shape[0], device=sim.device)
                mask = overlap > self.inhibition_threshold
                if mask.any():
                    beta = self.inhibition_beta * overlap * mask.float()
                    beta.fill_diagonal_(0)
                    push_live = beta @ w_live  # (n_live, d)
                    # 只推离活 expert
                    self.gate_weight.data[1:][live_idx] -= push_live

        # 3. 连接修剪: 长期不活跃 + 权重很弱 → 从 mask 剪除
        if step % 100 == 0:
            weak = (
                (self.gate_weight.abs() < self.prune_threshold)
                & (self.inactive_steps > self.prune_inactive).unsqueeze(1)
            )
            pruned = int(weak.sum().item())
            if pruned > 0:
                self.connect_mask.data = torch.where(
                    weak,
                    torch.zeros_like(self.connect_mask),
                    self.connect_mask,
                )
                _logger.info(f"[SparseHebbian] step={step} 修剪 {pruned} 个连接")

        # 4. 连接生长 (neurogenesis): 就近原则 + 存活选择 + 单向导通 (向量化)
        if step % self.growth_interval == 0:
            d = self.d_model
            n_routed = self.n_total - 1
            connected = self.connect_mask[1:]  # (n_routed, d)
            total_occupied = connected.sum(dim=0)  # (d,)
            window = max(1, d // 20)
            # 邻居核
            kernel = torch.tensor(
                [1.0 / (abs(di) + 1) for di in range(-window, window + 1)],
                device=self.connect_mask.device,
            ).view(1, 1, -1)  # (1,1,k)

            for j in range(n_routed):
                if loads[j] < 1e-4:
                    continue
                other_occupied = total_occupied - connected[j]
                available = (connected[j] == 0) & (other_occupied == 0)
                if available.sum() == 0:
                    continue

                # 卷积计算邻居得分
                conn_float = connected[j].float().view(1, 1, -1)
                neighbor_score = F.conv1d(conn_float, kernel, padding=window).squeeze()
                neighbor_score = neighbor_score * available.float()

                if neighbor_score.sum() > 0:
                    probs = neighbor_score / neighbor_score.sum()
                    n_conn = int(connected[j].sum().item())
                    n_growth = max(1, int(n_conn * self.growth_prob))
                    n_growth = min(n_growth, int(available.sum().item()))
                    if n_growth > 0:
                        growth_idx = torch.multinomial(probs, n_growth, replacement=False)
                        growth_mask = torch.zeros(d, device=self.connect_mask.device)
                        growth_mask[growth_idx] = 1.0
                        self.connect_mask[j + 1] = torch.clamp(
                            self.connect_mask[j + 1] + growth_mask, 0, 1
                        )
                        new_w = growth_mask * torch.randn_like(
                            self.gate_weight.data[j + 1]
                        ) * 0.01
                        self.gate_weight.data[j + 1] += new_w
        
        # 5. 存活选择: 新连接在下次修剪时接受检验
        # (新连接权重很小, 如果后续不被强化, 会在修剪阶段被清除)
        # 这一步由上面的修剪逻辑自动完成, 无需额外代码

        # 6. v2.9: ε-greedy 探索 — 偶尔随机扰动 gate_weight
        #   让 Router 跳出局部最优, 尝试新的路由组合。
        if random.random() < self.exploration_rate:
            noise = torch.randn_like(self.gate_weight[1:]) * 0.02
            self.gate_weight.data[1:] += noise
            self.exploration_rate *= self.exploration_decay
            if step % 500 == 0:  # 每500步汇报一次探索状态
                _logger.info(
                    f"[Explore] step={step} rate={self.exploration_rate:.4f} "
                    f"σ=0.02 applied"
                )

        # 7. v2.9: Router 奖励信号 — 基于冲突趋势调整连接强度
        #   (由 train_phase 调用时传入 rewards dict)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def sparsity_stats(self) -> dict:
        """返回当前连接稀疏度统计。"""
        mask = self.connect_mask[1:]  # 跳过 shared
        per_expert = mask.mean(dim=1).tolist()
        return {
            "mean_sparsity": float(mask.mean().item()),
            "per_expert": per_expert,
            "min": min(per_expert),
            "max": max(per_expert),
        }

    def overlap_matrix(self) -> Tensor:
        """返回 expert 间连接重叠矩阵 (余弦相似度, 不含 shared)。"""
        w = self.gate_weight.data[1:]  # (n_routed, d)
        norms = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
        w_norm = w / norms
        return w_norm @ w_norm.T


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def _selfcheck() -> bool:
    logging.basicConfig(level=logging.INFO)

    # 1. 初始化: 稀疏度正确
    sr = SparseRouter(d_model=64, n_total=3, top_k=1, sparsity=0.3)
    stats = sr.sparsity_stats()
    assert 0.15 < stats["mean_sparsity"] < 0.5, \
        f"稀疏度异常: {stats['mean_sparsity']:.3f}"
    assert stats["per_expert"][0] < 1.0 or stats["per_expert"][1] < 1.0, \
        "至少一个 expert 应该是稀疏的"
    assert sr.connect_mask[0].sum() > 0, "shared expert 应至少有一些连接"

    # 2. forward: 形状 + 梯度稀疏
    x = torch.randn(2, 16, 64, requires_grad=True)
    gates, indices, zl, rl = sr(x)
    assert gates.shape == (2, 16, 1)
    assert indices.shape == (2, 16, 1)
    # 反向传播检查梯度只流向 connect_mask=1 的维度
    loss = gates.sum()
    loss.backward()
    assert sr.gate_weight.grad is not None
    grad = sr.gate_weight.grad
    mask = sr.connect_mask
    # 未被 mask 的维度梯度应该为零
    masked_grad = grad.abs() * (1 - mask)
    assert masked_grad.max().item() < 1e-10, \
        f"梯度泄漏到未连接维度: {masked_grad.max().item():.2e}"

    # 3. expand: 新 expert 稀疏初始化
    old_n = sr.n_total
    sr.expand(source_idx=1, alpha=0.3)
    assert sr.n_total == old_n + 1
    new_mask = sr.connect_mask[-1]
    assert 0 < new_mask.sum() < sr.d_model, \
        f"新 expert 应为稀疏连接, 实际: {new_mask.sum().item()}/{sr.d_model}"
    assert sr.inactive_steps.shape[0] == sr.n_total

    # 4. expand_bridge: 桥接生长
    old_n = sr.n_total
    sr.expand_bridge(src_a=1, src_b=2, alpha=0.3)
    assert sr.n_total == old_n + 1

    # 5. hebbian_step: 衰减未选中 + 侧抑制
    loads = [0.0, 0.5]  # expert 1 未被选中, expert 2 被选中
    w_before = sr.gate_weight.data[1].clone()
    sr.hebbian_step(loads, step=10)
    w_after = sr.gate_weight.data[1]
    assert w_after.norm() < w_before.norm(), \
        "未被选中的 expert 连接应衰减"

    # 6. remove_row: 同步移除 buffer
    old_mask_rows = sr.connect_mask.shape[0]
    sr.remove_row(2)
    assert sr.connect_mask.shape[0] == old_mask_rows - 1
    assert sr.inactive_steps.shape[0] == old_mask_rows - 1

    # 7. restore: 重新随机稀疏
    sr.restore(1)
    assert sr.connect_mask[1].sum() > 0

    # 8. overlap_matrix
    om = sr.overlap_matrix()
    assert om.shape == (sr.n_total - 1, sr.n_total - 1)

    # 9. 连接生长: 就近原则 + 存活选择
    sr_g = SparseRouter(d_model=64, n_total=3, top_k=1, sparsity=0.3, growth_prob=0.3)
    # 让 expert 1 的连接集中在低维度
    sr_g.connect_mask.data[1] = torch.zeros(64)
    sr_g.connect_mask.data[1, :5] = 1.0  # 连接维度 0-4
    # expert 2 的连接集中在高维度
    sr_g.connect_mask.data[2] = torch.zeros(64)
    sr_g.connect_mask.data[2, 60:] = 1.0  # 连接维度 60-63
    
    # 模拟 expert 1 活跃, 多次触发生长
    initial_conn = sr_g.connect_mask[1].sum().item()
    for step in range(50, 200, 50):
        sr_g.hebbian_step(loads=[0.5, 0.0], step=step)  # expert 1 活跃
    grown_conn = sr_g.connect_mask[1].sum().item()
    assert grown_conn >= initial_conn, \
        f"活跃 expert 应生长连接: {initial_conn} → {grown_conn}"
    
    # 验证就近原则: 新连接应在已有连接附近
    new_connections = (sr_g.connect_mask[1] == 1)
    new_indices = torch.where(new_connections)[0]
    # 已有连接在 [0,4], 新连接应在附近 (允许 window 范围)
    far_new = [i for i in new_indices.tolist() if i > 10]
    # 大部分新连接应在已有连接附近 (允许少量远距探索)
    near_new = [i for i in new_indices.tolist() if i <= 10]
    if len(new_indices) > 0:
        assert len(near_new) >= len(far_new), \
            f"新连接应主要在附近: near={len(near_new)}, far={len(far_new)}"
    
    logging.info(f"[SparseRouter] 生长: {initial_conn} → {grown_conn}")

    # 10. 单向导通: connect_mask 无重叠
    sr_u = SparseRouter(d_model=64, n_total=4, top_k=1, sparsity=0.3)
    # 验证初始化时无重叠
    total_mask = sr_u.connect_mask[1:].sum(dim=0)  # (d,)
    assert (total_mask <= 1).all(), \
        f"初始化时 connect_mask 应无重叠, 最大重叠: {total_mask.max().item()}"
    
    # expand 后也无重叠
    sr_u.expand(source_idx=1, alpha=0.3)
    total_mask = sr_u.connect_mask[1:].sum(dim=0)
    assert (total_mask <= 1).all(), \
        f"expand 后 connect_mask 应无重叠, 最大重叠: {total_mask.max().item()}"
    
    # 生长后也无重叠
    for step in range(50, 200, 50):
        loads = [0.5, 0.3, 0.2, 0.1][:sr_u.n_total - 1]
        sr_u.hebbian_step(loads, step)
    total_mask = sr_u.connect_mask[1:].sum(dim=0)
    assert (total_mask <= 1).all(), \
        f"生长后 connect_mask 应无重叠, 最大重叠: {total_mask.max().item()}"
    
    logging.info("[SparseRouter] 单向导通: 无重叠 ✓")
    logging.info("[SparseRouter] 自检通过")
    return True


if __name__ == "__main__":
    assert _selfcheck()
