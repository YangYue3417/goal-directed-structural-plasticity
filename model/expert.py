"""M4: Expert FFN + ExpertContainer。SwiGLU 单专家 + 动态专家管理。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Expert(nn.Module):
    """
    SwiGLU FFN: d_model -> d_ff -> d_model

    计算: down_proj(SiLU(gate_proj(x)) * up_proj(x))

    初始化:
      - gate_proj, up_proj: N(0, 0.02)
      - down_proj: N(0, 0.02 / sqrt(2 * n_layers)), 但 n_layers 未知 -> 用 0.02/sqrt(20) ~= 0.0045
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: hidden dim (640)
            d_ff: FFN intermediate dim (1344)

        Raises:
            ValueError: d_ff 不能被 64 整除
        """
        super().__init__()
        if d_ff % 64 != 0:
            raise ValueError(
                f"d_ff must be divisible by 64, got d_ff={d_ff}"
            )
        self.d_model = d_model
        self.d_ff = d_ff

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """重新随机初始化所有参数。"""
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02 / (20 ** 0.5))

    def forward(self, x: Tensor) -> Tensor:
        """
        输入: (B, S, d_model) 或 (*, d_model)
        输出: 与输入 shape 完全相同

        Args:
            x: 输入张量

        Returns:
            经过 SwiGLU FFN 处理后的张量, shape 与输入相同
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)

    def clone(self, perturbation_std: float = 0.05) -> "Expert":
        """
        安全复制专家权重并添加扰动。

        步骤:
          1. 创建新 Expert(d_model, d_ff) 实例
          2. 用 self.state_dict() 加载权重 (load_state_dict)
          3. 每个参数加噪声: N(0, perturbation_std * std(p))

        Args:
            perturbation_std: 噪声标准差的比例因子, 默认 0.05

        Returns:
            新的 Expert 实例, 参数已注册, 与 self 结构相同但权重独立
        """
        clone = Expert(self.d_model, self.d_ff)
        clone.load_state_dict(self.state_dict())
        with torch.no_grad():
            for p in clone.parameters():
                noise_std = perturbation_std * p.std()
                p.add_(torch.randn_like(p) * noise_std)
        return clone


class ExpertContainer(nn.Module):
    """
    管理 routed experts 的动态列表。

    不包含 shared expert — shared expert 在 DynamicMoEFFN 中单独持有。

    routed experts 存储在 nn.ModuleList 中, 索引从 0 开始。
    shared expert 在 gate 中的 index 为 0, routed experts 的 gate index = routed_idx + 1。
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        初始化空容器。

        Args:
            d_model: hidden dim
            d_ff: FFN intermediate dim
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.experts: nn.ModuleList = nn.ModuleList()

    @property
    def n_routed(self) -> int:
        """当前 routed expert 总数 (含 frozen)"""
        return len(self.experts)

    def add_routed(self, expert: Expert):
        """追加一个 routed expert

        Args:
            expert: Expert 实例
        """
        self.experts.append(expert)

    def get_routed(self, idx: int) -> Expert:
        """按 routed index (0-based) 返回 Expert。支持负索引。

        Args:
            idx: routed expert 索引 (0-based)

        Returns:
            Expert 实例
        """
        return self.experts[idx]

    def forward(
        self,
        x: Tensor,            # (B, S, d)
        gate_weights: Tensor,  # (B, S, top_k)
        gate_indices: Tensor,  # (B, S, top_k) int64
    ) -> Tensor:
        """
        将输入 x 通过 top-k 选择的 routed experts。

        算法:
          output = torch.zeros_like(x)
          遍历 self.experts:
            # gate_indices 中的值是 gate 绝对索引 (含 shared expert)
            # 我们的 routed gate index = routed_idx + 1 (因为 gate[0]=shared)
            routed_gate_idx = routed_idx + 1
            mask = (gate_indices == routed_gate_idx)  # (B, S, top_k)
            if mask.any():
                # 对 matched tokens 计算 expert output
                # 收集所有需要处理的 token, 过 expert, 加权加回 output

        注意:
          - expert.requires_grad=False 时仍参与前向 (frozen expert)
          - 需要高效实现: 只对被选中的 token 计算 expert

        Args:
            x: 输入张量 (B, S, d)
            gate_weights: gate 权重 (B, S, top_k)
            gate_indices: gate 索引 (B, S, top_k), int64

        Returns:
            输出张量 (B, S, d)
        """
        B, S, d = x.shape
        K = gate_indices.shape[-1]  # top_k
        output = torch.zeros_like(x)
        n_routed = len(self.experts)

        # 每层只 sync 一次: 统计每个 expert 被选中的次数 (GPU bincount, 无 sync),
        # 单次 .tolist() 拿到有 token 的 expert 索引 \-> 160 次 sync/forward 降为 10 次
        counts = torch.bincount(gate_indices.flatten(), minlength=n_routed + 1)
        active_experts = (counts[1:] > 0).nonzero(as_tuple=True)[0].tolist()  # 单次 sync

        for routed_idx in active_experts:
            expert = self.experts[routed_idx]
            gate_idx = routed_idx + 1  # +1 because gate[0] = shared expert

            hit_mask = (gate_indices == gate_idx)  # (B, S, K)

            for k in range(K):
                slot_mask = hit_mask[:, :, k]  # (B, S)
                selected_x = x[slot_mask]
                if selected_x.shape[0] == 0:  # shape 是 CPU 元数据, 无 GPU sync
                    continue

                expert_out = expert(selected_x)  # (N, d)
                weights = gate_weights[:, :, k][slot_mask].unsqueeze(-1)  # (N, 1)
                output[slot_mask] += weights * expert_out

        return output


def _selfcheck() -> bool:
    import logging
    logging.basicConfig(level=logging.INFO)

    # 1. 参数校验: d_ff=100 -> ValueError
    try:
        Expert(640, 100)
        assert False, "应拒绝非 64 倍数的 d_ff"
    except ValueError:
        pass

    # 2. 输出 Shape 不变
    expert = Expert(640, 1344)
    x = torch.randn(2, 16, 640)
    y = expert(x)
    assert y.shape == x.shape, f"Shape: {y.shape} != {x.shape}"

    # 3. 输出无 NaN/Inf
    assert torch.isfinite(y).all(), "输出含 NaN 或 Inf"

    # 4. Clone 正确性
    clone = expert.clone(perturbation_std=0.05)
    assert len(list(clone.parameters())) == len(list(expert.parameters())), "参数数不一致"
    for p1, p2 in zip(expert.parameters(), clone.parameters()):
        assert not torch.allclose(p1, p2), "Clone 参数与原始完全相同—扰动未生效"

    # 5. Clone zero perturbation is identical
    clone_zero = expert.clone(perturbation_std=0.0)
    for p1, p2 in zip(expert.parameters(), clone_zero.parameters()):
        assert torch.allclose(p1, p2), "零扰动 clone 应与原版完全相同"

    # 6. ExpertContainer 基本操作
    container = ExpertContainer(640, 1344)
    assert container.n_routed == 0
    e1 = Expert(640, 1344)
    container.add_routed(e1)
    assert container.n_routed == 1
    assert container.get_routed(0) is e1
    assert container.get_routed(-1) is e1

    # 7. ExpertContainer forward
    B, S, d = 2, 16, 640
    gw = torch.tensor([[[1.0]]]).expand(B, S, 1)  # batch=2, seq=16, top_k=1, weight=1
    gi = torch.ones(B, S, 1, dtype=torch.long)  # gate index = 1 (routed idx 0)
    out = container(x, gw, gi)
    assert out.shape == x.shape

    # 8. Frozen expert 前向仍参与
    e1.requires_grad_(False)
    out2 = container(x, gw, gi)
    assert out2.shape == x.shape

    # 9. 多 expert 场景
    e2 = Expert(640, 1344)
    container.add_routed(e2)
    assert container.n_routed == 2
    # top_k=2, 一半 token 走 expert 0, 一半走 expert 1
    gw2 = torch.ones(B, S, 2) / 2.0
    gi2 = torch.zeros(B, S, 2, dtype=torch.long)
    gi2[:, :, 0] = 1  # gate idx for routed[0]
    gi2[:, :, 1] = 2  # gate idx for routed[1]
    out3 = container(x, gw2, gi2)
    assert out3.shape == x.shape
    assert torch.isfinite(out3).all()

    logging.info(f"[M4 selfcheck] Expert params: {sum(p.numel() for p in expert.parameters()):,}")
    logging.info(f"[M4 selfcheck] Clone params: {sum(p.numel() for p in clone.parameters()):,}")
    logging.info(f"[M4 selfcheck] Container routed: {container.n_routed}")
    return True


def register_conflict_hook(expert, conflict_ema, layer_idx, routed_idx, alpha=0.1, min_tokens=32,
                           separability_ema=None, grad_agg_ema=None):
    """在 expert 上注册 backward hook，收集归一化梯度冲突统计 (v2.1 §1.1)。

    per-step: c = 1 − ‖Σg‖²/(N·Σ‖g‖²) = Var(g)/E‖g‖² ∈ [0,1]
      ≈0: token 梯度方向一致 (健康);  →1: 破坏性干扰 (分裂候选)
    EMA 按 micro-batch 更新，初值 0.0（保守，防冷启动假阳性）。
    NaN 守卫: 梯度含非有限值时跳过本步更新。
    min_tokens 守卫: N < min_tokens 时统计量 ≈ 纯噪声，跳过 (reviewer 建议 32)。
    分裂后 clone 也需要调用此函数。

    v2.4 三判据门控扩展:
      separability_ema: PR (participation ratio) 有效方向数。
        子采样 ≤64 token, 单位化后 PR = n²/Σᵢⱼ(gᵢ·gⱼ)²。
        PR≈1: 梯度单方向 (单一专业, 不需要分); PR 大: 多维 (可分)。
        与 conflict 去相关: 反向共线梯度 conflict 高但 PR 低。
      grad_agg_ema: 聚合梯度方向 EMA (向量), 供分裂验证算 source/clone cos。

    Returns:
        RemovableHandle: hard prune 后摘除旧 hook 并重注册（索引重映射）。
    """
    def bw_hook(module, grad_input, grad_output):
        go = grad_output[0]
        N = go.shape[0] if go.dim() > 1 else 0
        if N < min_tokens:
            return  # reviewer 建议: 微批 token 太少时跳过，避免纯噪声
        go = go.detach().float()
        if not torch.isfinite(go).all():
            return  # NaN 守卫
        sq_sum = go.sum(dim=0).pow(2).sum()   # ‖Σg‖²
        sq_ind = go.pow(2).sum()              # Σ‖g‖²
        denom = go.shape[0] * sq_ind
        if denom <= 0:
            return
        c = float(1.0 - sq_sum / denom)
        k = (layer_idx, routed_idx)
        prev = conflict_ema.get(k, 0.0)
        conflict_ema[k] = (1.0 - alpha) * prev + alpha * c

        # v2.4: PR 可分性 (子采样 64 token, 单位化方向多样性)
        if separability_ema is not None:
            if N > 64:
                idx = torch.randperm(N, device=go.device)[:64]
                Gs = go[idx]
            else:
                Gs = go
            Gn = Gs / (Gs.norm(dim=1, keepdim=True) + 1e-12)
            n = Gn.shape[0]
            ggt = Gn @ Gn.T
            ggt_sum = float(ggt.pow(2).sum())
            if ggt_sum > 1e-12:
                pr = float(n * n) / ggt_sum
            else:
                pr = 0.0
            prev_s = separability_ema.get(k, 0.0)
            separability_ema[k] = (1.0 - alpha) * prev_s + alpha * pr

        # v2.4: 聚合梯度方向 (分裂验证: cos(source_agg, clone_agg))
        if grad_agg_ema is not None:
            g_mean = go.mean(dim=0)  # (d,)
            prev_v = grad_agg_ema.get(k)
            if prev_v is None:
                grad_agg_ema[k] = g_mean.clone()
            else:
                prev_v.mul_(1.0 - alpha).add_(g_mean, alpha=alpha)
    return expert.register_full_backward_hook(bw_hook)


if __name__ == "__main__":
    assert _selfcheck()
    print("[M4] 自检通过")
