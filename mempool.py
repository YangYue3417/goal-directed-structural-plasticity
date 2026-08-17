"""mempool.py — 记忆池 (LIF 泄漏积分器神经元池)。

神经元 = 泄漏积分器: V_m(t) = (1-1/τ)·V_m(t-1) + z_t
- τ 多样化 (per-neuron): 快神经元 (瞬态) ↔ 慢神经元 (节奏/相位)
- 短期记忆: 电位残留编码前几个状态 → 时序判断
- 生长: 克隆 W_in 行 + τ + W_out 列 (难样本定向)
- 淘汰: 激活率低 → 回收

接口兼容 SparseUnit 风格 (masked top-k 稀疏 + 激活率统计)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemPool(nn.Module):
    def __init__(self, d_model=64, d_pool=512, top_k=64,
                 tau_min=2.0, tau_max=48.0, init_active=128,
                 learn_tau=True):
        super().__init__()
        self.d_model, self.d_pool, self.top_k = d_model, d_pool, top_k
        self.W_in = nn.Parameter(torch.randn(d_pool, d_model) / d_model ** 0.5)
        self.W_out = nn.Parameter(torch.randn(d_model, d_pool) / d_pool ** 0.5)
        # τ 多样化: 对数均匀 (快→慢), 可学习
        tau = torch.logspace(torch.log10(torch.tensor(tau_min)),
                             torch.log10(torch.tensor(tau_max)), d_pool)
        self.tau_log = nn.Parameter(torch.log(tau)) if learn_tau \
            else torch.log(tau)
        self.register_buffer("active_mask",
                             torch.zeros(d_pool, dtype=torch.bool))
        self.active_mask[:init_active] = True
        self.register_buffer("act_rate", torch.zeros(d_pool))
        self.register_buffer("vm", torch.zeros(d_pool))
        self.growth_log = []
        self._source = None          # 生长克隆的源神经元
        self._growth_phase = []      # 生长时的 τ 值 (相位分工证据)
        # 新生神经元 (baby): 生长→试探→连接
        self.register_buffer("baby_mask", torch.zeros(d_pool, dtype=torch.bool))
        self.register_buffer("baby_age", torch.zeros(d_pool))
        self.register_buffer("baby_rate", torch.zeros(d_pool))  # 试探期激活率
        # 储备生命周期: 塑造期→评估→保留/凋亡→再出生
        self.register_buffer("reserve_age", torch.zeros(d_pool))
        self.register_buffer("reserve_rate", torch.zeros(d_pool))  # shadow 选中率
        self.register_buffer("reserve_born", torch.zeros(d_pool))   # 出生代次
        self.reserve_stats = {"apoptosis": 0, "reborn": 0, "kept": 0}

    def _reinit_neuron(self, idx):
        """重新随机初始化神经元 (凋亡后再出生)。"""
        with torch.no_grad():
            self.W_in.data[idx] = torch.randn_like(self.W_in.data[idx]) / self.d_model ** 0.5
            self.W_out.data[:, idx] = torch.randn_like(self.W_out.data[:, idx]) / self.d_pool ** 0.5
            self.tau_log.data[idx] = torch.log(torch.tensor(8.0))
            self.reserve_age[idx] = 0.0
            self.reserve_rate[idx] = 0.0
            self.reserve_born[idx] += 1
        self.reserve_stats["reborn"] += 1

    def tau(self):
        return torch.exp(self.tau_log).clamp(1.5, 100.0)

    def reset_state(self, batch):
        """序列前: 膜电位归零。"""
        self.vm = torch.zeros(batch, self.d_pool,
                              device=self.W_in.device)

    def forward(self, z_seq, mask=None):
        """输入序列 (B, T, d_model) → (B, T, d_model), selected (B, T, K)。

        逐时间步: V_m 泄漏积分 → top-k 稀疏 → 读出。
        """
        B, T, D = z_seq.shape
        self.reset_state(B)
        tau = self.tau().unsqueeze(0)          # (1, pool)
        leak = 1.0 - 1.0 / tau                  # (1, pool)
        am = self.active_mask.to(self.W_in.device)
        baby = self.baby_mask.to(self.W_in.device)
        p_explore = 0.3   # 试探: 强制激活概率
        outs, sels = [], []
        self._shadow = []
        for t in range(T):
            z = z_seq[:, t]
            self.vm = self.vm * leak + z @ self.W_in.t()
            pre = self.vm.masked_fill(~am.unsqueeze(0), -1e9)
            # 试探: baby 神经元以概率强制进入 top-k 探测 (低权重)
            if baby.any():
                explore = torch.rand(B, baby.sum().item(),
                                     device=pre.device) < p_explore
                b_idx = baby.nonzero().flatten()
                pre_b = pre[:, b_idx]
                pre_b = pre_b.masked_fill(~explore, -1e9)
                pre[:, b_idx] = torch.maximum(pre[:, b_idx], pre_b)
            vals, idx = pre.topk(self.top_k, dim=1)          # (B, K)
            sparse = torch.zeros_like(pre)
            # baby 输出用低权重 (试探不主导)
            w_scale = torch.where(baby, torch.full_like(baby, 0.3),
                                  torch.ones_like(baby))
            sparse.scatter_(1, idx, F.gelu(vals) * w_scale[idx])
            outs.append(sparse @ self.W_out.t())
            sels.append(idx)
            # shadow: 储备神经元 (未激活) top-k → 方向塑造 (暗中学习)
            reserve = ~am & ~baby
            if reserve.any():
                r_idx = reserve.nonzero().flatten()
                pre_r = self.vm[:, r_idx]
                r_k = min(self.top_k, r_idx.numel())
                vals_r, idx_r = pre_r.topk(r_k, dim=1)
                sh_r = torch.zeros(B, r_idx.numel(), device=pre.device)
                sh_r.scatter_(1, idx_r, F.gelu(vals_r))
                self._shadow.append(sh_r)
                # 储备选中率 (shadow top-k 内) + 年龄 (塑造期计时)
                with torch.no_grad():
                    oh_r = torch.zeros(B, r_idx.numel(), device=pre.device)
                    oh_r.scatter_(1, idx_r, 1.0)
                    self.reserve_rate[r_idx] = 0.999 * self.reserve_rate[r_idx] \
                                               + 0.001 * oh_r.mean(0)
                    self.reserve_age[r_idx] += 1.0
            # 激活率 + baby 年龄/激活
            with torch.no_grad():
                oh = torch.zeros(B, self.d_pool, device=pre.device)
                oh.scatter_(1, idx, 1.0)
                self.act_rate = 0.999 * self.act_rate.to(pre.device) \
                                + 0.001 * oh.mean(0)
                if baby.any():
                    self.baby_age[baby] += 1.0
                    self.baby_rate[baby] = 0.99 * self.baby_rate[baby] \
                                           + 0.01 * oh.mean(0)[baby]
        # 打包 shadow (B,T,reserve) + 记录储备索引
        if self._shadow:
            self.shadow_stack = torch.stack(self._shadow, 1)
            self.reserve_idx = (~self.active_mask & ~self.baby_mask)                 .nonzero().flatten()
        else:
            self.shadow_stack = None
        return torch.stack(outs, 1), torch.stack(sels, 1)

    def settle_reserve(self, age_thresh=2000.0, rate_thresh=0.003):
        """储备成熟评估: 塑造期结束 → 激活率高的保留 (有方向),
        激活率低的凋亡 → 再出生 (重新塑造)。"""
        rmask = (~self.active_mask & ~self.baby_mask)
        done = rmask & (self.reserve_age >= age_thresh)
        if not done.any():
            return 0, 0
        good = done & (self.reserve_rate >= rate_thresh)   # 有贡献 → 保留
        bad = done & ~good                                  # 无贡献 → 凋亡
        n_bad = int(bad.sum().item())
        if n_bad:
            idx = bad.nonzero().flatten()
            for i in idx:
                self._reinit_neuron(int(i))
            self.reserve_stats["apoptosis"] += n_bad
        self.reserve_stats["kept"] += int(good.sum().item())
        return int(good.sum().item()), n_bad

    def dream_grow(self, shadow_sel_hard, n=2):
        """做梦生长: 激活储备神经元中对难样本 shadow 预测最好的。

        shadow_sel_hard: (B,) 储备池内 top-k 索引 (难样本时刻)
        """
        if self.shadow_stack is None or len(self.reserve_idx) == 0:
            return 0
        cnt = torch.zeros(len(self.reserve_idx), device=shadow_sel_hard.device)
        for row in shadow_sel_hard.flatten():
            cnt[row] += 1
        cand = torch.argsort(cnt, descending=True)[:n]
        cand = cand[torch.tensor(
            [self.active_mask[i].item() for i in
             self.reserve_idx[cand].cpu()]) == False]
        n_grow = min(len(cand), 2)
        with torch.no_grad():
            for i in cand[:n_grow]:
                tgt = int(self.reserve_idx[i])
                self.active_mask[tgt] = True
                self.growth_log.append(tgt)
        return n_grow

    def grow(self, sel_hard, perturb=0.15, n=2):
        """难样本定向生长: 克隆难样本激活的神经元 (权重+τ)。

        源 = 片段内误差最大的时刻 top-k 激活的神经元。
        """
        inactive = (~self.active_mask).nonzero().flatten()
        if len(inactive) == 0:
            return 0
        cnt = torch.zeros(self.d_pool, device=sel_hard.device)
        for row in sel_hard.flatten():
            cnt[row] += 1
        cand = torch.argsort(cnt * self.active_mask.float(), descending=True)
        cand = cand[self.active_mask[cand]][:n]
        n_grow = min(len(cand), len(inactive))
        if n_grow == 0:
            return 0
        with torch.no_grad():
            for src, tgt in zip(cand, inactive[:n_grow]):
                src, tgt = int(src), int(tgt)
                # 生长→试探→连接: 弱初始化 (不完全继承), 电位零, τ 发展
                self.W_in.data[tgt] = 0.5 * self.W_in.data[src] \
                    + 0.3 * torch.randn_like(self.W_in.data[src])
                self.W_out.data[:, tgt] = 0.5 * self.W_out.data[:, src] \
                    + 0.3 * torch.randn_like(self.W_out.data[:, src])
                self.tau_log.data[tgt] = self.tau_log.data[src] \
                    * (0.7 + 0.6 * torch.rand(()).item())  # 发展自己的 τ
                self.active_mask[tgt] = True
                self.baby_mask[tgt] = True                  # 试用期
                self.baby_age[tgt] = 0.0
                self.baby_rate[tgt] = 0.0
                self.vm[:, tgt] = 0.0                       # 电位从零
                self.growth_log.append(tgt)
                self._source = src
                self._growth_phase.append(float(self.tau()[tgt]))
        return n_grow

    def settle_babies(self, age_thresh=400.0, rate_thresh=0.01):
        """试探期结束: 激活达标 → 巩固 (转正式); 不达标 → 剪枝淘汰。"""
        done = self.baby_mask & (self.baby_age >= age_thresh)
        if not done.any():
            return 0, 0
        good = done & (self.baby_rate >= rate_thresh)
        bad = done & ~good
        with torch.no_grad():
            # 巩固: 权重恢复全量 (结束试探期低权重)
            self.baby_mask[good] = False
            # 淘汰: 剪枝回收
            self.active_mask[bad] = False
            self.baby_mask[bad] = False
        return int(good.sum().item()), int(bad.sum().item())

    def prune(self, threshold=0.005):
        """淘汰: 激活率低的神经元回收 (含生长神经元)。"""
        inactive = (~self.active_mask).nonzero().flatten()
        low = (self.act_rate < threshold) & self.active_mask
        n = int(low.sum().item())
        self.active_mask[low] = False
        return n, int(inactive.numel())


class StaticPool(nn.Module):
    """对照: 无记忆池 (τ→∞, 等价静态 top-k 池)。接口同 MemPool。"""
    def __init__(self, d_model=64, d_pool=512, top_k=64, init_active=128):
        super().__init__()
        self.d_model, self.d_pool, self.top_k = d_model, d_pool, top_k
        self.W_in = nn.Parameter(torch.randn(d_pool, d_model) / d_model ** 0.5)
        self.W_out = nn.Parameter(torch.randn(d_model, d_pool) / d_pool ** 0.5)
        self.register_buffer("active_mask",
                             torch.zeros(d_pool, dtype=torch.bool))
        self.active_mask[:init_active] = True
        self.register_buffer("act_rate", torch.zeros(d_pool))
        self.growth_log = []
        self._growth_phase = []

    def reset_state(self, batch):
        pass

    def forward(self, z_seq, mask=None):
        B, T, D = z_seq.shape
        am = self.active_mask.to(self.W_in.device)
        outs, sels = [], []
        self._shadow = []
        for t in range(T):
            z = z_seq[:, t]
            pre = (z @ self.W_in.t()).masked_fill(~am.unsqueeze(0), -1e9)
            vals, idx = pre.topk(self.top_k, dim=1)
            sparse = torch.zeros_like(pre)
            sparse.scatter_(1, idx, F.gelu(vals))
            outs.append(sparse @ self.W_out.t())
            sels.append(idx)
            with torch.no_grad():
                oh = torch.zeros(B, self.d_pool, device=pre.device)
                oh.scatter_(1, idx, 1.0)
                self.act_rate = 0.999 * self.act_rate.to(pre.device) \
                                + 0.001 * oh.mean(0)
        # 打包 shadow (B,T,reserve) + 记录储备索引
        if self._shadow:
            self.shadow_stack = torch.stack(self._shadow, 1)
            self.reserve_idx = (~self.active_mask & ~self.baby_mask)                 .nonzero().flatten()
        else:
            self.shadow_stack = None
        return torch.stack(outs, 1), torch.stack(sels, 1)

    def dream_grow(self, shadow_sel_hard, n=2):
        """做梦生长: 激活储备神经元中对难样本 shadow 预测最好的。

        shadow_sel_hard: (B,) 储备池内 top-k 索引 (难样本时刻)
        """
        if self.shadow_stack is None or len(self.reserve_idx) == 0:
            return 0
        cnt = torch.zeros(len(self.reserve_idx), device=shadow_sel_hard.device)
        for row in shadow_sel_hard.flatten():
            cnt[row] += 1
        cand = torch.argsort(cnt, descending=True)[:n]
        cand = cand[torch.tensor(
            [self.active_mask[i].item() for i in
             self.reserve_idx[cand].cpu()]) == False]
        n_grow = min(len(cand), 2)
        with torch.no_grad():
            for i in cand[:n_grow]:
                tgt = int(self.reserve_idx[i])
                self.active_mask[tgt] = True
                self.growth_log.append(tgt)
        return n_grow

    def grow(self, sel_hard, perturb=0.15, n=2):
        inactive = (~self.active_mask).nonzero().flatten()
        if len(inactive) == 0:
            return 0
        cnt = torch.zeros(self.d_pool, device=sel_hard.device)
        for row in sel_hard.flatten():
            cnt[row] += 1
        cand = torch.argsort(cnt * self.active_mask.float(), descending=True)
        cand = cand[self.active_mask[cand]][:n]
        n_grow = min(len(cand), len(inactive))
        if n_grow == 0:
            return 0
        with torch.no_grad():
            for src, tgt in zip(cand, inactive[:n_grow]):
                src, tgt = int(src), int(tgt)
                self.W_in.data[tgt] = self.W_in.data[src] \
                    + perturb * torch.randn_like(self.W_in.data[src])
                self.W_out.data[:, tgt] = self.W_out.data[:, src]
                self.active_mask[tgt] = True
                self.growth_log.append(tgt)
        return n_grow

    def prune(self, threshold=0.005):
        inactive = (~self.active_mask).nonzero().flatten()
        low = (self.act_rate < threshold) & self.active_mask
        n = int(low.sum().item())
        self.active_mask[low] = False
        return n, int(inactive.numel())
