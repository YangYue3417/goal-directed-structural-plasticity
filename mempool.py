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
        outs, sels = [], []
        for t in range(T):
            z = z_seq[:, t]
            self.vm = self.vm * leak + z @ self.W_in.t()
            pre = self.vm.masked_fill(~am.unsqueeze(0), -1e9)
            vals, idx = pre.topk(self.top_k, dim=1)          # (B, K)
            sparse = torch.zeros_like(pre)
            sparse.scatter_(1, idx, F.gelu(vals))
            outs.append(sparse @ self.W_out.t())
            sels.append(idx)
            # 激活率 (电位激活, 含记忆参与)
            with torch.no_grad():
                oh = torch.zeros(B, self.d_pool, device=pre.device)
                oh.scatter_(1, idx, 1.0)
                self.act_rate = 0.999 * self.act_rate.to(pre.device) \
                                + 0.001 * oh.mean(0)
        return torch.stack(outs, 1), torch.stack(sels, 1)

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
            for i, (src, tgt) in enumerate(zip(cand, inactive[:n_grow])):
                src, tgt = int(src), int(tgt)
                self.W_in.data[tgt] = self.W_in.data[src] \
                    + perturb * torch.randn_like(self.W_in.data[src])
                self.W_out.data[:, tgt] = self.W_out.data[:, src]
                self.tau_log.data[tgt] = self.tau_log.data[src]  # 继承 τ
                self.active_mask[tgt] = True
                self.growth_log.append(tgt)
                self._source = src
                self._growth_phase.append(float(self.tau()[tgt]))
        return n_grow

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
        return torch.stack(outs, 1), torch.stack(sels, 1)

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
