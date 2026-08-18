"""lif_pool.py — 真 LIF 阈值发放池 (还原神经元完整动力学)。

完整动力学链 (非 MoE top-k):
  ① 膜电位积分: V_m(t) = (1-1/τ)·V_m(t-1) + Σ W_in·z   (突触输入)
  ② 阈值检测:   V_m > θ → 发放 (spike)
  ③ 发放后复位: V_m[spike] = 0 (不应期)
  ④ 时间常数 τ: 快/慢神经元 (记忆长度)
  ⑤ 稀疏性: 阈值自然产生 (该发才发, 非强制 k 个)
  ⑥ 生长: 目标驱动, 弱初始化 (V_m=0, 试探期)
  ⑦ 淘汰: 长期不发放 → 回收
  ⑧ 可塑性: 权重 ES/梯度 (无梯度框架兼容)

无 top-k! 发放数自适应。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LIFPool(nn.Module):
    def __init__(self, d_model=64, d_pool=512, tau_min=2.0, tau_max=48.0,
                 theta=0.5, learn_tau=True, learn_theta=False,
                 init_active=128):
        super().__init__()
        self.d_model, self.d_pool = d_model, d_pool
        self.W_in = nn.Parameter(torch.randn(d_pool, d_model) / d_model ** 0.5)
        self.W_out = nn.Parameter(torch.randn(d_model, d_pool) / d_pool ** 0.5)
        # τ 多样化 (快/慢记忆)
        tau = torch.logspace(torch.log10(torch.tensor(tau_min)),
                             torch.log10(torch.tensor(tau_max)), d_pool)
        self.tau_log = nn.Parameter(torch.log(tau)) if learn_tau else torch.log(tau)
        # 发放阈值
        self.register_buffer("theta", torch.full((d_pool,), theta))
        # 连接掩码: 哪些神经元存在 (未激活 = 未生长/已淘汰)
        self.register_buffer("alive", torch.zeros(d_pool, dtype=torch.bool))
        self.alive[:init_active] = True
        # 发放统计
        self.register_buffer("f_rate", torch.zeros(d_pool))   # 发放率 EMA
        self.register_buffer("f_count", torch.zeros(d_pool))
        self.growth_log = []

    def tau(self):
        return torch.exp(self.tau_log).clamp(1.5, 100.0)

    def reset_state(self, batch):
        self.vm = torch.zeros(batch, self.d_pool, device=self.W_in.device)

    def forward(self, z_seq):
        """序列 (B,T,d) → 发放稀疏输出 (B,T,d)。无 top-k!"""
        B, T = z_seq.shape[:2]
        self.reset_state(B)
        tau = self.tau().unsqueeze(0)
        leak = 1.0 - 1.0 / tau
        alive = self.alive.to(self.W_in.device)
        th = self.theta.to(self.W_in.device)
        outs, spikes = [], []
        for t in range(T):
            z = z_seq[:, t]
            # ① 膜电位积分 (突触输入)
            self.vm = self.vm * leak + z @ self.W_in.t()
            # 只允许存活的神经元发放 (未激活 = 不存在)
            self.vm = self.vm.masked_fill(~alive.unsqueeze(0), -1e9)
            # ② 阈值检测 → 发放事件 (非输出!)
            spike = self.vm > th.unsqueeze(0)
            # ③ 发放后部分复位 (不应期, 电势回落到基线)
            self.vm[spike] = self.vm[spike] * 0.1
            # 读出 = 连续膜电位 (电势有高低! 表示信息)
            out = F.gelu(self.vm) @ self.W_out.t()
            outs.append(out)
            spikes.append(spike.float())
            # ④ 发放统计 (可塑性/淘汰依据)
            with torch.no_grad():
                self.f_rate = 0.999 * self.f_rate.to(self.W_in.device) \
                              + 0.001 * spike.float().mean(0)
                self.f_count += spike.float().mean(0)
        return torch.stack(outs, 1), torch.stack(spikes, 1)

    def grow(self, spike_sel, n=2, alpha=0.2, perturb=0.2):
        """⑥ 生长: 目标驱动 (目标样本发放的神经元) → 弱继承新神经元。"""
        dead = (~self.alive).nonzero().flatten()
        if len(dead) == 0:
            return 0
        cnt = torch.zeros(self.d_pool, device=spike_sel.device)
        for row in spike_sel.flatten().long():
            cnt[row] += 1
        cand = torch.argsort(cnt * self.alive.float(), descending=True)[:n]
        cand = cand[self.alive[cand]][:n]
        n_grow = min(len(cand), len(dead))
        with torch.no_grad():
            for src, tgt in zip(cand, dead[:n_grow]):
                src, tgt = int(src), int(tgt)
                # 弱继承 (新神经元从近零开始, 试探)
                self.W_in.data[tgt] = alpha * self.W_in.data[src] \
                    + perturb * torch.randn_like(self.W_in.data[src])
                self.W_out.data[:, tgt] = alpha * self.W_out.data[:, src]
                self.tau_log.data[tgt] = self.tau_log.data[src] \
                    * (0.7 + 0.6 * torch.rand(()).item())
                self.alive[tgt] = True
                self.vm[:, tgt] = 0.0
                self.growth_log.append(tgt)
        return n_grow

    def prune(self, thresh=0.001, min_count=50):
        """⑦ 淘汰: 长期不发放 → 回收。"""
        low = self.alive & (self.f_rate < thresh)
        n = int(low.sum().item())
        self.alive[low] = False
        self.f_rate[low] = 0.0
        self.f_count[low] = 0.0
        return n

    def n_active(self):
        return int(self.alive.sum().item())
