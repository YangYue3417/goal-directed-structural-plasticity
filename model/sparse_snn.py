"""LIF 稀疏连接脉冲网络 (Sparse SNN) — 决策来自网络动力学。

核心 (用户观点):
  - 稀疏连接: 每神经元只连少数神经元 (类脑突触)
  - spike 传导: 信号以脉冲在 T 时间步传播 (LIF 膜电位)
  - 无检索头: 输出 = 最后发放率 (动力学结果, 非查表)

架构:
  输入 → 脉冲编码 (7 维 obs → 发放率)
  → 稀疏连接神经元层 (N 神经元, 每神经元 k 稀疏连接)
  → T 步 LIF 传播
  → 输出层: 最后发放率 → 3 动作 (softmax)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SpikeFunction(torch.autograd.Function):
    """硬阈值发放 + 替代梯度 (fast sigmoid surrogate)。

    前向: s = (u >= threshold) 硬脉冲 (真实动力学)
    反向: dL/du ≈ dL/ds · 1/(1+|10(u-th)|)²  (快速 sigmoid 导数)
    无替代梯度时 d(u>=th)/du ≡ 0 → LIF 层永远学不到 (实测 bug)。
    """

    @staticmethod
    def forward(ctx, u, threshold):
        ctx.save_for_backward(u)
        ctx.threshold = threshold
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors
        th = ctx.threshold
        # fast sigmoid surrogate: σ'(u-th) = 1/(1+|10(u-th)|)²
        grad = grad_output / (1.0 + (10.0 * (u - th)).abs()) ** 2
        return grad, None


class SparseLIFLayer(nn.Module):
    """稀疏连接 LIF 神经元层。

    每神经元连接前层的 k 个神经元 (随机稀疏图, 固定掩码)。
    LIF 动力学: u(t) = βu(t-1) + Σw·s(t-1); s(t) = 1 if u ≥ θ
    """

    def __init__(self, in_size: int, n_neurons: int, k: int = 8,
                 beta: float = 0.5, threshold: float = 1.0, seed: int = 0):
        super().__init__()
        self.n_neurons = n_neurons
        self.k = k
        self.beta = beta
        self.threshold = threshold
        # 稀疏连接掩码: 每神经元连 k 个输入 (固定随机图)
        rng = np.random.RandomState(seed)
        mask = torch.zeros(n_neurons, in_size)
        for i in range(n_neurons):
            idx = rng.choice(in_size, k, replace=False)
            mask[i, idx] = 1.0
        self.register_buffer("connect_mask", mask)
        # 突触权重 (仅连接处可学)
        self.weight = nn.Parameter(torch.randn(n_neurons, in_size) * 0.3)
        self.bias = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, s_in: torch.Tensor, T: int = 8) -> torch.Tensor:
        """s_in: (B, in_size) 输入发放率 [0,1] → (B, n_neurons) 输出发放率"""
        B = s_in.shape[0]
        w = self.weight * self.connect_mask  # (n, in) 稀疏权重
        u = torch.zeros(B, self.n_neurons, device=s_in.device)
        s = torch.zeros_like(u)
        s_history = []
        for t in range(T):
            # LIF: 膜电位更新 (输入脉冲 × 权重)
            u = self.beta * u + s_in @ w.T + self.bias
            # 发放 (替代梯度: 前向硬脉冲, 反向 fast sigmoid)
            s_new = SpikeFunction.apply(u, self.threshold)
            u = u - s_new * self.threshold  # 发放后复位
            s = s_new
            s_history.append(s)
        # 输出 = 平均发放率 (最后 4 步)
        out = torch.stack(s_history[-4:]).mean(0)  # (B, n)
        return out

    def forward_u(self, s_in: torch.Tensor, T: int = 8) -> torch.Tensor:
        """返回最终膜电位 (连续量, 无脉冲量化) — 世界模型读出用。"""
        B = s_in.shape[0]
        w = self.weight * self.connect_mask
        u = torch.zeros(B, self.n_neurons, device=s_in.device)
        for t in range(T):
            u = self.beta * u + s_in @ w.T + self.bias
            s_new = SpikeFunction.apply(u, self.threshold)
            u = u - s_new * self.threshold
        return u


class SparseSNN(nn.Module):
    """完整稀疏 SNN: 输入感知 → 两层稀疏 LIF → 动作。"""

    def __init__(self, obs_size: int = 6, n1: int = 256, n2: int = 256,
                 k1: int = 6, k2: int = 12, n_actions: int = 3, T: int = 8):
        super().__init__()
        self.T = T
        self.layer1 = SparseLIFLayer(obs_size, n1, k=k1)
        self.layer2 = SparseLIFLayer(n1, n2, k=k2)
        self.action_head = nn.Linear(n2, n_actions)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """obs: (B, 7) → (action_logits, stats)"""
        s1 = self.layer1(obs, self.T)      # (B, n1) 发放率
        s2 = self.layer2(s1, self.T)       # (B, n2) 发放率
        logits = self.action_head(s2)
        stats = {
            "l1_rate": s1,
            "l2_rate": s2,
        }
        return logits, stats


if __name__ == "__main__":
    net = SparseSNN()
    obs = torch.randn(4, 7)
    logits, stats = net(obs)
    print(f"动作 logits: {logits.shape}")
    print(f"L2 发放率: {stats['l2_rate'].shape}, 均值={stats['l2_rate'].mean():.3f}")
    assert torch.isfinite(logits).all()
    print("稀疏 SNN 冒烟通过 ✓")
