"""train_quantity.py — 从连续量学数 (量变到质变, 量化涌现进位)。

婴儿式: 先有量感 (连续量) → 量化成数字
输入: 连续数量 x (0-20)
输出: 数字 (0-20)
学习: 量化映射 (x → round(x)) — 阈值自动学!

进位 = 量化阈值 (自然涌现):
  量 9.3 → 9; 量 9.8 → 10 (量化阈值 ≈ 9.5!)
  = 进位不是规则, 是"量变到质变" (量化溢出)!

验证: 量化精度 / 数数 (量递增) / 进位点 (9→10 阈值)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lif_pool import LIFPool


class QuantityModel(nn.Module):
    """连续量 → LIF → 数字。量化 = 学习的映射 (阈值涌现)。"""
    def __init__(self, n_digit=21, d=32, pool=256, theta=0.3):
        super().__init__()
        self.inp = nn.Linear(1, d)          # 连续量 → 特征
        self.pool = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=12)
        self.head = nn.Linear(d, n_digit)   # 数字 (0-20)

    def forward(self, x, spikes_out=False):
        z = torch.tanh(self.inp(x))
        out, spikes = self.pool(z.unsqueeze(1))
        logits = self.head(out[:, -1])
        return (logits, spikes) if spikes_out else logits


def gen_data(n=30000, seed=42, lo=0.0, hi=20.0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(lo, hi, n).astype(np.float32)
    y = np.round(x).astype(np.int64)  # 量化目标 (round)
    return torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 从连续量学数: 量化涌现进位 (量变到质变) ===")
    model = QuantityModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    X, Y = gen_data()
    n = int(0.9 * len(X))
    Xd, Yd = X.to(dev), Y.to(dev)

    for ep in range(100):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i+512]
            loss = F.cross_entropy(model(Xd[idx]), Yd[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    # 验证: 量化精度
    model.eval()
    with torch.no_grad():
        acc = (model(Xd[n:]).argmax(-1) == Yd[n:]).float().mean()
    print(f"量化精度 (0-20): {acc.item():.3f}")

    # 验证: 量化边界 (进位点!)
    print("\n=== 量化阈值 (进位点 = 量变到质变) ===")
    with torch.no_grad():
        for x in [8.4, 8.9, 9.1, 9.4, 9.5, 9.6, 9.9, 10.1, 10.5]:
            logits = model(torch.tensor([[x]]).to(dev))
            pred = int(logits.argmax(-1))
            gold = int(round(x))
            print(f"  量 {x} → 数字 {pred} (round={gold}) {'✓' if pred==gold else '✗'}")

    # 验证: 数数 (量递增)
    print("\n=== 数数 (连续量递增 → 数字序列) ===")
    with torch.no_grad():
        seq = []
        for x in np.arange(0.1, 13, 1.0):
            logits = model(torch.tensor([[float(x)]]).to(dev))
            seq.append(int(logits.argmax(-1)))
    print(f"量 0.1,1.1,...  → 数字: {seq}")
    print(f"{'✅ 数数从量涌现 (含进位 9→10!)' if seq == list(range(13)) else '⚠️ 检查'}")


if __name__ == "__main__":
    main()
