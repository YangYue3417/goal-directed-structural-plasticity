"""train_growth_addition.py — 学一部分生长一部分 (渐进专群)。

发育式数学学习:
  阶段 1: 学"一位加法无进位" (0-9+0-9, 和≤9) → 稳定 (旧神经元)
  生长:   进位样本 (8+3, 9+5...) 激活的神经元 → 生长新神经元 (进位专群!)
  阶段 2: 学"带进位加法" (全) → 新神经元学进位, 旧神经元保留

验证: 旧能力 (无进位) 保留 + 进位学会 + 删生长神经元 (进位崩?)
= "学一部分生长一部分": 每部分能力 = 生长的新神经元群!
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lif_pool import LIFPool


class AddModel(nn.Module):
    """一位加法: (a, b) → 和 (0-18) + 进位标志。"""
    def __init__(self, n_digit=10, d=32, pool=256, theta=0.3):
        super().__init__()
        self.embed = nn.Embedding(n_digit, d)
        self.pool = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=12)
        self.head = nn.Linear(d, 19)  # 和 0-18

    def forward(self, a, b):
        za = torch.tanh(self.embed(a))
        zb = torch.tanh(self.embed(b))
        seq = torch.stack([za, zb], 1)
        out, spikes = self.pool(seq)
        return self.head(out[:, -1]), spikes


def gen_pairs(max_sum=9, n=20000, seed=42):
    """加法对: 和 ≤ max_sum (无进位) 或全部。"""
    rng = np.random.RandomState(seed)
    A, B, S = [], [], []
    while len(A) < n:
        a, b = rng.randint(10), rng.randint(10)
        if a + b <= max_sum:
            A.append(a); B.append(b); S.append(a + b)
    return (torch.tensor(A).long(), torch.tensor(B).long(), torch.tensor(S).long())


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 学一部分生长一部分: 加法渐进专群 ===")
    model = AddModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    # 阶段 1: 无进位加法 (和≤9) — 旧神经元
    A1, B1, S1 = gen_pairs(9, 20000, 42)
    A1d, B1d, S1d = A1.to(dev), B1.to(dev), S1.to(dev)
    for ep in range(40):
        idx = torch.randperm(len(A1))[:512]
        logits, _ = model(A1d[idx], B1d[idx])
        loss = F.cross_entropy(logits, S1d[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        a1 = (model(A1d[:2000], B1d[:2000])[0].argmax(-1) == S1d[:2000]).float().mean()
    print(f"阶段 1 (无进位): 加法 acc {a1.item():.3f} (活跃 {model.pool.n_active()})")

    # 生长: 进位样本 (8+3, 9+5... 和>9) 激活的神经元 → 新专群
    A_c, B_c, S_c = gen_pairs(18, 20000, 7)
    carry_mask = (A_c + B_c > 9)
    Ac, Bc, Sc = A_c[carry_mask][:4000], B_c[carry_mask][:4000], S_c[carry_mask][:4000]
    with torch.no_grad():
        _, spikes = model(Ac.to(dev), Bc.to(dev))
    n_grow = model.pool.grow(spikes, n=6, alpha=0.2)
    print(f"生长: {n_grow} 个进位专群神经元 (总 {len(model.pool.growth_log)})")

    # 阶段 2: 全部加法 (含进位) — 新神经元学进位
    A2, B2, S2 = gen_pairs(18, 30000, 11)
    A2d, B2d, S2d = A2.to(dev), B2.to(dev), S2.to(dev)
    for ep in range(60):
        idx = torch.randperm(len(A2))[:512]
        logits, _ = model(A2d[idx], B2d[idx])
        loss = F.cross_entropy(logits, S2d[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    # 验证: 旧保留 + 进位学会
    model.eval()
    with torch.no_grad():
        a_nc = (model(A1d[:2000], B1d[:2000])[0].argmax(-1) == S1d[:2000]).float().mean()
        carry_test = (A2d + B2d > 9)
        ct = model(A2d[carry_test][:2000], B2d[carry_test][:2000])[0].argmax(-1)
        a_c = (ct == S2d[carry_test][:2000]).float().mean()
        a_all = (model(A2d[:3000], B2d[:3000])[0].argmax(-1) == S2d[:3000]).float().mean()
    print(f"阶段 2 后: 无进位保留 {a_nc.item():.3f} | 进位 {a_c.item():.3f} | 全部 {a_all.item():.3f}")

    # 删生长神经元 → 进位功能?
    if model.pool.growth_log:
        backup = model.pool.alive.clone()
        with torch.no_grad():
            for nid in model.pool.growth_log:
                model.pool.alive[nid] = False
        with torch.no_grad():
            ct2 = model(A2d[carry_test][:2000], B2d[carry_test][:2000])[0].argmax(-1)
            a_c_del = (ct2 == S2d[carry_test][:2000]).float().mean()
            a_nc_del = (model(A1d[:2000], B1d[:2000])[0].argmax(-1) == S1d[:2000]).float().mean()
        model.pool.alive = backup
        print(f"删生长后: 进位 {a_c_del.item():.3f} (应降) | 无进位 {a_nc_del.item():.3f} (应保持)")
        print(f"{'✅ 生长专群 = 进位功能!' if a_c_del < a_c.item() - 0.05 and a_nc_del > a_nc.item() - 0.05 else '⚠️ 生长非专群'}")


if __name__ == "__main__":
    main()
