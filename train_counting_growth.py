"""train_counting_growth.py — 数数自我训练 + 进位处目标驱动生长。

设计:
  ① 自我训练: 生成-验证-改进 (从 0 数到错, 修正训练)
  ② 目标驱动生长: 进位时刻 (个位=9 → 10/20/30) = 难处
     → 生长专门神经元 (进位检测器!)
  ③ 验证: 从 99 数 (进位规则泛化?) + 删生长神经元 (功能性)

架构: 嵌入 + LIFPool (含生长) + 数位头
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lif_pool import LIFPool


def to_seq(n): return [n // 10, n % 10]


class CountingGrowth(nn.Module):
    """LIF 池 + 数位头 (可生长: 进位检测神经元)。"""
    def __init__(self, n_digit=10, d=32, pool=256, theta=0.3):
        super().__init__()
        self.embed = nn.Embedding(n_digit, d)
        self.pool = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=12)
        self.head_t = nn.Linear(d, 10)
        self.head_o = nn.Linear(d, 10)

    def forward(self, seq):
        out, spikes = self.pool(torch.tanh(self.embed(seq)))
        h = out[:, -1]
        return self.head_t(h), self.head_o(h), spikes

    def predict_next(self, n, device):
        seq = torch.tensor([[n // 10, n % 10]]).to(device)
        with torch.no_grad():
            pt, po, _ = self(seq)
        return int(pt.argmax(-1)) * 10 + int(po.argmax(-1))

    def grow_at_carry(self, carry_samples, device):
        """目标驱动生长: 进位样本 (个位=9) 的池发放神经元 → 克隆。"""
        if not carry_samples:
            return 0
        seq = torch.tensor([to_seq(n) for n in carry_samples]).to(device)
        with torch.no_grad():
            _, _, spikes = self(seq)
        return self.pool.grow(spikes, n=2, alpha=0.2)


def count_from(model, start, max_n=60, device="cuda"):
    chain = [start]
    cur = start
    for _ in range(max_n):
        nxt = model.predict_next(cur, device)
        if nxt != cur + 1:
            break
        chain.append(nxt)
        cur = nxt
    return chain


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 数数自我训练 + 进位处生长 (进位检测神经元) ===")
    model = CountingGrowth().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    for rnd in range(15):
        # ① 自主生成 (从 0 数到错)
        chain = count_from(model, 0, max_n=60, device=dev)
        # ② 验证修正样本: (n → n+1)
        samples = [(n, n + 1) for n in chain]
        if len(chain) < 60:
            samples.append((chain[-1], chain[-1] + 1))  # 修正错的
        # ③ 学习
        tr_in = torch.tensor([to_seq(n) for n, _ in samples]).to(dev)
        tr_t = torch.tensor([t // 10 for _, t in samples]).to(dev)
        tr_o = torch.tensor([t % 10 for _, t in samples]).to(dev)
        model.train()
        for _ in range(30):
            pt, po, _ = model(tr_in)
            loss = F.cross_entropy(pt, tr_t) + F.cross_entropy(po, tr_o)
            opt.zero_grad(); loss.backward(); opt.step()
        # ④ 目标驱动生长: 进位样本 (个位=9)
        carry = [n for n in chain if n % 10 == 9]
        n_grow = model.grow_at_carry(carry, dev)
        if rnd % 5 == 4 or n_grow:
            print(f"  轮 {rnd+1}: 数到 {chain[-1]}, 进位样本 {len(carry)}, "
                  f"生长 {n_grow} (总 {len(model.pool.growth_log)})")

    # 验证: 从 99 数 (进位规则泛化?)
    c99 = count_from(model, 99, max_n=20, device=dev)
    c50 = count_from(model, 50, max_n=20, device=dev)
    c0 = count_from(model, 0, max_n=60, device=dev)
    print(f"\n从 99 数: {c99}")
    print(f"从 50 数: {c50}")
    print(f"从 0 数: {c0}")
    ok_99 = c99 and c99[-1] > 99
    print(f"\n{'✅ 进位规则泛化 (99→100+!)' if ok_99 else '❌ 仍查表'}")
    print(f"生长神经元: {len(model.pool.growth_log)}")

    # 删生长神经元 → 功能验证
    if model.pool.growth_log:
        backup = model.pool.alive.clone()
        with torch.no_grad():
            for nid in model.pool.growth_log:
                model.pool.alive[nid] = False
        c99_del = count_from(model, 99, max_n=20, device=dev)
        model.pool.alive = backup
        ok_del = c99_del and c99_del[-1] > 99
        print(f"删生长后从 99 数: {c99_del}")
        print(f"{'✅ 生长神经元 = 进位功能 (删了失效!)' if ok_99 and not ok_del else '⚠️ 生长非关键'}")


if __name__ == "__main__":
    main()
