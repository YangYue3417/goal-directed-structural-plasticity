"""train_counting.py — 位置系统数数 + 举一反三 (进位规则泛化)。

位值系统: 数字 = [十位, 个位] 序列 ("10" = [1,0], "23" = [2,3])
数数规则: 个位+1; 个位=9 → 十位+1, 个位=0 (进位!)

举一反三: 训练 0-19 (20 个) → 测试 20-99 (80 个未见)
  如果进位规则泛化 → 数任意大 (能数到 99+)

架构: 数字序列 (2 token) → LIF → 预测下一序列 (十位, 个位)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lif_pool import LIFPool


def to_seq(n):
    """数字 → [十位, 个位] (0-9 → [0, n])。"""
    return [n // 10, n % 10]


class CountingModel(nn.Module):
    def __init__(self, n_digit=10, d=32, pool=256, theta=0.3):
        super().__init__()
        self.embed = nn.Embedding(n_digit, d)
        self.lif = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=12)
        self.head_t = nn.Linear(d, 10)  # 下一十位
        self.head_o = nn.Linear(d, 10)  # 下一个位

    def forward(self, seq):
        """seq: (B, 2) [十位, 个位] → 下一序列预测。"""
        out, _ = self.lif(torch.tanh(self.embed(seq)))
        h = out[:, -1]
        return self.head_t(h), self.head_o(h)


def eval_count(model, lo, hi, dev):
    """数数链: 从 lo 数到 hi, 每步预测下一。返回正确率。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for n in range(lo, hi):
            seq = torch.tensor([to_seq(n)]).to(dev)
            pt, po = model(seq)
            next_n = int(pt.argmax(-1)) * 10 + int(po.argmax(-1))
            gold = n + 1
            correct += (next_n == gold)
            total += 1
    return correct / max(total, 1)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 位置系统数数: 训练 0-19 → 举一反三测试 20-99 ===")
    model = CountingModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    # 训练: 0-19 (数数对)
    tr_in = torch.tensor([to_seq(n) for n in range(20)]).to(dev)
    tr_t = torch.tensor([(n + 1) // 10 for n in range(20)]).to(dev)
    tr_o = torch.tensor([(n + 1) % 10 for n in range(20)]).to(dev)

    for ep in range(300):
        pt, po = model(tr_in)
        loss = F.cross_entropy(pt, tr_t) + F.cross_entropy(po, tr_o)
        opt.zero_grad(); loss.backward(); opt.step()

    # 训练范围检查
    acc_tr = eval_count(model, 0, 19, dev)
    print(f"训练范围 (0-19): 数数 acc {acc_tr:.3f}")

    # 举一反三: 未见范围
    acc_te = eval_count(model, 20, 50, dev)
    acc_te2 = eval_count(model, 50, 99, dev)
    print(f"举一反三 (20-49, 未见): {acc_te:.3f}")
    print(f"举一反三 (50-98, 未见): {acc_te2:.3f}")

    # 展示数数链
    chain = []
    with torch.no_grad():
        for n in range(15, 32):
            seq = torch.tensor([to_seq(n)]).to(dev)
            pt, po = model(seq)
            nxt = int(pt.argmax(-1)) * 10 + int(po.argmax(-1))
            chain.append(f"{n}→{nxt}")
    print(f"数数链 (15-31): {' '.join(chain)}")

    print(f"\n{'✅ 举一反三: 进位规则泛化 (数到99+)' if acc_te > 0.9 and acc_te2 > 0.7 else '⚠️ 泛化有限'}")


if __name__ == "__main__":
    main()
