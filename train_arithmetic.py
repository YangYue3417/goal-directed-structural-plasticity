"""train_arithmetic.py — 算术 + 温故知新 (复习旧知识防遗忘)。

课程延续 (从 0 教小孩):
  已学: 数数 (顺序) + 大小比较 (数量语义)
  新学: 算术 (a+b=c 数量操作)

温故知新: 训练算术时交错复习数数/比较
  对比: 只学新 (遗忘旧) vs 温故知新 (保持旧 + 学新)

验证: 温故 (旧 acc 保持) + 知新 (算术 acc) + 不遗忘
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumberModel(nn.Module):
    def __init__(self, n_digit=10, d=32):
        super().__init__()
        self.embed = nn.Embedding(n_digit, d)
        self.next_head = nn.Linear(d, n_digit)      # 数数
        self.cmp_head = nn.Linear(d * 2, 1)         # 比较
        self.arith_head = nn.Linear(d * 2, n_digit) # 算术: a+b → c

    def next_logits(self, a): return self.next_head(self.embed(a))
    def cmp_logits(self, a, b): return self.cmp_head(torch.cat([self.embed(a), self.embed(b)], -1)).squeeze(-1)
    def arith_logits(self, a, b): return self.arith_head(torch.cat([self.embed(a), self.embed(b)], -1))


def eval_all(m, dev):
    m.eval()
    digits = torch.arange(10).to(dev)
    with torch.no_grad():
        # 数数
        nxt_in, nxt_tgt = digits[:9], digits[1:]
        acc_next = (m.next_logits(nxt_in).argmax(-1) == nxt_tgt).float().mean().item()
        # 比较
        pairs = torch.tensor([(a, b) for a in range(10) for b in range(10) if a != b]).to(dev)
        tgt = torch.tensor([1.0 if a < b else 0.0 for a, b in pairs.tolist()]).to(dev)
        acc_cmp = ((m.cmp_logits(pairs[:, 0], pairs[:, 1]) > 0) == tgt.bool()).float().mean().item()
        # 算术: 所有 a+b=c (0-9, c<10)
        ap = torch.tensor([(a, b) for a in range(10) for b in range(10) if a + b < 10]).to(dev)
        at = torch.tensor([a + b for a, b in ap.tolist()]).to(dev)
        acc_ar = (m.arith_logits(ap[:, 0], ap[:, 1]).argmax(-1) == at).float().mean().item()
    return acc_next, acc_cmp, acc_ar


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 算术 + 温故知新 (复习旧知识防遗忘) ===")
    digits = torch.arange(10).to(dev)
    nxt_in, nxt_tgt = digits[:9], digits[1:]
    cmp_pairs = torch.tensor([(a, b) for a in range(10) for b in range(10) if a != b]).to(dev)
    cmp_tgt = torch.tensor([1.0 if a < b else 0.0 for a, b in cmp_pairs.tolist()]).to(dev)
    ar_pairs = torch.tensor([(a, b) for a in range(10) for b in range(10) if a + b < 10]).to(dev)
    ar_tgt = torch.tensor([a + b for a, b in ar_pairs.tolist()]).to(dev)

    for mode in ["warm", "only_new"]:
        torch.manual_seed(42)
        m = NumberModel().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        # 阶段 1: 数数 + 比较 (旧知识)
        for ep in range(100):
            nl = m.next_logits(nxt_in)
            l1 = F.cross_entropy(nl, nxt_tgt)
            cl = m.cmp_logits(cmp_pairs[:, 0], cmp_pairs[:, 1])
            l2 = F.binary_cross_entropy_with_logits(cl, cmp_tgt)
            opt.zero_grad(); (l1 + l2).backward(); opt.step()
        a1, b1, _ = eval_all(m, dev)

        # 阶段 2: 学算术 (温故知新 vs 只学新)
        for ep in range(100):
            if mode == "warm":
                # 温故: 交错复习数数/比较 (1/3 时间)
                r = ep % 3
                if r == 0:
                    nl = m.next_logits(nxt_in)
                    loss = F.cross_entropy(nl, nxt_tgt)
                elif r == 1:
                    cl = m.cmp_logits(cmp_pairs[:, 0], cmp_pairs[:, 1])
                    loss = F.binary_cross_entropy_with_logits(cl, cmp_tgt)
                else:
                    al = m.arith_logits(ar_pairs[:, 0], ar_pairs[:, 1])
                    loss = F.cross_entropy(al, ar_tgt)
            else:
                # 只学新: 纯算术 (不复习)
                al = m.arith_logits(ar_pairs[:, 0], ar_pairs[:, 1])
                loss = F.cross_entropy(al, ar_tgt)
            opt.zero_grad(); loss.backward(); opt.step()

        a2, b2, c2 = eval_all(m, dev)
        print(f"[{mode}] 阶段1后: 数数 {a1:.3f} 比较 {b1:.3f} | "
              f"阶段2后: 数数 {a2:.3f} ({'保持' if a2>a1-0.05 else '遗忘!'}) "
              f"比较 {b2:.3f} 算术 {c2:.3f}")
        if mode == "warm":
            w = (a2, b2, c2)
        else:
            o = (a2, b2, c2)

    print(f"\n温故知新: 数数 {w[0]:.3f} 比较 {w[1]:.3f} 算术 {w[2]:.3f}")
    print(f"只学新:   数数 {o[0]:.3f} 比较 {o[1]:.3f} 算术 {o[2]:.3f}")
    print(f"{'✅ 温故知新有效 (旧知识保持 + 新算术学到)' if w[0] > o[0] + 0.05 else '⚠️ 对比'}")


if __name__ == "__main__":
    main()
