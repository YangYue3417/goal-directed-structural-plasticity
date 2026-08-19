"""train_number_sense.py — 数感教学: 从 0 教小孩理解数字。

儿童数字学习课程 (数感发展):
  阶段 1 数数: 0,1,2,... 顺序 (前后关系) — 符号顺序
  阶段 2 数量 tag: 符号 ↔ 数量 (5 = 五个单位) — 数量概念
  阶段 3 大小比较: 3 < 5 (数量多少 → 符号比较) — 数值语义
  阶段 4 (后续): 算术

设计: 数字符号 → 嵌入 → 数量 tag (内部数量表征)
验证: 嵌入 PCA 数值线 (单调) + 大小比较正确 = 理解数字
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumberModel(nn.Module):
    """数字嵌入 (学数量结构) + 数数头 + 大小比较头。"""
    def __init__(self, n_digit=10, d=32):
        super().__init__()
        self.embed = nn.Embedding(n_digit, d)
        self.next_head = nn.Linear(d, n_digit)   # 数数: 预测下一个
        self.cmp_head = nn.Linear(d * 2, 1)      # 大小: (a,b) → a<b?

    def forward(self, a, b=None):
        ea, eb = self.embed(a), self.embed(b) if b is not None else None
        next_logits = self.next_head(ea)
        if eb is not None:
            cmp = self.cmp_head(torch.cat([ea, eb], -1)).squeeze(-1)
            return next_logits, cmp
        return next_logits, None


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 数感教学: 从 0 教小孩 (数数 → 数量 → 比较) ===")
    model = NumberModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    digits = torch.arange(10).to(dev)
    # 数数训练对: (0→1, 1→2, ..., 8→9)
    nxt_in = digits[:9]; nxt_tgt = digits[1:]
    # 大小比较: 所有 (a,b) 对, a<b → 1
    pairs = [(a, b) for a in range(10) for b in range(10) if a != b]
    cmp_in = torch.tensor(pairs).to(dev)
    cmp_tgt = torch.tensor([1.0 if a < b else 0.0 for a, b in pairs]).to(dev)

    for ep in range(200):
        model.train()
        # 阶段 1: 数数 (顺序)
        nl, _ = model(nxt_in)
        loss_next = F.cross_entropy(nl, nxt_tgt)
        # 阶段 2+3: 大小比较 (数量语义)
        _, c = model(cmp_in[:, 0], cmp_in[:, 1])
        loss_cmp = F.binary_cross_entropy_with_logits(c, cmp_tgt)
        loss = loss_next + loss_cmp
        opt.zero_grad(); loss.backward(); opt.step()

    # 验证 1: 数数正确率
    model.eval()
    with torch.no_grad():
        nl, _ = model(nxt_in)
        acc_next = (nl.argmax(-1) == nxt_tgt).float().mean()
        _, c = model(cmp_in[:, 0], cmp_in[:, 1])
        acc_cmp = ((c > 0) == cmp_tgt.bool()).float().mean()

    # 验证 2: 数字嵌入数值线 (PCA 单调)
    emb = model.embed.weight.detach().cpu().numpy()
    U, S, Vt = np.linalg.svd(emb - emb.mean(0), full_matrices=False)
    proj = (emb - emb.mean(0)) @ Vt.T
    pc1 = proj[:, 0]
    order = np.argsort(pc1)
    is_sorted = np.array_equal(order, np.arange(10)) or np.array_equal(order, np.arange(10)[::-1])

    print(f"数数 acc: {acc_next.item():.3f} | 大小比较 acc: {acc_cmp.item():.3f}")
    print(f"数字嵌入 PCA 顺序: {order.tolist()}")
    print(f"{'✅ 数值线形成 (理解数字顺序!)' if is_sorted else '⚠️ 数值线未完全形成'}")
    print(f"PCA 投影: {np.round(pc1, 2)}")


if __name__ == "__main__":
    main()
