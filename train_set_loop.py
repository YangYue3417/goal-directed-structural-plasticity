"""train_set_loop.py — 持续训练: 新数据 → 外推检查 → 涌现?

循环:
  ① 生成新训练集 (每轮不同 seed → 不重复集合!)
  ② 增量训练
  ③ 外推评估 (大集合, 固定测试集)
  ④ 涌现? (外推 acc 显著提升 → 记录) 否则继续

确保: 测试集固定 (一致评估) + 训练集每轮新 (不重复)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from train_set_cardinality import SetCardinality, gen_set


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    N_ELEM = 20
    print("=== 持续训练: 新数据 → 外推涌现? ===")
    model = SetCardinality(N_ELEM).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    # 固定测试集 (外推评估一致): 大集合 ≤20
    S1t, C1t, S2at, S2bt, C2t = gen_set(N_ELEM, 20, 5000, seed=999)
    S1t, C1t, S2at, S2bt, C2t = (S1t.to(dev), C1t.to(dev),
                                 S2at.to(dev), S2bt.to(dev), C2t.to(dev))

    best_card, best_add = 0, 0
    for rnd in range(30):
        # ① 新训练集: seed = rnd → 不同集合 (不重复)
        S1, C1, S2a, S2b, C2 = gen_set(N_ELEM, 12, 20000, seed=rnd * 7 + 1)
        model.train()
        for i in range(0, len(S1), 512):
            idx = torch.randperm(len(S1))[:512]
            l1 = F.cross_entropy(model.card(S1[idx].to(dev)), C1[idx].to(dev))
            l2 = F.cross_entropy(model.add(S2a[idx].to(dev), S2b[idx].to(dev)), C2[idx].to(dev))
            opt.zero_grad(); (l1 + l2).backward(); opt.step()
        # ③ 外推评估
        model.eval()
        with torch.no_grad():
            a1 = (model.card(S1t).argmax(-1) == C1t).float().mean().item()
            a2 = (model.add(S2at, S2bt).argmax(-1) == C2t).float().mean().item()
        improved = a1 > best_card + 0.02 or a2 > best_add + 0.02
        best_card, best_add = max(best_card, a1), max(best_add, a2)
        mark = " ↑涌现!" if improved else ""
        print(f"轮 {rnd+1}: 外推 基数 {a1:.3f} 加法 {a2:.3f} "
              f"(best {best_card:.3f}/{best_add:.3f}){mark}")
        if a1 > 0.85 and a2 > 0.6:
            print(f"✅ 涌现达成: 外推泛化 (结构学习!)")
            break

    print(f"\n最终: 基数 {best_card:.3f} 加法 {best_add:.3f}")


if __name__ == "__main__":
    main()
