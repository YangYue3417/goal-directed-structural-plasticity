"""self_train_counting.py — 自我训练 (o1 式): 生成-验证-改进 循环。

"自己训练自己": 
  ① 生成: 模型自主数数 (从 0 开始, 每步预测下一)
  ② 验证: 可计算检查 (next == n+1) — 无外部标注!
  ③ 改进: 用验证修正的数据监督训练 (教师 = 计算)
  ④ 迭代: 反复 → 渐进突破 (0-9 → 进位 10 → 19-20 → 十位)

关键: 模型自己数到哪 → 修正到哪 → 课程自我引导 (试错教学)
最终: 能否自我发现进位规则 → 举一反三 (20-99)?
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_counting import CountingModel, to_seq, eval_count


def self_generate(model, dev, max_n=120):
    """模型自主数数: 从 0 开始, 每步预测下一; 返回链与每步对错。"""
    model.eval()
    chain, corrects = [], []
    with torch.no_grad():
        cur = 0
        for i in range(max_n):
            seq = torch.tensor([to_seq(cur)]).to(dev)
            pt, po = model(seq)
            nxt = int(pt.argmax(-1)) * 10 + int(po.argmax(-1))
            chain.append(cur)
            ok = (nxt == cur + 1)
            corrects.append(ok)
            if not ok:  # 数错 → 停止 (试错)
                break
            cur = nxt
    return chain, corrects


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 自我训练: 生成-验证-改进 (模型自己教自己数数) ===")
    model = CountingModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    for rnd in range(8):
        # ① 生成: 自主数数 (数到错为止)
        chain, corrects = self_generate(model, dev)
        reach = chain[-1] if chain else 0
        n_ok = sum(corrects)
        # ② 验证 + ③ 改进: 用 (n → n+1) 修正监督 (教师=计算!)
        # 只教"数对的延续 + 错的下一个" (验证修正)
        teach = []
        for n in chain:
            teach.append((n, n + 1))          # 验证过的正确延续
        if len(chain) > 0 and not corrects[-1]:
            teach.append((chain[-1], chain[-1] + 1))  # 修正错的那步!
        # 训练
        model.train()
        tr_in = torch.tensor([to_seq(n) for n, _ in teach]).to(dev)
        tr_t = torch.tensor([t // 10 for _, t in teach]).to(dev)
        tr_o = torch.tensor([t % 10 for _, t in teach]).to(dev)
        for ep in range(40):
            pt, po = model(tr_in)
            loss = F.cross_entropy(pt, tr_t) + F.cross_entropy(po, tr_o)
            opt.zero_grad(); loss.backward(); opt.step()
        # 记录
        print(f"轮 {rnd+1}: 数到 {reach} (正确 {n_ok}/{len(chain)}) "
              f"教 {len(teach)} 个延续 + 修正 {1 if len(chain)>0 and not corrects[-1] else 0}")

    # 最终验证
    acc_tr = eval_count(model, 0, 19, dev)
    acc_te = eval_count(model, 20, 50, dev)
    acc_te2 = eval_count(model, 50, 98, dev)
    print(f"\n最终: 0-19 {acc_tr:.3f} | 20-49 {acc_te:.3f} | 50-98 {acc_te2:.3f}")
    # 展示自我数数链
    chain, corrects = self_generate(model, dev)
    print(f"自主数数: {chain}")
    print(f"{'✅ 自我训练: 数到 ' + str(chain[-1]) if chain and len(chain) > 30 else '⚠️ 有限'}")


if __name__ == "__main__":
    main()
