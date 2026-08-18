"""train_integrated.py — 知识整合: 共享 LIF 逻辑引擎 + 多任务头。

问题: 单独学习的知识 (符号逻辑/受控语言/数学) 能整合吗?
方案: 统一词表 + 共享嵌入 + 共享 LIF (通用逻辑引擎) + 每任务头
训练: 交替采样各任务 (多任务学习)
验证: 整合后各任务 acc vs 独立训练 (共享是否保留/迁移)

任务:
  ① 符号逻辑: 序列 → 下一符号/不确定 (5 类)
  ② 受控语言: 句对 → 蕴含/矛盾/无关/不确定 (4 类)
  ③ 数学: 前提+结论 → 有效/无效/不确定 (3 类)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from lif_pool import LIFPool

# ============ 统一词表 ============
TOKENS = ["<pad>"]
# 符号逻辑 (A-D)
TOKENS += ["A", "B", "C", "D"]
# 受控语言
TOKENS += ["所有", "一些", "没有", "鸟", "鱼", "狗", "花", "猫",
           "会", "不会", "飞", "游泳", "叫", "跑"]
# 数学
TOKENS += ["+", "=", "v"] + [str(i) for i in range(10)]
VOCAB = {t: i for i, t in enumerate(TOKENS)}


def gen_symbol(n=6000, seed=1):
    from prove_general import gen_multi_rule
    X, Y, C = gen_multi_rule(n, seed)
    return X, Y.long(), C.long(), 5


def gen_lang(n=6000, seed=2):
    from train_controlled_lang import gen_data
    X1, X2, Y = gen_data(n, seed)
    return (X1, X2), Y.long(), None, 4


def gen_math(n=6000, seed=3):
    from train_math_logic import gen_data
    P, C, Y = gen_data(n, seed)
    return (P, C), Y.long(), None, 3


class IntegratedModel(nn.Module):
    """共享嵌入 + 共享 LIF 逻辑引擎 + 任务头。"""
    def __init__(self, vocab, d=64, pool=512, theta=0.3):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), d)
        self.lif = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=24)
        self.head_sym = nn.Linear(d, 5)
        self.head_lang = nn.Linear(d, 4)
        self.head_math = nn.Linear(d * 3, 3)   # 2 前提 + 结论

    def seq_state(self, x):
        """序列 → LIF 积累 → 状态。x: (B, L) 或 (B, N, L)。"""
        if x.dim() == 3:  # 多序列 (数学: 2 前提 + 结论)
            outs = []
            for j in range(x.shape[1]):
                out, _ = self.lif(torch.tanh(self.embed(x[:, j])))
                outs.append(out[:, -1])
            return torch.cat(outs, -1)
        out, _ = self.lif(torch.tanh(self.embed(x)))
        return out[:, -1]

    def forward(self, task, x):
        if task == 0:   # 符号: 序列 → 状态
            return self.head_sym(self.seq_state(x))
        if task == 1:   # 语言: 两句拼接
            x1, x2 = x
            return self.head_lang(self.seq_state(torch.cat([x1, x2], 1)))
        if task == 2:   # 数学: 前提+结论
            P, C = x
            state = torch.cat([self.seq_state(P), self.seq_state(C)], -1)
            return self.head_math(state)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 知识整合: 共享 LIF 逻辑引擎 + 多任务头 ===")
    # 各任务数据
    sym = gen_symbol(); lang = gen_lang(); math = gen_math()
    tasks = [
        ("符号逻辑", *sym),
        ("受控语言", *lang),
        ("数学推理", *math),
    ]
    model = IntegratedModel(VOCAB).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def eval_task(ti, X, Y, n_test):
        model.eval()
        with torch.no_grad():
            if ti == 0:
                acc = (model(0, X[n_test:].to(dev)).argmax(-1).cpu() == Y[n_test:]).float().mean()
            elif ti == 1:
                x1, x2 = X
                acc = (model(1, (x1[n_test:].to(dev), x2[n_test:].to(dev)))
                       .argmax(-1).cpu() == Y[n_test:]).float().mean()
            else:
                P, C = X
                acc = (model(2, (P[n_test:].to(dev), C[n_test:].to(dev)))
                       .argmax(-1).cpu() == Y[n_test:]).float().mean()
        return acc.item()

    n_tests = []
    for name, X, Y, C, n_cls in tasks:
        n_tests.append(len(X[0]) if isinstance(X, tuple) else len(X))

    for ep in range(60):
        model.train()
        # 交替训练 3 任务
        for ti in range(3):
            name, X, Y, C, n_cls = tasks[ti]
            n_use = int(len(Y) * 0.9)
            if ti == 0:
                Xd, Yd = X[:n_use].to(dev), Y[:n_use].to(dev)
                loss = F.cross_entropy(model(0, Xd), Yd)
            elif ti == 1:
                x1, x2 = X
                loss = F.cross_entropy(model(1, (x1[:n_use].to(dev), x2[:n_use].to(dev))),
                                      Y[:n_use].to(dev))
            else:
                P, C = X
                loss = F.cross_entropy(model(2, (P[:n_use].to(dev), C[:n_use].to(dev))),
                                      Y[:n_use].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 15 == 14:
            accs = [eval_task(i, tasks[i][1], tasks[i][2], int(len(tasks[i][2])*0.1))
                    for i in range(3)]
            print(f"  epoch {ep+1}: 符号 {accs[0]:.3f} | 语言 {accs[1]:.3f} | 数学 {accs[2]:.3f}")

    # 最终 (整合)
    accs = [eval_task(i, tasks[i][1], tasks[i][2], int(len(tasks[i][2])*0.1)) for i in range(3)]
    print(f"\n整合模型: 符号 {accs[0]:.3f} | 语言 {accs[1]:.3f} | 数学 {accs[2]:.3f}")
    print("对照 (独立): 符号 0.860 | 语言 0.957 | 数学 0.78")
    print(f"{'✅ 整合可行 (共享引擎保留性能)' if accs[0]>0.7 and accs[1]>0.8 else '⚠️ 有干扰'}")


if __name__ == "__main__":
    main()
