"""prove_reasoning.py — 证明推理有效: 消融 + 序列依赖任务。

设计 (查表必败):
  一致链: 每步 = 规则 (A→B→C→D→A), 输出下一符号 (确定)
  含跳跃链: 中间某步 ≠ 规则 → "不确定" (需要看中间位置!)
  
单符号模型 (查表): 只看最后符号 → 无法检测中间跳跃 → 必败
LIF 序列模型: 积累 → 检测跳跃 → 正确判断

对比: 确定 acc + 不确定 acc → 证明序列推理 (非查表)
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

SYMBOLS = ["A", "B", "C", "D"]
RULE = {"A": "B", "B": "C", "C": "D", "D": "A"}
RULE_IDX = [1, 2, 3, 0]


def gen_seq_data(n=20000, seed=42, max_len=4):
    """一致链 (确定) vs 含跳跃链 (中间步错 → 不确定)。"""
    rng = np.random.RandomState(seed)
    X, Y, C = [], [], []
    for _ in range(n):
        L = rng.randint(2, max_len + 1)
        start = rng.randint(4)
        seq = [start]
        for i in range(1, L):
            if rng.rand() < 0.8:
                seq.append(RULE_IDX[seq[-1]])     # 一致
            else:
                seq.append(rng.randint(4))        # 跳跃 (中间错)
        X.append(seq)
        # 判断: 最后一步一致 → 下一符号确定; 中间有跳跃 → 不确定
        consistent = all(seq[i+1] == RULE_IDX[seq[i]] for i in range(L-1))
        if consistent:
            Y.append(RULE_IDX[seq[-1]]); C.append(1.0)
        else:
            Y.append(4); C.append(0.0)            # 不确定 (含跳跃)
    max_len = max(len(s) for s in X)
    Xp = np.zeros((len(X), max_len), np.int64)
    for i, s in enumerate(X):
        Xp[i, :len(s)] = s
    return (torch.from_numpy(Xp).long(), torch.from_numpy(np.array(Y)).long(),
            torch.from_numpy(np.array(C)).float())


class LIFModel(nn.Module):
    """LIF 序列模型 (完整动力学链)。"""
    def __init__(self, n_sym=4, d=64, pool=512, theta=0.3):
        super().__init__()
        self.embed = nn.Embedding(n_sym, d)
        self.pool = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=24)
        self.head = nn.Linear(d, n_sym + 1)

    def forward(self, seq):
        out, spikes = self.pool(torch.tanh(self.embed(seq)))
        return self.head(out[:, -1]), spikes


class SingleSymModel(nn.Module):
    """单符号模型 (查表对照): 只看最后符号, 无序列。"""
    def __init__(self, n_sym=4, d=64):
        super().__init__()
        self.embed = nn.Embedding(n_sym, d)
        self.head = nn.Linear(d, n_sym + 1)

    def forward(self, seq):
        last = seq[:, -1]
        return self.head(torch.tanh(self.embed(last))), None


def train_eval(model, Xtr, Ytr, Ctr, Xte, Yte, Cte, dev, name="", epochs=60):
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for ep in range(int(epochs)):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            logits, _ = model(Xtr[idx].to(dev))
            target = Ytr[idx].to(dev); conf = Ctr[idx].to(dev)
            lp = F.log_softmax(logits, -1)
            loss = -(lp[range(len(idx)), target] * conf).mean() \
                   - (lp[:, -1] * (1 - conf)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xte.to(dev))
        pred = logits.argmax(-1).cpu()
        det_acc = ((pred == Yte) * (Cte > 0)).sum() / (Cte > 0).sum()
        unc_acc = ((pred == 4) * (Cte < 1)).sum() / (Cte < 1).sum()
        all_acc = (pred == Yte).float().mean()
    print(f"[{name}] 确定acc {det_acc.item():.3f} | 不确定acc {unc_acc.item():.3f} "
          f"| 总acc {all_acc.item():.3f}")
    return det_acc.item(), unc_acc.item(), all_acc.item()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 证明推理: LIF序列 vs 单符号查表 (序列依赖任务) ===")
    X, Y, C = gen_seq_data(20000)
    n = int(0.9 * len(X))
    Xtr, Ytr, Ctr = X[:n], Y[:n], C[:n]
    Xte, Yte, Cte = X[n:], Y[n:], C[n:]
    print(f"数据: 确定 {C.mean().item():.2f} / 不确定 {1-C.mean().item():.2f}")

    print("\n--- 单符号模型 (查表: 只看最后符号) ---")
    train_eval(SingleSymModel(), Xtr, Ytr, Ctr, Xte, Yte, Cte, dev, "单符号查表")
    print("\n--- LIF 序列模型 (完整动力学积累) ---")
    train_eval(LIFModel(), Xtr, Ytr, Ctr, Xte, Yte, Cte, dev, "LIF序列")


if __name__ == "__main__":
    main()
