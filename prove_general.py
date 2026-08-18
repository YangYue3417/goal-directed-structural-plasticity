"""prove_general.py — 通用逻辑验证: 多规则训练 + 新规则测试。

任务 (规则无关): 检测序列"自洽性" (每步符合某规律)
  一致链: 每一步符合规律 → 下一符号可推 (确定)
  跳跃链: 某步打破规律 → 不确定

训练: 随机规则 (24 种排列) 生成序列 — 学"自洽检测" (非特定规则)
测试: 新随机规则 (未见排列) — 若 acc 高 → 规则无关的通用逻辑
      (不能背规则表: 测试规则训练没见过!)

单符号查表对照: 只看最后符号 → 无法检测中间跳跃 (必败)
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


def gen_multi_rule(n=20000, seed=42, max_len=4, rules=None):
    """随机规则生成: 一致链 (确定) vs 跳跃链 (不确定)。"""
    rng = np.random.RandomState(seed)
    all_rules = list(np.random.RandomState(1).permutation(4) for _ in range(24))
    if rules is None:
        rules = all_rules
    X, Y, C = [], [], []
    for _ in range(n):
        rule = rules[rng.randint(len(rules))]
        L = rng.randint(2, max_len + 1)
        start = rng.randint(4)
        seq = [start]
        for i in range(1, L):
            if rng.rand() < 0.8:
                seq.append(int(rule[seq[-1]]))     # 符合规则
            else:
                seq.append(rng.randint(4))         # 跳跃
        X.append(seq)
        consistent = all(seq[i+1] == rule[seq[i]] for i in range(L-1))
        if consistent:
            Y.append(int(rule[seq[-1]])); C.append(1.0)
        else:
            Y.append(4); C.append(0.0)
    max_len = max(len(s) for s in X)
    Xp = np.zeros((len(X), max_len), np.int64)
    for i, s in enumerate(X):
        Xp[i, :len(s)] = s
    return (torch.from_numpy(Xp).long(), torch.from_numpy(np.array(Y)).long(),
            torch.from_numpy(np.array(C)).float())


class LIFModel(nn.Module):
    def __init__(self, n_sym=4, d=64, pool=512, theta=0.3):
        super().__init__()
        self.embed = nn.Embedding(n_sym, d)
        self.pool = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=24)
        self.head = nn.Linear(d, n_sym + 1)

    def forward(self, seq):
        out, spikes = self.pool(torch.tanh(self.embed(seq)))
        return self.head(out[:, -1]), spikes


class SingleSymModel(nn.Module):
    def __init__(self, n_sym=4, d=64):
        super().__init__()
        self.embed = nn.Embedding(n_sym, d)
        self.head = nn.Linear(d, n_sym + 1)

    def forward(self, seq):
        return self.head(torch.tanh(self.embed(seq[:, -1]))), None


def train_eval(model, Xtr, Ytr, Ctr, Xte, Yte, Cte, dev, name, epochs=60):
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            logits, _ = model(Xtr[idx].to(dev))
            lp = F.log_softmax(logits, -1)
            loss = -(lp[range(len(idx)), Ytr[idx].to(dev)] * Ctr[idx].to(dev)).mean() \
                   - (lp[:, -1] * (1 - Ctr[idx].to(dev))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xte.to(dev))
        pred = logits.argmax(-1).cpu()
        det = ((pred == Yte) * (Cte > 0)).sum() / (Cte > 0).sum()
        unc = ((pred == 4) * (Cte < 1)).sum() / (Cte < 1).sum()
        tot = (pred == Yte).float().mean()
    print(f"[{name}] 确定 {det.item():.3f} | 跳跃检测 {unc.item():.3f} | 总 {tot.item():.3f}")
    return tot.item()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    all_rules = list(np.random.RandomState(1).permutation(4) for _ in range(24))
    train_rules = all_rules[:16]   # 训练: 16 种规则
    test_rules = all_rules[16:]    # 测试: 8 种新规则 (未见!)

    print("=== 通用逻辑: 多规则训练 (16) + 新规则测试 (8) ===")
    Xtr, Ytr, Ctr = gen_multi_rule(20000, 42, rules=train_rules)
    Xte, Yte, Cte = gen_multi_rule(4000, 7, rules=test_rules)
    print(f"训练: {len(Xtr)} (规则 {len(train_rules)}) | 测试: {len(Xte)} "
          f"(新规则 {len(test_rules)})")

    print("\n--- 单符号查表 (新规则下必败) ---")
    train_eval(SingleSymModel(), Xtr, Ytr, Ctr, Xte, Yte, Cte, dev, "查表")
    print("\n--- LIF 序列 (通用逻辑: 自洽检测) ---")
    train_eval(LIFModel(), Xtr, Ytr, Ctr, Xte, Yte, Cte, dev, "LIF")


if __name__ == "__main__":
    main()
