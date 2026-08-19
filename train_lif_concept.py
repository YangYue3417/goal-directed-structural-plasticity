"""train_lif_concept.py — LIF 符号原理实现: 概念 = 阈值发放神经元。

数学原理 (推导):
  ① 符号 = 发放事件: 概念神经元 j 发放 ⟺ 输入 ∈ 概念 j
  ② 逻辑 = 时间积分: 序列在 V_m 中累积 (非单步)
  ③ 规则 = 阈值条件: V > θ → 发放 (if-then)

实现 (非 softmax!):
  输出层 = 概念神经元 (每个数字/符号一个)
  训练: margin 损失 (正确概念发放 V≥θ, 错误不发放 V<θ)
  = 直接训练"发放边界" (符号化), 不是概率!

任务: 一位加法 (a+b → 和 0-18, 概念神经元发放)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lif_pool import LIFPool


class LIFConcept(nn.Module):
    """LIF 池 (时间积分) + 概念神经元层 (阈值发放 = 符号)。"""
    def __init__(self, n_digit=10, n_concept=19, d=32, pool=256,
                 theta=0.3, out_theta=0.5):
        super().__init__()
        self.embed = nn.Embedding(n_digit, d)
        self.pool = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=12)
        self.concept = nn.Linear(d, n_concept)  # 概念打分 (无 softmax!)
        self.out_theta = out_theta              # 概念阈值 (发放边界)
        self.n_concept = n_concept

    def forward(self, a, b):
        """输入 (a,b) 序列 → 概念打分。"""
        za = torch.tanh(self.embed(a))
        zb = torch.tanh(self.embed(b))
        seq = torch.stack([za, zb], 1)
        out, spikes = self.pool(seq)           # 时间积分
        h = out[:, -1]
        scores = self.concept(h)                # 概念打分 (连续)
        # 符号化: 发放 = score > θ (阈值)
        return scores, spikes

    def predict(self, a, b):
        """预测: 概念神经元发放的 (score > θ) 中最高分? 或发放的。"""
        scores, _ = self.forward(a, b)
        # 符号 = 发放事件: 发放的概念 (score > θ); 无发放 → 最高分
        fired = scores > self.out_theta
        if fired.any():
            return scores.argmax(-1)  # 发放中最高 (最活跃概念)
        return scores.argmax(-1)


def gen_pairs(n=30000, max_sum=18, seed=42):
    """只生成和 ≤ max_sum 的样本 (max_sum 过滤!)."""
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
    print("=== LIF 符号原理: 概念 = 阈值发放神经元 (margin 训练) ===")
    model = LIFConcept().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    A, B, S = gen_pairs()
    n = int(0.9 * len(A))
    Ad, Bd, Sd = A.to(dev), B.to(dev), S.to(dev)

    for ep in range(60):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i+512]
            scores, _ = model(Ad[idx], Bd[idx])
            # margin 损失: 正确概念 V ≥ θ, 错误概念 V < θ
            target = Sd[idx]
            margin = torch.zeros_like(scores)
            # 正确概念要超过阈值
            correct = scores[torch.arange(len(idx)), target]
            loss_fire = F.relu(model.out_theta - correct).mean()
            # 错误概念要低于阈值
            wrong = scores.clone()
            wrong[torch.arange(len(idx)), target] = -1e9
            wrong_max = wrong.max(-1).values
            loss_silent = F.relu(wrong_max - model.out_theta).mean()
            loss = loss_fire + loss_silent
            opt.zero_grad(); loss.backward(); opt.step()

    # 验证: 概念发放正确率
    model.eval()
    with torch.no_grad():
        scores, _ = model(Ad[n:], Bd[n:])
        pred = scores.argmax(-1)
        acc = (pred == Sd[n:]).float().mean()
        # 发放率 (符号化质量)
        fired_rate = (scores > model.out_theta).float().mean()
    print(f"加法概念 acc: {acc.item():.3f} | 概念发放率: {fired_rate.item():.3f}")

    # 外推: 训练 ≤9 (小和) → 测试大和 (10-18)?
    print("\n=== 外推测试 (训练和≤9, 测大和) ===")
    model2 = LIFConcept().to(dev)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=2e-3)
    A1, B1, S1 = gen_pairs(30000, max_sum=9)
    A1d, B1d, S1d = A1.to(dev), B1.to(dev), S1.to(dev)
    for ep in range(60):
        idx = torch.randperm(len(A1))[:512]
        scores, _ = model2(A1d[idx], B1d[idx])
        target = S1d[idx]
        correct = scores[torch.arange(len(idx)), target]
        lf = F.relu(model2.out_theta - correct).mean()
        wrong = scores.clone()
        wrong[torch.arange(len(idx)), target] = -1e9
        lw = F.relu(wrong.max(-1).values - model2.out_theta).mean()
        opt2.zero_grad(); (lf + lw).backward(); opt2.step()
    # 测大和 (10-18, 未见)
    A2, B2, S2 = gen_pairs(5000, max_sum=18, seed=7)
    big = (A2 + B2 > 9)
    A2d, B2d, S2d = A2.to(dev), B2.to(dev), S2.to(dev)
    with torch.no_grad():
        s_big = model2(A2d[big], B2d[big])[0].argmax(-1)
        acc_big = (s_big == S2d[big]).float().mean()
    print(f"训练和≤9 → 测大和 (10-18): {acc_big.item():.3f}")


if __name__ == "__main__":
    main()
