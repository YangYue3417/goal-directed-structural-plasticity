"""train_set_cardinality.py — 集合基数: 数字概念 (数量) + 并集加法。

数字概念输入 (NALU 理念): 集合 = 数量直接可见 (非符号!)
  ① 集合 → 基数: one-hot 集合 → 数量 (量化映射)
  ② 并集 → 加法: setA ∪ setB → |A∪B| (本质操作)

验证: 外推! (训练小集合 ≤5 元素 → 测大集合 6-20)

增加训练 (回答"训练太少"):
  ① 无限生成: 集合组合 2^N — 每次随机采样, 基数可计算!
  ② 自我训练: 生成 → 可计算验证 → 改进 (无限数据循环!)
  ③ 课程: 小 → 大 (渐进)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lif_pool import LIFPool


class SetCardinality(nn.Module):
    """集合 (one-hot) → LIF → 基数/加法。"""
    def __init__(self, n_elem=20, d=32, pool=256, theta=0.3, n_out=21):
        super().__init__()
        self.inp = nn.Linear(n_elem, d)
        self.pool = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=12)
        self.head_card = nn.Linear(d, n_out)      # 集合 → 基数 (0-20)
        self.head_add = nn.Linear(d * 2, n_out)   # 并集 → 基数 (加法)

    def card(self, s):
        """集合 → 基数。s: (B, n_elem) one-hot。"""
        z = torch.tanh(self.inp(s))
        out, _ = self.pool(z.unsqueeze(1))
        return self.head_card(out[:, -1])

    def add(self, s1, s2):
        """并集 → 基数 (加法)。"""
        z1 = torch.tanh(self.inp(s1)); z2 = torch.tanh(self.inp(s2))
        o1, _ = self.pool(z1.unsqueeze(1))
        o2, _ = self.pool(z2.unsqueeze(1))
        return self.head_add(torch.cat([o1[:, -1], o2[:, -1]], -1))


def gen_set(n_elem, max_size, n=50000, seed=42, overlap=0.0):
    """生成集合样本: 基数 (量化) + 并集加法。"""
    rng = np.random.RandomState(seed)
    S1, C1 = [], []
    S2a, S2b, C2 = [], [], []
    for _ in range(n):
        k = rng.randint(0, max_size + 1)
        s1 = np.zeros(n_elem, np.float32)
        s1[rng.choice(n_elem, k, replace=False)] = 1
        S1.append(s1); C1.append(k)
        # 并集: 两集合 (控制重叠 → 可算 |A∪B|)
        k2a = rng.randint(0, max_size + 1); k2b = rng.randint(0, max_size + 1)
        s2a = np.zeros(n_elem, np.float32)
        s2a[rng.choice(n_elem, k2a, replace=False)] = 1
        s2b = np.zeros(n_elem, np.float32)
        # 与 A 部分重叠 (overlap 概率)
        s2b[rng.choice(n_elem, k2b, replace=False)] = 1
        union = int((s2a + s2b > 0).sum())
        S2a.append(s2a); S2b.append(s2b); C2.append(union)
    return (torch.from_numpy(np.array(S1)), torch.tensor(C1),
            torch.from_numpy(np.array(S2a)), torch.from_numpy(np.array(S2b)),
            torch.tensor(C2))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    N_ELEM = 20
    print("=== 集合基数: 数字概念 (数量) + 并集加法 + 外推 ===")
    # 训练: 小集合 (≤5) — 外推测试: 大集合 (6-20)
    S1, C1, S2a, S2b, C2 = gen_set(N_ELEM, 5, 30000, seed=42)
    model = SetCardinality(N_ELEM).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    S1d, C1d = S1.to(dev), C1.to(dev)
    S2ad, S2bd, C2d = S2a.to(dev), S2b.to(dev), C2.to(dev)
    n = len(S1)

    for ep in range(60):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i+512]
            l1 = F.cross_entropy(model.card(S1d[idx]), C1d[idx])
            l2 = F.cross_entropy(model.add(S2ad[idx], S2bd[idx]), C2d[idx])
            loss = l1 + l2
            opt.zero_grad(); loss.backward(); opt.step()

    # 外推测试: 大集合 (6-20 元素, 训练未见!)
    print("\n=== 外推验证 (训练 ≤5, 测试大集合) ===")
    for test_max in [8, 12, 20]:
        S1t, C1t, S2at, S2bt, C2t = gen_set(N_ELEM, test_max, 5000, seed=7)
        model.eval()
        with torch.no_grad():
            a1 = (model.card(S1t.to(dev)).argmax(-1) == C1t.to(dev)).float().mean()
            a2 = (model.add(S2at.to(dev), S2bt.to(dev)).argmax(-1) == C2t.to(dev)).float().mean()
        print(f"  测试大小 ≤{test_max}: 集合→基数 {a1.item():.3f} | 并集→加法 {a2.item():.3f}")

    print("\n=== 无限数据自我训练 (可计算验证!) ===")
    # 自我训练: 每轮生成新数据 (无限), 验证基数 (可算), 训练
    for rnd in range(5):
        S1n, C1n, S2an, S2bn, C2n = gen_set(N_ELEM, 12, 20000, seed=rnd)
        model.train()
        for i in range(0, len(S1n), 512):
            idx = torch.randperm(len(S1n))[:512]
            l1 = F.cross_entropy(model.card(S1n[idx].to(dev)), C1n[idx].to(dev))
            l2 = F.cross_entropy(model.add(S2an[idx].to(dev), S2bn[idx].to(dev)), C2n[idx].to(dev))
            opt.zero_grad(); (l1 + l2).backward(); opt.step()
    S1t, C1t, S2at, S2bt, C2t = gen_set(N_ELEM, 20, 5000, seed=9)
    with torch.no_grad():
        a1 = (model.card(S1t.to(dev)).argmax(-1) == C1t.to(dev)).float().mean()
        a2 = (model.add(S2at.to(dev), S2bt.to(dev)).argmax(-1) == C2t.to(dev)).float().mean()
    print(f"自我训练后 (测试 ≤20): 基数 {a1.item():.3f} | 加法 {a2.item():.3f}")


if __name__ == "__main__":
    main()
