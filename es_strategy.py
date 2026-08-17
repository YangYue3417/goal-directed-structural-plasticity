"""es_strategy.py — EGGROLL 风格 ES 策略 (低秩进化, 无梯度) + 生长冲突探索。

EGGROLL 核心: 低秩扰动 + 进化选择, backprop-free。
应用到我们的框架:
  策略网络: 抽象观测 z → 神经元池 → 动作 logits (直接学动作分布)
  ES 优化: 低秩扰动参数 → 环境得分 → 加权更新
  生长: 目标驱动生长 (与 ES 正交) — 探索冲突

验证:
  ① ES 策略得分 vs 价值+MPC (2.8) vs 随机 (73.5) — 食物迷宫
  ② ES + 生长: 生长神经元功能性 (删了 → 得分崩)? 冲突?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from abstract_layer import FoodGame


class Policy(nn.Module):
    """策略网络: 抽象观测 → 池 → 动作 logits (神经元层直接输出动作)。"""
    def __init__(self, z_dim=3, n_act=5, pool=256, top_k=32, d=32):
        super().__init__()
        self.embed = nn.Linear(z_dim, d)
        self.act_mask = torch.zeros(pool, dtype=torch.bool)
        self.act_mask[:96] = True
        self.W = nn.Linear(d, pool)
        self.head = nn.Linear(pool, n_act)
        self.growth_log = []

    def forward(self, z):
        z_ = torch.tanh(self.embed(z))
        pre = self.W(z_)
        m = self.act_mask.to(pre.device)
        pre = pre.masked_fill(~m[None], -1e9)
        vals, idx = pre.topk(32, dim=1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(1, idx, F.gelu(vals))
        return self.head(sparse), idx

    def act(self, z, greedy=False):
        with torch.no_grad():
            logits, _ = self.forward(torch.from_numpy(z).float().unsqueeze(0))
            if greedy:
                return int(logits.argmax(-1).item())
            return int(torch.distributions.Categorical(logits=logits).sample().item())


def low_rank_noise(policy, rank=4, sigma=0.15, seed=None):
    """EGGROLL 低秩扰动: 展平参数 + 低秩随机投影 (U·V)。"""
    rng = np.random.RandomState(seed)
    params = [p.detach().flatten() for p in policy.parameters()]
    total = sum(len(p) for p in params)
    # 低秩: 扰动 = U·v (rank 维) — 远小于全参数
    U = rng.randn(total, rank).astype(np.float32) * sigma / rank ** 0.5
    v = rng.randn(rank).astype(np.float32)
    delta = U @ v  # (total,) 低秩扰动
    return delta


def apply_delta(policy, delta, sign=1.0):
    offset = 0
    with torch.no_grad():
        for p in policy.parameters():
            n = p.numel()
            p.add_(sign * torch.from_numpy(
                delta[offset:offset + n]).float().reshape(p.shape))
            offset += n


def eval_score(policy, env, n_eps=8, max_steps=200):
    scores = []
    for _ in range(n_eps):
        s = env.reset(); done = False; sc = 0.0
        steps = 0
        while not done and steps < max_steps:
            a = policy.act(s)
            o2, r, d = env.step(a)
            sc += r; s = o2; done = d; steps += 1
        scores.append(sc)
    return float(np.mean(scores))


def eggroll(policy, env_fn, n_pop=24, n_iter=40, lr=0.12, sigma=0.15,
            rank=4, greedy_eval=True, seed=0):
    """EGGROLL 风格 ES: 低秩扰动 → 得分 → 加权更新。"""
    rng = np.random.RandomState(seed)
    history = []
    for it in range(n_iter):
        deltas, scores = [], []
        for i in range(n_pop):
            d = low_rank_noise(policy, rank, sigma, seed=rng.randint(10**6))
            apply_delta(policy, d, +1.0)
            sc = eval_score(policy, env_fn())
            apply_delta(policy, d, -1.0)  # 还原
            deltas.append(d); scores.append(sc)
        scores = np.array(scores)
        # 加权更新 (ES): 归一化得分作为权重
        w = (scores - scores.mean()) / (scores.std() + 1e-8)
        w = np.clip(w, -2, 2)
        for d, wi in zip(deltas, w):
            apply_delta(policy, d, lr * wi / (n_pop * sigma))
        history.append(scores.mean())
        if it % 10 == 9:
            g = eval_score(policy, env_fn(), n_eps=8)
            print(f"  iter {it+1}: 扰动均分 {scores.mean():.1f} | 当前策略 {g:.1f}")
    return history


def goal_grow(policy, env, n_eps=200, seed=0):
    """目标驱动生长: 吃食物状态激活的神经元 → 克隆 (baby 弱初始化)。"""
    rng = np.random.RandomState(seed)
    sel_goal = []
    for _ in range(n_eps):
        s = env.reset(); done = False
        while not done:
            a = int(rng.randint(5))
            o2, r, d = env.step(a)
            if r > 0:  # 目标达成时刻
                with torch.no_grad():
                    _, idx = policy.forward(torch.from_numpy(
                        s).float().unsqueeze(0))
                    sel_goal.append(idx[0])
            s = o2; done = d
    if not sel_goal:
        return 0
    # 克隆: 目标状态激活神经元 → 未激活池 (弱继承)
    inactive = (~policy.act_mask).nonzero().flatten()
    if len(inactive) == 0:
        return 0
    cnt = torch.zeros(policy.W.out_features)
    for row in sel_goal:
        cnt[row] += 1
    cand = torch.argsort(cnt * policy.act_mask.float(), descending=True)[:4]
    cand = cand[policy.act_mask[cand]]
    n_grow = 0
    with torch.no_grad():
        for src, tgt in zip(cand, inactive[:len(cand)]):
            src, tgt = int(src), int(tgt)
            policy.W.weight.data[tgt] = 0.5 * policy.W.weight.data[src] + 0.3 * torch.randn_like(policy.W.weight.data[src])
            policy.head.weight.data[:, tgt] = 0.5 * policy.head.weight.data[:, src]
            policy.act_mask[tgt] = True
            policy.growth_log.append(tgt)
            n_grow += 1
    return n_grow


if __name__ == "__main__":
    print("=== EGGROLL 风格 ES 策略 (食物迷宫) + 生长冲突 ===")
    # 随机基线
    print(f"随机基线: ~73.5/ep | 价值+MPC 对照: 2.8/ep (之前)")
    # 模式 A: 纯 ES
    torch.manual_seed(42)
    pA = Policy()
    eggroll(pA, FoodGame, n_pop=16, n_iter=30, greedy_eval=True)
    sA = eval_score(pA, FoodGame(), n_eps=30)
    print(f"[纯 ES] 得分 {sA:.1f}/ep")

    # 模式 B: ES + 目标驱动生长 (探索冲突)
    torch.manual_seed(42)
    pB = Policy()
    for it in range(30):
        if it % 5 == 0 and len(pB.growth_log) < 20:
            n = goal_grow(pB, FoodGame())
        eggroll(pB, FoodGame, n_pop=16, n_iter=1, greedy_eval=True)
    sB = eval_score(pB, FoodGame(), n_eps=30)
    print(f"[ES+生长] 得分 {sB:.1f}/ep (生长 {len(pB.growth_log)})")

    # 生长冲突: 删生长神经元 → 得分变化
    if pB.growth_log:
        backup = pB.act_mask.clone()
        with torch.no_grad():
            for nid in pB.growth_log:
                pB.act_mask[nid] = False
        sB_del = eval_score(pB, FoodGame(), n_eps=30)
        pB.act_mask = backup
        rel = (sB - sB_del) / max(abs(sB), 1e-6) * 100
        print(f"删生长神经元: {sB:.1f} → {sB_del:.1f} ({rel:+.0f}%) "
              f"{'= 生长与ES无冲突 (功能性)' if rel > 0 else '= 冲突 (生长有害)'}")
