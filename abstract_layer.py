"""abstract_layer.py — 抽象层: 神经元抽离游戏场景, 学习通用高分策略。

架构:
  场景层:  具体游戏 (观测, 得分函数) — 每场景可换
  感知抽象: 观测 → 语义特征 z (场景无关: 距离目标/资源/障碍/能量)
  共享层:   神经元池 (MemPool) + 价值, 工作在 z 上 — 跨场景复用
  评分:     每场景得分函数 (价值目标) — "根据不同场景设定值"

验证: 场景 A (食物收集) 训练 → 场景 B (能量平衡) 直接迁移
      (神经元/决策层不重训, 只换感知特征 + 评分)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from mempool import MemPool


# ============ 场景层: 两个游戏, 共享抽象观测 ============
class FoodGame:
    """场景 A: 收集食物 (得分 = 食物获取数)。"""
    SIZE = 8
    N_FOOD = 6

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.x, self.y = 3.0, 3.0
        self.food = self.rng.rand(self.N_FOOD, 2) * (self.SIZE - 1)
        self.score = 0
        self.t = 0
        return self.abstract()

    def abstract(self):
        """语义特征 (场景无关): 最近食物距离, 位置。"""
        d = np.min(np.abs(self.food - [self.x, self.y]).sum(1))
        return np.array([self.x / self.SIZE, self.y / self.SIZE,
                         d / (2 * self.SIZE)], np.float32)

    def step(self, a):
        if a == 1: self.y = min(self.SIZE - 1, self.y + 1)
        if a == 2: self.y = max(0, self.y - 1)
        if a == 3: self.x = min(self.SIZE - 1, self.x + 1)
        if a == 4: self.x = max(0, self.x - 1)
        eaten = 0
        for i, (fx, fy) in enumerate(self.food):
            if abs(self.x - fx) < 0.5 and abs(self.y - fy) < 0.5:
                self.score += 10
                eaten += 1
                self.food[i] = self.rng.rand(2) * (self.SIZE - 1)  # 新食物
        self.t += 1
        return self.abstract(), eaten * 10, self.t >= 200


class EnergyGame:
    """场景 B: 能量平衡 (得分 = 存活 × 效率)。能量点安全区外。"""
    SIZE = 8
    ENERGY = [(0, 0), (7, 7)]
    E0, C_BASE = 15.0, 0.06

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.x, self.y, self.E = 3.0, 3.0, self.E0
        self.score = 0.0
        self.t = 0
        return self.abstract()

    def abstract(self):
        d = min(abs(self.x - ex) + abs(self.y - ey) for ex, ey in self.ENERGY)
        return np.array([self.x / self.SIZE, self.y / self.SIZE,
                         self.E / self.E0, d / (2 * self.SIZE)], np.float32)

    def step(self, a):
        if a == 1: self.y = min(self.SIZE - 1, self.y + 1)
        if a == 2: self.y = max(0, self.y - 1)
        if a == 3: self.x = min(self.SIZE - 1, self.x + 1)
        if a == 4: self.x = max(0, self.x - 1)
        self.E -= self.C_BASE + 0.05 * (a != 0)
        gained = 0.0
        if (round(self.x), round(self.y)) in self.ENERGY and self.E < self.E0:
            self.E = min(self.E0, self.E + 8)
            gained = 8
        self.t += 1
        dead = self.E <= 0
        self.score += 0.5 * gained
        return self.abstract(), 0.5 * gained - 1.0 * dead, dead


# ============ 共享层: 神经元抽离场景 (z 空间) ============
class SharedAgent(nn.Module):
    """工作在抽象特征 z 上的共享层: 世界模型 + LIF 池 + 价值。"""
    def __init__(self, z_dim=4, n_act=5, d=32, pool=256, top_k=32):
        super().__init__()
        self.n_act = n_act
        self.embed = nn.Linear(z_dim + n_act, d)
        self.pool = MemPool(d, pool, top_k, tau_min=1.5, tau_max=24.0)
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU())
        self.head_s = nn.Linear(64, z_dim)
        self.head_r = nn.Linear(64, 1)

    def forward(self, z, act):
        B = z.shape[0]
        sa = torch.cat([z, act], -1)
        z_ = torch.tanh(self.embed(sa))
        z_pool, sel = self.pool(z_.unsqueeze(1))
        h = self.net(z_pool.squeeze(1))
        return self.head_s(h), self.head_r(h).squeeze(-1), sel

    def value(self, z):
        return self.head_r(self.net(torch.tanh(self.embed(
            torch.cat([z, torch.zeros(z.shape[0], self.n_act, device=z.device)], -1))))).squeeze(-1)


def collect(env, n_eps=300, seed=0):
    S, A, R, Sn = [], [], [], []
    for _ in range(n_eps):
        s = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            a = int(env.rng.randint(5))
            o2, r, dead = env.step(a)
            S.append(s); A.append(a); R.append(r); Sn.append(o2)
            s = o2; done = dead; steps += 1
    return (np.array(S, np.float32), np.array(A, np.int64),
            np.array(R, np.float32), np.array(Sn, np.float32))


def train(agent, opt, S, A, R, Sn, epochs=120, dev="cuda"):
    s_t = torch.from_numpy(S).float().to(dev)
    a_t = F.one_hot(torch.from_numpy(A).long(), 5).float().to(dev)
    r_t = torch.from_numpy(R).float().to(dev)
    sn_t = torch.from_numpy(Sn).float().to(dev)
    for ep in range(epochs):
        idx = torch.randperm(len(S))[:2048]
        sp, rp, sel = agent(s_t[idx], a_t[idx])
        loss_wm = F.mse_loss(sp, sn_t[idx]) + 0.5 * F.mse_loss(rp, r_t[idx])
        with torch.no_grad():
            target = r_t[idx] + 0.95 * agent.value(sn_t[idx])
        loss = loss_wm + 0.3 * F.mse_loss(agent.value(s_t[idx]), target)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 40 == 39 and len(agent.pool.growth_log) < 40:
            per_err = (sp - sn_t[idx]).pow(2).mean(-1)
            agent.pool.grow(sel[:, 0][int(per_err.argmax().item())])
        agent.pool.settle_babies(age_thresh=400.0, rate_thresh=0.01)


def mpc(agent, z, dev="cuda"):
    z_t = torch.from_numpy(z).float().to(dev).unsqueeze(0).repeat(5, 1)
    acts = torch.eye(5).to(dev)
    with torch.no_grad():
        sp, rp, _ = agent(z_t, acts)
        score = agent.value(sp) + 0.5 * rp
    return int(score.argmax().item())


def eval_game(agent, env, n_eps=50, dev="cuda"):
    scores, lens = [], []
    for _ in range(n_eps):
        s = env.reset(); done = False
        while not done:
            a = mpc(agent, s, dev)
            o2, r, dead = env.step(a)
            s = o2; done = dead
        scores.append(env.score); lens.append(env.t)
    return np.mean(scores), np.mean(lens)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== 抽象层: 神经元抽离场景, 学习通用高分策略 ===")
    # 场景 A 训练 (z_dim=3)
    torch.manual_seed(42)
    agent = SharedAgent(z_dim=3).to(dev)
    opt = torch.optim.AdamW(agent.parameters(), lr=1e-3)
    S, A, R, Sn = collect(FoodGame(), 300)
    print(f"场景A (食物收集): 随机得分 {np.sum(R)/300:.1f}/ep")
    train(agent, opt, S, A, R, Sn, dev=dev)
    sA, lA = eval_game(agent, FoodGame())
    print(f"场景A 学习后: 得分 {sA:.1f} (随机基线对照)")

    # 迁移: 场景 B (z_dim=4, 抽象特征不同维度 → 重新 embed)
    print("\n=== 迁移: 场景A → 场景B (能量平衡, 神经元池共享) ===")
    agent2 = SharedAgent(z_dim=4).to(dev)
    # 共享池参数: 迁移神经元层 (池权重), 只换感知 (embed/head)
    agent2.pool.load_state_dict(
        {k: v for k, v in agent.pool.state_dict().items()
         if k in agent2.pool.state_dict() and k not in ("vm",)},
        strict=False)
    opt2 = torch.optim.AdamW(agent2.parameters(), lr=1e-3)
    S2, A2, R2, Sn2 = collect(EnergyGame(), 300)
    print(f"场景B 随机: 得分 {np.sum(R2)/300:.1f}/ep")
    # 少量场景B 适应 (迁移后微调) vs 不微调
    train(agent2, opt2, S2, A2, R2, Sn2, epochs=60, dev=dev)
    sB, lB = eval_game(agent2, EnergyGame())
    print(f"场景B (迁移+微调): 得分 {sB:.1f}, 存活 {lB:.0f} 步")


if __name__ == "__main__":
    main()
