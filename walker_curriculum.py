"""walker_curriculum.py — 发育课程: 先学会站立, 再学会移动 (自然涌现)。

无手工模块: 统一池策略 + 课程阶段
  阶段 1 (站立): 无驱动规则, 目标 = 不跌倒 (平衡)
    → 神经元自发分化: 姿态敏感群 (平衡专群涌现)
  阶段 2 (移动): 加驱动规则 + 前进, 在阶段1基础上继续
    → 神经元新增: 前进/步态敏感群

统一策略: 观测窗口 (历史 L) → 稀疏池 → 动作 (时序由窗口表达)
ES 优化 (无梯度, Loihi 2 友好)
验证: 阶段1 站立能力 → 阶段2 移动能力 + 神经元群分化
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from walker_energy_env import WalkerEnergyEnv


class CurriculumEnv(WalkerEnergyEnv):
    """课程环境: 阶段1 无驱动规则 (站立), 阶段2 有 (移动)。"""
    def __init__(self, move=False, **kw):
        super().__init__(**kw)
        self.move = move

    def step(self, act):
        # 能量 + 输出 (不变), 驱动规则按阶段
        omega = np.abs(self.obs[4:8] - self.prev_obs[4:8])
        self.E -= self.c_out * float(np.sum(np.abs(act) * omega)) + self.c_base
        self.obs, r_env, done_env, _, _ = self.env.step(act)
        self.prev_obs = self.obs.copy()
        self.t += 1
        vx = self.obs[2]
        self.dist += max(0.0, vx)
        if self.move:  # 阶段2: 驱动规则
            self.slow_steps = self.slow_steps + 1 if vx < self.v_min else 0
            dead_slow = self.slow_steps > self.slow_n
        else:         # 阶段1: 无规则 (只平衡)
            dead_slow = False
        gained = 0.0
        while self.dist >= self.D_goal:
            self.dist -= self.D_goal
            self.E = min(self.E0 * 1.5, self.E + self.E_goal)
            gained += self.E_goal
        dead = done_env or dead_slow or self.E <= 0
        r = (0.3 if self.move else 0.1) * vx + 0.1 * gained - 1.0 * dead
        return self._obs(), r, dead


class UnifiedAgent(nn.Module):
    """统一池策略: 观测窗口 → 稀疏池 → 动作 (无手工模块)。"""
    def __init__(self, obs=26, L=8, act=4, d=64, pool=256, top_k=32):
        super().__init__()
        self.L = L
        self.embed = nn.Linear(obs * L, d)
        self.act_mask = torch.zeros(pool, dtype=torch.bool)
        self.act_mask[:96] = True
        self.W = nn.Linear(d, pool)
        self.head = nn.Linear(pool, act)
        self.growth_log = []

    def forward(self, x):
        z = torch.tanh(self.embed(x))
        pre = self.W(z)
        m = self.act_mask.to(pre.device)
        pre = pre.masked_fill(~m[None], -1e9)
        vals, idx = pre.topk(32, dim=1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(1, idx, torch.tanh(vals))
        return torch.tanh(self.head(sparse)), idx

    def act(self, hist, noise=0.0):
        dev = next(self.parameters()).device
        x = torch.from_numpy(hist).float().to(dev).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.forward(x)
            if noise:
                a = a + noise * torch.randn_like(a)
            return np.clip(a[0].cpu().numpy(), -1, 1).astype(np.float32)

    def neurons(self, x):
        """池激活 (用于聚类验证)。"""
        with torch.no_grad():
            _, idx = self.forward(x)
        return idx[0].cpu().numpy()


def eval_ep(agent, env, max_steps=1600, noise=0.0, seed=0):
    hist = np.zeros((agent.L, 26), np.float32)
    s = env.reset()
    total_r, dist = 0.0, 0.0
    hist[1:] = hist[:-1]; hist[0] = s
    for _ in range(max_steps):
        a = agent.act(hist.flatten(), noise=noise)
        o2, r, d = env.step(a)
        total_r += r; dist = env.dist
        hist[1:] = hist[:-1]; hist[0] = o2
        s = o2
        if d:
            break
    return total_r, env.t, dist


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 发育课程: 先站立 (阶段1) → 后移动 (阶段2) ===")
    agent = UnifiedAgent().to(dev)
    lr, sigma = 0.3, 0.2

    # ===== 阶段 1: 站立 (无驱动规则) =====
    print("阶段 1: 学会站立 (平衡, 无移动规则)...")
    env1 = CurriculumEnv(move=False)
    for it in range(20):
        deltas, scores = [], []
        for i in range(10):
            delta = []
            for p in agent.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d); p.data.add_(d)
            sc, t, dist = eval_ep(agent, env1)
            for p, d in zip(agent.parameters(), delta): p.data.sub_(d)
            deltas.append(delta); scores.append(sc)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(agent.parameters(), delta):
                p.data.add_(lr * wi / (10 * sigma) * d)
        if it % 5 == 4:
            sc, t, dist = eval_ep(agent, env1)
            print(f"  阶段1 iter {it+1}: 站立存活 {t} 步")

    # ===== 阶段 2: 移动 (加驱动规则) =====
    print("阶段 2: 学会移动 (驱动规则 + 前进)...")
    env2 = CurriculumEnv(move=True)
    for it in range(25):
        deltas, scores, dists = [], [], []
        for i in range(10):
            delta = []
            for p in agent.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d); p.data.add_(d)
            sc, t, dist = eval_ep(agent, env2)
            for p, d in zip(agent.parameters(), delta): p.data.sub_(d)
            deltas.append(delta); scores.append(sc); dists.append(dist)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(agent.parameters(), delta):
                p.data.add_(lr * wi / (10 * sigma) * d)
        if it % 5 == 4:
            sc, t, dist = eval_ep(agent, env2)
            print(f"  阶段2 iter {it+1}: 存活 {t}, 前进 {dist:.0f}")

    # 最终: 站立能力 + 移动能力
    t1s, t2s, d2s = [], [], []
    for _ in range(6):
        e1 = CurriculumEnv(move=False)
        _, t1, _ = eval_ep(agent, e1); t1s.append(t1)
        e2 = CurriculumEnv(move=True)
        _, t2, d2 = eval_ep(agent, e2); t2s.append(t2); d2s.append(d2)
    print(f"\n站立能力 (阶段1环境): 存活 {np.mean(t1s):.0f} 步")
    print(f"移动能力 (阶段2环境): 存活 {np.mean(t2s):.0f}, 前进 {np.mean(d2s):.0f}")
    print(f"{'✅ 站立→移动课程生效' if np.mean(t1s) > 300 and np.mean(d2s) > 20 else '⚠️ 有限'}")


if __name__ == "__main__":
    main()
