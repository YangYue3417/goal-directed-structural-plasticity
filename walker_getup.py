"""walker_getup.py — 三段发育课程: 起立 → 站立 → 移动。

阶段 0 (起立): 从随机初始姿态 (跌倒/侧倾) 恢复直立
  判定: |hull_angle| < 0.3 (直立) + 触地 → 离地
  环境: reset 时 hull 随机倾斜 (模拟跌倒), 或中途绊倒
阶段 1 (站立): 直立 + 静止 + 省能 (已有判定)
阶段 2 (移动): 从真站立出发, 驱动规则 + 前进

统一池策略 (无手工模块) + ES 每阶段优化
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from walker_energy_env import WalkerEnergyEnv


class GetUpEnv(WalkerEnergyEnv):
    """起立环境: 随机初始姿态 (hull 倾斜) + 中途绊倒。"""
    def __init__(self, stage=0, **kw):
        self.stage = stage  # 0=起立, 1=站立, 2=移动
        super().__init__(**kw)

    def reset(self):
        self.obs, _ = self.env.reset(seed=int(self.rng.randint(10**6)))
        self.prev_obs = self.obs.copy()
        self.E = self.E0; self.slow_steps = 0
        self.dist = 0.0; self.t = 0
        if self.stage == 0:
            # 随机初始姿态: hull 倾斜 (跌倒模拟) — 需起立
            tilt = self.rng.uniform(-1.2, 1.2)  # ±70°
            self.env.unwrapped.hull.angle = tilt
            self.env.unwrapped.hull.angularVelocity = self.rng.uniform(-2, 2)
            # 一步刷新观测 (物理反映倾斜)
            self.obs, _, _, _, _ = self.env.step(np.zeros(4, np.float32))
            self.prev_obs = self.obs.copy()
        return self._obs()

    def step(self, act):
        omega = np.abs(self.obs[4:8] - self.prev_obs[4:8])
        self.E -= self.c_out * float(np.sum(np.abs(act) * omega)) + self.c_base
        self.obs, r_env, done_env, _, _ = self.env.step(act)
        self.prev_obs = self.obs.copy()
        self.t += 1
        vx = self.obs[2]
        self.dist += max(0.0, vx)
        if self.stage == 2:
            self.slow_steps = self.slow_steps + 1 if vx < self.v_min else 0
            dead_slow = self.slow_steps > self.slow_n
        else:
            dead_slow = False
        gained = 0.0
        while self.dist >= self.D_goal:
            self.dist -= self.D_goal
            self.E = min(self.E0 * 1.5, self.E + self.E_goal)
            gained += self.E_goal
        dead = done_env or dead_slow or self.E <= 0
        # 阶段奖励
        upright = abs(self.obs[0]) < 0.3
        if self.stage == 0:
            r = 2.0 * upright - 1.0 * dead   # 起立: 直立奖励
        elif self.stage == 1:
            r = 0.2 * upright - 0.1 * (self.dist > 5) - 1.0 * dead  # 站立: 直立+静止
        else:
            r = 0.3 * vx + 0.1 * gained - 1.0 * dead
        return self._obs(), r, dead


class Agent(nn.Module):
    """统一池策略 (窗口): 观测 → 动作。"""
    def __init__(self, obs=26, L=8, act=4, d=64, pool=256, top_k=32):
        super().__init__()
        self.L = L
        self.embed = nn.Linear(obs * L, d)
        self.act_mask = torch.zeros(pool, dtype=torch.bool)
        self.act_mask[:96] = True
        self.W = nn.Linear(d, pool)
        self.head = nn.Linear(pool, act)

    def forward(self, x):
        z = torch.tanh(self.embed(x))
        pre = self.W(z)
        m = self.act_mask.to(pre.device)
        pre = pre.masked_fill(~m[None], -1e9)
        vals, idx = pre.topk(32, dim=1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(1, idx, torch.tanh(vals))
        return torch.tanh(self.head(sparse))

    def act(self, hist, noise=0.0):
        dev = next(self.parameters()).device
        with torch.no_grad():
            a = self.forward(torch.from_numpy(hist).float().to(dev).unsqueeze(0))
            if noise:
                a = a + noise * torch.randn_like(a)
            return np.clip(a[0].cpu().numpy(), -1, 1).astype(np.float32)


def eval_ep(agent, env, max_steps=1600, noise=0.0):
    L = agent.L
    hist = np.zeros((L, 26), np.float32)
    s = env.reset(); hist[0] = s
    total_r, dist, upright_steps = 0.0, 0.0, 0
    for _ in range(max_steps):
        a = agent.act(hist.flatten(), noise=noise)
        o2, r, d = env.step(a)
        total_r += r; dist = env.dist
        upright_steps += abs(o2[0]) < 0.3
        hist[1:] = hist[:-1]; hist[0] = o2
        s = o2
        if d:
            break
    return total_r, env.t, dist, upright_steps


def train_stage(agent, env, iters, lr=0.3, sigma=0.2, label="", n_pop=10):
    for it in range(iters):
        deltas, scores = [], []
        for i in range(n_pop):
            delta = []
            for p in agent.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d); p.data.add_(d)
            sc, t, dist, up = eval_ep(agent, env)
            for p, d in zip(agent.parameters(), delta): p.data.sub_(d)
            deltas.append(delta); scores.append(sc)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(agent.parameters(), delta):
                p.data.add_(lr * wi / (n_pop * sigma) * d)
        if it % 4 == 3:
            sc, t, dist, up = eval_ep(agent, env)
            print(f"  {label} iter {it+1}: 存活 {t}, 直立 {up} 步, 前进 {dist:.0f}")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 三段发育: 起立 → 站立 → 移动 ===")
    agent = Agent().to(dev)

    print("阶段 0: 起立 (随机初始姿态恢复直立)...")
    env0 = GetUpEnv(stage=0)
    train_stage(agent, env0, 15, label="起立")

    print("阶段 1: 站立 (直立+静止+省能)...")
    env1 = GetUpEnv(stage=1)
    train_stage(agent, env1, 12, label="站立")

    print("阶段 2: 移动 (驱动规则+前进)...")
    env2 = GetUpEnv(stage=2)
    train_stage(agent, env2, 20, label="移动")

    # 最终: 三段能力
    r0, t0, d0, up0 = eval_ep(agent, GetUpEnv(stage=0))
    r1, t1, d1, up1 = eval_ep(agent, GetUpEnv(stage=1))
    r2, t2, d2, up2 = eval_ep(agent, GetUpEnv(stage=2))
    print(f"\n起立: 存活 {t0}, 直立 {up0}/t0 | 站立: 存活 {t1}, 直立 {up1}, 前进 {d1:.0f} "
          f"| 移动: 存活 {t2}, 前进 {d2:.0f}")
    print(f"{'✅ 起立→站立→移动' if up0 > 30 and t1 > 100 and d2 > 15 else '⚠️ 有限'}")


if __name__ == "__main__":
    main()
