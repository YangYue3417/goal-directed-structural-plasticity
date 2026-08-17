"""energy_balance.py — 能量平衡生存 (用户 4 条约束)。

约束:
  ① 能量效率最大化 (每单位能量获取最多生存)
  ② 运动消耗能量; 能量补充点在安全区外 (必须离开安全区)
  ③ 无目标 (无任务终点), 但有惩罚 (能量耗尽死亡)
  ④ Loihi 2 推理风格: LIF 记忆池 + 稀疏 top-k + 在线学习

环境: 迷宫, 中心安全区 (低消耗), 四角能量点 (安全区外, 补充能量)
每步: E -= c_base; 移动额外 E -= c_move; 到能量点 E += ΔE
惩罚: E ≤ 0 死亡 (负回报 -1); 无正任务目标
最优: 周期外出觅食 → 回安全区, 能量效率最大化
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


class EnergyMaze:
    """能量平衡迷宫 (物理功能量模型)。

    能量 = 机械功: dE = m·a·ds = m·|a|·|v|·dt (功率: 力×速度)
    质量 m 使加速度大/速度快的运动消耗更多。
    状态: (x, y, vx, vy, E); 动作 → 加速度 (4 向 ±a0 或 0)。
    """
    SIZE = 8
    SAFE = ((2, 2), (5, 5))          # 中心安全区
    ENERGY_POINTS = [(0, 0), (7, 0), (0, 7), (7, 7)]   # 四角 (安全区外)
    E0 = 20.0
    MASS = 1.0                        # 质量
    A0 = 0.8                          # 动作加速度 (4 向)
    C_BASE = 0.06                     # 基础代谢/步 (无觅食 ~330 步死)
    DELTA_E = 8.0                     # 能量点补充 (往返净收益小)
    V_MAX = 2.0

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.x, self.y, self.vx, self.vy = 3.0, 3.0, 0.0, 0.0
        self.E = self.E0
        self.t = 0
        return self.obs()

    def _dist_energy(self):
        return min(abs(self.x - ex) + abs(self.y - ey) for ex, ey in self.ENERGY_POINTS)

    def in_safe(self, x, y):
        (x0, y0), (x1, y1) = self.SAFE
        return x0 <= x <= x1 and y0 <= y <= y1

    def obs(self):
        d = min(abs(self.x - ex) + abs(self.y - ey) for ex, ey in self.ENERGY_POINTS)
        return np.array([self.x / self.SIZE, self.y / self.SIZE,
                         self.vx / self.V_MAX, self.vy / self.V_MAX,
                         self.E / self.E0, d / (2 * self.SIZE),
                         float(self.in_safe(self.x, self.y))], np.float32)

    def step(self, a):
        """a: 0=无加速度(滑行), 1-4=±y,±x 加速度。能量 = m·a·v·dt。"""
        d0 = self._dist_energy()
        ax = ay = 0.0
        if a == 1: ay = self.A0
        if a == 2: ay = -self.A0
        if a == 3: ax = self.A0
        if a == 4: ax = -self.A0
        # 动力学: v += a·dt; x += v·dt (惯性保持)
        self.vx = np.clip(self.vx + ax, -self.V_MAX, self.V_MAX)
        self.vy = np.clip(self.vy + ay, -self.V_MAX, self.V_MAX)
        self.x = np.clip(self.x + self.vx, 0, self.SIZE - 1)
        self.y = np.clip(self.y + self.vy, 0, self.SIZE - 1)
        # 能量 = 机械功: 功率 m·|a|·|v| + 基础代谢
        a_mag = (ax ** 2 + ay ** 2) ** 0.5
        v_mag = (self.vx ** 2 + self.vy ** 2) ** 0.5
        self.E -= self.MASS * a_mag * v_mag + self.C_BASE
        gained = 0.0
        if (round(self.x), round(self.y)) in self.ENERGY_POINTS and self.E < self.E0:
            self.E = min(self.E0, self.E + self.DELTA_E)
            gained = self.DELTA_E
        self.t += 1
        dead = self.E <= 0
        # 奖励: 能量补充 (约束②) + 距离势塑形 (引导觅食) + 死亡惩罚 (约束③)
        r = 0.5 * gained + 0.08 * (d0 - self._dist_energy()) - 1.0 * dead
        return self.obs(), r, dead, gained


class Agent(nn.Module):
    """Loihi 2 风格: LIF 记忆池 + 稀疏 top-k + 世界模型 + 价值。"""
    def __init__(self, obs_dim=7, n_act=5, d=32, pool=256, top_k=32):
        super().__init__()
        self.n_act = n_act
        self.embed = nn.Linear(obs_dim + n_act, d)
        self.pool = MemPool(d, pool, top_k, tau_min=1.5, tau_max=24.0)
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU())
        self.head_s = nn.Linear(64, obs_dim)
        self.head_r = nn.Linear(64, 1)

    def forward(self, obs, act):
        B = obs.shape[0]
        sa = torch.cat([obs, act], -1)
        z = torch.tanh(self.embed(sa))
        z_pool, sel = self.pool(z.unsqueeze(1))
        h = self.net(z_pool.squeeze(1))
        return self.head_s(h), self.head_r(h).squeeze(-1), sel


class EnergyValue(nn.Module):
    """能量生存价值: V(s) = 预期还能活多久 (TD, 惩罚驱动)。"""
    def __init__(self, obs_dim=7, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, s):
        return self.net(s).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n_eps", type=int, default=400)
    p.add_argument("--grow_every", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)

    print("=== 能量平衡生存 (约束: 能量效率/安全区外觅食/无目标有惩罚) ===")
    # 收集: 随机策略
    env = EnergyMaze()
    S, A, R, Sn, G = [], [], [], [], []
    for _ in range(args.n_eps):
        s = env.reset()
        done = False
        while not done:
            a = int(env.rng.randint(5))
            o2, r, dead, gained = env.step(a)
            S.append(s); A.append(a); R.append(r); Sn.append(o2)
            G.append(gained)
            s = o2
            done = dead
    S = np.array(S, np.float32); A = np.array(A, np.int64)
    R = np.array(R, np.float32); Sn = np.array(Sn, np.float32)
    print(f"随机基线: 平均存活 {len(S)/args.n_eps:.0f} 步, "
          f"能量获取 {sum(G)/args.n_eps:.1f}/ep")

    # 训练: 世界模型 + 能量价值 (TD, 无奖励目标, 惩罚驱动)
    torch.manual_seed(42)
    agent = Agent().to(dev)
    V = EnergyValue().to(dev)
    opt = torch.optim.AdamW(list(agent.parameters()) + list(V.parameters()), lr=1e-3)
    s_t = torch.from_numpy(S).float().to(dev)
    a_t = F.one_hot(torch.from_numpy(A).long(), 5).float().to(dev)
    r_t = torch.from_numpy(R).float().to(dev)
    sn_t = torch.from_numpy(Sn).float().to(dev)

    for ep in range(args.epochs):
        idx = torch.randperm(len(S))[:2048]
        sp, rp, sel = agent(s_t[idx], a_t[idx])
        # 世界模型 (预测转移 + 惩罚)
        loss_wm = F.mse_loss(sp, sn_t[idx]) + 0.5 * F.mse_loss(rp, r_t[idx])
        # 能量价值: TD (V(s) ≈ r + γV(s'))
        with torch.no_grad():
            target = r_t[idx] + 0.95 * V(sn_t[idx])
        loss_v = F.mse_loss(V(s_t[idx]), target)
        loss = loss_wm + 0.3 * loss_v
        opt.zero_grad(); loss.backward(); opt.step()
        # 生长 (难样本定向, Loihi 风格稀疏池)
        if ep % args.grow_every == args.grow_every - 1 and len(agent.pool.growth_log) < 40:
            per_err = (sp - sn_t[idx]).pow(2).mean(-1)
            agent.pool.grow(sel[:, 0][int(per_err.argmax().item())])
        agent.pool.settle_babies(age_thresh=400.0, rate_thresh=0.01)

    # 评估: 能量效率策略 (MPC: 用世界模型+价值)
    def mpc_act(obs):
        obs_t = torch.from_numpy(obs).float().to(dev).unsqueeze(0).repeat(5, 1)
        acts = torch.eye(5).to(dev)
        with torch.no_grad():
            sp, rp, _ = agent(obs_t, acts)
            score = V(sp)
        return int(score.argmax().item())

    lens, gains = [], []
    for _ in range(100):
        s = env.reset(); done = False; g = 0.0
        while not done:
            a = mpc_act(s)
            o2, r, dead, gained = env.step(a)
            g += gained; s = o2; done = dead
        lens.append(env.t); gains.append(g)
    print(f"能量价值+MPC: 平均存活 {np.mean(lens):.0f} 步, "
          f"能量获取 {np.mean(gains):.1f}/ep, 达 500 步 {sum(np.array(lens)>=500)}/100")
    # 效率: 能量获取 / 移动步数
    print(f"能量效率: 获取 {np.mean(gains):.1f} / 存活 {np.mean(lens):.0f} = "
          f"{np.mean(gains)/max(np.mean(lens),1):.3f} 能量/步")


if __name__ == "__main__":
    main()
