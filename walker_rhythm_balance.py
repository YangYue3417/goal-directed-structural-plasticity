"""walker_rhythm_balance.py — 节奏感步态 + 平衡专精神经群 + 抗扰动。

架构 (生物对应):
  步态层 (CPG):  相位状态机, 接触耦合 → 周期性交替 (节奏感)
  心跳层:        状态耦合振荡器 → 节奏调制 (活动→频率)
  平衡专精神经群 (小脑): 独立模块, 只处理躯干姿态
                  → 实时矫正 (抗扰动), 与步态分离
  动作 = 步态 + 平衡矫正

验证: 前进 + 平稳 (存活) + 抗扰动 (噪声下存活变化)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from walker_energy_env import WalkerEnergyEnv
from walker_gait import GaitPhase


class HeartbeatRhythm:
    """状态耦合心跳: 活动 → 频率; 调制步态幅度。"""
    def __init__(self):
        self.t = 0

    def intensity(self, obs):
        return abs(obs[2]) + abs(obs[3])  # 速度 = 活动强度

    def modulate(self, obs, base_amp):
        """活动强 → 步态幅度略增 (运动反馈)。"""
        I = self.intensity(obs)
        return base_amp * (1 + 0.15 * I)


class BalanceGroup(nn.Module):
    """平衡专精神经群 (小脑): 躯干姿态 → 矫正动作。

    输入: hull_angle, hull_angvel, vel (obs 0-3)
    输出: 4 维矫正 (叠加到步态)
    ES 进化矫正权重 (PID 式反惯的可学习版)。
    """
    def __init__(self, n_in=4):
        super().__init__()
        # 可学习矫正: 姿态误差 → 关节矫正 (相当于进化 PID 增益)
        self.W = nn.Parameter(torch.randn(n_in, 4) * 0.5)
        self.b = nn.Parameter(torch.zeros(4))

    def forward(self, obs):
        """obs: (B, 26) → 矫正 (B, 4)。姿态偏离 → 对抗矫正。"""
        pose = torch.stack([obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]], -1)
        corr = pose @ self.W + self.b
        return torch.tanh(corr) * 0.5   # 矫正幅度上限


class RhythmPolicy(nn.Module):
    """策略: 观测 → 步态参数 (每周期决策)。"""
    def __init__(self, obs=26, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, d), nn.ReLU(),
                                 nn.Linear(d, 2))
        self.scale = torch.tensor([0.5, 0.4])
        self.offset = torch.tensor([0.5, 0.5])

    def forward(self, s):
        return torch.sigmoid(self.net(s)) * self.scale.to(s.device) + self.offset.to(s.device)

    def params_for(self, s):
        dev = next(self.parameters()).device
        with torch.no_grad():
            p = self.forward(torch.from_numpy(s).float().to(dev).unsqueeze(0))[0]
        return p.cpu().numpy()


def eval_ep(rhythm_policy, balance, env, max_steps=1600, noise=0.0, seed=0):
    s = env.reset()
    leg1 = GaitPhase(A=0.8)
    leg2 = GaitPhase(A=0.8); leg2.ph = GaitPhase.SWING_FLEX
    hb = HeartbeatRhythm()
    total_r, dist = 0.0, 0.0
    for _ in range(max_steps):
        c1 = s[8] > 0.5; c2 = s[10] > 0.5
        a1 = leg1.action(); a2 = leg2.action()
        base = np.concatenate([a1, a2]).astype(np.float32)
        # 心跳调制幅度 (节奏感)
        amp = hb.modulate(s, 1.0)
        base = base * amp
        # 平衡专精矫正 (小脑)
        obs_t = torch.from_numpy(s).float().to(next(balance.parameters()).device).unsqueeze(0)
        with torch.no_grad():
            corr = balance(obs_t)[0].cpu().numpy()
        if noise > 0:
            corr += noise * np.random.randn(4).astype(np.float32)
        act = np.clip(base + corr, -1, 1)
        o2, r, d = env.step(act)
        total_r += r; dist = env.dist
        leg1.update(c1); leg2.update(c2)
        if leg1.ph == 0 and leg2.ph == 2:
            p = rhythm_policy.params_for(s)
            leg1.A = max(0.2, p[0]); leg2.A = max(0.2, p[0])
        s = o2
        if d:
            break
    return total_r, env.t, dist


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 节奏步态 + 平衡专精神经群 + 抗扰动 ===")
    env = WalkerEnergyEnv()
    rhythm = RhythmPolicy().to(dev)
    balance = BalanceGroup().to(dev)
    params = list(rhythm.parameters()) + list(balance.parameters())
    lr, sigma = 0.3, 0.2

    for it in range(30):
        deltas, scores = [], []
        for i in range(12):
            delta = []
            for p in params:
                d = torch.randn_like(p) * sigma
                delta.append(d); p.data.add_(d)
            sc, t, dist = eval_ep(rhythm, balance, env)
            for p, d in zip(params, delta): p.data.sub_(d)
            deltas.append(delta); scores.append(sc)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(params, delta):
                p.data.add_(lr * wi / (12 * sigma) * d)
        if it % 5 == 4:
            sc, t, dist = eval_ep(rhythm, balance, env)
            print(f"  iter {it+1}: 得分 {sc:+.1f} | 存活 {t} | 前进 {dist:.0f}")

    # 最终 + 抗扰动验证
    times, dists = [], []
    for seed in range(6):
        env2 = WalkerEnergyEnv()
        _, t, dist = eval_ep(rhythm, balance, env2)
        times.append(t); dists.append(dist)
    print(f"无扰动: 存活 {np.mean(times):.0f}, 前进 {np.mean(dists):.0f}")

    # 抗扰动: 动作噪声 + 姿态扰动
    t_noise, d_noise = [], []
    for seed in range(6):
        env3 = WalkerEnergyEnv()
        _, t, dist = eval_ep(rhythm, balance, env3, noise=0.1)
        t_noise.append(t); d_noise.append(dist)
    print(f"抗扰动 (动作噪声 0.1): 存活 {np.mean(t_noise):.0f} ({100*np.mean(t_noise)/max(np.mean(times),1):.0f}%), "
          f"前进 {np.mean(d_noise):.0f}")
    print(f"{'✅ 走起来+抗扰动' if np.mean(dists) > 40 else '⚠️ 有限'}")


if __name__ == "__main__":
    main()
