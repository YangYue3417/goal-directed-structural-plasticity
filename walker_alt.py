"""walker_alt.py — 交替周期 = 局部单元 (用户设计)。

策略: 观测 → 每交替周期的参数 (A_hip, A_knee, T)
执行器: 左腿 sin(ph), 右腿 -sin(ph) 反相交替 (周期结构先验)
ES 优化参数策略 (低维: 3 参数/周期 vs 4 动作/步)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from walker_energy_env import WalkerEnergyEnv


class AltPolicy(nn.Module):
    """策略: 观测 → 交替参数 (每周期一次决策)。"""
    def __init__(self, obs=26, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, d), nn.ReLU(),
                                 nn.Linear(d, 3))
        # 输出: A_hip ∈ [0.2,1], A_knee ∈ [0,0.6], T ∈ [20,80]
        self.scale = torch.tensor([0.4, 0.3, 30.0])
        self.offset = torch.tensor([0.6, 0.3, 50.0])

    def forward(self, s):
        return torch.sigmoid(self.net(s)) * self.scale.to(s.device) + self.offset.to(s.device)

    def params_for(self, s):
        dev = next(self.parameters()).device
        with torch.no_grad():
            p = self.forward(torch.from_numpy(s).float().to(dev).unsqueeze(0))[0]
        return p.cpu().numpy()


class Alternator:
    """交替执行器: 左腿 sin, 右腿 -sin (反相), 周期 T 参数化。"""
    def __init__(self):
        self.ph = 0.0
        self.T = 50.0
        self.A1 = 0.7
        self.A2 = 0.3

    def set(self, p):
        self.A1, self.A2, self.T = p[0], p[1], max(10, p[2])

    def action(self):
        ph = 2 * np.pi * self.ph / self.T
        self.ph += 1
        if self.ph >= self.T:
            self.ph = 0.0
        hip = self.A1 * np.sin(ph)
        knee = self.A2 * np.sin(ph + np.pi / 2)
        return np.array([hip, knee, -hip, -knee], np.float32)


def eval_ep(policy, env, max_steps=1600, seed=0):
    s = env.reset()
    alt = Alternator()
    alt.set(policy.params_for(s))
    total_r, dist = 0.0, 0.0
    for _ in range(max_steps):
        a = alt.action()
        o2, r, d = env.step(a)
        total_r += r
        dist = env.dist
        s = o2
        # 每周期更新参数 (观测驱动的局部决策)
        if alt.ph == 0:
            alt.set(policy.params_for(s))
        if d:
            break
    return total_r, env.t, dist


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== Walker 交替周期局部单元 (ES 优化参数策略) ===")
    env = WalkerEnergyEnv()
    policy = AltPolicy().to(dev)
    lr, sigma = 0.3, 0.2

    r0, t0, d0 = eval_ep(policy, env)
    print(f"初始 (随机参数): 存活 {t0} 步, 前进 {d0:.0f}")

    for it in range(25):
        deltas, scores, dists = [], [], []
        for i in range(12):
            delta = []
            for p in policy.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d)
                p.data.add_(d)
            sc, t, dist = eval_ep(policy, env)
            for p, d in zip(policy.parameters(), delta):
                p.data.sub_(d)
            deltas.append(delta); scores.append(sc); dists.append(dist)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(policy.parameters(), delta):
                p.data.add_(lr * wi / (12 * sigma) * d)
        if it % 5 == 4:
            sc, t, dist = eval_ep(policy, env)
            print(f"  iter {it+1}: 得分 {sc:+.1f} | 存活 {t} | 前进 {dist:.0f}")

    times, dists = [], []
    for seed in range(6):
        env2 = WalkerEnergyEnv()
        _, t, dist = eval_ep(policy, env2)
        times.append(t); dists.append(dist)
    print(f"\n最终: 存活 {np.mean(times):.0f}, 前进 {np.mean(dists):.0f} "
          f"(vs 逐动作 ES 11)")
    print(f"{'✅ 交替局部单元有效 (前进显著)' if np.mean(dists) > 30 else '⚠️ 有限'}")


if __name__ == "__main__":
    main()
