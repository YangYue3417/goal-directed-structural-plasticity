"""walker_es.py — Walker 优化: ES 策略 (连续动作) + 能量+驱动规则环境。

之前的失败 (价值+MPC 静止陷阱) → 用 ES 策略解决:
  策略: 观测 (26) → MLP → 4 维连续动作 (tanh)
  ES:   低秩扰动 → 跑 episode (能量环境) → 得分 → 加权更新
  环境: WalkerEnergyEnv (输出能量 + 不动即死规则 + 阶段补充)

验证: 存活 + 前进距离 (vs 之前 81 步/前进 2)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from walker_energy_env import WalkerEnergyEnv


class WalkerPolicy(nn.Module):
    """连续动作策略: 观测 → 4 维动作 (tanh ∈ [-1,1])。"""
    def __init__(self, obs=26, act=4, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, d), nn.ReLU(),
                                 nn.Linear(d, d), nn.ReLU(),
                                 nn.Linear(d, act))
        self.act_scale = 1.0

    def forward(self, s):
        return torch.tanh(self.net(s)) * self.act_scale

    def act(self, s, noise=0.0):
        dev = next(self.parameters()).device
        with torch.no_grad():
            a = self.forward(torch.from_numpy(s).float().to(dev).unsqueeze(0))[0]
            if noise > 0:
                a = a + noise * torch.randn_like(a)
            return np.clip(a.cpu().numpy(), -1, 1).astype(np.float32)


def eval_episode(policy, env, max_steps=1600, noise=0.0, seed=0):
    """跑一个 episode: 返回 (得分, 存活, 前进)。"""
    s = env.reset()
    total_r, dist = 0.0, 0.0
    for _ in range(max_steps):
        a = policy.act(s, noise=noise)
        o2, r, d = env.step(a)
        total_r += r
        dist = env.dist
        s = o2
        if d:
            break
    return total_r, env.t, dist


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== Walker 优化: ES 策略 (能量+驱动规则) ===")
    env = WalkerEnergyEnv()
    policy = WalkerPolicy().to(dev)
    lr, sigma = 0.2, 0.15

    # 基线: 零动作 (驱动规则死亡线) / 随机
    _, t0, d0 = eval_episode(policy, env, noise=1.0)  # 纯随机
    print(f"随机动作: 存活 {t0} 步, 前进 {d0:.0f}")
    print(f"对照: 零动作 81 步死 (驱动规则), 价值+MPC 之前 81 步/前进 2")

    for it in range(30):
        deltas, scores, dists = [], [], []
        for i in range(12):
            delta = []
            for p in policy.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d)
                p.data.add_(d)
            sc, t, dist = eval_episode(policy, env, noise=0.0)
            for p, d in zip(policy.parameters(), delta):
                p.data.sub_(d)
            deltas.append(delta); scores.append(sc); dists.append(dist)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(policy.parameters(), delta):
                p.data.add_(lr * wi / (12 * sigma) * d)
        if it % 5 == 4:
            sc, t, dist = eval_episode(policy, env)
            print(f"  iter {it+1}: 得分 {sc:+.1f} | 存活 {t} 步 | 前进 {dist:.0f}")

    # 最终评估 (多 seed)
    times, dists = [], []
    for seed in range(6):
        env2 = WalkerEnergyEnv()
        _, t, dist = eval_episode(policy, env2)
        times.append(t); dists.append(dist)
    print(f"\n最终: 存活 {np.mean(times):.0f} 步, 前进 {np.mean(dists):.0f} "
          f"(vs 之前价值+MPC 81步/前进2)")
    print(f"{'✅ Walker 学会移动' if np.mean(dists) > 20 else '⚠️ 未学会移动'}")


if __name__ == "__main__":
    main()
