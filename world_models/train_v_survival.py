"""train_v_survival.py — 多地图价值函数 (V: 食物可达性)。

planner 用 D=3 想象 + V bootstrap → 跟随价值梯度 (绕过气味局部峰)。
V 训练: 多地图随机策略 episode 的折扣回报回归 (MC)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from envs.survival_maze import SurvivalMaze
from world_models.train_wm_energy import ValueNet
from world_models.train_wm_explore import ACT


def collect_returns(env, n_episodes=3000, gamma=0.95, seed=0):
    """多地图随机策略: (obs, 折扣回报)。"""
    rng = np.random.RandomState(seed)
    X, G = [], []
    for _ in range(n_episodes):
        env.energy = env.E0
        obs, _ = env.reset_day()
        traj = []
        for _ in range(env.day_steps):
            a = int(rng.randint(ACT))
            obs_next, r, done = env.step(a)
            traj.append((obs, r))
            obs = obs_next
            if done:
                break
        g = 0.0
        for obs_t, r in reversed(traj):
            g = r + gamma * g
            X.append(obs_t)
            G.append(g)
    return np.array(X, np.float32), np.array(G, np.float32)


def main():
    device = torch.device("cuda")
    env = SurvivalMaze(**cfg.SURVIVAL_ENV)
    print("收集多地图回报...", flush=True)
    X, G = collect_returns(env, n_episodes=3000)
    print(f"  {len(X)} 样本, 回报范围 [{G.min():.1f}, {G.max():.1f}]", flush=True)
    V = ValueNet(obs_dim=14).to(device)
    opt = torch.optim.AdamW(V.parameters(), lr=1e-3)
    xt = torch.from_numpy(X).float().to(device)
    gt = torch.from_numpy(G).float().to(device)
    for ep in range(80):
        idx = torch.randperm(len(X))[:8192].to(device)
        loss = F.mse_loss(V(xt[idx]), gt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 20 == 19:
            print(f"  ep {ep+1}: loss={loss.item():.4f}", flush=True)
    torch.save({"model": V.state_dict()}, "runs/v_survival.pt")
    print("保存: runs/v_survival.pt")


if __name__ == "__main__":
    main()
