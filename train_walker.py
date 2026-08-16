"""train_walker.py — BipedalWalker: 多关节运动神经元协作。

用户洞察: Walker 4 关节 (2髋+2膝) = 4 个运动神经元集群,
协作产生走路 — 生长/删神经元 → 关节协作破坏。

迁移:
  生存 → 走路 (不跌倒, 前进)
  世界模型: (obs24 + act4连续) → (next obs + reward)
  策略: 世界模型 MPC (采样连续动作, 评分选最优)
  关节分组: 隐层神经元与各关节状态 (髋/膝角度) 的互信息 → 分组 → 删组验证

验证:
  ① 世界模型学到 Walker 动力学
  ② 采样 MPC 能否走路 (前进距离)
  ③ 关节神经元分组: 删"髋1控制组" → 该关节预测崩, 走路崩
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

import config as cfg


class ValueNet(nn.Module):
    """学习价值: 从经验学"哪些状态能走路" (MC 回报, 自研)。"""
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
    def forward(self, s):
        return self.net(s).squeeze(-1)


class WalkerWM(nn.Module):
    """世界模型: (obs24 + act4) → (next_obs24 + reward1)。"""

    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_s = nn.Linear(hidden, 24)
        self.head_r = nn.Linear(hidden, 1)

    def forward(self, sa):
        h = self.net(sa)
        return self.head_s(h), self.head_r(h).squeeze(-1)


def collect(n_eps=600, max_steps=60, seed=42):
    import gymnasium as gym
    env = gym.make('BipedalWalker-v3')
    rng = np.random.RandomState(seed)
    S, A, R, Sn = [], [], [], []
    lens = []
    for _ in range(n_eps):
        obs, _ = env.reset()
        t = 0
        for t in range(max_steps):
            a = rng.uniform(-1, 1, 4).astype(np.float32)
            obs_next, r, done, _, _ = env.step(a)
            S.append(obs); A.append(a); R.append(r); Sn.append(obs_next)
            obs = obs_next
            if done:
                break
        lens.append(t)
    env.close()
    print(f"  平均回合长度 {np.mean(lens):.0f}/{max_steps} (随机策略)")
    return (np.array(S, np.float32), np.array(A, np.float32),
            np.array(R, np.float32), np.array(Sn, np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--n_eps", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("收集 Walker 数据 (随机策略)...", flush=True)
    S, A, R, Sn = collect(args.n_eps)
    n = len(S)
    s_t = torch.from_numpy(S).float().to(device)
    a_t = torch.from_numpy(A).float().to(device)
    r_t = torch.from_numpy(R).float().to(device)
    sn_t = torch.from_numpy(Sn).float().to(device)
    sa_t = torch.cat([s_t, a_t], -1)

    model = WalkerWM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(n)[:8192]
        sp, rp = model(sa_t[idx])
        loss = F.mse_loss(sp, sn_t[idx]) + 0.5 * F.mse_loss(rp, r_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 30 == 29:
            model.eval()
            with torch.no_grad():
                sp, rp = model(sa_t[:2000])
                err = F.mse_loss(sp, sn_t[:2000]).item()
            print(f"  ep {ep+1}: loss={loss.item():.4f} 预测误差={err:.4f}", flush=True)
    torch.save({"model": model.state_dict()}, "runs/walker_wm.pt")

    # 学习 V: 从收集轨迹的 MC 回报
    print("训练学习价值 V...", flush=True)
    V = ValueNet().to(device)
    optv = torch.optim.AdamW(V.parameters(), lr=1e-3)
    idx = torch.randperm(n)[:20000]
    Xt, Gt = s_t[idx], r_t[idx]
    # 简化 MC: 用收集轨迹重算回报 (按 episode 边界, 这里用单步奖励近似)
    for ep in range(100):
        loss = F.mse_loss(V(Xt), Gt)
        optv.zero_grad(); loss.backward(); optv.step()
    torch.save({"model": V.state_dict()}, "runs/walker_v.pt")

    # 采样 MPC + 学习 V 评分 (纯框架)
    def mpc_act(obs, K=200):
        rng = np.random.RandomState()
        acts = rng.uniform(-1, 1, (K, 4)).astype(np.float32)
        sa = torch.from_numpy(
            np.concatenate([np.tile(obs, (K, 1)), acts], -1)).float().to(device)
        with torch.no_grad():
            sp, rp = model(sa)
            score = rp + 0.95 * V(sp)   # 学习价值, 非手动规则
        return acts[int(score.argmax().item())]

    import gymnasium as gym
    env = gym.make('BipedalWalker-v3')
    for tag, policy in [("采样MPC", mpc_act), ("随机", lambda o: np.random.uniform(-1, 1, 4))]:
        dists = []
        for _ in range(10):
            obs, _ = env.reset()
            total = 0.0
            done = False
            for _ in range(300):
                obs, r, done, _, _ = env.step(policy(obs))
                total += r
                if done:
                    break
            dists.append(total)
        print(f"  [{tag}] 平均奖励 {np.mean(dists):.1f} (300=满)", flush=True)
    env.close()
    print("保存: runs/walker_wm.pt")


if __name__ == "__main__":
    main()
