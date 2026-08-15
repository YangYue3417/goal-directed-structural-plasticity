"""C-M6: 能量觅食 — 饥饿权衡 (世界模型 + MC 价值 + 规划)。

回答: 闻不到远处食物时, agent 是否因饥饿压力选择远程觅食?

管线:
  1. WM10: 世界模型 (10 维 obs 含能量) 枚举训练 → 学能量动力学/死亡
  2. V: MC 价值函数 (随机策略 episode 的折扣回报回归)
  3. 规划: D=3 lookahead + V(s_D) bootstrap → 饥饿时远期价值可见
  4. 评估: 食物率 / 饿死率 / 能量-行为响应

对照: 无能量规划 (M2, 0.333) vs 有能量规划 vs 随机
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from envs.energy_maze import EnergyMaze
from world_models.train_world_model import WorldModel, ACT_SIZE

OBS10 = 10
ENERGY_LEVELS = [0.15, 0.5, 1.0]


def enumerate_transitions(env: EnergyMaze):
    """全状态 × 能量水平 × 动作 枚举 (能量使状态含能量维度)。"""
    S, A, R, Sn = [], [], [], []
    for x in range(env.size):
        for y in range(env.size):
            if env.grid[y, x]:
                continue
            for d in range(4):
                for ef in ENERGY_LEVELS:
                    for a in range(ACT_SIZE):
                        env.x, env.y, env.dir = x, y, d
                        env.energy = ef * env.E0
                        env.food_eaten = [False] * env.n_foods
                        env._visited = {(x, y)}
                        env.steps = 0
                        s = env.observe()
                        sn, r, _ = env.step(a)
                        S.append(s); A.append(a); R.append(r); Sn.append(sn)
    S = np.array(S, dtype=np.float32)
    A = np.array(A, dtype=np.int64)
    R = np.array(R, dtype=np.float32)
    Sn = np.array(Sn, dtype=np.float32)
    # 食物过采样
    food = R > 5
    if food.any():
        idx = np.repeat(np.nonzero(food)[0], 20)
        S = np.concatenate([S, S[idx]])
        A = np.concatenate([A, A[idx]])
        R = np.concatenate([R, R[idx]])
        Sn = np.concatenate([Sn, Sn[idx]])
    return S, A, R, Sn


def collect_returns(env, n_episodes=800, gamma=0.95, seed=0):
    """随机策略收集 (obs, 折扣回报) → 价值函数训练数据。"""
    rng = np.random.RandomState(seed)
    X, G = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        traj = []
        for _ in range(200):
            a = int(rng.randint(ACT_SIZE))
            obs_next, r, done = env.step(a)
            traj.append((obs, r))
            obs = obs_next
            if done:
                break
        G_ = 0.0
        for obs_t, r in reversed(traj):
            G_ = r + gamma * G_
            X.append(obs_t)
            G.append(G_)
    return np.array(X, dtype=np.float32), np.array(G, dtype=np.float32)


class ValueNet(nn.Module):
    def __init__(self, obs_dim=OBS10, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def plan_step(wm, V, obs, D=3, gamma=0.98, device="cuda"):
    """想象 rollout + 价值 bootstrap: score(seq) = Σ γ^k r_pred + γ^D V(s_D)"""
    states = [obs.copy()]
    scores = [0.0]
    for d in range(D):
        inp = np.stack([np.concatenate([s, np.eye(ACT_SIZE)[a]])
                        for s in states for a in range(ACT_SIZE)])
        s_pred, r_pred, _ = wm(torch.from_numpy(inp).float().to(device))
        s_pred = s_pred.cpu().numpy()
        r_pred = r_pred.cpu().numpy()
        ns, nsc = [], []
        g = gamma ** d
        for i, (s, sc) in enumerate(zip(states, scores)):
            for a in range(ACT_SIZE):
                idx = i * ACT_SIZE + a
                ns.append(s_pred[idx])
                nsc.append(sc + g * r_pred[idx])
        states, scores = ns, nsc
    # 价值 bootstrap
    inp = torch.from_numpy(np.stack(states)).float().to(device)
    v_pred = V(inp).cpu().numpy()
    scores = [s + gamma ** D * v for s, v in zip(scores, v_pred)]
    best = int(np.argmax(scores))
    return (best // (ACT_SIZE ** (D - 1))) % ACT_SIZE


def eval_planner(wm, V, env, n_episodes=50, D=3, device="cuda", random=False):
    foods, deaths = 0, 0
    for _ in range(n_episodes):
        obs = env.reset()
        died = False
        for _ in range(200):
            if random:
                a = np.random.randint(ACT_SIZE)
            else:
                a = plan_step(wm, V, obs, D, device=device)
            obs, r, done = env.step(a)
            if r > 5:
                foods += 1
            if done:
                if env.energy <= 0:
                    deaths += 1
                break
    n = n_episodes * env.n_foods
    return round(foods / n, 3), round(deaths / n_episodes, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--D", type=int, default=3)
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    env = EnergyMaze(size=20, n_foods=3, seed=42)
    print("=== 1. 世界模型 (10 维, 含能量) ===", flush=True)
    S, A, R, Sn = enumerate_transitions(env)
    n = len(S)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    split = int(n * 0.8)
    def to_t(a): return torch.from_numpy(a).float().to(device)
    S_t, Sn_t = to_t(S), to_t(Sn)
    A_oh = to_t(np.eye(ACT_SIZE)[A])
    R_t = to_t(R)
    wm = WorldModel(obs_size=OBS10, T=4).to(device)
    opt = torch.optim.AdamW(wm.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        wm.train()
        tot = 0.0
        for i in range(0, split, 512):
            idx = perm[i:i + 512]
            sa = torch.cat([S_t[idx], A_oh[idx]], 1)
            op, rp, _ = wm(sa)
            loss = F.mse_loss(op, Sn_t[idx]) + 0.2 * F.mse_loss(rp, R_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 9:
            wm.eval()
            with torch.no_grad():
                idx = perm[split:]
                sa = torch.cat([S_t[idx], A_oh[idx]], 1)
                op, rp, _ = wm(sa)
                print(f"  ep {ep+1}: loss={tot/split:.4f} pos_MAE={((op-Sn_t[idx])[:,6:8].abs()).mean():.4f} "
                      f"rew_MAE={(rp-R_t[idx]).abs().mean():.3f}", flush=True)
    torch.save({"model": wm.state_dict()}, "runs/wm_energy.pt")

    print("=== 2. MC 价值函数 ===", flush=True)
    X, G = collect_returns(env, n_episodes=3000, seed=args.seed)
    print(f"  {len(X)} 状态-回报对", flush=True)
    V = ValueNet().to(device)
    optv = torch.optim.AdamW(V.parameters(), lr=1e-3)
    xt = torch.from_numpy(X).float().to(device)
    gt = torch.from_numpy(G).float().to(device)
    for ep in range(100):
        idx = torch.randperm(len(X))[:4096].to(device)
        loss = F.mse_loss(V(xt[idx]), gt[idx])
        optv.zero_grad(); loss.backward(); optv.step()
        if ep % 20 == 19:
            print(f"  v ep {ep+1}: loss={loss.item():.4f}", flush=True)
    torch.save({"model": V.state_dict()}, "runs/v_energy.pt")

    print("=== 3. 规划评估 ===", flush=True)
    wm.load_state_dict(torch.load("runs/wm_energy.pt",
                                  map_location="cpu", weights_only=False)["model"])
    V.load_state_dict(torch.load("runs/v_energy.pt",
                                 map_location="cpu", weights_only=False)["model"])
    wm.eval(); V.eval()
    for tag, rnd in [("规划+D价值", False), ("随机对照", True)]:
        f, d = eval_planner(wm, V, env, args.n_episodes, args.D, device, rnd)
        print(f"  [{tag}] 食物率={f} 饿死率={d}")


if __name__ == "__main__":
    main()
