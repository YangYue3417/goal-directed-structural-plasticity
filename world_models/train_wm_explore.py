"""train_wm_explore.py — 去地图世界模型: 探索中自建认知地图。

中心实验 (自行发育神经元 × 专精 × 环境诱导):
  观测 = 局部传感器 (无位置, 13 维) → 位置观测唯一 (88/88 验证)
  模型 = GRU 循环积分 (路径积分) + SparseUnit 神经元池 (专精载体)
  目标 = 每步预测下一观测 (多步 BPTT) → 隐状态被迫编码空间结构

验证 (认知地图):
  隐状态/池激活 → 线性探针解码 (x,y): 无位置输入也能 >> 随机?

后续: 生长接入 (误差驱动募集) + 删神经元因果 + 30 昼夜回归
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

from envs.survival_maze import SurvivalMaze
from units.sparse_unit import SparseUnit

ACT = 3
SENSOR_DIMS = {"strong": 14, "weak": 8, "walls": 4}
OBS = SENSOR_DIMS["strong"]  # 默认; --sensor 时覆盖


class WorldExplore(nn.Module):
    """GRU 积分 + 稀疏神经元池的世界模型。"""

    def __init__(self, obs_dim=OBS, d=64, pool=512, top_k=64,
                 active_ratio=0.25, hidden=128):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(obs_dim + ACT, d)
        # 神经元池: 专精载体 (可生长: active_mask 由池管理)
        self.unit = SparseUnit(d_model=d, d_pool=pool, top_k=top_k)
        n_init = max(1, int(pool * active_ratio))
        self.unit.register_buffer("active_mask",
                                  torch.zeros(pool, dtype=torch.bool))
        self.unit.active_mask[:n_init] = True
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.head_obs = nn.Linear(hidden, obs_dim)
        self.head_rew = nn.Linear(hidden, 1)
        self.skip_obs = nn.Linear(obs_dim + ACT, obs_dim)
        self.skip_rew = nn.Linear(obs_dim + ACT, 1)

    def forward(self, obs, actions):
        """obs: (B, T, obs_dim) actions: (B, T) → (obs_pred, rew_pred, h_seq, pool_stats)"""
        B, T = obs.shape[:2]
        sa = torch.cat([obs, F.one_hot(actions.long(), ACT).float()], dim=-1)
        emb = torch.tanh(self.embed(sa))  # (B, T, d)
        # 神经元池: 每步稀疏激活
        pool_out, pool_stats = self.unit(emb)  # (B, T, d)
        h, _ = self.gru(pool_out)  # (B, T, hidden)
        obs_pred = self.head_obs(h) + self.skip_obs(sa)
        rew_pred = self.head_rew(h).squeeze(-1) + self.skip_rew(sa).squeeze(-1)
        return obs_pred, rew_pred, h, pool_stats


def collect_trajectories(env, n_episodes=2000, max_steps=60, seed=42):
    """多地图轨迹 (通用规律): 每 episode 新地图, 能量探测时重置。"""
    rng = np.random.RandomState(seed)
    S, A, R, Sn, D = [], [], [], [], []
    P = []  # 位置 (评估用)
    for _ in range(n_episodes):
        env.energy = env.E0  # 探测: 满能量出发 (不受跨天能量影响)
        obs, _ = env.reset_day()  # 新地图 + 新食物
        for _ in range(max_steps):
            a = int(rng.randint(ACT))
            obs_next, r, done = env.step(a)
            S.append(obs); A.append(a); R.append(r)
            Sn.append(obs_next); D.append(done)
            P.append([env.x / env.size, env.y / env.size])
            obs = obs_next
            if done:
                break
    S = np.array(S, np.float32); A = np.array(A, np.int64)
    R = np.array(R, np.float32); Sn = np.array(Sn, np.float32)
    D = np.array(D, np.float32); P = np.array(P, np.float32)
    # 食物过采样 (奖励头学 +10)
    food = R > 5
    if food.any():
        idx = np.repeat(np.nonzero(food)[0], 50)
        S = np.concatenate([S, S[idx]]); A = np.concatenate([A, A[idx]])
        R = np.concatenate([R, R[idx]]); Sn = np.concatenate([Sn, Sn[idx]])
        D = np.concatenate([D, D[idx]]); P = np.concatenate([P, P[idx]])
    return S, A, R, Sn, D, P


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--T", type=int, default=30, help="BPTT 轨迹长度")
    p.add_argument("--n_episodes", type=int, default=2000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sensor", type=str, default="strong",
                   choices=["strong", "weak", "walls"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    global OBS
    OBS = SENSOR_DIMS[args.sensor]
    env = SurvivalMaze(size=10, n_foods=6, seed=42, E0=200.0, day_steps=60,
                       food_restore=80.0, sensor=args.sensor)
    print("收集轨迹...", flush=True)
    S, A, R, Sn, D, P = collect_trajectories(env, args.n_episodes)
    n = len(S)
    print(f"  {n} 步, 死亡 {(D==1).sum()} 条", flush=True)

    model = WorldExplore(obs_dim=OBS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    s_t = torch.from_numpy(S).float().to(device)
    a_t = torch.from_numpy(A).long().to(device)
    r_t = torch.from_numpy(R).float().to(device)
    sn_t = torch.from_numpy(Sn).float().to(device)

    # 轨迹窗口切分 (BPTT)
    n_win = n // args.T
    idx_w = np.random.RandomState(0).permutation(n_win)
    split = int(n_win * 0.8)

    print(f"训练 (GRU + 神经元池, BPTT T={args.T})...", flush=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for i in range(0, split * args.bs, args.bs):
            wi = idx_w[i // args.bs:(i // args.bs) + (args.bs if False else 1)]
            # 采 batch 个窗口
            sel = np.random.choice(split, args.bs, replace=False)
            idx = np.concatenate([np.arange(w * args.T, (w + 1) * args.T) for w in sel])
            sb = s_t[idx].view(args.bs, args.T, OBS)
            ab = a_t[idx].view(args.bs, args.T)
            rb = r_t[idx].view(args.bs, args.T)
            snb = sn_t[idx].view(args.bs, args.T, OBS)
            op, rp, _, _ = model(sb, ab)
            loss = F.mse_loss(op, snb) + 0.5 * F.mse_loss(rp, rb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        # 验证: 预测 + 认知地图探针 (隐状态解码位置)
        model.eval()
        with torch.no_grad():
            wi = np.arange(split, n_win)
            sel = wi[:args.bs]
            idx = np.concatenate([np.arange(w * args.T, (w + 1) * args.T) for w in sel])
            sb = s_t[idx].view(len(sel), args.T, OBS)
            ab = a_t[idx].view(len(sel), args.T)
            rb = r_t[idx].view(len(sel), args.T)
            snb = sn_t[idx].view(len(sel), args.T, OBS)
            op, rp, h, ps = model(sb, ab)
            obs_mae = (op - snb).abs().mean().item()
            wall_acc = ((op[:, :, 3:6] > 0.5).float() == (snb[:, :, 3:6] > 0.5)).float().mean().item()
            # 认知地图探针: 末步隐状态 → 解码位置 (P 是外部记录, 模型没见过)
            p_t = torch.from_numpy(P[idx]).float().to(device).view(len(sel), args.T, 2)
            h_last = h[:, -1].cpu().numpy()
            y_last = p_t[:, -1].cpu().numpy()
            if ep % 5 == 4:
                # ridge 探针 (用一半窗口训, 一半测)
                n2 = len(h_last)
                Xtr, ytr = h_last[:n2//2], y_last[:n2//2]
                Xva, yva = h_last[n2//2:], y_last[n2//2:]
                lam = 1e-3 * Xtr.shape[1]
                W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
                pred = Xva @ W
                acc = float(np.mean(np.abs(pred - yva).max(1) < 0.05))
                print(f"  认知地图: 位置解码 acc={acc:.3f} (随机≈0.01)", flush=True)
        print(f"  ep {ep+1}: loss={tot/split:.4f} | obs_MAE={obs_mae:.4f} wall_acc={wall_acc:.3f}",
              flush=True)

    torch.save({"model": model.state_dict(), "config": vars(args)},
               f"runs/wm_explore_{args.sensor}.pt")
    print("保存: runs/wm_explore.pt")


if __name__ == "__main__":
    main()
