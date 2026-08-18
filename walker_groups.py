"""walker_groups.py — 并行子任务专群: 平衡群 + 移动群 协作。

两群独立参数, 独立目标, 输出并行组合:
  平衡群 (阶段1): 姿态 → 矫正 (目标: 不跌倒) — 持续工作
  移动群 (阶段2): 状态窗口 → 步态 (目标: 前进) — 并行工作
  动作 = 移动群输出 + 平衡群矫正 (协作)
  移动群 ES 只更新自己参数 (不碰平衡群) — 专群独立优化
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from walker_curriculum import CurriculumEnv


class BalanceGroup(nn.Module):
    """平衡专群: 姿态 (angle, angvel, vel) → 矫正。"""
    def __init__(self, n_in=4, n_act=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.Tanh(), nn.Linear(32, n_act))

    def forward(self, obs):
        pose = obs[..., :4]
        return torch.tanh(self.net(pose)) * 0.4


class MoveGroup(nn.Module):
    """移动专群: 状态窗口 → 步态动作。"""
    def __init__(self, obs=26, L=8, act=4, d=64):
        super().__init__()
        self.L = L
        self.embed = nn.Linear(obs * L, d)
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, act))

    def forward(self, x):
        B = x.shape[0]
        return torch.tanh(self.net(torch.tanh(self.embed(x.reshape(B, -1)))))


def eval_ep(balance, move, env, move_only=False, max_steps=1600, seed=0):
    L = move.L
    hist = np.zeros((L, 26), np.float32)
    s = env.reset()
    hist[0] = s
    total_r, dist = 0.0, 0.0
    dev = next(move.parameters()).device
    for _ in range(max_steps):
        x = torch.from_numpy(hist).float().to(dev).unsqueeze(0)
        o_t = torch.from_numpy(s).float().to(dev).unsqueeze(0)
        with torch.no_grad():
            m = move(x)[0]
            b = balance(o_t)[0] if not move_only else torch.zeros_like(m)
        act = np.clip((m + b).cpu().numpy(), -1, 1).astype(np.float32)
        o2, r, d = env.step(act)
        total_r += r; dist = env.dist
        hist[1:] = hist[:-1]; hist[0] = o2
        s = o2
        if d:
            break
    return total_r, env.t, dist


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 并行子任务专群: 平衡群 + 移动群 协作 ===")
    balance = BalanceGroup().to(dev)
    move = MoveGroup().to(dev)
    lr, sigma = 0.3, 0.2

    # 阶段 1: 平衡群 (站立, 无规则)
    env1 = CurriculumEnv(move=False)
    for it in range(12):
        deltas, scores = [], []
        for i in range(10):
            delta = []
            for p in balance.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d); p.data.add_(d)
            sc, t, dist = eval_ep(balance, move, env1, move_only=True)
            for p, d in zip(balance.parameters(), delta): p.data.sub_(d)
            deltas.append(delta); scores.append(sc)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(balance.parameters(), delta):
                p.data.add_(lr * wi / (10 * sigma) * d)
        if it % 4 == 3:
            sc, t, _ = eval_ep(balance, move, env1, move_only=True)
            print(f"  平衡群 iter {it+1}: 站立存活 {t}")

    # 阶段 2: 移动群 (前进, 只更新 move, 平衡群保持)
    print("移动群学习 (平衡群保持参数, 并行协作)...")
    env2 = CurriculumEnv(move=True)
    for it in range(25):
        deltas, scores, dists = [], [], []
        for i in range(10):
            delta = []
            for p in move.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d); p.data.add_(d)
            sc, t, dist = eval_ep(balance, move, env2)
            for p, d in zip(move.parameters(), delta): p.data.sub_(d)
            deltas.append(delta); scores.append(sc); dists.append(dist)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(move.parameters(), delta):
                p.data.add_(lr * wi / (10 * sigma) * d)
        if it % 5 == 4:
            sc, t, dist = eval_ep(balance, move, env2)
            print(f"  移动群 iter {it+1}: 存活 {t}, 前进 {dist:.0f}")

    # 最终: 站立能力 (平衡群) + 移动能力 (协作)
    t_stand, t_move, d_move = [], [], []
    for _ in range(6):
        e1 = CurriculumEnv(move=False)
        _, t1, _ = eval_ep(balance, move, e1); t_stand.append(t1)
        e2 = CurriculumEnv(move=True)
        _, t2, d2 = eval_ep(balance, move, e2); t_move.append(t2); d_move.append(d2)
    print(f"\n站立 (协作): {np.mean(t_stand):.0f} 步 | 移动: 存活 {np.mean(t_move):.0f}, "
          f"前进 {np.mean(d_move):.0f}")
    print(f"{'✅ 并行协作: 平衡保持+移动推进' if np.mean(t_stand) > 300 and np.mean(d_move) > 20 else '⚠️ 有限'}")


if __name__ == "__main__":
    main()
