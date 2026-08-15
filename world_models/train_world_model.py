"""C-M1: SNN 世界模型 — 学习 (s, a) → (s', r)。监督学习, 零 RL 方差。

动机 (DESIGN_neural_world_model.md):
  预测目标使空间信息成为硬需求 → 隐状态被迫编码空间 (M3 验证)。

管线:
  1. 随机探索收集 transitions (20×20 固定迷宫, 覆盖状态空间)
  2. WorldModel: LIF (obs+action → 隐状态) → 双头 (obs_next 回归 + reward 回归)
  3. 监督训练 (MSE), 无策略梯度
  4. 评估: 一步预测 per-component MAE + 对照 (持久性基线)

用法: python train_world_model.py [--no_pos 移除 x,y 输入 (W2 臂)]
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

from envs.maze_nav import MazeNav
from model.sparse_snn import SparseLIFLayer

OBS_SIZE = 9
ACT_SIZE = 3


class WorldModel(nn.Module):
    """LIF 世界模型: (obs, action) → (obs_next, reward)。"""

    def __init__(self, obs_size: int = OBS_SIZE, n1: int = 256, n2: int = 256,
                 k1: int = 6, k2: int = 12, T: int = 4, seed: int = 0,
                 out_size: int | None = None):
        super().__init__()
        self.T = T
        self.obs_size = obs_size
        self.out_size = obs_size if out_size is None else out_size
        self.layer1 = SparseLIFLayer(obs_size + ACT_SIZE, n1, k=k1, seed=seed)
        self.layer2 = SparseLIFLayer(n1, n2, k=k2, seed=seed)
        self.head_obs = nn.Linear(n2, self.out_size)
        self.head_rew = nn.Linear(n2, 1)
        # 直连旁路: LIF 动力学 + 线性残差 (脉冲量化不伤连续回归精度)
        self.skip_obs = nn.Linear(obs_size + ACT_SIZE, self.out_size)
        self.skip_rew = nn.Linear(obs_size + ACT_SIZE, 1)

    def forward(self, s_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """s_a: (B, obs+3) → (obs_next_pred, reward_pred, stats)
        读出用膜电位 u (连续量) — 脉冲发放率量化伤连续回归精度。
        """
        u1 = self.layer1.forward_u(s_a, self.T)
        u2 = self.layer2.forward_u(u1, self.T)
        obs_pred = self.head_obs(u2) + self.skip_obs(s_a)
        rew_pred = self.head_rew(u2).squeeze(-1) + self.skip_rew(s_a).squeeze(-1)
        stats = {"l2_rate": torch.sigmoid(u2), "l2_u": u2}
        return obs_pred, rew_pred, stats


def collect(env: MazeNav, n_episodes: int = 600, max_steps: int = 100,
            seed: int = 0, no_pos: bool = False):
    """全状态枚举: 每 (位置, 方向, 动作) 一条 transition (确定性世界完整覆盖)。

    比随机探索好: 食物 transition 全部包含, 无冗余, 一步到位。
    """
    from envs.maze_nav import DIRS
    size = env.size
    transitions = []
    for x in range(size):
        for y in range(size):
            if env.grid[y, x]:
                continue  # 墙位置
            for d in range(4):
                for a in range(ACT_SIZE):
                    env.x, env.y, env.dir = x, y, d
                    env.food_eaten = [False] * env.n_foods
                    env._visited = {(x, y)}
                    env.steps = 0
                    s = env.observe()
                    sn, r, _ = env.step(a)
                    transitions.append((s, a, r, sn))
    # 转成数组
    n = len(transitions)
    S = np.zeros((n, OBS_SIZE), dtype=np.float32)
    A = np.zeros((n, ACT_SIZE), dtype=np.float32)
    R = np.zeros(n, dtype=np.float32)
    Sn = np.zeros((n, OBS_SIZE), dtype=np.float32)
    for i, (s, a, r, sn) in enumerate(transitions):
        S[i] = s
        A[i, a] = 1.0
        R[i] = r
        Sn[i] = sn
    return S, A, R, Sn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no_pos", action="store_true", help="W2 臂: 去掉 x,y 输入")
    p.add_argument("--n_episodes", type=int, default=600)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    tag = "W2_nopos" if args.no_pos else "W1_pos"
    env = MazeNav(size=20, n_foods=3, seed=42)
    print(f"收集数据 (全状态枚举, {tag})...", flush=True)
    S, A, R, Sn = collect(env, args.n_episodes, args.max_steps, args.seed, args.no_pos)
    # 食物 transition 过采样 (rare events: 只有 ~9 条, MSE 忽略它们 → 奖励学不到)
    food_mask = R > 5
    if food_mask.any():
        oversample = np.repeat(np.nonzero(food_mask)[0], 20)
        S = np.concatenate([S, S[oversample]])
        A = np.concatenate([A, A[oversample]])
        R = np.concatenate([R, R[oversample]])
        Sn = np.concatenate([Sn, Sn[oversample]])
    n = len(S)
    split = int(n * 0.8)
    # 持久性基线 (预测 obs_next = obs, 位置列): 用全量数据计算
    base_pos = np.abs(S[split:] - Sn[split:])[:, 6:8].mean()
    if args.no_pos:
        S = np.delete(S, [6, 7], axis=1)  # W2: 输入去掉位置, 输出仍预测全量
    print(f"  {n} transitions (食物过采样后), obs_dim={S.shape[1]}", flush=True)

    # 分割: 8:2 训练/验证 (随机洗牌 → iid, 枚举是按坐标排序的)
    rng = np.random.RandomState(0)
    perm_all = rng.permutation(n)
    def to_t(a): return torch.from_numpy(a).float().to(device)
    tr = (to_t(S[perm_all[:split]]), to_t(A[perm_all[:split]]),
          to_t(R[perm_all[:split]]), to_t(Sn[perm_all[:split]]))
    va = (to_t(S[perm_all[split:]]), to_t(A[perm_all[split:]]),
          to_t(R[perm_all[split:]]), to_t(Sn[perm_all[split:]]))

    model = WorldModel(obs_size=S.shape[1], T=args.T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"WorldModel: obs_dim={S.shape[1]} LIF(256×2, T={args.T})", flush=True)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(split)
        tot = 0.0
        for i in range(0, split, args.batch_size):
            idx = perm[i:i + args.batch_size]
            s_b, a_b, r_b, sn_b = [t[idx] for t in tr]
            obs_pred, rew_pred, _ = model(torch.cat([s_b, a_b], dim=1))
            loss = F.mse_loss(obs_pred, sn_b) + 0.1 * F.mse_loss(rew_pred, r_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        # 验证
        model.eval()
        with torch.no_grad():
            s_b, a_b, r_b, sn_b = va
            obs_pred, rew_pred, _ = model(torch.cat([s_b, a_b], dim=1))
            mae = (obs_pred - sn_b).abs()
            pos_mae = mae[:, 6:8].mean().item()  # 位置列恒在 6:8 (W2 也预测全量)
            wall_acc = ((obs_pred[:, 3:6] > 0.5).float() == sn_b[:, 3:6]).float().mean().item()
            smell_mae = mae[:, :3].mean().item()
            rew_mae = (rew_pred - r_b).abs().mean().item()
        print(f"  ep {ep+1}: loss={tot/split:.4f} | pos_MAE={pos_mae:.4f} "
              f"wall_acc={wall_acc:.3f} smell_MAE={smell_mae:.4f} rew_MAE={rew_mae:.3f}",
              flush=True)

    # 持久性基线 (预测 obs_next = obs): 位置不变时误差
    print(f"持久性基线 pos_MAE={base_pos:.4f} (预测=原地)")
    out = Path(f"runs/world_model_{tag}.pt")
    torch.save({"model": model.state_dict(), "config": vars(args)}, out)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
