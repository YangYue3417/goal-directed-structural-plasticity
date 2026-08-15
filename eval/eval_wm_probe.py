"""C-M3: 隐状态位置解码 — 世界模型是否在隐状态编码空间?

核心可证伪预测 (DESIGN §4):
  P1: 世界模型隐状态位置解码显著 > 直接策略 (W1 vs D1)
  P2: 无位置输入的 W2 仍能解码位置 → 空间自组织

方法: 枚举迷宫全部 (x,y,朝向) 状态 → 提取隐状态 (L2 膜电位/发放率)
      → 线性探针 (ridge/logistic) 解 (x,y) → 解码准确率 (容差 1 格)
对比: W1 世界模型 / W2 世界模型 / D1 直接策略 (runs/maze_nav_fixed_best.pt)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from envs.maze_nav import MazeNav, DIRS
from world_models.train_world_model import WorldModel, OBS_SIZE, ACT_SIZE
from model.sparse_snn import SparseSNN


def encode_all_states(env, enc_fn, device):
    """枚举全部 (x,y,dir) → 隐状态特征 (LIF 膜电位 sigmoid)。"""
    feats, pos = [], []
    for x in range(env.size):
        for y in range(env.size):
            if env.grid[y, x]:
                continue
            for d in range(4):
                env.x, env.y, env.dir = x, y, d
                env.food_eaten = [False] * env.n_foods
                env.steps = 0
                env._visited = {(x, y)}
                obs = env.observe()
                f = enc_fn(obs)
                feats.append(f)
                pos.append([x / env.size, y / env.size])
    return np.array(feats), np.array(pos)


def make_encoder_wm(wm, no_pos=False):
    @torch.no_grad()
    def enc(obs):
        o = np.delete(obs, [6, 7]) if no_pos else obs
        sa = torch.from_numpy(np.concatenate([o, np.zeros(ACT_SIZE)])[None]).float()
        _, _, stats = wm(sa.cuda())
        return stats["l2_rate"].cpu().numpy()[0]  # (256,)
    return enc


def make_encoder_policy(net):
    @torch.no_grad()
    def enc(obs):
        sa = torch.from_numpy(obs[None]).float()
        _, stats = net(sa.cuda())
        return stats["l2_rate"].cpu().numpy()[0]
    return enc


def linear_probe(X, y, seed=0):
    """ridge 回归 + 留出, 返回 MAE (归一化单位) 和 1 格容差准确率。"""
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    tr, va = idx[:int(n * 0.7)], idx[int(n * 0.7):]
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    # ridge: (X^T X + λI)^-1 X^T y
    lam = 1e-3 * Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    pred = Xva @ W
    mae = np.abs(pred - yva).mean()
    acc = float(np.mean(np.abs(pred - yva).max(1) < (1.0 / 20)))
    return mae, acc


def main():
    env = MazeNav(size=20, n_foods=3, seed=42)
    device = torch.device("cuda")
    results = {}

    # W1: 有位置输入的世界模型
    wm1 = WorldModel(obs_size=OBS_SIZE, T=4).to(device)
    wm1.load_state_dict(torch.load("runs/world_model_W1_pos.pt",
                                   map_location="cpu", weights_only=False)["model"])
    wm1.eval()
    X, y = encode_all_states(env, make_encoder_wm(wm1), device)
    mae, acc = linear_probe(X, y)
    results["W1 世界模型(有位置)"] = (mae, acc)

    # W2: 无位置输入
    wm2 = WorldModel(obs_size=OBS_SIZE - 2, T=4).to(device)
    wm2.load_state_dict(torch.load("runs/world_model_W2_nopos.pt",
                                   map_location="cpu", weights_only=False)["model"])
    wm2.eval()
    X2, _ = encode_all_states(env, make_encoder_wm(wm2, no_pos=True), device)
    mae2, acc2 = linear_probe(X2, y)
    results["W2 世界模型(无位置)"] = (mae2, acc2)

    # D1: 直接策略
    net = SparseSNN(obs_size=9, n1=256, n2=256).to(device)
    net.load_state_dict(torch.load("runs/maze_nav_fixed_best.pt",
                                   map_location="cpu", weights_only=False)["model"])
    net.eval()
    Xd, _ = encode_all_states(env, make_encoder_policy(net), device)
    maed, accd = linear_probe(Xd, y)
    results["D1 直接策略"] = (maed, accd)

    # 随机基线 (随机特征)
    Xr = np.random.RandomState(0).randn(len(X), 256)
    maer, accr = linear_probe(Xr, y)
    results["随机特征基线"] = (maer, accr)

    print("=== 隐状态位置解码 (线性探针) ===")
    print(f"{'来源':<22}{'位置MAE':>10}{'1格准确率':>10}")
    for k, (m, a) in results.items():
        print(f"{k:<22}{m:>10.4f}{a:>10.3f}")


if __name__ == "__main__":
    main()
