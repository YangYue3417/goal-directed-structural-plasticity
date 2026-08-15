"""C-M5 探针: 时间积分模型的隐状态能否解码位置 (自组织测试)。

对比基线 (M3): W1 有位置 94.9% | W2 单步 5.3% | 随机 0.0%
预期: 窗口隐状态解码显著 > 5.3% → 时间积分自组织成立
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from envs.maze_nav import MazeNav
from train_wm_seq import LIFIntegrator, collect_windows, OBS_SIZE


def probe(X, y, seed=0):
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    tr, va = idx[:int(n * 0.7)], idx[int(n * 0.7):]
    lam = 1e-3 * X.shape[1]
    W = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ y[tr])
    pred = X[va] @ W
    mae = np.abs(pred - y[va]).mean()
    acc = float(np.mean(np.abs(pred - y[va]).max(1) < (1.0 / 20)))
    return mae, acc


def main():
    K = 8
    env = MazeNav(size=20, n_foods=3, seed=42)
    device = torch.device("cuda")

    model = LIFIntegrator(obs_dim=7, T=K).to(device)
    model.load_state_dict(torch.load("runs/wm_seq_W2_window.pt",
                                     map_location="cpu", weights_only=False)["model"])
    model.eval()

    print("收集窗口数据...", flush=True)
    S, A, R, Sn = collect_windows(env, K, n_episodes=800, no_pos=True)
    n = len(S)
    feats = np.zeros((n, 256))
    with torch.no_grad():
        for i in range(0, n, 512):
            sb = torch.from_numpy(S[i:i + 512]).float().to(device)
            ab = torch.from_numpy(A[i:i + 512]).long().to(device)
            _, _, u = model(sb, ab)
            feats[i:i + 512] = u.cpu().numpy()
    y = Sn[:, 6:8]  # 目标位置

    mae, acc = probe(feats, y)
    print(f"=== 时间积分 (K={K}, 无位置输入) 隐状态位置解码 ===")
    print(f"  位置MAE={mae:.4f}  1格准确率={acc:.3f}")
    print(f"  对比: W1有位置 94.9% | W2单步 5.3% | 随机 0.0%")

    # 随机基线
    Xr = np.random.RandomState(0).randn(n, 256)
    mr, ar = probe(Xr, y)
    print(f"  随机特征基线: MAE={mr:.4f} acc={ar:.3f}")


if __name__ == "__main__":
    main()
