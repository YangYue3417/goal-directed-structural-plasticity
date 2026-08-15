"""eval_wm_explore.py — 认知地图评估 (固定地图参照系)。

训练: 多地图通用规律 (train_wm_explore.py)
验证: 固定地图内 (seed=777) 跑轨迹 → 隐状态解码位置
  = "到新地图当天构建认知地图" (通用规律 + 轨迹积分)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from envs.survival_maze import SurvivalMaze
from world_models.train_wm_explore import WorldExplore, ACT, OBS


def ridge_probe(X, y, seed=0):
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    tr, va = idx[:int(n * 0.7)], idx[int(n * 0.7):]
    lam = 1e-3 * X.shape[1]
    W = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ y[tr])
    pred = X[va] @ W
    mae = np.abs(pred - y[va]).mean()
    acc = float(np.mean(np.abs(pred - y[va]).max(1) < 0.05))
    return mae, acc


def main():
    device = torch.device("cuda")
    env = SurvivalMaze(size=20, n_foods=3, seed=42)
    model = WorldExplore().to(device)
    model.load_state_dict(torch.load("runs/wm_explore.pt",
                                     map_location="cpu", weights_only=False)["model"])
    model.eval()

    # 固定地图 (seed=777), 天内多 episode 随机轨迹
    print("收集固定地图轨迹...", flush=True)
    env._day_seed = 777
    env.energy = env.E0
    env.reset_day()
    rng = np.random.RandomState(7)
    S, A, P = [], [], []
    for _ in range(300):
        obs = env.reset()
        for _ in range(60):
            a = int(rng.randint(ACT))
            obs_next, r, done = env.step(a)
            S.append(obs); A.append(a)
            P.append([env.x / env.size, env.y / env.size])
            obs = obs_next
            if done:
                break
    S = np.array(S, np.float32)
    A = np.array(A, np.int64)
    P = np.array(P, np.float32)
    n = len(S)
    s_t = torch.from_numpy(S).float().to(device)
    a_t = torch.from_numpy(A).long().to(device)

    T = 30
    n_win = n // T
    H, Hp, Y = [], [], []
    with torch.no_grad():
        for w0 in range(0, n_win, 64):
            wsel = np.arange(w0, min(w0 + 64, n_win))
            idx = np.concatenate([np.arange(w * T, (w + 1) * T) for w in wsel])
            nb = len(idx) // T
            sb = s_t[idx].view(nb, T, OBS)
            ab = a_t[idx].view(nb, T)
            _, _, h, ps = model(sb, ab)
            H.append(h.cpu().numpy().reshape(-1, 128))
            sel = ps.selected.cpu().numpy()[:, 0, :]
            act = np.zeros((nb, 512))
            np.put_along_axis(act, sel, 1.0, axis=1)
            Hp.append(act.reshape(-1, 512))
            p_t = torch.from_numpy(P[idx]).float()
            Y.append(p_t.view(nb, T, 2).cpu().numpy().reshape(-1, 2))
    H = np.concatenate(H)
    Hp = np.concatenate(Hp)
    Y = np.concatenate(Y)
    idx_all = np.concatenate([np.arange(w * T, (w + 1) * T)
                              for w in range(n_win)])
    S_win = S[idx_all]
    print(f"  {len(H)} 样本 (固定地图 seed=777)", flush=True)

    print("=== 认知地图探针 (固定地图内) ===")
    for tag, X in [("GRU 隐状态", H), ("池神经元", Hp)]:
        mae, acc = ridge_probe(X, Y)
        print(f"  [{tag}] MAE={mae:.4f} acc={acc:.3f}")
    mae1, acc1 = ridge_probe(S_win, Y)
    print(f"  [单帧观测] MAE={mae1:.4f} acc={acc1:.3f}")
    Xr = np.random.RandomState(0).randn(len(H), 128)
    maer, accr = ridge_probe(Xr, Y)
    print(f"  [随机特征] MAE={maer:.4f} acc={accr:.3f} (基线含先验)")


if __name__ == "__main__":
    main()
