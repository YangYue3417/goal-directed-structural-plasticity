"""train_dream.py v2 — 睡眠结构化重放 (高效版)。

生物: sharp-wave ripple 按序重放轨迹 → 信用沿链完全分配。
确定性环境 → 轨迹的精确折扣回报 = 反向 DP backup (O(T) 无 autograd),
一次传播全链 (等价 TD(1)/MC)。随机采样 TD bootstrap 每 pass 只回传 1 步。

昼夜:
  白天: 随机探索收集轨迹 (含死亡)
  睡眠: 重放 (MC-return 回归) vs 随机 TD — 比传播速度
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from envs.energy_maze import EnergyMaze
from world_models.train_wm_energy import ValueNet, eval_planner
from world_models.train_world_model import WorldModel, ACT_SIZE


def collect_day(env, n_episodes=2000, seed=42):
    rng = np.random.RandomState(seed)
    traj_ids = []
    buffer = []
    for _ in range(n_episodes):
        obs = env.reset()
        start = len(buffer)
        for _ in range(200):
            a = int(rng.randint(ACT_SIZE))
            obs_next, r, done = env.step(a)
            buffer.append((obs, r, obs_next, done))
            obs = obs_next
            if done:
                break
        traj_ids.append((start, len(buffer)))
    return buffer, traj_ids


def dream_replay(V, obs_all, rew_all, done_all, traj_ids, n_nights, gamma=0.95,
                 lr=1e-3, device="cuda", epochs=3):
    """结构化睡眠: 反向算折扣回报 + 显著性加权回归 (死亡区 ×15)。"""
    opt = torch.optim.AdamW(V.parameters(), lr=lr)
    all_t, all_y = [], []
    with torch.no_grad():
        for s, e in traj_ids:
            G = 0.0
            for t in range(e - 1, s - 1, -1):
                G = rew_all[t].item() + gamma * G * (1 - done_all[t].item())
                all_t.append(obs_all[t])
                all_y.append(G)
    all_t = torch.stack(all_t)
    all_y = torch.from_numpy(np.array(all_y, dtype=np.float32)).to(device)
    # 显著性权重: 死亡附近 (|G|>3) ×15 (生物: 显著事件优先固化)
    w = torch.ones_like(all_y)
    sal = all_y.abs() > 3
    w[sal] = 15.0
    # 夜间训练集: 显著子集全部 + 普通随机子集 (每夜 5 epoch)
    sal_idx = sal.nonzero().flatten()
    for _ in range(n_nights * 5):
        idx_s = sal_idx[torch.randperm(len(sal_idx))[:2048]]
        idx_r = torch.randint(0, len(all_t), (2048,), device=device)
        idx = torch.cat([idx_s, idx_r])
        loss = F.mse_loss(V(all_t[idx]), all_y[idx], reduction="none") * w[idx]
        opt.zero_grad(); loss.mean().backward(); opt.step()


def random_td(V, obs_all, rew_all, nxt_all, done_all, n_updates, gamma=0.95,
              lr=1e-3, device="cuda"):
    """随机重放: 死亡 transition 显著性加权 ×15 (bootstrap)。"""
    opt = torch.optim.AdamW(V.parameters(), lr=lr)
    n = len(obs_all)
    w_all = torch.ones(n, device=device)
    w_all[done_all.bool()] = 15.0
    for _ in range(n_updates):
        idx = torch.randint(0, n, (512,), device=device)
        with torch.no_grad():
            target = rew_all[idx] + gamma * (1 - done_all[idx]) * V(nxt_all[idx])
        loss = F.mse_loss(V(obs_all[idx]), target, reduction="none") * w_all[idx]
        opt.zero_grad(); loss.mean().backward(); opt.step()


def death_metric(V, env, device):
    diffs = []
    for x, y in [(2, 2), (5, 5), (10, 10)]:
        vals = {}
        for ef in [0.01, 0.5]:
            env.x, env.y, env.dir = x, y, 1
            env.energy = ef * env.E0
            env.food_eaten = [False] * 3
            env._visited = {(x, y)}
            env.steps = 0
            obs = env.observe()
            with torch.no_grad():
                vals[ef] = V(torch.from_numpy(obs[None]).float().to(device)).item()
        diffs.append(vals[0.01] - vals[0.5])
    return np.mean(diffs)


def main():
    device = torch.device("cuda")
    env = EnergyMaze(size=20, n_foods=3, seed=42, step_cost=0.5)
    wm = WorldModel(obs_size=10, T=4).to(device)
    wm.load_state_dict(torch.load("runs/wm_energy_nodth.pt",
                                  map_location="cpu", weights_only=False)["model"])
    wm.eval()

    print("=== 白天 ===", flush=True)
    buffer, traj_ids = collect_day(env, n_episodes=2000)
    n_die = sum(1 for b in buffer if b[3])
    print(f"  {len(buffer)} transitions, {len(traj_ids)} 轨迹, 死亡 {n_die}", flush=True)
    obs_all = torch.from_numpy(np.stack([b[0] for b in buffer])).float().to(device)
    rew_all = torch.from_numpy(np.array([b[1] for b in buffer])).float().to(device)
    nxt_all = torch.from_numpy(np.stack([b[2] for b in buffer])).float().to(device)
    done_all = torch.from_numpy(np.array([b[3] for b in buffer], dtype=np.float32)).to(device)

    print("=== 睡眠 × 2 ===", flush=True)
    for mode in ["dream", "random"]:
        V = ValueNet().to(device)
        curve = []
        if mode == "dream":
            for k in range(1, 4):
                dream_replay(V, obs_all, rew_all, done_all, traj_ids, 1, device=device)
                curve.append(round(death_metric(V, env, device), 3))
        else:
            n_t = sum(e - s for s, e in traj_ids)
            for k in range(1, 4):
                random_td(V, obs_all, rew_all, nxt_all, done_all, n_t // 512,
                          device=device)
                curve.append(round(death_metric(V, env, device), 3))
        V.eval()
        f, d = eval_planner(wm, V, env, 30, 3, device, False)
        ok = "✓" if curve[-1] < -3 else "✗"
        print(f"[{mode}] 每夜后 V(2步死)-V(安全): {curve} → {ok} "
              f"| 食物率={f} 饿死率={d}", flush=True)


if __name__ == "__main__":
    main()
