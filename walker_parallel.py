"""walker_parallel.py — 并行环境 Walker 自举 (8 envs, 数据速率 ×8)。

完整框架: ΔWM世界模型 + 难样本生长 + 淘汰 + 做梦 + SR安全访问价值 + 批量MPC
并行: 8 envs 同时收集, 批量 MPC 评分 (N×K 动作一次前向)

目标: 数据量 ×8 → 更快覆盖走路状态 → 存活稳步增长
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from walker_full import DeltaWM, ValueNet, grow_hard, prune, train_v_surv


def random_collect_parallel(n_envs=8, n_steps=50000, seed=42):
    """轮 0: 随机策略并行收集。"""
    import gymnasium as gym
    envs = [gym.make('BipedalWalker-v3') for _ in range(n_envs)]
    rng = np.random.RandomState(seed)
    obs = [e.reset()[0] for e in envs]
    S, A, R, Sn = [], [], [], []
    while len(S) < n_steps:
        for i in range(n_envs):
            a = rng.uniform(-1, 1, 4).astype(np.float32)
            o2, r, d, _, _ = envs[i].step(a)
            S.append(obs[i]); A.append(a); R.append(r); Sn.append(o2)
            obs[i] = o2
            if d:
                obs[i] = envs[i].reset()[0]
    for e in envs: e.close()
    return (np.array(S, np.float32), np.array(A, np.float32),
            np.array(R, np.float32), np.array(Sn, np.float32))


def batch_mpc(model, V, obs_list, K=200, eps=0.05, device="cuda"):
    """批量 MPC: N envs × K 动作采样 → 一次前向 → 每 env 选最优。"""
    N = len(obs_list)
    rng = np.random.RandomState()
    acts = rng.uniform(-1, 1, (N, K, 4)).astype(np.float32)
    obs_t = torch.from_numpy(np.repeat(np.stack(obs_list), K, axis=0)).float().to(device)
    act_t = torch.from_numpy(acts.reshape(-1, 4)).float().to(device)
    with torch.no_grad():
        sp, rp, _ = model(obs_t, act_t)
        score = 0.95 * V(sp).view(N, K)
    best = score.argmax(1)
    chosen = acts[np.arange(N), best.cpu().numpy()]
    # ε 探索: 部分 env 随机动作 (覆盖)
    explore = rng.rand(N) < eps
    chosen[explore] = rng.uniform(-1, 1, (int(explore.sum()), 4))
    return chosen


def survive_collect_parallel(model, V, n_envs=8, n_steps=50000, eps=0.1,
                             device="cuda"):
    """并行持续生存收集, 返回 episodes。"""
    import gymnasium as gym
    envs = [gym.make('BipedalWalker-v3') for _ in range(n_envs)]
    obs = [e.reset()[0] for e in envs]
    eps_list = [[] for _ in range(n_envs)]   # 每 env 当前 episode
    episodes = []
    total = 0
    while total < n_steps:
        acts = batch_mpc(model, V, obs, K=200, eps=eps, device=device)
        for i in range(n_envs):
            o2, r, d, _, _ = envs[i].step(acts[i])
            eps_list[i].append((obs[i], acts[i], r, o2))
            obs[i] = o2
            if d or len(eps_list[i]) >= 1600:
                episodes.append(eps_list[i])
                eps_list[i] = []
                obs[i] = envs[i].reset()[0]
        total = sum(len(e) for e in episodes)
    for e in envs: e.close()
    S = np.array([t[0] for ep in episodes for t in ep], np.float32)
    A = np.array([t[1] for ep in episodes for t in ep], np.float32)
    R = np.array([t[2] for ep in episodes for t in ep], np.float32)
    Sn = np.array([t[3] for ep in episodes for t in ep], np.float32)
    return (S, A, R, Sn), episodes


def dream(model, episodes, n_passes=3, lr=1e-4, L=20, device="cuda"):
    """做梦: 生存优先片段回放。"""
    lens = [len(e) for e in episodes]
    order = np.argsort(lens)[::-1]
    good = [episodes[i] for i in order[:max(1, len(episodes)//3)]]
    if not good:
        return
    rng = np.random.RandomState(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_passes):
        for _ in range(30):
            ep = good[rng.randint(len(good))]
            seg = ep if len(ep) <= L else ep[rng.randint(0, len(ep)-L):][:L]
            if rng.rand() < 0.5:
                seg = seg[::-1]
            if len(seg) < 5:
                continue
            S = np.array([t[0] for t in seg], np.float32)
            A = np.array([t[1] for t in seg], np.float32)
            R = np.array([t[2] for t in seg], np.float32)
            Sn = np.array([t[3] for t in seg], np.float32)
            sp, rp, _ = model(torch.from_numpy(S).to(device),
                              torch.from_numpy(A).to(device))
            loss = F.mse_loss(sp, torch.from_numpy(Sn).to(device)) \
                   + 0.5 * F.mse_loss(rp, torch.from_numpy(R).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--steps_per_round", type=int, default=50000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--train_epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"=== Walker 并行自举 (n_envs={args.n_envs}) ===", flush=True)
    model = DeltaWM(24, 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def train_wm(S, A, R, Sn, epochs):
        s_t = torch.from_numpy(S).float().to(device)
        a_t = torch.from_numpy(A).float().to(device)
        r_t = torch.from_numpy(R).float().to(device)
        sn_t = torch.from_numpy(Sn).float().to(device)
        for ep in range(epochs):
            model.train()
            idx = torch.randperm(len(S))[:8192]
            sp, rp, sel = model(s_t[idx], a_t[idx])
            loss = F.mse_loss(sp, sn_t[idx]) + 0.5 * F.mse_loss(rp, r_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 20 == 19 and len(model.growth_log) < 60:
                per_err = (sp - sn_t[idx]).pow(2).mean(-1)
                grow_hard(model, sel[int(per_err.argmax().item())])

    # 轮 0: 随机并行收集
    print("轮 0: 随机并行收集...", flush=True)
    S0, A0, R0, Sn0 = random_collect_parallel(args.n_envs, args.steps_per_round)
    train_wm(S0, A0, R0, Sn0, args.train_epochs)
    V = train_v_surv([list(zip(S0, A0, R0, Sn0))], device=device)
    buffer = [S0, A0, R0, Sn0]
    print(f"  轮 0 完成 (随机 {len(S0)} 步)", flush=True)

    # 自举
    for rnd in range(1, args.rounds + 1):
        (S, A, R, Sn), episodes = survive_collect_parallel(
            model, V, args.n_envs, args.steps_per_round, device=device)
        avg_len = np.mean([len(e) for e in episodes])
        buffer = [np.concatenate([b, x]) for b, x in zip(buffer, [S, A, R, Sn])]
        train_wm(*[x[-60000:] for x in buffer], args.train_epochs)
        n_prune = prune(model, 0.005)
        dream(model, episodes, device=device)
        V = train_v_surv(episodes, device=device)
        print(f"  轮 {rnd}: 存活 {avg_len:.0f} 步 (生长 {len(model.growth_log)}, "
              f"淘汰 {n_prune})", flush=True)

    torch.save({"model": model.state_dict(), "v": V.state_dict()},
               "runs/walker_parallel.pt")
    print(f"保存: runs/walker_parallel.pt")


if __name__ == "__main__":
    main()
