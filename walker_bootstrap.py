"""walker_bootstrap.py — Walker 生存自举循环 (用户洞察落地)。

结构 (与迷宫 30 天一致):
  持续生存 (跌倒=死亡, 重生继续) + 连续经验 + 迭代自举

循环:
  轮 0: 随机收集初始数据 → 世界模型 + V
  轮 1..N:
    1. 当前模型 + V + 采样MPC 控制 agent 持续生存 (跌倒重生)
    2. 收集经验 (含跌倒前临界状态)
    3. 增量训练世界模型 + 难样本生长 (临界状态定向)
    4. V 从累积经验更新
  指标: 每轮平均存活步数 → 应增长; 生长神经元专精临界状态

纯框架: 世界模型 + 生长 + 淘汰 + 学习V + MPC, 无外部 RL。
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
from train_gdsp import (GDSPModel, ValueNet, grow_hard, prune,
                        train_value, mpc_act)


def random_collect(env, n_steps=20000, seed=42):
    """轮 0: 随机策略收集。"""
    rng = np.random.RandomState(seed)
    S, A, R, Sn = [], [], [], []
    obs, _ = env.reset()
    while len(S) < n_steps:
        a = rng.uniform(-1, 1, 4).astype(np.float32)
        o2, r, d, _, _ = env.step(a)
        S.append(obs); A.append(a); R.append(r); Sn.append(o2)
        obs = o2
        if d:
            obs, _ = env.reset()
    return (np.array(S, np.float32), np.array(A, np.float32),
            np.array(R, np.float32), np.array(Sn, np.float32))


def survive_collect(model, V, env, n_steps=20000, device="cuda"):
    """持续生存收集: MPC 控制, 跌倒=死亡, 重生继续。返回 episodes。"""
    episodes = []
    lens = []
    obs, _ = env.reset()
    ep = []
    while True:
        a = mpc_act(model, V, obs, 24, 4, False, device=device)
        o2, r, d, _, _ = env.step(a)
        ep.append((obs, a, r, o2))
        obs = o2
        if d:
            episodes.append(ep)
            lens.append(len(ep))
            ep = []
            obs, _ = env.reset()
        if sum(len(e) for e in episodes) >= n_steps:
            break
    # 扁平化
    S = np.array([t[0] for e in episodes for t in e], np.float32)
    A = np.array([t[1] for e in episodes for t in e], np.float32)
    R = np.array([t[2] for e in episodes for t in e], np.float32)
    Sn = np.array([t[3] for e in episodes for t in e], np.float32)
    return (S, A, R, Sn), episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--steps_per_round", type=int, default=20000)
    p.add_argument("--train_epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    import gymnasium as gym
    env = gym.make('BipedalWalker-v3')

    model = GDSPModel(24, 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def train_wm(S, A, R, Sn, epochs):
        s_t = torch.from_numpy(S).float().to(device)
        a_t = torch.from_numpy(A).float().to(device)
        r_t = torch.from_numpy(R).float().to(device)
        sn_t = torch.from_numpy(Sn).float().to(device)
        for ep in range(epochs):
            model.train()
            idx = torch.randperm(len(S))[:8192]
            sp, rp, zp, _, sel = model(s_t[idx], a_t[idx])
            loss = F.mse_loss(sp, sn_t[idx]) + 0.5 * F.mse_loss(rp, r_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 20 == 19:
                # 难样本定向生长
                if len(model.growth_log) < 60:
                    per_err = (sp - sn_t[idx]).pow(2).mean(-1)
                    hard = int(per_err.argmax().item())
                    grow_hard(model, sel[hard])

    # 轮 0: 随机初始
    print("=== 轮 0: 随机收集 + 初始训练 ===", flush=True)
    S0, A0, R0, Sn0 = random_collect(env, args.steps_per_round)
    train_wm(S0, A0, R0, Sn0, args.train_epochs)
    V = train_value(lambda: gym.make('BipedalWalker-v3'), 24, 4, False,
                    device=device)
    print(f"  初始世界模型就绪", flush=True)

    def v_from_episodes(episodes, gamma=0.95):
        """从自举经验 (episodes) 学 MC 回报价值。"""
        X, G = [], []
        for ep in episodes:
            g = 0.0
            for s, a, r, sn in reversed(ep):
                g = r + gamma * g
                X.append(s); G.append(g)
        V = ValueNet(24).to(device)
        optv = torch.optim.AdamW(V.parameters(), lr=1e-3)
        Xt = torch.from_numpy(np.array(X, np.float32)).to(device)
        Gt = torch.from_numpy(np.array(G, np.float32)).to(device)
        for ep in range(150):
            idx = torch.randperm(len(Xt))[:4096]
            loss = F.mse_loss(V(Xt[idx]), Gt[idx])
            optv.zero_grad(); loss.backward(); optv.step()
        return V

    # 自举循环
    buffer = [S0, A0, R0, Sn0]  # 累积经验防遗忘
    print("=== 生存自举循环 ===", flush=True)
    for rnd in range(1, args.rounds + 1):
        # 1. 持续生存收集 (MPC 控制) → episodes
        (S, A, R, Sn), episodes = survive_collect(model, V, env,
                                                   args.steps_per_round, device)
        avg_len = np.mean([len(e) for e in episodes])
        # 2. 混合 buffer 训练 (旧+新, 防遗忘)
        buffer = [np.concatenate([b, x]) for b, x in zip(buffer, [S, A, R, Sn])]
        S_all, A_all, R_all, Sn_all = buffer
        train_wm(S_all[-60000:], A_all[-60000:], R_all[-60000:], Sn_all[-60000:],
                 args.train_epochs)
        # 3. V 从自举经验更新
        V = v_from_episodes(episodes)
        # 4. 评估
        print(f"  轮 {rnd}: 平均存活 {avg_len:.0f} 步 "
              f"(生长 {len(model.growth_log)})", flush=True)

    # 最终评估
    print("=== 最终评估 ===", flush=True)
    scores = []
    for _ in range(10):
        obs, _ = env.reset()
        total, done = 0.0, False
        for _ in range(300):
            a = mpc_act(model, V, obs, 24, 4, False, device=device)
            obs, r, done, _, _ = env.step(a)
            total += r
            if done: break
        scores.append(total)
    env.close()
    print(f"最终平均得分 {np.mean(scores):.1f} (300=满) | "
          f"生长神经元 {len(model.growth_log)}")
    torch.save({"model": model.state_dict(), "v": V.state_dict()},
               "runs/walker_bootstrap.pt")


if __name__ == "__main__":
    main()
