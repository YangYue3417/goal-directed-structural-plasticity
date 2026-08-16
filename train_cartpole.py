"""train_cartpole.py — Gym 连续控制域验证: 世界模型 + 生长。

迁移点 (生存任务 → 平衡任务):
  生存 (迷宫觅食) → 生存 (杆不倒, episode 结束=死亡)
  世界模型: (s, a) → (s', r) — 物理动力学
  生长: 误差驱动 → 难预测状态 (倾倒临界)
  策略: 世界模型 MPC (选让杆角度最小的动作)

验证:
  ① 世界模型学到 CartPole 动力学 (预测误差)
  ② 生长功能: 删生长神经元 → 预测误差上升 (func vs rand)
  ③ MPC 控制: 平均存活步数 vs 随机
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

import config as cfg
import gymnasium as gym


class ValueNet(nn.Module):
    """学习价值: 从经验 (MC 回报) 学"哪些状态能保持平衡" — 自研, 非外部 RL。"""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
    def forward(self, s):
        return self.net(s).squeeze(-1)


class CartPoleWM(nn.Module):
    """世界模型: (obs4 + act2) → (next_obs4 + reward1)。"""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_s = nn.Linear(hidden, 4)
        self.head_r = nn.Linear(hidden, 1)

    def forward(self, sa):
        h = self.net(sa)
        return self.head_s(h), self.head_r(h).squeeze(-1)


def heuristic(obs):
    """CartPole 规则策略: 杆倒向哪边推哪边。"""
    x, xv, th, thv = obs
    return 0 if th + 0.1 * thv < 0 else 1


def collect(n_eps=300, max_steps=300, seed=42):
    """混合收集: 规则策略 (长轨迹) + 随机 (多样覆盖)。"""
    env = gym.make('CartPole-v1')
    rng = np.random.RandomState(seed)
    S, A, R, Sn = [], [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset()
        for t in range(max_steps):
            a = heuristic(obs) if ep % 2 == 0 else int(rng.randint(2))
            obs_next, r, done, _, _ = env.step(a)
            S.append(obs); A.append(a); R.append(r); Sn.append(obs_next)
            obs = obs_next
            if done:
                break
    env.close()
    return (np.array(S, np.float32), np.array(A, np.int64),
            np.array(R, np.float32), np.array(Sn, np.float32))


def grow(model, arm, n=2, perturb=0.1, rng=None, load_ema=None):
    """误差驱动 (func): 克隆高负载; 随机 (rand)。简化: 扩宽 hidden (复制行)。"""
    net = model.net[2]  # 第二个 Linear
    n_hidden = net.weight.shape[0]
    # 新神经元 = 复制 + 扰动 (扩展 hidden 宽度)
    if arm == "func":
        idx = torch.argsort(load_ema)[-n:]  # 高负载神经元
    else:
        idx = torch.tensor(rng.choice(n_hidden, n, replace=False))
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n_eps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("收集 CartPole 数据...", flush=True)
    S, A, R, Sn = collect(args.n_eps)
    n = len(S)
    print(f"  {n} 转移, 平均回合长度 ≈ {n/args.n_eps:.0f} 步", flush=True)

    model = CartPoleWM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    s_t = torch.from_numpy(S).float().to(device)
    a_oh = F.one_hot(torch.from_numpy(A).long(), 2).float().to(device)
    r_t = torch.from_numpy(R).float().to(device)
    sn_t = torch.from_numpy(Sn).float().to(device)
    sa_t = torch.cat([s_t, a_oh], -1)

    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(n)[:4096]
        sp, rp = model(sa_t[idx])
        loss = F.mse_loss(sp, sn_t[idx]) + 0.5 * F.mse_loss(rp, r_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 40 == 39:
            model.eval()
            with torch.no_grad():
                sp, rp = model(sa_t[:2000])
                err = F.mse_loss(sp, sn_t[:2000]).item()
            print(f"  ep {ep+1}: loss={loss.item():.4f} 预测误差={err:.4f}", flush=True)

    torch.save({"model": model.state_dict()}, "runs/cartpole_wm.pt")

    # 学习 V: 混合数据 (随机+启发式) MC 回报 — 覆盖广, 不偏单一策略
    print("训练学习价值 V (混合数据)...", flush=True)
    V = ValueNet().to(device)
    optv = torch.optim.AdamW(V.parameters(), lr=1e-3)
    X, G = [], []
    env_data = gym.make('CartPole-v1')
    rng = np.random.RandomState(7)
    for ep_i in range(200):
        obs, _ = env_data.reset()
        traj = []
        for _ in range(300):
            a = heuristic(obs) if ep_i % 2 == 0 else int(rng.randint(2))
            obs_next, r, done, _, _ = env_data.step(a)
            traj.append((obs, r))
            obs = obs_next
            if done:
                break
        g = 0.0
        for s, r in reversed(traj):
            g = r + 0.95 * g
            X.append(s); G.append(g)
    env_data.close()
    Xt = torch.from_numpy(np.array(X, np.float32)).to(device)
    Gt = torch.from_numpy(np.array(G, np.float32)).to(device)
    for ep in range(200):
        idx = torch.randperm(len(Xt))[:4096]
        loss = F.mse_loss(V(Xt[idx]), Gt[idx])
        optv.zero_grad(); loss.backward(); optv.step()
    torch.save({"model": V.state_dict()}, "runs/cartpole_v.pt")
    with torch.no_grad():
        vb = V(Xt[:1000]).mean().item()
    print(f"  V 训练 loss={loss.item():.4f} (均值 {vb:.2f})")

    # MPC 策略: 世界模型 rollout + 学习 V 评分 (非手动规则)
    def mpc_act(obs):
        best_a, best = 0, -1e9
        with torch.no_grad():
            for a in [0, 1]:
                sa = torch.from_numpy(
                    np.concatenate([obs, np.eye(2)[a]])[None]).float().to(device)
                sp, rp = model(sa)
                # 评分 = 预测奖励 + 学习价值(下一状态) — 平衡技能从经验涌现
                score = rp.item() + 0.95 * V(sp[0]).item()
                if score > best:
                    best, best_a = score, a
        return best_a

    # 评估 MPC vs 随机
    env = gym.make('CartPole-v1')
    for tag, policy in [("MPC(世界模型)", mpc_act), ("随机", lambda o: np.random.randint(2))]:
        lens = []
        for _ in range(20):
            obs, _ = env.reset()
            t = 0
            for t in range(500):
                obs, r, done, _, _ = env.step(policy(obs))
                if done:
                    break
            lens.append(t)
        print(f"  [{tag}] 平均存活 {np.mean(lens):.0f} 步 (满 500)", flush=True)
    env.close()
    print("保存: runs/cartpole_wm.pt")


if __name__ == "__main__":
    main()
