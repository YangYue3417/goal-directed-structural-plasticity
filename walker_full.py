"""walker_sr.py — Successor Measure 版 Walker 自举循环。

核心升级 (SR 思想):
  之前: V = E[回报] (MC, 依赖奖励+策略分布 → Walker 轮3-7崩溃)
  现在: V_surv = E[Σ γ^t · 1(safe)] (安全状态访问, TD 自监督, 无奖励!)

  生存目标 = 最大化未来安全状态访问 (successor measure)
  → 不需要奖励函数, 从自举状态转移直接 TD 学
  → 不受奖励分布漂移影响 (修复崩溃根因)

结构 (同迷宫 30 天): 持续生存 + 连续经验 + 迭代自举
组件: 世界模型 + 难样本生长 + V_surv(TD) + MPC(未来安全评分)
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
from train_gdsp import ValueNet, grow_hard, prune, mpc_act
from units.sparse_unit import SparseUnit


class DeltaWM(torch.nn.Module):
    """Δ 残差预测世界模型: 预测状态变化 (s'-s) 而非绝对 s'。

    连续高速动力学 (Walker) 中, 预测偏移比绝对位置容易得多。"""
    def __init__(self, obs_dim=24, act_dim=4, d=64, pool=512, top_k=64,
                 hidden=128, active_ratio=0.25):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.embed = torch.nn.Linear(obs_dim + act_dim, d)
        self.unit = SparseUnit(d_model=d, d_pool=pool, top_k=top_k)
        self.unit.register_buffer("active_mask",
                                  torch.zeros(pool, dtype=torch.bool))
        self.unit.active_mask[:int(pool * active_ratio)] = True
        self.unit.register_buffer("act_rate", torch.zeros(pool))
        self.unit.register_buffer("_load_ema", torch.zeros(pool))
        self.growth_log = []
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU())
        self.head_d = torch.nn.Linear(hidden, obs_dim)  # Δ 预测
        self.head_r = torch.nn.Linear(hidden, 1)

    def forward(self, obs, act):
        B = obs.shape[0]
        sa = torch.cat([obs, act], -1)
        z = torch.tanh(self.embed(sa))
        z_pool, ps = self.unit(z.unsqueeze(1))
        h = self.net(z_pool.squeeze(1))
        delta = self.head_d(h)
        s_pred = obs + delta  # s' = s + Δ
        r_pred = self.head_r(h).squeeze(-1)
        with torch.no_grad():
            onehot = torch.zeros(self.unit.d_pool, device=obs.device)
            onehot[ps.selected[0, 0]] = 1.0
            self.unit.act_rate = 0.999 * self.unit.act_rate + 0.001 * onehot
            self.unit._load_ema = 0.99 * self.unit._load_ema + 0.01 * ps.load.mean(0)
        return s_pred, r_pred, ps.selected


def is_safe(obs):
    """安全: 躯干直立 (hull angle 小) + 未跌倒。"""
    return (abs(obs[0]) < 0.5).astype(np.float32)


def random_collect(env, n_steps=20000, seed=42):
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


def delta_mpc_act(model, V, obs, n_samples=200, device="cuda"):
    """ΔWM 采样 MPC: 评分 = V(预测安全状态)。"""
    rng = np.random.RandomState()
    acts = rng.uniform(-1, 1, (n_samples, 4)).astype(np.float32)
    obs_t = torch.from_numpy(np.tile(obs, (n_samples, 1))).float().to(device)
    act_t = torch.from_numpy(acts).float().to(device)
    with torch.no_grad():
        sp, rp, _ = model(obs_t, act_t)
        score = 0.95 * V(sp)
    return acts[int(score.argmax().item())]


def survive_collect(model, V, env, n_steps=20000, device="cuda", eps=0.1):
    """MPC 控制持续生存, 跌倒重生。返回 episodes。"""
    rng = np.random.RandomState(0)
    episodes = []
    obs, _ = env.reset()
    ep = []
    max_ep = 1600  # Walker 环境截断
    while True:
        if rng.rand() < eps:
            a = rng.uniform(-1, 1, 4).astype(np.float32)
        else:
            a = delta_mpc_act(model, V, obs, device=device)
        o2, r, d, _, _ = env.step(a)
        ep.append((obs, a, r, o2))
        obs = o2
        if d or len(ep) >= max_ep:
            episodes.append(ep)
            ep = []
            obs, _ = env.reset()
        if sum(len(e) for e in episodes) >= n_steps:
            break
    S = np.array([t[0] for e in episodes for t in e], np.float32)
    A = np.array([t[1] for e in episodes for t in e], np.float32)
    R = np.array([t[2] for e in episodes for t in e], np.float32)
    Sn = np.array([t[3] for e in episodes for t in e], np.float32)
    return (S, A, R, Sn), episodes


def train_v_surv(episodes, gamma=0.98, epochs=100, device="cuda"):
    """V_surv: 安全状态访问价值 (TD 自监督, 无奖励)。

    V(s) = 1(safe) + γ·V(s')  ← TD, 从状态转移学, 不需要奖励
    """
    X, Y = [], []
    for ep in episodes:
        for i in range(len(ep)):
            s = ep[i][0]
            s_next = ep[i][3] if i + 1 < len(ep) else None
            target = is_safe(s) + gamma * (is_safe(s_next) if s_next is not None else 0.0)
            X.append(s)
            Y.append(target)
    Xt = torch.from_numpy(np.array(X, np.float32)).to(device)
    Yt = torch.from_numpy(np.array(Y, np.float32)).to(device)
    V = ValueNet(24).to(device)
    opt = torch.optim.AdamW(V.parameters(), lr=1e-3)
    for ep in range(epochs):
        idx = torch.randperm(len(Xt))[:8192]
        loss = F.mse_loss(V(Xt[idx]), Yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return V


def dream(model, episodes, n_passes=3, lr=1e-4, L=20, device="cuda"):
    """做梦: 生存优先片段回放 (M7 方法迁移)。

    重放"活得好"的轨迹 (按存活长度优先) 的随机片段 × 随机方向
    → 世界模型强化生存相关结构 → 难样本变真 → 生长学结构
    → 坏神经元 (对好轨迹不激活) 激活率低 → 被淘汰回收
    """
    # 按存活长度排序, 取 top 长轨迹
    lens = [len(e) for e in episodes]
    order = np.argsort(lens)[::-1]
    good = [episodes[i] for i in order[:max(1, len(episodes)//3)]]
    if not good:
        return
    rng = np.random.RandomState(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_passes):
        for _ in range(20):
            ep = good[rng.randint(len(good))]
            if len(ep) <= L:
                seg = ep
            else:
                start = rng.randint(0, len(ep) - L)
                seg = ep[start:start + L]
            if rng.rand() < 0.5:
                seg = seg[::-1]  # 反向片段
            if len(seg) < 5:
                continue
            S = np.array([t[0] for t in seg], np.float32)
            A = np.array([t[1] for t in seg], np.float32)
            R = np.array([t[2] for t in seg], np.float32)
            Sn = np.array([t[3] for t in seg], np.float32)
            sp, rp, _ = model(torch.from_numpy(S).to(device),
                                     torch.from_numpy(A).to(device))
            loss = F.mse_loss(sp, torch.from_numpy(Sn).to(device))                    + 0.5 * F.mse_loss(rp, torch.from_numpy(R).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--steps_per_round", type=int, default=15000)
    p.add_argument("--train_epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    import gymnasium as gym
    env = gym.make('BipedalWalker-v3')
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

    print("=== 轮 0: 随机初始 ===", flush=True)
    S0, A0, R0, Sn0 = random_collect(env, args.steps_per_round)
    train_wm(S0, A0, R0, Sn0, args.train_epochs)
    V = train_v_surv([list(zip(S0, A0, R0, Sn0))], device=device)
    buffer = [S0, A0, R0, Sn0]

    print("=== SR 自举循环 (V=安全访问, 无奖励) ===", flush=True)
    for rnd in range(1, args.rounds + 1):
        (S, A, R, Sn), episodes = survive_collect(model, V, env,
                                                   args.steps_per_round, device)
        avg_len = np.mean([len(e) for e in episodes])
        # 混合 buffer (旧+新, 防遗忘 + 数据累积)
        buffer = [np.concatenate([b, x]) for b, x in zip(buffer, [S, A, R, Sn])]
        # 白天: 世界模型训练 (混合 buffer + 难样本生长)
        train_wm(*[x[-60000:] for x in buffer], args.train_epochs)
        # 夜晚: 做梦 (生存优先片段回放) — 强化好结构, 淘汰坏神经元
        n_prune = prune(model, 0.005)
        dream(model, episodes, device=device)
        # 生长: 难样本定向 (在训练中已发生, 这里补充)
        V = train_v_surv(episodes, device=device)          # V_surv (TD 自监督)
        print(f"  轮 {rnd}: 平均存活 {avg_len:.0f} 步 (生长 {len(model.growth_log)}, "
              f"淘汰 {n_prune})", flush=True)

    env.close()
    print(f"最终: 生长神经元 {len(model.growth_log)}")
    torch.save({"model": model.state_dict(), "v": V.state_dict()},
               "runs/walker_full.pt")


if __name__ == "__main__":
    main()
