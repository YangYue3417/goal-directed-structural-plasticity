"""walker_mem.py — Walker 记忆世界模型 (LIF + 生长→试探→连接)。

新机制迁移:
  ① 记忆池 (MemPool LIF): 神经元短期记忆 → 相位/节奏检测 (走路步态)
  ② baby 生长 (生长→试探→连接): 弱初始化 + 试探期 + 巩固/剪枝
  ③ Δ残差预测: s' = s + Δ (连续动力学)
  ④ 并行自举 + SR 安全访问价值 + 做梦 (生存优先回放)

MPC: 历史片段滚动电位 → 采样动作 → 预测 (共享历史, 一次前向)
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

from mempool import MemPool
from walker_full import ValueNet, train_v_surv


class MemWM(nn.Module):
    """Δ + 记忆池世界模型。输入序列 (B,T,obs+act) → 每步预测 Δ, r。"""
    def __init__(self, obs_dim=24, act_dim=4, d=64, pool=512, top_k=64,
                 hidden=128, tau_min=2.0, tau_max=48.0):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.embed = nn.Linear(obs_dim + act_dim, d)
        self.pool = MemPool(d, pool, top_k, tau_min, tau_max)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.head_d = nn.Linear(hidden, obs_dim)
        self.head_r = nn.Linear(hidden, 1)

    def forward(self, obs_seq, act_seq):
        """(B,T,D) 序列 → 每步预测 Δ, r。"""
        B, T = obs_seq.shape[:2]
        sa = torch.cat([obs_seq, act_seq], -1)
        z = torch.tanh(self.embed(sa))
        z_pool, sel = self.pool(z)
        h = self.net(z_pool)
        delta = self.head_d(h)
        s_pred = obs_seq + delta
        r_pred = self.head_r(h).squeeze(-1)
        return s_pred, r_pred, sel

    def rollout(self, hist_obs, hist_act, obs, act_k):
        """MPC: 历史滚动电位 + K 候选动作 → 预测。

        hist: (L, obs)/(L, act) 最近历史; obs: (obs,); act_k: (K, act)
        序列 = [hist 广播 ×K, (obs, act_k)] → 最后一步预测 (K, obs)
        """
        K = len(act_k)
        dev = self.embed.weight.device
        h_o = torch.from_numpy(hist_obs).float().to(dev).repeat(K, 1, 1)
        h_a = torch.from_numpy(hist_act).float().to(dev).repeat(K, 1, 1)
        o_t = torch.from_numpy(obs).float().to(dev).repeat(K, 1)
        a_t = torch.from_numpy(act_k).float().to(dev)
        seq_o = torch.cat([h_o, o_t.unsqueeze(1)], 1)   # (K, L+1, obs)
        seq_a = torch.cat([h_a, a_t.unsqueeze(1)], 1)
        with torch.no_grad():
            sp, rp, _ = self(seq_o, seq_a)
        return sp[:, -1].cpu().numpy()


def collect_parallel(model, V, n_envs=8, n_steps=30000, eps=0.05, L=12,
                     device="cuda"):
    """并行持续生存收集。维护每 env 历史 ring buffer (L 步)。"""
    import gymnasium as gym
    envs = [gym.make('BipedalWalker-v3') for _ in range(n_envs)]
    obs = [e.reset()[0] for e in envs]
    hist = [([], []) for _ in range(n_envs)]
    eps_list = [[] for _ in range(n_envs)]
    episodes, total = [], 0
    rng = np.random.RandomState(0)
    while total < n_steps:
        acts = np.zeros((n_envs, 4), np.float32)
        for i in range(n_envs):
            if rng.rand() < eps:
                acts[i] = rng.uniform(-1, 1, 4)
            else:
                ho = np.array(hist[i][0][-L:], np.float32) if hist[i][0] else \
                    np.zeros((1, 24), np.float32)
                ha = np.array(hist[i][1][-L:], np.float32) if hist[i][1] else \
                    np.zeros((1, 4), np.float32)
                cand = rng.uniform(-1, 1, (200, 4)).astype(np.float32)
                sp = model.rollout(ho, ha, obs[i], cand)
                sp_t = torch.from_numpy(sp).float().to(device)
                score = 0.95 * V(sp_t)
                acts[i] = cand[int(score.argmax().item())]
        for i in range(n_envs):
            o2, r, d, _, _ = envs[i].step(acts[i])
            eps_list[i].append((obs[i], acts[i], r, o2))
            hist[i][0].append(obs[i]); hist[i][1].append(acts[i])
            obs[i] = o2
            if d or len(eps_list[i]) >= 1600:
                episodes.append(eps_list[i])
                eps_list[i] = []
                hist[i] = ([], [])
                obs[i] = envs[i].reset()[0]
        total = sum(len(e) for e in episodes)
    for e in envs: e.close()
    S = np.array([t[0] for ep in episodes for t in ep], np.float32)
    A = np.array([t[1] for ep in episodes for t in ep], np.float32)
    R = np.array([t[2] for ep in episodes for t in ep], np.float32)
    Sn = np.array([t[3] for ep in episodes for t in ep], np.float32)
    return (S, A, R, Sn), episodes


def seg_batch(episodes, L=16, n=256, device="cuda", seed=0):
    """从 episodes 抽片段 (含随机起始), 拼 batch。"""
    rng = np.random.RandomState(seed)
    valid = [e for e in episodes if len(e) >= 5]
    if not valid:
        return None
    S, A, R, Sn = [], [], [], []
    for _ in range(n):
        ep = valid[rng.randint(len(valid))]
        if len(ep) <= L:
            seg = ep
        else:
            s0 = rng.randint(0, len(ep) - L)
            seg = ep[s0:s0 + L]
        S.append([t[0] for t in seg]); A.append([t[1] for t in seg])
        R.append([t[2] for t in seg]); Sn.append([t[3] for t in seg])
    return (torch.from_numpy(np.array(S, np.float32)).to(device),
            torch.from_numpy(np.array(A, np.float32)).to(device),
            torch.from_numpy(np.array(R, np.float32)).to(device),
            torch.from_numpy(np.array(Sn, np.float32)).to(device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--steps_per_round", type=int, default=30000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--L", type=int, default=12, help="历史/片段长度")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)

    print(f"=== Walker 记忆世界模型 (LIF+baby生长) n_envs={args.n_envs} ===")
    model = MemWM(24, 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 轮 0: 随机并行
    (S, A, R, Sn), episodes = collect_parallel(
        model, torch.zeros(1, 24).to(device), args.n_envs,
        args.steps_per_round, eps=1.0, L=args.L, device=device)
    V = train_v_surv(episodes, device=device)
    data = [S, A, R, Sn]
    print(f"轮 0: 随机 {len(S)} 步", flush=True)

    for rnd in range(1, args.rounds + 1):
        (S, A, R, Sn), episodes = collect_parallel(
            model, V, args.n_envs, args.steps_per_round, eps=0.05,
            L=args.L, device=device)
        avg_len = np.mean([len(e) for e in episodes])
        data = [np.concatenate([d, x]) for d, x in zip(data, [S, A, R, Sn])]
        # 训练 (片段序列)
        for ep_i in range(args.epochs):
            b = seg_batch(episodes + [list(zip(*data))], L=args.L,
                          device=device)
            if b is None:
                break
            S_b, A_b, R_b, Sn_b = b
            sp, rp, sel = model(S_b, A_b)
            loss = F.mse_loss(sp, Sn_b) + 0.5 * F.mse_loss(rp, R_b)
            opt.zero_grad(); loss.backward(); opt.step()
            if ep_i % 20 == 19 and len(model.pool.growth_log) < 48:
                per_err = (sp - Sn_b).pow(2).mean(-1).mean(0)  # (L,)
                t_hard = int(per_err.argmax())
                model.pool.grow(sel[:, t_hard])
            model.pool.settle_babies(age_thresh=3000.0, rate_thresh=0.005)
        n_prune = model.pool.prune(0.004)[0]
        # 做梦: 生存优先片段
        dream(model, episodes, device=device)
        V = train_v_surv(episodes, device=device)
        print(f"轮 {rnd}: 存活 {avg_len:.0f} 步 (生长 {len(model.pool.growth_log)}, "
              f"baby {int(model.pool.baby_mask.sum().item())}, 淘汰 {n_prune})",
              flush=True)

    torch.save({"model": model.state_dict(), "v": V.state_dict()},
               "runs/walker_mem.pt")
    print("保存: runs/walker_mem.pt")


def dream(model, episodes, n_passes=3, lr=1e-4, L=20, device="cuda"):
    """做梦: 生存优先片段回放 (好轨迹多回放)。"""
    lens = [len(e) for e in episodes]
    order = np.argsort(lens)[::-1]
    good = [episodes[i] for i in order[:max(1, len(episodes)//3)]]
    if not good:
        return
    rng = np.random.RandomState(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_passes):
        b = seg_batch(good, L=L, n=64, device=device, seed=rng.randint(10**6))
        if b is None:
            continue
        S_b, A_b, R_b, Sn_b = b
        if rng.rand() < 0.5:
            S_b, A_b, R_b, Sn_b = (S_b.flip(1), A_b.flip(1),
                                   R_b.flip(1), Sn_b.flip(1))
        sp, rp, _ = model(S_b, A_b)
        loss = F.mse_loss(sp, Sn_b) + 0.5 * F.mse_loss(rp, R_b)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


if __name__ == "__main__":
    main()
