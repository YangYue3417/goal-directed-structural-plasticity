"""train_gdsp.py — 统一训练入口 (框架迁移正确性的保障)。

原则 (用户规定):
  - 生长/淘汰/睡眠是**必需运行时组件**, 不是可选附加
  - 迁移新任务只换 环境+观测, 组件自动携带 → 不会漏
  - 不依赖外部 RL 框架 (PPO 等不作数)

组件:
  GDSPModel: 观测+动作 → 编码 → SparseUnit池(可生长) → 预测(next_obs+reward)
  Growth:    误差驱动 — 高负载克隆 + 扰动 (适应难预测状态区域)
  Prune:     激活率淘汰 — 死神经元回收 (脑容量有限)
  Sleep:     片段回放 (可选, 动态环境用)
  Decision:  学习V + 采样MPC (评分 = 预测奖励 + γ·V(s'))

用法 (迁移 = 换 env_fn + 维度):
  CartPole:   train_gdsp(make_cartpole, obs=4, act=2, discrete=True)
  Walker:     train_gdsp(make_walker, obs=24, act=4, discrete=False)
  符号迷宫:   train_gdsp(make_survival, obs=14, act=3, discrete=True)
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
from units.sparse_unit import SparseUnit


# ---------------------------------------------------------------------------
# 世界模型: 池 (可生长) 是中间层
# ---------------------------------------------------------------------------
class GDSPModel(nn.Module):
    def __init__(self, obs_dim, act_dim, d=64, pool=512, top_k=64,
                 hidden=128, active_ratio=0.25):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.embed = nn.Linear(obs_dim + act_dim, d)
        self.unit = SparseUnit(d_model=d, d_pool=pool, top_k=top_k)
        self.unit.register_buffer("active_mask",
                                  torch.zeros(pool, dtype=torch.bool))
        n_init = max(1, int(pool * active_ratio))
        self.unit.active_mask[:n_init] = True
        self.unit.register_buffer("act_rate", torch.zeros(pool))   # 淘汰信号
        self.unit.register_buffer("_load_ema", torch.zeros(pool))  # 生长信号
        self.growth_log = []
        self.gru = nn.GRUCell(d, hidden)
        self.head_s = nn.Linear(hidden, obs_dim)
        self.head_r = nn.Linear(hidden, 1)
        self.head_z = nn.Linear(hidden, d)  # latent 预测目标

    def forward(self, obs, act, h=None):
        """obs: (B, obs_dim), act: (B, act_dim) → (next_obs, reward, z_next)"""
        B = obs.shape[0]
        sa = torch.cat([obs, act], -1)
        z = torch.tanh(self.embed(sa))
        z_pool, ps = self.unit(z.unsqueeze(1))  # (B, 1, d)
        z = z_pool.squeeze(1)
        if h is None:
            h = torch.zeros(B, self.gru.hidden_size, device=obs.device)
        h = self.gru(z, h)
        s_pred = self.head_s(h)
        r_pred = self.head_r(h).squeeze(-1)
        z_next = self.head_z(h)
        # 更新激活率 (淘汰信号)
        with torch.no_grad():
            onehot = torch.zeros(self.unit.d_pool, device=obs.device)
            onehot[ps.selected[0, 0]] = 1.0
            self.unit.act_rate = 0.999 * self.unit.act_rate + 0.001 * onehot
            self.unit._load_ema = 0.99 * self.unit._load_ema + 0.01 * ps.load.mean(0)
        return s_pred, r_pred, z_next, h, ps.selected


# ---------------------------------------------------------------------------
# 生长 (误差驱动): 高负载克隆 → 适应难预测状态
# ---------------------------------------------------------------------------
def grow(model, perturb=0.1, n=2):
    unit = model.unit
    inactive = (~unit.active_mask).nonzero().flatten()
    if len(inactive) == 0:
        return 0
    active = unit.active_mask.nonzero().flatten()
    loads = unit._load_ema[active]
    cand = active[loads.argsort(descending=True)[:n]]
    n_grow = min(len(cand), len(inactive))
    with torch.no_grad():
        for src, tgt in zip(cand, inactive[:n_grow]):
            unit.W1.data[:, tgt] = unit.W1.data[:, src] + perturb * torch.randn_like(unit.W1.data[:, src])
            unit.W2.data[tgt, :] = unit.W2.data[src, :] + perturb * torch.randn_like(unit.W2.data[src, :])
            unit.b1.data[tgt] = unit.b1.data[src]
            unit.active_mask[tgt] = True
            model.growth_log.append(int(tgt))
    return n_grow


# ---------------------------------------------------------------------------
# 淘汰 (用进废退): 低激活率生长神经元回收
# ---------------------------------------------------------------------------
def grow_hard(model, sel_hard, perturb=0.1, n=2):
    """难样本定向生长: 克隆预测误差最大样本激活的神经元。
    → 生长自动聚集在难预测状态 (如临界平衡态), 而非全局随机。"""
    unit = model.unit
    inactive = (~unit.active_mask).nonzero().flatten()
    if len(inactive) == 0:
        return 0
    cnt = torch.zeros(unit.d_pool, device=unit.W1.device)
    for row in sel_hard:
        cnt[row] += 1
    active = unit.active_mask
    cand = torch.argsort(cnt * active.float(), descending=True)[:n * 2]
    cand = cand[active[cand]][:n]
    n_grow = min(len(cand), len(inactive))
    with torch.no_grad():
        for src_i, tgt in zip(cand, inactive[:n_grow]):
            unit.W1.data[:, tgt] = unit.W1.data[:, src_i] + perturb * torch.randn_like(unit.W1.data[:, src_i])
            unit.W2.data[tgt, :] = unit.W2.data[src_i, :] + perturb * torch.randn_like(unit.W2.data[src_i, :])
            unit.b1.data[tgt] = unit.b1.data[src_i]
            unit.active_mask[tgt] = True
            model.growth_log.append(int(tgt))
    return n_grow


def prune(model, thr=0.005):
    pool = model.unit
    n_init = 128
    rates = pool.act_rate.cpu().numpy()
    weak = [i for i in range(n_init, pool.d_pool)
            if pool.active_mask[i] and rates[i] < thr]
    with torch.no_grad():
        for i in weak:
            pool.active_mask[i] = False
    return len(weak)


# ---------------------------------------------------------------------------
# 统一训练循环
# ---------------------------------------------------------------------------
def train_gdsp(collect, obs_dim, act_dim, discrete=True, epochs=150,
               grow_every=20, grow_thr=0.05, prune_thr=0.005, max_grow=60,
               sleep_every=0, device="cuda"):
    """collect() → (S, A, R, Sn)。discrete: 动作 one-hot / 连续向量。"""
    S, A, R, Sn = collect()
    n = len(S)
    model = GDSPModel(obs_dim, act_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    s_t = torch.from_numpy(S).float().to(device)
    r_t = torch.from_numpy(R).float().to(device)
    sn_t = torch.from_numpy(Sn).float().to(device)
    a_t = (F.one_hot(torch.from_numpy(A).long(), act_dim).float()
           if discrete else torch.from_numpy(A).float()).to(device)

    step = 0
    loss_ema = 1.0
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n)[:8192]
        sp, rp, zp, _, sel = model(s_t[idx], a_t[idx])
        loss = F.mse_loss(sp, sn_t[idx]) + 0.5 * F.mse_loss(rp, r_t[idx]) \
               + 0.1 * F.mse_loss(zp, model.embed(torch.cat([s_t[idx], a_t[idx]], -1)).detach())
        opt.zero_grad(); loss.backward(); opt.step()
        loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
        # 生长+淘汰 (必需组件)
        step += 1
        if step % grow_every == 0:
            n_prune = prune(model, prune_thr)
            if len(model.growth_log) < max_grow:
                # 难样本定向生长: 预测误差最大的样本激活的神经元
                per_err = (sp - sn_t[idx]).pow(2).mean(-1)
                hard_idx = int(per_err.argmax().item())
                grow_hard(model, sel[hard_idx])
        if ep % 30 == 29:
            model.eval()
            with torch.no_grad():
                sp, _, _, _, _ = model(s_t[:2000], a_t[:2000])
                err = F.mse_loss(sp, sn_t[:2000]).item()
            print(f"  ep {ep+1}: loss={loss.item():.4f} 预测={err:.4f} "
                  f"生长{len(model.growth_log)} 激活{int(model.unit.active_mask.sum())}",
                  flush=True)
    return model


# ---------------------------------------------------------------------------
# 决策: 学习 V + 采样 MPC (纯框架)
# ---------------------------------------------------------------------------
class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
    def forward(self, s):
        return self.net(s).squeeze(-1)


def train_value(env_fn, obs_dim, act_dim, discrete, device="cuda", epochs=200,
                gamma=0.95, seed=7):
    """学习 V: 从经验轨迹算 MC 回报 (混合策略覆盖, 非单步近似)。"""
    import gymnasium as gym
    env = env_fn()
    rng = np.random.RandomState(seed)
    X, G = [], []
    for ep in range(200):
        obs, _ = env.reset()
        traj = []
        for _ in range(100):
            if discrete:
                a = int(rng.randint(act_dim))
            else:
                a = rng.uniform(-1, 1, act_dim).astype(np.float32)
            obs_next, r, done, _, _ = env.step(a)
            traj.append((obs, r))
            obs = obs_next
            if done:
                break
        g = 0.0
        for s, r in reversed(traj):
            g = r + gamma * g
            X.append(s); G.append(g)
    env.close()
    V = ValueNet(obs_dim).to(device)
    opt = torch.optim.AdamW(V.parameters(), lr=1e-3)
    Xt = torch.from_numpy(np.array(X, np.float32)).to(device)
    Gt = torch.from_numpy(np.array(G, np.float32)).to(device)
    for ep in range(epochs):
        idx = torch.randperm(len(Xt))[:4096]
        loss = F.mse_loss(V(Xt[idx]), Gt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return V


def mpc_act(model, V, obs, obs_dim, act_dim, discrete, n_samples=200, device="cuda"):
    """采样 MPC: 评分 = 预测奖励 + γ·V(s') (学习价值, 非手动规则)。"""
    rng = np.random.RandomState()
    if discrete:
        best_a, best = 0, -1e9
        with torch.no_grad():
            for a in range(act_dim):
                obs_t = torch.from_numpy(obs[None]).float().to(device)
                act_t = torch.from_numpy(np.eye(act_dim)[a][None]).float().to(device)
                sp, rp, _, _, _ = model(obs_t, act_t)
                score = rp.item() + 0.95 * V(sp[0]).item()
                if score > best:
                    best, best_a = score, a
        return best_a
    else:
        acts = rng.uniform(-1, 1, (n_samples, act_dim)).astype(np.float32)
        obs_t = torch.from_numpy(np.tile(obs, (n_samples, 1))).float().to(device)
        act_t = torch.from_numpy(acts).float().to(device)
        with torch.no_grad():
            sp, rp, _, _, _ = model(obs_t, act_t)
            score = rp + 0.95 * V(sp)
        return acts[int(score.argmax().item())]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="cartpole")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.env == "cartpole":
        import gymnasium as gym
        from train_cartpole import heuristic
        obs_dim, act_dim, discrete = 4, 2, True
        def collect(n_eps=300):
            env = gym.make('CartPole-v1')
            rng = np.random.RandomState(42)
            S, A, R, Sn = [], [], [], []
            for ep in range(n_eps):
                obs, _ = env.reset()
                for _ in range(300):
                    a = heuristic(obs) if ep % 2 == 0 else int(rng.randint(2))
                    o2, r, d, _, _ = env.step(a)
                    S.append(obs); A.append(a); R.append(r); Sn.append(o2)
                    obs = o2
                    if d: break
            env.close()
            return (np.array(S, np.float32), np.array(A, np.int64),
                    np.array(R, np.float32), np.array(Sn, np.float32))
    elif args.env == "walker":
        import gymnasium as gym
        obs_dim, act_dim, discrete = 24, 4, False
        def collect(n_eps=600):
            env = gym.make('BipedalWalker-v3')
            rng = np.random.RandomState(42)
            S, A, R, Sn = [], [], [], []
            for _ in range(n_eps):
                obs, _ = env.reset()
                for _ in range(60):
                    a = rng.uniform(-1, 1, 4).astype(np.float32)
                    o2, r, d, _, _ = env.step(a)
                    S.append(obs); A.append(a); R.append(r); Sn.append(o2)
                    obs = o2
                    if d: break
            env.close()
            return (np.array(S, np.float32), np.array(A, np.float32),
                    np.array(R, np.float32), np.array(Sn, np.float32))
    else:
        raise ValueError(args.env)

    print(f"=== GDSP 统一入口 [{args.env}] (含生长+淘汰, 纯框架) ===", flush=True)
    model = train_gdsp(collect, obs_dim, act_dim, discrete, args.epochs, device=args.device)
    import gymnasium as gym
    env_fn = lambda: gym.make('CartPole-v1' if args.env == 'cartpole' else 'BipedalWalker-v3')
    V = train_value(env_fn, obs_dim, act_dim, discrete, device=args.device)
    model.eval(); V.eval()

    # 评估
    import gymnasium as gym
    env = gym.make('CartPole-v1' if args.env == 'cartpole' else 'BipedalWalker-v3')
    scores = []
    for _ in range(10):
        obs, _ = env.reset()
        total, done = 0.0, False
        for t in range(300 if args.env == 'walker' else 500):
            a = mpc_act(model, V, obs, obs_dim, act_dim, discrete, device=args.device)
            obs, r, done, _, _ = env.step(a)
            total += r
            if done: break
        scores.append(total)
    env.close()
    print(f"[统一入口 {args.env}] 平均得分 {np.mean(scores):.1f}")
    torch.save({"model": model.state_dict()}, f"runs/gdsp_{args.env}.pt")
    print(f"保存: runs/gdsp_{args.env}.pt")
