"""train_wm_image.py — 视觉世界模型: 图像观测迁移验证。

图像 = 局部第一人称视野 (5×5 格 → 20×20 像素, 含食物/墙/agent标记)
  90 唯一/340 状态 → 有歧义 → 路径积分有任务 (相机类比)

架构 (Dreamer 式 latent 化):
  CNN 编码器: image → z (64)          [不预测像素, 预测潜在特征]
  GRU: z 序列积分 → h (认知地图载体)
  头: 预测 z_{t+1} (latent) + 奖励
  位置解码: h → (x,y) 线性探针

验证:
  ① 单帧解码 vs 积分后解码 (路径积分是否在视觉下工作)
  ② 对比符号实验 (walls 96.7%) — 视觉迁移
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

from envs.survival_maze import SurvivalMaze, render_image
from world_models.train_wm_explore import ACT

IMG = 20  # 局部视野像素 (5格×4px)


class CNNEnc(nn.Module):
    """图像 → 潜在特征 z。"""

    def __init__(self, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 10x10
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 5x5
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),                  # 5x5
            nn.Flatten(),
        )
        self.proj = nn.Linear(32 * 5 * 5, z_dim)

    def forward(self, img):
        return self.proj(self.net(img))


class WorldImage(nn.Module):
    def __init__(self, z_dim=64, hidden=128, pool=512, top_k=64, active_ratio=0.25):
        super().__init__()
        self.z_dim = z_dim
        self.enc = CNNEnc(z_dim)
        # 神经元池 (可生长): z → 池 → GRU
        from units.sparse_unit import SparseUnit
        self.pool = SparseUnit(d_model=z_dim, d_pool=pool, top_k=top_k)
        self.pool.register_buffer("active_mask",
                                  torch.zeros(pool, dtype=torch.bool))
        n_init = max(1, int(pool * active_ratio))
        self.pool.active_mask[:n_init] = True
        self.pool._load_ema = torch.zeros(pool)
        self.dec_in = nn.Linear(z_dim + ACT, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head_z = nn.Linear(hidden, z_dim)   # latent 预测
        self.head_r = nn.Linear(hidden, 1)       # 奖励

    def forward(self, img, actions, h=None):
        """img: (B, T, 1, 20, 20) actions: (B, T) → (z_pred, r_pred, h_seq)"""
        B, T = img.shape[:2]
        z_cnn = self.enc(img.reshape(-1, 1, IMG, IMG)).view(B, T, self.z_dim)
        z_pool, ps = self.pool(z_cnn)  # 神经元池稀疏激活
        z = z_pool
        h_seq = []
        if h is None:
            h = torch.zeros(B, self.gru.hidden_size, device=img.device)
        for t in range(T):
            sa = torch.cat([z[:, t], F.one_hot(actions[:, t].long(), ACT).float()], -1)
            h = self.gru(torch.tanh(self.dec_in(sa)), h)
            h_seq.append(h)
        h_seq = torch.stack(h_seq, 1)  # (B, T, hidden)
        z_pred = self.head_z(h_seq)
        r_pred = self.head_r(h_seq).squeeze(-1)
        return z_pred, r_pred, h_seq, z, ps, z_cnn


def collect_image(env, n_episodes=1500, max_steps=50, seed=0):
    """随机策略收集 (图像观测, 局部视野)。"""
    rng = np.random.RandomState(seed)
    S, A, R, Sn, P = [], [], [], [], []
    for _ in range(n_episodes):
        env.energy = env.E0
        obs = env.reset()
        for _ in range(max_steps):
            img = render_image(env, 'local', window=5)  # step 前的观测
            a = int(rng.randint(ACT))
            obs_next, r, done = env.step(a)
            S.append(img)
            A.append(a)
            R.append(r)
            Sn.append(render_image(env, 'local', window=5))
            P.append([env.x / env.size, env.y / env.size])
            obs = obs_next
            if done:
                break
    return (np.array(S, np.float32), np.array(A, np.int64),
            np.array(R, np.float32), np.array(Sn, np.float32),
            np.array(P, np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--T", type=int, default=25)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--n_episodes", type=int, default=1500)
    p.add_argument("--arm", type=str, choices=["func", "rand", "none"], default="none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    env = SurvivalMaze(**cfg.SURVIVAL_ENV)
    print("收集图像轨迹...", flush=True)
    S, A, R, Sn, P = collect_image(env, args.n_episodes)
    n = len(S)
    print(f"  {n} 步, 图像 {S.shape}", flush=True)

    model = WorldImage().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    s_t = torch.from_numpy(S).float().to(device)
    a_t = torch.from_numpy(A).long().to(device)
    r_t = torch.from_numpy(R).float().to(device)
    sn_t = torch.from_numpy(Sn).float().to(device)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        step = 0
        for i in range(0, n - args.T, args.bs):
            idx = np.random.randint(0, n - args.T, args.bs)
            idx = np.concatenate([np.arange(j, j + args.T) for j in idx])
            sb = s_t[idx].view(args.bs, args.T, 1, IMG, IMG)
            ab = a_t[idx].view(args.bs, args.T)
            rb = r_t[idx].view(args.bs, args.T)
            snb = sn_t[idx].view(args.bs, args.T, 1, IMG, IMG)
            z_pred, rp, h, z, ps, z_cnn = model(sb, ab)
            # latent 预测目标: 池输出 (正则化目标, 模型更好)
            z_next = torch.cat([z[:, 1:], z[:, -1:]], 1)
            loss = F.mse_loss(z_pred, z_next.detach()) + 0.5 * F.mse_loss(rp, rb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            # 负载 EMA + 生长
            with torch.no_grad():
                model.pool._load_ema = 0.99 * model.pool._load_ema.to(ps.load.device) + 0.01 * ps.load.mean(0)
            step += 1
            grown_so_far = int(model.pool.active_mask.sum()) - 128
            if args.arm != "none" and step % 300 == 0 and step > 300 and grown_so_far < 60:
                grow_neurons(model, args.arm, 2, rng=np.random.RandomState(args.seed))
        if (ep + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "config": vars(args)},
                       f"runs/wm_image_{args.arm}.pt")
        if ep % 5 == 4:
            model.eval()
            with torch.no_grad():
                acc_single = decode_pos(S, A, P, model, mode="single", device=device)
                acc_gru = decode_pos(S, A, P, model, mode="gru", T=args.T, device=device)
                print(f"  ep {ep+1}: loss={tot/(n//args.bs):.4f} | "
                      f"单帧解码={acc_single:.3f} GRU积分={acc_gru:.3f}", flush=True)

    torch.save({"model": model.state_dict(), "config": vars(args)}, f"runs/wm_image_{args.arm}.pt")
    print("保存: runs/wm_image.pt")




def grow_neurons(model, arm, n=2, perturb=0.05, rng=None):
    """误差驱动 (func): 克隆高负载活跃神经元; 随机 (rand)。"""
    unit = model.pool
    inactive = (~unit.active_mask).nonzero().flatten()
    if len(inactive) == 0:
        return 0
    active = unit.active_mask.nonzero().flatten()
    if arm == "func":
        loads = unit._load_ema.to(unit.W1.device)[active]
        cand = active[loads.argsort(descending=True)[:n]]
    else:
        cand = active[rng.choice(len(active), min(n, len(active)), replace=False)]
    n_grow = min(len(cand), len(inactive))
    with torch.no_grad():
        for src_i, tgt in zip(cand, inactive[:n_grow]):
            unit.W1.data[:, tgt] = unit.W1.data[:, src_i] + perturb * torch.randn_like(unit.W1.data[:, src_i])
            unit.W2.data[tgt, :] = unit.W2.data[src_i, :] + perturb * torch.randn_like(unit.W2.data[src_i, :])
            unit.b1.data[tgt] = unit.b1.data[src_i]
            unit.active_mask[tgt] = True
    return n_grow


def eval_removal(model, S, A, Sn, device, K=30):
    """删 K 个生长神经元 → latent 预测误差上升 (固定目标)。"""
    n = len(S)
    s_t = torch.from_numpy(S).float().to(device)
    a_t = torch.from_numpy(A).long().to(device)
    idx = np.random.RandomState(0).choice(n, 2000, replace=False)
    sb = s_t[idx].view(1, 2000, 1, IMG, IMG)
    ab = a_t[idx].view(1, 2000)
    with torch.no_grad():
        _, _, _, _, _, z_cnn_full = model(sb, ab)
        z_next_fixed = torch.cat([z_cnn_full[:, 1:], z_cnn_full[:, -1:]], 1).detach()
        z_pred0, _, _, _, _, _ = model(sb, ab)
        e0 = F.mse_loss(z_pred0, z_next_fixed).item()
    n_active = int(model.pool.active_mask.sum())
    grow_idx = np.arange(128, n_active)
    K = min(K, len(grow_idx))
    with torch.no_grad():
        for nid in grow_idx[:K]:
            model.pool.W2.data[nid, :] = 0
    with torch.no_grad():
        z_pred1, _, _, _, _, _ = model(sb, ab)
        e1 = F.mse_loss(z_pred1, z_next_fixed).item()
    return e0, e1, len(grow_idx), K

def decode_pos(S, A, P, model=None, mode="single", T=25, device="cuda"):
    """位置解码 (单帧图像 or GRU 积分)。"""
    n = len(S)
    s_t = torch.from_numpy(S).float().to(device)
    idx = np.arange(n)
    if mode == "single":
        enc = model.enc if model else None
        with torch.no_grad():
            z = enc(s_t[:4000].view(-1, 1, IMG, IMG)).cpu().numpy()
        X, y = z, P[:4000]
    else:
        n_win = n // T
        H = []
        a_t = torch.from_numpy(A).long().to(device)
        with torch.no_grad():
            for w0 in range(0, n_win, 64):
                wsel = np.arange(w0, min(w0 + 64, n_win))
                idx = np.concatenate([np.arange(w * T, (w + 1) * T) for w in wsel])
                nb = len(idx) // T
                sb = s_t[idx].view(nb, T, 1, IMG, IMG)
                ab = a_t[idx].view(nb, T)
                _, _, h, _, _, _ = model(sb, ab)
                H.append(h.cpu().numpy().reshape(-1, model.gru.hidden_size))
        X = np.concatenate(H)
        idx_all = np.concatenate([np.arange(w * T, (w + 1) * T)
                                  for w in range(n_win)])
        y = P[idx_all]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(X))
    tr, va = perm[:int(len(X) * 0.7)], perm[int(len(X) * 0.7):]
    lam = 1e-3 * X.shape[1]
    W = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ y[tr])
    pred = X[va] @ W
    return float(np.mean(np.abs(pred - y[va]).max(1) < 0.05))


if __name__ == "__main__":
    main()
