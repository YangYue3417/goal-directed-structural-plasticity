"""train_wm_image_v3.py — 视觉 v3: 连接掩码学习 (感受野自然浮现)。

核心 (用户驱动的设计修正):
  - 空间保留 CNN: 20×20 图像 → 5×5×16 特征图 (保留位置)
  - 可学习连接池: 每神经元有位置亲和度 score (5×5)
    → softmax = 连接掩码 m (连哪里)
    → 打分 = Σ_pos m·w·F[pos] (该位置模式检测器)
    → top-k 稀疏激活
  - 熵正则: 逼 m 尖峰 → 感受野自然浮现 (V1 式)
  - 生长: 新神经元 score 由误差贡献初始化 (连难预测区域)

验证:
  ① 感受野形成: 每神经元 m 是否集中局部区域 (熵下降)
  ② 认知地图: GRU 隐状态位置解码 (像素 → 位置)
  ③ 生长: 误差驱动, 删神经元 → 预测/解码下降
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

from envs.survival_maze import SurvivalMaze, render_image
from world_models.train_wm_explore import ACT

IMG = 20
H, W, C = 5, 5, 16


class SpatialCNN(nn.Module):
    """空间保留 CNN: 图像 → 5×5×16 特征图。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, C, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, img):
        return self.net(img)  # (B, C, 5, 5)


class ConnectPool(nn.Module):
    """可学习连接池: 神经元选择"连特征图哪里" (softmax 掩码)。

    每神经元: score (25 位置) → m = softmax(score/τ) → 连接掩码
              w (25×C) → 每位置模式权重
    打分: s_i = Σ_pos m_i[pos] · (w_i[pos] · F[pos])   # 标量检测器
    激活: top-k 稀疏 (只有选中的神经元发放)
    """

    def __init__(self, n_neurons=512, top_k=64, tau=1.0, n_pos=H * W):
        super().__init__()
        self.n_neurons = n_neurons
        self.top_k = top_k
        self.tau = tau
        self.score = nn.Parameter(torch.randn(n_neurons, n_pos) * 0.5)
        self.w = nn.Parameter(torch.randn(n_neurons, n_pos, C) * 0.1)
        self.register_buffer("active_mask", torch.zeros(n_neurons, dtype=torch.bool))
        self.active_mask[:128] = True
        self.register_buffer("_grow_pos", torch.zeros(n_neurons))
        self.register_buffer("act_rate", torch.zeros(n_neurons))  # 选中频率 EMA
        self.growth_log = []  # 生长历史 (神经元 id)

    def forward(self, F_img):
        """F_img: (B, C, H, W) → (out: B,n_neurons 稀疏, m: n×25, scores)"""
        B = F_img.shape[0]
        Ff = F_img.flatten(2).permute(0, 2, 1)  # (B, 25, C)
        m = F.softmax(self.score / self.tau, dim=1)  # (n, 25)
        m_b = m[None].expand(B, -1, -1)  # (B, n, 25)
        # 打分: (B, n, 25) × (n, 25, C) × (B, 25, C) → (B, n)
        s = torch.einsum('bnj,njc,bjc->bn', m_b, self.w, Ff)
        # top-k 稀疏 (只活跃神经元)
        active = self.active_mask
        s_masked = s.masked_fill(~active[None], -1e9)
        vals, idx = s_masked.topk(self.top_k, dim=1)
        sparse = torch.zeros_like(s)
        sparse.scatter_(1, idx, vals)
        # 激活率 EMA (选中频率)
        with torch.no_grad():
            onehot = torch.zeros_like(s)
            onehot.scatter_(1, idx, 1.0)
            self.act_rate = 0.999 * self.act_rate + 0.001 * onehot.mean(0)
        return sparse, m, s

    def entropy(self):
        """连接掩码熵 (尖峰 → 低熵 = 感受野形成)。"""
        m = F.softmax(self.score / self.tau, dim=1)
        return -(m * m.clamp_min(1e-9).log()).sum(1)  # (n,)

    def receptive_field_size(self, thresh=0.3):
        """每神经元感受野大小: m 超过阈值的格子数。"""
        m = F.softmax(self.score / 1.0, dim=1)
        return (m > thresh).sum(1)


class WorldImageV3(nn.Module):
    """视觉世界模型 v3: SpatialCNN → ConnectPool → GRU → latent 预测。"""

    def __init__(self, hidden=128, pool_n=512, top_k=64, z_dim=64):
        super().__init__()
        self.cnn = SpatialCNN()
        self.pool = ConnectPool(pool_n, top_k)
        self.dec_in = nn.Linear(z_dim + ACT, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head_z = nn.Linear(hidden, z_dim)
        self.head_r = nn.Linear(hidden, 1)
        # latent 编码 (认知地图用)
        self.z_enc = nn.Linear(pool_n, z_dim)

    def forward(self, img, actions, h=None):
        """img: (B, T, 1, 20, 20) actions: (B,T) → (z_pred, r_pred, h_seq, pool_out, m)"""
        B, T = img.shape[:2]
        F_img = self.cnn(img.reshape(-1, 1, IMG, IMG))  # (B*T, C, 5, 5)
        pool_out, m, s = self.pool(F_img)  # (B*T, n) 稀疏
        pool_out = pool_out.view(B, T, -1)
        z = torch.tanh(self.z_enc(pool_out))  # (B, T, z_dim) 认知特征
        h_seq = []
        if h is None:
            h = torch.zeros(B, self.gru.hidden_size, device=img.device)
        for t in range(T):
            sa = torch.cat([z[:, t], F.one_hot(actions[:, t].long(), ACT).float()], -1)
            h = self.gru(torch.tanh(self.dec_in(sa)), h)
            h_seq.append(h)
        h_seq = torch.stack(h_seq, 1)
        z_pred = self.head_z(h_seq)
        r_pred = self.head_r(h_seq).squeeze(-1)
        return z_pred, r_pred, h_seq, z, m




def grow_v3(model, err_pos, arm, n=2, rng=None, perturb=0.1):
    """v3 生长: 新神经元连接由位置误差决定 (func) 或随机 (rand)。

    func: 高误差位置 p* → 新神经元 score 尖峰指向 p* (连接那里)
    rand: 随机位置
    """
    pool = model.pool
    inactive = (~pool.active_mask).nonzero().flatten()
    if len(inactive) == 0:
        return 0
    if arm == "func":
        p_star = int(np.argmax(err_pos))
    else:
        p_star = int(rng.randint(25))
    n_grow = min(n, len(inactive))
    with torch.no_grad():
        for tgt in inactive[:n_grow]:
            # 连接掩码: 尖峰指向 p* (周围也留一点)
            pos_vec = torch.zeros(25)
            pos_vec[p_star] = 3.0
            for d in [-1, 1]:
                for ax in [0, 1]:
                    q = p_star + d * (1 if ax == 0 else 5)
                    if 0 <= q < 25 and (q % 5 == p_star % 5 or q // 5 == p_star // 5):
                        pos_vec[q] = 1.0
            pool.score.data[tgt] = pos_vec
            # 权重: 从活跃中按负载克隆 (误差驱动: 负载 EMA 高的)
            active = pool.active_mask.nonzero().flatten()
            loads = pool.act_rate[active]
            src = active[torch.argmax(loads)] if arm == "func" and len(active) else                   active[int(rng.randint(len(active)))]
            pool.w.data[tgt] = pool.w.data[src] + perturb * torch.randn_like(pool.w.data[src])
            pool.active_mask[tgt] = True
            pool.growth_log.append(int(tgt))
    return n_grow



def prune_weak(model, thr=0.005):
    """优胜劣汰: 淘汰低激活率的生长神经元 (槽位释放供再生长)。"""
    pool = model.pool
    grow_idx = np.arange(128, pool.n_neurons)
    rates = pool.act_rate.cpu().numpy()
    weak = [i for i in grow_idx if pool.active_mask[i] and rates[i] < thr]
    with torch.no_grad():
        for i in weak:
            pool.active_mask[i] = False
    return len(weak)

def collect_image(env, n_episodes=1200, max_steps=50, seed=0):
    rng = np.random.RandomState(seed)
    S, A, R, Sn, P = [], [], [], [], []
    for _ in range(n_episodes):
        env.energy = env.E0
        obs = env.reset()
        for _ in range(max_steps):
            img = render_image(env, 'local', window=5)
            a = int(rng.randint(ACT))
            obs_next, r, done = env.step(a)
            S.append(img); A.append(a); R.append(r)
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
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--T", type=int, default=25)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--n_episodes", type=int, default=1200)
    p.add_argument("--entropy_w", type=float, default=0.05, help="连接熵正则")
    p.add_argument("--arm", type=str, choices=["func", "rand", "none"], default="none")
    p.add_argument("--grow_every", type=int, default=800)
    p.add_argument("--prune_thr", type=float, default=0.005, help="淘汰阈值")
    p.add_argument("--max_grow", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    env = SurvivalMaze(size=10, n_foods=6, seed=42, E0=200.0, day_steps=60, food_restore=80.0)
    print("收集图像轨迹...", flush=True)
    S, A, R, Sn, P = collect_image(env, args.n_episodes)
    n = len(S)
    print(f"  {n} 步", flush=True)

    model = WorldImageV3().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng_grow = np.random.RandomState(args.seed + 1)  # 持久 rng (修复 rand bug)
    s_t = torch.from_numpy(S).float().to(device)
    a_t = torch.from_numpy(A).long().to(device)
    r_t = torch.from_numpy(R).float().to(device)

    for ep in range(args.epochs):
        model.train()
        tot, ent_tot = 0.0, 0.0
        for i in range(0, n - args.T, args.bs):
            idx = np.random.randint(0, n - args.T, args.bs)
            idx = np.concatenate([np.arange(j, j + args.T) for j in idx])
            sb = s_t[idx].view(args.bs, args.T, 1, IMG, IMG)
            ab = a_t[idx].view(args.bs, args.T)
            rb = r_t[idx].view(args.bs, args.T)
            z_pred, rp, h, z, m = model(sb, ab)
            z_next = torch.cat([z[:, 1:], z[:, -1:]], 1)
            ent = model.pool.entropy().mean()
            loss = F.mse_loss(z_pred, z_next.detach()) + 0.5 * F.mse_loss(rp, rb) \
                   + args.entropy_w * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            ent_tot += ent.item()
            # 优胜劣汰: 淘汰 + 生长
            step = ep * (n // args.T // args.bs) + i // args.bs
            if args.arm != "none" and step % args.grow_every == 0:
                n_prune = prune_weak(model, args.prune_thr)
                grown = len(model.pool.growth_log)
                if grown < args.max_grow:
                    with torch.no_grad():
                        F_img = model.cnn(sb.reshape(-1, 1, IMG, IMG))
                        Ff = F_img.flatten(2).permute(0, 2, 1)
                        resid = (Ff - Ff.mean(0, keepdim=True)).abs().mean(0).mean(0)
                    grow_v3(model, resid.cpu().numpy(), args.arm, 2, rng=rng_grow)
                if n_prune > 0 and step % (args.grow_every * 5) == 0:
                    print(f"    淘汰 {n_prune} 弱神经元 (step {step})", flush=True)
        if ep % 5 == 4:
            model.eval()
            rf = model.pool.receptive_field_size().float().mean().item()
            print(f"  ep {ep+1}: loss={tot/(n//args.bs):.4f} 熵={ent_tot/(n//args.bs):.3f} "
                  f"感受野大小={rf:.1f}格/25", flush=True)

    torch.save({"model": model.state_dict(), "config": vars(args)},
               f"runs/wm_image_v3_{args.arm}.pt")
    print("保存: runs/wm_image_v3.pt")


if __name__ == "__main__":
    main()
