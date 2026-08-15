"""train_atari_wm.py — Atari (Ms. Pac-Man) 最小机制验证。

验证三件事:
  ① 世界模型在真实游戏图像上能否学到动力学 (latent 预测收敛)
  ② 连接掩码学习 → 感受野在真实图像上是否浮现 (空间选择性)
  ③ 奖励预测 (吃豆 +1 / 幽灵接触)

观测: 灰度 84×84 → CNN → 5×5×16 特征图 → ConnectPool → GRU → latent 预测
数据: 随机策略 ~50K 帧 (机制验证, 非 SOTA)
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

from world_models.train_wm_image_v3 import ConnectPool, grow_v3, prune_weak

IMG = 84
H, W, C = 5, 5, 16


class AtariCNN(nn.Module):
    """84×84 灰度 → 5×5×16 特征图 (空间保留)。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 42
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 21
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 10
            nn.Conv2d(32, C, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 5
        )

    def forward(self, img):
        return self.net(img)


class AtariWorld(nn.Module):
    def __init__(self, n_act=9, hidden=128, pool_n=512, top_k=64, z_dim=64):
        super().__init__()
        self.n_act = n_act
        self.cnn = AtariCNN()
        self.pool = ConnectPool(pool_n, top_k)
        self.dec_in = nn.Linear(z_dim + n_act, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.head_z = nn.Linear(hidden, z_dim)
        self.head_r = nn.Linear(hidden, 1)
        self.z_enc = nn.Linear(pool_n, z_dim)

    def forward(self, img, actions, h=None):
        B, T = img.shape[:2]
        F_img = self.cnn(img.reshape(-1, 1, IMG, IMG))
        pool_out, m, s = self.pool(F_img)
        pool_out = pool_out.view(B, T, -1)
        z = torch.tanh(self.z_enc(pool_out))
        h_seq = []
        if h is None:
            h = torch.zeros(B, self.gru.hidden_size, device=img.device)
        for t in range(T):
            sa = torch.cat([z[:, t], F.one_hot(actions[:, t].long(), self.n_act).float()], -1)
            h = self.gru(torch.tanh(self.dec_in(sa)), h)
            h_seq.append(h)
        h_seq = torch.stack(h_seq, 1)
        return self.head_z(h_seq), self.head_r(h_seq).squeeze(-1), h_seq, z, m


def collect_atari(n_frames=50000, seed=0, max_ep=500):
    """随机策略收集: 灰度 84×84, (帧, 动作, 奖励, 下一帧)。"""
    import ale_py  # noqa: 注册 ALE 命名空间
    import gymnasium as gym
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    rng = np.random.RandomState(seed)
    S, A, R, Sn = [], [], [], []
    ep = 0
    obs, _ = env.reset()
    while len(S) < n_frames and ep < max_ep:
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        import cv2
        img = cv2.resize(gray, (IMG, IMG)).astype(np.float32) / 255.0
        a = int(rng.randint(env.action_space.n))
        obs_next, r, done, _, _ = env.step(a)
        S.append(img); A.append(a); R.append(r)
        gray_n = np.dot(obs_next[..., :3], [0.299, 0.587, 0.114])
        Sn.append(cv2.resize(gray_n, (IMG, IMG)).astype(np.float32) / 255.0)
        if done:
            obs, _ = env.reset()
            ep += 1
        else:
            obs = obs_next
    env.close()
    return (np.array(S, np.float32), np.array(A, np.int64),
            np.array(R, np.float32), np.array(Sn, np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--bs", type=int, default=24)
    p.add_argument("--n_frames", type=int, default=50000)
    p.add_argument("--entropy_w", type=float, default=0.03)
    p.add_argument("--arm", type=str, choices=["func", "rand", "none"], default="none")
    p.add_argument("--grow_every", type=int, default=600)
    p.add_argument("--prune_thr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("收集 Atari 随机数据...", flush=True)
    S, A, R, Sn = collect_atari(args.n_frames, args.seed)
    n = len(S)
    print(f"  {n} 帧, 奖励范围 [{R.min()}, {R.max()}], 非零奖励 {(R != 0).mean():.3f}", flush=True)

    model = AtariWorld().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng_grow = np.random.RandomState(args.seed + 1)
    s_t = torch.from_numpy(S).float().to(device)
    a_t = torch.from_numpy(A).long().to(device)
    r_t = torch.from_numpy(R).float().to(device)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
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
            # 生长+淘汰
            step = ep * (n // args.T // args.bs) + i // args.bs
            if args.arm != "none" and step % args.grow_every == 0:
                n_prune = prune_weak(model, args.prune_thr)
                if len(model.pool.growth_log) < 60:
                    with torch.no_grad():
                        F_img = model.cnn(sb.reshape(-1, 1, IMG, IMG))
                        Ff = F_img.flatten(2).permute(0, 2, 1)
                        resid = (Ff - Ff.mean(0, keepdim=True)).abs().mean(0).mean(0)
                    grow_v3(model, resid.cpu().numpy(), args.arm, 2, rng=rng_grow)
        if ep % 5 == 4:
            model.eval()
            rf = model.pool.receptive_field_size().float().mean().item()
            grown = len(model.pool.growth_log)
            alive = sum(1 for i in model.pool.growth_log if model.pool.active_mask[i])
            print(f"  ep {ep+1}: loss={tot/(n//args.bs):.4f} 熵={model.pool.entropy().mean().item():.3f} "
                  f"感受野={rf:.1f}格 生长{alive}/{grown}", flush=True)

    torch.save({"model": model.state_dict(), "config": vars(args)},
               f"runs/atari_wm_{args.arm}.pt")
    print(f"保存: runs/atari_wm_{args.arm}.pt")


if __name__ == "__main__":
    main()
