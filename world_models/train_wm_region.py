"""train_wm_region.py — 区域专精: 新区域出现 → 生长神经元专精新区域。

设计 (中心问题最后拼图):
  固定地图 (seed=777)
  阶段 1: 只在区域 A (x<10) 收集轨迹训练 WM
  阶段 2: 引入区域 B (x>=10) → 误差驱动生长 (func arm)
  
验证:
  ① 生长神经元 (idx>=n_init) 的激活位置分布 → 是否集中区域 B
  ② 删生长神经元 → 区域 B 预测误差上升 >> 区域 A (区域专精!)
  对照: rand (随机生长) — 应无区域选择性
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

from envs.survival_maze import SurvivalMaze
from world_models.train_wm_explore import WorldExplore, ACT, OBS


def collect_region(env, n_episodes=1500, max_steps=50, region="A", seed=0):
    """固定地图, 区域过滤收集。region: 'A'=x<10, 'B'=x>=10, 'ALL'"""
    rng = np.random.RandomState(seed)
    S, A, R, Sn, P = [], [], [], [], []
    for _ in range(n_episodes):
        env.energy = env.E0
        obs = env.reset()
        for _ in range(max_steps):
            a = int(rng.randint(ACT))
            obs_next, r, done = env.step(a)
            x = env.x / env.size
            inA = x < 0.5
            if region == "A" and not inA:
                pass  # 跳过 B 样本
            elif region == "B" and inA:
                pass
            elif region != "ALL" and region == "A" and not inA:
                pass
            elif region == "ALL" or (region == "A" and inA) or (region == "B" and not inA):
                S.append(obs); A.append(a); R.append(r); Sn.append(obs_next)
                P.append([env.x / env.size, env.y / env.size])
            obs = obs_next
            if done:
                break
    return (np.array(S, np.float32), np.array(A, np.int64),
            np.array(R, np.float32), np.array(Sn, np.float32),
            np.array(P, np.float32))


def grow_region_neurons(model, S_B, A_B, n=2, perturb=0.05, device="cuda"):
    """区域驱动生长: 克隆在区域 B 上高激活的活跃神经元 (B 专精候选)。"""
    unit = model.unit
    inactive = (~unit.active_mask).nonzero().flatten()
    if len(inactive) == 0:
        return 0
    # B 区域前向, 统计激活计数
    sb = torch.from_numpy(S_B[:1024]).float().to(device)
    ab = torch.from_numpy(A_B[:1024]).long().to(device)
    with torch.no_grad():
        emb = torch.tanh(model.embed(torch.cat([sb, F.one_hot(ab, ACT).float()], -1)))
        _, ps = model.unit(emb.unsqueeze(1))
    sel = ps.selected.cpu().numpy()[:, 0, :]
    cnt = np.zeros(unit.d_pool)
    for row in sel:
        cnt[row] += 1
    # 在 B 高激活的活跃神经元 (排除已满)
    active = unit.active_mask.cpu().numpy()
    cand_score = cnt * active
    cand = np.argsort(cand_score)[::-1][:n * 3]
    cand = cand[active[cand]]
    n_grow = min(len(cand), len(inactive), n)
    with torch.no_grad():
        for src, tgt in zip(cand[:n_grow], inactive[:n_grow]):
            unit.W1.data[:, tgt] = unit.W1.data[:, src] + perturb * torch.randn_like(unit.W1.data[:, src])
            unit.W2.data[tgt, :] = unit.W2.data[src, :] + perturb * torch.randn_like(unit.W2.data[src, :])
            unit.b1.data[tgt] = unit.b1.data[src]
            unit.active_mask[tgt] = True
    return n_grow


def train_batches(model, S, A, R, Sn, epochs, bs=64, T=30, device="cuda",
                  grow_arm=None, grow_every=200, grow_n=2, rng=None):
    """BPTT 训练 (可选生长)。"""
    s_t = torch.from_numpy(S).float().to(device)
    a_t = torch.from_numpy(A).long().to(device)
    r_t = torch.from_numpy(R).float().to(device)
    sn_t = torch.from_numpy(Sn).float().to(device)
    n = len(S)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    unit = model.unit
    if not hasattr(unit, "_load_ema"):
        unit._load_ema = torch.zeros(unit.d_pool, device=device)
    step = 0
    for ep in range(epochs):
        for i in range(0, n - T, bs):
            idx = np.random.randint(0, n - T, bs)
            idx = np.concatenate([np.arange(j, j + T) for j in idx])
            sb = s_t[idx].view(bs, T, OBS)
            ab = a_t[idx].view(bs, T)
            rb = r_t[idx].view(bs, T)
            snb = sn_t[idx].view(bs, T, OBS)
            op, rp, _, ps = model(sb, ab)
            loss = F.mse_loss(op, snb) + 0.5 * F.mse_loss(rp, rb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                unit._load_ema = 0.99 * unit._load_ema + 0.01 * ps.load.mean(0)
            step += 1
            if grow_arm and step % grow_every == 0 and step > 100:
                if grow_arm == "func":
                    s_b, a_b = unit._grow_b
                    grow_region_neurons(model, s_b.cpu().numpy(), a_b.cpu().numpy(),
                                        grow_n, device=device)
                else:
                    grow_neurons(model, grow_arm, grow_n, rng=rng)
    return unit.active_mask.sum().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", type=str, choices=["func", "rand"], default="func")
    p.add_argument("--epochs_p1", type=int, default=20)
    p.add_argument("--epochs_p2", type=int, default=25)
    p.add_argument("--n_episodes", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.RandomState(args.seed)

    env = SurvivalMaze(**cfg.SURVIVAL_ENV)
    env._day_seed = 777  # 固定地图
    env.energy = env.E0
    env.reset_day()

    print("=== 阶段 1: 只训区域 A (x<10) ===", flush=True)
    S1, A1, R1, Sn1, P1 = collect_region(env, args.n_episodes, region="A")
    print(f"  区域 A: {len(S1)} 样本", flush=True)
    model = WorldExplore().to(device)
    n1 = train_batches(model, S1, A1, R1, Sn1, args.epochs_p1, device=device,
                       rng=rng)

    print("=== 阶段 2: 引入区域 B, 区域驱动生长 ===", flush=True)
    S2, A2, R2, Sn2, P2 = collect_region(env, args.n_episodes, region="B")
    # B 过采样: 让 B 在训练流中占 50%
    S_all = np.concatenate([S1[:len(S2)], S2]); A_all = np.concatenate([A1[:len(A2)], A2])
    R_all = np.concatenate([R1[:len(R2)], R2]); Sn_all = np.concatenate([Sn1[:len(Sn2)], Sn2])
    print(f"  全部: {len(S_all)} 样本 (A:{len(S2)}, B:{len(S2)})", flush=True)
    s_b = torch.from_numpy(S2).float().to(device)
    a_b = torch.from_numpy(A2).long().to(device)
    model.unit._grow_b = (s_b, a_b)
    n_active = train_batches(model, S_all, A_all, R_all, Sn_all, args.epochs_p2,
                             device=device, grow_arm=args.arm, rng=rng)
    print(f"  生长后激活: {n_active}/512", flush=True)

    torch.save({"model": model.state_dict(), "config": vars(args)},
               f"runs/wm_region_{args.arm}.pt")
    print(f"保存: runs/wm_region_{args.arm}.pt")


if __name__ == "__main__":
    main()
