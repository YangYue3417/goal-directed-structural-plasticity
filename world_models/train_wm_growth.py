"""train_wm_growth.py — 生长三臂: 自行发育神经元是否有专精功能?

中心实验 (回到主线):
  池容量受限 (512 池, 25% 激活 = 128) → 神经元没空间专精 (0.207 解码)
  生长: 误差高时募集新神经元 (克隆高负载 + 扰动) → 池扩容

三臂:
  func: 误差驱动 (误差 EMA 超阈值 → 克隆高负载神经元)
  rand: 随机募集 (同数量, 随机位置克隆)
  none: 无生长 (容量不变)

验证:
  ① 池神经元位置解码: 生长后是否 > 无生长 (专精提升)
  ② 删神经元因果: 删 G 神经元 → 预测误差上升 (func 应 >> rand)
  ③ 生长位置: func 是否更集中在高误差区域
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from envs.survival_maze import SurvivalMaze
from world_models.train_wm_explore import WorldExplore, collect_trajectories, OBS


def grow_neurons(model, arm, n=2, perturb=0.05, rng=None):
    """募集 n 个神经元。func: 克隆高负载活跃; rand: 随机克隆。"""
    unit = model.unit
    inactive = (~unit.active_mask).nonzero().flatten()
    if len(inactive) == 0:
        return 0
    active = unit.active_mask.nonzero().flatten()
    if arm == "func":
        loads = unit._load_ema[active]
        cand = active[loads.argsort(descending=True)[:n]]
    else:  # rand
        cand = active[rng.choice(len(active), min(n, len(active)), replace=False)]
    n_grow = min(len(cand), len(inactive))
    with torch.no_grad():
        for src, tgt in zip(cand, inactive[:n_grow]):
            unit.W1.data[:, tgt] = unit.W1.data[:, src] + perturb * torch.randn_like(unit.W1.data[:, src])
            unit.W2.data[tgt, :] = unit.W2.data[src, :] + perturb * torch.randn_like(unit.W2.data[src, :])
            unit.b1.data[tgt] = unit.b1.data[src]
            unit.active_mask[tgt] = True
    return n_grow


def eval_pool_decode(model, S, A, P, device, n_sample=20000):
    """池神经元激活 → 位置解码。"""
    model.eval()
    idx = np.random.RandomState(0).choice(len(S), n_sample, replace=False)
    acts, ys = [], []
    with torch.no_grad():
        for i in range(0, n_sample, 512):
            sb = torch.from_numpy(S[idx[i:i + 512]]).float().to(device)
            ab = torch.from_numpy(A[idx[i:i + 512]]).long().to(device)
            emb = torch.tanh(model.embed(torch.cat([sb, F.one_hot(ab, 3).float()], -1)))
            _, ps = model.unit(emb.unsqueeze(1))
            # 每样本激活向量 (top-k 索引散射, 512 维)
            sel = ps.selected.cpu().numpy()[:, 0, :]  # (B, k)
            B = len(sel)
            act = np.zeros((B, model.unit.d_pool))
            np.put_along_axis(act, sel, 1.0, axis=1)
            acts.append(act)
            ys.append(P[idx[i:i + 512]])
    X = np.concatenate(acts)
    y = np.concatenate(ys)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(X))
    tr, va = perm[:int(len(X) * 0.7)], perm[int(len(X) * 0.7):]
    lam = 1e-3 * X.shape[1]
    W = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ y[tr])
    pred = X[va] @ W
    return float(np.mean(np.abs(pred - y[va]).max(1) < 0.05))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", type=str, choices=["func", "rand", "none"], default="func")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--n_episodes", type=int, default=3000)
    p.add_argument("--grow_interval", type=int, default=100)
    p.add_argument("--grow_n", type=int, default=2)
    p.add_argument("--err_thresh", type=float, default=0.02,
                   help="误差 EMA 阈值 (超过才生长)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.RandomState(args.seed)

    env = SurvivalMaze(size=20, n_foods=3, seed=42)
    S, A, R, Sn, D, P = collect_trajectories(env, args.n_episodes)
    n = len(S)
    s_t = torch.from_numpy(S).float().to(device)
    a_t = torch.from_numpy(A).long().to(device)
    r_t = torch.from_numpy(R).float().to(device)
    sn_t = torch.from_numpy(Sn).float().to(device)
    n_win = n // args.T
    idx_w = np.random.RandomState(0).permutation(n_win)
    split = int(n_win * 0.8)

    model = WorldExplore().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # 池负载 EMA (生长信号)
    unit = model.unit
    unit._load_ema = torch.zeros(unit.d_pool, device=device)
    err_ema = 0.0
    grown = []

    print(f"生长臂: {args.arm} | 池 512 (初始 128 激活)", flush=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for i in range(0, split, args.bs):
            sel = np.random.choice(split, args.bs, replace=False)
            idx = np.concatenate([np.arange(w * args.T, (w + 1) * args.T) for w in sel])
            sb = s_t[idx].view(args.bs, args.T, OBS)
            ab = a_t[idx].view(args.bs, args.T)
            rb = r_t[idx].view(args.bs, args.T)
            snb = sn_t[idx].view(args.bs, args.T, OBS)
            op, rp, _, ps = model(sb, ab)
            loss = F.mse_loss(op, snb) + 0.2 * F.mse_loss(rp, rb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            # 负载 EMA (每步)
            with torch.no_grad():
                unit._load_ema = 0.99 * unit._load_ema + 0.01 * ps.load.mean(0)
            # 生长 (误差驱动: 误差 EMA 超阈值才长)
            err = loss.item()
            err_ema = 0.99 * err_ema + 0.01 * err
            step = ep * (split // args.bs) + i // args.bs
            if step % args.grow_interval == 0 and args.arm != "none":
                if args.arm == "func" and err_ema < args.err_thresh:
                    continue
                g = grow_neurons(model, args.arm, args.grow_n, rng=rng)
                if g:
                    grown.append(step)
        if ep % 5 == 4:
            acc = eval_pool_decode(model, S, A, P, device)
            n_act = int(unit.active_mask.sum())
            print(f"  ep {ep+1}: loss={tot/split:.4f} | 池激活={n_act}/512 "
                  f"池解码acc={acc:.3f} 生长事件={len(grown)}", flush=True)

    torch.save({"model": model.state_dict(), "grown": grown,
                "config": vars(args)}, f"runs/wm_growth_{args.arm}.pt")
    print(f"保存: runs/wm_growth_{args.arm}.pt (生长 {len(grown)} 事件, "
          f"激活 {int(unit.active_mask.sum())}/512)")


if __name__ == "__main__":
    main()
