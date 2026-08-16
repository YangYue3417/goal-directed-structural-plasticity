"""cycle_task.py — 周期状态机任务: 神经元短期记忆 → 状态转换能力。

任务: 环境状态在 S0 ↔ S1 之间按周期 T 切换 (状态机)。
观测含当前状态, 但**不含相位** (在 S0 待了多久不可见)。
→ 预测"下一步状态/切换时刻"必须靠记忆 (前几个观测的电位残留)。

对照:
  记忆池 (MemPool, LIF)  vs  静态池 (StaticPool, 无记忆)
  两者同生长/淘汰机制。

验证:
  ① 记忆池学会周期 (切换时刻预测误差低), 静态池学不到
  ② 生长神经元 (难样本定向, 落在切换时刻) → 删了 → 切换时刻预测崩
  ③ τ 分工: 生长神经元继承的 τ 分布 (快/慢)
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

from mempool import MemPool, StaticPool


class CycleWM(nn.Module):
    """周期任务世界模型: 观测嵌入 → 池 → 预测下一观测。"""
    def __init__(self, obs_dim, pool_type="mem", pool=512, top_k=64,
                 d=64, hidden=64, tau_min=2.0, tau_max=48.0):
        super().__init__()
        self.embed = nn.Linear(obs_dim, d)
        if pool_type == "mem":
            self.pool = MemPool(d, pool, top_k, tau_min, tau_max)
        else:
            self.pool = StaticPool(d, pool, top_k)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, obs_dim)

    def forward(self, obs_seq):
        B, T, D = obs_seq.shape
        z = torch.tanh(self.embed(obs_seq))
        z_pool, sel = self.pool(z)
        h = self.net(z_pool)
        return self.head(h), sel


def gen_cycle_seq(T_switch=8, n_seg=400, L=16, seed=0):
    """生成周期状态机序列: 状态 0/1 每 T_switch 步切换。

    观测 = 当前状态 (0/1) + 一个随相位变化的小信号? 否 — 纯状态。
    预测目标 = 下一步观测 (含切换时刻)。
    相位不可见 → 预测切换必须靠记忆。
    """
    rng = np.random.RandomState(seed)
    seqs = []
    for _ in range(n_seg):
        s = 0
        phase = 0
        seg = []
        for _ in range(L):
            seg.append(float(s))
            phase += 1
            if phase >= T_switch:
                s = 1 - s
                phase = 0
        seqs.append(seg)
    X = np.array(seqs, np.float32).reshape(-1, L, 1)   # (n, L, 1)
    Y = np.roll(X, -1, axis=1)                          # 预测下一步
    Y[:, -1] = X[:, -1]
    return torch.from_numpy(X), torch.from_numpy(Y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=8, help="切换周期")
    p.add_argument("--L", type=int, default=16, help="片段长度")
    p.add_argument("--n_seg", type=int, default=400)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    X, Y = gen_cycle_seq(args.T, args.n_seg, args.L)
    X, Y = X.to(device), Y.to(device)
    # 切换时刻 mask (验证用)
    switch = (X[:, 1:] != X[:, :-1]).any(-1)   # (n, L-1) 当前步后发生切换
    switch = torch.cat([switch, torch.zeros_like(switch[:, :1])], 1)

    print(f"=== 周期任务 T={args.T}: 记忆池 vs 静态池 ===")
    results = {}
    for ptype in ["mem", "static"]:
        torch.manual_seed(42)
        model = CycleWM(1, ptype, pool=512, top_k=64).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        n_grow_total = 0
        for ep in range(args.epochs):
            model.train()
            idx = torch.randperm(len(X))[:256]
            sp, sel = model(X[idx])
            loss = F.mse_loss(sp, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 100 == 99 and len(model.pool.growth_log) < 40:
                per_err = (sp - Y[idx]).pow(2).mean(-1).mean(0)  # (L,)
                t_hard = int(per_err.argmax())
                n_grow_total += model.pool.grow(sel[:, t_hard])

        # 评估: 总误差 + 切换时刻误差 + 非切换误差
        model.eval()
        sp, _ = model(X)
        err = (sp - Y).pow(2).mean(-1)                     # (n, L)
        err_switch = err[switch].mean().item()
        err_stable = err[~switch].mean().item()
        err_all = err.mean().item()

        # 删生长神经元 → 切换时刻误差变化
        grow_idx = model.pool.growth_log
        if grow_idx:
            with torch.no_grad():
                for nid in grow_idx:
                    model.pool.active_mask[nid] = False
            sp2, _ = model(X)
            err2 = (sp2 - Y).pow(2).mean(-1)
            d_switch = (err2[switch].mean().item() - err_switch) / max(err_switch, 1e-9) * 100
            d_stable = (err2[~switch].mean().item() - err_stable) / max(err_stable, 1e-9) * 100
        else:
            d_switch = d_stable = 0.0
        results[ptype] = (err_all, err_switch, err_stable, len(grow_idx), d_switch, d_stable)
        print(f"[{ptype}] 误差: 总 {err_all:.4f} | 切换 {err_switch:.4f} | "
              f"稳定 {err_stable:.4f}")
        print(f"       生长 {len(grow_idx)} | 删后: 切换 {d_switch:+.0f}% | "
              f"稳定 {d_stable:+.0f}%")

    mem = results["mem"]; st = results["static"]
    print(f"\n=== 结论 ===")
    print(f"切换预测: 记忆池 {mem[1]:.4f} vs 静态池 {st[1]:.4f} "
          f"({(1-mem[1]/max(st[1],1e-9))*100:.0f}% 改善)")
    print(f"生长功能: 记忆池删神经元 → 切换 {mem[4]:+.0f}% | "
          f"静态池 → 切换 {st[4]:+.0f}%")


if __name__ == "__main__":
    main()
