"""walker_mem_policy.py — 记忆池策略: 神经元自发学时间规律 (LIF)。

核心原则: 神经元自发涌现学习规律 (top1)
- 输入: 单帧观测 (无窗口, 无手动耦合)
- MemPool (LIF): 膜电位积分 = 时间感 → 交替节奏自发涌现
- 输出: 4 维动作 (两腿协调由神经元学, 非结构先验)
- ES 优化 (无梯度)

复用: mempool.MemPool (LIF 池), walker_safe_zone.SafeZoneEnv
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mempool import MemPool
from walker_safe_zone import SafeZoneEnv


class MemPolicy(nn.Module):
    """记忆池策略: 单帧输入, 膜电位积分 (时间感), 输出动作。"""
    def __init__(self, obs=26, act=4, d=64, pool=256, top_k=32,
                 tau_min=2.0, tau_max=48.0):
        super().__init__()
        self.embed = nn.Linear(obs, d)
        self.pool = MemPool(d, pool, top_k, tau_min, tau_max)
        self.head = nn.Linear(d, act)
        # 运行时膜电位状态
        self._vm = None

    def reset_state(self, batch=1):
        self._vm = torch.zeros(batch, self.pool.d_pool,
                               device=self.embed.weight.device)

    def forward_seq(self, obs_seq):
        """训练/分析: 序列 (B,T,obs) → 动作 (B,T,act)。"""
        B, T = obs_seq.shape[:2]
        z = torch.tanh(self.embed(obs_seq))
        z_pool, sel = self.pool(z)
        return torch.tanh(self.head(z_pool))

    def act(self, obs, noise=0.0):
        """单步: 保持膜电位 (时间感连续)。"""
        dev = self.embed.weight.device
        if self._vm is None:
            self.reset_state()
        o = torch.from_numpy(obs).float().to(dev).unsqueeze(0)
        z = torch.tanh(self.embed(o))
        tau = self.pool.tau().unsqueeze(0)
        leak = 1.0 - 1.0 / tau
        self._vm = self._vm * leak + z @ self.pool.W_in.t()
        am = self.pool.active_mask.to(dev)
        pre = self._vm.masked_fill(~am.unsqueeze(0), -1e9)
        vals, idx = pre.topk(self.pool.top_k, dim=1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(1, idx, F.gelu(vals))
        out = torch.tanh(self.head(sparse @ self.pool.W_out.t()))
        if noise:
            out = out + noise * torch.randn_like(out)
        return np.clip(out[0].detach().cpu().numpy(), -1, 1).astype(np.float32)


def eval_ep(policy, env, max_steps=1600, noise=0.0, seed=0):
    s = env.reset()
    policy.reset_state()
    total_r, dist, safe_n = 0.0, 0.0, 0
    alt_n = 0
    for _ in range(max_steps):
        a = policy.act(s, noise=noise)
        o2, r, d = env.step(a)
        total_r += r; dist = env.dist
        safe_n += env._hull_y() >= env.h_safe
        alt_n += a[0] * a[2] < 0
        s = o2
        if d:
            break
    return total_r, env.t, dist, safe_n, alt_n


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 记忆池策略: 神经元自发学时间 (LIF, 无窗口/无手动耦合) ===")
    env = SafeZoneEnv(h_safe=5.6)
    policy = MemPolicy().to(dev)
    lr, sigma = 0.3, 0.15

    r0, t0, d0, s0, a0 = eval_ep(policy, env)
    print(f"初始: 存活 {t0}, 安全 {s0}/{t0}, 交替 {a0}/{t0}, 前进 {d0:.0f}")

    for it in range(30):
        deltas, scores = [], []
        for i in range(10):
            delta = []
            for p in policy.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d); p.data.add_(d)
            sc, t, dist, safe, alt = eval_ep(policy, env)
            for p, d in zip(policy.parameters(), delta): p.data.sub_(d)
            deltas.append(delta); scores.append(sc)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(policy.parameters(), delta):
                p.data.add_(lr * wi / (10 * sigma) * d)
        if it % 5 == 4:
            sc, t, dist, safe, alt = eval_ep(policy, env)
            print(f"  iter {it+1}: 存活 {t}, 安全 {safe}/{t}, 交替 {alt}/{t}, 前进 {dist:.0f}")

    # 最终
    ts, ss, as_, ds = [], [], [], []
    for _ in range(6):
        e2 = SafeZoneEnv(h_safe=5.6)
        _, t, dist, safe, alt = eval_ep(policy, e2)
        ts.append(t); ss.append(safe); as_.append(alt); ds.append(dist)
    print(f"\n最终: 存活 {np.mean(ts):.0f}, 安全 {np.mean(ss)/np.mean(ts)*100:.0f}%, "
          f"交替 {np.mean(as_)/np.mean(ts)*100:.0f}%, 前进 {np.mean(ds):.0f}")
    torch.save(policy.state_dict(), "runs/walker_mem_policy.pt")
    print(f"{'✅ 神经元自发时间感 (交替涌现)' if np.mean(as_)/np.mean(ts) > 0.5 else '⚠️ 有限'}")


if __name__ == "__main__":
    main()
