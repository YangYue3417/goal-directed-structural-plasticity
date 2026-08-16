"""discrim_experiment.py — 核心 claim 判别实验。

Claim: 生长的功能价值 = f(基础模型能否分辨"真正的困难")
  - 结构难 (数据充足, 模型系统性错): 难样本=真难 → 生长有效 (删神经元伤该区域)
  - 噪声难 (数据稀疏, 模型偶然错):   难样本=噪声 → 生长有害/无效

控制: 同一迷宫, 同一"难区域" (左半 x<5), 只改难区域数据量:
  Cond 1 (结构难): 难区域样本充足 → 模型在难区域一致犯错
  Cond 2 (噪声难): 难区域样本稀疏 → 模型在难区域偶然犯错

验证: 难样本定向生长 → 删生长神经元 → 难区域 vs 易区域 预测误差
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
from envs.survival_maze import SurvivalMaze


class WM(nn.Module):
    """符号世界模型 (简化: 全连接 + 池)。"""
    def __init__(self, obs_dim=14, act_dim=3, pool=512, top_k=64):
        super().__init__()
        self.embed = nn.Linear(obs_dim + act_dim, 64)
        self.unit = nn.Linear(64, pool)          # 神经元层 (可生长: 克隆行)
        self.act_mask = torch.ones(pool, dtype=torch.bool)  # 初始全激活? 用池子
        self.act_mask[:128] = True
        self.act_mask[128:] = False
        self.act_rate = torch.zeros(pool)
        self.growth_log = []
        self.head = nn.Linear(pool, obs_dim)
        self.head_r = nn.Linear(pool, 1)

    def forward(self, obs, act, mask=None):
        sa = torch.cat([obs, act], -1)
        z = torch.tanh(self.embed(sa))
        pre = self.unit(z)
        m = self.act_mask.to(pre.device)
        pre = pre.masked_fill(~m[None], -1e9)
        vals, idx = pre.topk(64, dim=1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(1, idx, vals)
        act_one = F.gelu(sparse)
        # 更新激活率
        with torch.no_grad():
            oh = torch.zeros(self.unit.weight.shape[0], device=pre.device)
            oh[idx[0]] = 1.0
            self.act_rate = 0.999 * self.act_rate.to(pre.device) + 0.001 * oh
        return self.head(act_one), self.head_r(act_one).squeeze(-1), idx


def grow_hard_sym(model, sel_hard, perturb=0.1, n=2):
    inactive = (~model.act_mask).nonzero().flatten()
    if len(inactive) == 0:
        return 0
    cnt = torch.zeros(model.unit.weight.shape[0])
    for row in sel_hard:
        cnt[row] += 1
    cand = torch.argsort(cnt * model.act_mask.float(), descending=True)[:n]
    cand = cand[model.act_mask[cand]][:n]
    n_grow = min(len(cand), len(inactive))
    with torch.no_grad():
        for src, tgt in zip(cand, inactive[:n_grow]):
            model.unit.weight.data[tgt] = model.unit.weight.data[src] + perturb * torch.randn_like(model.unit.weight.data[src])
            model.unit.bias.data[tgt] = model.unit.bias.data[src]
            model.head.weight.data[:, tgt] = model.head.weight.data[:, src]
            model.act_mask[tgt] = True
            model.growth_log.append(int(tgt))
    return n_grow


def collect(env, n_eps, cond, seed=0, hard_ratio=0.1):
    """收集轨迹。cond='noise': 难区域样本稀疏 (过滤)。"""
    rng = np.random.RandomState(seed)
    S, A, R, Sn, P = [], [], [], [], []
    for _ in range(n_eps):
        obs = env.reset()
        for _ in range(60):
            a = int(rng.randint(3))
            o2, r, d = env.step(a)
            in_hard = env.x / env.size < 0.5
            if cond == "noise" and in_hard and rng.rand() > hard_ratio:
                pass  # 难区域样本稀疏
            else:
                S.append(obs); A.append(a); R.append(r); Sn.append(o2)
                P.append(in_hard)
            obs = o2
            if d:
                break
    return (np.array(S, np.float32), np.array(A, np.int64),
            np.array(R, np.float32), np.array(Sn, np.float32), np.array(P, np.bool_))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n_eps", type=int, default=1500)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    env = SurvivalMaze(**cfg.SURVIVAL_ENV)
    env._day_seed = 777; env.energy = env.E0; env.reset_day()

    print("=== 判别实验: 结构难 vs 噪声难 → 生长效果 ===")
    for cond in ["struct", "noise"]:
        torch.manual_seed(42)
        S, A, R, Sn, Phard = collect(env, args.n_eps, cond)
        model = WM().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        s_t = torch.from_numpy(S).float().to(device)
        a_t = F.one_hot(torch.from_numpy(A).long(), 3).float().to(device)
        r_t = torch.from_numpy(R).float().to(device)
        sn_t = torch.from_numpy(Sn).float().to(device)
        for ep in range(args.epochs):
            model.train()
            idx = torch.randperm(len(S))[:4096]
            sp, rp, sel = model(s_t[idx], a_t[idx])
            loss = F.mse_loss(sp, sn_t[idx]) + 0.5 * F.mse_loss(rp, r_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 50 == 49 and len(model.growth_log) < 30:
                per_err = (sp - sn_t[idx]).pow(2).mean(-1)
                grow_hard_sym(model, sel[int(per_err.argmax().item())])
        # 验证: 删生长神经元 → 难/易区域误差
        def err_by_region():
            errs = {}
            for region, m_idx in [("难区域", Phard), ("易区域", ~Phard)]:
                if m_idx.sum() == 0:
                    continue
                idx = np.where(m_idx)[0][:1500]
                sp, rp, _ = model(s_t[idx], a_t[idx])
                errs[region] = F.mse_loss(sp, sn_t[idx]).item()
            return errs
        e0 = err_by_region()
        with torch.no_grad():
            for nid in model.growth_log:
                model.act_mask[nid] = False
        e1 = err_by_region()
        delta = {k: (e1[k]-e0[k])/max(e0[k],1e-9)*100 for k in e0}
        print(f"[{cond}] 生长{len(model.growth_log)}个 | 删后误差变化: "
              f"难区域 {delta.get('难区域',0):+.0f}% | 易区域 {delta.get('易区域',0):+.0f}%")


if __name__ == "__main__":
    main()
