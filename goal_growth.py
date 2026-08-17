"""goal_growth.py — 目标驱动专精: 神经元生长的专精方向由目标决定?

对比三种生长信号:
  随机生长     (无方向): 基线
  误差驱动     (预测难): 当前框架 — 学"哪里难预测"
  目标驱动     (目标关键): 学"哪里对达成目标重要"

验证: 删生长神经元 → 目标达成 (得分) 下降多少
  + 专精方向: 删生长神经元 → 目标部分崩 (食物获取) vs 非目标部分 (存活)?

环境: 食物收集迷宫 (目标 = 获得食物高分)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from abstract_layer import FoodGame, collect


class WM(nn.Module):
    """世界模型 (符号): 状态 + 动作 → 下一状态 + 奖励。"""
    def __init__(self, obs_dim=3, n_act=5, pool=512, top_k=64, d=32):
        super().__init__()
        self.embed = nn.Linear(obs_dim + n_act, d)
        self.act_mask = torch.zeros(pool, dtype=torch.bool)
        self.act_mask[:128] = True
        self.act_rate = torch.zeros(pool)
        self.growth_log = []
        self.W = nn.Linear(d, pool)
        self.head = nn.Linear(pool, obs_dim)
        self.head_r = nn.Linear(pool, 1)

    def forward(self, obs, act):
        B = obs.shape[0]
        sa = torch.cat([obs, act], -1)
        z = torch.tanh(self.embed(sa))
        pre = self.W(z)
        m = self.act_mask.to(pre.device)
        pre = pre.masked_fill(~m[None], -1e9)
        vals, idx = pre.topk(64, dim=1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(1, idx, F.gelu(vals))
        with torch.no_grad():
            oh = torch.zeros(self.W.out_features, device=pre.device)
            oh[idx[0]] = 1.0
            self.act_rate = 0.999 * self.act_rate.to(pre.device) + 0.001 * oh
        return self.head(sparse), self.head_r(sparse).squeeze(-1), idx

    def grow_at(self, sel, n=2, perturb=0.1):
        dev = sel.device
        inactive = (~self.act_mask).nonzero().flatten().to(dev)
        if len(inactive) == 0:
            return 0
        cnt = torch.zeros(self.W.out_features, device=dev)
        for row in sel.flatten():
            cnt[row] += 1
        am = self.act_mask.to(dev)
        cand = torch.argsort(cnt * am.float(), descending=True)[:n]
        cand = cand[am[cand]][:n]
        with torch.no_grad():
            for src, tgt in zip(cand, inactive[:min(n, len(inactive))]):
                src, tgt = int(src), int(tgt)
                self.W.weight.data[tgt] = self.W.weight.data[src] + perturb * torch.randn_like(self.W.weight.data[src])
                self.head.weight.data[:, tgt] = self.head.weight.data[:, src]
                self.act_mask[tgt] = True
                self.growth_log.append(tgt)
        return len(self.growth_log[-min(n, len(inactive)):])


def run_growth_mode(mode, seed=42):
    """一种生长模式: 训练世界模型 + 定向生长 + 删神经元验证。"""
    torch.manual_seed(seed)
    env = FoodGame()
    S, A, R, Sn = collect(env, 400, seed=seed)
    # 目标关键状态: 食物被吃的时刻 (目标达成) — 用于目标驱动生长
    goal_mask = np.array(R) > 0  # 吃食物 = 目标达成

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    s_t = torch.from_numpy(S).float().to(dev)
    a_t = F.one_hot(torch.from_numpy(A).long(), 5).float().to(dev)
    r_t = torch.from_numpy(R).float().to(dev)
    sn_t = torch.from_numpy(Sn).float().to(dev)
    g_idx = torch.from_numpy(np.where(goal_mask)[0]).to(dev)

    for ep in range(200):
        idx = torch.randperm(len(S))[:2048]
        sp, rp, sel = model(s_t[idx], a_t[idx])
        loss = F.mse_loss(sp, sn_t[idx]) + 0.5 * F.mse_loss(rp, r_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 49 and len(model.growth_log) < 30:
            per_err = (sp - sn_t[idx]).pow(2).mean(-1)
            if mode == "error":      # 误差驱动: 预测难处
                model.grow_at(sel[int(per_err.argmax().item())])
            elif mode == "goal":     # 目标驱动: 目标关键状态 (吃食物处)
                if len(g_idx) > 0:
                    tg = g_idx[torch.randperm(len(g_idx))[:64]]
                    with torch.no_grad():
                        _, _, sel_t = model(s_t[tg], a_t[tg])
                    model.grow_at(sel_t)
            else:                    # 随机
                model.grow_at(sel[torch.randint(0, len(sel), (1,))])

    # 验证: 删生长神经元 → 得分下降 (目标驱动专精 = 得分崩)
    def eval_score(mask_off=None):
        if mask_off:
            with torch.no_grad():
                for nid in mask_off:
                    model.act_mask[nid] = False
        env2 = FoodGame()
        total = 0.0
        for _ in range(60):
            s = env2.reset(); done = False
            while not done:
                z_t = torch.from_numpy(s).float().to(dev).unsqueeze(0).repeat(5, 1)
                acts = torch.eye(5).to(dev)
                with torch.no_grad():
                    sp, rp, _ = model(z_t, acts)
                a = int((0.9 * rp).argmax().item())
                o2, r, dead = env2.step(a)
                total += r; s = o2; done = dead
        return total / 60

    base = eval_score()
    gl = model.growth_log
    drop = eval_score(gl) if gl else base
    rel = (base - drop) / max(abs(base), 1e-6) * 100
    return len(gl), base, drop, rel


if __name__ == "__main__":
    print("=== 目标驱动专精: 生长方向由目标决定? ===")
    for mode in ["random", "error", "goal"]:
        n, base, drop, rel = run_growth_mode(mode)
        print(f"[{mode:6s}] 生长 {n:2d} | 得分 完整 {base:6.1f} → 删生长 {drop:6.1f} "
              f"| 变化 {rel:+6.0f}%")
