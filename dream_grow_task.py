"""dream_grow_task.py — 验证: 训练塑造方向 + 做梦生长。

机制: 
  白天训练: 活跃池主预测 + 储备池 shadow 预测 (暗中塑造方向)
  做梦生长: 重放难样本 → 激活对难样本 shadow 预测最好的储备神经元
  对比: 训练期生长 (直接克隆) vs 做梦期生长 (有方向)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from mempool import MemPool
from cycle_task import CycleWM, gen_cycle_seq


class ShadowWM(nn.Module):
    """带 shadow 储备的周期世界模型。"""
    def __init__(self, obs_dim=1, pool=512, top_k=64, d=64, hidden=64,
                 tau_min=2.0, tau_max=48.0, shadow_w=0.5):
        super().__init__()
        self.shadow_w = shadow_w
        self.embed = nn.Linear(obs_dim, d)
        self.pool = MemPool(d, pool, top_k, tau_min, tau_max)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, obs_dim)

    def forward(self, obs_seq, with_shadow=True):
        B, T = obs_seq.shape[:2]
        z = torch.tanh(self.embed(obs_seq))
        z_pool, sel = self.pool(z)
        h = self.net(z_pool)
        out = self.head(h)
        shadow_out = None
        if with_shadow and self.pool.shadow_stack is not None:
            sh = self.pool.shadow_stack              # (B, T, reserve)
            sh_pad = torch.zeros(
                B, T, self.pool.d_pool, device=sh.device)
            sh_pad[:, :, self.pool.reserve_idx] = sh
            h_sh = self.net(sh_pad @ self.pool.W_out.t())  # 投影回 d
            shadow_out = self.head(h_sh)
        return out, sel, shadow_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = gen_cycle_seq(8, 400, 16)
    X, Y = X.to(device), Y.to(device)

    print("=== 做梦生长 vs 训练期生长 (周期任务) ===")
    for mode in ["dream", "train"]:
        torch.manual_seed(42)
        m = ShadowWM().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        for ep in range(800):
            m.train()
            idx = torch.randperm(len(X))[:256]
            sp, sel, sp_sh = m(X[idx])
            loss = F.mse_loss(sp, Y[idx])
            if sp_sh is not None:
                # shadow 预测也学 Δ (方向塑造), 但目标 = 全量 Y
                loss = loss + m.shadow_w * F.mse_loss(sp_sh, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 100 == 99:
                per_err = (sp - Y[idx]).pow(2).mean(-1).mean(0)
                t_hard = int(per_err.argmax())
                if mode == "train":
                    m.pool.grow(sel[:, t_hard])
                else:
                    # 做梦生长: 用 shadow top-k 激活的储备神经元
                    sh = m.pool.shadow_stack              # (B, T, R)
                    r_err = (sp_sh - Y[idx]).pow(2).mean(-1)  # (B, T)
                    _, t_h = r_err[:, t_hard].topk(8)
                    cand_rows = sh[t_h, t_hard].argmax(-1)
                    m.pool.dream_grow(cand_rows)
                m.pool.settle_babies(age_thresh=800.0, rate_thresh=0.01)

        m.eval()
        sp, _, _ = m(X)
        err = (sp - Y).pow(2).mean(-1)
        sw = torch.cat([(X[:, 1:] != X[:, :-1]).any(-1),
                        torch.zeros(X.shape[0], 1, dtype=torch.bool, device=X.device)], 1)
        e_sw = err[sw].mean().item()
        e_st = err[~sw].mean().item()
        # 删生长神经元
        gl = m.pool.growth_log
        if gl:
            with torch.no_grad():
                for nid in gl:
                    m.pool.active_mask[nid] = False
            sp2, _, _ = m(X)
            err2 = (sp2 - Y).pow(2).mean(-1)
            d_sw = (err2[sw].mean().item() - e_sw) / max(e_sw, 1e-9) * 100
        else:
            d_sw = 0.0
        print(f"[{mode}] 生长 {len(gl)} | 切换 {e_sw:.4f} | 稳定 {e_st:.4f} "
              f"| 删后切换 {d_sw:+.0f}%")


if __name__ == "__main__":
    main()
