"""rhythm_growth.py — 节奏/心跳调制生长 (活动依赖神经发生)。

设计: 生长 = 目标事件 × 节奏相位窗口, 速率 ∝ 活动强度
对照: 目标驱动生长 (无节奏) vs 目标驱动 + 节奏调制
验证: 生长神经元功能性 (删了 → 得分变化) + 生长效率
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abstract_layer import FoodGame
from goal_growth import WM, run_growth_mode


class Heartbeat:
    """状态耦合心跳: 频率/幅度 = f(活动强度)。"""
    def __init__(self, base=0.05):
        self.ph = 0.0
        self.base = base

    def tick(self, activity):
        # 活动 → 频率 (运动越强心跳越快)
        freq = self.base * (1 + 2.0 * activity)
        self.ph = (self.ph + freq) % 1.0
        return self.ph

    def window(self, phase, width=0.3):
        """生长窗口: 只在相位 < width 时允许生长。"""
        return phase < width


def collect_activity(env, n_eps=150, seed=0):
    """收集带活动强度的经验 (移动 = 活动)。"""
    rng = np.random.RandomState(seed)
    acts = []
    for _ in range(n_eps):
        s = env.reset(); done = False
        while not done:
            a = int(rng.randint(5))
            o2, r, d = env.step(a)
            acts.append((s, a, r, o2, float(a != 0)))  # 活动 = 移动
            s = o2; done = d
    return acts


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== 节奏/心跳调制生长 (活动依赖神经发生) ===")
    # 快速: 用 goal_growth 的训练, 但生长时机加节奏
    for mode in ["goal", "goal_rhythm"]:
        torch.manual_seed(42)
        env = FoodGame()
        S, A, R, Sn = [], [], [], []
        acts_data = collect_activity(env, 300)
        for s, a, r, o2, act in acts_data:
            S.append(s); A.append(a); R.append(r); Sn.append(o2)
        S = np.array(S, np.float32); A = np.array(A, np.int64)
        R = np.array(R, np.float32); Sn = np.array(Sn, np.float32)
        goal_mask = np.array(R) > 0

        model = WM().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        s_t = torch.from_numpy(S).float().to(dev)
        a_t = F.one_hot(torch.from_numpy(A).long(), 5).float().to(dev)
        r_t = torch.from_numpy(R).float().to(dev)
        sn_t = torch.from_numpy(Sn).float().to(dev)
        g_idx = torch.from_numpy(np.where(goal_mask)[0]).to(dev)
        hb = Heartbeat()

        for ep in range(150):
            idx = torch.randperm(len(S))[:2048]
            sp, rp, sel = model(s_t[idx], a_t[idx])
            loss = F.mse_loss(sp, sn_t[idx]) + 0.5 * F.mse_loss(rp, r_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 50 == 49 and len(model.growth_log) < 30:
                if mode == "goal":
                    tg = g_idx[torch.randperm(len(g_idx))[:64]]
                    with torch.no_grad():
                        _, _, sel_t = model(s_t[tg], a_t[tg])
                    model.grow_at(sel_t)
                else:  # goal_rhythm: 目标 × 节奏窗口
                    # 心跳相位窗口内才生长 (活动强度高 → 窗口更宽)
                    act = np.array([t[4] for t in acts_data])
                    act_hi = act[np.where(goal_mask)[0]]
                    window_ok = hb.window(hb.tick(0.5), width=0.5)
                    if window_ok:
                        tg = g_idx[torch.randperm(len(g_idx))[:64]]
                        with torch.no_grad():
                            _, _, sel_t = model(s_t[tg], a_t[tg])
                        model.grow_at(sel_t, n=2)

        # 验证: 删生长 → 得分
        gl = model.growth_log
        def score():
            env2 = FoodGame()
            total = 0.0
            for _ in range(40):
                s = env2.reset(); done = False
                while not done:
                    z = torch.from_numpy(s).float().to(dev).unsqueeze(0).repeat(5, 1)
                    with torch.no_grad():
                        sp, rp, _ = model(z, torch.eye(5).to(dev))
                    a = int((0.9 * rp).argmax().item())
                    o2, r, d = env2.step(a)
                    total += r; s = o2; done = d
            return total / 40
        base = score()
        if gl:
            with torch.no_grad():
                for nid in gl: model.act_mask[nid] = False
            drop = score()
            rel = (base - drop) / max(abs(base), 1e-6) * 100
        else:
            rel = 0.0
        print(f"[{mode:12s}] 生长 {len(gl):2d} | 删后得分 {rel:+.0f}% "
              f"{'= 功能性' if rel > 10 else '= 中性'} | 完整 {base:.1f}")


if __name__ == "__main__":
    main()
