"""walker_kuramoto.py — Kuramoto 耦合振荡器步态 (复用现有模块, 不修改)。

Un-0 启发: 耦合振荡器作为计算原语。
双腿 = 两个耦合振荡器 (反相锁相 φ₁-φ₂=π):
  dφ₁/dt = ω + K·sin(φ₂-φ₁) + B(姿态)    (腿1)
  dφ₂/dt = ω + K·sin(φ₁-φ₂) + B(姿态)    (腿2)
  K 耦合 → 反相锁定 (受扰自动回到 π — 抗扰动)
  B(姿态) = 平衡反馈 (hull 倾斜 → 相位/动作调制)
  接触耦合 → 触地微调相位 (感觉耦合)

动作 = 相位曲线 (hip/knee sin) × 幅度 + 平衡矫正
ES 优化振荡器参数 (ω, K, 幅度, offset, 平衡增益)
复用: WalkerEnergyEnv (能量+驱动规则), 不修改现有文件。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from walker_energy_env import WalkerEnergyEnv


class KuramotoGait(nn.Module):
    """Kuramoto 步态振荡器: 两腿耦合 + 平衡反馈 + 接触耦合。

    可进化参数: ω(频率), K(耦合), A1/A2(幅度), off(膝偏移),
                g_bal(平衡增益), g_contact(接触耦合)。
    """
    def __init__(self):
        super().__init__()
        # 可进化参数 (ES 扰动)
        self.params = nn.Parameter(torch.tensor([
            0.25,   # ω: 步态频率
            0.6,    # K: 反相耦合强度
            0.6,    # A_hip: 髋幅度
            0.4,    # A_knee: 膝幅度
            1.2,    # 膝相位偏移
            0.8,    # g_bal: 平衡反馈增益
            0.2,    # g_contact: 接触耦合
            0.3,    # bal_amp: 平衡矫正幅度
        ]))

    def step(self, obs, ph1, ph2, contact1, contact2, dt=1.0):
        """一步: 更新相位 → 返回动作。obs: (26,)。"""
        w, K, A1, A2, off, gb, gc, ba = self.params.detach().cpu().numpy()
        w = max(0.05, abs(w))
        # Kuramoto 耦合: 反相锁定 (φ₁-φ₂ → π)
        d1 = w + K * np.sin(ph2 - ph1) + gb * obs[1] * 0.3
        d2 = w + K * np.sin(ph1 - ph2) + gb * obs[1] * 0.3
        # 接触耦合: 触地时锁定相位 (感觉耦合)
        if contact1:
            d1 += gc * (np.pi - ph1) * 0.5
        if contact2:
            d2 += gc * (np.pi - ph2) * 0.5
        ph1 += d1 * dt
        ph2 += d2 * dt
        # 步态曲线 (相位 → 关节)
        hip1 = A1 * np.sin(ph1)
        knee1 = A2 * np.sin(ph1 + off)
        hip2 = A1 * np.sin(ph2)
        knee2 = A2 * np.sin(ph2 + off)
        # 平衡矫正 (hull 倾斜 → 反推)
        bal = ba * np.clip(obs[0] * 1.5, -1, 1)
        return (np.array([hip1 + bal, knee1, hip2 - bal, knee2],
                         np.float32), ph1, ph2)

    def init_phase(self):
        return 0.0, np.pi  # 反相启动


def eval_ep(gait, env, max_steps=1600, noise=0.0, seed=0):
    s = env.reset()
    ph1, ph2 = gait.init_phase()
    total_r, dist = 0.0, 0.0
    for _ in range(max_steps):
        c1 = s[8] > 0.5; c2 = s[10] > 0.5
        a, ph1, ph2 = gait.step(s, ph1, ph2, c1, c2)
        if noise:
            a = a + noise * np.random.randn(4).astype(np.float32)
        o2, r, d = env.step(np.clip(a, -1, 1))
        total_r += r; dist = env.dist
        s = o2
        if d:
            break
    return total_r, env.t, dist


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== Kuramoto 耦合振荡器步态 (双腿反相 + 平衡反馈) ===")
    gait = KuramotoGait().to(dev)
    env = WalkerEnergyEnv()
    lr, sigma = 0.3, 0.15

    r0, t0, d0 = eval_ep(gait, env)
    print(f"初始: 存活 {t0}, 前进 {d0:.0f}")

    for it in range(30):
        deltas, scores, dists = [], [], []
        for i in range(12):
            delta = torch.randn_like(gait.params) * sigma
            gait.params.data.add_(delta)
            sc, t, dist = eval_ep(gait, env)
            gait.params.data.sub_(delta)
            deltas.append(delta); scores.append(sc); dists.append(dist)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            gait.params.data.add_(lr * wi / (12 * sigma) * delta)
        if it % 5 == 4:
            sc, t, dist = eval_ep(gait, env)
            print(f"  iter {it+1}: 得分 {sc:+.1f} | 存活 {t} | 前进 {dist:.0f}")

    times, dists, heights = [], [], []
    for _ in range(6):
        e2 = WalkerEnergyEnv()
        _, t, dist = eval_ep(gait, e2)
        times.append(t); dists.append(dist)
    print(f"\n最终: 存活 {np.mean(times):.0f}, 前进 {np.mean(dists):.0f}")
    print(f"参数: {gait.params.detach().cpu().numpy().round(2)}")
    print(f"{'✅ Kuramoto 步态有效' if np.mean(dists) > 25 else '⚠️ 有限'}")


if __name__ == "__main__":
    main()
