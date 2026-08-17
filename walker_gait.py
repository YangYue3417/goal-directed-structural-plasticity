"""walker_gait.py — Walker 走起来: 感觉耦合相位状态机 (局部单元)。

"局部"的定义:
  每条腿 4 相位状态机: 支撑早→支撑晚→摆动屈→摆动伸
  两腿反相 (左摆右撑), 触地/离地触发切换 (感觉耦合)
  决策单元 = 每交替周期选参数 (幅度/偏置) — ES 优化

支撑相: 腿伸直推地 (hip 后摆, knee 伸) — 推进
摆动相: 腿屈曲抬起前移 (hip 前摆, knee 屈→伸) — 跨步
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from walker_energy_env import WalkerEnergyEnv


class GaitPhase:
    """单腿相位状态机 (接触触发切换)。"""
    STANCE_EARLY, STANCE_LATE, SWING_FLEX, SWING_EXTEND = 0, 1, 2, 3

    def __init__(self, A=0.8):
        self.A = A
        self.ph = self.STANCE_EARLY
        self.t_in_ph = 0

    def action(self):
        """当前相位动作 (hip, knee) — 走路的基本运动学。"""
        A = self.A
        if self.ph == self.STANCE_EARLY:
            return np.array([-0.6 * A, 0.3], np.float32)   # 腿后摆推地
        elif self.ph == self.STANCE_LATE:
            return np.array([0.3 * A, 0.2], np.float32)    # 重心前移
        elif self.ph == self.SWING_FLEX:
            return np.array([0.8 * A, -0.9], np.float32)   # 屈膝抬腿前摆
        else:  # SWING_EXTEND
            return np.array([0.2 * A, 0.7], np.float32)    # 伸腿准备触地

    def update(self, contact, dt_max=12):
        """接触耦合切换: 触地→支撑, 离地→摆动。"""
        self.t_in_ph += 1
        if self.ph in (self.STANCE_EARLY, self.STANCE_LATE):
            # 支撑相: 离地 → 摆动; 支撑过长强制切
            if not contact or self.t_in_ph > dt_max:
                self.ph = (self.ph + 1) % 4
                self.t_in_ph = 0
        else:
            # 摆动相: 触地 → 支撑; 摆动过长强制切
            if contact or self.t_in_ph > dt_max:
                self.ph = (self.ph + 1) % 4
                self.t_in_ph = 0
        return self.ph


class GaitPolicy(nn.Module):
    """策略: 观测 → 交替参数 (每周期决策)。"""
    def __init__(self, obs=26, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, d), nn.ReLU(),
                                 nn.Linear(d, 3))
        self.scale = torch.tensor([0.5, 0.4, 20.0])
        self.offset = torch.tensor([0.5, 0.5, 8.0])

    def forward(self, s):
        return torch.sigmoid(self.net(s)) * self.scale.to(s.device) + self.offset.to(s.device)

    def params_for(self, s):
        dev = next(self.parameters()).device
        with torch.no_grad():
            p = self.forward(torch.from_numpy(s).float().to(dev).unsqueeze(0))[0]
        return p.cpu().numpy()


def eval_ep(policy, env, max_steps=1600, seed=0):
    """双相位腿 (反相) + 接触耦合 + 每周期参数决策。"""
    s = env.reset()
    leg1 = GaitPhase(A=0.8)   # 腿 1
    leg2 = GaitPhase(A=0.8)   # 腿 2
    leg2.ph = GaitPhase.SWING_FLEX  # 反相启动
    total_r, dist = 0.0, 0.0
    for _ in range(max_steps):
        # 接触 (观测 dim 8=腿1, 10=腿2)
        c1 = s[8] > 0.5
        c2 = s[10] > 0.5
        a1 = leg1.action()
        a2 = leg2.action()
        act = np.concatenate([a1, a2]).astype(np.float32)
        o2, r, d = env.step(act)
        total_r += r; dist = env.dist
        leg1.update(c1); leg2.update(c2)
        # 每周期 (两腿都完成一轮) 更新参数
        if leg1.ph == 0 and leg2.ph == 2:
            p = policy.params_for(s)
            leg1.A = max(0.2, p[0])
            leg2.A = max(0.2, p[0])
        s = o2
        if d:
            break
    return total_r, env.t, dist


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== Walker 走起来: 感觉耦合相位状态机 (局部单元) ===")
    env = WalkerEnergyEnv()
    policy = GaitPolicy().to(dev)
    lr, sigma = 0.3, 0.2

    r0, t0, d0 = eval_ep(policy, env)
    print(f"初始 (随机参数): 存活 {t0} 步, 前进 {d0:.0f}")

    for it in range(30):
        deltas, scores, dists = [], [], []
        for i in range(12):
            delta = []
            for p in policy.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d)
                p.data.add_(d)
            sc, t, dist = eval_ep(policy, env)
            for p, d in zip(policy.parameters(), delta):
                p.data.sub_(d)
            deltas.append(delta); scores.append(sc); dists.append(dist)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(policy.parameters(), delta):
                p.data.add_(lr * wi / (12 * sigma) * d)
        if it % 5 == 4:
            sc, t, dist = eval_ep(policy, env)
            print(f"  iter {it+1}: 得分 {sc:+.1f} | 存活 {t} | 前进 {dist:.0f}")

    times, dists = [], []
    for seed in range(6):
        env2 = WalkerEnergyEnv()
        _, t, dist = eval_ep(policy, env2)
        times.append(t); dists.append(dist)
    print(f"\n最终: 存活 {np.mean(times):.0f}, 前进 {np.mean(dists):.0f} "
          f"(vs sin交替 4-9, 逐动作 11)")
    print(f"{'✅ 走起来了!' if np.mean(dists) > 40 else '⚠️ 有限'}")


if __name__ == "__main__":
    main()
