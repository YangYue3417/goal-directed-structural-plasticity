"""walker_safe_zone.py — 姿态安全区: hull 站起来的高度 = 安全探索区。

安全 = hull 高度 > h_safe (真站立, 站直)
危险 = hull 高度 < h_safe 持续 N 步 → 死亡 (蹲/跪/跌倒 = 脆弱)

融合:
  安全区 = 姿态 (站姿), 非空间位置
  探索 = 保持站姿移动 (安全探索)
  能量 = 输出层 + 阶段补充 (已有)
  驱动 = 不动死 (已有)

复用 WalkerEnergyEnv (继承, 加姿态安全规则), 不改现有文件。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from walker_energy_env import WalkerEnergyEnv


class SafeZoneEnv(WalkerEnergyEnv):
    """动态安全区: 站姿(高) + 左侧区域逐渐危险 (必须右移探索)。"""
    def __init__(self, h_safe=5.6, low_n=60, x_left=2.0, x_n=40, **kw):
        self.h_safe = h_safe    # 边缘站立高度 (腿伸长≥85%: 3.35+2.29×0.85+0.29)
        self.low_n = low_n      # 低姿态持续 N 步 → 危险死亡
        self.x_left = x_left    # 左侧危险边界 (x < 此值)
        self.x_n = x_n          # 左侧停留 N 步 → 危险死亡
        super().__init__(**kw)

    def _hull_y(self):
        return self.env.unwrapped.hull.position.y

    def reset(self):
        o = super().reset()
        self.low_steps = 0
        self.x_left_steps = 0
        return o

    def step(self, act):
        # 能量 + 驱动 (父类逻辑, 但需 hull 高度)
        omega = np.abs(self.obs[4:8] - self.prev_obs[4:8])
        self.E -= self.c_out * float(np.sum(np.abs(act) * omega)) + self.c_base
        self.obs, r_env, done_env, _, _ = self.env.step(act)
        self.prev_obs = self.obs.copy()
        self.t += 1
        vx = self.obs[2]
        self.dist += max(0.0, vx)
        self.slow_steps = self.slow_steps + 1 if vx < self.v_min else 0
        dead_slow = self.slow_steps > self.slow_n
        # 姿态安全: 低姿态持续 → 危险死亡
        hy = self._hull_y()
        self.low_steps = self.low_steps + 1 if hy < self.h_safe else 0
        dead_low = self.low_steps > self.low_n
        # 动态安全: 左侧区域逐渐危险 (x 低 → 停留累积)
        hx = self.env.unwrapped.hull.position.x
        self.x_left_steps = self.x_left_steps + 1 if hx < self.x_left else 0
        dead_left = self.x_left_steps > self.x_n
        gained = 0.0
        while self.dist >= self.D_goal:
            self.dist -= self.D_goal
            self.E = min(self.E0 * 1.5, self.E + self.E_goal)
            gained += self.E_goal
        dead = done_env or dead_slow or dead_low or dead_left or self.E <= 0
        # 踏步: 髋前摆 + 屈膝抬腿 (膝可弯, 幅度可见)
        hip1, hip2 = self.obs[4], self.obs[6]
        knee1, knee2 = self.obs[5], self.obs[7]
        # 交替 = hip 反相 (hip1·hip2<0) 且幅度显著 (踏步可见)
        alt_amp = abs(hip1) + abs(hip2)
        alternating = (hip1 * hip2 < 0) and (alt_amp > 0.15)
        # 屈膝允许 (踏步时膝盖弯曲 = 抬腿)
        knee_act = abs(knee1) + abs(knee2)  # 膝活跃 (弯曲=抬腿)
        # 奖励: 唯一硬约束 hull 安全区; 踏步 (交替+髋摆+屈膝) + 前进
        safe_bonus = 0.3 if hy >= self.h_safe else 0.0   # 硬约束
        step_bonus = 0.15 if alternating else 0.0        # 交替踏步
        knee_bonus = 0.03 * min(2.0, knee_act)           # 屈膝抬腿鼓励
        right_bonus = 0.1 * max(0.0, min(1.0, hx / 5.0))
        r = 0.3 * vx + safe_bonus + step_bonus + knee_bonus + right_bonus + 0.1 * gained - 1.0 * dead
        return self._obs(), r, dead


class Agent(nn.Module):
    """统一池策略 (窗口): 观测 → 腿1模式, 腿2 = 反相耦合 (结构保证交替)。"""
    def __init__(self, obs=26, L=8, d=64, pool=256, top_k=32):
        super().__init__()
        self.L = L
        self.embed = nn.Linear(obs * L, d)
        self.act_mask = torch.zeros(pool, dtype=torch.bool)
        self.act_mask[:96] = True
        self.W = nn.Linear(d, pool)
        self.head = nn.Linear(pool, 2)   # 输出 (hip, knee) 模式

    def forward(self, x):
        z = torch.tanh(self.embed(x))
        pre = self.W(z)
        m = self.act_mask.to(pre.device)
        pre = pre.masked_fill(~m[None], -1e9)
        vals, idx = pre.topk(32, dim=1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(1, idx, torch.tanh(vals))
        return torch.tanh(self.head(sparse))   # (B, 2): hip, knee

    def act(self, hist, noise=0.0):
        dev = next(self.parameters()).device
        with torch.no_grad():
            m = self.forward(torch.from_numpy(hist).float().to(dev).unsqueeze(0))[0]
            if noise:
                m = m + noise * torch.randn_like(m)
        h, k = float(m[0]), float(m[1])
        # 结构耦合: 腿2 = 反相 (hip2=-h, knee2=-k) — 两腿必然交替
        return np.array([h, k, -h, -k], np.float32)


def eval_ep(agent, env, max_steps=1600, noise=0.0):
    L = agent.L
    hist = np.zeros((L, 26), np.float32)
    s = env.reset(); hist[0] = s
    total_r, dist, safe_steps = 0.0, 0.0, 0
    for _ in range(max_steps):
        a = agent.act(hist.flatten(), noise=noise)
        o2, r, d = env.step(a)
        total_r += r; dist = env.dist
        safe_steps += env._hull_y() >= env.h_safe
        hist[1:] = hist[:-1]; hist[0] = o2
        s = o2
        if d:
            break
    return total_r, env.t, dist, safe_steps


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 姿态安全区: hull 站起高度 = 安全, 低姿态 = 危险 ===")
    env = SafeZoneEnv(h_safe=5.0)
    agent = Agent().to(dev)
    lr, sigma = 0.3, 0.2

    r0, t0, d0, s0 = eval_ep(agent, env)
    print(f"初始: 存活 {t0}, 安全帧 {s0}/{t0}, 前进 {d0:.0f}")

    for it in range(30):
        deltas, scores, dists, safes = [], [], [], []
        for i in range(10):
            delta = []
            for p in agent.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d); p.data.add_(d)
            sc, t, dist, safe = eval_ep(agent, env)
            for p, d in zip(agent.parameters(), delta): p.data.sub_(d)
            deltas.append(delta); scores.append(sc); dists.append(dist); safes.append(safe)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(agent.parameters(), delta):
                p.data.add_(lr * wi / (10 * sigma) * d)
        if it % 5 == 4:
            sc, t, dist, safe = eval_ep(agent, env)
            print(f"  iter {it+1}: 存活 {t}, 安全 {safe}/{t}, 前进 {dist:.0f}")

    # 最终 (多 seed)
    ts, ds, ss = [], [], []
    for _ in range(6):
        e2 = SafeZoneEnv(h_safe=5.0)
        _, t, dist, safe = eval_ep(agent, e2)
        ts.append(t); ds.append(dist); ss.append(safe)
    print(f"\n最终: 存活 {np.mean(ts):.0f}, 安全帧 {np.mean(ss)/max(np.mean(ts),1)*100:.0f}%, "
          f"前进 {np.mean(ds):.0f}")
    print(f"{'✅ 站姿安全探索 (站直+移动)' if np.mean(ss)/max(np.mean(ts),1) > 0.6 and np.mean(ds) > 15 else '⚠️ 有限'}")


if __name__ == "__main__":
    main()
