"""walker_energy_env.py — Walker 能量+驱动规则环境 (用户设计修正)。

① 能量 = 输出层隐形约束 (动作代价), 非神经元活动约束:
   E -= Σ|τ_i·ω_i|·dt + c_base   (关节力矩×角速度 = 机械功率)
   神经元活动不消耗能量
   阶段目标达成 (前进距离 D) → 能量补充 +E_goal (不区分目标)

② 驱动规则 (环境层, 与 MPC 无关):
   不持续向右移动 (vx < v_min 连续 N 步) → 死 (惩罚)
   → 站桩策略失效, 必须移动
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class WalkerEnergyEnv:
    OBS_DIM = 26  # 24 + E(1) + 阶段进度(1)

    def __init__(self, v_min=0.4, slow_n=80, E0=100.0, c_out=0.5,
                 c_base=0.02, D_goal=40.0, E_goal=40.0, seed=0):
        self.v_min = v_min
        self.slow_n = slow_n
        self.E0 = E0
        self.c_out = c_out      # 输出能量系数
        self.c_base = c_base    # 基础代谢
        self.D_goal = D_goal    # 阶段目标: 前进距离
        self.E_goal = E_goal    # 阶段奖励: 能量补充
        self.env = gym.make('BipedalWalker-v3')
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.obs, _ = self.env.reset(seed=int(self.rng.randint(10**6)))
        self.prev_obs = self.obs.copy()
        self.E = self.E0
        self.slow_steps = 0
        self.x_total = 0.0      # 累计前进
        self.last_x = self.obs[0]
        self.dist = 0.0         # 阶段进度
        self.t = 0
        return self._obs()

    def _obs(self):
        return np.concatenate([self.obs, [self.E / self.E0,
                                          self.dist / self.D_goal]]).astype(np.float32)

    def step(self, act):
        # 能量: 输出能量 = Σ|τ·ω| (力矩 act × 关节角速度, 角度差分近似)
        omega = np.abs(self.obs[4:8] - self.prev_obs[4:8])  # 4 关节角速度
        power = self.c_out * float(np.sum(np.abs(act) * omega))
        self.E -= power + self.c_base
        # 环境 step
        self.obs, r_env, done_env, _, _ = self.env.step(act)
        self.prev_obs = self.obs.copy()
        self.t += 1
        # 前进量 (hull x 位置估计 = 累计 vel_x)
        vx = self.obs[2]
        self.dist += max(0.0, vx)
        # 驱动规则: 不持续向右 → 死 (与 MPC 无关, 环境规则)
        if vx < self.v_min:
            self.slow_steps += 1
        else:
            self.slow_steps = 0
        dead_slow = self.slow_steps > self.slow_n
        # 阶段目标达成 → 能量补充 (不区分目标)
        gained = 0.0
        while self.dist >= self.D_goal:
            self.dist -= self.D_goal
            self.E = min(self.E0 * 1.5, self.E + self.E_goal)
            gained += self.E_goal
        # 死亡条件: 环境跌倒 / 驱动规则 / 能量耗尽
        dead = done_env or dead_slow or self.E <= 0
        # 奖励: 前进微量 + 阶段能量补充 + 死亡惩罚 (惩罚驱动, 无任务目标)
        r = 0.01 * vx + 0.1 * gained - 1.0 * dead
        return self._obs(), r, dead

    def close(self):
        self.env.close()


if __name__ == "__main__":
    print("=== Walker 能量+驱动规则环境测试 ===")
    env = WalkerEnergyEnv()
    # 测试 1: 不移动 (零动作) → 应死于驱动规则
    o = env.reset()
    steps = 0
    for _ in range(500):
        o, r, d = env.step(np.zeros(4, np.float32))
        steps += 1
        if d:
            break
    print(f"零动作: {steps} 步后死 (驱动规则: 不移动死) {'✅' if steps < 200 else '❌'}")

    # 测试 2: 固定前进动作 (向右推) → 应活更久
    env2 = WalkerEnergyEnv()
    o = env2.reset()
    steps2 = 0
    for _ in range(500):
        a = np.array([1.0, 1.0, 0.5, 0.5], np.float32)  # 持续蹬腿
        o, r, d = env2.step(a)
        steps2 += 1
        if d:
            break
    print(f"持续前进动作: {steps2} 步 (零动作 {steps} 步) — 驱动规则区分 {'✅' if steps2 > steps else '❌'}")
    print(f"能量: 输出层约束 (动作×角速度), 阶段目标 {env2.D_goal} 距离补充 {env2.E_goal}")
