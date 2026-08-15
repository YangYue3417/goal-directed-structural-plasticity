"""气味导航迷宫 — 多信号协调的具身觅食任务。

环境:
  - 迷宫 (网格 + 墙)
  - 多个食物点 (随机位置), 每个发出气味 (高斯场, 随距离衰减)
  - agent: 位置 + 方向, 感知 3 方向气味浓度 + 位置

信号协调 (防作弊):
  ① 气味梯度 (趋利方向线索): 前方/左/右气味浓度
  ② 到达食物: +10 (多巴胺)
  ③ 撞墙: -1 (疼痛)
  ④ 每步: -0.05 (能量消耗)
  ⑤ 饥饿: 步数超限未进食 → 负奖励 (必须持续觅食)

目标: agent 学会朝气味浓的方向走, 绕开墙, 找到多个食物点
  = 嗅觉趋性 (taxis) + 空间记忆 + 多目标
"""
from __future__ import annotations

import numpy as np

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W


class OdorMaze:
    def __init__(self, size: int = 12, wall_density: float = 0.12,
                 n_foods: int = 3, seed: int = 0):
        self.size = size
        rng = np.random.RandomState(seed)
        self.grid = np.zeros((size, size), dtype=np.int8)
        # 随机墙
        for _ in range(int(size * size * wall_density)):
            x, y = rng.randint(size), rng.randint(size)
            if (x, y) != (0, 0):
                self.grid[y, x] = 1
        # 食物点 (空地, 非起点)
        self.foods = []
        while len(self.foods) < n_foods:
            x, y = rng.randint(size), rng.randint(size)
            if (x, y) != (0, 0) and not self.grid[y, x]:
                self.foods.append((x, y))
        self.food_eaten = [False] * n_foods
        self.odor_sigma = size / 2.0  # 气味扩散范围
        self.reset()

    def reset(self):
        self.x, self.y = 0, 0
        self.dir = 1  # 朝东
        self.steps = 0
        n = len(self.foods) if hasattr(self, 'foods') else 3
        self.food_eaten = [False] * n
        return self.observe()

    def odor_at(self, x, y, food_idx):
        """食物点的气味浓度 (高斯衰减)。"""
        fx, fy = self.foods[food_idx]
        if self.food_eaten[food_idx]:
            return 0.0
        d2 = (x - fx) ** 2 + (y - fy) ** 2
        return float(np.exp(-d2 / (2 * self.odor_sigma ** 2)))

    def observe(self):
        """感知: [前气味, 左气味, 右气味, x/12, y/12, 饥饿度]"""
        x, y = self.x, self.y
        # 3 方向的气味 (前方/左/右 一格处)
        def smell(dir_idx):
            dx, dy = DIRS[dir_idx]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                return 0.0
            return sum(self.odor_at(nx, ny, f) for f in range(len(self.foods)))
        fwd = DIRS[self.dir]
        left = DIRS[(self.dir - 1) % 4]
        right = DIRS[(self.dir + 1) % 4]
        def wall(dx, dy):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                return 1.0
            return float(self.grid[ny, nx])
        # 饥饿度: 步数越多越饿 (0→1)
        hunger = min(1.0, self.steps / 40.0)
        obs = np.array([
            smell(self.dir), smell((self.dir - 1) % 4), smell((self.dir + 1) % 4),
            wall(*fwd), wall(*left), wall(*right),
            x / self.size, y / self.size, hunger,
        ], dtype=np.float32)
        return obs

    def step(self, action):
        """0=前进 1=左转 2=右转. 返回 (obs, reward, done)."""
        self.steps += 1
        if action == 1:
            self.dir = (self.dir - 1) % 4
            return self.observe(), -0.05, False
        if action == 2:
            self.dir = (self.dir + 1) % 4
            return self.observe(), -0.05, False
        # 前进
        dx, dy = DIRS[self.dir]
        nx, ny = self.x + dx, self.y + dy
        if not (0 <= nx < self.size and 0 <= ny < self.size) or self.grid[ny, nx]:
            return self.observe(), -1.0, False  # 撞墙 (疼痛)
        self.x, self.y = nx, ny
        # 检查食物
        for fi, (fx, fy) in enumerate(self.foods):
            if (self.x, self.y) == (fx, fy) and not self.food_eaten[fi]:
                self.food_eaten[fi] = True
                return self.observe(), 10.0, False  # 多巴胺!
        # 饥饿惩罚: 步数太多未进食
        reward = -0.05
        if self.steps % 15 == 0 and not any(self.food_eaten):
            reward -= 1.0  # 饿 (时间压力)
        return self.observe(), reward, False

    def render(self):
        g = self.grid.astype(str)
        g[self.y, self.x] = 'A'
        for fx, fy in self.foods:
            g[fy, fx] = 'F'
        print('\n'.join(' '.join(row) for row in g))


if __name__ == "__main__":
    env = OdorMaze(seed=42)
    env.render()
    obs = env.reset()
    print(f"\n食物: {env.foods}")
    print(f"obs: {obs}")
    print(f"气味维度: 前={obs[0]:.3f} 左={obs[1]:.3f} 右={obs[2]:.3f}")
