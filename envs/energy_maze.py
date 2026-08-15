"""EnergyMaze — 能量预算觅食 (饥饿权衡的载体)。

机制 (M6):
  E0=100 初始能量, 每步 -1 (转向/移动都耗能), 食物 +10 且恢复 +60,
  E≤0 → 饿死 (episode 终止, reward -10)。
  obs 10 维: [气味×3, 墙×3, x, y, 饥饿, 能量分数 E/E0]

作用: 远处食物从"无关"变"生存必需" → 饥饿压力调制觅食决策
      (optimal foraging: 能量低 → 冒险远程觅食; 能量高 → 保守)
"""
from __future__ import annotations

import numpy as np

from envs.maze_nav import MazeNav, DIRS


class EnergyMaze(MazeNav):
    def __init__(self, size=20, n_foods=3, seed=42, E0=100.0,
                 step_cost=1.0, food_restore=60.0, death_reward=-10.0):
        self.E0 = E0
        self.step_cost = step_cost
        self.food_restore = food_restore
        self.death_reward = death_reward
        self.energy = E0  # MazeNav.__init__ 会调 self.reset() → observe() 需要 energy
        super().__init__(size=size, n_foods=n_foods, seed=seed)

    def reset(self):
        obs = super().reset()
        self.energy = self.E0
        return obs

    def observe(self):
        base = super().observe()  # 9 维 (含 hunger)
        return np.append(base, self.energy / self.E0)  # 10 维

    def step(self, action):
        self.steps += 1
        self.energy -= self.step_cost
        # 转向: 不改变位置
        if action in (1, 2):
            self.dir = (self.dir - 1) % 4 if action == 1 else (self.dir + 1) % 4
            if self.energy <= 0:
                return self.observe(), self.death_reward, True
            return self.observe(), 0.0, False
        # 前进
        dx, dy = DIRS[self.dir]
        nx, ny = self.x + dx, self.y + dy
        if not (0 <= nx < self.size and 0 <= ny < self.size) or self.grid[ny, nx]:
            if self.energy <= 0:
                return self.observe(), self.death_reward, True
            return self.observe(), -1.0, False
        self.x, self.y = nx, ny
        novelty = 0.0
        if (self.x, self.y) not in self._visited:
            self._visited.add((self.x, self.y))
            novelty = 0.05
        for fi, (fx, fy) in enumerate(self.foods):
            if (self.x, self.y) == (fx, fy) and not self.food_eaten[fi]:
                self.food_eaten[fi] = True
                self.energy += self.food_restore
                if self.energy <= 0:
                    return self.observe(), self.death_reward, True
                return self.observe(), 10.0 + novelty, False
        if self.energy <= 0:
            return self.observe(), self.death_reward, True
        return self.observe(), 0.0 + novelty, False


if __name__ == "__main__":
    e = EnergyMaze(seed=42)
    print(f"obs dim: {len(e.reset())} | 食物: {e.foods}")
    obs, r, done = e.step(1)
    print(f"转向: r={r}, energy={e.energy:.0f}")
