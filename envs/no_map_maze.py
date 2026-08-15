"""NoMapMaze — 无全局地图的觅食环境 (探索世界机制)。

关键设计: 观测**不含 (x,y) 位置**。要建立空间认知, 只能靠:
  - 局部地标: 多距离墙感 (近/中/远) + 气味场 (sigma=6, 较大)
  - 路径积分: 运动本身 (self-motion) 随时间累积

观测 (13 维):
  [气味前/左/右,          # 3 气味梯度
   墙近前/左/右,           # 3 (1格)    ← 多距离墙感 = 更独特的地标
   墙中前/左/右,           # 3 (2-3格)
   墙远前/左/右,           # 3 (5-6格)
   能量]                   # 1

奖励: 食物 +10 (恢复 +80), 墙 -1, 普通 0, 死亡 -10 (E≤0, episode 结束)
"""
from __future__ import annotations

import numpy as np

from envs.maze_nav import MazeNav, DIRS


class NoMapMaze(MazeNav):
    def __init__(self, size=20, n_foods=3, seed=42, E0=100.0,
                 step_cost=0.5, food_restore=80.0, death_reward=-10.0,
                 sensor="strong"):
        self.sensor = sensor
        self.E0 = E0
        self.step_cost = step_cost
        self.food_restore = food_restore
        self.death_reward = death_reward
        self.energy = E0
        self.odor_sigma = 15.0 if sensor == "strong" else 5.0
        super().__init__(size=size, n_foods=n_foods, seed=seed)

    def reset(self):
        obs = super().reset()
        self.energy = self.E0
        return obs

    def observe(self):
        """13 维: 气味×3 + 墙近×3 + 墙中×3 + 墙远×3 + 能量。无位置!"""
        x, y = self.x, self.y

        def smell(di):
            dx, dy = DIRS[di]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                return 0.0
            return sum(self.odor_at(nx, ny, f) for f in range(self.n_foods)
                       if not self.food_eaten[f])

        def wall_at(dist, di):
            dx, dy = DIRS[di]
            nx, ny = x + dx * dist, y + dy * dist
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                return 1.0
            return float(self.grid[ny, nx])

        obs = []
        if self.sensor != "walls":
            for di in [self.dir, (self.dir + 2) % 4, (self.dir - 1) % 4, (self.dir + 1) % 4]:
                obs.append(smell(di))
        if self.sensor == "strong":
            for dist in [1, 3, 6]:
                for di in [self.dir, (self.dir - 1) % 4, (self.dir + 1) % 4]:
                    obs.append(wall_at(dist, di))
        elif self.sensor == "weak":
            for di in [self.dir, (self.dir - 1) % 4, (self.dir + 1) % 4]:
                obs.append(wall_at(1, di))
        else:  # walls: 纯墙感 (无气味) — 开阔区观测相同 → 必须路径积分
            for di in [self.dir, (self.dir - 1) % 4, (self.dir + 1) % 4]:
                obs.append(wall_at(1, di))
        if self.sensor != "walls":
            obs = obs[:4] + obs[4:]  # 保持结构
        obs.append(self.energy / self.E0)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        self.energy -= self.step_cost
        if action in (1, 2):
            self.dir = (self.dir - 1) % 4 if action == 1 else (self.dir + 1) % 4
            if self.energy <= 0:
                return self.observe(), self.death_reward, True
            return self.observe(), 0.0, False
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
    e = NoMapMaze(seed=42)
    print(f"obs dim: {len(e.reset())} (无位置) | 食物: {e.foods}")
    print("obs:", np.round(e.observe(), 3))
