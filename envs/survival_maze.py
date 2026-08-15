"""SurvivalMaze — 动态环境生存 (地图/食物每天变化 + 跨天能量)。

每天清晨 (reset_day):
  - 新迷宫: 随机墙布局 (wall_density=0.10) + 3 个随机食物位置
  - 能量跨天延续: 第一天 E0=100, 之后 = 昨天剩余
  - 食物必须 BFS 可达 (否则重生成)

天内 (reset → step):
  - 从 (0,0) 出发, 150 步白天
  - cost 1.0/步, 食物 +60 恢复, 死亡 E≤0 (-10, done)

观测 (13 维, 无位置): [气味前/左/右, 墙近×3, 墙中×3, 墙远×3, 能量]

30 天不死 = 每天在新地图靠通用规律觅食 (不能背位置/地图)
"""
from __future__ import annotations

import numpy as np

from envs.no_map_maze import NoMapMaze, DIRS


class SurvivalMaze(NoMapMaze):
    def __init__(self, size=20, n_foods=3, seed=42, E0=100.0,
                 step_cost=1.0, food_restore=60.0, death_reward=-10.0,
                 day_steps=150, wall_density=0.10, sensor="strong", odor_sigma=None):
        self.E0 = E0
        self.step_cost = step_cost
        self.food_restore = food_restore
        self.death_reward = death_reward
        self.day_steps = day_steps
        self.wall_density = wall_density
        self.sensor = sensor
        self.day = 0
        self._day_seed = seed
        self.energy = E0
        self.odor_sigma = odor_sigma if odor_sigma is not None else (15.0 if sensor == "strong" else 5.0)
        # 直接构造 (不调 MazeNav.__init__ 的默认 _build)
        self.size = size
        self.n_foods = n_foods
        self.seed = seed
        self.randomize = False
        self.start = (0, 0)
        self._rng = np.random.RandomState(seed)
        self._build()
        self.reset()

    def reset_day(self):
        """清晨: 新迷宫 + 新食物 (能量跨天延续)。返回 (obs, 是否饿死)。"""
        self.day += 1
        self._build()  # 新墙 + 新食物
        if self.energy <= 0:
            return self.observe(), True  # 昨天就死了 (不应发生, 防御)
        obs = self.reset()
        return obs, False

    def _build(self):
        """新一天的迷宫: 随机墙 + 随机食物 (BFS 可达保证)。"""
        from envs.maze_nav import bfs_dist
        for attempt in range(50):
            grid = np.zeros((self.size, self.size), dtype=np.int8)
            for _ in range(int(self.size * self.size * self.wall_density)):
                x, y = self._rng.randint(self.size), self._rng.randint(self.size)
                if (x, y) != self.start:
                    grid[y, x] = 1
            # 食物: 随机空地, 必须从起点可达
            empty = [(x, y) for x in range(self.size) for y in range(self.size)
                     if not grid[y, x] and (x, y) != self.start]
            self._rng.shuffle(empty)
            foods = empty[:self.n_foods]
            dist = bfs_dist(grid, self.start, foods)
            if all(np.isfinite(dist[f]) for f in foods):
                self.grid = grid
                self.foods = foods
                self.food_eaten = [False] * self.n_foods
                return
        raise RuntimeError("无法生成可达迷宫")

    def step(self, action):
        """区域 B (x>=0.5) 移动耗能 2× (区域功能差异 → 神经元区域专精)。"""
        self.steps += 1
        cost = self.step_cost * (2.0 if self.x / self.size >= 0.5 else 1.0)
        self.energy -= cost
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

    def reset(self):
        """当天开始: 回到起点, 位置重置, 能量延续。"""
        self.x, self.y = self.start
        self.dir = 1
        self.steps = 0
        self.food_eaten = [False] * self.n_foods
        self._visited = {(self.x, self.y)}
        return self.observe()


if __name__ == "__main__":
    e = SurvivalMaze(seed=42)
    obs, died = e.reset_day()
    print(f"Day {e.day}: obs={len(obs)} 食物={e.foods} 能量={e.energy:.0f}")
    obs, died = e.reset_day()
    print(f"Day {e.day}: 新食物={e.foods} (应不同) 能量={e.energy:.0f}")


def render_image(env, mode="full", window=5, cell=4):
    """渲染迷宫为像素图像。
    full: 上帝视角 (40x40, 无 agent 标记 — 纯环境布局)
    local: agent 局部视野 (window×window 格, 第一人称, 有 agent 标记)
    编码: 0=空地, 1=墙, 2=食物, (local 3=agent)
    """
    import numpy as np
    size = env.size
    if mode == "full":
        img = np.zeros((size, size), dtype=np.float32)
        img[env.grid == 1] = 1.0
        for fx, fy in env.foods:
            if not env.food_eaten[env.foods.index((fx, fy))]:
                img[fy, fx] = 2.0
        img = np.kron(img, np.ones((cell, cell)))
        img = img[None]  # (1, H, W)
        return img
    else:  # local 第一人称视野 (以 agent 为中心, 朝前方向)
        w = window // 2
        img = np.zeros((window, window), dtype=np.float32)
        for dy in range(-w, w + 1):
            for dx in range(-w, w + 1):
                wx, wy = env.x + dx, env.y + dy
                if not (0 <= wx < size and 0 <= wy < size):
                    img[dy + w, dx + w] = 1.0  # 边界=墙
                elif env.grid[wy, wx]:
                    img[dy + w, dx + w] = 1.0
                elif (wx, wy) in env.foods and not env.food_eaten[env.foods.index((wx, wy))]:
                    img[dy + w, dx + w] = 2.0
        img[w, w] = 3.0  # agent 位置
        img = np.kron(img, np.ones((cell, cell)))
        img = img[None]
        return img
