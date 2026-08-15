"""迷宫导航 — 大迷宫 + 固定食物点 + 固定起点, 测空间记忆与最短路径。

与 odor_maze 的区别 (针对 Q5 升级: 记忆型分化):
  - 大迷宫 (24×24+), 墙密度可调
  - 食物点固定位置 (固定臂: 每 episode 相同) — 可记忆 → 位置细胞可涌现
  - 固定起点 (0,0) — 每 episode 同一出发
  - 随机臂: 每 episode 重新随机布局 + 食物 → 记忆无意义 (对照)

obs (9 维, 与 SparseSNN 兼容): [前/左/右气味, 前/左/右墙, x/size, y/size, 饥饿]
reward: 食物 +10, 撞墙 -1, 每步 -0.1 (效率压力 → 逼最短路径)

度量 (eval 用):
  - BFS 最短路径 (贪心最近食物 TSP 近似, 访问所有食物)
  - 路径效率 = 最短步数 / 实际步数
"""
from __future__ import annotations

from collections import deque

import numpy as np

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W


def bfs_dist(grid, start, targets):
    """BFS: start 到所有 target 的最短路径 (墙绕行)。返回距离列表 (inf=不可达)。"""
    size = grid.shape[0]
    dist = {t: float("inf") for t in targets}
    q = deque([(start[0], start[1], 0)])
    seen = {start}
    remaining = set(targets)
    while q and remaining:
        x, y, d = q.popleft()
        if (x, y) in remaining:
            dist[(x, y)] = d
            remaining.remove((x, y))
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and not grid[ny, nx] and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny, d + 1))
    return dist


def greedy_shortest(grid, start, foods):
    """访问所有食物的最短路径近似: 贪心最近食物 (BFS 距离)。"""
    remaining = list(foods)
    pos = start
    total = 0
    while remaining:
        dist = bfs_dist(grid, pos, remaining)
        best = min(remaining, key=lambda t: dist[t])
        d = dist[best]
        if d == float("inf"):
            return float("inf")
        total += d
        pos = best
        remaining.remove(best)
    return total


class MazeNav:
    def __init__(self, size: int = 24, wall_density: float = 0.12,
                 n_foods: int = 4, seed: int = 0, randomize: bool = False,
                 start=(0, 0)):
        self.size = size
        self.wall_density = wall_density
        self.n_foods = n_foods
        self.seed = seed
        self.randomize = randomize
        self.start = start
        self._rng = np.random.RandomState(seed)  # 持久 rng (randomize 时每回合推进)
        self._build()
        self.reset()

    def _build(self):
        rng = self._rng  # 共享 rng: 每次 build 推进状态, 随机臂每回合新迷宫
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        for _ in range(int(self.size * self.size * self.wall_density)):
            x, y = rng.randint(self.size), rng.randint(self.size)
            if (x, y) != self.start:
                self.grid[y, x] = 1
        # 食物 (空地, 非起点, 互不重叠)
        self.foods = []
        while len(self.foods) < self.n_foods:
            x, y = rng.randint(self.size), rng.randint(self.size)
            if (x, y) != self.start and not self.grid[y, x] \
                    and (x, y) not in self.foods:
                self.foods.append((x, y))
        self.odor_sigma = 4.0  # 气味梯度范围 (宽到搜索可达, 仍局部于大迷宫)
        self.shortest_total = greedy_shortest(self.grid, self.start, self.foods)

    def reset(self):
        if self.randomize:
            self._build()  # 随机臂: 每 episode 新迷宫
        self.x, self.y = self.start
        self.dir = 1
        self.steps = 0
        self.food_eaten = [False] * self.n_foods
        self._visited = {(self.x, self.y)}
        return self.observe()

    def odor_at(self, x, y, fi):
        if self.food_eaten[fi]:
            return 0.0
        fx, fy = self.foods[fi]
        d2 = (x - fx) ** 2 + (y - fy) ** 2
        return float(np.exp(-d2 / (2 * self.odor_sigma ** 2)))

    def observe(self):
        x, y = self.x, self.y
        def smell(di):
            dx, dy = DIRS[di]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                return 0.0
            return sum(self.odor_at(nx, ny, f) for f in range(self.n_foods))
        def wall(dx, dy):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                return 1.0
            return float(self.grid[ny, nx])
        hunger = min(1.0, self.steps / (8 * self.size))
        return np.array([
            smell(self.dir), smell((self.dir - 1) % 4), smell((self.dir + 1) % 4),
            wall(*DIRS[self.dir]), wall(*DIRS[(self.dir - 1) % 4]), wall(*DIRS[(self.dir + 1) % 4]),
            x / self.size, y / self.size, hunger,
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        if action == 1:  # 左转
            self.dir = (self.dir - 1) % 4
            return self.observe(), -0.1, False
        if action == 2:  # 右转
            self.dir = (self.dir + 1) % 4
            return self.observe(), -0.1, False
        dx, dy = DIRS[self.dir]
        nx, ny = self.x + dx, self.y + dy
        if not (0 <= nx < self.size and 0 <= ny < self.size) or self.grid[ny, nx]:
            return self.observe(), -1.0, False
        self.x, self.y = nx, ny
        # 新奇奖励: 首次访问新格子 (探索激励, 不泄露食物位置)
        novelty = 0.0
        if (self.x, self.y) not in self._visited:
            self._visited.add((self.x, self.y))
            novelty = 0.05
        for fi, (fx, fy) in enumerate(self.foods):
            if (self.x, self.y) == (fx, fy) and not self.food_eaten[fi]:
                self.food_eaten[fi] = True
                done = all(self.food_eaten)
                return self.observe(), 10.0 + novelty, done
        return self.observe(), -0.1 + novelty, False


if __name__ == "__main__":
    env = MazeNav(size=12, seed=1)
    print(f"食物: {env.foods}")
    print(f"BFS 最短路径 (贪心): {env.shortest_total}")
    print(f"obs: {env.observe()}")
