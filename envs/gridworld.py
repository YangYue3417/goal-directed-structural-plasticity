"""网格世界避障环境 — 位置细胞验证任务。

环境: 10×10 网格, 随机墙
智能体: 位置 (x,y) + 方向 (N/E/S/W)
感知: 前方/左/右是否有墙 (3 bits) + 归一化位置 (x/10, y/10)
动作: 0=前进, 1=左转, 2=右转
奖励: 撞墙 -1, 前进 +0.05, 每步 -0.01 (鼓励效率)

目标: 训练避障行为 (撞墙会拐弯), 同时验证位置细胞 (神经元编码位置)
"""
from __future__ import annotations

import numpy as np

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W


class GridWorld:
    def __init__(self, size: int = 10, wall_density: float = 0.15, seed: int = 0):
        self.size = size
        rng = np.random.RandomState(seed)
        # 网格: 0=空地, 1=墙
        self.grid = np.zeros((size, size), dtype=np.int8)
        n_walls = int(size * size * wall_density)
        for _ in range(n_walls):
            x, y = rng.randint(size), rng.randint(size)
            if (x, y) != (0, 0):  # 起点不堵
                self.grid[y, x] = 1
        self.reset()

    def reset(self, seed=None):
        self.x, self.y = 0, 0
        self.dir = 0  # 朝东
        self.steps = 0
        return self.observe()

    def observe(self):
        """感知: [x/10, y/10, 方向one-hot(4)] — 无墙标志!
        模型必须从位置+方向推断前方是否有墙 (空间记忆)。"""
        x, y = self.x, self.y
        dir_oh = [0.0] * 4
        dir_oh[self.dir] = 1.0
        obs = np.array([
            x / self.size, y / self.size,
            *dir_oh,
            (self.size - 1 - x) / self.size,  # 到目标距离
        ], dtype=np.float32)
        return obs

    def step(self, action):
        """动作: 0=前进, 1=左转, 2=右转. 类多巴胺奖励 (RPE 驱动)."""
        self.steps += 1
        if action in (1, 2):  # 转弯: 轻时间惩罚
            self.dir = (self.dir + (1 if action == 1 else -1)) % 4
            return self.observe(), -0.05, False
        # 前进
        dx, dy = DIRS[self.dir]
        nx, ny = self.x + dx, self.y + dy
        old_dist = abs(self.x - (self.size - 1)) + abs(self.y - (self.size - 1))
        if not (0 <= nx < self.size and 0 <= ny < self.size) or self.grid[ny, nx]:
            return self.observe(), -1.0, False  # 撞墙: 负向 RPE
        self.x, self.y = nx, ny
        new_dist = abs(self.x - (self.size - 1)) + abs(self.y - (self.size - 1))
        if (self.x, self.y) == (self.size - 1, self.size - 1):
            return self.observe(), 10.0, True  # 到达: 大奖励
        # 类多巴胺: 进展奖励 (距离减少) + 前进鼓励 + 时间惩罚
        progress = (old_dist - new_dist) * 0.3   # 接近目标 → 多巴胺
        reward = -0.05 + 0.1 + progress          # 时间 + 移动 + 进展
        return self.observe(), reward, False

    def render(self):
        g = self.grid.astype(str)
        g[self.y, self.x] = 'A'
        g[self.size - 1, self.size - 1] = 'G'
        print('\n'.join(' '.join(row) for row in g))


if __name__ == "__main__":
    env = GridWorld(seed=42)
    env.render()
    print(f"obs: {env.observe()}")
    # 手动试: 一直前进看撞墙
    obs, r, d = env.step(0)
    print(f"前进: r={r} done={d} obs={obs}")
