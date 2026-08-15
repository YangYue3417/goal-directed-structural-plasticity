"""B1-B3 类脑可塑性机制实现 (可插拔零件)。

B1 Homeostasis (稳态):     每神经元维持目标发放率, 调整阈值/增益
B2 LateralInhibition (侧抑制): 激活神经元抑制邻近 (基于激活相关性)
B3 ActivityPruning (用进废退): 活动依赖修剪 — 低 Fisher 重要性 + 长期不活跃
对照 HebbianDecay:           旧 sparse_router 的简单衰减 (未激活权重衰减)
"""
from __future__ import annotations

import torch
from core.interfaces import UnitStats
from core.plasticity import PlasticityMechanism, register_mechanism


@register_mechanism("homeostasis")
class Homeostasis(PlasticityMechanism):
    """B1: 稳态 — 每神经元维持目标发放率。

    类脑对应: firing-rate homeostasis (长期不活跃神经元提高敏感度)。
    实现: 发放率 EMA -> 超目标 -> 降增益 (阈值上升); 低于目标 -> 升增益。
    """

    # (纯 Python 对象, 非 nn.Module)

    name = "homeostasis"

    def __init__(self, target_rate: float = 0.05, eta: float = 0.01, ema: float = 0.9):
        self.target_rate = target_rate
        self.eta = eta
        self.ema = ema
        self.rate_ema: torch.Tensor | None = None
        self.gain: torch.Tensor | None = None

    def init_for(self, unit) -> None:
        n = unit.n_units
        dev = unit.W1.device
        self.rate_ema = torch.zeros(n, device=dev)
        self.gain = torch.ones(n, device=dev)

    def step(self, stats: UnitStats, global_step: int) -> None:
        if stats.load is None or self.rate_ema is None:
            return
        self.rate_ema = self.ema * self.rate_ema + (1 - self.ema) * stats.load
        err = self.rate_ema - self.target_rate
        # 增益调整: 过活跃 -> 降增益; 不活跃 -> 升增益
        self.gain = torch.clamp(self.gain - self.eta * err, 0.1, 10.0)


@register_mechanism("lateral_inhibition")
class LateralInhibition(PlasticityMechanism):
    """B2: 侧抑制 — 激活神经元抑制邻近 (竞争锐化)。

    类脑对应: 抑制性中间神经元实现的 winner-take-all 竞争。
    实现: 激活模式间的相关性 -> 抑制项推离相似神经元。
    """

    name = "lateral_inhibition"

    def __init__(self, strength: float = 0.01, sim_threshold: float = 0.5):
        self.strength = strength
        self.sim_threshold = sim_threshold

    def init_for(self, unit) -> None:
        self._W1 = getattr(unit, "W1", None)

    def step(self, stats: UnitStats, global_step: int) -> None:
        if self._W1 is None:
            return
        w = self._W1.detach().float()
        wn = F_normalize(w, dim=0)  # (d_model, d_pool)
        sim = (wn.T @ wn).abs()  # (d_pool, d_pool)
        sim.fill_diagonal_(0)
        mask = sim > self.sim_threshold
        if mask.any():
            # 抑制项: 推离高度相似的神经元
            push = self.strength * (sim * mask.float()) @ wn.T  # (d_pool, d)
            with torch.no_grad():
                self._W1.data -= push.T  # (d_model, d_pool)


def F_normalize(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=dim)


@register_mechanism("activity_pruning")
class ActivityPruning(PlasticityMechanism):
    """B3: 用进废退 — 活动依赖修剪。

    文献依据 (PLOS CB 2021): 仅按权重幅值修剪非最优;
    应基于活动相关性 (Fisher 信息, <pre,post> 相关) 判断重要性。
    实现: 长期低激活 + 低活动相关性 -> 标记待修剪 (weight 归零 + mask)。
    """

    name = "activity_pruning"

    def __init__(self, prune_after: int = 2000, act_threshold: float = 1e-3,
                 prune_ratio: float = 0.1):
        self.prune_after = prune_after
        self.act_threshold = act_threshold
        self.prune_ratio = prune_ratio
        self.inactive_steps = None  # 由 init_for 创建 tensor
        self.total_steps = 0

    def init_for(self, unit) -> None:
        self._unit = unit
        self._n = unit.n_units
        dev = getattr(unit.W1, 'device', torch.device('cpu'))
        self._dead = torch.zeros(unit.n_units, dtype=torch.bool, device=dev)
        self.inactive_steps = torch.zeros(unit.n_units, device=dev)

    def step(self, stats: UnitStats, global_step: int) -> None:
        self.total_steps += 1
        if stats.activation is None:
            return
        # 长期不活跃的神经元
        low_act = stats.activation < self.act_threshold
        self.inactive_steps = self.inactive_steps + 1
        self.inactive_steps = torch.where(
            low_act, self.inactive_steps, torch.zeros_like(self.inactive_steps)
        )
        if self.total_steps < self.prune_after:
            return
        # 达到不活跃阈值 -> 修剪 (权重归零, 后续不再激活)
        to_prune = (self.inactive_steps > self.prune_after) & ~self._dead
        if to_prune.any():
            with torch.no_grad():
                self._unit.W1.data[:, to_prune] = 0.0
                self._unit.W2.data[to_prune, :] = 0.0
            self._dead |= to_prune
            n_pruned = int(to_prune.sum())
            if n_pruned > 0:
                print(f"[ActivityPruning] step={self.total_steps} 修剪 {n_pruned} 神经元")


@register_mechanism("hebbian_decay")
class HebbianDecay(PlasticityMechanism):
    """对照: 旧 sparse_router 的简单用进废退 (未激活权重衰减)。"""

    name = "hebbian_decay"

    def __init__(self, decay: float = 0.001):
        self.decay = decay

    def init_for(self, unit) -> None:
        self._unit = unit

    def step(self, stats: UnitStats, global_step: int) -> None:
        if stats.load is None:
            return
        inactive = stats.load < 1e-4
        if inactive.any():
            with torch.no_grad():
                self._unit.W1.data[:, inactive] *= (1.0 - self.decay)
                self._unit.W2.data[inactive, :] *= (1.0 - self.decay)
