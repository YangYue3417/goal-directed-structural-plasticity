"""B4: 神经元生长 v2 (固定池 + 启用掩码)。

v1 (动态池) 的问题: 新参数不在优化器, 训练失效。
v2: 池大小固定 (max_pool), 生长 = 启用掩码 (0→1)。
  - 所有神经元参数一直存在 (优化器全覆盖)
  - 未启用的神经元被 mask 掉 (不参与 top-k, 无梯度)
  - 分裂: 复制源神经元权重到目标位置 + 扰动, 启用 mask

这更接近真实神经发育: 神经元池恒定, 生长是"募集"新神经元。
"""
from __future__ import annotations

import torch

from core.interfaces import UnitStats
from core.plasticity import PlasticityMechanism, register_mechanism


@register_mechanism("neurogenesis")
class Neurogenesis(PlasticityMechanism):
    """B4: 神经元生长 (固定池 + 启用掩码)。"""

    name = "neurogenesis"

    def __init__(self, grow_after: int = 3000, load_mult: float = 2.0,
                 perturb_std: float = 0.1, grow_interval: int = 500):
        super().__init__()
        self.grow_after = grow_after
        self.load_mult = load_mult
        self.perturb_std = perturb_std
        self.grow_interval = grow_interval
        self._step = 0

    def init_for(self, unit) -> None:
        self._unit = unit
        self._load_ema = None
        # 启用掩码: 1 = 活跃, 0 = 未募集 (在 SparseUnit 上创建)
        if not hasattr(unit, "active_mask"):
            unit.register_buffer(
                "active_mask",
                torch.ones(unit.n_units, dtype=torch.bool),
            )
            # 初始只激活一部分 (n_init 比例)
            n_init = max(1, unit.n_units // 4)
            unit.active_mask[n_init:] = False
        self._unit = unit

    def step(self, stats: UnitStats, global_step: int) -> None:
        self._step += 1
        if stats.load is None:
            return
        u = self._unit
        if self._step < self.grow_after:
            return
        if self._step % self.grow_interval != 0:
            return
        if u.active_mask.all():
            return  # 全部已募集
        if self._load_ema is None:
            self._load_ema = stats.load.clone()
        else:
            self._load_ema = 0.9 * self._load_ema + 0.1 * stats.load
        # 候选: 活跃且负载高
        active_idx = u.active_mask.nonzero().flatten()
        if active_idx.numel() == 0:
            return
        loads = self._load_ema[active_idx]
        mean_load = loads.mean()
        cand = active_idx[loads > mean_load * self.load_mult]
        if cand.numel() == 0:
            return
        src = int(cand[0].item())
        # 找下一个未募集的神经元
        inactive = (~u.active_mask).nonzero().flatten()
        if inactive.numel() == 0:
            return
        tgt = int(inactive[0].item())
        with torch.no_grad():
            u.W1.data[:, tgt] = u.W1.data[:, src] + self.perturb_std * torch.randn_like(u.W1.data[:, src])
            u.W2.data[tgt, :] = u.W2.data[src, :] + self.perturb_std * torch.randn_like(u.W2.data[src, :])
            u.b1.data[tgt] = u.b1.data[src] + self.perturb_std * torch.randn_like(u.b1.data[src])
        u.active_mask[tgt] = True
        self._load_ema[tgt] = self._load_ema[src]
        n_active = int(u.active_mask.sum())
        print(f"[Neurogenesis] step={self._step} {src}→{tgt} 激活 {n_active}")
