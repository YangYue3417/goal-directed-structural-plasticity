"""B0: 神经元级稀疏激活单元 (SparseUnit)。

核心主张的实现: 专精载体 = 单个神经元 (FFN 中间层单元), 无块级 router。

机制:
  - 一个大的 FFN: x -> W1 (d_model, d_pool) -> 神经元池 (d_pool)
  - top-k 稀疏: 每输入激活 top-k 神经元, 其余置 0 (STE 反向)
  - 选择者 = 内容者: W1 行既决定"对什么激活"也参与"输出计算"
  - 无独立 router 参数 — 神经元自己竞争 (隐式选择)

梯度: 前向 hard top-k (稀疏), 反向 STE (soft 梯度流到全部被激活神经元)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.interfaces import ConditionalUnit, UnitStats, register_unit


@register_unit("sparse")
class SparseUnit(ConditionalUnit):
    def __init__(
        self,
        d_model: int,
        d_pool: int = 1024,
        top_k: int = 64,
        temperature: float = 1.0,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_pool = d_pool
        self.top_k = top_k
        self.n_units = d_pool  # 神经元池 = 单元池
        self.temperature = temperature

        self.W1 = nn.Parameter(torch.empty(d_model, d_pool))
        self.b1 = nn.Parameter(torch.zeros(d_pool))
        self.W2 = nn.Parameter(torch.empty(d_pool, d_model))
        nn.init.normal_(self.W1, std=init_scale)
        nn.init.normal_(self.W2, std=init_scale)

        # 可塑性机制注册 (B1-B3 由 pipeline 挂载)
        self.plasticity = None

    def set_plasticity(self, pipeline) -> None:
        """挂载可塑性机制 (B1-B3)。"""
        self.plasticity = pipeline
        pipeline.init_for(self)
        # 机制内部状态 tensor 移到单元所在 device
        self._move_plasticity(self.W1.device)

    def _move_plasticity(self, device) -> None:
        if self.plasticity is None:
            return
        for m in self.plasticity.mechanisms:
            for k, v in vars(m).items():
                if isinstance(v, torch.Tensor):
                    setattr(m, k, v.to(device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, UnitStats]:
        B, S, d = x.shape
        # 神经元激活
        pre = x @ self.W1 + self.b1  # (B, S, d_pool)

        # 生长机制: 未募集神经元屏蔽 (active_mask)
        if hasattr(self, "active_mask"):
            if self.active_mask.device != self.W1.device:
                self.active_mask = self.active_mask.to(self.W1.device)
            masked_pre = pre.masked_fill(~self.active_mask.view(1, 1, -1), float("-inf"))
        else:
            masked_pre = pre

        # top-k 稀疏 (hard 前向, 只在活跃神经元中选)
        topk_vals, topk_idx = masked_pre.topk(self.top_k, dim=-1)  # (B, S, k)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, topk_idx, topk_vals)
        # STE: 反向时让梯度流向全部被激活神经元 (sparse 本身就是被激活的)
        # 激活函数
        act = F.gelu(sparse)

        out = act @ self.W2  # (B, S, d)

        # 统计 (表示用于对比学习时需保留梯度)
        load = (sparse > 0).float().mean(dim=(0, 1))  # (d_pool,) 激活频率
        act_strength = sparse.mean(dim=(0, 1))  # (d_pool,) 平均激活强度
        stats = UnitStats(
            activation=act_strength.detach(),
            selected=topk_idx.detach(),
            load=load.detach(),
            input_repr=x.mean(dim=1).detach(),
            extra={"input_repr_grad": x.mean(dim=1)},  # 可导版本 (对比学习用)
        )

        # 可塑性机制 (B1-B3)
        if self.plasticity is not None:
            self.plasticity.step(stats, 0)

        return out, stats

    def plasticity_step(self, stats: UnitStats, global_step: int) -> None:
        if self.plasticity is not None:
            self.plasticity.step(stats, global_step)
