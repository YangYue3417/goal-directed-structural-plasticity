"""统一接口: 条件计算单元 (ConditionalUnit)。

所有可插拔的稀疏计算单元实现此接口:
  - MoEUnit      (块级, 来自旧 dynamic-moe, 可作对照)
  - SparseUnit   (神经元级 top-k, B0)
  - LIFUnit      (动力学, 阶段2)

接口保证: 训练循环 / 数据 / checkpoint / 分析工具无需感知具体单元。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class UnitStats:
    """单元前向统计 (供分析/可塑性机制使用)。"""

    # 激活模式: 本批每个单元 (神经元/expert) 的激活强度 (n_units,)
    activation: Optional[torch.Tensor] = None
    # 选择索引: 本批被选中的单元 id (top-k 索引)
    selected: Optional[torch.Tensor] = None
    # 负载: 各单元被选中的 token 比例 (n_units,)
    load: Optional[torch.Tensor] = None
    # 输入表示: mean-pooled 输入 (供概念分析)
    input_repr: Optional[torch.Tensor] = None
    # 自定义扩展字段
    extra: dict = field(default_factory=dict)


class ConditionalUnit(nn.Module):
    """条件计算单元基类 (可插拔契约)。"""

    #: 单元数量 (专家数 / 神经元池大小)
    n_units: int = 0

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, UnitStats]:
        """前向 + 统计。

        Args:
            x: (B, S, d_model) 输入。
        Returns:
            (output, stats): (B, S, d_model) 输出 + 激活统计。
        """
        raise NotImplementedError

    # -- 可塑性机制接口 (B1-B3, 默认 no-op) ---------------------------
    def plasticity_step(self, stats: UnitStats, global_step: int) -> None:
        """每训练步调用, 让可塑性机制调整权重 (稳态/侧抑制/用进废退)。"""
        pass

    # -- 结构演化接口 (生长/修剪, 默认 no-op) --------------------------
    def grow(self, **kwargs) -> None:
        """增加单元 (神经元级生长)。"""
        pass

    def prune(self, **kwargs) -> None:
        """删除单元 (用进废退修剪)。"""
        pass


def register_unit(name: str):
    """单元注册表: 使单元可从配置字符串创建 (可插拔)。"""

    def deco(cls):
        ConditionalUnit.registry[name] = cls
        return cls

    return deco


ConditionalUnit.registry: dict[str, type["ConditionalUnit"]] = {}
