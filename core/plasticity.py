"""可塑性机制基类 (PlasticityMechanism)。

B1-B3 阶梯机制统一实现此接口, 可组合、可消融:
  - Homeostasis        (B1): 稳态 — 维持目标发放率
  - LateralInhibition  (B2): 侧抑制 — 激活抑制邻近
  - ActivityPruning    (B3): 用进废退 — 活动依赖修剪 (Fisher)
  - HebbianDecay       (对照): 旧 sparse_router 里的衰减

注意: 机制是纯 Python 对象 (非 nn.Module), 避免与 unit 形成
循环引用 (unit 持有 pipeline, pipeline 持有 unit 引用)。
内部状态为 torch.Tensor, 由 init_for 时手动放到正确 device。
"""
from __future__ import annotations

import torch

from core.interfaces import UnitStats


class PlasticityMechanism:
    """可塑性机制基类。子类实现 step(), 可选 init_for()。"""

    name: str = "base"

    def init_for(self, unit) -> None:
        """绑定目标单元 (读取其权重形状初始化状态)。"""
        pass

    def step(self, stats: UnitStats, global_step: int) -> None:
        """每训练步调用, 基于本批激活统计更新机制内部状态 + 调整权重。"""
        raise NotImplementedError


class PlasticityPipeline:
    """多个可塑性机制的组合 (阶梯: B0+B1+B2+B3 = pipeline of all)。"""

    def __init__(self, mechanisms: list[PlasticityMechanism]):
        self.mechanisms = list(mechanisms)

    def init_for(self, unit) -> None:
        for m in self.mechanisms:
            m.init_for(unit)

    def step(self, stats: UnitStats, global_step: int) -> None:
        for m in self.mechanisms:
            m.step(stats, global_step)

    def names(self) -> list[str]:
        return [m.name for m in self.mechanisms]


def register_mechanism(name: str):
    """机制注册表: 从配置字符串创建可插拔机制。"""

    def deco(cls):
        PlasticityMechanism.registry[name] = cls
        return cls

    return deco


PlasticityMechanism.registry: dict[str, type[PlasticityMechanism]] = {}
