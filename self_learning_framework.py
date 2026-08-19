"""self_learning_framework.py — 自主学习框架 (o1 式: 生成-验证-改进)。

核心: 可计算验证 = 自动奖励 (无人工标注! 数字/逻辑任务的独特优势)

循环:
  ① explore (探索): 模型生成尝试 (数数/推理/算术)
  ② verify (验证): 可计算检查正确性 (next=n+1, 3+2=5)
  ③ learn (改进): 验证结果 → 训练 (正确强化 / 错误修正)
  ④ curriculum (扩展): 能力提升 → 尝试更难 (自我驱动课程)
  ⑤ log (记录): 能力轨迹 (学了多少/学到哪)

vs o1: o1 用 RLHF/PRM (人工奖励); 我们用计算验证 (0 成本, 可解释)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LearningLog:
    """学习轨迹记录 (能力曲线)。"""
    rounds: list = field(default_factory=list)   # 每轮: {round, reach, correct, new_learned}
    errors: list = field(default_factory=list)   # 错误模式 (可解释)
    started: str = field(default_factory=lambda: datetime.now().isoformat())

    def record(self, rnd, reach, n_correct, n_total, new_learned, errors=None):
        self.rounds.append({"round": rnd, "reach": reach,
                            "correct": n_correct, "total": n_total,
                            "new": new_learned})
        if errors:
            self.errors.extend(errors)

    def summary(self):
        if not self.rounds:
            return "无记录"
        last = self.rounds[-1]
        return (f"{len(self.rounds)} 轮 | 最新到达: {last['reach']} "
                f"(正确 {last['correct']}/{last['total']}, 新增 {last['new']})")

    def save(self, path="runs/self_learning.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"started": self.started, "rounds": self.rounds,
                       "errors": self.errors}, f, ensure_ascii=False, indent=2)


class SelfLearningFramework:
    """自主学习框架 (领域无关, 验证器可插拔)。"""

    def __init__(self, model, optimizer, verifier, curriculum,
                 explore_fn, learn_fn,
                 device="cuda", log_path="runs/self_learning.json"):
        self.model = model
        self.optimizer = optimizer
        self.verify = verifier        # (生成结果) → (正确/错误, 修正) — 可计算!
        self.curriculum = curriculum  # (能力) → 目标难度
        self._explore_fn = explore_fn  # 领域生成 (注入)
        self._learn_fn = learn_fn      # 领域学习 (注入)
        self.device = torch.device(device)
        self.log = LearningLog()
        self.log_path = log_path

    def explore(self, target):
        """① 生成尝试: 领域注入。"""
        return self._explore_fn(self, target)

    def learn(self, samples):
        """③ 改进: 领域注入。"""
        return self._learn_fn(self, samples)

    def run(self, n_rounds=20, verbose=True):
        """自主学习主循环。"""
        for rnd in range(n_rounds):
            t0 = time.time()
            # ① 探索: 当前能力 → 目标难度
            target = self.curriculum(self.log)
            samples, meta = self.explore(target)
            # ② 验证: 可计算检查 (自动奖励!)
            verified, errors = self.verify(samples)
            # ③ 学习: 修正训练
            new_learned = self.learn(verified)
            # ④ 记录
            self.log.record(rnd, meta.get("reach", 0),
                            meta.get("n_correct", 0), meta.get("n_total", 0),
                            new_learned, errors)
            if verbose:
                print(f"  轮 {rnd+1}: {self.log.summary()} | {time.time()-t0:.1f}s")
            self.log.save(self.log_path)
        return self.log


# ============ 数字域: 数数 (可计算验证) ============
class CountingPolicy(nn.Module):
    """位置系统数数策略: [十位,个位] → 下一序列。"""
    def __init__(self, n_digit=10, d=32, pool=256, theta=0.3):
        super().__init__()
        from lif_pool import LIFPool
        self.embed = nn.Embedding(n_digit, d)
        self.lif = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=12)
        self.head_t = nn.Linear(d, 10)
        self.head_o = nn.Linear(d, 10)

    def forward(self, seq):
        out, _ = self.lif(torch.tanh(self.embed(seq)))
        h = out[:, -1]
        return self.head_t(h), self.head_o(h)

    def predict_next(self, n, device):
        """预测 n 的下一数字。"""
        seq = torch.tensor([[n // 10, n % 10]]).to(device)
        with torch.no_grad():
            pt, po = self(seq)
        return int(pt.argmax(-1)) * 10 + int(po.argmax(-1))


def make_counting_domain(device="cuda"):
    """数字数数领域: 探索/验证/学习 的具体实现。"""
    model = CountingPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    def curriculum(log):
        """课程: 基于当前能力 (数到哪 → 目标扩展)。"""
        reach = log.rounds[-1]["reach"] if log.rounds else 0
        return max(10, reach + 20)  # 目标 = 当前 + 20 (渐进)

    def explore(fw, target):
        """生成: 从 0 开始数到 target (自主尝试)。"""
        chain = []
        cur = 0
        for _ in range(target + 1):
            chain.append(cur)
            nxt = fw.model.predict_next(cur, device)
            if nxt != cur + 1:
                break  # 数错 → 停 (试错边界)
            cur = nxt
        return chain, {"reach": chain[-1], "n_total": len(chain)}

    def verify(chain):
        """验证: next == n+1 (可计算!), 返回修正样本。"""
        samples = []
        errors = []
        for n in chain:
            nxt = model.predict_next(n, device)
            correct = (nxt == n + 1)
            samples.append((n, n + 1))  # 修正 = 计算答案
            if not correct:
                errors.append({"n": n, "predicted": nxt, "correct": n + 1})
        return samples, errors
    # (verify 用闭包 model — 保留)

    def learn(fw, samples):
        """训练: 用验证修正的 (n → n+1) 监督。"""
        import torch.nn.functional as F
        from train_counting import to_seq
        fw.model.train()
        tr_in = torch.tensor([to_seq(n) for n, _ in samples]).to(device)
        tr_t = torch.tensor([t // 10 for _, t in samples]).to(device)
        tr_o = torch.tensor([t % 10 for _, t in samples]).to(device)
        for _ in range(30):
            pt, po = fw.model(tr_in)
            loss = F.cross_entropy(pt, tr_t) + F.cross_entropy(po, tr_o)
            fw.optimizer.zero_grad(); loss.backward(); fw.optimizer.step()
        return len(samples)

    return SelfLearningFramework(model, opt, verify, curriculum,
                                 explore, learn, device=device), model


if __name__ == "__main__":
    print("=== 自主学习框架 (o1 式: 生成-验证-改进, 计算验证) ===")
    fw, model = make_counting_domain()
    log = fw.run(n_rounds=12)
    print(f"\n最终: {log.summary()}")
    # 最终自主能力
    chain, _ = fw.explore(200)
    print(f"自主数数: {chain}")
    print(f"数到: {chain[-1]} (0-9=单数, 10+=进位!)")
    print(f"{'✅ 自主学习: 模型自己数出进位!' if chain[-1] > 9 else '⚠️ 待更多轮'}")
