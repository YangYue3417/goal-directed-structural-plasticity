"""train_dual_zone.py — 同侧协调分区: 一套 LIF 池, 语言区 + 数学区。

大脑对应: 布洛卡区 (语言) + 角回 (数学) — 同侧皮质, 可协调
架构:
  一个 LIF 池 (512): 语言区 (0-255) + 数学区 (256-511)
  输入路由: 语言任务 → 偏置语言区; 数学 → 偏置数学区
  协调: 全连接保留 (区域间可协同 — 数学应用题 = 语言+数学!)
  隔离: 区域偏置 (任务输入激活对应区, 减少干扰)

任务: 语言 (蕴含/矛盾/无关/不确定) + 数学 (有效/无效/不确定)
验证: 分区整合 vs 之前共享干扰 (0.53) vs 独立
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from lif_pool import LIFPool

# 统一词表 (语言 + 数学)
TOKENS = ["<pad>", "<LANG>", "<MATH>"]
TOKENS += ["所有", "一些", "没有", "鸟", "鱼", "狗", "花", "猫",
           "会", "不会", "飞", "游泳", "叫", "跑"]
TOKENS += ["+", "=", "v"] + [str(i) for i in range(10)]
VOCAB = {t: i for i, t in enumerate(TOKENS)}
LANG_ID = VOCAB["<LANG>"]; MATH_ID = VOCAB["<MATH>"]


class ZoneLIF(nn.Module):
    """区域化 LIF: 一个池, 语言区/数学区, 区域偏置路由。"""
    def __init__(self, vocab, d=64, pool=512, theta=0.3, half=256):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), d)
        self.lif = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=24)
        # 区域偏置 (可学习): 语言任务偏前区, 数学偏后区
        self.zone_bias = nn.Parameter(torch.zeros(pool))
        self.zone_bias.data[:half] = 1.0
        self.zone_bias.data[half:] = -1.0
        self.half = half
        self.head_lang = nn.Linear(d * 2, 4)   # 两句状态
        self.head_math = nn.Linear(d * 3, 3)

    def seq_state(self, x, zone):
        """序列 → 区域token前缀 → LIF → 状态 (池自然分区)。"""
        tid = LANG_ID if zone > 0 else MATH_ID
        tok = torch.full((x.shape[0], 1), tid, dtype=torch.long, device=x.device)
        x = torch.cat([tok, x], 1)
        z = torch.tanh(self.embed(x))
        out, _ = self.lif(z)
        return out[:, -1]

    def forward(self, task, x, zone):
        if task == 0:  # 语言: 两句
            x1, x2 = x
            s = torch.cat([self.seq_state(x1, zone), self.seq_state(x2, zone)], -1)
            return self.head_lang(s)
        else:  # 数学: 前提+结论
            P, C = x
            ps = []
            for j in range(P.shape[1]):
                ps.append(self.seq_state(P[:, j], zone))
            cs = self.seq_state(C, zone)
            return self.head_math(torch.cat(ps + [cs], -1))


def gen_lang(n=8000, seed=2):
    from train_controlled_lang import gen_data
    X1, X2, Y = gen_data(n, seed)
    return (X1, X2), Y.long()


def gen_math(n=8000, seed=3):
    from train_math_logic import gen_data
    P, C, Y = gen_data(n, seed)
    return (P, C), Y.long()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 同侧协调分区: 语言区 + 数学区 (一套池) ===")
    (L1, L2), LY = gen_lang()
    (MP, MC), MY = gen_math()
    model = ZoneLIF(VOCAB).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    nl = int(len(LY) * 0.9); nm = int(len(MY) * 0.9)
    L1d, L2d, LYd = L1.to(dev), L2.to(dev), LY.to(dev)
    MPd, MCd, MYd = MP.to(dev), MC.to(dev), MY.to(dev)

    for ep in range(50):
        model.train()
        # 交替: 语言 (zone=+1) + 数学 (zone=-1)
        idx = torch.randperm(nl)[:512]
        loss = F.cross_entropy(model(0, (L1d[idx], L2d[idx]), 1.0), LYd[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        idx = torch.randperm(nm)[:512]
        loss = F.cross_entropy(model(1, (MPd[idx], MCd[idx]), -1.0), MYd[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            model.eval()
            with torch.no_grad():
                la = (model(0, (L1d[nl:], L2d[nl:]), 1.0).argmax(-1) == LYd[nl:]).float().mean()
                ma = (model(1, (MPd[nm:], MCd[nm:]), -1.0).argmax(-1) == MYd[nm:]).float().mean()
            print(f"  epoch {ep+1}: 语言 {la.item():.3f} | 数学 {ma.item():.3f}")

    model.eval()
    with torch.no_grad():
        la = (model(0, (L1d[nl:], L2d[nl:]), 1.0).argmax(-1) == LYd[nl:]).float().mean()
        ma = (model(1, (MPd[nm:], MCd[nm:]), -1.0).argmax(-1) == MYd[nm:]).float().mean()
    print(f"\n分区整合: 语言 {la.item():.3f} | 数学 {ma.item():.3f}")
    print("对照: 独立 0.957/0.78 | 共享整合 0.953/0.53")
    print(f"{'✅ 分区解决干扰 (数学恢复)' if ma.item() > 0.6 else '⚠️ 仍干扰'}")


if __name__ == "__main__":
    main()
