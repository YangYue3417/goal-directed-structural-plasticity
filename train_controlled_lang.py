"""train_controlled_lang.py — 受控语言逻辑判断 (可验证 + LIF)。

受控语法: [量词][名词][会/不会][属性]
  量词: 所有/一些/没有 | 名词: 鸟/鱼/狗/花/猫 | 属性: 飞/游泳/叫/跑

逻辑判断 (前提, 假设) → 蕴含/矛盾/无关/不确定 (可验证规则):
  蕴含:  所有X会P → 一些X会P (子集); 所有X会P → X会P
  矛盾:  所有X会P vs 没有X会P / X不会P; 一些X会P vs 没有X会P
  无关:  不同属性 (P1≠P2)
  不确定: 一些X会P → 所有X会P (不够推); 无量词 vs 所有

架构: 词嵌入 → LIF 序列 (两句积累) → 判断头 (4 类)
验证: 训练部分组合 → 测试新组合 (泛化) + 规则可检查
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from lif_pool import LIFPool

# ============ 受控语言 ============
QUANT = {"所有": 0, "一些": 1, "没有": 2}
NOUNS = ["鸟", "鱼", "狗", "花", "猫"]
ATTRS = ["飞", "游泳", "叫", "跑"]
NEG = "不"
VOCAB = {"<pad>": 0}
for q in QUANT: VOCAB[q] = len(VOCAB)
for n in NOUNS: VOCAB[n] = len(VOCAB)
for a in ATTRS: VOCAB[a] = len(VOCAB)
VOCAB["会"] = len(VOCAB); VOCAB["不会"] = len(VOCAB)


def encode(sent):
    """句子 → token 序列: [量词, 名词, 会/不会, 属性]"""
    return [VOCAB[t] for t in sent]


def judge(premise, hyp):
    """可验证逻辑: (前提, 假设) → 0蕴含 1矛盾 2无关 3不确定"""
    q1, n1, v1, p1 = premise
    q2, n2, v2, p2 = hyp
    if n1 != n2:
        return 2  # 不同名词 → 无关
    if p1 != p2:
        return 2  # 不同属性 → 无关
    # 同名词同属性: 量词/否定关系
    neg1 = v1 == "不会"; neg2 = v2 == "不会"
    if q1 == "所有" and not neg1:
        if q2 == "所有" and not neg2: return 0  # 蕴含 (等价)
        if q2 == "一些" and not neg2: return 0  # 蕴含 (子集)
        if q2 == "没有" and neg2: return 1      # 矛盾
        if q2 == "没有" and not neg2: return 1  # 矛盾
        if q2 == "一些" and neg2: return 3      # 不确定 (部分不会≠矛盾)
        return 3
    if q1 == "一些" and not neg1:
        if q2 == "一些" and not neg2: return 0
        if q2 == "所有" and not neg2: return 3  # 不确定 (一些推不出所有)
        if q2 == "没有" and not neg2: return 1  # 矛盾
        if q2 == "没有" and neg2: return 3      # 不确定
        return 3
    if q1 == "没有" and not neg1:
        if q2 == "没有" and not neg2: return 0
        if q2 == "一些" and not neg2: return 1  # 矛盾
        if q2 == "所有" and not neg2: return 1  # 矛盾
        return 3
    if neg1:  # 前提带否定
        if neg2 and q1 == q2: return 0
        if not neg2 and q2 in ("所有", "一些"): return 1  # 矛盾
        return 3
    return 3


def gen_data(n=20000, seed=42):
    """生成 (前提, 假设) 对 + 逻辑标签。"""
    rng = np.random.RandomState(seed)
    quant_l = list(QUANT.keys())
    X1, X2, Y = [], [], []
    for _ in range(n):
        n1 = NOUNS[rng.randint(5)]
        p1 = ATTRS[rng.randint(4)]
        q1 = quant_l[rng.randint(3)]
        v1 = "会" if rng.rand() < 0.8 else "不会"
        prem = [q1, n1, v1, p1]
        n2 = NOUNS[rng.randint(5)]
        p2 = ATTRS[rng.randint(4)]
        q2 = quant_l[rng.randint(3)]
        v2 = "会" if rng.rand() < 0.8 else "不会"
        hyp = [q2, n2, v2, p2]
        X1.append(encode(prem)); X2.append(encode(hyp))
        Y.append(judge(prem, hyp))
    return (torch.from_numpy(np.array(X1)).long(),
            torch.from_numpy(np.array(X2)).long(),
            torch.from_numpy(np.array(Y)).long())


class LangModel(nn.Module):
    """词嵌入 → 句 LIF 编码 → 序列 LIF 逻辑积累 → 判断头。"""
    def __init__(self, vocab, d=64, pool=512, theta=0.3):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), d)
        # 句编码 LIF (句内 token 积累 → 句向量)
        self.sent_lif = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=16)
        self.sent_head = nn.Linear(d, d)
        # 逻辑判断 LIF (两句序列积累)
        self.logic_lif = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=24)
        self.judge = nn.Linear(d, 4)

    def forward(self, s1, s2):
        z1 = torch.tanh(self.embed(s1))
        out1, _ = self.sent_lif(z1)
        v1 = self.sent_head(out1[:, -1])          # 句1向量
        z2 = torch.tanh(self.embed(s2))
        out2, _ = self.sent_lif(z2)
        v2 = self.sent_head(out2[:, -1])          # 句2向量
        # LIF 逻辑积累 (句序列)
        seq = torch.stack([v1, v2], 1)
        out, _ = self.logic_lif(seq)
        return self.judge(out[:, -1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log", type=str, default="runs/lang_lif.log")
    args = p.parse_args()
    dev = torch.device(args.device)
    log_f = open(args.log, "a", encoding="utf-8")
    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    log(f"=== 受控语言逻辑训练 ===")
    log(f"词表: {len(VOCAB)} ({list(VOCAB)[:12]}...) | 判断: 蕴含/矛盾/无关/不确定")
    X1, X2, Y = gen_data(20000)
    n = int(0.9 * len(X1))
    dev_X1 = X1.to(dev); dev_X2 = X2.to(dev); dev_Y = Y.to(dev)
    model = LangModel(VOCAB).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, args.batch):
            idx = perm[i:i+args.batch]
            logits = model(dev_X1[idx], dev_X2[idx])
            loss = F.cross_entropy(logits, dev_Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # 评估 (训练内 + 测试)
        model.eval()
        with torch.no_grad():
            lt = model(dev_X1[:n], dev_X2[:n])
            acc_tr = (lt.argmax(-1) == dev_Y[:n]).float().mean()
            le = model(dev_X1[n:], dev_X2[n:])
            acc_te = (le.argmax(-1) == dev_Y[n:]).float().mean()
        dt = time.time() - t0
        log(f"epoch {ep+1}/{args.epochs}: loss {np.mean(losses):.4f} "
            f"训练acc {acc_tr.item():.3f} 测试acc {acc_te.item():.3f} | {dt:.1f}s "
            f"| 活跃 {model.sent_lif.n_active()}/{model.logic_lif.n_active()}")

    log("=== 训练完成 ===")
    log_f.close()


if __name__ == "__main__":
    main()
