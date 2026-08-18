"""train_syllogism.py — 三段论推理验证 (大前提/小前提/结论)。

大前提: 所有[Y]会[P]  (∀x: Y(x)→P(x))
小前提: [X]是[Y]       (X⊆Y)
结论:   [X]会[P]       → 有效 (传递推理: X⊆Y⊆会P)

验证类型:
  有效 (蕴含):  所有动物会进食 + 人是动物 → 人会进食 ✓
  无效 (逆推):  所有动物会进食 + 人会进食 → 人是动物 ✗ (共因)
  不确定:       信息不足/不匹配

架构: 词嵌入 → 句 LIF 编码 → 三段序列 LIF 积累 → 判断 (有效/无效/不确定)
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

NOUNS = ["动物", "人", "鸟", "鱼", "猫"]
ATTRS = ["进食", "呼吸", "移动"]
VOCAB = {"<pad>": 0}
for n in NOUNS: VOCAB[n] = len(VOCAB)
for a in ATTRS: VOCAB[a] = len(VOCAB)
for t in ["所有", "会", "不会", "是", "不是"]:
    VOCAB[t] = len(VOCAB)


def enc(s): return [VOCAB[t] for t in s]


def judge(major, minor, concl):
    """三段论: 有效(0) / 无效(1) / 不确定(2)。可验证规则。"""
    # 大: [所有, Y, 会/不会, P]  小: [X, 是/不是, Y]  结: [X, 会/不会, P]
    _, Y, mv, P = major
    X, sub, Y2 = minor
    X2, cv, P2 = concl
    if X != X2 or P != P2:
        return 2  # 结论不匹配 → 不确定
    if Y != Y2:
        return 2
    # 小前提 X 是 Y, 大前提 Y 会/不会 P
    if sub == "是":
        if mv == "会" and cv == "会": return 0   # 有效: X⊆Y⊆会P
        if mv == "不会" and cv == "不会": return 0  # 有效: X⊆Y⊆不会P
        if mv == "会" and cv == "不会": return 1  # 矛盾 (无效)
        if mv == "不会" and cv == "会": return 1  # 矛盾
    if sub == "不是":
        if mv == "会" and cv == "不会": return 0  # 逆否: X∉Y, 逆否? (简化: 有效)
        if mv == "不会" and cv == "会": return 0
        return 2
    return 2


def gen_data(n=20000, seed=42):
    rng = np.random.RandomState(seed)
    M, m, C, Y = [], [], [], []
    for _ in range(n):
        Yn = NOUNS[rng.randint(5)]
        X = NOUNS[rng.randint(5)]
        P = ATTRS[rng.randint(3)]
        mv = "会" if rng.rand() < 0.7 else "不会"
        sub = "是" if rng.rand() < 0.7 else "不是"
        cv = "会" if rng.rand() < 0.5 else "不会"
        major = ["所有", Yn, mv, P]
        minor = [X, sub, Yn]
        concl = [X, cv, P]
        M.append(enc(major)); m.append(enc(minor)); C.append(enc(concl))
        Y.append(judge(major, minor, concl))
    max_len = 4
    def pad(lst):
        a = np.zeros((len(lst), max_len), np.int64)
        for i, s in enumerate(lst): a[i, :len(s)] = s
        return torch.from_numpy(a).long()
    return pad(M), pad(m), pad(C), torch.from_numpy(np.array(Y)).long()


class SyllogismModel(nn.Module):
    """词嵌入 → 句 LIF → 三段序列 LIF 积累 → 判断。"""
    def __init__(self, vocab, d=64, pool=512, theta=0.3):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), d)
        self.sent_lif = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=16)
        self.sent_head = nn.Linear(d, d)
        self.chain_lif = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=24)
        self.judge = nn.Linear(d, 3)

    def forward(self, M, m, C):
        def enc_sent(s):
            out, _ = self.sent_lif(torch.tanh(self.embed(s)))
            return self.sent_head(out[:, -1])
        seq = torch.stack([enc_sent(M), enc_sent(m), enc_sent(C)], 1)
        out, _ = self.chain_lif(seq)
        return self.judge(out[:, -1])


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 三段论推理: 大前提/小前提/结论 ===")
    M, m, C, Y = gen_data(20000)
    n = int(0.9 * len(M))
    Md, md, Cd, Yd = M.to(dev), m.to(dev), C.to(dev), Y.to(dev)
    model = SyllogismModel(VOCAB).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for ep in range(60):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            loss = F.cross_entropy(model(Md[idx], md[idx], Cd[idx]), Yd[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 15 == 14:
            model.eval()
            with torch.no_grad():
                acc_tr = (model(Md[:n], md[:n], Cd[:n]).argmax(-1) == Yd[:n]).float().mean()
                acc_te = (model(Md[n:], md[n:], Cd[n:]).argmax(-1) == Yd[n:]).float().mean()
            print(f"  epoch {ep+1}: 训练 {acc_tr.item():.3f} 测试 {acc_te.item():.3f}")

    # 具体三段论验证
    LABELS = ["有效", "无效", "不确定"]
    tests = [
        (["所有","动物","会","进食"], ["人","是","动物"], ["人","会","进食"]),   # 有效!
        (["所有","动物","会","进食"], ["人","会","进食"], ["人","是","动物"]),   # 无效 (共因)
        (["所有","动物","会","进食"], ["人是动物"], ["人会进食"]),  # 演示
    ]
    model.eval()
    with torch.no_grad():
        for major, minor, concl in tests[:2]:
            if len(minor) == 1: continue
            Mt = torch.tensor([enc(major)]).to(dev)
            mt = torch.tensor([enc(minor)]).to(dev)
            Ct = torch.tensor([enc(concl)]).to(dev)
            pred = int(model(Mt, mt, Ct).argmax(-1))
            gold = judge(major, minor, concl)
            print(f"大'{''.join(major)}' 小'{''.join(minor)}' → 结论'{''.join(concl)}' "
                  f"= 预测 {LABELS[pred]} | 规则 {LABELS[gold]} {'✓' if pred==gold else '✗'}")


if __name__ == "__main__":
    main()
