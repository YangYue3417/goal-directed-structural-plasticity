"""train_math_logic.py — 符号化数学推理逻辑 (LIF 学习验证)。

数学推理 (可计算验证):
  ① 等式传递: a=b, b=c → a=c (有效); a=b, b=c → a=d (无效)
  ② 算术验证: a+b=c 前提 → 结论 b+a=c (交换, 有效); a+b=d (无效)
  ③ 结合: (a+b)+c vs a+(b+c) — 结合律

符号: 数字 0-9, +, =, 变量
架构: 符号嵌入 → LIF 序列积累 → 判断 (有效/无效/不确定)
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

TOKENS = ["<pad>", "+", "=", "v"]  # v = 变量
for i in range(10): TOKENS.append(str(i))
VOCAB = {t: i for i, t in enumerate(TOKENS)}


def enc(s): return [VOCAB[t] for t in s]


def eval_expr(expr):
    """计算符号表达式 (数字/变量/+-) 值; 变量不可算返回 None。"""
    parts = expr.split("+")
    total = 0
    for p in parts:
        p = p.strip()
        if p == "v": return None
        total += int(p)
    return total


def judge_math(premises, concl):
    """数学推理: 前提集 → 结论 (有效0/无效1/不确定2)。可计算验证。"""
    # 前提: [a=b] 或 [a+b=c] 形式 (token 列表)
    # 简化: 全部前提与结论 = 等式, 检查结论是否由前提推出
    # 传递: a=b, b=c → a=c
    eqs = []
    for pr in premises:
        s = "".join(pr)
        if "=" in s:
            l, r = s.split("=")
            eqs.append((l, r))
    c = "".join(concl)
    if "=" not in c: return 2
    cl, cr = c.split("=")
    # 计算两边值
    vl, vr = eval_expr(cl), eval_expr(cr)
    if vl is not None and vr is not None:
        return 0 if vl == vr else 1  # 可计算: 相等/不等
    # 变量: 尝试传递 (a=b, b=c → a=c)
    if len(eqs) >= 2:
        (a1, b1), (a2, b2) = eqs[0], eqs[1]
        if b1 == a2 and cl == a1 and cr == b2: return 0  # 传递链
        if b1 == b2 and cl == a1 and cr == a2: return 0  # 等量代换
    return 2  # 无法确定


def gen_data(n=20000, seed=42):
    rng = np.random.RandomState(seed)
    P, C, Y = [], [], []
    for _ in range(n):
        typ = rng.randint(3)
        a, b, c = rng.randint(9) + 1, rng.randint(9) + 1, rng.randint(9) + 1
        if typ == 0:  # 传递: a=b, b=c → a=c (有效) 或 a=d (无效)
            d = c if rng.rand() < 0.5 else (c + 1) % 10
            pre = [["v", "=", str(b)], ["v", "=", str(c)]]
            concl = ["v", "=", str(d)]
            Y.append(0 if d == c else 1)
        elif typ == 1:  # 算术: a+b=c → b+a=c (交换有效) / a+b=d (无效)
            d = c if rng.rand() < 0.5 else (c + 1) % 10
            pre = [[str(a), "+", str(b), "=", str(c)]]
            concl = [str(b), "+", str(a), "=", str(d)]
            Y.append(0 if d == c else 1)
        else:  # 变量代换: v=b, v=c (b≠c) → 矛盾 (不确定)
            pre = [["v", "=", str(b)], ["v", "=", str(c)]]
            concl = ["v", "=", str(c)]
            Y.append(2 if b != c else 0)
        P.append(pre); C.append(concl)
    # pad 前提 (2 前提 × 5 token) 和结论
    Pp = np.zeros((len(P), 2, 5), np.int64)
    for i, pre in enumerate(P):
        for j, pr in enumerate(pre):
            Pp[i, j, :len(pr)] = enc(pr)
    Cp = np.zeros((len(C), 5), np.int64)
    for i, c_ in enumerate(C):
        Cp[i, :len(c_)] = enc(c_)
    return (torch.from_numpy(Pp).long(), torch.from_numpy(Cp).long(),
            torch.from_numpy(np.array(Y)).long())


class MathModel(nn.Module):
    """符号嵌入 → 前提 LIF 积累 → 结论 LIF → 判断。"""
    def __init__(self, vocab, d=64, pool=512, theta=0.3):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), d)
        self.pre_lif = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=16)
        self.concl_lif = LIFPool(d, pool, theta=theta, tau_min=2, tau_max=16)
        self.judge = nn.Linear(d * 3, 3)   # 2 前提状态 + 结论状态

    def forward(self, P, C):
        # 前提 (B, 2, 5) → 每个前提 LIF → 拼接
        outs = []
        for j in range(P.shape[1]):
            out, _ = self.pre_lif(torch.tanh(self.embed(P[:, j])))
            outs.append(out[:, -1])
        p_state = torch.cat(outs, -1)
        out, _ = self.concl_lif(torch.tanh(self.embed(C)))
        c_state = out[:, -1]
        return self.judge(torch.cat([p_state, c_state], -1))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 符号化数学推理 (LIF): 传递/交换/变量 ===")
    P, C, Y = gen_data(20000)
    n = int(0.9 * len(P))
    Pd, Cd, Yd = P.to(dev), C.to(dev), Y.to(dev)
    model = MathModel(VOCAB).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for ep in range(60):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            loss = F.cross_entropy(model(Pd[idx], Cd[idx]), Yd[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 15 == 14:
            model.eval()
            with torch.no_grad():
                acc_tr = (model(Pd[:n], Cd[:n]).argmax(-1) == Yd[:n]).float().mean()
                acc_te = (model(Pd[n:], Cd[n:]).argmax(-1) == Yd[n:]).float().mean()
            print(f"  epoch {ep+1}: 训练 {acc_tr.item():.3f} 测试 {acc_te.item():.3f}")

    # 具体数学推理验证
    LABELS = ["有效", "无效", "不确定"]
    def show(pre, concl):
        pr = [[VOCAB[t] for t in x] for x in pre]
        cc = [VOCAB[t] for t in concl]
        Pp = torch.zeros(1, 2, 5, dtype=torch.long).to(dev)
        for j, x in enumerate(pr): Pp[0, j, :len(x)] = torch.tensor(x)
        Cc = torch.zeros(1, 5, dtype=torch.long).to(dev)
        Cc[0, :len(cc)] = torch.tensor(cc)
        with torch.no_grad():
            pred = int(model(Pp, Cc).argmax(-1))
        gold = judge_math(pre, concl)
        pstr = " ".join("".join(x) for x in pre)
        print(f"前提 [{pstr}] → 结论 {''.join(concl)} = 预测 {LABELS[pred]} | 规则 {LABELS[gold]} "
              f"{'✓' if pred==gold else '✗'}")

    show([["v","=","3"], ["v","=","3"]], ["v","=","3"])       # 传递有效
    show([["v","=","3"], ["v","=","4"]], ["v","=","4"])       # 矛盾 → 不确定
    show([["2","+","3","=","5"]], ["3","+","2","=","5"])      # 交换有效
    show([["2","+","3","=","5"]], ["2","+","3","=","6"])      # 无效
    show([["1","+","2","=","3"], ["3","+","4","=","7"]], ["1","+","2","+","4","=","7"])  # 组合


if __name__ == "__main__":
    main()
