"""train_logic_lif.py — 符号逻辑序列 + LIF 逻辑理解 (可恢复训练)。

任务: 符号规则链 A→B→C→D→A
  一致序列 (如 A,B): 下一符号确定 (C) — 逻辑推理
  跳跃序列 (如 A,C): 规则无定义 → "不确定" (宁缺毋滥)

架构: 符号嵌入 → LIF 池 (时间积分, 完整动力学链)
      → 逻辑头 (下一符号分布 + 不确定概率)

日志完备: TeeLogger (时间/step/loss/acc/不确定率/配置)
可恢复: checkpoint (model/optimizer/epoch/best) + --resume
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

# ============ 数据: 符号逻辑链 ============
SYMBOLS = ["A", "B", "C", "D"]
RULE = {"A": "B", "B": "C", "C": "D", "D": "A"}  # A→B→C→D→A


class TeeLogger:
    """完备日志: 控制台 + 文件, 时间戳, 可重定向。"""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def gen_data(n=20000, seed=42, max_len=4):
    """生成序列: 一致链 (下一符号确定) vs 跳跃链 (不确定)。"""
    rng = np.random.RandomState(seed)
    X, Y, confident = [], [], []
    for _ in range(n):
        L = rng.randint(1, max_len)
        start = rng.randint(4)
        seq = [(start + i) % 4 for i in range(L)]
        X.append(seq)
        if rng.rand() < 0.7:
            # 一致: 下一符号 = 规则 (确定)
            nxt = RULE[SYMBOLS[seq[-1]]]
            Y.append(SYMBOLS.index(nxt))
            confident.append(1.0)
        else:
            # 跳跃: 随机符号 (规则外 → 不确定)
            wrong = rng.randint(4)
            Y.append(wrong)
            confident.append(0.0)
    return X, Y, confident


class LogicModel(nn.Module):
    """符号嵌入 + LIF 池 (时间积分) + 逻辑头 (确定符号 + 不确定)。"""
    def __init__(self, n_sym=4, d=64, pool=512, theta=0.3,
                 tau_min=2.0, tau_max=24.0):
        super().__init__()
        self.embed = nn.Embedding(n_sym, d)
        self.pool = LIFPool(d, pool, tau_min=tau_min, tau_max=tau_max,
                            theta=theta)
        self.head = nn.Linear(d, n_sym + 1)  # 4 符号 + 1 不确定

    def forward(self, seq, train_pool=True):
        """seq: (B, L) 符号索引 → 逻辑输出 (B, n_sym+1)。"""
        B, L = seq.shape
        z = self.embed(seq)          # (B, L, d)
        # 用 LIF 池 (pad 0 符号仍积分 — 逻辑在有效位置)
        out, spikes = self.pool(z)   # LIF 时间积分 (B, L, d)
        h = out[:, -1]               # 最后状态 (逻辑积累)
        logits = self.head(h)        # (B, 5): 4 符号 + 不确定
        return logits, spikes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data_n", type=int, default=20000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default=None, help="checkpoint 路径")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--log", type=str, default="runs/logic_lif.log")
    p.add_argument("--ckpt", type=str, default="runs/logic_lif.pt")
    args = p.parse_args()
    dev = torch.device(args.device)
    logger = TeeLogger(args.log)
    logger.log(f"=== 训练启动 {datetime.now()} ===")
    logger.log(f"配置: {json.dumps(vars(args))}")

    # 数据 (pad 到等长)
    X, Y, C = gen_data(args.data_n)
    max_len = max(len(s) for s in X)
    Xp = np.zeros((len(X), max_len), np.int64)
    for i, s in enumerate(X):
        Xp[i, :len(s)] = s
    X = torch.from_numpy(Xp).long()
    Y = torch.from_numpy(np.array(Y)).long()
    C = torch.from_numpy(np.array(C)).float()
    n_train = int(0.9 * len(X))
    Xtr, Ytr, Ctr = X[:n_train].to(dev), Y[:n_train].to(dev), C[:n_train].to(dev)
    Xte, Yte, Cte = X[n_train:].to(dev), Y[n_train:].to(dev), C[n_train:].to(dev)
    logger.log(f"数据: 训练 {n_train}, 测试 {len(X)-n_train} (一致率 {C.mean().item():.2f})")

    # 模型
    model = LogicModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_acc = 0.0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_acc = ck.get("best_acc", 0.0)
        logger.log(f"恢复训练: epoch {start_epoch} (best_acc {best_acc:.3f})")

    # 训练
    n_steps = 0
    for ep in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        losses, accs, unc_accs = [], [], []
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            seq = Xtr[idx]
            # 补齐到等长 (padding = 0 符号, mask 后)
            logits, spikes = model(seq)
            target = Ytr[idx]
            conf = Ctr[idx]
            # 损失: 确定样本 → 下一符号 CE; 不确定样本 → 不确定类
            log_p = F.log_softmax(logits, -1)
            loss_conf = -(log_p[range(len(idx)), target] * conf).mean()
            loss_unc = -(log_p[:, -1] * (1 - conf)).mean()
            loss = loss_conf + loss_unc
            opt.zero_grad(); loss.backward(); opt.step()
            n_steps += 1
            # 统计
            pred = logits.argmax(-1)
            acc = ((pred == target) * (conf > 0)).sum() / max((conf > 0).sum(), 1)
            unc_acc = ((pred == 4) * (conf < 1)).sum() / max((conf < 1).sum(), 1)
            losses.append(loss.item()); accs.append(acc.item()); unc_accs.append(unc_acc.item())
            if n_steps % 200 == 0:
                logger.log(f"  step {n_steps}: loss {np.mean(losses):.4f} "
                           f"acc {np.mean(accs):.3f} 不确定率 {np.mean(unc_accs):.3f}")
        # 评估
        model.eval()
        with torch.no_grad():
            logits, _ = model(Xte)
            pred = logits.argmax(-1)
            acc = ((pred == Yte) * (Cte > 0)).sum() / (Cte > 0).sum()
            unc_acc = ((pred == 4) * (Cte < 1)).sum() / (Cte < 1).sum()
            acc_all = (pred == Yte).float().mean()
        dt = time.time() - t0
        logger.log(f"epoch {ep+1}/{args.epochs}: loss {np.mean(losses):.4f} "
                   f"| 确定acc {acc.item():.3f} 不确定acc {unc_acc.item():.3f} "
                   f"总acc {acc_all.item():.3f} | {dt:.1f}s "
                   f"| 活跃神经 {model.pool.n_active()}")
        # checkpoint
        if (ep + 1) % args.save_every == 0 or acc.item() > best_acc:
            best_acc = max(best_acc, acc.item())
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": ep,
                "best_acc": best_acc,
                "config": vars(args),
                "n_steps": n_steps,
            }, args.ckpt)
            logger.log(f"  ✓ checkpoint 保存 (best_acc {best_acc:.3f})")

    logger.log(f"=== 训练完成: best_acc {best_acc:.3f} ===")
    logger.close()


if __name__ == "__main__":
    main()
