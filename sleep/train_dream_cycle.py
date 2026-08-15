"""train_dream_cycle.py — 多日昼夜循环: 每晚做梦重放全部历史 + 降标。

大脑: 每晚 ripple 重放旧记忆 (不只是当天的) → 旧记忆持久。
对照: 无梦 (同样学习, 无夜间重放/降标)。

日程:
  Day k: 学新任务组 (0.8×N_lim)
  Night k: 做梦 = 重放全部历史任务 (小 lr) + 突触降标
  Morning k: 测所有已学任务 (记忆保持)

任务组: A=0-3, B=4-6, C=7-9
指标: 每天早上各任务准确率 (遗忘率), 权重范数变化
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from train_mnist import VisionSparseModel

TASKS = {"A": [0, 1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
ORDER = ["A", "B", "C"]


def load_mnist(device):
    img = np.load("data/mnist/train_img.npy")
    lab = np.load("data/mnist/train_lab.npy")
    t_img = np.load("data/mnist/test_img.npy")
    t_lab = np.load("data/mnist/test_lab.npy")
    x = torch.from_numpy(img).float().unsqueeze(1) / 255.0
    y = torch.from_numpy(lab).long()
    tx = torch.from_numpy(t_img).float().unsqueeze(1) / 255.0
    ty = torch.from_numpy(t_lab).long()
    return x.to(device), y.to(device), tx.to(device), ty.to(device)


def acc(model, x, y, digits, bs=256):
    model.eval()
    idx = torch.isin(y, torch.tensor(digits, device=y.device))
    sel = torch.nonzero(idx).flatten()
    if len(sel) == 0:
        return 0.0
    correct = 0
    with torch.no_grad():
        for i in range(0, len(sel), bs):
            s = sel[i:i + bs]
            logits, _ = model(x[s])
            correct += (logits.argmax(-1) == y[s]).sum().item()
    return correct / len(sel)


def train_steps(model, opt, x, y, digits, steps, bs=128):
    model.train()
    idx = torch.nonzero(torch.isin(y, torch.tensor(digits, device=y.device))).flatten()
    for _ in range(steps):
        order = torch.randperm(len(idx))[:bs]
        s = idx[order]
        opt.zero_grad()
        logits, _ = model(x[s])
        F.cross_entropy(logits, y[s]).backward()
        opt.step()


def dream(model, x, y, past_digits, replay_epochs=2, scale=0.9, lr=1e-4):
    """夜: 重放全部历史 (小 lr) + 突触降标。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    idx = torch.nonzero(torch.isin(y, torch.tensor(past_digits, device=y.device))).flatten()
    n = len(idx)
    for _ in range(replay_epochs):
        order = torch.randperm(n)
        for i in range(0, n, 128):
            s = idx[order[i:i + 128]]
            opt.zero_grad()
            logits, _ = model(x[s])
            F.cross_entropy(logits, y[s]).backward()
            opt.step()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, tx, ty = load_mnist(device)
    n_lim = 200  # 预扫结果 (A 饱和步数); 0.8x = 160

    print("=== 多日循环 (0.8x 活动 + 每晚做梦) ===", flush=True)
    for arm, do_dream in [("做梦", True), ("无梦", False)]:
        model = VisionSparseModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        past = []
        print(f"[{arm}]", flush=True)
        for day, task in enumerate(ORDER):
            # Day: 学新任务 (0.8×N_lim)
            train_steps(model, opt, x, y, TASKS[task], int(n_lim * 0.8))
            past.extend(TASKS[task])
            # Night: 做梦重放全部历史 + 降标
            if do_dream:
                w_b = sum(p.norm().item() for p in model.parameters())
                dream(model, x, y, past, scale=0.9)
                w_a = sum(p.norm().item() for p in model.parameters())
            # Morning: 测所有已学任务
            accs = " ".join(f"{t}={acc(model, tx, ty, TASKS[t]):.3f}"
                            for t in ORDER[:day + 1])
            norm = f" (范数{w_b:.0f}→{w_a:.0f})" if do_dream else ""
            print(f"  晨{day+1} {accs}{norm}", flush=True)


if __name__ == "__main__":
    main()
