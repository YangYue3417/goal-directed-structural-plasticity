"""walker_robustness.py — SR vs MC-V 策略的抗扰动评估。

扰动矩阵:
  1. 动作噪声: 执行 a + N(0, σ) (执行器扰动)
  2. 观测噪声: MPC 输入 obs + N(0, σ) (传感器扰动)
  3. 初始扰动: 起始 hull angle 偏移 (推离平衡)

对比: SR (安全访问价值) vs MC-V (奖励回报价值)
指标: 平均存活步数 (300 上限) + 得分, 以及相对基线的下降率
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from train_gdsp import GDSPModel, ValueNet, mpc_act


def load(ckpt):
    dev = torch.device("cuda")
    model = GDSPModel(24, 4).to(dev)
    V = ValueNet(24).to(dev)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    V.load_state_dict(state["v"])
    model.eval(); V.eval()
    return model, V


def evaluate(model, V, obs_noise=0.0, act_noise=0.0, init_tilt=0.0,
             n_eps=10, max_steps=300, device="cuda"):
    import gymnasium as gym
    env = gym.make('BipedalWalker-v3')
    rng = np.random.RandomState(0)
    lens, scores = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        if init_tilt > 0:
            obs = obs.copy()
            obs[0] += init_tilt  # 躯干角度偏移 (推离平衡)
        total, done = 0.0, False
        t = 0
        for t in range(max_steps):
            obs_in = obs + rng.randn(24) * obs_noise if obs_noise > 0 else obs
            a = mpc_act(model, V, obs_in, 24, 4, False, device=device)
            if act_noise > 0:
                a = np.clip(a + rng.randn(4) * act_noise, -1, 1)
            obs, r, done, _, _ = env.step(a)
            total += r
            if done:
                break
        lens.append(t + 1)
        scores.append(total)
    env.close()
    return np.mean(lens), np.mean(scores)


def main():
    print("加载模型...")
    sr = load("runs/walker_sr.pt")
    mcv = load("runs/walker_bootstrap.pt")

    tests = [
        ("基线 (无扰动)", dict()),
        ("动作噪声 0.1", dict(act_noise=0.1)),
        ("动作噪声 0.2", dict(act_noise=0.2)),
        ("观测噪声 0.05", dict(obs_noise=0.05)),
        ("观测噪声 0.1", dict(obs_noise=0.1)),
        ("初始倾斜 0.2", dict(init_tilt=0.2)),
        ("初始倾斜 0.4", dict(init_tilt=0.4)),
        ("综合 (动作0.1+观测0.05)", dict(act_noise=0.1, obs_noise=0.05)),
    ]
    print(f"\n{'扰动':<28}{'SR存活':>10}{'MC-V存活':>10}{'SR得分':>9}{'MC-V得分':>9}")
    base_sr, base_mcv = None, None
    for tag, kw in tests:
        l_sr, s_sr = evaluate(sr[0], sr[1], **kw)
        l_mcv, s_mcv = evaluate(mcv[0], mcv[1], **kw)
        if tag.startswith("基线"):
            base_sr, base_mcv = l_sr, l_mcv
        drop_sr = (base_sr - l_sr) / max(base_sr, 1) * 100
        drop_mcv = (base_mcv - l_mcv) / max(base_mcv, 1) * 100
        print(f"{tag:<28}{l_sr:>8.0f}步{'-'+str(int(drop_sr))+'%':>10}"
              f"{l_mcv:>8.0f}步{'-'+str(int(drop_mcv))+'%':>10}{s_sr:>9.1f}{s_mcv:>9.1f}")


if __name__ == "__main__":
    main()
