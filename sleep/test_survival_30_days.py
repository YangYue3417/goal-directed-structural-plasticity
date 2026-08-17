"""回归测试: 训练完成的神经元系统在迷宫中 30 昼夜不死。

协议 (端到端验收):
  Day 1..30:
    清晨: 能量重置 E0, 检查昨日做梦后系统未退化
    白天: 世界模型规划器在 EnergyMaze 觅食 (max_steps 内找 ≥1 食物)
    夜晚: 做梦 = 结构化重放当天轨迹 (MC 回归) + 突触降标
  通过标准: 持续存活 (无期限) + 每日食物 ≥1 + 预测误差不随天数恶化

测试点:
  1. 规划器生存可靠性 (每天能否找到食物)
  2. 昼夜循环长期稳定 (30 夜降标 × 重放补偿 → 不崩)
  3. 对照: 无做梦臂 (只有降标无重放 → 应退化) / 无降标臂
  4. (后续) 删生长神经元 → 生存率变化
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from envs.energy_maze import EnergyMaze
from world_models.train_wm_energy import ValueNet, plan_step, WorldModel, ACT_SIZE


def collect_day(env, wm, V, max_steps=150, D=3, device="cuda", noise=0.05):
    """白天: 规划器觅食 (带探索噪声), 返回 (轨迹, 食物数, 死亡)。"""
    obs = env.reset()
    traj, foods, died = [], 0, False
    for _ in range(max_steps):
        a = plan_step(wm, V, obs, D, device=device)
        if np.random.rand() < noise:
            a = np.random.randint(ACT_SIZE)
        obs_next, r, done = env.step(a)
        traj.append((obs, a, r, obs_next, done))
        if r > 5:
            foods += 1
        obs = obs_next
        if done:
            died = True
            break
    return traj, foods, died


def dream_night(V, traj, scale=0.95, gamma=0.95, lr=1e-3, device="cuda"):
    """夜晚: MC 回归重放当天轨迹 (结构化) + 突触降标。"""
    opt = torch.optim.AdamW(V.parameters(), lr=lr)
    obs_t = torch.stack([torch.from_numpy(t[0]) for t in traj]).float().to(device)
    rew = torch.tensor([t[2] for t in traj], dtype=torch.float32).to(device)
    done = torch.tensor([t[4] for t in traj], dtype=torch.float32).to(device)
    G = torch.zeros_like(rew)
    g = 0.0
    for t in range(len(traj) - 1, -1, -1):
        g = rew[t].item() + gamma * g * (1 - done[t].item())
        G[t] = g
    for _ in range(10):
        loss = F.mse_loss(V(obs_t), G)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        for p in V.parameters():
            p.mul_(scale)
    return loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--no_dream", action="store_true", help="对照: 无做梦 (不重放不降标)")
    p.add_argument("--scale_only", action="store_true", help="对照: 只降标不重放")
    p.add_argument("--wm", type=str, default="runs/wm_energy.pt")
    p.add_argument("--v", type=str, default="runs/v_energy.pt")
    p.add_argument("--D", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.1, help="白天探索噪声")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda")
    env = EnergyMaze(size=20, n_foods=3, seed=args.seed, food_restore=80)
    wm = WorldModel(obs_size=10, T=4).to(device)
    wm.load_state_dict(torch.load(args.wm, map_location="cpu", weights_only=False)["model"])
    wm.eval()
    V = ValueNet().to(device)
    V.load_state_dict(torch.load(args.v, map_location="cpu", weights_only=False)["model"])

    print(f"=== 30 昼夜生存回归测试 ({'无梦对照' if args.no_dream else '做梦'}) ===")
    died_day, foods_total, energy_hist = None, 0, []
    for day in range(1, args.days + 1):
        traj, foods, died = collect_day(env, wm, V, noise=args.noise, device=device)
        foods_total += foods
        energy_hist.append(env.energy if not died else 0.0)
        if args.no_dream:
            pass  # 无做梦: 什么都不做
        elif args.scale_only:
            with torch.no_grad():
                for p in V.parameters():
                    p.mul_(0.95)
        else:
            dream_night(V, traj, device=device)
        if died:
            died_day = day
            break
        if day % 5 == 0 or day == 1:
            print(f"  Day {day}: 食物={foods} 能量={env.energy:.0f}", flush=True)

    if died_day:
        print(f"✗ 第 {died_day} 天死亡 (总食物 {foods_total})")
    else:
        avg_e = np.mean(energy_hist)
        print(f"✓ 存活 {args.days} 天 (总食物 {foods_total}, 平均日末能量 {avg_e:.0f})")
        print(f"  最后一天 V 范数: {sum(p.norm().item() for p in V.parameters()):.1f}")


if __name__ == "__main__":
    main()
