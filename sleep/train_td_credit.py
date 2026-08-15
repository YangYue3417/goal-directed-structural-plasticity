"""TD 信用分配实验 — 不用枚举终态, 模型能否自己学会饿死代价?

问题 (用户提出的核心):
  枚举终态 = 作弊 (直接标注因果链终点)。生物靠 TD 信号沿时间链
  回溯信用。本实验: 去掉枚举作弊, 用 TD(0) 让死亡惩罚自然传播。

设计:
  A. WM 无作弊版: 枚举不含近空能量 (奖励头对死亡"失明") — 半个信号
  B. V-TD: 随机轨迹 TD(0) 训练 (死亡 -10 沿链回溯)
  C. 评估:
     1) 模型级 rollout 权衡 (WM 无作弊, 无 V): 预期反了 (只看到移动耗能)
     2) planner (WM 无作弊 + TD-V): TD-V 携带死亡信号 → 是否救回权衡?
     3) 死亡信号传播: V(低能量远点) vs V(高能量远点) 是否梯度正确
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

from envs.energy_maze import EnergyMaze
from world_models.train_wm_energy import ValueNet, plan_step, eval_planner
from world_models.train_world_model import WorldModel, ACT_SIZE


def train_wm_no_death(env, device, epochs=50):
    """枚举不含近空能量 → 奖励头学不到死亡 (-10 不可见)。"""
    S, A, R, Sn = [], [], [], []
    for x in range(env.size):
        for y in range(env.size):
            if env.grid[y, x]:
                continue
            for d in range(4):
                for ef in [0.15, 0.5, 1.0]:  # 无 0.001 → 无死亡 transition
                    for a in range(ACT_SIZE):
                        env.x, env.y, env.dir = x, y, d
                        env.energy = ef * env.E0
                        env.food_eaten = [False] * env.n_foods
                        env._visited = {(x, y)}
                        env.steps = 0
                        s = env.observe()
                        sn, r, _ = env.step(a)
                        S.append(s); A.append(a); R.append(r); Sn.append(sn)
    S = np.array(S, np.float32); A = np.array(A, np.int64)
    R = np.array(R, np.float32); Sn = np.array(Sn, np.float32)
    food = R > 5
    if food.any():
        idx = np.repeat(np.nonzero(food)[0], 20)
        S = np.concatenate([S, S[idx]]); A = np.concatenate([A, A[idx]])
        R = np.concatenate([R, R[idx]]); Sn = np.concatenate([Sn, Sn[idx]])
    print(f"  枚举 (无死亡): {len(S)} transitions, 死亡 {(R < -5).sum()} 条", flush=True)
    n = len(S); rng = np.random.RandomState(0); perm = rng.permutation(n)
    split = int(n * 0.8)
    def to_t(a): return torch.from_numpy(a).float().to(device)
    S_t, Sn_t = to_t(S), to_t(Sn); A_oh = to_t(np.eye(ACT_SIZE)[A]); R_t = to_t(R)
    wm = WorldModel(obs_size=10, T=4).to(device)
    opt = torch.optim.AdamW(wm.parameters(), lr=1e-3)
    for ep in range(epochs):
        for i in range(0, split, 512):
            idx = perm[i:i + 512]
            sa = torch.cat([S_t[idx], A_oh[idx]], 1)
            op, rp, _ = wm(sa)
            loss = F.mse_loss(op, Sn_t[idx]) + 0.3 * F.mse_loss(rp, R_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save({"model": wm.state_dict()}, "runs/wm_energy_nodth.pt")
    return wm


def train_v_td(env, device, n_episodes=2000, gamma=0.95, passes=40):
    """TD(0): 死亡惩罚沿时间链逐步回溯。"""
    rng = np.random.RandomState(42)
    buffer = []  # (obs, reward, obs_next, done)
    for _ in range(n_episodes):
        obs = env.reset()
        for _ in range(200):
            a = int(rng.randint(ACT_SIZE))
            obs_next, r, done = env.step(a)
            buffer.append((obs, r, obs_next, done))
            obs = obs_next
            if done:
                break
    obs_t = torch.from_numpy(np.stack([b[0] for b in buffer])).float().to(device)
    rew_t = torch.from_numpy(np.array([b[1] for b in buffer])).float().to(device)
    nxt_t = torch.from_numpy(np.stack([b[2] for b in buffer])).float().to(device)
    done_t = torch.from_numpy(np.array([b[3] for b in buffer], dtype=np.float32)).to(device)
    V = ValueNet().to(device)
    opt = torch.optim.AdamW(V.parameters(), lr=1e-3)
    print(f"  TD 训练: {len(buffer)} transitions, {passes} passes", flush=True)
    for p in range(passes):
        idx = torch.randperm(len(buffer))[:16384].to(device)
        with torch.no_grad():
            target = rew_t[idx] + gamma * (1 - done_t[idx]) * V(nxt_t[idx])
        loss = F.mse_loss(V(obs_t[idx]), target)
        opt.zero_grad(); loss.backward(); opt.step()
        if p % 10 == 9:
            print(f"    pass {p+1}: td_loss={loss.item():.4f}", flush=True)
    torch.save({"model": V.state_dict()}, "runs/v_td.pt")
    return V


def rollout_tradeoff(wm, env, device):
    """模型级权衡 (无 V): 保守转圈 vs 觅食直走。"""
    def rollout(obs, policy, steps=40, gamma=0.95):
        s = obs.copy(); total = 0.0; g = 1.0; deaths = 0
        for _ in range(steps):
            a = policy(s)
            sa = torch.from_numpy(np.concatenate([s, np.eye(ACT_SIZE)[a]])[None]).float().to(device)
            with torch.no_grad():
                sp, rp, _ = wm(sa)
            total += g * rp.item(); g *= gamma
            s = sp.cpu().numpy()[0]
            if s[-1] <= 0.001:
                deaths += 1
        return total, deaths
    print("  模型级 rollout (WM 无作弊, 无 V):")
    for ef in [0.30]:
        env.x, env.y, env.dir = 18, 18, 1
        env.energy = ef * env.E0; env.food_eaten = [False] * 3
        env._visited = {(18, 18)}; env.steps = 0
        obs = env.observe()
        r_t, d_t = rollout(obs, lambda s: 1)
        r_f, d_f = rollout(obs, lambda s: 0)
        print(f"    能量 {ef}: 原地转={r_t:+.1f}(死{d_t}) 直走={r_f:+.1f}(死{d_f}) "
              f"→ {'权衡正确' if r_f > r_t else '半个信号(反了)'}")


def main():
    device = torch.device("cuda")
    env = EnergyMaze(**cfg.ENERGY_ENV)

    print("=== A. WM 无作弊 (枚举无死亡) ===", flush=True)
    wm = train_wm_no_death(env, device)
    wm.eval()

    print("=== B. V-TD (生物式信用分配) ===", flush=True)
    V = train_v_td(env, device)
    V.eval()

    print("=== C. 评估 ===", flush=True)
    rollout_tradeoff(wm, env, device)

    # V 梯度检查: 远点低/高能量
    print("  V 饿死风险梯度 (远点 18,18):")
    for ef in [0.05, 0.2, 0.5, 1.0]:
        env.x, env.y, env.dir = 18, 18, 1
        env.energy = ef * env.E0; env.food_eaten = [False] * 3
        env._visited = {(18, 18)}; env.steps = 0
        obs = env.observe()
        with torch.no_grad():
            v = V(torch.from_numpy(obs[None]).float().to(device)).item()
        print(f"    能量 {ef:.2f}: V={v:+.2f}")

    print("  planner (WM无作弊 + TD-V):")
    for tag, rnd in [("规划+TD-V", False), ("随机", True)]:
        f, d = eval_planner(wm, V, env, 50, 3, device, rnd)
        print(f"    [{tag}] 食物率={f} 饿死率={d}")


if __name__ == "__main__":
    main()
