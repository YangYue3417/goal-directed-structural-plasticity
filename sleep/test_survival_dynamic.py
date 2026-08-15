"""test_survival_dynamic.py — 动态环境 30 昼夜生存回归 (最终验收)。

环境: SurvivalMaze — 地图/食物每天变化, 能量跨天延续, cost 1.0
系统: 多地图世界模型 (通用规律) + GRU 隐状态规划器 (气味+奖励目标)
夜晚: 做梦 = 重放当天轨迹 (小 lr 适应当天地图) + 突触降标 (×0.95)

通过标准: 30 天不死 (能量跨天, 每天新地图新食物)。
对照: 无做梦 (不适应当天地图) / 随机动作。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from envs.survival_maze import SurvivalMaze
from world_models.train_wm_explore import WorldExplore, ACT, OBS
from world_models.train_wm_energy import ValueNet


class GRUPlanner:
    """世界模型规划器: D 步想象树 + 价值 bootstrap。"""

    def __init__(self, wm, device, V=None, smell_coef=0.5, D=3,
                 novelty_coef=0.0):
        self.wm = wm
        self.device = device
        self.V = V
        self.smell_coef = smell_coef
        self.novelty_coef = novelty_coef
        self.D = D
        self.h = None  # (1, 1, 128)

    def reset(self):
        self.h = None

    def _roll(self, obs, h, a):
        wm = self.wm
        sa = torch.from_numpy(
            np.concatenate([obs, np.eye(ACT)[a]])[None, None]).float().to(self.device)
        emb = torch.tanh(wm.embed(sa))
        pool_out, _ = wm.unit(emb)
        h_out, _ = wm.gru(pool_out, h)
        obs_pred = wm.head_obs(h_out) + wm.skip_obs(sa)
        rew_pred = wm.head_rew(h_out).squeeze(-1) + wm.skip_rew(sa).squeeze(-1)
        return obs_pred, rew_pred, h_out

    @torch.no_grad()
    def act(self, obs):
        """动作树搜索: 3^D 序列批量展开 (可转弯绕墙), 气味+新异+奖励。"""
        wm = self.wm
        # 层级展开 (每层: 状态×3动作 → 单批 forward)
        states = [obs]
        hs = [self.h]
        scores = [0.0]
        g = 1.0
        for d in range(self.D):
            inp = np.stack([np.concatenate([s, np.eye(ACT)[a]])
                            for s in states for a in range(ACT)])
            sa = torch.from_numpy(inp)[None].float().to(self.device)  # (1, N, obs+act)
            emb = torch.tanh(wm.embed(sa))
            po, _ = wm.unit(emb)
            # 每序列自己的 GRU 状态: 需逐样本处理
            outs = []
            for i in range(sa.shape[1]):
                h_i = hs[i // ACT] if d == 0 else hs_all[i // ACT]
                ho, _ = wm.gru(po[:, i:i+1], h_i)
                outs.append((ho, wm.head_obs(ho) + wm.skip_obs(sa[:, i:i+1]),
                             wm.head_rew(ho).squeeze(-1) + wm.skip_rew(sa[:, i:i+1]).squeeze(-1)))
            new_states, new_hs, new_scores = [], [], []
            for i, (ho, op, rp) in enumerate(outs):
                opn = op[0, 0].cpu().numpy()
                parent = i // ACT
                novelty = float(np.abs(opn - states[parent]).sum())
                sc = scores[parent] + g * (rp.item() + self.smell_coef * opn[:4].sum()
                                           + self.novelty_coef * novelty)
                new_states.append(opn)
                new_hs.append(ho)
                new_scores.append(sc)
            states, hs_all, scores = new_states, new_hs, new_scores
            g *= 0.9
        # V bootstrap: 叶节点价值 = 食物可达性 (长时程)
        if self.V is not None:
            with torch.no_grad():
                leaf_t = torch.from_numpy(np.stack(states)).float().to(self.device)
                v_leaf = self.V(leaf_t).cpu().numpy()
            scores = [s + g * v for s, v in zip(scores, v_leaf)]
        best = int(np.argmax(scores))
        best_a = (best // (ACT ** (self.D - 1))) % ACT
        # 执行 best: 更新真实隐状态
        op, rp, h1 = self._roll(obs, self.h, best_a)
        self.h = h1
        return best_a


def dream_night(wm, traj, scale=0.95, lr=1e-4, device="cuda",
               mode="fragment", L=15, n_frag=8):
    """做梦: 片段回放 + 随机正/反向 (biologically: ripple 片段 + fwd/rev 混合)。

    mode:
      full_fwd: 完整轨迹正向 (旧, 动态下有害)
      fragment: 随机片段 × 随机方向 — 学习局部转换规则 (通用), 不绑定路径
    """
    if len(traj) < 10:
        return
    rng = np.random.RandomState(0)
    opt = torch.optim.AdamW(wm.parameters(), lr=lr)
    wm.train()
    segs = []
    if mode == "full_fwd":
        segs = [traj]
    else:
        for _ in range(n_frag):
            start = rng.randint(0, max(1, len(traj) - L + 1))
            seg = traj[start:start + L]
            if rng.rand() < 0.5:
                seg = seg[::-1]  # 反向片段
            segs.append(seg)
    for seg in segs:
        if len(seg) < 5:
            continue
        obs_t = torch.from_numpy(np.stack([t[0] for t in seg]))[None].float().to(device)
        act_t = torch.from_numpy(np.array([t[1] for t in seg]))[None].long().to(device)
        rew_t = torch.from_numpy(np.array([t[2] for t in seg]))[None].float().to(device)
        sn_t = torch.from_numpy(np.stack([t[3] for t in seg]))[None].float().to(device)
        op, rp, _, _ = wm(obs_t, act_t)
        loss = F.mse_loss(op, sn_t) + 0.2 * F.mse_loss(rp, rew_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    wm.eval()
    with torch.no_grad():
        for p in wm.parameters():
            p.mul_(scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--no_dream", action="store_true", help="对照: 无做梦")
    p.add_argument("--dream_mode", type=str, default="fragment",
                   choices=["fragment", "full_fwd"], help="做梦回放模式")
    p.add_argument("--scale", type=float, default=0.95, help="突触降标")
    p.add_argument("--random", action="store_true", help="对照: 随机动作")
    p.add_argument("--wm", type=str, default="runs/wm_explore.pt")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda")
    env = SurvivalMaze(size=10, n_foods=6, seed=args.seed, E0=200.0, day_steps=60, food_restore=80.0)
    wm = WorldExplore().to(device)
    wm.load_state_dict(torch.load(args.wm, map_location="cpu",
                                  weights_only=False)["model"])
    wm.eval()
    V = ValueNet(obs_dim=14).to(device)
    V.load_state_dict(torch.load("runs/v_survival.pt", map_location="cpu",
                                 weights_only=False)["model"])
    V.eval()
    planner = GRUPlanner(wm, device, V=V)

    tag = "随机对照" if args.random else ("无做梦" if args.no_dream else "做梦")
    print(f"=== 动态环境 30 昼夜生存 ({tag}) ===")
    died_day, foods_total, energy_day = None, 0, []
    for day in range(1, args.days + 1):
        obs, died = env.reset_day()
        planner.reset()
        traj = []
        for _ in range(env.day_steps):
            a = (np.random.randint(ACT) if args.random
                 else planner.act(obs))
            if np.random.rand() < args.noise and not args.random:
                a = np.random.randint(ACT)
            obs_next, r, done = env.step(a)
            traj.append((obs, a, r, obs_next, done))
            if r > 5:
                foods_total += 1
            obs = obs_next
            if done:
                died = True
                break
        energy_day.append(env.energy if not died else 0.0)
        if not args.random and not args.no_dream:
            dream_night(wm, traj, scale=args.scale, mode=args.dream_mode, device=device)
            planner.reset()
        if died:
            died_day = day
            break
        if day % 5 == 0 or day == 1:
            print(f"  Day {day}: 食物累计={foods_total} 能量={env.energy:.0f}", flush=True)

    if died_day:
        print(f"✗ 第 {died_day} 天死亡 (总食物 {foods_total})")
    else:
        print(f"✓ 存活 {args.days} 天 (总食物 {foods_total}, "
              f"平均日末能量 {np.mean(energy_day):.0f})")


if __name__ == "__main__":
    main()
