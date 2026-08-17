"""combo_verify.py — 组合框架验证: 世界模型 + TD价值 + 价值引导 ES。

任务: 长走廊觅食 (稀疏奖励 + 长视界):
  走廊 20 格, 食物在尽头 — 中间无奖励, 到达才 +10
  → 信用分配难 (稀疏), 需要多步规划 (长视界)

组合:
  世界模型: 预测 (s,a)→(s',r) — 想象评估基础
  TD 价值:  从经验学稠密信号 (走廊中段也有价值梯度)
  价值引导 ES: 扰动策略 → 评分 = 环境得分 + λ·Σ V(状态) (稠密化)

验证: 组合能否学会走完整条走廊 (对比: 只看稀疏得分会失败)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Corridor:
    """长走廊: 状态 x ∈ [0,20], 动作 0=左 1=右 2=停。食物在 x=20。"""
    def __init__(self, L=20, seed=0):
        self.L, self.rng = L, np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.x, self.t = 0.0, 0
        return np.array([self.x / self.L], np.float32)

    def step(self, a):
        if a == 1: self.x = min(self.L, self.x + 0.8)
        if a == 0: self.x = max(0, self.x - 0.8)
        self.t += 1
        got = self.x >= self.L - 0.1
        r = 10.0 if got else 0.0          # 稀疏: 只有到达有奖励
        done = got or self.t > 400
        if got: self.x = 0.0
        return np.array([self.x / self.L], np.float32), r, done


class Policy(nn.Module):
    def __init__(self, obs=1, n_act=3, d=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, d), nn.Tanh(), nn.Linear(d, n_act))

    def forward(self, s):
        return self.net(s)

    def act(self, s, greedy=True):
        dev = next(self.parameters()).device
        with torch.no_grad():
            logits = self.forward(torch.from_numpy(s).float().to(dev).unsqueeze(0))
            return int(logits.argmax(-1).item()) if greedy else \
                int(torch.distributions.Categorical(logits=logits).sample().item())


class WorldModel(nn.Module):
    def __init__(self, obs=1, n_act=3, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs + n_act, d), nn.ReLU(),
                                 nn.Linear(d, obs + 1))

    def forward(self, s, a):
        out = self.net(torch.cat([s, a], -1))
        return out[..., :1], out[..., 1]


class Value(nn.Module):
    def __init__(self, obs=1, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, s):
        return self.net(s).squeeze(-1)


def collect(env, policy, n_eps=200, seed=0):
    S, A, R, Sn = [], [], [], []
    rng = np.random.RandomState(seed)
    for _ in range(n_eps):
        s = env.reset(); done = False
        while not done:
            a = int(rng.randint(3))
            o2, r, d = env.step(a)
            S.append(s); A.append(a); R.append(r); Sn.append(o2)
            s = o2; done = d
    return (np.array(S, np.float32), np.array(A, np.int64),
            np.array(R, np.float32), np.array(Sn, np.float32))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("=== 组合验证: 长走廊觅食 (稀疏+长视界) ===")
    env = Corridor()
    S, A, R, Sn = collect(env, None, 300)
    print(f"随机: 到达率 {np.sum(R > 0) / 300 * 100:.0f}% (稀疏奖励: 只有到达 +10)")

    # 组合: 世界模型 + 价值 (TD) + ES (价值引导)
    wm = WorldModel().to(dev)
    V = Value().to(dev)
    opt = torch.optim.AdamW(list(wm.parameters()) + list(V.parameters()), lr=1e-3)
    s_t = torch.from_numpy(S).float().to(dev)
    a_t = F.one_hot(torch.from_numpy(A).long(), 3).float().to(dev)
    r_t = torch.from_numpy(R).float().to(dev)
    sn_t = torch.from_numpy(Sn).float().to(dev)
    for ep in range(150):
        idx = torch.randperm(len(S))[:1024]
        sp, rp = wm(s_t[idx], a_t[idx])
        loss_wm = F.mse_loss(sp, sn_t[idx]) + F.mse_loss(rp, r_t[idx])
        with torch.no_grad():
            target = r_t[idx] + 0.9 * V(sn_t[idx])
        loss = loss_wm + F.mse_loss(V(s_t[idx]), target)
        opt.zero_grad(); loss.backward(); opt.step()

    # 价值引导 ES: 评分 = 环境得分 + λ·Σ V (稠密化稀疏)
    lam = 1.0
    policy = Policy().to(dev)
    lr, sigma = 0.3, 0.2

    def eval_policy(lam=1.0, n_eps=10):
        env2 = Corridor()
        hits = 0
        for _ in range(n_eps):
            s = env2.reset(); done = False
            while not done:
                a = policy.act(s)
                o2, r, d = env2.step(a)
                if r > 0: hits += 1
                s = o2; done = d
        return hits / n_eps, env2.t

    for it in range(40):
        # 扰动 → 评估: 真实得分为主 (到达=+10×5) + 价值辅助 (λ 小)
        deltas, scores = [], []
        for i in range(16):
            delta = []
            for p in policy.parameters():
                d = torch.randn_like(p) * sigma
                delta.append(d)
                p.data.add_(d)
            # 评估: 跑 episode, 稠密评分 = Σ r + λ·Σ V(s)
            env2 = Corridor()
            s = env2.reset(); done = False
            total = 0.0
            with torch.no_grad():
                while not done:
                    a = policy.act(s)
                    o2, r, d = env2.step(a)
                    total += 5 * r + 0.2 * V(torch.from_numpy(s).float().to(dev)).item()
                    s = o2; done = d
            scores.append(total)
            for p, d in zip(policy.parameters(), delta):
                p.data.sub_(d)
            deltas.append(delta)
        scores = np.array(scores)
        w = np.clip((scores - scores.mean()) / (scores.std() + 1e-8), -2, 2)
        for delta, wi in zip(deltas, w):
            for p, d in zip(policy.parameters(), delta):
                p.data.add_(lr * wi / (16 * sigma) * d)
        if it % 10 == 9:
            hit, steps = eval_policy()
            print(f"  iter {it+1}: 稠密均分 {scores.mean():.1f} | 到达率 {hit * 100:.0f}%")

    hit, steps = eval_policy(n_eps=30)
    print(f"\n组合 (价值引导 ES): 到达率 {hit * 100:.0f}% — "
          f"{'✅ 学会长走廊 (稀疏奖励破解)' if hit > 0.5 else '⚠️ 未学会'}")


if __name__ == "__main__":
    main()
