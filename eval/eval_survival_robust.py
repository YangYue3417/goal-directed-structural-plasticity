"""稳定性 + 迁移验证: 多 seed × 多环境配置 的持续生存。"""
import sys
sys.path.insert(0, '.')
import numpy as np, torch
from envs.survival_maze import SurvivalMaze
from world_models.train_wm_explore import WorldExplore
from world_models.train_wm_energy import ValueNet
from sleep.test_survival_dynamic import GRUPlanner, dream_night

def run_once(seed, cfg, wm, V, device, days=30, mode="nodream", scale=1.0):
    env = SurvivalMaze(**cfg, seed=seed)
    planner = GRUPlanner(wm, device, V=V, D=3)
    foods_total = 0
    for day in range(days):
        obs, _ = env.reset_day()
        planner.reset()
        traj = []
        for _ in range(env.day_steps):
            a = planner.act(obs)
            obs_next, r, done = env.step(a)
            traj.append((obs, a, r, obs_next, done))
            if r > 5: foods_total += 1
            obs = obs_next
            if done: return day + 1, foods_total
        if mode == "dream":
            dream_night(wm, traj, scale=scale, mode="fragment", device=device)
            planner.reset()
    return days + 1, foods_total

def main():
    device = torch.device("cuda")
    wm = WorldExplore().to(device)
    wm.load_state_dict(torch.load("runs/wm_explore.pt", map_location="cpu", weights_only=False)["model"])
    V = ValueNet(obs_dim=14).to(device)
    V.load_state_dict(torch.load("runs/v_survival.pt", map_location="cpu", weights_only=False)["model"])
    wm.eval(); V.eval()

    base = dict(cfg.SURVIVAL_ENV)
    print("=== 稳定性: 训练分布内多 seed ===")
    for seed in [43, 44, 45, 7]:
        d, f = run_once(seed, dict(size=10, **base), wm, V, device)
        print(f"  seed={seed}: 存活至第{d}天 (食物{f})")

    print("=== 迁移: 环境参数变化 ===")
    migs = [
        ("尺寸 8×8", dict(size=8, **base)),
        ("尺寸 12×12", dict(size=12, **base)),
        ("食物 4 个", dict(size=10, n_foods=4, **{k:v for k,v in base.items() if k!='n_foods'})),
        ("食物 8 个", dict(size=10, n_foods=8, **{k:v for k,v in base.items() if k!='n_foods'})),
        ("墙密度 0.03", dict(size=10, wall_density=0.03, **{k:v for k,v in base.items() if k!='wall_density'})),
        ("墙密度 0.08", dict(size=10, wall_density=0.08, **{k:v for k,v in base.items() if k!='wall_density'})),
    ]
    for tag, cfg in migs:
        d, f = run_once(42, cfg, wm, V, device)
        print(f"  {tag}: 存活至第{d}天 (食物{f})")

if __name__ == "__main__":
    main()
