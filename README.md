<div align="center">

# 🧠 Goal-Directed Structural Plasticity

### How specialization emerges from experience — and when it becomes functional

---

**Abstract.** Biological brains rewire as the environment demands: V1 neurons form receptive fields, hippocampal cells encode places, and circuits specialize for survival-relevant structure. Neural networks rarely self-organize this way — MoE experts collapse into generalists, and growing networks add dead capacity. We study *when* specialization emerges and becomes functional. Across symbolic, pixel, and Atari domains, we show that **(1)** prediction objectives force structured representations, **(2)** error-driven growth produces functionally critical neurons, **(3)** connection-mask learning self-organizes receptive fields, **(4)** cognitive maps emerge from path integration, and **(5)** consolidation has environment-dependent boundary conditions. All findings are validated by causal deletion and survive in a daily-changing world indefinitely (continuous survival, no fixed horizon).

</div>

## 🧭 Motivation

Division of labor is how brains organize — yet NNs rarely self-organize it. Mixture-of-Experts models *assume* specialization, but in practice experts collapse into generalists; growing networks add capacity, but the new capacity is rarely *functional*. What conditions make specialization emerge — and can the environment induce it?

```
division of labor is how brains organize
  → yet NNs rarely self-organize it (experts collapse into generalists)
  → more capacity ≠ more function (growth is often dead capacity)
  → what conditions make specialization emerge & functional?
  → can the environment induce it? how does it persist?
```

## 🔬 Method

```mermaid
flowchart TD
    G["Survival Goals<br/>foraging · navigation · prediction · vision"]
    V["Value Signals<br/>learned V · usage frequency · prediction error"]
    S["Structural Growth<br/>split · recruit · connect-adapt · prune"]
    N["Specialized Neurons<br/>E1, E2, ... — each serves a sub-goal"]
    C["Causal Validation<br/>delete Ei - goal i collapses · transfer test"]

    G -->|defines| V
    V -->|triggers| S
    S -->|produces| N
    N -->|verified by| C
    C -.->|new goals emerge| G

    style G fill:#2E5AAC,color:#fff
    style V fill:#2E7D46,color:#fff
    style S fill:#B3392F,color:#fff
    style N fill:#6C4AA6,color:#fff
    style C fill:#A87F2E,color:#fff
```

The framework runs a world model (encoder → connection-mask pool → GRU integration → latent prediction) under goal-driven structural growth, with pruning for resource constraints and sleep-based consolidation. Each claim below is validated by causal deletion: remove the specialized neurons and the corresponding goal collapses.

## 📊 Results

The research chain — each step asks *why*:

| # | Question | Why we ran it | Finding |
|:---:|:---|:---|:---|
| **P1** | At what level does specialization live? | Block-level MoE failed — architecture or capacity? | Neuron-level 0.855 ≫ block-level 0.202 — **granularity decides specialization** |
| **P2** | What training condition forces structure? | Neuron-level specialization was memory — what makes it functional? | Prediction 94.9% vs policy 4.2% — **task demand shapes representation** |
| **P3** | Can the environment induce growth? | Is growth "adding capacity" or "purposeful"? | Error-driven 1.47× more critical (causal deletion); connection adaptation → receptive fields |
| **P4** | How is space built without a map? | How do animals navigate without GPS? | Path integration 96.7% — **cognitive maps emerge from motion** |
| **P5** | How does knowledge persist? | Dynamic world + finite brain | Sleep boundaries: fragment replay works, downscaling needs capacity pressure; **continuous survival** |
| **P6** | Is the mechanism general? | Toy-specific or universal principle? | Symbolic → vision → Atari, zero architecture change — **observation-agnostic** |

![Results](assets/results.png)

![Transfer](assets/transfer.png)

### 🎬 Demos (trained agents, no external RL)

| Domain | Agent | World model internals |
|:---:|:---|:---|
| **Bipedal Walker (stepping in place)** | ![walker](assets/walker_tread.gif) | Hull safety zone (≥5.6) is the **only hard constraint**; visible stepping: hip swing 0.9 + knee flexion 1.0 (legs alternate 84%, knee bends freely) |
| **Bipedal Walker (early)** | ![walker](assets/walker_demo.gif) | Δ-residual world model + SR value + MPC — balances without falling (safety-optimal policy is stationary; locomotion is a documented boundary) |
| **CartPole** | ![cartpole](assets/cartpole_demo.gif) | World-model prediction selects the action that keeps the pole upright |
| **Ms. Pac-Man** | ![pacman](assets/atari_pacman_wm.gif) | Top: real gameplay · Bottom: spatial structure of world-model neurons (learned 5×5 field connectivity) |

## 🧩 Architecture: one entry, components guaranteed

A single entry point (`train_gdsp.py`) makes **growth, pruning, and consolidation required runtime components** — migrating to a new domain means swapping the environment, never re-assembling the framework:

```
train_gdsp(env_fn, obs_dim, act_dim)      # swap env to transfer
    → world model (connection-mask / Δ-residual / LIF memory pool)
    → growth (error-driven, hard-sample)  # automatic, required
    → pruning (usage-based)               # automatic, required
    → consolidation (fragment replay)     # automatic, required
    → decision (learned V + MPC, 1-3 step) # pure framework, no external RL
```

**Mechanism family** (each verified by causal deletion):

| Mechanism | Finding |
|:---|:---|
| **Growth → trial → connect** (baby neurons) | Direct cloning breaks the pool (-89%); weak-init + probe + settle makes grown neurons functional (+378%) |
| **Dream-phase growth** (training shapes reserve directions via shadow prediction, dreaming activates them) | Grown neurons functional on activation (+140%); reserve lifecycle (apoptosis/reborn) improves 2-3× |
| **Memory pool** (LIF membrane potential, per-neuron τ) | Neurons acquire state-transition ability: cycle task 48× better at phase prediction |
| **Multi-step MPC** (D=1→3) | Decision horizon, not data or sensing, was Walker's bottleneck: 40% → 70% complete episodes |

| Domain | Obs → Action | Result |
|:---|:---|:---|
| CartPole | 4-dim → discrete 2 | world model 0.0046, MPC beats random |
| Walker | 24-dim joints → continuous 4 | **balances 771-1600 steps (no falls, no rewards)**; displacement ≈ 0 — safety-optimal policy is stationary, coordinated locomotion beyond random-sampling MPC |
| Cycle task | 1-dim state machine | memory pool learns phase: 0.016 vs 0.77 error (48×) |

### Walker 训练脚本系列 (持续生存自举)

```
walker_bootstrap.py   # 自举循环 (随机收集 → 学习 → 再收集)
walker_sr.py          # SR 安全访问价值 (无奖励, 抗扰动 <±8%)
walker_full.py        # 整套: 白天学习+生长, 夜晚做梦+淘汰 (ΔWM)
walker_parallel.py    # 8 envs 并行, 数据速率 ×8
walker_mem.py         # 记忆池 (LIF) 世界模型
```

### 神经元机制实验脚本 (符号, 分钟级)

```
mempool.py            # 记忆池 (LIF + τ 分工) + baby 生长 + shadow 储备 + 生命周期
cycle_task.py         # 周期状态机: 记忆 → 状态转换能力
discrim_experiment.py # 结构难 vs 噪声难 → 生长功能差异
dream_grow_task.py    # 训练塑造方向 + 做梦生长 + 储备凋亡/再出生
```

## ⚠️ Limitations

Honest boundaries of this work:

1. **Toy-scale validation** — 10×10 mazes and small MNIST; no real-scale experiments yet
2. **Single-seed results** for the headline numbers (1.47×, +42%) — multi-seed statistics pending
3. **Visual growth deletion is noisy** — decode-probe refitting sensitivity; activation rate is the reliable metric
4. **Atari is mechanism-level** — 50K random frames, not a performance benchmark
5. **"Growth = capacity" reversal is task-dependent** — holds for prediction tasks; classification (memory) tasks still work with random growth
6. **Survival is a metaphor at the LLM scale** — "data = environment, objective = survival" is a mapping, not a literal pressure signal

## 🔭 Roadmap

- [ ] Value-driven (learned V) vs error-driven growth — which survives better
- [ ] Multi-goal environments → 1:1 neuron-goal mapping purity
- [ ] New goal appears → new specialization follows
- [ ] LLM transfer: streaming domains → value-driven expert allocation
- [ ] Multi-seed statistical rigor

## 🧩 Future: world-model learning & explaining functional clustering

**Application — learning regularities inside a world model.** World models are a frontier direction (Dreamer, Genie, JEPA) for robots, autonomous driving, and increasingly LLMs. Our framework adds what they lack: **structure that adapts to goals**. Capacity is not fixed — it grows where value signals point, and connection masks self-organize into functional units. In an LLM context, where *data = environment* and *training objective = survival*, this maps to value-driven expert allocation and goal-aligned capacity.

**Science — explaining why neurons cluster functionally.** Neuroscience observes that neurons cluster by function: V1 cells tuned to edges at specific locations, hippocampal place cells, even induction heads in LLMs. This project reproduces the *mechanism* behind such clustering:

```
prediction task   → position-tuned neurons  (hidden-state decode 94.9%)
connection masks  → spatial receptive fields (entropy 0.05)
error-driven growth → functionally critical neurons (deletion collapses the goal)
environment stats → transition-type specialists (wall / food / open detectors)
```

*Functional clustering is not an assumption — it is a consequence of what the network must predict.*

## 🔁 Reproduction

```bash
git clone git@github.com:YangYue3417/goal-directed-structural-plasticity.git
cd goal-directed-structural-plasticity

# P4: cognitive map from ambiguous sensing (~15 min)
python world_models/train_wm_explore.py --sensor walls

# P3: visual receptive fields self-organize (~25 min)
python world_models/train_wm_image_v3.py --epochs 25

# P5: continuous survival in a daily-changing world (~10 min)
python sleep/test_survival_dynamic.py --days 100   # no fixed horizon — always survive

# Unified entry: growth/pruning automatic — swap env to transfer
python train_gdsp.py --env cartpole   # 4-dim state, discrete action
python train_gdsp.py --env walker     # 24-dim joints, continuous action

# Walker: bootstrap survival (continuous, no rewards)
python walker_full.py --rounds 8 --steps_per_round 10000   # day: learn+grow, night: dream+prune
python walker_parallel.py --rounds 20 --steps_per_round 50000 --n_envs 8  # parallel

# Neuron mechanisms (symbolic, minutes)
python cycle_task.py          # memory pool → state-transition ability
python discrim_experiment.py  # structural vs noisy difficulty → growth function
python dream_grow_task.py     # train shapes direction + dream-phase growth
```

*Atari: `pip install gymnasium[atari] ale-py opencv-python-headless`, then `python world_models/train_atari_wm.py`*

All experiment configurations are centralized in [`config.py`](config.py). Full evidence tables in [TECHNICAL.md](docs/TECHNICAL.md).

## 📚 Documentation

| Doc | Content |
|:---|:---|
| **[TECHNICAL.md](docs/TECHNICAL.md)** | Evidence tables, methods, configs, full reproduction |
| **docs/MASTER_SUMMARY.md** | Research narrative & finding chain (中文) |
| **docs/FINDINGS_growth_functional.md** | Center experiments: growth / survival / transfer (中文) |
| **docs/FINDINGS_world_model.md** | World models, credit assignment, dreaming (中文) |
| **docs/FINDINGS_control_domain.md** | Control domain (CartPole/Walker): mechanism family + honest boundaries (中文) |

## 📖 Citation

```bibtex
@software{goal_directed_structural_plasticity,
  title = {Goal-Directed Structural Plasticity},
  author = {Yang, Yue},
  year = {2026},
  url = {https://github.com/YangYue3417/goal-directed-structural-plasticity},
}
```

## 📄 License

MIT

---

*Importance is not predefined — it emerges from what keeps you alive.*
