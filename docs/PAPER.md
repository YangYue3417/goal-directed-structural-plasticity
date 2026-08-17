# Goal-Directed Structural Plasticity — 论文草稿

> 目标会议: NeurIPS / ICLR (主会)。当前状态: 草稿 + 部分实验待补 (标注 [待跑])。

---

## Title

**Goal-Directed Structural Plasticity: How Specialization Emerges from Experience and When It Becomes Functional**

## Abstract

Biological brains rewire as the environment demands: V1 neurons form receptive fields, hippocampal cells encode places, and circuits specialize for survival-relevant structure. Neural networks rarely self-organize this way — mixture-of-experts collapses into generalists, and growing networks add dead capacity. We study when specialization emerges and becomes functional. Across symbolic, pixel, and Atari domains, we show: (1) prediction objectives force structured representations (94.9% vs 4.2% position decoding); (2) error-driven growth produces functionally critical neurons (1.47× over random, causal deletion); (3) connection-mask learning self-organizes receptive fields without pre-set geometry; (4) cognitive maps emerge from path integration (96.7% from ambiguous sensing); (5) consolidation has environment-dependent boundary conditions. All findings survive causal-deletion tests and continuous dynamic-world evaluation (no fixed survival horizon).

## 1. Introduction

[从 README Motivation 扩展: 分工的普遍性 (生物) vs 缺失 (NN), 核心问题, 三条路线 → 我们的框架]

## 2. Related Work

### 2.1 专精的粒度
- 块级 MoE: Switch Transformer (Fedus et al.), DeepSeek fine-grained experts — 细粒度提升但仍是块级
- 神经元级: superposition (Elhage/Anthropic), sparse features — 分布式编码, 非显式专精
- **我们的位置**: 同任务直接对比 神经元级 vs 块级 (0.855 vs 0.202), 粒度本身是变量

### 2.2 世界模型与表征
- Dreamer 系列 (Hafner et al.): 潜在世界模型 + 想象 — 我们采用其 latent 化 (不预测像素)
- JEPA (LeCun et al.): 预测潜在表征
- **我们的位置**: 世界模型作为"表征强迫结构"的工具 (预测目标 vs 策略目标对比)

### 2.3 动态网络与生长
- DEN (Yoon et al. 2018), Lifelong learning 动态扩展
- 神经发生计算模型 (neurogenesis): 募集/修剪
- **我们的位置**: 误差驱动生长 + 因果删除验证 + 优胜劣汰 (生长配淘汰), 翻案"生长=容量" (在预测任务)

### 2.4 空间表征
- 网格细胞/位置细胞 (Banino et al. 2018 Nature): 导航 RNN 涌现网格细胞 — 我们的路径积分 96.7% 同源但分离了"指纹 vs 积分"
- SLAM 类方法: 显式地图 — 我们是隐式认知地图

### 2.5 睡眠与巩固
- SRC (Bazhenov et al. 2022 Nat Commun): 无监督睡眠重放防遗忘
- van de Ven et al. 2020: 生成式重放
- **我们的位置**: 边界条件 — 重放内容须匹配环境稳定性; 降标依赖容量压力 (动态环境新发现)

### 2.6 连接学习
- 稀疏化 (pruning at init, SNIP), 可学习掩码
- **我们的位置**: 连接掩码 + 熵正则 → 感受野自组织 (无预设几何)

## 3. Method

[从 README Method 扩展, 结构: 3.1 环境 (生存压力/动态) 3.2 世界模型 3.3 生长与淘汰 3.4 认知地图 3.5 睡眠巩固 3.6 验证协议 (因果删除)]

## 4. Experiments

[从 Results P1-P6 扩展, 每节: 设置 → 结果 → 解读]

### 4.1 粒度 (P1) [补充实验/已有]
### 4.2 预测强迫结构 (P2)
### 4.3 生长功能 (P3) — 核心
### 4.4 认知地图 (P4)
### 4.5 生存与睡眠 (P5)
### 4.6 跨域迁移 (P6)

## 5. Ablations [待跑]

| 消融 | 设计 | 预期 |
|:---|:---|:---|
| 生长信号对比 | 价值V vs 误差 vs 环境统计 → 删神经元生存影响 | 价值>误差 (生存导向) |
| 淘汰必要性 | 生长 vs 生长+淘汰 (同预算) | 淘汰显著提升存活率 |
| 连接掩码 | 固定掩码 vs 可学习掩码 | 可学习更优 (感受野适配) |
| 睡眠内容 | 当天重放 vs 片段 vs 无 | 动态环境: 片段>完整>无 |
| 目标数量 | 1目标 vs 多目标 (子目标分化) | 多目标 → 1:1 专精映射 |

## 6. Discussion

### 6.1 为什么预测强迫结构
预测目标要求"世界如何运作"的知识 (多步一致), 策略目标只要求"下一步做什么" — 表征需求层级不同。这与 JEPA/世界模型文献一致, 但我们给出同任务内的干净对照。

### 6.2 生长的功能价值: 任务依赖
分类 (记忆) 任务: 随机生长就够 (容量红利); 预测 (结构) 任务: 误差驱动 > 随机 (定向补结构)。→ "生长=容量" 的翻案有条件: 任务需要结构时, 定向生长才有功能优势。

### 6.3 认知的双系统
地标识别 (指纹) vs 自运动积分 (路径积分) 自然分离 — 对应海马/新皮层分工 (CLS 理论)。连接掩码学习使"感受野"从环境统计中自组织 — V1 发育的计算版本。

### 6.4 睡眠巩固的边界
静态环境: 降标恢复可塑性 (M8); 动态环境: 降标侵蚀通用知识, 重放须匹配稳定性。→ 巩固机制不是万能, 内容须匹配环境时间尺度。

### 6.5 与 LLM 的联系
数据=环境, 目标=生存 (广义 RL)。MoE 专家管理是本文框架的工程实例 — 容量分配应按任务价值, 而非纯路由相关性。

## 7. Limitations

[README 的 6 条, 扩展: 玩具规模/单seed/视觉噪声/Atari非基准/任务依赖/LLM比喻]

## 8. Conclusion

[总结 5 个发现 + 框架 + 开放问题]

---

## 待办 (论文级证据)

- [ ] **多 seed 统计**: 符号 1.47×, 视觉 +42%, 30 天生存 × 10 seed → 均值±std + 显著性
- [ ] **消融矩阵** (见 §5): 各机制独立贡献
- [ ] 中等规模验证 (ProcGen / 更大迷宫) — 摆脱玩具标签
- [ ] 与 Dreamer-lite 对比 (世界模型部分对齐文献)
- [ ] 与固定容量/经典持续学习 (EWC/replay) 对比
- [ ] 图: 框架图 / 结果图 (README 已有 assets)
