# Loihi 2 推理实现映射

## Intel Loihi 2 核心特性 (参考: arXiv 2310.03251, Intel Loihi 2 brief)

| Loihi 2 特性 | 本框架对应 | 状态 |
|:---|:---|:---|
| 异步事件驱动 (asynchronous, event-driven) | top-k 稀疏激活 (每次只发 64/512 神经元) | ✅ mempool.py |
| 有状态神经元模型 (stateful neuron) | LIF 膜电位 V_m(t) = (1-1/τ)·V_m(t-1) + z | ✅ mempool.py |
| 每核 8192 神经元, 128 核 | 池 512-1024 神经元, 可生长到上限 | ✅ |
| 稀疏计算 (sparse) | masked top-k, 稀疏矩阵 | ✅ |
| 在线学习 (online/on-chip learning) | 增量训练, EMA 统计, 生长/淘汰在线 | ✅ |
| 结构可塑性 (structural plasticity) | 生长→试探→连接 / 储备生命周期 | ✅ |
| 突触延迟/时间常数 (per-neuron τ) | τ 多样化 (快/慢神经元分工) | ✅ |

## 推理实现要点 (对照实现)

```python
# Loihi 2 风格: 稀疏事件前向 (mempool.py MemPool)
for t in range(T):
    vm = vm * leak + z @ W_in.T        # 有状态神经元 (膜电位积分)
    pre = vm.masked_fill(~active, -1e9) # 稀疏事件
    vals, idx = pre.topk(top_k)         # top-k 发放 (事件驱动)
    sparse = scatter(idx, gelu(vals))   # 稀疏激活
    out = sparse @ W_out.T
```

- **事件驱动**: 只有 top-k 神经元"发放" (Loihi 同步: 稀疏脉冲)
- **有状态**: 膜电位跨时间步保持 (Loihi 异步状态)
- **在线**: 生长 (难样本定向)/淘汰 (激活率) 无需重训

## 能量约束 (用户 4 约束, energy_balance.py)

1. **能量效率最大化**: 目标 = 每单位能量获取最多生存
2. **运动消耗能量, 能量点在安全区外**: 必须离开安全区觅食
   - 能量 = 物理功: dE = m·|a|·|v|·dt (质量×加速度×速度), 非固定移动消耗
3. **无目标有惩罚**: 无任务终点; 惩罚 = 能量耗尽死亡 (r=-1)
4. **奖励**: 能量补充 +0.5·gained; 距离势塑形 0.08·Δd (引导觅食)

## 诚实边界 (energy_balance.py 4 次尝试)

**框架 (世界模型+TD+MPC) 未学会能量觅食**:
- 随机 631 步 vs MPC 105-293 步, 能量获取 0/ep (从不觅食)
- 根因:
  ① 站桩陷阱: 基础代谢 (0.06/步) 慢死 vs 移动消耗 (m·a·v) 快 → V 倾向站桩
  ② 稀疏奖励: 能量点事件 (+4) 稀疏 → TD 传播慢, 世界模型 r 头学不准
  ③ MPC 视野: 单步看"移动=消耗" (负); 多步 D=5 误差累积预测漂移
- 觅食 = 周期决策 (外出→获取→返回), 超出单步 TD + 采样 MPC

**改进方向** (若继续):
- 奖励重标定 / 势函数强化 (能量效率作为显式目标)
- 显式行为序列 (CPG 式周期策略, 而非逐动作采样)
- 策略学习 (偏离无奖励框架)
