# 🧬 代码架构

## genesis_lr 的整体架构

```{figure} ../../_static/images/code_structure.png
```

整个项目可以分为 3 个部分：仿真器（simulator）、环境（environments）和算法（algorithms）。

## 仿真器（legged_gym/simulator）

仿真器层为不同的仿真器提供统一的 API，包括 IsaacGym 和 Genesis。用户可以选择使用任意一个仿真器进行训练。

除了仿真器内置的传感器外，我们还实现了外部增强传感器，主要用于 IsaacGym。目前，我们提供基于 [warp](https://github.com/NVIDIA/warp) 的深度相机传感器，以加速 IsaacGym 中的渲染。与 IsaacGym 内置的深度相机相比，基于 warp 的深度相机在无头（headless）训练期间可以提供 2-3 倍的渲染速度提升。

## 环境（legged_gym/envs）

环境是智能体（agent）收集数据并从中获得奖励信号的地方。用户可以定义任务和动力学模型。环境采用继承风格。用户可以定义继承基类（legged_robot.py）的新环境类。环境中的配置文件（config files）负责存储环境的参数和设置。

## 算法（rsl_rl）

算法定义了训练中使用的强化学习算法。目前我们实现了 PPO（Proximal Policy Optimization，近端策略优化）、[SPO（Simple Policy Optimization，简单策略优化）](https://github.com/MyRepositories-hub/Simple-Policy-Optimization)以及基于它们的几种训练架构（显式估计器 Explicit Estimator、教师-学生 Teacher Student、并发 TS Concurrent TS、DreamWaQ）。

## 仿真到仿真（Sim2Sim）

仿真到仿真（Simulation-to-Simulation）验证是开发流程中的关键步骤，用于在真实世界部署前在不同物理仿真器中验证训练好的策略。这个过程有助于发现对特定仿真器伪影的过拟合，并确保策略的鲁棒性。

### 为什么 Sim2Sim 很重要

在单一仿真器中训练策略可能导致对物理引擎特性的利用。例如，仅在 IsaacGym 中训练的策略可能会利用特定的接触处理行为，而这些行为在 Genesis 或真实硬件中可能不同。通过跨仿真器验证，你可以：

- 发现对特定接触模型或摩擦近似的过拟合
- 识别对特定数值积分方案的依赖
- 确保在不同物理后端上的一致行为
- 增加 Sim2Real 迁移前的信心

### 训练循环架构

`OnPolicyRunner` 中的训练循环遵循清晰的四阶段模式：

#### 1. 初始化阶段

运行器按顺序初始化三个核心组件：

```
环境 → 演员-评论家网络 → 算法 → 回放缓冲区
```

首先，环境提供观察和动作空间规范。然后运行器使用配置的策略类（标准或循环）实例化演员-评论家网络。PPO 算法用优化逻辑包装这个网络。最后，回放缓冲区分配缓冲区用于收集经验。

#### 2. 回滚收集阶段

在每次迭代期间，运行器收集 `num_steps_per_env` 个转移：

```python
for step in range(num_steps_per_env):
    actions = alg.act(obs, critic_obs)      # 策略前向传播
    obs, rewards, dones, infos = env.step(actions)  # 环境步进
    alg.process_env_step(rewards, dones, infos)     # 存储转移
```

此阶段在 `torch.inference_mode()` 下运行以提高效率。使用滚动缓冲区跟踪回合统计数据，捕获已完成回合的奖励总和和回合长度。

#### 3. 学习阶段

收集回滚后，算法执行更新：

```python
alg.compute_returns(critic_obs)    # 使用 GAE 计算优势
mean_value_loss, mean_surrogate_loss = alg.update()  # PPO 更新
```

更新在收集数据的多个小批量上运行多个 epoch，计算裁剪的替代损失和价值函数损失。基于 KL 散度的学习率调度发生在此阶段。

#### 4. 日志记录和检查点阶段

指标被记录到 TensorBoard（以及可选的 WandB）：

- **性能指标**：FPS、收集时间、学习时间
- **损失指标**：价值损失、替代损失、熵
- **策略指标**：平均动作噪声标准差
- **回合指标**：平均奖励、回合长度

检查点以配置的间隔保存，包含模型权重、优化器状态和迭代计数。

### 检查点文件结构

检查点以 PyTorch `.pt` 文件形式保存在日志目录中：

```
logs/
└── experiment_name/
    ├── model_50.pt       # 第 50 次迭代的检查点
    ├── model_100.pt      # 第 100 次迭代的检查点
    └── exported/
        └── policy.pt     # 用于部署的 JIT 导出策略
```

每个检查点包含：

| 键 | 内容 |
|-----|----------|
| `model_state_dict` | 演员-评论家网络权重 |
| `optimizer_state_dict` | Adam 优化器状态 |
| `iter` | 当前学习迭代次数 |
| `infos` | 可选的元数据字典 |

加载检查点会同时恢复模型权重和优化器状态，允许无缝恢复训练：

```python
runner.load("model_500.pt", load_optimizer=True)
runner.learn(num_learning_iterations=1000)  # 从第 500 次迭代继续
```

### 组件交互概述

组件之间的数据流遵循循环模式：

```
┌─────────────────┐     观察值     ┌──────────────────┐
│   环境   │ ←──────────────────→ │  演员-评论家网络    │
│ (legged_gym)  │       动作        │    网络       │
└────────┬────────┘                      └────────┬─────────┘
         │                                        │
         │ 状态、奖励、完成标志                │ 策略
         │                                        │ 分布
         ↓                                        ↓
┌──────────────────────────────────────────────────────────┐
│                    算法 (PPO)                       │
│  ┌────────────────────────────────────────────────────┐  │
│  │               回放缓冲区                      │  │
│  │  存储：观察值、动作、奖励、价值、对数概率 │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  compute_returns() → update() → 优化步           │
└──────────────────────────────────────────────────────────┘
```

**环境接口**：提供 `step()`、`reset()` 和 `get_observations()` 方法。为所有并行环境返回批处理张量。

**演员-评论家**：实现 `act()`（带探索噪声的训练模式）和 `act_inference()`（确定性推理模式）。处理观察处理和价值估计。

**算法**：编排学习过程。调用 `init_storage()` 准备缓冲区，`process_env_step()` 存储转移，`compute_returns()` 进行优势估计，`update()` 进行梯度步进。

**缓冲区**：维护当前回滚的循环缓冲区。在被查询时计算回报和优势，处理更新的小批量采样。

### Sim2Sim 测试的推理流程

`play.py` 脚本处理 Sim2Sim 验证：

1. **配置覆盖**：减少环境数量以进行可视化，启用调试模式，调整地形以进行测试
2. **策略加载**：从检查点恢复模型并切换到评估模式
3. **导出阶段**：将训练好的策略转换为 JIT 格式以实现跨平台兼容性
4. **交互循环**：在环境中运行策略并进行实时可视化

不同的任务类型（教师-学生、显式估计器等）有专门的观察处理和导出逻辑：

```python
if task_type == "ts":
    actions = policy(obs_buf, obs_history)  # 使用历史缓冲区
elif task_type == "ee":
    actions = policy(estimator_features)    # 使用估计状态
else:
    actions = policy(obs_buf)               # 标准观察
```

### Sim2Sim 验证的最佳实践

1. **匹配观察空间**：确保所有仿真器提供相同的观察格式
2. **一致归一化**：在所有仿真器中使用相同的归一化统计信息
3. **验证动作范围**：确认关节限制和动作缩放匹配
4. **测试地形变化**：在不同地形类型上验证以确保泛化
5. **监控回合长度**：显著差异可能表明物理差异
6. **比较接触行为**：足部接触模式通常是仿真器特定的
