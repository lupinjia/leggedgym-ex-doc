# 常见问题解答 (FAQ)

本FAQ涵盖了关于LeggedGym-Ex的常见问题。如果在这里找不到答案，请查阅详细的文档章节或加入我们的社区讨论。

---

## 概述

### 什么是LeggedGym-Ex？

LeggedGym-Ex是一个用于训练腿式机器人运动策略的强化学习框架。它扩展了原始的[legged_gym](https://github.com/leggedrobotics/legged_gym)框架，增加了对多个仿真器（Genesis、IsaacGym、IsaacSim）的支持，并实现了10多种近期机器人研究论文中的方法。

### 支持哪些机器人？

LeggedGym-Ex目前支持宇树Go2（四足）、宇树G1（双足）、LimX TRON1（四足）和博恩思K1（四足）。您也可以按照{doc}`/developer_guide/how_to/add_new_robot`中的指南添加您自己的机器人。

### LeggedGym-Ex与原始legged_gym有什么区别？

LeggedGym-Ex在保持与legged_gym API兼容的同时，增加了多仿真器支持，并实现了教师-学生网络、显式估计器、DreamWaQ、DeepMimic和AMP等高级方法。完整的功能对比请参阅[README](https://github.com/lupinjia/LeggedGym-Ex)。

### 我可以将此框架用于研究论文发表吗？

可以。LeggedGym-Ex是开源的，专为研究用途设计。发表论文时，请同时引用原始的legged_gym论文并说明您使用的具体方法（DeepMimic、AMP等）。

### 允许商业使用吗？

允许。本框架采用BSD-3-Clause许可证发布。您可以将其用于商业应用，包括在商用机器人上部署策略。

### 在哪里可以获得帮助？

您可以通过以下渠道获得帮助：GitHub issues页面用于错误报告，飞书群组（QR码在README中）用于社区讨论，文档提供详细的指南。

### 如何报告错误或请求功能？

在[GitHub仓库](https://github.com/lupinjia/LeggedGym-Ex)上提交issue。报告错误时，请包含您的仿真器版本、Python版本和最小复现脚本。

---

## 安装

### 我应该使用哪个仿真器？

根据您的需求选择：

- **IsaacGym**：训练速度最快，适合快速迭代，需要Python 3.8
- **Genesis**：速度与渲染质量的良好平衡，支持软体材料，需要Python 3.10+
- **IsaacLab/IsaacSim**：最佳渲染质量，NVIDIA官方支持，需要Python 3.11+

每个仿真器的详细设置说明请参阅{doc}`getting_started/installation`。

### 我需要什么硬件？

最低要求：

- **CPU**：Intel Core i7或AMD Ryzen 7（推荐i9）
- **GPU**：NVIDIA RTX 3060 8GB+（推荐RTX 3080 10GB+）
- **内存**：推荐32GB
- **操作系统**：Ubuntu 20.04或22.04
- **NVIDIA驱动**：570或更新版本

可以通过减少`num_envs`和地形复杂度在较小的GPU上训练。

### 我需要为不同的仿真器创建单独的conda环境吗？

需要。IsaacGym需要Python 3.8，而Genesis和IsaacLab需要Python 3.10+。我们提供了`switch_simulator.sh`脚本帮助您轻松切换环境。

### 如何在仿真器之间切换？

使用提供的切换脚本：

```bash
source ./switch_simulator.sh
```

或者手动激活适当的环境并设置`SIMULATOR`环境变量。详情请参阅{doc}`getting_started/quick_start`。

### 安装过程中出现"CUDA out of memory"错误

这在首次运行时是正常的。测试脚本尝试创建大量环境。减少测试环境的数量：

```bash
python legged_gym/scripts/train.py --task=go2 --num_envs=100 --headless
```

### 我可以使用AMD或Intel GPU吗？

不可以。LeggedGym-Ex需要支持CUDA的NVIDIA GPU。仿真器（IsaacGym、Genesis、IsaacSim）都依赖于CUDA。

### 我可以在Windows或macOS上安装吗？

官方仅支持Ubuntu Linux。Windows用户可以尝试WSL2，但这不受官方支持。由于CUDA要求，不支持macOS。

### ImportError: libpython3.8.so.1.0 not found

将您的conda环境的lib目录添加到`LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH=/home/username/miniconda3/envs/lr_gym/lib:$LD_LIBRARY_PATH
```

---

## 训练

### 我需要多少个并行环境？

默认是4096个环境，在RTX 3080上运行良好。您可以根据GPU内存进行调整：

- **RTX 3060 12GB**：2048-3072个环境
- **RTX 3080 10GB**：4096个环境
- **RTX 4090 24GB**：8192+个环境

在配置文件中或通过命令行`--num_envs=2048`调整`num_envs`。

### 我需要训练多长时间？

训练时间取决于任务复杂度：

- **平地行走**：500-1000次迭代（2-4小时）
- **崎岖地形（TS/EE）**：1500-3000次迭代（6-12小时）
- **高级方法（DeepMimic、AMP）**：3000+次迭代（12+小时）

使用TensorBoard监控奖励何时达到平稳状态。

### 什么是好的奖励值？

这因任务而异。一般情况：

- **平地**：平均奖励>15表示行走良好
- **崎岖地形**：平均奖励>10且回合长度较长（>800步）表示具有鲁棒性
- **追踪奖励**：值>0.8表示命令跟随良好

同时监控`episode_length`。较长的回合意味着机器人保持稳定。

### 训练损失为NaN。我该怎么办？

常见原因和解决方案：

1. **学习率过高**：降低PPO配置中的`learning_rate`（尝试3e-4）
2. **奖励比例过大**：确保惩罚奖励正确为负值
3. **观测值爆炸**：验证观测值缩放因子是否合理
4. **梯度裁剪**：确保`clip_param`设置为0.2

### 训练速度很慢。如何加速？

- 使用`--headless`模式禁用渲染
- 如果GPU内存受限，减少`num_envs`
- 使用IsaacGym获得最快的训练速度
- 如果任务不需要，禁用`measure_heights`
- 减少`episode_length_s`以获得更快的迭代速度

### 如何从检查点恢复训练？

```bash
python -m legged_gym.scripts.train --task=go2 --resume --load_run=Sep03_16-30-16_
```

将运行名称替换为您的实际训练会话文件夹名称。

### 我可以在仅CPU上训练吗？

不可以。LeggedGym-Ex需要支持CUDA的GPU进行训练。并行环境模拟是GPU加速的。

### 如何监控训练进度？

训练指标会自动记录到TensorBoard：

```bash
tensorboard --logdir logs/go2/
```

通过在训练命令中添加`--sync_wandb`启用Weights & Biases。

### 哪些是最重要的需要调整的超参数？

对于大多数任务，重点关注：

1. **奖励比例**：平衡追踪奖励与惩罚
2. **命令范围**：从保守开始，使用课程学习扩展
3. **域随机化**：对sim-to-real迁移至关重要
4. **学习率**：默认1e-3，如果训练不稳定则降低

奖励调整指南请参阅{doc}`/developer_guide/how_to/custom_rewards`。

---

## 仿真器

### 我可以在所有仿真器中使用相同的代码吗？

可以。LeggedGym-Ex通过统一的API抽象了仿真器差异。只需设置`SIMULATOR`环境变量并使用适当的conda环境。

### 为什么我的策略在不同仿真器中表现不同？

物理引擎处理接触、摩擦和数值积分的方式不同。这是预期的。对于关键应用，在一个仿真器中训练，在另一个中验证（Sim2Sim测试）。最佳实践请参阅{doc}`getting_started/code_architecture`。

### Genesis显示网格地形碰撞问题

Genesis在自定义网格地形碰撞检测方面有已知的限制。改用`mesh_type = "heightfield"`或`mesh_type = "plane"。详情请参阅{doc}`/developer_guide/known_issues/genesis`。

### IsaacGym窗口不显示渲染的世界

使用NVIDIA驱动>=570时，设置Vulkan ICD路径：

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### IsaacLab域随机化错误

IsaacLab要求域随机化张量在CPU上。这由LeggedGym-Ex内部处理，但要注意IsaacLab中的域随机化可能比其他仿真器慢。

### 我可以为训练渲染深度图像吗？

深度相机支持因仿真器而异。IsaacGym支持基于warp的深度渲染。Genesis和IsaacLab有自己的相机实现。有关示例，请查看特定任务的实现，如`go2_ts_depth`。

---

## 部署

### 我可以将训练好的策略部署到真实机器人吗？

可以。LeggedGym-Ex支持部署到宇树Go2、TRON1和其他支持的机器人。详细说明请参阅{doc}`getting_started/deploy_to_real_robot`。

### 真实机器人部署需要更改什么？

主要更改是从观测值中移除`base_lin_vel`，因为线速度无法直接在真实机器人上测量。训练一个没有此观测分量的新策略。完整检查清单请参阅部署指南。

### 如何导出策略进行部署？

训练完成后，运行：

```bash
python -m legged_gym.scripts.play --task=go2 --load_run=your_run_name
```

这会创建一个JIT编译的策略文件，位于`logs/experiment_name/exported/policy.pt`。

### 什么是Sim2Sim测试？

Sim2Sim（Simulation-to-Simulation）在真实部署前在不同的仿真器中验证您的策略。这可以捕获仿真器特定的过拟合。我们为Go2和TRON1提供了基于MuJoCo的Sim2Sim设置。请参阅{doc}`getting_started/code_architecture`。

### 在真实部署前如何在MuJoCo中测试我的策略？

安装Sim2Sim框架（参见{doc}`getting_started/installation`），导出您的策略，然后运行：

```bash
./go2_deploy simple_rl
```

这会在MuJoCo中测试您的策略，而不会危及真实机器人。

### 我可以使用自己的机器人URDF吗？

可以。按照{doc}`/developer_guide/how_to/add_new_robot`中的指南添加您的机器人。您需要：

1. 有效的URDF（用于IsaacGym/IsaacLab）或MJCF XML（用于Genesis）
2. 关节名称和默认位置
3. 正确定义的碰撞网格

### 我需要为不同的仿真器使用不同的策略吗？

策略通常可以在仿真器之间迁移，但由于物理差异，性能可能会有所不同。为获得最佳结果，使用相同的仿真器进行训练和部署，或使用域随机化提高跨仿真器的鲁棒性。

### 真实机器人应该使用什么控制频率？

默认是50Hz（`dt = 0.02`），这与典型的机器人硬件能力相匹配。一些机器人支持100Hz或更高。确保您的部署代码与训练配置保持一致的时序。

### 我可以部署到官方不支持的机器人吗？

可以，但您需要编写自己的部署代码。关键是将观测值和动作与机器人的传感器接口和电机控制器对齐。使用提供的Go2/TRON1部署代码作为参考。

### 如何处理与机器人的网络通信？

对于宇树Go2，使用提供的`go2_deploy`框架处理DDS通信。对于其他机器人，基于机器人的SDK实现自己的通信层。策略本身只需要观测张量作为输入。

---

## 快速参考

### 基本命令

```bash
# 训练
python -m legged_gym.scripts.train --task=go2 --headless

# 可视化
python -m legged_gym.scripts.play --task=go2 --load_run=run_name

# 列出任务
python -c "from legged_gym.envs import task_registry; print(list(task_registry.task_classes.keys()))"

# TensorBoard
tensorboard --logdir logs/
```

### 关键配置文件

- 环境：`legged_gym/envs/go2/go2_config.py`
- 训练：配置文件中的PPO设置
- 奖励：配置文件中的`class rewards`
- 命令：配置文件中的`class commands`

### 有用链接

- [GitHub仓库](https://github.com/lupinjia/LeggedGym-Ex)
- [完整文档](https://genesis-lr-doc.readthedocs.io/)
- [legged_gym原版](https://github.com/leggedrobotics/legged_gym)
