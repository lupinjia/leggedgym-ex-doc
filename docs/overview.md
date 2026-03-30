# LeggedGym-Ex 文档

## 什么是 LeggedGym-Ex？

LeggedGym-Ex 是一个用于训练足式机器人运动策略的强化学习框架。它扩展了原始的 [legged_gym](https://github.com/leggedrobotics/legged_gym) 框架，支持多种物理模拟器，并实现了各种用于机器人运动的最新强化学习方法。

## 快速入门

### 安装

LeggedGym-Ex 支持三种物理模拟器。根据您的需求选择一种：

| 模拟器（Simulator） | Python 版本 | 最适用于 |
|-----------|---------------|----------|
| **Genesis** | 3.10+ | 支持软体物理的快速训练 |
| **IsaacGym** | 3.6-3.8 | 最大训练速度 |
| **IsaacSim/IsaacLab** | 3.10+ | 真实渲染 |

#### Genesis 安装

```bash
# 创建 conda 环境
conda create -n genesis python=3.10
conda activate genesis

# 安装 Genesis
pip install genesis-world

# 安装 LeggedGym-Ex
pip install -e .

# 设置模拟器
export SIMULATOR=genesis
```

#### IsaacGym 安装

```bash
# 创建 conda 环境
conda create -n isaacgym python=3.8
conda activate isaacgym

# 安装 IsaacGym（先从 NVIDIA 下载）
pip install <path_to_isaacgym>/python/isaacgym-*.whl

# 安装 LeggedGym-Ex
pip install -e .

# IsaacGym 自动检测（无需环境变量）
```

#### IsaacSim 安装

```bash
# 创建 conda 环境
conda create -n isaacsim python=3.10
conda activate isaacsim

# 安装 IsaacSim（遵循 NVIDIA 的安装指南）

# 安装 LeggedGym-Ex
pip install -e .

# 设置模拟器
export SIMULATOR=isaaclab
```

### 训练您的第一个策略

为 Unitree Go2 训练一个基础运动策略：

```bash
python legged_gym/scripts/train.py --task=go2
```

可用任务：
- `go2` - 基础 Go2 运动
- `go2_ts` - 教师-学生（Teacher-Student）训练
- `go2_ee` - 显式估计器（Explicit Estimator）
- `go2_wtw` - Walk These Ways
- `go2_dreamwaq` - DreamWaQ
- `go2_cts` - 并行教师-学生（Concurrent TS）
- `tron1pf_ee` - 带显式估计器的 TRON1
- `g1_deepmimic` - 使用 DeepMimic 的 G1 人形机器人
- `k1_amp` - 使用 AMP 的 Booster K1

### 播放训练好的策略

```bash
# 从特定检查点加载
python legged_gym/scripts/play.py --task=go2 --load_run=<run_name> --checkpoint=<checkpoint_num>

# 使用 CPU 渲染
python legged_gym/scripts/play.py --task=go2 --load_run=<run_name> --cpu

# 录制视频（仅 Genesis）
python legged_gym/scripts/play.py --task=go2 --load_run=<run_name> --record
```

### 摇杆控制

使用摇杆控制测试部署：

```bash
python legged_gym/scripts/joystick.py --task=go2 --load_run=<run_name>
```

## 主要特性

### 多模拟器支持

框架根据 Python 版本和 `SIMULATOR` 环境变量自动选择模拟器：

```python
from legged_gym import SIMULATOR

if SIMULATOR == "genesis":
    # 使用 Genesis
elif SIMULATOR == "isaacgym":
    # 使用 IsaacGym
elif SIMULATOR == "isaaclab":
    # 使用 IsaacLab
```

### 已实现的方法

LeggedGym-Ex 包含了近期论文中各种 RL 方法的实现：

| 方法（Method） | 描述（Description） | 任务示例 |
|--------|-------------|--------------|
| **Walk These Ways** | 使用指令参数的多行为训练 | `go2_wtw` |
| **Teacher-Student** | 特权教师训练盲学生 | `go2_ts` |
| **Explicit Estimator** | 联合训练策略和状态估计器 | `go2_ee` |
| **DreamWaQ** | 隐式地形想象 | `go2_dreamwaq` |
| **Concurrent TS** | 同时训练教师和学生 | `go2_cts` |
| **DeepMimic** | 从参考运动的模仿学习 | `g1_deepmimic` |
| **AMP** | 对抗运动先验 | `k1_amp` |
| **CaT** | 约束即终止 | `go2_cat` |
| **System ID** | 模拟到现实的系统辨识 | `go2_sysid` |

### 配置系统

所有超参数都在配置类中定义：

```python
from legged_gym.envs.go2.go2_config import Go2Cfg

cfg = Go2Cfg()

# 环境设置
cfg.env.num_envs = 4096
cfg.env.episode_length_s = 20

# 奖励权重
cfg.rewards.scales.tracking_lin_vel = 1.0
cfg.rewards.scales.tracking_ang_vel = 0.5

# 域随机化（Domain Randomization）
cfg.domain_rand.randomize_friction = True
cfg.domain_rand.friction_range = [0.5, 1.25]
```

### 命令行参数

训练和评估的常用参数：

```bash
# 训练参数
python legged_gym/scripts/train.py \
    --task=go2 \                    # 任务名称
    --num_envs=4096 \               # 并行环境数量
    --seed=1 \                      # 随机种子
    --run_name=test \               # 实验名称
    --max_iterations=1000 \         # 训练迭代次数
    --headless                      # 无可视化运行

# 恢复训练
python legged_gym/scripts/train.py \
    --task=go2 \
    --resume \
    --load_run=Jan01_00-00-00_test \
    --checkpoint=500
```

## 项目结构

```
LeggedGym-Ex/
├── legged_gym/
│   ├── envs/           # 环境实现
│   │   ├── base/       # 基类（LeggedRobot、配置）
│   │   ├── go2/        # Unitree Go2 机器人
│   │   ├── g1/         # Unitree G1 人形机器人
│   │   ├── k1/         # Booster K1 机器人
│   │   └── tron1*/     # LimX TRON1 机器人
│   ├── simulator/      # 模拟器抽象
│   ├── utils/          # 工具类（地形、辅助函数等）
│   └── scripts/        # 训练和评估脚本
├── resources/          # 机器人 URDF 和网格文件
└── docs/              # 文档
```

## 配置文件

每个任务都有一个配置文件，定义：

- **环境（Environment）**: 环境数量、回合长度、观测值
- **机器人资产（Robot Asset）**: URDF 路径、关节名称、默认位置
- **地形（Terrain）**: 地形类型、难度课程
- **控制（Control）**: PD 增益、降采样率（Decimation）、动作缩放
- **奖励（Rewards）**: 奖励函数及其权重
- **域随机化（Domain Randomization）**: 模拟到现实的随机化范围
- **训练（Training）**: PPO 超参数、网络架构

配置结构示例：

```python
class LeggedRobotCfg:
    class env:
        num_envs = 4096
        num_observations = 48
        episode_length_s = 20
    
    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "foot"
        
    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            torques = -0.00001
```

## 模拟到现实部署（Sim-to-Real Deployment）

在真实机器人上部署：

1. 启用域随机化进行训练
2. 将策略导出为 ONNX 或 PyTorch JIT 格式
3. 在机器人的机载计算机上运行推理
4. 将动作转换为电机指令

部署代码示例：

```python
import torch
import onnxruntime as ort

# 加载模型
session = ort.InferenceSession("policy.onnx")

# 推理循环
while True:
    obs = get_observations()  # 从机器人传感器获取
    action = session.run(None, {"obs": obs})[0]
    send_commands(action)  # 发送到机器人执行器
```

## 故障排除

### 导入错误

**Genesis 未找到**：
```bash
export SIMULATOR=genesis
conda activate genesis
```

**IsaacGym 未找到**：
```bash
conda activate isaacgym
# 确保 IsaacGym 已安装
```

### CUDA 内存不足

减少环境数量：
```bash
python legged_gym/scripts/train.py --task=go2 --num_envs=2048
```

### 训练速度慢

使用 IsaacGym 获得最大速度：
```bash
conda activate isaacgym
python legged_gym/scripts/train.py --task=go2 --headless
```

## 额外资源

- [用户指南](../user_guide/index.md) - 详细使用指南
- [开发者指南](../developer_guide/index.md) - API 参考和参数文档
- [GitHub 仓库](https://github.com/lupinjia/LeggedGym-Ex)
- [原始 legged_gym](https://github.com/leggedrobotics/legged_gym)

## 引用

如果您在研究中使用了 LeggedGym-Ex，请引用：

```bibtex
@software{leggedgym_ex,
  title={LeggedGym-Ex: A Multi-Simulator Framework for Legged Robot Learning},
  author={Yasen Jia},
  year={2025},
  url={https://github.com/lupinjia/LeggedGym-Ex}
}
```

## 支持

- GitHub Issues: [报告错误或请求功能](https://github.com/lupinjia/LeggedGym-Ex/issues)
- 飞书群：扫描 README 中的二维码获取社区支持
