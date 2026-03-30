# LeggedGym-Ex 设计文档

## 概述

LeggedGym-Ex 是一个用于足式机器人运动（Legged Robot Locomotion）的强化学习（Reinforcement Learning, RL）训练框架。它基于苏黎世联邦理工学院（ETH Zurich）机器人系统实验室（Robotic Systems Lab, RSL）的 [legged_gym](https://github.com/leggedrobotics/legged_gym) 和 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 构建。该框架提供了一个统一的接口，用于在多个物理模拟器（Physics Simulators）上训练机器人控制策略。

## 架构

### 高层架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LeggedGym-Ex                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  训练脚本       │    │  环境任务       │    │  模拟器抽象层           │  │
│  │                 │───▶│                 │───▶│                         │  │
│  │ • train.py      │    │ • LeggedRobot   │    │ • Simulator (Base)      │  │
│  │ • play.py       │    │ • Extensions    │    │ • GenesisSimulator      │  │
│  │ • joystick.py   │    │   (TS, EE, etc) │    │ • IsaacGymSimulator     │  │
│  └─────────────────┘    └─────────────────┘    │ • IsaacLabSimulator     │  │
│                                                └─────────────────────────┘  │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  配置系统       │    │  工具类         │    │  任务注册表             │  │
│  │                 │    │                 │    │                         │  │
│  │ • Config Classes│    │ • Math Utils    │    │ • Task Registration     │  │
│  │ • YAML/Classes  │    │ • Terrain Gen   │    │ • Env Creation          │  │
│  └─────────────────┘    │ • Helpers       │    │ • Runner Creation       │  │
│                         └─────────────────┘    └─────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. 模拟器抽象层（Simulator Abstraction Layer）(`legged_gym/simulator/`)

该框架通过统一接口抽象了三种不同的物理模拟器：

- **基类（Base Class）** (`simulator.py`): 定义模拟器接口的抽象基类
  - 环境管理（创建、重置、步进）
  - 状态访问（自由度（Degrees of Freedom, DOF）位置、速度、基座姿态等）
  - 域随机化（Domain Randomization）（摩擦、质量、质心等）
  - 传感器更新（接触力、深度相机）

- **GenesisSimulator** (`genesis_simulator.py`): Genesis 物理引擎集成
  - 支持 Python 3.10+
  - 支持软体物理（Soft Body Physics）的快速训练
  - GPU 加速模拟

- **IsaacGymSimulator** (`isaacgym_simulator.py`): NVIDIA Isaac Gym 集成
  - 支持 Python 3.6-3.8
  - 最快的训练速度
  - 基于 GPU 的物理模拟

- **IsaacLabSimulator** (`isaaclab_simulator.py`): NVIDIA IsaacLab/Isaac Sim 集成
  - 支持 Python 3.10+
  - 更真实的渲染效果
  - 高级传感器模拟

**模拟器选择（Simulator Selection）**: 由运行时 `SIMULATOR` 环境变量和 Python 版本决定：
```python
# Python 3.10+: Genesis 或 IsaacLab
# Python 3.6-3.8: IsaacGym
simulator_type = os.getenv("SIMULATOR")  # "genesis" 或 "isaaclab"
```

#### 2. 任务/环境层（Task/Environment Layer） (`legged_gym/envs/`)

**基类（Base Classes）**:
- `BaseTask` (`base/base_task.py`): 所有 RL 任务的基础，管理模拟器实例化
- `LeggedRobot` (`base/legged_robot.py`): 核心机器人运动环境，包含：
  - 观测值（Observation）计算
  - 奖励（Reward）计算
  - 终止条件（Termination）检查
  - 重置处理

**扩展类（Extensions）** (继承自 `LeggedRobot`):
- `LeggedRobotTS` (`base/legged_robot_ts.py`): 教师-学生（Teacher-Student）训练
- `LeggedRobotEE` (`base/legged_robot_ee.py`): 显式估计器（Explicit Estimator）
- `LeggedRobotDreamWaQ` (`base/legged_robot_dreamwaq.py`): DreamWaQ 地形想象
- `LeggedRobotCTS` (`base/legged_robot_cts.py`): 并行教师-学生（Concurrent Teacher-Student）
- `LeggedRobotAMP` (`base/legged_robot_amp.py`): 对抗运动先验（Adversarial Motion Priors）

**特定机器人实现（Robot-Specific Implementations）**:
```
envs/
├── go2/          # Unitree Go2 四足机器人
├── g1/           # Unitree G1 人形机器人
├── k1/           # Booster K1 机器人
├── tron1pf/      # LimX TRON1 (并联足)
├── tron1sf/      # LimX TRON1 (串联足)
└── bipedal_walker/  # 双足行走机器人
```

每个机器人目录包含：
- `<robot>.py`: 主环境类
- `<robot>_config.py`: 配置类
- 不同方法的子目录（如 `go2_ts/`、`go2_ee/`）

#### 3. 配置系统（Configuration System） (`legged_gym/envs/base/`)

配置通过 Python 类处理：

- `LeggedRobotCfg`: 主配置类，包含嵌套类：
  - `env`: 环境设置（环境数量、回合长度等）
  - `asset`: 机器人资产配置（URDF 路径、关节名称等）
  - `terrain`: 地形生成参数
  - `control`: 控制参数（PD 增益、降采样率等）
  - `rewards`: 奖励函数权重和参数
  - `domain_rand`: 域随机化范围
  - `sim`: 模拟器特定设置

- `LeggedRobotCfgPPO`: 训练配置，包含：
  - `runner`: 训练运行器设置
  - `algorithm`: PPO 算法参数
  - `policy`: 策略网络架构

#### 4. 任务注册表（Task Registry） (`legged_gym/utils/task_registry.py`)

集中式任务管理：
```python
task_registry = TaskRegistry()

# 注册
 task_registry.register(name, task_class, env_cfg, train_cfg)

# 环境创建
 env, env_cfg = task_registry.make_env(name, args)

# 算法运行器创建
 runner, train_cfg = task_registry.make_alg_runner(env, name, args)
```

#### 5. 工具类（Utilities） (`legged_gym/utils/`)

- `math_utils.py`: 数学工具（四元数操作、噪声生成）
- `terrain.py`: 程序化地形生成
- `terrain_utils.py`: 地形生成基元
- `helpers.py`: 配置加载、参数解析
- `logger.py`: 训练日志工具
- `motion_loader.py`: 参考运动加载（用于 DeepMimic/AMP）

## 数据流

### 训练循环

```
1. train.py
   └── task_registry.make_env() 
       └── LeggedRobot.__init__()
           └── BaseTask.__init__()
               └── Simulator.__init__()
                   └── _create_sim() -> _create_envs()
   └── task_registry.make_alg_runner()
       └── runner.learn()
           └── for iteration in range(max_iterations):
               └── env.step(actions)
                   ├── _pre_sim_step()
                   ├── simulator.step(actions)
                   └── post_physics_step()
                       ├── check_termination()
                       ├── compute_reward()
                       ├── reset_idx()
                       └── compute_observations()
```

### 观测流水线

```
机器人状态（模拟器）
    │
    ▼
LeggedRobot.compute_observations()
    │
    ├── 基础观测（重力、角速度、指令）
    ├── 关节观测（位置、速度）
    ├── 动作历史
    ├── 高度测量（可选）
    └── 深度图像（可选）
    │
    ▼
归一化和裁剪（Normalized & Clipped）
    │
    ▼
PPO 策略网络
```

## 关键设计决策

### 1. 模拟器抽象

**原理（Rationale）**: 允许相同代码在不同模拟器上无需修改即可运行。

**实现（Implementation）**: 抽象基类配合模拟器特定实现。所有模拟器特定代码都隔离在模拟器类中。

### 2. 代码即配置（Configuration as Code）

**原理（Rationale）**: 类型安全、IDE 支持、配置变体的继承。

**实现（Implementation）**: 使用 Python 数据类（Dataclasses）进行配置，支持继承：
```python
class Go2TSCfg(Go2Cfg):
    class env(Go2Cfg.env):
        num_observations = 48 + 3  # 添加特权信息
```

### 3. 任务注册模式（Task Registration Pattern）

**原理（Rationale）**: 环境定义和使用的清晰分离。

**实现（Implementation）**: 全局注册表将名称映射到 (类, 配置) 元组。

### 4. 模块化奖励系统（Modular Reward System）

**原理（Rationale）**: 便于使用不同奖励组合进行实验。

**实现（Implementation）**: 在配置中定义奖励函数，在 `compute_reward()` 中计算：
```python
for name, reward_func in self.reward_functions.items():
    rew = reward_func() * self.reward_scales[name]
    self.rew_buf += rew
```

### 5. 域随机化集成（Domain Randomization Integration）

**原理（Rationale）**: 模拟到现实（Sim-to-Real）迁移需要域随机化。

**实现（Implementation）**: 内置于模拟器，使用归一化的特权观测：
```python
@property
def dr_friction_values(self):
    return dr_normalize(self._friction_values, ...)
```

## 文件组织

```
legged_gym/
├── __init__.py              # 包初始化，模拟器选择
├── envs/
│   ├── __init__.py          # 任务注册
│   ├── base/
│   │   ├── base_task.py     # 基础任务类
│   │   ├── base_config.py   # 基础配置类
│   │   ├── legged_robot.py  # 主机器人环境
│   │   ├── legged_robot_config.py  # 主配置
│   │   └── ...              # 扩展（TS, EE 等）
│   ├── go2/                 # Go2 机器人
│   ├── g1/                  # G1 机器人
│   ├── k1/                  # K1 机器人
│   ├── tron1pf/             # TRON1 PF
│   ├── tron1sf/             # TRON1 SF
│   └── bipedal_walker/      # 双足行走机器人
├── simulator/
│   ├── __init__.py          # 模拟器导出
│   ├── simulator.py         # 基础模拟器类
│   ├── genesis_simulator.py # Genesis 实现
│   ├── isaacgym_simulator.py# IsaacGym 实现
│   └── isaaclab_simulator.py# IsaacLab 实现
├── utils/
│   ├── __init__.py
│   ├── task_registry.py     # 任务注册
│   ├── helpers.py           # 工具函数
│   ├── terrain.py           # 地形生成
│   └── ...
├── scripts/
│   ├── train.py             # 训练脚本
│   ├── play.py              # 播放/评估脚本
│   └── ...
└── warp/                    # 基于 Warp 的深度相机
    └── ...
```

## 扩展点

### 添加新机器人

1. 创建 `envs/<robot>/` 目录
2. 实现 `<robot>.py` 继承自 `LeggedRobot`
3. 实现 `<robot>_config.py` 使用 `LeggedRobotCfg` 子类
4. 在 `envs/__init__.py` 中注册

### 添加新模拟器

1. 创建 `simulator/<name>_simulator.py`
2. 继承自 `Simulator` 基类
3. 实现所有抽象方法
4. 添加到 `BaseTask` 模拟器选择逻辑

### 添加新方法

1. 在 `envs/base/` 中创建基类（如需要）
2. 在 `envs/<robot>/<method>/` 中创建特定机器人实现
3. 继承自适当的基类
4. 覆盖相关方法（奖励、观测等）

## 性能考虑

- **向量化（Vectorization）**: 所有操作都使用 PyTorch 张量在环境间进行向量化
- **GPU 加速**: 物理模拟和网络推理在 GPU 可用时运行于 GPU
- **JIT 编译**: 关键路径启用 PyTorch JIT 优化
- **内存管理**: 仔细的重用缓冲区以最小化分配

## 依赖项

- **PyTorch**: 深度学习框架
- **rsl_rl**: RSL 的 RL 库，用于 PPO 实现
- **Genesis/IsaacGym/IsaacLab**: 物理模拟器（需要一个）
- **NumPy**: 数值运算
- **其他**: 参见 `setup.py` 获取完整列表

## 未来方向

- 使用 NVIDIA Warp 的深度相机支持
- 额外的模拟器（MuJoCo、Drake）
- 多机器人训练
- 基于模型的 RL 集成
