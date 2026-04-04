# `legged_robot.py` - LeggedRobot 类 API 参考

本文档提供了 `LeggedRobot` 类的全面 API 文档，它是 LeggedGym-Ex 框架中所有足式机器人任务的基础环境类。

---

## 概述

`LeggedRobot` 类继承自 `BaseTask`，为使用强化学习训练足式机器人运动策略提供核心功能。它管理仿真环境、处理观察和奖励、处理终止条件并编排训练循环。

所有特定机器人的环境（Go2、G1、K1、TRON1）都继承自这个基础类，要么直接继承，要么通过专门的变体如 `LeggedRobotTS`（教师-学生）、`LeggedRobotEE`（显式估计器）或 `LeggedRobotAMP`（对抗运动先验）继承。

---

## 类型别名

该类使用几个类型别名以提高清晰度和文档说明：

```python
ObsBuf = Tensor  # 形状: (num_envs, obs_dim)
Action = Tensor  # 形状: (num_envs, num_actions)
Reward = Tensor  # 形状: (num_envs,)
EnvIds = Tensor  # 形状: (num_reset_envs,) - 环境索引的整数张量
```

---

## 类定义

```python
class LeggedRobot(BaseTask):
    """足式机器人运动任务的基础环境。
    
    提供核心功能：
    - 多环境仿真管理
    - 观察和奖励计算
    - 终止检查和环境重置
    - 课程学习支持
    - 域随机化
    """
```

---

## 初始化

### `__init__()`

```python
def __init__(
    self,
    cfg: LeggedRobotCfg,
    sim_params: dict[str, Any],
    sim_device: str | int,
    headless: bool
) -> None
```

初始化足式机器人环境。

**参数:**

- **cfg** (`LeggedRobotCfg`): 包含机器人、地形、控制和奖励参数的环境配置。必须包含 `env`、`normalization`、`sim` 和 `control` 部分。
- **sim_params** (`dict[str, Any]`): 传递给仿真器后端（IsaacGym、Genesis 或 IsaacSim）的仿真参数字典。
- **sim_device** (`str | int`): 运行仿真的设备。可以是 `'cuda'`、`'cpu'` 或设备 ID 整数如 `0` 或 `'cuda:0'`。
- **headless** (`bool`): 如果为 `True`，则不渲染运行以加速训练。

**抛出:**

- `AssertionError`: 如果配置缺少必需部分或有无效值。

**示例:**

```python
from legged_gym.envs.go2.go2_config import GO2Cfg

cfg = GO2Cfg()
sim_params = {"dt": 0.005, "substeps": 1}
env = LeggedRobot(cfg, sim_params, "cuda:0", headless=True)
```

**验证:**

初始化执行广泛的配置验证以尽早捕获错误：

- 验证必需的配置部分（`env`、`normalization`、`sim`、`control`）
- 检查观察和动作维度是否为正
- 确保仿真器创建有效数量的环境
- 准备奖励函数并初始化缓冲区

---

## 核心方法

### `step()`

```python
def step(
    self, 
    actions: Action
) -> Tuple[ObsBuf, ObsBuf | None, Reward, Tensor, Dict[str, Any]]
```

使用给定动作执行一次仿真步骤。

将动作应用于机器人，推进物理仿真，然后处理观察、奖励和终止状态。这是 RL 训练循环的主要接口。

**参数:**

- **actions** (`Action`): 形状为 `(num_envs, num_actions)` 的动作张量，根据控制模式包含目标关节位置或力矩。必须是 `float32` 数据类型。

**返回:**

包含以下内容的元组：
- **obs_buf** (`ObsBuf`): 形状为 `(num_envs, obs_dim)` 的观察缓冲区。
- **privileged_obs_buf** (`ObsBuf | None`): 形状为 `(num_envs, privileged_obs_dim)` 的特权观察值，如果不使用非对称 Actor-Critic 则为 `None`。
- **rew_buf** (`Reward`): 形状为 `(num_envs,)` 的奖励缓冲区。
- **reset_buf** (`Tensor`): 形状为 `(num_envs,)` 的重置标志，指示哪些环境需要重置（1 表示重置，0 否则）。
- **extras** (`Dict[str, Any]`): 包含额外信息的字典，包括回合统计和课程数据。

**抛出:**

- `AssertionError`: 如果动作形状或数据类型不正确。

**示例:**

```python
# 标准 RL 训练循环
actions = policy(obs)  # 策略网络输出
obs, priv_obs, rew, done, info = env.step(actions)

# 检查回合统计
if "episode" in info:
    for key, value in info["episode"].items():
        print(f"{key}: {value}")
```

**验证:**

- 验证动作形状是否匹配 `(num_envs, num_actions)`
- 确保动作为 `float32` 数据类型
- 将观察裁剪到配置的范围

---

### `reset_idx()`

```python
def reset_idx(self, env_ids: EnvIds) -> None
```

将指定的环境重置为初始状态。

对指定环境执行完全重置，包括课程更新、命令重采样、自由度状态重置和缓冲区重置。此方法会自动为终止的环境调用，但也可以手动调用。

**参数:**

- **env_ids** (`EnvIds`): 形状为 `(num_reset_envs,)` 的整数张量，包含要重置的环境索引。可以是空张量。

**示例:**

```python
# 重置特定环境
env_ids = torch.tensor([0, 5, 10], device="cuda")
env.reset_idx(env_ids)

# 重置所有环境
env_ids = torch.arange(env.num_envs, device="cuda")
env.reset_idx(env_ids)
```

**重置过程:**

1. 如果启用，更新地形课程
2. 如果启用，更新命令课程
3. 重采样速度命令
4. 使用小的随机扰动重置自由度位置
5. 重置根状态（基础位置、方向、速度）
6. 清除动作历史缓冲区
7. 重置回合长度和失败计数器
8. 将回合统计记录到 `extras` 字典

---

### `reset()`

继承自 `BaseTask`。重置所有环境。

```python
def reset(self) -> Tuple[ObsBuf, ObsBuf | None]:
    """重置所有环境。"""
    env_ids = torch.arange(self.num_envs, device=self.device)
    self.reset_idx(env_ids)
    return self.obs_buf, self.privileged_obs_buf
```

---

### `compute_observations()`

```python
def compute_observations(self) -> None
```

为所有环境计算观察值。

从仿真器状态构建观察张量并应用归一化尺度。此方法在每次物理步骤后自动调用，但可以重写以自定义观察结构。

**默认观察组件:**

| 组件 | 形状 | 描述 | 缩放因子 |
|-----------|-------|-------------|--------------|
| 基础线速度 | (3,) | 身体坐标系中的基础速度 | `obs_scales.lin_vel` |
| 投影重力 | (3,) | 世界坐标系中的身体 z 轴 | 1.0 |
| 基础角速度 | (3,) | 身体坐标系中的角速度 | `obs_scales.ang_vel` |
| 命令 | (3,) | 线速度 x、线速度 y、角速度偏航 | `commands_scale` |
| 自由度位置 | (num_dofs,) | 与默认位置的偏差 | `obs_scales.dof_pos` |
| 自由度速度 | (num_dofs,) | 关节速度 | `obs_scales.dof_vel` |
| 动作 | (num_actions,) | 历史中的前一个动作 | 1.0 |

**可选组件:**

- **高度测量**: 如果 `cfg.terrain.measure_heights` 为 `True` 则添加。形状: `(num_height_points,)`，缩放因子为 `obs_scales.height_measurements`。
- **观察噪声**: 如果 `cfg.noise.add_noise` 为 `True` 则添加。使用均匀噪声，缩放因子为 `noise_scale_vec`。

**特权观察:**

如果设置了 `num_privileged_obs`，则计算 `privileged_obs_buf` 并包含附加信息：

| 组件 | 形状 | 描述 |
|-----------|-------|-------------|
| 标准观察 | - | 常规观察的所有组件 |
| 上一个动作 | (num_actions,) | 两步之前的动作 |
| 摩擦值 | (1,) | 域随机化摩擦 |
| 添加的基础质量 | (1,) | 域随机化质量 |
| 基础质心偏差 | (3,) | 域随机化质心偏移 |
| 推力速度 | (2,) | 域随机化推力速度 |

**更新:**

- `self.obs_buf`: 形状 `(num_envs, obs_dim)`
- `self.privileged_obs_buf`: 形状 `(num_envs, privileged_obs_dim)` 或 `None`

**示例重写:**

```python
class MyRobot(LeggedRobot):
    def compute_observations(self):
        # 调用父类获取基础观察
        super().compute_observations()
        
        # 添加自定义观察
        custom_obs = self._compute_custom_features()
        self.obs_buf = torch.cat([self.obs_buf, custom_obs], dim=-1)
```

**重要:**

重写此方法时，确保最终观察大小与 `cfg.env.num_observations` 匹配。如果大小不匹配，类会包含引发 `AssertionError` 的验证。

---

### `get_observations()`

继承自 `BaseTask`。返回当前观察值而不重新计算。

```python
def get_observations(self) -> Tuple[ObsBuf, ObsBuf | None]:
    """获取当前观察值。"""
    return self.obs_buf, self.privileged_obs_buf
```

---

### `check_termination()`

```python
def check_termination(self) -> None
```

检查终止条件并更新重置缓冲区。

为每个环境评估三个终止条件：

1. **接触终止**: 与终止身体的接触力超过阈值（10.0N）。
2. **方向终止**: 投影重力 z 分量超过 `cfg.env.max_projected_gravity`。
3. **超时终止**: 回合超过 `max_episode_length`。

**更新:**

- `self.fail_buf`: 跟踪优雅终止逻辑的连续失败。
- `self.time_out_buf`: 指示超时的回合（非实际失败）。
- `self.reset_buf`: 指示需要重置的环境。

**优雅终止:**

`fail_buf` 计数器允许优雅终止：环境仅在失败持续 `cfg.env.fail_to_terminal_time_s / dt` 步后才重置，防止来自短暂扰动的过早终止。

---

### `check_timeout()`

在 `check_termination()` 中隐式处理。检查回合长度是否超过最大值。

```python
self.time_out_buf = self.episode_length_buf > self.max_episode_length
```

---

### `compute_reward()`

```python
def compute_reward(self) -> None
```

为所有环境计算奖励。

遍历 `cfg.rewards.scales` 中定义的所有具有非零尺度的奖励函数，计算每个奖励项，按 `dt` 缩放，并累加到总奖励中。

**奖励结构:**

奖励由配置的 `scales` 字典定义：

```python
cfg.rewards.scales.tracking_lin_vel = 1.0
cfg.rewards.scales.lin_vel_z = -0.5
cfg.rewards.scales.termination = -0.0
```

**更新:**

- `self.rew_buf`: 形状为 `(num_envs,)` 的总奖励。
- `self.episode_sums`: 用于记录每项累积奖励的字典。

**处理:**

1. 从考虑中移除零尺度
2. 非零尺度乘以 `dt` 以进行适当积分
3. 调用每个奖励函数 `_reward_<name>()`
4. 如果 `cfg.rewards.only_positive_rewards`，可选择将奖励裁剪为正值
5. 裁剪后添加终止惩罚

---

### `post_physics_step()`

```python
def post_physics_step(self) -> None
```

物理步骤后处理环境状态。

此方法在每次仿真步骤后编排 RL 训练循环逻辑。它由 `step()` 自动调用，不应直接调用。

**执行顺序:**

1. 更新回合和步数计数器
2. 处理仿真器后物理回调
3. 运行自定义回调 (`_post_physics_step_callback`)
4. 检查终止条件
5. 计算奖励
6. 重置终止的环境
7. 更新传感器
8. 计算观察值
9. 绘制调试可视化（如果启用）

---

### `set_viewer_camera()`

```python
def set_viewer_camera(
    self, 
    pos: np.ndarray, 
    lookat: np.ndarray
) -> None
```

设置查看器相机位置和方向以进行渲染。

**参数:**

- **pos** (`np.ndarray`): 世界坐标系中的相机位置，形状 `(3,)`。
- **lookat** (`np.ndarray`): 世界坐标系中要看向的点，形状 `(3,)`。

**示例:**

```python
# 将相机定位在机器人上方和后方
pos = np.array([2.0, 0.0, 1.0])
lookat = np.array([0.0, 0.0, 0.5])
env.set_viewer_camera(pos, lookat)
```

---

## 关键属性

### 观察和状态缓冲区

| 属性 | 类型 | 形状 | 描述 |
|-----------|------|-------|-------------|
| `obs_buf` | `Tensor` | `(num_envs, obs_dim)` | 主要观察缓冲区 |
| `privileged_obs_buf` | `Tensor \| None` | `(num_envs, privileged_obs_dim)` | 非对称训练的特权观察 |
| `reward_buf` | `Tensor` | `(num_envs,)` | 奖励缓冲区 |
| `reset_buf` | `Tensor` | `(num_envs,)` | 重置标志 |
| `time_out_buf` | `Tensor` | `(num_envs,)` | 超时标志 |

### 机器人状态

| 属性 | 类型 | 形状 | 描述 |
|-----------|------|-------|-------------|
| `dof_pos` | `Tensor` | `(num_envs, num_dofs)` | 通过 `simulator.dof_pos` 获取的关节位置 |
| `dof_vel` | `Tensor` | `(num_envs, num_dofs)` | 通过 `simulator.dof_vel` 获取的关节速度 |
| `dof_acc` | `Tensor` | `(num_envs, num_dofs)` | 关节加速度（计算得到） |
| `base_pos` | `Tensor` | `(num_envs, 3)` | 世界坐标系中的基础位置 |
| `base_quat` | `Tensor` | `(num_envs, 4)` | 基础方向四元数（x, y, z, w） |
| `base_lin_vel` | `Tensor` | `(num_envs, 3)` | 世界坐标系中的基础线速度 |
| `base_ang_vel` | `Tensor` | `(num_envs, 3)` | 身体坐标系中的基础角速度 |

### 控制和动作

| 属性 | 类型 | 形状 | 描述 |
|-----------|------|-------|-------------|
| `actions` | `Tensor` | `(num_envs, num_actions)` | 当前动作 |
| `last_actions` | `Tensor` | `(num_envs, num_actions)` | 上一步的动作 |
| `llast_actions` | `Tensor` | `(num_envs, num_actions)` | 两步之前的动作 |
| `commands` | `Tensor` | `(num_envs, num_commands)` | 速度命令 |

### 回合跟踪

| 属性 | 类型 | 形状 | 描述 |
|-----------|------|-------|-------------|
| `episode_length_buf` | `Tensor` | `(num_envs,)` | 当前回合的步数长度 |
| `fail_buf` | `Tensor` | `(num_envs,)` | 连续失败计数器 |
| `episode_sums` | `Dict[str, Tensor]` | - | 每项的累积奖励 |
| `extras` | `Dict[str, Any]` | - | 额外的回合信息 |

---

## 奖励自动发现机制

`LeggedRobot` 类实现了一个强大的奖励函数自动发现机制。这允许您通过简单地实现具有命名约定 `_reward_<name>()` 的方法来定义奖励项。

### 工作原理

1. **配置**: 在配置中定义奖励尺度：

```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            lin_vel_z = -0.5
            orientation = -0.5
            termination = -0.0
```

2. **实现**: 实现相应的方法：

```python
class MyRobot(LeggedRobot):
    def _reward_tracking_lin_vel(self) -> Reward:
        """追踪线速度命令的奖励。"""
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.simulator.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self) -> Reward:
        """惩罚 z 轴线速度。"""
        return torch.square(self.simulator.base_lin_vel[:, 2])
```

3. **自动发现**: 在初始化期间，`_prepare_reward_function()` 自动：
   - 查找所有具有非零值的奖励尺度
   - 查找对应的 `_reward_<name>()` 方法
   - 验证方法是否存在
   - 创建可调用奖励函数列表
   - 将尺度乘以 `dt` 以进行适当积分

### 内置奖励函数

基础类提供了许多可立即使用的奖励函数：

| 方法 | 描述 | 典型用途 |
|--------|-------------|-------------|
| `_reward_tracking_lin_vel()` | 追踪线速度命令 | 速度追踪任务 |
| `_reward_tracking_ang_vel()` | 追踪角速度命令 | 转向行为 |
| `_reward_lin_vel_z()` | 惩罚垂直速度 | 稳定运动 |
| `_reward_ang_vel_xy()` | 惩罚横滚/俯仰速率 | 稳定运动 |
| `_reward_orientation()` | 惩罚非平坦基础 | 稳定性 |
| `_reward_base_height()` | 维持目标高度 | 姿态控制 |
| `_reward_torques()` | 惩罚大力矩 | 能量效率 |
| `_reward_dof_vel()` | 惩罚关节速度 | 平滑运动 |
| `_reward_dof_acc()` | 惩罚关节加速度 | 平滑运动 |
| `_reward_action_rate()` | 惩罚动作变化 | 平滑控制 |
| `_reward_collision()` | 惩罚身体碰撞 | 安全性 |
| `_reward_feet_air_time()` | 奖励适当的摆动阶段 | 步态质量 |
| `_reward_dof_pos_limits()` | 惩罚关节限制 | 安全性 |
| `_reward_torque_limits()` | 惩罚力矩限制 | 硬件安全性 |

### 自定义奖励

添加自定义奖励：

```python
class MyRobot(LeggedRobot):
    def _reward_custom_balance(self) -> Reward:
        """自定义平衡奖励。"""
        # 实现您的奖励逻辑
        balance_error = torch.abs(self.simulator.projected_gravity[:, 2] - 1.0)
        return torch.exp(-balance_error)
```

然后添加到配置：

```python
class rewards:
    class scales:
        custom_balance = 0.5
```

### 验证

系统包含验证，如果出现以下情况则引发 `AssertionError`：
- 未定义奖励尺度
- 定义了奖励尺度但不存在相应的方法
- 方法没有返回正确的形状 `(num_envs,)`

---

## 回调方法（受保护）

这些方法可以被重写以自定义行为：

### `_pre_sim_step()`

```python
def _pre_sim_step(self, actions: Action) -> Action
```

在 `step()` 的开头调用，在仿真之前。默认实现裁剪动作并更新动作历史。

### `_post_physics_step_callback()`

```python
def _post_physics_step_callback(self) -> None
```

在物理步骤后调用，在终止检查之前。默认实现：
- 定期重采样命令
- 处理航向命令
- 对机器人应用随机推力

**重写示例:**

```python
class MyRobot(LeggedRobot):
    def _post_physics_step_callback(self):
        # 首先调用父类
        super()._post_physics_step_callback()
        
        # 添加自定义逻辑
        self._update_custom_curriculum()
```

### `_resample_commands()`

```python
def _resample_commands(self, env_ids: EnvIds) -> None
```

为指定环境重采样速度命令。在重置期间和回合期间定期调用。

### `_reset_dofs()`

```python
def _reset_dofs(self, env_ids: EnvIds) -> None
```

为指定环境重置自由度状态。默认在默认位置添加小的随机扰动。

### `_reset_root_states()`

```python
def _reset_root_states(self, env_ids: EnvIds) -> None
```

为指定环境重置根状态（基础位置、方向、速度）。

### `_update_terrain_curriculum()`

```python
def _update_terrain_curriculum(self, env_ids: EnvIds) -> None
```

根据机器人性能更新地形难度。实现游戏启发的课程，其中移动很远的机器人会进入更难的地形。

### `_update_command_curriculum()`

```python
def _update_command_curriculum(self, env_ids: EnvIds) -> None
```

根据追踪性能增加命令难度。

---

## 常见重写模式

### 自定义观察

```python
class MyRobot(LeggedRobot):
    def compute_observations(self):
        # 获取基础观察
        super().compute_observations()
        
        # 添加自定义传感器数据
        sensor_data = self._get_sensor_readings()  # 形状: (num_envs, sensor_dim)
        self.obs_buf = torch.cat([self.obs_buf, sensor_data], dim=-1)
        
        # 如果需要更新噪声向量
        if self.add_noise:
            self.noise_scale_vec = torch.cat([
                self.noise_scale_vec,
                torch.zeros(self.num_envs, sensor_dim, device=self.device)
            ], dim=-1)
```

**重要**: 重写 `compute_observations()` 时，您还必须更新 `cfg.env.num_observations` 以匹配新的观察维度。

### 自定义终止

```python
class MyRobot(LeggedRobot):
    def check_termination(self):
        # 调用父类获取默认终止条件
        super().check_termination()
        
        # 添加自定义终止
        custom_failure = self._check_custom_failure_condition()
        self.fail_buf += custom_failure
        
        # 更新重置缓冲区
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )
```

### 自定义奖励

```python
class MyRobot(LeggedRobot):
    def _reward_gait_quality(self) -> Reward:
        """奖励平滑、自然的步态模式。"""
        # 计算基于相位的步态度量
        gait_error = self._compute_gait_phase_error()
        return torch.exp(-gait_error / 0.1)
```

### 域随机化定制

```python
class MyRobot(LeggedRobot):
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        
        # 添加自定义随机化
        if self.common_step_counter % 1000 == 0:
            self._randomize_custom_parameters()
```

---

## 最佳实践

### 配置验证

基础类包含广泛的验证。始终确保：
- 配置具有必需部分：`env`、`normalization`、`sim`、`control`
- `num_observations` 与实际观察大小匹配
- `num_actions` 与自由度数量匹配
- 奖励尺度具有对应的 `_reward_*()` 方法

### 观察修改

修改观察时：
1. 更新 `cfg.env.num_observations`
2. 如果使用噪声，更新 `_get_noise_scale_vec()`
3. 启用断言进行测试以尽早捕获大小不匹配

### 奖励函数实现

- 始终返回形状 `(num_envs,)`
- 追踪任务使用指数奖励：`torch.exp(-error/sigma)`
- 正则化使用平方惩罚：`torch.square(quantity)`
- 记录奖励目的和参数

### 性能考虑

- 使用向量化操作而非循环
- 避免奖励函数中的 Python 控制流
- 缓存频繁访问的属性
- 非训练计算使用 `torch.no_grad()`

---

## 另请参阅

- {doc}`../parameter_reference/legged_robot_config` - 配置参数
- {doc}`../how_to/add_new_robot` - 添加新机器人的指南
- {doc}`../how_to/custom_rewards` - 自定义奖励函数指南
