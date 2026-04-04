# Simulator API 参考

Simulator 抽象层为在多个物理仿真器中运行腿足机器人强化学习提供了统一的接口。本文档涵盖抽象基类及其三个实现：Genesis、IsaacGym 和 IsaacLab。

## 概述

LeggedGym-Ex 通过共同的抽象接口支持三个仿真器：

| 仿真器 | 资源格式 | 主要特性 |
|-----------|--------------|--------------|
| Genesis | MJCF/XML | 流体/软体材料支持，训练速度快 |
| IsaacGym | URDF | 最快的训练速度，支持 warp 相机 |
| IsaacLab | URDF | 最佳渲染质量，IsaacSim 集成 |

## 仿真器选择

仿真器后端通过 `SIMULATOR` 环境变量进行选择：

```bash
# 使用 Genesis 仿真器
export SIMULATOR=genesis

# 使用 IsaacGym 仿真器  
export SIMULATOR=isaacgym

# 使用 IsaacLab/IsaacSim 仿真器
export SIMULATOR=isaaclab
```

框架在初始化期间会根据此环境变量自动导入适当的仿真器实现。

```{note}
仿真器选择必须在导入任何 legged_gym 模块之前设置。`SIMULATOR` 变量在导入时读取，无法在运行时更改。
```

## Simulator 抽象基类

`Simulator` 抽象基类定义了所有仿真器实现必须遵循的接口。它位于 `legged_gym/simulator/simulator.py`。

### 构造函数

```python
class Simulator(ABC):
    def __init__(self, cfg: LeggedRobotCfg, sim_params: dict, 
                 sim_device: str = "cuda:0", headless: bool = False)
```

**参数：**
- `cfg` (`LeggedRobotCfg`): 包含资源、控制、地形和域随机化设置的机器人配置
- `sim_params` (`dict`): 包含 dt、子步骤和物理引擎设置的仿真参数
- `sim_device` (`str`): 仿真设备（默认："cuda:0"）
- `headless` (`bool`): 是否在无可视化模式下运行（默认：False）

### 核心抽象方法

#### step()

```python
@abstractmethod
def step(self, actions: Tensor) -> None
```

执行仿真步骤，包括应用动作、推进物理仿真和更新内部状态缓冲区。该方法每个控制步骤运行 `cfg.control.decimation` 个物理步骤。

**参数：**
- `actions` (`Tensor`): 形状为 `(num_envs, num_actions)` 的动作张量，值通常在 `[-1, 1]` 范围内

**示例：**
```python
actions = policy(observation)
simulator.step(actions)
```

#### post_physics_step()

```python
@abstractmethod
def post_physics_step(self) -> None
```

每个控制步骤后调用，用于更新派生量。该方法刷新状态缓冲区，包括基础位置、朝向、速度、接触力和地形高度。

#### reset_idx()

```python
@abstractmethod
def reset_idx(self, env_ids: Tensor) -> None
```

将指定环境重置到初始状态。

**参数：**
- `env_ids` (`Tensor`): 要重置的环境 ID 整数张量

**示例：**
```python
# 重置已终止的环境
terminated_envs = dones.nonzero(as_tuple=False).flatten()
simulator.reset_idx(terminated_envs)
```

#### reset_dofs()

```python
@abstractmethod
def reset_dofs(self, env_ids: Tensor, dof_pos: Tensor, dof_vel: Tensor) -> None
```

为指定环境重置 DOF（自由度）位置和速度。

**参数：**
- `env_ids` (`Tensor`): 要重置的环境 ID
- `dof_pos` (`Tensor`): 目标 DOF 位置，形状 `(len(env_ids), num_dof)`
- `dof_vel` (`Tensor`): 目标 DOF 速度，形状 `(len(env_ids), num_dof)`

#### reset_root_states()

```python
@abstractmethod
def reset_root_states(self, env_ids: Tensor, base_pos: Tensor, 
                      base_quat: Tensor, base_lin_vel_w: Tensor, 
                      base_ang_vel_w: Tensor) -> None
```

重置根（基础）连杆状态，包括位置、朝向和速度。

**参数：**
- `env_ids` (`Tensor`): 要重置的环境 ID
- `base_pos` (`Tensor`): 世界坐标系中的基础位置，形状 `(len(env_ids), 3)`
- `base_quat` (`Tensor`): 四元数形式的基础朝向（xyzw 顺序），形状 `(len(env_ids), 4)`
- `base_lin_vel_w` (`Tensor`): 世界坐标系中的基础线速度，形状 `(len(env_ids), 3)`
- `base_ang_vel_w` (`Tensor`): 世界坐标系中的基础角速度，形状 `(len(env_ids), 3)`

```{warning}
整个框架中的四元数顺序为 **xyzw**，与 PyTorch/SciPy 约定一致。然而，Genesis 和 IsaacLab 内部使用 wxyz 顺序，需要进行转换。
```

#### update_sensors()

```python
@abstractmethod
def update_sensors(self) -> None
```

更新传感器读数，如深度相机和激光雷达传感器。在配置了传感器的情况下，在 `post_physics_step()` 之后调用。

#### update_terrain_curriculum()

```python
@abstractmethod
def update_terrain_curriculum(self, env_ids: Tensor, move_up: Tensor, 
                               move_down: Tensor) -> None
```

为指定环境更新地形难度课程。

**参数：**
- `env_ids` (`Tensor`): 要更新的环境 ID
- `move_up` (`Tensor`): 布尔张量，指示哪些环境应增加难度
- `move_down` (`Tensor`): 布尔张量，指示哪些环境应降低难度

#### push_robots()

```python
@abstractmethod
def push_robots(self) -> None
```

对机器人底座应用随机速度扰动，用于域随机化。扰动幅度由 `cfg.domain_rand.max_push_vel_xy` 控制。

#### push_links()

```python
@abstractmethod
def push_links(self) -> None
```

对机器人连杆应用随机力，用于域随机化。力的大小由 `cfg.domain_rand.max_push_force` 控制。

#### draw_debug_vis()

```python
@abstractmethod
def draw_debug_vis(self) -> None
```

绘制调试可视化效果，如高度采样点和关键体位置。仅在启用了 `cfg.env.debug_*` 标志时激活。

#### set_viewer_camera()

```python
@abstractmethod
def set_viewer_camera(self, eye: np.ndarray, target: np.ndarray) -> None
```

设置查看器相机的位置和朝向。

**参数：**
- `eye` (`np.ndarray`): 相机位置，形状 `(3,)`
- `target` (`np.ndarray`): 相机注视点，形状 `(3,)`

### 状态属性

Simulator 类提供了许多只读属性用于访问机器人状态：

| 属性 | 形状 | 描述 |
|----------|-------|-------------|
| `base_pos` | `(num_envs, 3)` | 世界坐标系中的基础位置 |
| `base_quat` | `(num_envs, 4)` | 基础朝向（xyzw 四元数） |
| `base_euler` | `(num_envs, 3)` | 基础朝向（欧拉角） |
| `base_lin_vel` | `(num_envs, 3)` | 基础坐标系中的基础线速度 |
| `base_ang_vel` | `(num_envs, 3)` | 基础坐标系中的基础角速度 |
| `dof_pos` | `(num_envs, num_dof)` | 关节位置 |
| `dof_vel` | `(num_envs, num_dof)` | 关节速度 |
| `feet_pos` | `(num_envs, num_feet, 3)` | 世界坐标系中的足端位置 |
| `link_contact_forces` | `(num_envs, num_bodies, 3)` | 所有体上的接触力 |
| `measured_heights` | `(num_envs, num_points)` | 机器人周围的地形高度 |
| `torques` | `(num_envs, num_dof)` | 施加的关节扭矩 |

### 域随机化属性

归一化的域随机化参数可通过前缀为 `dr_` 的属性获取：

| 属性 | 描述 |
|----------|-------------|
| `dr_friction_values` | 摩擦系数乘数 |
| `dr_restitution_values` | 恢复系数乘数 |
| `dr_added_base_mass` | 添加的基础质量（kg） |
| `dr_base_com_bias` | 质心偏移 |
| `dr_joint_armature` | 关节惯量值 |
| `dr_kp_scale` | PD 控制器 P 增益缩放 |
| `dr_kd_scale` | PD 控制器 D 增益缩放 |

## GenesisSimulator

`GenesisSimulator` 提供了与 Genesis 物理引擎的集成。它位于 `legged_gym/simulator/genesis_simulator.py`。

### 资源路径约定

Genesis 需要 **MJCF/XML** 格式的机器人资源：

```python
class MyRobotCfg:
    class asset:
        # 必需：Genesis 的 XML 文件路径
        xml_file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/my_robot/robot.xml"
```

```{important}
使用 Genesis 时 **必须** 提供 `xml_file` 参数。不直接支持 URDF 文件。请使用 MJCF 格式描述机器人。
```

### 主要特性

- **地形类型**：支持平面和高度场地形。三角网格地形尚未验证。
- **深度相机**：通过 Genesis 传感器 API 内置深度相机支持
- **域随机化**：完全支持摩擦、质量、COM、关节属性的随机化
- **四元数约定**：Genesis 内部使用 **wxyz** 顺序；自动转换为 xyzw

### 使用示例

```python
# 导入前必须设置环境变量
import os
os.environ["SIMULATOR"] = "genesis"

from legged_gym.envs import MyRobot, MyRobotCfg

cfg = MyRobotCfg()
simulator = GenesisSimulator(cfg, sim_params, device="cuda:0", headless=True)
```

## IsaacGymSimulator

`IsaacGymSimulator` 提供了 NVIDIA IsaacGym Preview 4 的最快训练性能。它位于 `legged_gym/simulator/isaacgym_simulator.py`。

### 资源路径约定

IsaacGym 使用 **URDF** 格式的机器人资源：

```python
class MyRobotCfg:
    class asset:
        # 必需：URDF 文件路径
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/my_robot/robot.urdf"
```

### 主要特性

- **地形类型**：支持平面、高度场和三角网格地形
- **深度相机**：两种模式：
  - IsaacGym 原生相机传感器（较慢）
  - 基于 Warp 的光线投射（较快，通过 `cfg.sensor.use_warp = True`）
- **Warp 相机**：使用 NVIDIA Warp 的高性能深度感知

### Warp 相机支持

```python
class MyRobotCfg:
    class sensor:
        add_depth = True
        use_warp = True  # 启用基于 Warp 的相机
        depth_camera_config:
            resolution = [87, 58]
            near_plane = 0.1
            far_plane = 3.0
```

### IsaacGym 重置 Bug

```{warning}
在 IsaacGym 中调用 `reset()` 后，必须在读取刚体状态之前调用一次 `simulator.forward()`。否则会导致状态数据过时。

```python
# 正确的模式
env_ids = terminated.nonzero().flatten()
simulator.reset_idx(env_ids)
simulator._gym.simulate(simulator._sim)  # 或等效的 forward 调用
# 现在状态张量已更新
```

这记录在 `g1_deepmimic.py:73` 的 G1 DeepMimic 实现中。
```

## IsaacLabSimulator

`IsaacLabSimulator` 集成了 NVIDIA IsaacSim 和 IsaacLab 框架。它位于 `legged_gym/simulator/isaaclab_simulator.py`。

### 资源路径约定

IsaacLab 使用 **URDF** 格式的机器人资源：

```python
class MyRobotCfg:
    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/my_robot/robot.urdf"
```

### 主要特性

- **地形类型**：支持平面和三角网格地形
- **渲染**：在所有仿真器中视觉质量最佳
- **Articulation API**：使用 IsaacLab 的 Articulation 类进行状态管理

```{warning}
IsaacLabSimulator **未实现**高度场地形。请改用三角网格或平面地形类型。

```python
# 这将引发 NotImplementedError
cfg.terrain.mesh_type = "heightfield"  # 不支持
```
```

### 域随机化 CPU 张量

```{important}
在 IsaacLab 中，域随机化方法需要 **CPU** 张量，而非 GPU：

```python
# 正确：传递 CPU 张量
material_props = self._robot.root_physx_view.get_material_properties()
self._robot.root_physx_view.set_material_properties(
    material_props.to('cpu'), 
    all_indices  # 必须在 CPU 上
)
```

这适用于：
- `set_material_properties()` 用于摩擦/恢复
- `set_masses()` 用于质量随机化
- `set_coms()` 用于 COM 位移
```

### 接触传感器排序

IsaacLab 的接触传感器具有与关节体排序不同的连杆排序。仿真器通过单独的属性处理此问题：

```python
feet_indices = simulator.feet_indices  # Articulation 体索引
feet_contact_indices = simulator.feet_contact_indices  # 接触传感器索引
```

## 资源路径总结

| 仿真器 | 格式 | 配置参数 | 示例 |
|-----------|--------|------------------|---------|
| Genesis | MJCF/XML | `xml_file` | `resources/robots/go2/go2.xml` |
| IsaacGym | URDF | `file` | `resources/robots/go2/urdf/go2.urdf` |
| IsaacLab | URDF | `file` | `resources/robots/go2/urdf/go2.urdf` |

## 常见实现模式

### PD 控制

所有仿真器在 `_compute_torques()` 中实现 PD 控制：

```python
def _compute_torques(self, actions):
    actions_scaled = actions * self._cfg.control.action_scale
    
    # 位置控制 (control_type='P')
    torques = self._kp_scale * self._p_gains * \
        (actions_scaled + self._default_dof_pos - self._dof_pos) - \
        self._kd_scale * self._d_gains * self._dof_vel
    
    return torch.clip(torques, -self._torque_limits, self._torque_limits)
```

### 动作延迟

通过动作队列实现控制延迟随机化：

```python
def _pre_simulator_step(self, actions):
    if self._cfg.domain_rand.randomize_ctrl_delay:
        self._action_queue[:, 1:] = self._action_queue[:, :-1].clone()
        self._action_queue[:, 0] = actions.clone()
        actions = self._action_queue[torch.arange(self._num_envs), 
                                      self._action_delay].clone()
    return actions
```

### 地形高度采样

机器人基础周围的高度采样：

```python
def _update_surrounding_heights(self):
    # 将高度采样点转换到世界坐标系
    points = quat_apply_yaw(self._base_quat.repeat(1, self._num_height_points), 
                            self._height_points) + self._base_pos.unsqueeze(1)
    
    # 从高度场采样
    points = (points / self._cfg.terrain.horizontal_scale).long()
    heights = self._height_samples[points[:, :, 0], points[:, :, 1]]
    
    self._measured_heights = heights * self._cfg.terrain.vertical_scale
```

## 调试可视化

在配置中启用调试标志以可视化内部状态：

```python
class MyRobotCfg:
    class env:
        debug_draw_height_points_around_base = True
        debug_draw_height_points_around_feet = True
        debug_draw_key_body_points = True
        debug_draw_depth_images = True
```

```{note}
调试可视化会显著降低仿真速度。仅在调试期间启用，训练时请勿启用。
```

## 错误处理

常见异常及其原因：

| 错误 | 原因 | 解决方案 |
|-------|-------|----------|
| `NotImplementedError` (Genesis) | 使用了三角网格地形 | 改用高度场 |
| `NotImplementedError` (IsaacLab) | 使用了高度场地形 | 改用三角网格或平面 |
| `ValueError` | 未知的地形网格类型 | 使用 'plane'、'heightfield' 或 'trimesh' |
| `NameError` | 未知的控制器类型 | 使用 'P'、'V' 或 'T' |

## 相关文档

- {doc}`../parameter_reference/legged_robot_config` - 配置参数
- {doc}`../../user_guide/getting_started/installation` - 仿真器安装指南
- {doc}`../known_issues/genesis` - 已知问题和解决方案
