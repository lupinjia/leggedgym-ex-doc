# 向 LeggedGym-Ex 添加新机器人

本指南将带你完成向 LeggedGym-Ex 框架添加新机器人的完整流程。按照以下步骤操作,你就能为任何自定义四足或双足机器人训练运动策略。

```{note}
本指南假设你已经成功安装 LeggedGym-Ex。如果尚未安装,请先参考安装说明。
```

## 概述

添加新机器人涉及六个关键步骤:

1. **创建机器人类** - 继承 `LeggedRobot` 并自定义行为
2. **创建配置类** - 定义机器人参数、奖励和训练设置
3. **定义资源路径** - 为每个模拟器配置 URDF/XML 路径
4. **注册任务** - 添加到任务注册表
5. **测试训练** - 用最小训练运行验证设置
6. **调优参数** - 优化奖励、指令和域随机化

```{tip}
添加新机器人最快的方法是复制一个现有的机器人实现并进行修改。Go2 的实现是四足机器人的良好起点,而 G1 则适合双足机器人。
```

---

## 步骤 1: 创建机器人类

### 1.1 目录结构

在 `legged_gym/envs/` 下为你的机器人创建新目录:

```
legged_gym/envs/
├── myrobot/
│   ├── __init__.py
│   ├── myrobot.py           # 机器人环境类
│   └── myrobot_config.py    # 配置类
```

### 1.2 基本机器人类

创建 `myrobot.py`,其中包含一个继承自 `LeggedRobot` 的类:

```python
# legged_gym/envs/myrobot/myrobot.py
from legged_gym import *
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float

class MyRobot(LeggedRobot):
    """继承 LeggedRobot 的自定义机器人环境。
    
    本类处理:
    - 自由度状态重置
    - 观测值计算
    - 根状态重置
    - 观测噪声缩放
    """
    
    def _reset_dofs(self, env_ids):
        """为指定的环境重置自由度位置和速度。
        
        自由度位置在默认站立姿态附近初始化,并带有小的随机偏移以增强鲁棒性。
        
        Args:
            env_ids: 要重置的环境索引张量
        """
        dof_pos = torch.zeros(
            (len(env_ids), self.num_actions), 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        dof_vel = torch.zeros(
            (len(env_ids), self.num_actions), 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        
        # 在默认位置附近初始化并带有随机化
        # 根据机器人的自由度结构调整索引
        dof_pos[:, :] = self.simulator.default_dof_pos[:] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_actions), self.device)
        
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def compute_observations(self):
        """为所有环境计算观测值。
        
        观测向量包括:
        - 指令 (3): lin_vel_x, lin_vel_y, ang_vel_yaw
        - 投影重力 (3): 世界坐标系中的机身 z 轴
        - 机身角速度 (3): 按 obs_scales.ang_vel 缩放
        - 自由度位置 (num_dofs): 偏离默认值
        - 自由度速度 (num_dofs): 按 obs_scales.dof_vel 缩放
        - 动作 (num_actions): 历史动作
        
        [注意]: 修改观测时必须更新 _get_noise_scale_vec()
        """
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,               # 3
            self.simulator.projected_gravity,                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,    # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) 
                * self.obs_scales.dof_pos,                            # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,         # num_dofs
            self.actions                                              # num_actions
        ), dim=-1)
        
        # 如果启用则添加高度测量
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5 - 
                self.measured_heights, -1, 1.
            ) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        # 如果启用则添加噪声
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        # 为非对称演员-评论家网络计算特权观测
        if self.num_privileged_obs is not None:
            self._compute_privileged_observations()
    
    def _compute_privileged_observations(self):
        """为评论家网络计算特权观测。
        
        包括在现实机器人上不可用的域随机化参数:
        - 摩擦系数
        - 添加的机身质量
        - 质心偏移
        - 外部推动速度
        """
        self.privileged_obs_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * 
                self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.last_actions,
            self.simulator._friction_values,        # 1
            self.simulator._added_base_mass,        # 1
            self.simulator._base_com_bias,          # 3
            self.simulator._rand_push_vels[:, :2],  # 2
        ), dim=-1)
        
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5 - 
                self.measured_heights, -1, 1.
            ) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat(
                (self.privileged_obs_buf, heights), dim=-1
            )
    
    def _get_noise_scale_vec(self):
        """设置观测噪声缩放的污染。
        
        [注意]: 观测结构变化时必须更新!
        这里的索引必须与 compute_observations() 中的观测顺序匹配。
        
        Returns:
            每个观测维度的噪声缩放张量
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # 匹配索引到观测结构
        noise_vec[:3] = 0.  # 指令 (无噪声)
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.  # 动作 (无噪声)
        
        if self.cfg.terrain.measure_heights:
            height_start = 9 + 3 * self.num_actions
            noise_vec[height_start:] = noise_scales.height_measurements * \
                noise_level * self.obs_scales.height_measurements
        
        return noise_vec
    
    def _reset_root_states(self, env_ids):
        """为指定的环境重置机身位置和速度。
        
        Args:
            env_ids: 要重置的环境索引张量
        """
        # 机身位置
        if self.simulator.custom_origins:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
            base_pos[:, :2] += torch_rand_float(
                -0.5, 0.5, (len(env_ids), 2), device=self.device
            )
        else:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
        
        # 机身方向 (四元数)
        base_quat = self.simulator.base_init_quat.reshape(1, -1).repeat(len(env_ids), 1)
        
        # 机身速度 (小的随机初始速度)
        base_lin_vel = torch_rand_float(-0.0, 0.0, (len(env_ids), 3), self.device)
        base_ang_vel = torch_rand_float(-0.0, 0.0, (len(env_ids), 3), self.device)
        
        self.simulator.reset_root_states(
            env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel
        )
```

### 1.3 需要重写的主要方法

| 方法 | 用途 | 何时重写 |
|------|------|----------|
| `_reset_dofs()` | 初始化关节位置 | 需要自定义站立姿态时 |
| `compute_observations()` | 构建观测向量 | 不同传感器套件时 |
| `_get_noise_scale_vec()` | 定义观测噪声 | 观测结构变化时 |
| `_reset_root_states()` | 初始化机身姿态 | 自定义生成行为时 |

```{warning}
如果修改了 `compute_observations()`,则**必须**更新 `_get_noise_scale_vec()` 以匹配新的观测索引。索引不匹配会导致静默训练失败!
```

---

## 步骤 2: 创建配置类

### 2.1 基本配置

使用嵌套配置类创建 `myrobot_config.py`:

```python
# legged_gym/envs/myrobot/myrobot_config.py
from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import get_simulator_suffix

class MyRobotCfg(LeggedRobotCfg):
    """MyRobot 运动训练的配置。
    
    继承自 LeggedRobotCfg 并覆盖特定参数。
    """
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096              # 并行环境数量
        num_observations = 45        # 观测维度 (根据机器人调整)
        num_privileged_obs = None    # 设置为整数以进行非对称训练
        num_actions = 12             # 驱动关节数量
        episode_length_s = 20        # 回合持续时间 (秒)
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]        # 初始生成位置 [x, y, z] (米)
        default_joint_angles = {     # 默认站立姿态
            'FL_hip_joint': 0.0,
            'FL_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,
            # ... 添加所有关节
        }
    
    class control(LeggedRobotCfg.control):
        control_type = 'P'           # 位置控制
        stiffness = {'joint': 20.0}  # PD 刚度 [N*m/rad]
        damping = {'joint': 0.5}     # PD 阻尼 [N*m*s/rad]
        action_scale = 0.25          # 动作到角度映射
        dt = 0.02                    # 控制频率: 50 Hz
        decimation = 4               # 每次控制的模拟步数
    
    class asset(LeggedRobotCfg.asset):
        name = "myrobot"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot.xml'
        foot_name = "foot"           # 标识脚部连杆的子串
        penalize_contacts_on = ["thigh", "calf"]  # 惩罚这些接触
        terminate_after_contacts_on = ["base"]    # 在这些接触上终止
        base_link_name = "base"
        dof_names = [                # 按动作顺序的关节名称
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        dof_armature = [0.01] * 12   # 关节惯量以保证稳定性
    
    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.25        # 跟踪奖励带宽
        base_height_target = 0.36    # 目标站立高度 [m]
        
        class scales(LeggedRobotCfg.rewards.scales):
            # 惩罚奖励 (负值)
            dof_pos_limits = -1.0
            collision = -1.0
            lin_vel_z = -0.5
            base_height = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -2.e-4
            
            # 正向奖励
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            feet_air_time = 1.0
            foot_clearance = 0.5
    
    class commands(LeggedRobotCfg.commands):
        curriculum = True            # 随时间增加指令难度
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0       # 指令变更间隔 [s]
        heading_command = True       # 使用航向代替角速度
        
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]  # [最小, 最大] m/s
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]    # rad/s
            heading = [-3.14, 3.14]  # rad
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_com_displacement = True


class MyRobotCfgPPO(LeggedRobotCfgPPO):
    """MyRobot 的 PPO 训练配置。"""
    
    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'simple_rl' + get_simulator_suffix()
        experiment_name = 'myrobot'
        save_interval = 200
        max_iterations = 1500
```

### 2.2 配置层次结构

配置系统使用嵌套类继承:

```
LeggedRobotCfg
├── env          # 环境设置
├── terrain      # 地形配置
├── init_state   # 初始机器人状态
├── control      # 控制参数
├── asset        # 机器人资源路径
├── rewards      # 奖励函数设置
├── commands     # 指令速度范围
├── domain_rand  # 域随机化
├── normalization # 观测/动作缩放
├── noise        # 观测噪声
└── sim          # 模拟参数
```

### 2.3 关键配置参数

#### 环境设置

| 参数 | 描述 | 典型值 |
|------|------|--------|
| `num_envs` | 并行环境 | 4096 (取决于 GPU) |
| `num_observations` | 观测维度 | 机器人特定 |
| `num_actions` | 驱动关节 | 12 (四足), 12+ (双足) |
| `episode_length_s` | 回合持续时间 | 10-20 秒 |

#### 控制设置

| 参数 | 描述 | 典型值 |
|------|------|--------|
| `control_type` | 控制模式 | 'P' (位置), 'V' (速度), 'T' (力矩) |
| `stiffness` | PD 刚度 | 20-100 N*m/rad |
| `damping` | PD 阻尼 | 0.5-2.0 N*m*s/rad |
| `action_scale` | 动作幅度 | 0.25-0.5 |
| `dt` | 控制周期 | 0.02s (50Hz) |
| `decimation` | 每次控制的模拟步数 | 4 (200Hz 模拟) |

#### 奖励缩放策略

```{tip}
从平衡的奖励缩放开始,并根据训练曲线进行调整:
1. 指令跟踪奖励 (正向): 0.5-2.0
2. 平滑惩罚 (负向): -0.001 到 -0.1
3. 安全惩罚 (负向): -1.0 到 -10.0
```

---

## 步骤 3: 为每个模拟器定义资源路径

### 3.1 多模拟器支持

LeggedGym-Ex 支持三个模拟器,每个有不同的资源要求:

| 模拟器 | 资源格式 | 配置键 |
|--------|----------|--------|
| IsaacGym | URDF | `asset.file` |
| IsaacLab | URDF | `asset.file` |
| Genesis | XML | `asset.xml_file` |

### 3.2 资源的目录结构

将你的机器人资源放在 resources 目录中:

```
resources/robots/
└── myrobot/
    ├── myrobot.urdf        # IsaacGym/IsaacLab 的 URDF
    ├── myrobot.xml         # Genesis 的 XML
    └── meshes/             # 视觉和碰撞网格
        ├── base.stl
        ├── thigh.stl
        └── ...
```

### 3.3 配置示例

```python
class asset(LeggedRobotCfg.asset):
    name = "myrobot"
    
    # IsaacGym 和 IsaacLab 的 URDF 路径
    file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot.urdf'
    
    # Genesis 的 XML 路径
    xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot.xml'
    
    # Genesis 特定设置
    links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    dof_vel_limits = [30.1, 30.1, 15.7] * 4  # 每个关节的 rad/s
```

### 3.4 URDF 要求

```{important}
你的 URDF 必须满足以下要求:
1. 所有关节必须是旋转关节 (腿部没有棱柱关节)
2. 必须定义关节限制
3. 惯性参数必须真实
4. 网格文件必须使用相对路径引用
```

URDF 结构示例:

```xml
<robot name="myrobot">
  <link name="base">
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
    </inertial>
    <visual>
      <geometry><mesh filename="meshes/base.stl"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="meshes/base.stl"/></geometry>
    </collision>
  </link>
  
  <joint name="FL_hip_joint" type="revolute">
    <parent link="base"/>
    <child link="FL_hip"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="30"/>
  </joint>
  <!-- ... 更多连杆和关节 ... -->
</robot>
```

### 3.5 Genesis XML 要求

Genesis 需要 MJCF 风格的 XML 文件:

```xml
<mujoco model="myrobot">
  <compiler angle="radian"/>
  
  <asset>
    <mesh name="base_mesh" file="meshes/base.stl"/>
    <!-- ... 更多网格 ... -->
  </asset>
  
  <worldbody>
    <body name="base" pos="0 0 0">
      <inertial mass="10.0" diaginertia="0.1 0.1 0.05"/>
      <geom type="mesh" mesh="base_mesh"/>
      
      <body name="FL_hip" pos="0.2 0.1 0">
        <joint name="FL_hip_joint" type="hinge" axis="0 0 1" 
               range="-0.5 0.5"/>
        <!-- ... 更多嵌套体 ... -->
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="FL_hip_motor" joint="FL_hip_joint" gear="1"/>
    <!-- ... 更多执行器 ... -->
  </actuator>
</mujoco>
```

---

## 步骤 4: 在 envs/__init__.py 中注册

### 4.1 导入并注册

将机器人添加到 `legged_gym/envs/__init__.py` 中的任务注册表:

```python
# legged_gym/envs/__init__.py

# ... 现有导入 ...

# MyRobot - 添加这些行
from legged_gym.envs.myrobot.myrobot import MyRobot
from legged_gym.envs.myrobot.myrobot_config import MyRobotCfg, MyRobotCfgPPO

# ... 在注册部分 ...
task_registry.register("myrobot", MyRobot, MyRobotCfg(), MyRobotCfgPPO())
```

### 4.2 注册参数

```python
task_registry.register(
    task_name="myrobot",      # 唯一任务标识符
    env_class=MyRobot,        # 环境类
    env_cfg=MyRobotCfg(),     # 环境配置 (实例化)
    train_cfg=MyRobotCfgPPO() # 训练配置 (实例化)
)
```

### 4.3 命名约定

遵循任务名称的以下约定:

| 模式 | 示例 | 用例 |
|------|------|------|
| `{robot}` | `go2` | 基础运动 |
| `{robot}_ts` | `go2_ts` | 教师-学生 |
| `{robot}_ee` | `go2_ee` | 显式估计器 |
| `{robot}_wtw` | `go2_wtw` | Walk These Ways |
| `{robot}_deepmimic` | `g1_deepmimic` | DeepMimic |
| `{robot}_amp` | `k1_amp` | 对抗运动先验 |

---

## 步骤 5: 用最小训练运行测试

### 5.1 快速验证

在完整训练之前验证你的设置:

```bash
# 设置模拟器 (选择一个)
export SIMULATOR=genesis
# export SIMULATOR=isaacgym
# export SIMULATOR=isaaclab

# 运行短期训练测试
python -m legged_gym.scripts.train --task=myrobot --headless --max_iterations=10
```

### 5.2 预期输出

成功的测试应显示:

```
[INFO] Creating environment: myrobot
[INFO] num_envs: 4096
[INFO] num_observations: 45
[INFO] num_actions: 12
[INFO] Starting training...
iteration 0: mean_reward=0.123, mean_episode_length=45.2
iteration 1: mean_reward=0.156, mean_episode_length=52.1
...
```

### 5.3 常见验证检查清单

- [ ] 环境无错误初始化
- [ ] 机器人在正确高度生成
- [ ] 观测具有预期形状
- [ ] 奖励正在计算
- [ ] 机器人不会立即倒下
- [ ] 无 CUDA 内存错误

### 5.4 调试模式

用于可视化调试:

```bash
# 不使用 headless 运行以查看机器人
python -m legged_gym.scripts.train --task=myrobot --num_envs=16
```

在配置中启用调试可视化:

```python
class env(LeggedRobotCfg.env):
    debug = True
    debug_draw_height_points_around_base = True
```

---

## 步骤 6: 调整奖励和指令

### 6.1 奖励调整过程

1. **从简单开始**: 仅从指令跟踪奖励开始
2. **添加平滑性**: 惩罚急促动作
3. **添加安全性**: 惩罚碰撞和限制
4. **迭代**: 根据观察到的行为调整缩放

### 6.2 常见奖励函数

```python
# 在你的配置中
class rewards(LeggedRobotCfg.rewards):
    class scales(LeggedRobotCfg.rewards.scales):
        # 指令跟踪 (鼓励目标达成)
        tracking_lin_vel = 1.0     # 跟踪 x/y 速度指令
        tracking_ang_vel = 0.5     # 跟踪偏航速度指令
        
        # 平滑性 (惩罚不稳定行为)
        lin_vel_z = -2.0           # 惩罚垂直速度
        ang_vel_xy = -0.05         # 惩罚横滚/俯仰率
        dof_acc = -2.5e-7          # 惩罚关节加速度
        action_rate = -0.01        # 惩罚动作变化
        
        # 能量效率
        torques = -1e-4            # 惩罚大力矩
        dof_power = -2e-4          # 惩罚功耗
        
        # 安全性
        collision = -1.0           # 惩罚机身碰撞
        dof_pos_limits = -10.0     # 惩罚关节限制违规
        termination = -0.0         # 终止惩罚
        
        # 步态质量
        feet_air_time = 1.0        # 奖励正确摆动阶段
        foot_clearance = 0.5       # 奖励摆动期间抬脚
```

### 6.3 指令调整

```python
class commands(LeggedRobotCfg.commands):
    curriculum = True              # 启用渐进式难度
    max_curriculum = 1.5           # 最大速度乘数
    
    class ranges(LeggedRobotCfg.commands.ranges):
        # 从保守开始,课程会扩展
        lin_vel_x = [-0.3, 0.3]    # 前/后
        lin_vel_y = [-0.3, 0.3]    # 横向
        ang_vel_yaw = [-0.5, 0.5]  # 转弯
```

### 6.4 域随机化调整

```python
class domain_rand(LeggedRobotCfg.domain_rand):
    # 物理随机化
    randomize_friction = True
    friction_range = [0.5, 1.5]    # 更宽以增强鲁棒性
    
    randomize_base_mass = True
    added_mass_range = [-2.0, 2.0] # kg 变化
    
    # 外部扰动
    push_robots = True
    push_interval_s = 10
    max_push_vel_xy = 0.5          # m/s 推动速度
    
    # 建模误差
    randomize_com_displacement = True
    com_pos_x_range = [-0.02, 0.02]
    com_pos_y_range = [-0.02, 0.02]
    com_pos_z_range = [-0.01, 0.01]
```

### 6.5 监控训练进度

使用 tensorboard 或 wandb:

```bash
# 查看训练曲线
tensorboard --logdir logs/myrobot/

# 或在配置中启用 wandb
class runner(LeggedRobotCfgPPO.runner):
    sync_wandb = True
```

要监控的关键指标:

| 指标 | 目标 | 差时操作 |
|------|------|----------|
| `rew_tracking_lin_vel` | > 0.8 | 增加缩放或减小速度范围 |
| `rew_tracking_ang_vel` | > 0.7 | 增加缩放或减小偏航范围 |
| `episode_length` | 接近最大值 | 检查终止条件 |
| `terrain_level` | 增加 | 训练正在进展 |

---

## 工作示例: 克隆 Go2 → MyRobot

本节提供一个完整的工作示例,展示如何通过克隆和修改 Go2 实现来创建新机器人。

### 逐步克隆过程

```bash
# 1. 创建目录结构
mkdir -p legged_gym/envs/myrobot

# 2. 复制 Go2 文件作为模板
cp legged_gym/envs/go2/go2.py legged_gym/envs/myrobot/myrobot.py
cp legged_gym/envs/go2/go2_config.py legged_gym/envs/myrobot/myrobot_config.py

# 3. 创建 __init__.py
touch legged_gym/envs/myrobot/__init__.py
```

### 修改后的 myrobot.py

```python
# legged_gym/envs/myrobot/myrobot.py
from legged_gym import *
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float

class MyRobot(LeggedRobot):
    """基于 Go2 模板的自定义四足机器人。"""
    
    def _reset_dofs(self, env_ids):
        """使用 MyRobot 特定初始化重置自由度。"""
        dof_pos = torch.zeros((len(env_ids), self.num_actions), 
                              dtype=torch.float, device=self.device)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), 
                              dtype=torch.float, device=self.device)
        
        # 为 MyRobot 的关节结构定制
        # 假设类似 FR, FL, RR, RL 的顺序
        hip_indices = [0, 3, 6, 9]
        thigh_indices = [1, 4, 7, 10]
        calf_indices = [2, 5, 8, 11]
        
        # 髋部: 0 附近小随机化
        dof_pos[:, hip_indices] = self.simulator.default_dof_pos[:, hip_indices] + \
            torch_rand_float(-0.1, 0.1, (len(env_ids), 4), self.device)
        
        # 大腿: 较大随机化
        dof_pos[:, thigh_indices] = self.simulator.default_dof_pos[:, thigh_indices] + \
            torch_rand_float(-0.3, 0.3, (len(env_ids), 4), self.device)
        
        # 小腿: 适度随机化
        dof_pos[:, calf_indices] = self.simulator.default_dof_pos[:, calf_indices] + \
            torch_rand_float(-0.3, 0.3, (len(env_ids), 4), self.device)
        
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    # 使用 LeggedRobot 的默认 compute_observations
    # 或根据需要自定义
```

### 修改后的 myrobot_config.py

```python
# legged_gym/envs/myrobot/myrobot_config.py
from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import get_simulator_suffix

class MyRobotCfg(LeggedRobotCfg):
    """MyRobot 的配置。"""
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45        # 根据传感器设置调整
        num_privileged_obs = None
        num_actions = 12             # 根据机器人的自由度调整
        episode_length_s = 20
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]       # 为你的机器人调整生成高度
        default_joint_angles = {
            # 为你的机器人关节名称定制
            'FR_hip_joint': 0.0,
            'FR_thigh_joint': 0.7,
            'FR_calf_joint': -1.4,
            'FL_hip_joint': 0.0,
            'FL_thigh_joint': 0.7,
            'FL_calf_joint': -1.4,
            'RR_hip_joint': 0.0,
            'RR_thigh_joint': 0.7,
            'RR_calf_joint': -1.4,
            'RL_hip_joint': 0.0,
            'RL_thigh_joint': 0.7,
            'RL_calf_joint': -1.4,
        }
    
    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 25.0}   # 为你的机器人调优
        damping = {'joint': 0.5}
        action_scale = 0.25
        dt = 0.02
        decimation = 4
    
    class asset(LeggedRobotCfg.asset):
        name = "myrobot"
        # 指向你的 URDF/XML 文件
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot.xml'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        base_link_name = "base"
        dof_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        dof_armature = [0.01] * 12
    
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32    # 为你的机器人调整
        only_positive_rewards = True
        
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            base_height = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -2.e-4
            feet_air_time = 1.0
            foot_clearance = 0.5
            dof_pos_limits = -1.0
            collision = -1.0
    
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = True
        
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = True


class MyRobotCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'simple_rl' + get_simulator_suffix()
        experiment_name = 'myrobot'
        save_interval = 200
        max_iterations = 1500
```

### 注册并测试

```python
# 添加到 legged_gym/envs/__init__.py

from legged_gym.envs.myrobot.myrobot import MyRobot
from legged_gym.envs.myrobot.myrobot_config import MyRobotCfg, MyRobotCfgPPO

task_registry.register("myrobot", MyRobot, MyRobotCfg(), MyRobotCfgPPO())
```

```bash
# 测试新机器人
export SIMULATOR=genesis
python -m legged_gym.scripts.train --task=myrobot --headless --max_iterations=100
```

---

## 故障排除

### 常见问题和解决方案

#### 生成后机器人立即倒下

**症状**: 机器人立即倒塌,回合快速终止。

**解决方案**:
1. 检查 `init_state.pos[2]` - 生成高度可能太低
2. 验证 `default_joint_angles` 是否匹配稳定姿态
3. 增加 `dof_armature` 以保证数值稳定性
4. 检查 URDF 惯性值是否真实

```python
# 调试生成高度
class init_state(LeggedRobotCfg.init_state):
    pos = [0.0, 0.0, 0.5]  # 尝试更高的生成位置
```

#### 观测形状不匹配

**症状**: 错误如 "expected shape (4096, 45) but got (4096, 48)"。

**解决方案**:
1. 手动计算观测维度
2. 确保 `num_observations` 匹配 `obs_buf.shape[1]`
3. 检查 `measure_heights` 是否添加额外维度

```python
# 计算观测:
# commands (3) + gravity (3) + ang_vel (3) + dof_pos (12) + dof_vel (12) + actions (12) = 45
```

#### CUDA 内存不足

**症状**: RuntimeError: CUDA out of memory.

**解决方案**:
1. 减少 `num_envs`
2. 减少 `terrain.num_rows` 和 `num_cols`
3. 如果不需要则禁用 `measure_heights`
4. 使用更小的观测历史

```python
class env(LeggedRobotCfg.env):
    num_envs = 2048  # 从 4096 减少
```

#### 训练没有进展

**症状**: 奖励在低值处平稳,机器人没有学会走路。

**解决方案**:
1. 增加 `tracking_lin_vel` 奖励缩放
2. 减小指令速度范围
3. 检查终止条件是否太严格
4. 验证域随机化是否太激进

```python
class commands(LeggedRobotCfg.commands):
    class ranges(LeggedRobotCfg.commands.ranges):
        lin_vel_x = [-0.2, 0.2]  # 从更小的指令开始

class rewards(LeggedRobotCfg.rewards):
    class scales(LeggedRobotCfg.rewards.scales):
        tracking_lin_vel = 2.0   # 增加跟踪奖励
```

#### 关节限制违规

**症状**: 机器人在关节限制处抽搐,动作不自然。

**解决方案**:
1. 检查 URDF 关节限制是否匹配真实机器人
2. 调整 `soft_dof_pos_limit`
3. 增加 `dof_pos_limits` 惩罚

```python
class rewards(LeggedRobotCfg.rewards):
    soft_dof_pos_limit = 0.9  # 在限制的 90% 处惩罚
    class scales(LeggedRobotCfg.rewards.scales):
        dof_pos_limits = -10.0  # 增加惩罚
```

#### Genesis 特定问题

**症状**: XML 加载错误,缺少链接。

**解决方案**:
1. 确保 `links_to_keep` 包含所有脚部链接
2. 检查 XML 使用正确的四元数约定 (wxyz)
3. 验证网格路径是相对于 XML 位置的

```python
class asset(LeggedRobotCfg.asset):
    links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    dof_vel_limits = [30.0] * 12  # Genesis 必需
```

#### IsaacGym 特定问题

**症状**: 重置错误,重置后状态不正确。

**解决方案**:
1. 重置后调用 `simulator.forward()`
2. 检查 `collapse_fixed_joints` 设置
3. 验证 `default_dof_drive_mode` 是否正确

```python
# 在 _reset_dofs 或类似函数中
self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
self.simulator.forward()  # IsaacGym 必需
```

#### IsaacLab 特定问题

**症状**: 域随机化错误,设备不匹配。

**解决方案**:
1. 确保 DR 张量在 IsaacLab 上使用 CPU
2. 使用正确的 `collapse_fixed_joints` 设置
3. 检查 `measure_heights` 兼容性

```python
# IsaacLab 需要 CPU 张量进行 DR
# 这已内部处理,但请注意
```

### 调试命令

```bash
# 列出所有已注册任务
python -c "from legged_gym.envs import task_registry; print(task_registry.task_classes.keys())"

# 检查配置值
python -c "from legged_gym.envs.myrobot.myrobot_config import MyRobotCfg; cfg = MyRobotCfg(); print(f'obs: {cfg.env.num_observations}, actions: {cfg.env.num_actions}')"

# 测试单个环境
python -m legged_gym.scripts.train --task=myrobot --num_envs=1 --headless
```

---

## 总结检查清单

在提交新机器人实现之前:

- [ ] 机器人类继承自 `LeggedRobot`
- [ ] 所有必需方法已实现或使用默认方法
- [ ] 配置类继承自 `LeggedRobotCfg`
- [ ] 所有模拟器的资源路径正确
- [ ] 关节名称与 URDF/XML 完全匹配
- [ ] 默认姿态稳定
- [ ] 任务在 `__init__.py` 中注册
- [ ] 基础训练无错误运行
- [ ] 机器人学会走路 (即使走得不好)
- [ ] 默认设置下无 CUDA 内存问题

---

## 后续步骤

成功添加机器人后:

1. **地形训练**: 添加崎岖地形以增强鲁棒性
   ```python
   class terrain(LeggedRobotCfg.terrain):
       mesh_type = "trimesh"
       curriculum = True
       measure_heights = True
   ```

2. **仿真到现实**: 准备部署
   - 从观测中移除 `base_lin_vel`
   - 添加观测噪声
   - 用域随机化训练

3. **高级方法**: 实现专门算法
   - 用于特权信息的教师-学生
   - 用于速度估计的显式估计器
   - 用于步态控制的 Walk These Ways

---

## 参考

- {doc}`../parameter_reference/legged_robot_config` - 完整参数参考
- {doc}`../api_reference/legged_robot` - API 文档
- [IsaacGym 文档](https://developer.nvidia.com/isaac-gym)
- [Genesis 文档](https://genesis-world.readthedocs.io/)
- [IsaacLab 文档](https://isaac-sim.github.io/IsaacLab/)
