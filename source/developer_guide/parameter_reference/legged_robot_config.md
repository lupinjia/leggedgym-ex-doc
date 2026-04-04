# `legged_robot_config.py`

本文档提供了 `legged_robot_config.py` 中定义的所有配置参数（configuration parameters）的详细说明。

---

## LeggedRobotCfg

### `env` 环境配置（Environment Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_envs` | int | 4096 | 并行环境（parallel environments）数量 |
| `num_observations` | int | 48 | 观测向量（observation vector）的维度 |
| `num_privileged_obs` | int | None | 特权观测（privileged observations）数量。如果不为 None，step() 返回 priviledge_obs_buf（用于非对称训练的 Critic 观测），否则返回 None |
| `num_actions` | int | 12 | 动作向量（action vector）的维度 |
| `send_timeouts` | bool | True | 是否向算法发送超时（timeout）信息 |
| `episode_length_s` | float | 20 | 回合（episode）长度，单位为秒 |
| `env_spacing` | float | 2.0 | 场景中环境之间的间距，仅适用于平面地形 |
| `fail_to_terminal_time_s` | float | 0.1 | 失败状态（fail state）导致环境重置前的等待时间（秒） |
| `debug` | bool | False | 是否在仿真器中启用调试绘制（debug drawings） |
| `debug_draw_height_points_around_base` | bool | False | 是否获取基座（base）周围的高度测量点 |
| `debug_draw_height_points_around_feet` | bool | False | 是否获取足部（feet）周围的高度测量点（每个足部周围9个点） |
| `debug_draw_terrain_height_points` | bool | False | 是否绘制地形的所有高度点 |
| `debug_draw_key_body_points` | bool | False | 是否为模仿任务（mimic tasks）绘制关键刚体点（key body points） |
| `max_projected_gravity` | float | -0.1 | Z 轴方向允许的最大投影重力（projected gravity） |

---

### `terrain` 地形配置（Terrain Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mesh_type` | str | 'plane' | 地形网格类型：'plane'（平面）、'heightfield'（高度场）、'trimesh'（三角网格） |
| `plane_length` | float | 200.0 | 平面尺寸 [m]，默认为 200x200x10 |
| `horizontal_scale` | float | 0.1 | X 和 Y 方向高度采样间距 [m] |
| `vertical_scale` | float | 0.005 | Z 方向高度采样间距 [m] |
| `border_size` | float | 5 | 地形周围边界长度 [m] |
| `border_height` | float | 1.0 | 地形周围边界高度 [m] |
| `curriculum` | bool | False | 是否使用地形课程学习（terrain curriculum），从简单地形开始逐渐增加难度 |
| `static_friction` | float | 1.0 | 地形静摩擦系数（coefficient of static friction） |
| `dynamic_friction` | float | 1.0 | 地形动摩擦系数（coefficient of dynamic friction） |
| `restitution` | float | 0. | 地形恢复系数（coefficient of restitution） |
| `obtain_terrain_info_around_feet` | bool | False | 是否获取足部周围的地形高度信息（默认每个足部周围9个点） |
| `measure_heights` | bool | False | 是否获取高度测量 |
| `measured_points_x` | list | [-0.8, ..., 0.8] | 基座周围高度采样点的 X 位置（相对于机器人基座），1m x 1.6m 矩形区域 |
| `measured_points_y` | list | [-0.5, ..., 0.5] | 基座周围高度采样点的 Y 位置（相对于机器人基座） |
| `selected` | bool | False | 是否选择唯一的地形类型并传递所有参数 |
| `terrain_kwargs` | dict | None | 选定地形的参数字典 |
| `max_init_terrain_level` | int | 1 | 初始课程级别（curriculum level） |
| `terrain_length` | float | 6.0 | 每个子地形（subterrain）的长度 [m]（X 方向） |
| `terrain_width` | float | 6.0 | 每个子地形的宽度 [m]（Y 方向） |
| `platform_size` | float | 3.0 | 每个子地形中心平整平台（platform）的尺寸 [m] |
| `num_rows` | int | 4 | 地形行数（级别），X 方向 |
| `num_cols` | int | 4 | 地形列数（类型），Y 方向 |
| `terrain_proportions` | list | [0.1, 0.1, 0.35, 0.25, 0.2] | 地形类型比例：[平滑斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散地形] |
| `slope_treshold` | float | 0.75 | 仅适用于三角网格：超过此阈值的斜坡将被修正为垂直表面 |

---

### `init_state` 初始状态配置（Initial State Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pos` | list | [0.0, 0.0, 1.] | 初始位置 x, y, z [m] |
| `rot` | list | [0.0, 0.0, 0.0, 1.0] | 初始朝向四元数（quaternion）x, y, z, w。注意：Gym 格式为 xyzw，Genesis 格式为 wxyz |
| `lin_vel` | list | [0.0, 0.0, 0.0] | 初始线速度 x, y, z [m/s] |
| `ang_vel` | list | [0.0, 0.0, 0.0] | 初始角速度 x, y, z [rad/s] |
| `roll_random_scale` | float | 0.0 | 横滚角（roll angle）随机化范围 |
| `pitch_random_scale` | float | 0.0 | 俯仰角（pitch angle）随机化范围 |
| `yaw_random_scale` | float | 0.0 | 偏航角（yaw angle）随机化范围 |
| `default_joint_angles` | dict | {"joint_a": 0., "joint_b": 0.} | action = 0.0 时的目标关节角度 |

---

### `control` 控制配置（Control Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `control_type` | str | 'P' | 控制类型：'P'（位置）、'V'（速度）、'T'（力矩） |
| `stiffness` | dict | {'joint_a': 10.0, 'joint_b': 15.} | PD 控制刚度（stiffness）[N*m/rad] |
| `damping` | dict | {'joint_a': 1.0, 'joint_b': 1.5} | PD 控制阻尼（damping）[N*m*s/rad] |
| `action_scale` | float | 0.5 | 动作缩放因子，目标角度 = actionScale * action + defaultAngle |
| `dt` | float | 0.02 | 控制频率，默认为 50Hz |
| `decimation` | int | 4 | 控制动作更新与策略更新的比率 |

---

### `asset` 资源配置（Asset Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | None | 资源名称 |
| `file` | str | "" | URDF 文件路径 |
| `foot_name` | str | "" | 足部刚体名称，包含此子字符串的刚体将被视为足部 |
| `key_bodies` | list | [] | 模仿任务中需要跟踪的重要刚体列表 |
| `penalize_contacts_on` | list | [] | 对包含这些子字符串的连杆的接触进行惩罚 |
| `terminate_after_contacts_on` | list | [] | 当包含这些子字符串的连杆发生接触后终止回合 |
| `fix_base_link` | bool | False | 是否将基座连杆固定到世界坐标系 |
| `obtain_link_contact_states` | bool | False | 是否获取特定连杆的接触状态，可用于特权策略（privileged policy） |
| `contact_state_link_names` | list | ["thigh", "calf", "foot"] | 用于获取接触状态的连杆名称 |
| `base_link_name` | str | "" | 基座连杆的完整名称 |
| `self_collisions` | int | 0 | 自碰撞（self-collision）设置：1 禁用，0 启用（位掩码过滤器） |
| `dof_names` | list | ["joint_a", "joint_b"] | 动作和观测中的自由度（DOF）序列 |
| `dof_armature` | list | [0.0] | 每个自由度的惯量（armature） |
| `links_to_keep` | list | [] | 仅限 Genesis：不因固定关节而合并的连杆 |
| `dof_vel_limits` | list | [] | 自由度速度限制 [rad/s]，从 URDF 获取 |
| `disable_gravity` | bool | False | 仅限 IsaacGym/IsaacLab：是否禁用重力 |
| `collapse_fixed_joints` | bool | True | 仅限 IsaacGym/IsaacLab：是否合并由固定关节连接的刚体 |
| `default_dof_drive_mode` | int | 3 | 仅限 IsaacGym/IsaacLab：自由度驱动模式（0 无，1 位置目标，2 速度目标，3 力矩） |
| `replace_cylinder_with_capsule` | bool | False | 仅限 IsaacGym/IsaacLab：将碰撞圆柱体替换为胶囊体，可提高仿真速度和稳定性 |
| `flip_visual_attachments` | bool | False | 仅限 IsaacGym/IsaacLab：是否将某些 .obj 网格从 y-up 翻转到 z-up |
| `density` | float | 0.001 | 密度 |
| `angular_damping` | float | 0. | 角阻尼 |
| `linear_damping` | float | 0. | 线阻尼 |
| `max_angular_velocity` | float | 1000. | 最大角速度 |
| `max_linear_velocity` | float | 1000. | 最大线速度 |
| `armature` | float | 0. | 惯量 |
| `thickness` | float | 0.01 | 厚度 |

---

### `rewards` 奖励配置（Reward Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `only_positive_rewards` | bool | True | 是否仅使用正奖励 |
| `tracking_sigma` | float | 0.25 | 跟踪奖励（tracking reward）计算参数：reward = exp(-error^2/sigma) |
| `soft_dof_pos_limit` | float | 1. | 关节位置软限制（soft limit）（URDF 限制的百分比），超过此限制的值将被惩罚 |
| `soft_dof_vel_limit` | float | 1. | 关节速度软限制 |
| `soft_torque_limit` | float | 1. | 力矩软限制 |
| `base_height_target` | float | 1. | 目标基座高度 |
| `foot_clearance_target` | float | 0.04 | 期望的足部离地间隙（clearance）[m] |
| `foot_height_offset` | float | 0.0 | 足部坐标原点距地面高度 [m] |
| `foot_clearance_tracking_sigma` | float | 0.01 | 足部离地间隙跟踪的 sigma 值 |

#### `scales` 奖励权重（Reward Weights）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `termination` | float | -0.0 | 终止惩罚（termination penalty）权重 |
| `tracking_lin_vel` | float | 0 | 线速度跟踪奖励权重 |
| `tracking_ang_vel` | float | 0 | 角速度跟踪奖励权重 |
| `lin_vel_z` | float | 0 | Z 方向线速度惩罚权重 |
| `ang_vel_xy` | float | 0 | XY 平面角速度惩罚权重 |
| `orientation` | float | -0. | 朝向惩罚权重 |
| `torques` | float | 0 | 力矩惩罚权重 |
| `dof_vel` | float | -0. | 关节速度惩罚权重 |
| `dof_acc` | float | 0 | 关节加速度惩罚权重 |
| `base_height` | float | -0. | 基座高度惩罚权重 |
| `feet_air_time` | float | 0 | 足部悬空时间（air time）奖励权重 |
| `collision` | float | 0 | 碰撞惩罚权重 |
| `feet_stumble` | float | -0.0 | 足部绊倒惩罚权重 |
| `action_rate` | float | 0 | 动作变化率惩罚权重 |
| `dof_pos_stand_still` | float | -0. | 静止站立时的关节位置惩罚权重 |

---

### `commands` 指令配置（Command Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `curriculum` | bool | False | 是否使用指令课程学习（command curriculum） |
| `max_curriculum` | float | 1. | 最大课程级别 |
| `num_commands` | int | 4 | 指令数量，默认：lin_vel_x, lin_vel_y, ang_vel_yaw, heading |
| `resampling_time` | float | 10. | 指令重采样（resampling）时间间隔 [s] |
| `heading_command` | bool | True | 如果为 True：从航向误差（heading error）计算角速度指令 |
| `curriculum_threshold` | float | 0.8 | 课程学习阈值，当跟踪奖励超过此阈值时增加指令范围 |
| `zero_cmd_prob` | float | 0.4 | 重采样期间采样零指令的概率，鼓励机器人学习静止站立行为 |

#### `ranges` 指令范围（Command Ranges）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lin_vel_x` | list | [-1.0, 1.0] | X 方向线速度范围 [m/s] |
| `lin_vel_y` | list | [-1.0, 1.0] | Y 方向线速度范围 [m/s] |
| `ang_vel_yaw` | list | [-1, 1] | 偏航角速度范围 [rad/s] |
| `heading` | list | [-3.14, 3.14] | 航向角范围 [rad] |

---

### `domain_rand` 域随机化配置（Domain Randomization Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `randomize_friction` | bool | True | 是否随机化刚体摩擦系数 |
| `friction_range` | list | [0.5, 1.25] | 摩擦系数随机化范围 |
| `randomize_base_mass` | bool | True | 是否随机化基座质量 |
| `added_mass_range` | list | [-1., 1.] | 附加质量范围 [kg] |
| `push_robots` | bool | True | 是否对基座施加随机速度扰动 |
| `push_interval_s` | float | 15 | 推动机器人的时间间隔 [s] |
| `max_push_vel_xy` | float | 1. | XY 平面最大推动速度 [m/s] |
| `randomize_com_displacement` | bool | True | 是否随机化质心（center of mass, CoM）位置以模拟建模误差 |
| `com_pos_x_range` | list | [-0.01, 0.01] | 质心 X 位置随机化范围 [m] |
| `com_pos_y_range` | list | [-0.01, 0.01] | 质心 Y 位置随机化范围 [m] |
| `com_pos_z_range` | list | [-0.01, 0.01] | 质心 Z 位置随机化范围 [m] |
| `randomize_ctrl_delay` | bool | False | 是否对动作应用随机延迟以模拟控制环路延迟（control loop latency） |
| `ctrl_delay_step_range` | list | [0, 1] | 控制延迟步数范围 |
| `randomize_pd_gain` | bool | False | 是否按缩放因子随机化 PD 增益 |
| `kp_range` | list | [0.8, 1.2] | Kp 缩放范围 |
| `kd_range` | list | [0.8, 1.2] | Kd 缩放范围 |
| `randomize_joint_armature` | bool | False | 是否随机化关节惯量（在 Genesis 中会显著降低仿真速度） |
| `joint_armature_range` | list | [0.0, 0.05] | 关节惯量范围 [N*m*s/rad] |
| `randomize_joint_friction` | bool | False | 是否随机化关节摩擦 |
| `joint_friction_range` | list | [0.0, 0.1] | 关节摩擦范围 |
| `randomize_joint_damping` | bool | False | 是否随机化关节阻尼 |
| `joint_damping_range` | list | [0.0, 1.0] | 关节阻尼范围 |
| `push_links` | bool | False | 是否对机器人连杆施加随机推力 |
| `max_push_force` | float | 10.0 | 施加到每个连杆的随机推力最大幅值 [N] |
| `push_links_interval_s` | float | 15.0 | 随机推力之间的时间间隔 [s] |

---

### `normalization` 归一化配置（Normalization Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `clip_observations` | float | 100. | 观测裁剪（clipping）范围 |
| `clip_actions` | float | 100. | 动作裁剪范围 |

#### `obs_scales` 观测缩放（Observation Scales）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lin_vel` | float | 1.0 | 线速度观测缩放因子 |
| `ang_vel` | float | 0.25 | 角速度观测缩放因子 |
| `dof_pos` | float | 1.0 | 关节位置观测缩放因子 |
| `dof_vel` | float | 0.05 | 关节速度观测缩放因子 |
| `height_measurements` | float | 5.0 | 高度测量观测缩放因子 |

---

### `noise` 噪声配置（Noise Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `add_noise` | bool | True | 是否添加噪声 |
| `noise_level` | float | 1.0 | 噪声缩放因子 |

#### `noise_scales` 噪声缩放（Noise Scales）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dof_pos` | float | 0.01 | 关节位置噪声缩放 |
| `dof_vel` | float | 0.5 | 关节速度噪声缩放 |
| `lin_vel` | float | 0.1 | 线速度噪声缩放 |
| `ang_vel` | float | 0.2 | 角速度噪声缩放 |
| `gravity` | float | 0.05 | 重力观测噪声缩放 |
| `height_measurements` | float | 0.1 | 高度测量噪声缩放 |

---

### `constraints` 约束配置（Constraints Configuration）

用于 CaT（Constraints as Termination，约束即终止）方法。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `limits` | class | - | 约束限制配置 |

---

### `viewer` 查看器配置（Viewer Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ref_env` | int | 0 | 参考环境索引 |
| `pos` | list | [4.0, 4.0, 2.0] | 相机位置（相对于机器人位置）[m] |
| `lookat` | list | [0., 0, 0.] | 相机注视点（相对于机器人位置）[m] |
| `rendered_envs_idx` | list | [0, 1, 2, 3, 4] | 仅限 Genesis：要渲染的环境索引列表 |

---

### `sensor` 传感器配置（Sensor Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `add_depth` | bool | False | 是否添加深度相机（depth camera） |
| `use_warp` | bool | False | 是否使用基于 warp 的模型 |

#### `depth_camera_config` 深度相机配置（Depth Camera Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_sensors` | int | 1 | 传感器数量 |
| `num_history` | int | 1 | 深度图像的历史帧数 |
| `near_clip` | float | 0.1 | 近裁剪平面（near clipping plane） |
| `far_clip` | float | 10.0 | 远裁剪平面（far clipping plane） |
| `near_plane` | float | 0.1 | 近平面 |
| `far_plane` | float | 10.0 | 远平面 |
| `resolution` | tuple | (80, 60) | 图像分辨率 |
| `horizontal_fov_deg` | float | 75 | 水平视场角 [度] |
| `pos` | tuple | (0.3, 0.0, 0.1) | 相机位置 |
| `euler` | tuple | (0.0, 0.0, 0.0) | 相机欧拉角（Euler angles） |
| `decimation` | int | 5 | 抽取（decimation）因子 |
| `calculate_depth` | bool | True | 仅限 Warp：是否计算深度 |
| `segmentation_camera` | bool | False | 仅限 Warp：是否使用分割相机（segmentation camera） |
| `return_pointcloud` | bool | False | 仅限 Warp：是否返回点云（point cloud） |
| `pointcloud_in_world_frame` | bool | False | 仅限 Warp：点云是否在世界坐标系中 |

---

### `sim` 仿真配置（Simulation Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dt` | float | 0.005 | 仿真时间步长，默认为 200Hz |
| `substeps` | int | 1 | 子步数 |
| `max_collision_pairs` | int | 100 | 仅限 Genesis：最大碰撞对（collision pairs）数量，更多碰撞对会占用更多 GPU 内存并降低仿真速度 |
| `IK_max_targets` | int | 2 | 仅限 Genesis：逆运动学（Inverse Kinematics, IK）目标数量，较少的目标会减少内存使用 |
| `gravity` | list | [0., 0., -9.81] | 仅限 IsaacGym：重力加速度 [m/s^2] |
| `up_axis` | int | 1 | 仅限 IsaacGym：向上轴方向，0 表示 y，1 表示 z |
| `use_gpu_pipeline` | bool | True | 仅限 IsaacGym：是否使用 GPU 管线 |

#### `physx` PhysX 引擎配置（PhysX Engine Configuration）（仅限 IsaacGym）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_gpu` | bool | True | 是否使用 GPU |
| `num_subscenes` | int | 0 | 子场景数量 |
| `num_threads` | int | 10 | 线程数 |
| `solver_type` | int | 1 | 求解器类型：0 表示 pgs，1 表示 tgs |
| `num_position_iterations` | int | 4 | 位置迭代次数 |
| `num_velocity_iterations` | int | 0 | 速度迭代次数 |
| `contact_offset` | float | 0.01 | 接触偏移 [m] |
| `rest_offset` | float | 0.0 | 静止偏移 [m] |
| `bounce_threshold_velocity` | float | 0.5 | 弹跳阈值速度 [m/s] |
| `max_depenetration_velocity` | float | 1.0 | 最大去穿透速度（maximum depenetration velocity） |
| `max_gpu_contact_pairs` | int | 2**23 | 最大 GPU 接触对 |
| `default_buffer_size_multiplier` | int | 5 | 默认缓冲区大小乘数 |
| `contact_collection` | int | 2 | 接触收集模式：0 从不，1 最后子步，2 所有子步 |

---

## LeggedRobotCfgPPO

PPO 算法配置类。

### 基本参数（Basic Parameters）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | int | 1 | 随机种子（random seed） |
| `runner_class_name` | str | 'OnPolicyRunner' | 运行器（runner）类名 |

---

### `policy` 策略配置（Policy Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `clip_actions` | float | LeggedRobotCfg.normalization.clip_actions | 动作裁剪范围 |
| `init_noise_std` | float | 1.0 | 初始噪声标准差（standard deviation） |
| `actor_hidden_dims` | list | [512, 256, 128] | Actor 网络隐藏层维度 |
| `critic_hidden_dims` | list | [512, 256, 128] | Critic 网络隐藏层维度 |
| `activation` | str | 'elu' | 激活函数（activation function），选项：elu, relu, selu, crelu, lrelu, tanh, sigmoid |

#### RNN 配置（RNN Configuration）（仅适用于 ActorCriticRecurrent）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rnn_type` | str | 'lstm' | RNN 类型 |
| `rnn_hidden_size` | int | 512 | RNN 隐藏层大小 |
| `rnn_num_layers` | int | 1 | RNN 层数 |

---

### `algorithm` 算法配置（Algorithm Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `value_loss_coef` | float | 1.0 | 价值损失（value loss）系数 |
| `use_clipped_value_loss` | bool | True | 是否使用裁剪价值损失 |
| `clip_param` | float | 0.2 | PPO 裁剪参数（clipping parameter） |
| `entropy_coef` | float | 0.01 | 熵（entropy）系数 |
| `num_learning_epochs` | int | 5 | 学习轮数（learning epochs） |
| `num_mini_batches` | int | 4 | 小批量（mini batches）数量，小批量大小 = num_envs * nsteps / nminibatches |
| `learning_rate` | float | 1.e-3 | 学习率（learning rate） |
| `schedule` | str | 'adaptive' | 学习率调度（schedule）：'adaptive' 或 'fixed' |
| `gamma` | float | 0.99 | 折扣因子（discount factor） |
| `lam` | float | 0.95 | GAE lambda 参数 |
| `desired_kl` | float | 0.01 | 目标 KL 散度（KL divergence） |
| `max_grad_norm` | float | 1. | 最大梯度裁剪范数（maximum gradient clipping norm） |
| `use_spo` | bool | False | 是否使用 SPO（Simple Policy Optimization，简单策略优化）。注意：SPO 可能与默认 PPO 参数不兼容，建议：learning_rate=2.5e-4, schedule='fixed' |

---

### `runner` 运行器配置（Runner Configuration）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `policy_class_name` | str | 'ActorCritic' | 策略类名 |
| `algorithm_class_name` | str | 'PPO' | 算法类名 |
| `num_steps_per_env` | int | 24 | 每个环境每次迭代的步数 |
| `max_iterations` | int | 1500 | 策略更新最大迭代次数 |
| `sync_wandb` | bool | False | 是否同步日志到 wandb |
| `save_interval` | int | 50 | 每隔这么多迭代检查是否保存 |
| `experiment_name` | str | 'test' | 实验名称 |
| `run_name` | str | '' | 运行名称 |
| `resume` | bool | False | 是否从检查点（checkpoint）恢复 |
| `load_run` | int | -1 | 要加载的运行 ID，-1 表示最后一次运行 |
| `checkpoint` | int | -1 | 检查点 ID，-1 表示最后保存的模型 |
| `resume_path` | str | None | 恢复路径，从 load_run 和 checkpoint 更新 |

## 附录：Main 新增内容（中英对照）

- Intro: This document provides detailed descriptions of all configuration parameters defined in . 本文档提供了  中定义的所有配置参数（configuration parameters）的详细说明。
- Terrain: Terrain configuration in main adds/broadens parameter descriptions. 地形配置在主分支中添加/扩展了参数描述，以下为要点对照：
  - mesh_type: Terrain mesh type: plane, heightfield, trimesh; 地形网格类型：plane、heightfield、trimesh
  - plane_length: Plane size [m], default 200; 平面尺寸 [m]，默认 200
  - horizontal_scale: Distance between height samples in x/y direction [m]; X/Y 方向高度采样点间距 [m]
