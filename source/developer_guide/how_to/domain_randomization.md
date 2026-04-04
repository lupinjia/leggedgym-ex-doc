# 域随机化调优指南

域随机化（DR）是强化学习中实现 Sim-to-Real 转移的关键技术。通过在训练期间随机化仿真参数，策略学会对仿真与现实之间不可避免的差异具有鲁棒性。本指南涵盖 LeggedGym-Ex 中完整的域随机化系统，包括可用参数、仿真器特定考虑因素和经过验证的调优策略。

## 理解域随机化

Sim-to-Real 转移的根本挑战源于现实差距，即仿真与现实动力学之间的差异。由于未建模的物理特性、传感器噪声、执行器延迟和环境变化，完美的仿真保真度是不可能的。域随机化通过在动力学分布上训练策略而不是在单一确定性模型上训练来解决这个问题。

关键见解是，在广泛参数变化上训练的策略学习到超越任何单一仿真实例特定动力学的特征和策略。部署时，真实机器人落在这个训练分布的某个位置，使策略无需显式微调即可适应。

## 可用随机化选项

LeggedGym-Ex 通过 `domain_rand` 配置部分提供全面的域随机化。所有参数都在 `LeggedRobotCfg.domain_rand` 中定义，可以独立启用或禁用。

### 摩擦随机化

地面摩擦在不同表面之间变化巨大，从抛光地板（μ ≈ 0.2）到粗糙地形（μ ≈ 1.5）。摩擦随机化训练策略处理这种可变性而无需显式表面识别。

```python
class domain_rand:
    randomize_friction: bool = True
    friction_range: List[float] = [0.5, 1.25]
```

**推荐范围**：一般运动为 `[0.5, 1.25]`。在多样化表面上部署时扩展到 `[0.2, 1.7]`。

**应用时机**：在回合重置时应用。摩擦系数从指定范围均匀采样并应用于机器人的所有连杆。

**物理解释**：摩擦范围覆盖典型的橡胶对混凝土（μ ≈ 0.8-1.0）、橡胶对木材（μ ≈ 0.6-0.8）和更滑的表面。大于 1.5 的值不常见，但可能代表高摩擦橡胶或专用地形。

### 恢复系数随机化

恢复系数（coefficient of restitution）控制接触的弹性。虽然经常被忽视，但恢复系数影响冲击动力学，并能显著影响步态稳定性。

```python
class domain_rand:
    randomize_restitution: bool = False
    restitution_range: List[float] = [0.0, 0.5]
```

**推荐范围**：`[0.0, 0.5]`。大多数现实表面恢复系数低于 0.3。启用以实现鲁棒的冲击处理。

**默认值**：默认禁用，因为大多数地形表现出低恢复系数。在具有显著弹性的表面上部署时启用（健身房地板、蹦床）。

### 质量随机化

基座质量随机化考虑有效载荷变化、电池重量差异和 URDF 中的建模误差。这是最有影响力的随机化之一。

```python
class domain_rand:
    randomize_base_mass: bool = True
    added_mass_range: List[float] = [-1.0, 1.0]
```

**推荐范围**：对于像 Go2 这样的中型四足机器人（基座质量约 15kg），`[-1.0, 1.0]` kg。按其他机器人尺寸比例缩放（基座质量的 ±5-10%）。

**物理解释**：该范围覆盖电池更换、传感器有效载荷或建模误差的典型变化。15kg 机器人上 1kg 的变化代表约 6.7% 的质量不确定性。

### 质心位移

质心（CoM）位置很少被精确知道。制造变化、电缆布线和组件放置都会影响真实的 CoM 位置。

```python
class domain_rand:
    randomize_com_displacement: bool = True
    com_pos_x_range: List[float] = [-0.01, 0.01]
    com_pos_y_range: List[float] = [-0.01, 0.01]
    com_pos_z_range: List[float] = [-0.01, 0.01]
```

**推荐范围**：每个轴 `[-0.01, 0.01]` m（±1cm）。更大的机器人可以使用 `[-0.03, 0.03]` m。

**物理解释**：1cm 的 CoM 偏移代表显著但现实的误差。对于 15kg 机器人，1cm 偏移在中等加速度下产生约 0.15 Nm 的意外力矩。

### PD 增益随机化

PD 控制器增益直接影响动作如何转换为关节力矩。随机化这些增益训练策略对执行器建模误差和增益校准误差具有鲁棒性。

```python
class domain_rand:
    randomize_pd_gain: bool = False
    kp_range: List[float] = [0.8, 1.2]  # Scale factor
    kd_range: List[float] = [0.8, 1.2]  # Scale factor
```

**推荐范围**：中等变化为 `[0.8, 1.2]`（±20%）。更宽的 `[0.5, 1.5]` 以获得更强的鲁棒性。

**工作原理**：缩放因子乘以 `cfg.control.stiffness` 和 `cfg.control.damping` 中定义的名义 PD 增益。1.0 的缩放意味着名义增益；0.8 意味着低 20% 的增益。

**影响**：这种随机化显著增加训练难度。仅在实现其他随机化的稳定运动后才启用。

### 外部推动扰动

随机推动训练恢复行为并提高对扰动的鲁棒性。对于将在非结构化环境中运行的策略这是必不可少的。

```python
class domain_rand:
    push_robots: bool = True
    push_interval_s: int = 15
    max_push_vel_xy: float = 1.0
```

**推荐设置**：
- `push_interval_s = 15`：每 15 秒推动一次（较不频繁允许策略稳定）
- `max_push_vel_xy = 1.0`：最大推动速度 1.0 m/s（中等扰动）

**调优指南**：
- 从不频繁的推动（`push_interval_s = 15-20`）和中等速度开始
- 增加频率以获得更强的恢复行为
- 更高的速度（`1.5-2.0 m/s`）创造更具挑战性的场景

### 连杆推动力

除了基座推动外，随机力可以应用于单个连杆以模拟接触扰动和阵风吹袭。

```python
class domain_rand:
    push_links: bool = False
    max_push_force: float = 10.0  # Newtons
    push_links_interval_s: float = 15.0
```

**推荐范围**：四足机器人为 `5.0-15.0` N。超过 20N 的力可能不现实地大。

**用例**：当机器人将在有频繁接触扰动的环境中运行时启用（拥挤空间、机械手交互）。

### 控制延迟随机化

真实系统由于通信开销、电机响应时间和计算而在命令生成和执行之间经历延迟。随机化此延迟提高对延迟的鲁棒性。

```python
class domain_rand:
    randomize_ctrl_delay: bool = False
    ctrl_delay_step_range: List[int] = [0, 1]  # Number of simulation steps
```

**推荐范围**：50Hz 控制下 `[0, 3]` 步（0-60ms 延迟）。真实系统通常有 20-50ms 的延迟。

**影响**：延迟随机化显著增加训练难度。仅推荐在延迟是已知问题的部署场景中启用此功能。

### 关节动力学随机化

关节级参数捕获执行器建模误差和单个电机之间的变化。

```python
class domain_rand:
    randomize_joint_armature: bool = False
    joint_armature_range: List[float] = [0.0, 0.05]  # N*m*s/rad
    
    randomize_joint_friction: bool = False
    joint_friction_range: List[float] = [0.0, 0.1]  # N*m
    
    randomize_joint_damping: bool = False
    joint_damping_range: List[float] = [0.0, 1.0]  # N*m*s/rad
```

**物理解释**：
- **转动惯量**：来自电机转子和齿轮系的附加转动惯量
- **关节摩擦**：阻碍关节运动的库仑摩擦
- **关节阻尼**：与关节速度成比例的粘性阻尼

**建议**：使用系统辨识来确定适当的范围。这些参数显著影响关节动力学，应基于真实机器人数据进行校准。

### 相机随机化

对于基于视觉的策略，相机位姿随机化提高对传感器安装变化的鲁棒性。

```python
class domain_rand:
    randomize_camera_pos: bool = False
    camera_com_displacement_range: List[float] = [0.01, 0.01, 0.01]
    
    randomize_camera_euler: bool = False
    camera_euler_range: List[float] = [0.1, 0.1, 0.1]  # radians
```

**推荐范围**：1-2cm 的位置偏移和 0.1-0.2 弧度（5-10 度）的角度偏移。

## 仿真器特定考虑因素

LeggedGym-Ex 支持三种具有不同域随机化能力和性能特征的仿真器。

### IsaacGym

IsaacGym 提供最快的训练速度和完整的 DR 支持。主要考虑因素：

- **性能**：最高吞吐量，适合广泛的 DR 训练
- **摩擦**：在回合重置时应用，不能在回合中途修改
- **关节参数**：通过 PhysX DOF 属性支持
- **限制**：对于某些参数，环境创建后无法修改刚体属性

### Genesis

Genesis 提供出色的物理精度，但某些 DR 操作有性能考虑：

```{warning}
在 Genesis 中随机化关节转动惯量、摩擦和阻尼需要批处理 DOF/连杆信息，这会显著降低仿真速度。建议在使用 Genesis 时将 `randomize_joint_armature`、`randomize_joint_friction` 和 `randomize_joint_damping` 设置为 `False`。如果需要这些随机化，请使用 IsaacGym 或 IsaacLab。
```

**Genesis 特定说明**：
- 摩擦和质量随机化高效工作
- 避免关节级随机化（转动惯量、摩擦、阻尼）由于性能影响
- 用于表面特性和质量变化是主要关注的场景

### IsaacLab

IsaacLab 提供最逼真的渲染和良好的 DR 支持，但需要特殊处理某些操作：

**CPU 张量要求**：对于某些操作，域随机化张量必须在 CPU 上：

```python
# From isaaclab_simulator.py
# Tensors passed to set_material_properties, set_masses, set_coms must be on CPU
all_indices = torch.arange(self._robot.root_physx_view.count, device="cpu")
self._robot.root_physx_view.set_material_properties(
    target_material_props.to('cpu'), all_indices
)
```

**IsaacLab 特定说明**：
- 所有材料和质量属性修改需要 CPU 张量
- 支持全套随机化选项
- 当视觉真实感重要或需要与 IsaacSim 功能集成时使用

## 渐进随机化策略

同时启用所有随机化通常会阻止训练收敛。渐进策略产生更好的结果：

### 阶段 1：建立基线（迭代 0-500）

从最小随机化开始学习基本运动：

```python
class domain_rand:
    randomize_friction = True
    friction_range = [0.8, 1.0]  # Narrow range
    randomize_base_mass = True
    added_mass_range = [-0.5, 0.5]
    push_robots = True
    push_interval_s = 15
    max_push_vel_xy = 0.5
    
    # Disable all others
    randomize_com_displacement = False
    randomize_pd_gain = False
    randomize_ctrl_delay = False
    randomize_joint_armature = False
    randomize_joint_friction = False
    randomize_joint_damping = False
```

### 阶段 2：扩展核心随机化（迭代 500-1000）

逐渐扩展已启用随机化的范围：

```python
class domain_rand:
    randomize_friction = True
    friction_range = [0.5, 1.25]  # Expanded
    randomize_base_mass = True
    added_mass_range = [-1.0, 1.0]  # Expanded
    randomize_com_displacement = True  # Now enabled
    com_pos_x_range = [-0.01, 0.01]
    push_robots = True
    push_interval_s = 15
    max_push_vel_xy = 1.0  # Increased
```

### 阶段 3：完全随机化（迭代 1000+）

启用剩余随机化以获得最大鲁棒性：

```python
class domain_rand:
    randomize_friction = True
    friction_range = [0.3, 1.5]
    randomize_base_mass = True
    added_mass_range = [-1.5, 1.5]
    randomize_com_displacement = True
    com_pos_x_range = [-0.02, 0.02]
    randomize_pd_gain = True
    kp_range = [0.8, 1.2]
    kd_range = [0.8, 1.2]
    push_robots = True
    push_interval_s = 10
    max_push_vel_xy = 1.5
    
    # Optional: Enable for specific deployment scenarios
    randomize_ctrl_delay = True
    ctrl_delay_step_range = [0, 2]
```

## 调优工作流程

遵循这种系统方法为您的特定部署调优域随机化：

### 步骤 1：系统辨识

在调优 DR 之前，使用系统辨识校准您的仿真：

1. 使用正弦关节命令从真实机器人收集轨迹数据
2. 运行系统辨识脚本以查找匹配参数
3. 更新机器人配置中的默认值

有关详细的系统辨识程序，请参阅 [Sim-to-Real 转移指南](../../user_guide/blind_locomotion/sim2real_skills.md)。

### 步骤 2：识别部署条件

记录预期的操作条件：

- **地形类型**：光滑地板、粗糙地形、斜坡、楼梯？
- **有效载荷变化**：固定有效载荷还是可变有效载荷？
- **环境扰动**：拥挤空间、风、接触？
- **表面条件**：干燥、潮湿、多尘？

### 步骤 3：配置随机化范围

根据部署条件设置范围：

| 条件 | 推荐配置 |
|-----------|--------------------------|
| 室内光滑地板 | `friction_range = [0.4, 0.8]`，窄质量范围 |
| 混合室内/室外 | `friction_range = [0.3, 1.5]`，中等质量范围 |
| 粗糙室外地形 | `friction_range = [0.5, 1.7]`，宽质量范围 |
| 可变有效载荷 | `added_mass_range = [-3.0, 3.0]` 或更宽 |
| 非结构化环境 | 启用频繁间隔的 push_robots |

### 步骤 4：通过 Sim2Sim 验证

在真实部署之前，使用跨仿真器转移进行验证：

1. 在 IsaacGym/Genesis 中配置 DR 训练策略
2. 在不同仿真器（通过 go2_deploy 的 MuJoCo）中测试
3. 如果转移失败，扩展 DR 范围

### 步骤 5：迭代改进

使用以下诊断来改进 DR 设置：

| 症状 | 可能原因 | 调整 |
|---------|-------------|------------|
| 策略在仿真中成功但在真实机器人上失败 | DR 范围太窄 | 扩展范围 20-50% |
| 训练无法收敛 | DR 范围太宽 | 缩小范围，使用渐进策略 |
| 真实机器人上出现不稳定振荡 | 执行器动力学不匹配 | 启用 PD 增益和关节随机化 |
| 策略在扰动下崩溃 | 推动训练不足 | 增加推动频率和速度 |
| 静态行为良好但运动差 | 摩擦随机化不足 | 扩展摩擦范围 |

## 最佳实践总结

1. **从保守开始**：从窄随机化范围开始并逐步扩展
2. **优先考虑影响**：首先关注摩擦、质量和推动随机化，这些影响最大
3. **匹配部署**：配置范围以覆盖预期的现实世界条件
4. **系统验证**：在真实部署之前始终通过 sim2sim 转移进行验证
5. **记录性能**：追踪哪些 DR 配置适用于哪些部署场景
6. **考虑仿真器**：基于 DR 需求选择仿真器，Genesis 用于基本 DR，IsaacGym/IsaacLab 用于关节级随机化
7. **使用系统辨识**：在广泛的 DR 训练之前校准仿真参数
8. **监控训练**：观察奖励曲线，如果训练变得不稳定，缩小 DR 范围

## 与教师-学生训练集成

当与教师-学生框架结合时，域随机化最有效。教师策略接收关于随机化参数的特权信息，而学生策略学习从观测历史中估计它们。

当使用 DR 与教师-学生训练时，随机化参数自动包含在特权观测中：

```python
# From legged_robot.py - privileged observations include DR parameters
self.privileged_obs_buf = torch.cat((
    # ... standard observations ...
    self.simulator.dr_friction_values,        # Friction value
    self.simulator.dr_added_base_mass,        # Added mass
    self.simulator.dr_base_com_bias,          # CoM displacement
    self.simulator.dr_rand_push_vels[:, :2],  # Push velocities
), dim=-1)
```

这允许学生策略从可观测的量（关节位置、速度、IMU 数据）学习隐式系统辨识。

## 常见陷阱

### 同时启用所有随机化

**问题**：训练永不收敛或奖励曲线保持平坦。
**解决方案**：使用上述渐进随机化策略。

### 不现实的随机化范围

**问题**：策略学习利用仿真特性的不现实行为。
**解决方案**：基于物理测量和制造商规格设定范围。

### 忽视仿真器限制

**问题**：Genesis 中的关节随机化导致显著减速。
**解决方案**：对于关节级随机化使用 IsaacGym 或 IsaacLab。

### 跳过验证

**问题**：策略在训练中似乎有效但在真实机器人上失败。
**解决方案**：在真实部署之前始终通过 sim2sim 转移进行验证。

## 参考文献

1. [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) - 基础 DR 论文
2. [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) - 随机化策略
3. [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251) - 使用 DR 的教师-学生方法
