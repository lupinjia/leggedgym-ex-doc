# 开发自定义奖励函数

本文档介绍如何在 LeggedGym-Ex 中创建自定义奖励函数。该框架使用自动奖励发现机制，使添加新奖励变得简单且易于维护。

---

## 概述

LeggedGym-Ex 中的奖励函数遵循基于约定的发现模式。只需将奖励方法命名为带有 `_reward_` 前缀的名称，并在配置中添加相应的比例，框架就会自动将您的奖励集成到训练循环中。

**主要优势：**
- 无需手动注册
- 自动奖励累加和追踪
- 内置回合统计记录
- 易于调试和监控

---

## 自动发现机制

### 工作原理

`LeggedRobot` 中的 `_prepare_reward_function()` 方法会在初始化时自动发现所有奖励方法：

```python
def _prepare_reward_function(self) -> None:
    """Prepares a list of reward functions, which will be called 
    to compute the total reward.
    """
    # Remove zero scales and multiply non-zero ones by dt
    for key in list(self.reward_scales.keys()):
        scale = self.reward_scales[key]
        if scale == 0:
            self.reward_scales.pop(key)
        else:
            self.reward_scales[key] *= self.dt
    
    # Prepare list of functions
    self.reward_functions = []
    self.reward_names = []
    for name, scale in self.reward_scales.items():
        if name == "termination":
            continue
        self.reward_names.append(name)
        method_name = '_reward_' + name
        
        # Validate that the method exists
        assert hasattr(self, method_name), (
            f"Reward function '{method_name}' not found for reward scale '{name}'. "
            f"You must implement a method '_reward_{name}()'."
        )
        self.reward_functions.append(getattr(self, method_name))
    
    # Initialize episode sums for logging
    self.episode_sums = {
        name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for name in self.reward_scales.keys()
    }
```

### 发现过程

1. **配置解析**：框架从您的配置类中读取 `cfg.rewards.scales`
2. **比例过滤**：移除零比例奖励；非零比例乘以 `dt`
3. **方法发现**：对于每个奖励名称，查找 `_reward_<name>()` 方法
4. **验证**：断言每个配置的奖励都有对应的方法
5. **存储**：存储函数引用和名称以进行高效计算

---

## 奖励函数模式

### 命名约定

奖励方法必须遵循此模式：

```python
def _reward_<name>(self) -> Reward:
    """Compute reward for <name>.
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Your reward computation here
    return reward
```

**规则：**
- 方法名必须以 `_reward_` 开头
- `_reward_` 后的后缀必须与 `cfg.rewards.scales` 中的键匹配
- 必须返回形状为 `(num_envs,)` 的张量
- 返回类型应为 `Reward`（`Tensor` 的别名）

### 配置

在配置中添加您的奖励比例：

```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            my_custom_reward = 0.5  # Will look for _reward_my_custom_reward()
```

### 计算流程

在训练期间，每个步骤都会调用 `compute_reward()`：

```python
def compute_reward(self) -> None:
    """Compute rewards for all environments."""
    self.rew_buf[:] = 0.
    
    for i in range(len(self.reward_functions)):
        name = self.reward_names[i]
        rew = self.reward_functions[i]() * self.reward_scales[name]
        self.rew_buf += rew
        self.episode_sums[name] += rew
    
    # Optionally clip to positive rewards
    if self.cfg.rewards.only_positive_rewards:
        self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
```

---

## 常见奖励模式

### 模式 1：追踪奖励

追踪奖励鼓励机器人遵循指令或目标值。它们通常使用指数核函数以获得平滑的梯度。

```python
def _reward_tracking_lin_vel(self) -> Reward:
    """Track linear velocity commands."""
    lin_vel_error = torch.sum(torch.square(
        self.commands[:, :2] - self.simulator.base_lin_vel[:, :2]
    ), dim=1)
    return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
```

**适用场景：** 跟随速度指令、追踪参考动作、保持目标姿态。

**关键特征：**
- 使用指数核函数：`exp(-error / sigma)`
- 返回值在 `[0, 1]` 范围内
- 平滑梯度鼓励稳定收敛

### 模式 2：惩罚奖励

惩罚用于阻止不良行为。它们通常返回平方误差或绝对值。

```python
def _reward_torques(self) -> Reward:
    """Penalize large torques for energy efficiency."""
    return torch.sum(torch.square(self.simulator.torques), dim=1)

def _reward_action_rate(self) -> Reward:
    """Penalize rapid action changes for smoothness."""
    return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
```

**适用场景：** 能量效率、平滑运动、避免关节极限、防止碰撞。

**关键特征：**
- 返回非负值
- 通常使用 `torch.square()` 或 `torch.abs()`
- 可使用较小的权重进行缩放（例如 `-0.001`）

### 模式 3：条件奖励

条件奖励仅在特定条件下应用，例如在特定运动阶段期间。

```python
def _reward_feet_air_time(self) -> Reward:
    """Reward long steps (feet air time)."""
    contact = self.simulator.link_contact_forces[
        :, self.simulator.feet_contact_indices, 2
    ] > 1.
    contact_filt = torch.logical_or(contact, self.last_contacts)
    self.last_contacts = contact
    
    first_contact = (self.feet_air_time > 0.) * contact_filt
    self.feet_air_time += self.dt
    
    rew_airTime = torch.sum(
        (self.feet_air_time - 0.3) * first_contact, dim=1
    )
    # Only reward when moving
    rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.2
    self.feet_air_time *= ~contact_filt
    
    return rew_airTime
```

**适用场景：** 步态特定奖励、相位相关行为、条件惩罚。

**关键特征：**
- 使用状态依赖条件
- 可能需要维护额外的缓冲区（例如 `feet_air_time`）
- 通常与指令条件结合使用

### 模式 4：参考追踪奖励

对于运动模仿任务，奖励追踪参考运动数据。

```python
def _reward_tracking_ref_dof_pos(self) -> Reward:
    """Track reference motion DOF positions."""
    dof_pos_error = torch.sum(torch.square(
        self.simulator.dof_pos - 
        self.motion_loader.get_ref_dof_pos()
    ), dim=-1)
    
    return torch.exp(-dof_pos_error / self.cfg.rewards.tracking_dof_pos_sigma)
```

**适用场景：** DeepMimic 风格的运动模仿、AMP、风格迁移。

**关键特征：**
- 需要参考运动数据加载器
- 追踪多个方面（位置、速度、方向）
- 使用指数核函数以获得平滑梯度

---

## 示例实现

### 示例 1：能量效率奖励

一个平衡速度和能耗的自定义奖励：

```python
def _reward_energy_efficiency(self) -> Reward:
    """Reward energy-efficient locomotion.
    
    Balances forward velocity with power consumption.
    Higher velocity with lower power = higher reward.
    """
    # Forward velocity (positive x in body frame)
    forward_vel = self.simulator.base_lin_vel[:, 0]
    
    # Power consumption (torque * velocity)
    power = torch.sum(
        torch.abs(self.simulator.torques * self.simulator.dof_vel), 
        dim=1
    )
    
    # Efficiency = velocity / power (with small epsilon to avoid division by zero)
    efficiency = forward_vel / (power + 0.01)
    
    # Only reward positive forward velocity
    reward = torch.clamp(efficiency, min=0.0)
    
    return reward
```

**配置：**
```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            energy_efficiency = 0.1
```

### 示例 2：步态对称奖励

鼓励腿部对称运动：

```python
def _reward_gait_symmetry(self) -> Reward:
    """Penalize asymmetric leg movements.
    
    Computes the difference between left and right leg joint positions
    and velocities to encourage symmetric gaits.
    """
    dof_pos = self.simulator.dof_pos
    dof_vel = self.simulator.dof_vel
    
    # Assuming 12 DOFs: FL(3), FR(3), RL(3), RR(3)
    # Front legs symmetry
    front_pos_diff = torch.sum(torch.square(
        dof_pos[:, 0:3] - dof_pos[:, 3:6]
    ), dim=1)
    front_vel_diff = torch.sum(torch.square(
        dof_vel[:, 0:3] - dof_vel[:, 3:6]
    ), dim=1)
    
    # Rear legs symmetry
    rear_pos_diff = torch.sum(torch.square(
        dof_pos[:, 6:9] - dof_pos[:, 9:12]
    ), dim=1)
    rear_vel_diff = torch.sum(torch.square(
        dof_vel[:, 6:9] - dof_vel[:, 9:12]
    ), dim=1)
    
    # Combine penalties
    symmetry_penalty = front_pos_diff + 0.1 * front_vel_diff + \
                       rear_pos_diff + 0.1 * rear_vel_diff
    
    return symmetry_penalty
```

**配置：**
```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            gait_symmetry = -0.01  # Negative scale for penalty
```

### 示例 3：基座稳定性奖励

奖励在运动中保持基座稳定：

```python
def _reward_base_stability(self) -> Reward:
    """Reward stable base orientation and height.
    
    Penalizes orientation deviation and height oscillation
    for smoother locomotion.
    """
    # Orientation stability (projected gravity should be [0, 0, 1])
    orientation_error = torch.sum(torch.square(
        self.simulator.projected_gravity[:, :2]
    ), dim=1)
    
    # Height stability (base height should be consistent)
    base_height = self.simulator.base_pos[:, 2]
    height_error = torch.square(
        base_height - self.cfg.rewards.base_height_target
    )
    
    # Angular velocity penalty (should be minimal)
    ang_vel_error = torch.sum(torch.square(
        self.simulator.base_ang_vel
    ), dim=1)
    
    # Combined stability reward using exponential kernel
    stability_error = orientation_error + height_error + 0.1 * ang_vel_error
    
    return torch.exp(-stability_error / 0.5)
```

**配置：**
```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            base_stability = 0.5
        base_height_target = 0.35  # meters
```

### 示例 4：足部放置奖励

鼓励行走时正确的足部放置：

```python
def _reward_foot_placement(self) -> Reward:
    """Reward proper foot placement relative to body.
    
    Encourages feet to land in a good support polygon
    under the body.
    """
    # Foot positions relative to base
    foot_pos_rel = self.simulator.feet_pos - self.simulator.base_pos.unsqueeze(1)
    
    # Desired foot positions (spread out for stability)
    # For a quadruped: FL, FR, RL, RR
    desired_spread = 0.2  # lateral spread
    desired_length = 0.2  # forward/backward offset
    
    # Create desired foot positions
    desired_pos = torch.zeros_like(foot_pos_rel)
    desired_pos[:, 0, 0] = desired_length   # FL x
    desired_pos[:, 0, 1] = desired_spread   # FL y
    desired_pos[:, 1, 0] = desired_length   # FR x
    desired_pos[:, 1, 1] = -desired_spread  # FR y
    desired_pos[:, 2, 0] = -desired_length  # RL x
    desired_pos[:, 2, 1] = desired_spread   # RL y
    desired_pos[:, 3, 0] = -desired_length  # RR x
    desired_pos[:, 3, 1] = -desired_spread  # RR y
    
    # Compute placement error only for feet in contact
    contacts = self.simulator.link_contact_forces[
        :, self.simulator.feet_contact_indices, 2
    ] > 1.0
    
    placement_error = torch.zeros(self.num_envs, device=self.device)
    for i in range(4):
        foot_error = torch.sum(torch.square(
            foot_pos_rel[:, i, :2] - desired_pos[:, i, :2]
        ), dim=1)
        placement_error += foot_error * contacts[:, i]
    
    return torch.exp(-placement_error / 0.01)
```

**配置：**
```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            foot_placement = 0.3
```

---

## 奖励比例配置

### 配置结构

奖励比例在嵌套配置类中定义：

```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            # Tracking rewards (positive scales)
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            
            # Penalties (negative scales)
            torques = -0.0001
            dof_vel = -0.001
            action_rate = -0.01
            
            # Custom rewards
            my_custom_reward = 0.5
```

### 比例指南

1. **正比例**：鼓励行为（追踪、成就）
2. **负比例**：阻止行为（惩罚、成本）
3. **比例幅度**：对惩罚从小的值开始，根据经验调整

**典型比例范围：**

| 奖励类型 | 典型比例范围 | 示例 |
|------------|---------------------|---------|
| 追踪奖励 | `0.1 - 2.0` | `tracking_lin_vel = 1.0` |
| 平滑性惩罚 | `-0.001 - -0.1` | `action_rate = -0.01` |
| 能量惩罚 | `-0.0001 - -0.01` | `torques = -0.0001` |
| 终止惩罚 | `-1.0 - -10.0` | `termination = -1.0` |

### 比例与 dt 相乘

框架自动将奖励比例乘以 `dt`：

```python
self.reward_scales[key] *= self.dt
```

这确保奖励与时间步长无关。配置比例时，按单位时间步长（1 秒）设置值。

---

## 调试技巧

### 1. 打印奖励分量

添加调试打印以了解奖励贡献：

```python
def _reward_my_custom(self) -> Reward:
    reward = self.compute_my_reward()
    
    if self.cfg.env.debug_rewards:
        print(f"my_custom reward: mean={reward.mean():.4f}, "
              f"min={reward.min():.4f}, max={reward.max():.4f}")
    
    return reward
```

### 2. 记录回合统计

框架自动记录奖励统计：

```python
# In reset_idx(), episode statistics are logged:
self.extras["episode"]['rew_tracking_lin_vel'] = torch.mean(
    self.episode_sums['tracking_lin_vel'][env_ids]
) / self.max_episode_length_s
```

在 TensorBoard 或训练日志中监控这些值。

### 3. 检查奖励形状

断言奖励具有正确的形状：

```python
def _reward_my_custom(self) -> Reward:
    reward = self.compute_my_reward()
    
    assert reward.shape == (self.num_envs,), (
        f"Reward shape mismatch: expected ({self.num_envs},), "
        f"got {reward.shape}"
    )
    
    return reward
```

### 4. 可视化奖励分量

创建一个调试方法来可视化所有奖励：

```python
def debug_rewards(self):
    """Print all reward components for debugging."""
    print("\n=== Reward Components ===")
    for i, name in enumerate(self.reward_names):
        rew = self.reward_functions[i]()
        print(f"{name:30s}: mean={rew.mean():8.4f}, "
              f"std={rew.std():8.4f}, "
              f"scale={self.reward_scales[name]:8.4f}")
    print(f"{'TOTAL':30s}: mean={self.rew_buf.mean():8.4f}")
```

在训练期间调用此方法：

```python
if self.common_step_counter % 100 == 0:
    self.debug_rewards()
```

### 5. 检查 NaN 值

检测数值问题：

```python
def _reward_my_custom(self) -> Reward:
    reward = self.compute_my_reward()
    
    if torch.any(torch.isnan(reward)):
        print(f"WARNING: NaN detected in my_custom reward!")
        reward = torch.nan_to_num(reward, nan=0.0)
    
    return reward
```

### 6. 分析奖励计算

为性能计时奖励计算：

```python
import time

def compute_reward(self):
    self.rew_buf[:] = 0.
    
    for i, name in enumerate(self.reward_names):
        start = time.perf_counter()
        rew = self.reward_functions[i]() * self.reward_scales[name]
        elapsed = time.perf_counter() - start
        
        if elapsed > 0.001:  # Flag slow rewards
            print(f"Slow reward: {name} took {elapsed*1000:.2f}ms")
        
        self.rew_buf += rew
        self.episode_sums[name] += rew
```

---

## 最佳实践

### 1. 对追踪使用指数核函数

```python
# Good: Smooth gradient, bounded output
error = torch.sum(torch.square(target - current), dim=1)
reward = torch.exp(-error / sigma)

# Avoid: Unbounded output, harsh gradients
reward = -error  # Can grow arbitrarily large
```

### 2. 归一化奖励幅度

保持奖励在相似范围内以避免主导：

```python
# Good: Bounded in [0, 1]
reward = torch.exp(-error / sigma)

# If using penalties, scale appropriately
reward = error * small_scale  # e.g., 0.0001
```

### 3. 使用指令条件

仅在相关时应用奖励：

```python
# Only reward tracking when there's a non-zero command
tracking_reward *= (torch.norm(self.commands[:, :2], dim=1) > 0.1)

# Only penalize motion when standing still
stand_still_penalty *= (torch.norm(self.commands[:, :3], dim=1) < 0.2)
```

### 4. 避免奖励作弊

设计没有简单漏洞的奖励：

```python
# Bad: Robot can just stand still
def _reward_forward_velocity(self):
    return self.simulator.base_lin_vel[:, 0]

# Good: Require both forward velocity and active stepping
def _reward_forward_velocity(self):
    forward_vel = self.simulator.base_lin_vel[:, 0]
    is_moving = torch.norm(self.commands[:, :2], dim=1) > 0.1
    return forward_vel * is_moving
```

### 5. 仔细维护状态

如果您的奖励需要状态追踪：

```python
def _init_buffers(self):
    super()._init_buffers()
    self.my_state_buffer = torch.zeros(
        self.num_envs, device=self.device
    )

def _reward_needs_state(self) -> Reward:
    # Use the buffer
    reward = self.compute_with_state(self.my_state_buffer)
    
    # Update buffer for next step
    self.my_state_buffer = self.update_state()
    
    return reward
```

---

## 故障排除

### 奖励未被调用

**症状：** 自定义奖励未影响训练。

**原因：**
1. 配置中的比例为零或缺失
2. 方法名与配置键不匹配
3. 方法不在正确的类中

**解决方案：**
```python
# Check that your method exists
assert hasattr(self, '_reward_my_custom'), "Method not found!"

# Check that scale is configured
assert 'my_custom' in self.reward_scales, "Scale not configured!"
```

### NaN 奖励

**症状：** 训练因 NaN 损失而失败。

**原因：**
1. 除以零
2. 零或负数的对数
3. 数值溢出

**解决方案：**
```python
# Add epsilon to divisions
reward = value / (denominator + 1e-8)

# Clamp values before log
reward = torch.log(torch.clamp(value, min=1e-8))

# Check for NaN and replace
reward = torch.nan_to_num(reward, nan=0.0)
```

### 奖励不平衡

**症状：** 一个奖励主导总奖励。

**原因：**
1. 比例过大
2. 奖励幅度过大
3. 指数核函数的 sigma 值错误

**解决方案：**
```python
# Normalize reward output
reward = torch.exp(-error / sigma)  # Bounds to [0, 1]

# Use smaller scales
my_reward_scale = 0.01  # Instead of 1.0

# Monitor reward statistics
print(f"Reward range: [{reward.min()}, {reward.max()}]")
```

---

## 总结

在 LeggedGym-Ex 中创建自定义奖励遵循一个简单的模式：

1. **实现** 一个名为 `_reward_<name>()` 的方法，返回形状为 `(num_envs,)`
2. **配置** `cfg.rewards.scales.<name>` 中的比例
3. **调试** 使用回合统计和调试打印
4. **调整** 基于训练性能的经验调整比例

自动发现机制自动处理集成，让您可以专注于为您的特定运动任务设计有效的奖励函数。
