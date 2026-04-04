# Developing Custom Reward Functions

This guide explains how to create custom reward functions in LeggedGym-Ex. The framework uses an automatic reward discovery mechanism that makes adding new rewards straightforward and maintainable.

---

## Overview

Reward functions in LeggedGym-Ex follow a convention-based discovery pattern. By simply naming your reward method with the `_reward_` prefix and adding a corresponding scale to the configuration, the framework automatically integrates your reward into the training loop.

**Key Benefits:**
- No manual registration required
- Automatic reward accumulation and tracking
- Built-in episode statistics logging
- Easy debugging and monitoring

---

## Auto-Discovery Mechanism

### How It Works

The `_prepare_reward_function()` method in `LeggedRobot` automatically discovers all reward methods at initialization:

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

### Discovery Process

1. **Configuration Parsing**: The framework reads `cfg.rewards.scales` from your config class
2. **Scale Filtering**: Zero-scale rewards are removed; non-zero scales are multiplied by `dt`
3. **Method Discovery**: For each reward name, it looks for `_reward_<name>()` method
4. **Validation**: Asserts that each configured reward has a corresponding method
5. **Storage**: Stores function references and names for efficient computation

---

## Reward Function Pattern

### Naming Convention

Reward methods must follow this pattern:

```python
def _reward_<name>(self) -> Reward:
    """Compute reward for <name>.
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Your reward computation here
    return reward
```

**Rules:**
- Method name must start with `_reward_`
- The suffix after `_reward_` must match the key in `cfg.rewards.scales`
- Must return a tensor of shape `(num_envs,)`
- Return type should be `Reward` (alias for `Tensor`)

### Configuration

Add your reward scale in the configuration:

```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            my_custom_reward = 0.5  # Will look for _reward_my_custom_reward()
```

### Computation Flow

During training, `compute_reward()` is called each step:

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

## Common Reward Patterns

### Pattern 1: Tracking Rewards

Tracking rewards encourage the robot to follow commands or target values. They typically use exponential kernels for smooth gradients.

```python
def _reward_tracking_lin_vel(self) -> Reward:
    """Track linear velocity commands."""
    lin_vel_error = torch.sum(torch.square(
        self.commands[:, :2] - self.simulator.base_lin_vel[:, :2]
    ), dim=1)
    return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
```

**When to use:** Following velocity commands, tracking reference motions, maintaining target poses.

**Key characteristics:**
- Uses exponential kernel: `exp(-error / sigma)`
- Returns values in `[0, 1]` range
- Smooth gradient encourages stable convergence

### Pattern 2: Penalty Rewards

Penalties discourage undesirable behaviors. They typically return squared errors or absolute values.

```python
def _reward_torques(self) -> Reward:
    """Penalize large torques for energy efficiency."""
    return torch.sum(torch.square(self.simulator.torques), dim=1)

def _reward_action_rate(self) -> Reward:
    """Penalize rapid action changes for smoothness."""
    return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
```

**When to use:** Energy efficiency, smooth motion, avoiding joint limits, preventing collisions.

**Key characteristics:**
- Returns non-negative values
- Often uses `torch.square()` or `torch.abs()`
- Can be scaled with small weights (e.g., `-0.001`)

### Pattern 3: Conditional Rewards

Conditional rewards apply only under specific conditions, such as during specific motion phases.

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

**When to use:** Gait-specific rewards, phase-dependent behaviors, conditional penalties.

**Key characteristics:**
- Uses state-dependent conditions
- May require maintaining additional buffers (e.g., `feet_air_time`)
- Often combined with command conditions

### Pattern 4: Reference Tracking Rewards

For motion imitation tasks, rewards track reference motion data.

```python
def _reward_tracking_ref_dof_pos(self) -> Reward:
    """Track reference motion DOF positions."""
    dof_pos_error = torch.sum(torch.square(
        self.simulator.dof_pos - 
        self.motion_loader.get_ref_dof_pos()
    ), dim=-1)
    
    return torch.exp(-dof_pos_error / self.cfg.rewards.tracking_dof_pos_sigma)
```

**When to use:** DeepMimic-style motion imitation, AMP, style transfer.

**Key characteristics:**
- Requires reference motion data loader
- Tracks multiple aspects (position, velocity, orientation)
- Uses exponential kernels for smooth gradients

---

## Example Implementations

### Example 1: Energy Efficiency Reward

A custom reward that balances speed and energy consumption:

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

**Configuration:**
```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            energy_efficiency = 0.1
```

### Example 2: Gait Symmetry Reward

Encourage symmetric leg movements:

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

**Configuration:**
```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            gait_symmetry = -0.01  # Negative scale for penalty
```

### Example 3: Base Stability Reward

Reward keeping the base stable during locomotion:

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

**Configuration:**
```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            base_stability = 0.5
        base_height_target = 0.35  # meters
```

### Example 4: Foot Placement Reward

Encourage proper foot placement during walking:

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

**Configuration:**
```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            foot_placement = 0.3
```

---

## Reward Scales Configuration

### Configuration Structure

Reward scales are defined in the nested configuration class:

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

### Scale Guidelines

1. **Positive scales**: Encourage the behavior (tracking, achievements)
2. **Negative scales**: Discourage the behavior (penalties, costs)
3. **Scale magnitude**: Start small for penalties, tune empirically

**Typical scale ranges:**

| Reward Type | Typical Scale Range | Example |
|------------|---------------------|---------|
| Tracking rewards | `0.1 - 2.0` | `tracking_lin_vel = 1.0` |
| Smoothness penalties | `-0.001 - -0.1` | `action_rate = -0.01` |
| Energy penalties | `-0.0001 - -0.01` | `torques = -0.0001` |
| Termination penalty | `-1.0 - -10.0` | `termination = -1.0` |

### Scale vs. dt Multiplication

The framework automatically multiplies reward scales by `dt`:

```python
self.reward_scales[key] *= self.dt
```

This ensures rewards are timestep-independent. When configuring scales, set values as if for a unit timestep (1 second).

---

## Debugging Techniques

### 1. Print Reward Components

Add debug prints to understand reward contributions:

```python
def _reward_my_custom(self) -> Reward:
    reward = self.compute_my_reward()
    
    if self.cfg.env.debug_rewards:
        print(f"my_custom reward: mean={reward.mean():.4f}, "
              f"min={reward.min():.4f}, max={reward.max():.4f}")
    
    return reward
```

### 2. Log Episode Statistics

The framework automatically logs reward statistics:

```python
# In reset_idx(), episode statistics are logged:
self.extras["episode"]['rew_tracking_lin_vel'] = torch.mean(
    self.episode_sums['tracking_lin_vel'][env_ids]
) / self.max_episode_length_s
```

Monitor these in TensorBoard or your training logs.

### 3. Check Reward Shapes

Assert that rewards have the correct shape:

```python
def _reward_my_custom(self) -> Reward:
    reward = self.compute_my_reward()
    
    assert reward.shape == (self.num_envs,), (
        f"Reward shape mismatch: expected ({self.num_envs},), "
        f"got {reward.shape}"
    )
    
    return reward
```

### 4. Visualize Reward Components

Create a debug method to visualize all rewards:

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

Call this method during training:

```python
if self.common_step_counter % 100 == 0:
    self.debug_rewards()
```

### 5. Check for NaN Values

Detect numerical issues:

```python
def _reward_my_custom(self) -> Reward:
    reward = self.compute_my_reward()
    
    if torch.any(torch.isnan(reward)):
        print(f"WARNING: NaN detected in my_custom reward!")
        reward = torch.nan_to_num(reward, nan=0.0)
    
    return reward
```

### 6. Profile Reward Computation

Time reward computation for performance:

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

## Best Practices

### 1. Use Exponential Kernels for Tracking

```python
# Good: Smooth gradient, bounded output
error = torch.sum(torch.square(target - current), dim=1)
reward = torch.exp(-error / sigma)

# Avoid: Unbounded output, harsh gradients
reward = -error  # Can grow arbitrarily large
```

### 2. Normalize Reward Magnitudes

Keep rewards in similar ranges to avoid dominance:

```python
# Good: Bounded in [0, 1]
reward = torch.exp(-error / sigma)

# If using penalties, scale appropriately
reward = error * small_scale  # e.g., 0.0001
```

### 3. Use Command Conditioning

Apply rewards only when relevant:

```python
# Only reward tracking when there's a non-zero command
tracking_reward *= (torch.norm(self.commands[:, :2], dim=1) > 0.1)

# Only penalize motion when standing still
stand_still_penalty *= (torch.norm(self.commands[:, :3], dim=1) < 0.2)
```

### 4. Avoid Reward Hacking

Design rewards that don't have easy exploits:

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

### 5. Maintain State Carefully

If your reward needs state tracking:

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

## Troubleshooting

### Reward Not Being Called

**Symptom:** Custom reward not affecting training.

**Causes:**
1. Scale is zero or missing from config
2. Method name doesn't match config key
3. Method not in the correct class

**Solution:**
```python
# Check that your method exists
assert hasattr(self, '_reward_my_custom'), "Method not found!"

# Check that scale is configured
assert 'my_custom' in self.reward_scales, "Scale not configured!"
```

### NaN Rewards

**Symptom:** Training fails with NaN loss.

**Causes:**
1. Division by zero
2. Log of zero or negative
3. Numerical overflow

**Solution:**
```python
# Add epsilon to divisions
reward = value / (denominator + 1e-8)

# Clamp values before log
reward = torch.log(torch.clamp(value, min=1e-8))

# Check for NaN and replace
reward = torch.nan_to_num(reward, nan=0.0)
```

### Unbalanced Rewards

**Symptom:** One reward dominates total reward.

**Causes:**
1. Scale too large
2. Reward magnitude too large
3. Exponential kernel with wrong sigma

**Solution:**
```python
# Normalize reward output
reward = torch.exp(-error / sigma)  # Bounds to [0, 1]

# Use smaller scales
my_reward_scale = 0.01  # Instead of 1.0

# Monitor reward statistics
print(f"Reward range: [{reward.min()}, {reward.max()}]")
```

---

## Summary

Creating custom rewards in LeggedGym-Ex follows a simple pattern:

1. **Implement** a method named `_reward_<name>()` returning shape `(num_envs,)`
2. **Configure** the scale in `cfg.rewards.scales.<name>`
3. **Debug** using episode statistics and debug prints
4. **Tune** scales empirically based on training performance

The auto-discovery mechanism handles integration automatically, allowing you to focus on designing effective reward functions for your specific locomotion task.
