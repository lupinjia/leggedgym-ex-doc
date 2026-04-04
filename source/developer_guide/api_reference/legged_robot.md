# `legged_robot.py` - LeggedRobot Class API Reference

This document provides comprehensive API documentation for the `LeggedRobot` class, the base environment class for all legged robot tasks in the LeggedGym-Ex framework.

---

## Overview

The `LeggedRobot` class extends `BaseTask` and provides the core functionality for training legged robot locomotion policies using reinforcement learning. It manages simulation environments, handles observations and rewards, processes termination conditions, and orchestrates the training loop.

All robot-specific environments (Go2, G1, K1, TRON1) inherit from this base class, either directly or through specialized variants like `LeggedRobotTS` (Teacher-Student), `LeggedRobotEE` (Explicit Estimator), or `LeggedRobotAMP` (Adversarial Motion Priors).

---

## Type Aliases

The class uses several type aliases for clarity and documentation:

```python
ObsBuf = Tensor  # Shape: (num_envs, obs_dim)
Action = Tensor  # Shape: (num_envs, num_actions)
Reward = Tensor  # Shape: (num_envs,)
EnvIds = Tensor  # Shape: (num_reset_envs,) - integer tensor of environment indices
```

---

## Class Definition

```python
class LeggedRobot(BaseTask):
    """Base environment for legged robot locomotion tasks.
    
    Provides core functionality for:
    - Multi-environment simulation management
    - Observation and reward computation
    - Termination checking and environment resetting
    - Curriculum learning support
    - Domain randomization
    """
```

---

## Initialization

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

Initialize the legged robot environment.

**Parameters:**

- **cfg** (`LeggedRobotCfg`): Environment configuration containing robot, terrain, control, and reward parameters. Must include `env`, `normalization`, `sim`, and `control` sections.
- **sim_params** (`dict[str, Any]`): Dictionary of simulation parameters passed to the simulator backend (IsaacGym, Genesis, or IsaacSim).
- **sim_device** (`str | int`): Device for running the simulation. Can be `'cuda'`, `'cpu'`, or a device ID integer like `0` or `'cuda:0'`.
- **headless** (`bool`): If `True`, run without rendering for faster training.

**Raises:**

- `AssertionError`: If configuration is missing required sections or has invalid values.

**Example:**

```python
from legged_gym.envs.go2.go2_config import GO2Cfg

cfg = GO2Cfg()
sim_params = {"dt": 0.005, "substeps": 1}
env = LeggedRobot(cfg, sim_params, "cuda:0", headless=True)
```

**Validation:**

The initialization performs extensive configuration validation to catch errors early:

- Validates required config sections (`env`, `normalization`, `sim`, `control`)
- Checks observation and action dimensions are positive
- Ensures simulator creates valid number of environments
- Prepares reward functions and initializes buffers

---

## Core Methods

### `step()`

```python
def step(
    self, 
    actions: Action
) -> Tuple[ObsBuf, ObsBuf | None, Reward, Tensor, Dict[str, Any]]
```

Execute one simulation step with the given actions.

Applies actions to the robots, advances the physics simulation, then processes observations, rewards, and termination states. This is the main interface for the RL training loop.

**Parameters:**

- **actions** (`Action`): Action tensor of shape `(num_envs, num_actions)` containing target joint positions or torques depending on control mode. Must be `float32` dtype.

**Returns:**

A tuple containing:
- **obs_buf** (`ObsBuf`): Observation buffer of shape `(num_envs, obs_dim)`.
- **privileged_obs_buf** (`ObsBuf | None`): Privileged observations of shape `(num_envs, privileged_obs_dim)` or `None` if not using asymmetric actor-critic.
- **rew_buf** (`Reward`): Reward buffer of shape `(num_envs,)`.
- **reset_buf** (`Tensor`): Reset flags of shape `(num_envs,)` indicating which environments need reset (1 for reset, 0 otherwise).
- **extras** (`Dict[str, Any]`): Dictionary of additional information including episode statistics and curriculum data.

**Raises:**

- `AssertionError`: If action shape or dtype is incorrect.

**Example:**

```python
# Standard RL training loop
actions = policy(obs)  # Policy network output
obs, priv_obs, rew, done, info = env.step(actions)

# Check for episode statistics
if "episode" in info:
    for key, value in info["episode"].items():
        print(f"{key}: {value}")
```

**Validation:**

- Verifies action shape matches `(num_envs, num_actions)`
- Ensures actions are `float32` dtype
- Clips observations to configured range

---

### `reset_idx()`

```python
def reset_idx(self, env_ids: EnvIds) -> None
```

Reset specified environments to initial states.

Performs complete reset of specified environments including curriculum updates, command resampling, DOF state reset, and buffer resets. This method is called automatically for terminated environments, but can also be called manually.

**Parameters:**

- **env_ids** (`EnvIds`): Integer tensor of shape `(num_reset_envs,)` containing indices of environments to reset. Can be an empty tensor.

**Example:**

```python
# Reset specific environments
env_ids = torch.tensor([0, 5, 10], device="cuda")
env.reset_idx(env_ids)

# Reset all environments
env_ids = torch.arange(env.num_envs, device="cuda")
env.reset_idx(env_ids)
```

**Reset Process:**

1. Updates terrain curriculum if enabled
2. Updates command curriculum if enabled
3. Resamples velocity commands
4. Resets DOF positions with small random perturbations
5. Resets root states (base position, orientation, velocities)
6. Clears action history buffers
7. Resets episode length and failure counters
8. Logs episode statistics to `extras` dict

---

### `reset()`

Inherited from `BaseTask`. Resets all environments.

```python
def reset(self) -> Tuple[ObsBuf, ObsBuf | None]:
    """Reset all environments."""
    env_ids = torch.arange(self.num_envs, device=self.device)
    self.reset_idx(env_ids)
    return self.obs_buf, self.privileged_obs_buf
```

---

### `compute_observations()`

```python
def compute_observations(self) -> None
```

Compute observations for all environments.

Constructs observation tensors from simulator states and applies normalization scales. This method is called automatically after each physics step, but can be overridden to customize observation structure.

**Default Observation Components:**

| Component | Shape | Description | Scale Factor |
|-----------|-------|-------------|--------------|
| Base linear velocity | (3,) | Base velocity in body frame | `obs_scales.lin_vel` |
| Projected gravity | (3,) | Body z-axis in world frame | 1.0 |
| Base angular velocity | (3,) | Angular velocity in body frame | `obs_scales.ang_vel` |
| Commands | (3,) | Linear x, linear y, angular yaw velocities | `commands_scale` |
| DOF positions | (num_dofs,) | Deviation from default position | `obs_scales.dof_pos` |
| DOF velocities | (num_dofs,) | Joint velocities | `obs_scales.dof_vel` |
| Actions | (num_actions,) | Previous action for history | 1.0 |

**Optional Components:**

- **Height measurements**: Added if `cfg.terrain.measure_heights` is `True`. Shape: `(num_height_points,)`, scaled by `obs_scales.height_measurements`.
- **Observation noise**: Added if `cfg.noise.add_noise` is `True`. Uses uniform noise scaled by `noise_scale_vec`.

**Privileged Observations:**

If `num_privileged_obs` is set, computes `privileged_obs_buf` with additional information:

| Component | Shape | Description |
|-----------|-------|-------------|
| Standard observations | - | All components from regular obs |
| Last actions | (num_actions,) | Action two steps ago |
| Friction values | (1,) | Domain randomized friction |
| Added base mass | (1,) | Domain randomized mass |
| Base CoM bias | (3,) | Domain randomized CoM offset |
| Push velocities | (2,) | Domain randomized push velocities |

**Updates:**

- `self.obs_buf`: Shape `(num_envs, obs_dim)`
- `self.privileged_obs_buf`: Shape `(num_envs, privileged_obs_dim)` or `None`

**Example Override:**

```python
class MyRobot(LeggedRobot):
    def compute_observations(self):
        # Call parent to get base observations
        super().compute_observations()
        
        # Add custom observations
        custom_obs = self._compute_custom_features()
        self.obs_buf = torch.cat([self.obs_buf, custom_obs], dim=-1)
```

**Important:**

When overriding this method, ensure the final observation size matches `cfg.env.num_observations`. The class includes validation that raises `AssertionError` if sizes don't match.

---

### `get_observations()`

Inherited from `BaseTask`. Returns current observations without recomputing.

```python
def get_observations(self) -> Tuple[ObsBuf, ObsBuf | None]:
    """Get current observations."""
    return self.obs_buf, self.privileged_obs_buf
```

---

### `check_termination()`

```python
def check_termination(self) -> None
```

Check termination conditions and update reset buffer.

Evaluates three termination conditions for each environment:

1. **Contact termination**: Body contacts with termination bodies exceed force threshold (10.0N).
2. **Orientation termination**: Projected gravity z-component exceeds `cfg.env.max_projected_gravity`.
3. **Timeout termination**: Episode exceeds `max_episode_length`.

**Updates:**

- `self.fail_buf`: Tracks consecutive failures for graceful termination logic.
- `self.time_out_buf`: Indicates episodes that timed out (not actual failures).
- `self.reset_buf`: Indicates environments needing reset.

**Graceful Termination:**

The `fail_buf` counter allows for graceful termination: environments are only reset after failures persist for `cfg.env.fail_to_terminal_time_s / dt` steps, preventing premature termination from brief perturbations.

---

### `check_timeout()`

Implicitly handled in `check_termination()`. Checks if episode length exceeds maximum.

```python
self.time_out_buf = self.episode_length_buf > self.max_episode_length
```

---

### `compute_reward()`

```python
def compute_reward(self) -> None
```

Compute rewards for all environments.

Iterates through all reward functions with non-zero scales (defined in `cfg.rewards.scales`), computes each reward term, scales it by `dt`, and accumulates into the total reward.

**Reward Structure:**

Rewards are defined by the configuration's `scales` dictionary:

```python
cfg.rewards.scales.tracking_lin_vel = 1.0
cfg.rewards.scales.lin_vel_z = -0.5
cfg.rewards.scales.termination = -0.0
```

**Updates:**

- `self.rew_buf`: Total reward of shape `(num_envs,)`.
- `self.episode_sums`: Dictionary tracking cumulative rewards per term for logging.

**Processing:**

1. Zero-scales are removed from consideration
2. Non-zero scales are multiplied by `dt` for proper integration
3. Each reward function `_reward_<name>()` is called
4. Optionally clips rewards to positive values if `cfg.rewards.only_positive_rewards`
5. Adds termination penalty after clipping

---

### `post_physics_step()`

```python
def post_physics_step(self) -> None
```

Process environment state after physics step.

This method orchestrates the RL training loop logic after each simulation step. It's called automatically by `step()` and should not be called directly.

**Execution Order:**

1. Update episode and step counters
2. Process simulator post-physics callbacks
3. Run custom callbacks (`_post_physics_step_callback`)
4. Check termination conditions
5. Compute rewards
6. Reset terminated environments
7. Update sensors
8. Compute observations
9. Draw debug visualizations (if enabled)

---

### `set_viewer_camera()`

```python
def set_viewer_camera(
    self, 
    pos: np.ndarray, 
    lookat: np.ndarray
) -> None
```

Set viewer camera position and orientation for rendering.

**Parameters:**

- **pos** (`np.ndarray`): Camera position in world frame, shape `(3,)`.
- **lookat** (`np.ndarray`): Point to look at in world frame, shape `(3,)`.

**Example:**

```python
# Position camera above and behind robot
pos = np.array([2.0, 0.0, 1.0])
lookat = np.array([0.0, 0.0, 0.5])
env.set_viewer_camera(pos, lookat)
```

---

## Key Attributes

### Observation and State Buffers

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `obs_buf` | `Tensor` | `(num_envs, obs_dim)` | Main observation buffer |
| `privileged_obs_buf` | `Tensor \| None` | `(num_envs, privileged_obs_dim)` | Privileged observations for asymmetric training |
| `reward_buf` | `Tensor` | `(num_envs,)` | Reward buffer |
| `reset_buf` | `Tensor` | `(num_envs,)` | Reset flags |
| `time_out_buf` | `Tensor` | `(num_envs,)` | Timeout flags |

### Robot State

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `dof_pos` | `Tensor` | `(num_envs, num_dofs)` | Joint positions via `simulator.dof_pos` |
| `dof_vel` | `Tensor` | `(num_envs, num_dofs)` | Joint velocities via `simulator.dof_vel` |
| `dof_acc` | `Tensor` | `(num_envs, num_dofs)` | Joint accelerations (computed) |
| `base_pos` | `Tensor` | `(num_envs, 3)` | Base position in world frame |
| `base_quat` | `Tensor` | `(num_envs, 4)` | Base orientation quaternion (x, y, z, w) |
| `base_lin_vel` | `Tensor` | `(num_envs, 3)` | Base linear velocity in world frame |
| `base_ang_vel` | `Tensor` | `(num_envs, 3)` | Base angular velocity in body frame |

### Control and Actions

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `actions` | `Tensor` | `(num_envs, num_actions)` | Current actions |
| `last_actions` | `Tensor` | `(num_envs, num_actions)` | Actions from previous step |
| `llast_actions` | `Tensor` | `(num_envs, num_actions)` | Actions from two steps ago |
| `commands` | `Tensor` | `(num_envs, num_commands)` | Velocity commands |

### Episode Tracking

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `episode_length_buf` | `Tensor` | `(num_envs,)` | Current episode length in steps |
| `fail_buf` | `Tensor` | `(num_envs,)` | Consecutive failure counter |
| `episode_sums` | `Dict[str, Tensor]` | - | Cumulative rewards per term |
| `extras` | `Dict[str, Any]` | - | Additional episode information |

---

## Reward Auto-Discovery Mechanism

The `LeggedRobot` class implements a powerful auto-discovery mechanism for reward functions. This allows you to define reward terms by simply implementing methods with the naming convention `_reward_<name>()`.

### How It Works

1. **Configuration**: Define reward scales in your config:

```python
class MyRobotCfg(LeggedRobotCfg):
    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            lin_vel_z = -0.5
            orientation = -0.5
            termination = -0.0
```

2. **Implementation**: Implement corresponding methods:

```python
class MyRobot(LeggedRobot):
    def _reward_tracking_lin_vel(self) -> Reward:
        """Reward for tracking linear velocity commands."""
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.simulator.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self) -> Reward:
        """Penalize z-axis linear velocity."""
        return torch.square(self.simulator.base_lin_vel[:, 2])
```

3. **Auto-Discovery**: During initialization, `_prepare_reward_function()` automatically:
   - Finds all reward scales with non-zero values
   - Looks for corresponding `_reward_<name>()` methods
   - Validates that methods exist
   - Creates a list of callable reward functions
   - Multiplies scales by `dt` for proper integration

### Built-in Reward Functions

The base class provides many reward functions ready to use:

| Method | Description | Typical Use |
|--------|-------------|-------------|
| `_reward_tracking_lin_vel()` | Track linear velocity commands | Velocity tracking tasks |
| `_reward_tracking_ang_vel()` | Track angular velocity commands | Turning behavior |
| `_reward_lin_vel_z()` | Penalize vertical velocity | Stable locomotion |
| `_reward_ang_vel_xy()` | Penalize roll/pitch rates | Stable locomotion |
| `_reward_orientation()` | Penalize non-flat base | Stability |
| `_reward_base_height()` | Maintain target height | Posture control |
| `_reward_torques()` | Penalize large torques | Energy efficiency |
| `_reward_dof_vel()` | Penalize joint velocities | Smooth motion |
| `_reward_dof_acc()` | Penalize joint accelerations | Smooth motion |
| `_reward_action_rate()` | Penalize action changes | Smooth control |
| `_reward_collision()` | Penalize body collisions | Safety |
| `_reward_feet_air_time()` | Reward proper swing phase | Gait quality |
| `_reward_dof_pos_limits()` | Penalize joint limits | Safety |
| `_reward_torque_limits()` | Penalize torque limits | Hardware safety |

### Custom Rewards

To add custom rewards:

```python
class MyRobot(LeggedRobot):
    def _reward_custom_balance(self) -> Reward:
        """Custom reward for balance."""
        # Implement your reward logic
        balance_error = torch.abs(self.simulator.projected_gravity[:, 2] - 1.0)
        return torch.exp(-balance_error)
```

Then add to config:

```python
class rewards:
    class scales:
        custom_balance = 0.5
```

### Validation

The system includes validation that raises `AssertionError` if:
- No reward scales are defined
- A reward scale is defined but no corresponding method exists
- Method doesn't return correct shape `(num_envs,)`

---

## Callback Methods (Protected)

These methods can be overridden to customize behavior:

### `_pre_sim_step()`

```python
def _pre_sim_step(self, actions: Action) -> Action
```

Called at the beginning of `step()`, before simulation. Default implementation clips actions and updates action history.

### `_post_physics_step_callback()`

```python
def _post_physics_step_callback(self) -> None
```

Called after physics step, before termination checks. Default implementation:
- Resamples commands periodically
- Handles heading commands
- Applies random pushes to robots

**Override Example:**

```python
class MyRobot(LeggedRobot):
    def _post_physics_step_callback(self):
        # Call parent first
        super()._post_physics_step_callback()
        
        # Add custom logic
        self._update_custom_curriculum()
```

### `_resample_commands()`

```python
def _resample_commands(self, env_ids: EnvIds) -> None
```

Resample velocity commands for specified environments. Called during reset and periodically during episodes.

### `_reset_dofs()`

```python
def _reset_dofs(self, env_ids: EnvIds) -> None
```

Reset DOF states for specified environments. Default adds small random perturbations to default positions.

### `_reset_root_states()`

```python
def _reset_root_states(self, env_ids: EnvIds) -> None
```

Reset root states (base position, orientation, velocities) for specified environments.

### `_update_terrain_curriculum()`

```python
def _update_terrain_curriculum(self, env_ids: EnvIds) -> None
```

Update terrain difficulty based on robot performance. Implements game-inspired curriculum where robots that travel far advance to harder terrains.

### `_update_command_curriculum()`

```python
def _update_command_curriculum(self, env_ids: EnvIds) -> None
```

Increase command difficulty based on tracking performance.

---

## Common Override Patterns

### Custom Observations

```python
class MyRobot(LeggedRobot):
    def compute_observations(self):
        # Get base observations
        super().compute_observations()
        
        # Add custom sensor data
        sensor_data = self._get_sensor_readings()  # Shape: (num_envs, sensor_dim)
        self.obs_buf = torch.cat([self.obs_buf, sensor_data], dim=-1)
        
        # Update noise vector if needed
        if self.add_noise:
            self.noise_scale_vec = torch.cat([
                self.noise_scale_vec,
                torch.zeros(self.num_envs, sensor_dim, device=self.device)
            ], dim=-1)
```

**Important**: When overriding `compute_observations()`, you must also update `cfg.env.num_observations` to match the new observation dimension.

### Custom Termination

```python
class MyRobot(LeggedRobot):
    def check_termination(self):
        # Call parent for default termination conditions
        super().check_termination()
        
        # Add custom termination
        custom_failure = self._check_custom_failure_condition()
        self.fail_buf += custom_failure
        
        # Update reset buffer
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )
```

### Custom Reward

```python
class MyRobot(LeggedRobot):
    def _reward_gait_quality(self) -> Reward:
        """Reward smooth, natural gait patterns."""
        # Compute phase-based gait metric
        gait_error = self._compute_gait_phase_error()
        return torch.exp(-gait_error / 0.1)
```

### Domain Randomization Customization

```python
class MyRobot(LeggedRobot):
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        
        # Add custom randomization
        if self.common_step_counter % 1000 == 0:
            self._randomize_custom_parameters()
```

---

## Best Practices

### Configuration Validation

The base class includes extensive validation. Always ensure:
- Config has required sections: `env`, `normalization`, `sim`, `control`
- `num_observations` matches actual observation size
- `num_actions` matches DOF count
- Reward scales have corresponding `_reward_*()` methods

### Observation Modifications

When modifying observations:
1. Update `cfg.env.num_observations`
2. Update `_get_noise_scale_vec()` if using noise
3. Test with assertions enabled to catch size mismatches early

### Reward Function Implementation

- Always return shape `(num_envs,)`
- Use exponential rewards for tracking tasks: `torch.exp(-error/sigma)`
- Use squared penalties for regularization: `torch.square(quantity)`
- Document reward purpose and parameters

### Performance Considerations

- Use vectorized operations instead of loops
- Avoid Python control flow in reward functions
- Cache frequently accessed attributes
- Use `torch.no_grad()` for non-training computations

---

## See Also

- {doc}`../parameter_reference/legged_robot_config` - Configuration parameters
- {doc}`../how_to/add_new_robot` - Guide on adding new robots
- {doc}`../how_to/custom_rewards` - Guide on custom reward functions
