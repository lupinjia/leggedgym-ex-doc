# Adding a New Robot to LeggedGym-Ex

This guide walks you through the complete process of adding a new robot to the LeggedGym-Ex framework. By following these steps, you'll be able to train locomotion policies for any custom quadrupedal or bipedal robot.

```{note}
This guide assumes you have a working installation of LeggedGym-Ex. If not, please refer to the installation instructions first.
```

## Overview

Adding a new robot involves six key steps:

1. **Create Robot Class** - Inherit from `LeggedRobot` and customize behavior
2. **Create Configuration Class** - Define robot parameters, rewards, and training settings
3. **Define Asset Paths** - Configure URDF/XML paths for each simulator
4. **Register the Task** - Add to the task registry
5. **Test Training** - Verify the setup with a minimal training run
6. **Tune Parameters** - Refine rewards, commands, and domain randomization

```{tip}
The fastest way to add a new robot is to copy an existing robot implementation and modify it. The Go2 implementation is a good starting point for quadrupeds, while G1 works well for bipeds.
```

---

## Step 1: Create Robot Class

### 1.1 Directory Structure

Create a new directory for your robot under `legged_gym/envs/`:

```
legged_gym/envs/
├── myrobot/
│   ├── __init__.py
│   ├── myrobot.py           # Robot environment class
│   └── myrobot_config.py    # Configuration class
```

### 1.2 Basic Robot Class

Create `myrobot.py` with a class inheriting from `LeggedRobot`:

```python
# legged_gym/envs/myrobot/myrobot.py
from legged_gym import *
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float

class MyRobot(LeggedRobot):
    """Custom robot environment extending LeggedRobot.
    
    This class handles:
    - DOF state resets
    - Observation computation
    - Root state resets
    - Noise scaling for observations
    """
    
    def _reset_dofs(self, env_ids):
        """Reset DOF positions and velocities for specified environments.
        
        DOF positions are initialized near default standing pose with small
        random offsets to encourage robustness.
        
        Args:
            env_ids: Tensor of environment indices to reset
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
        
        # Initialize near default positions with randomization
        # Adjust indices based on your robot's DOF structure
        dof_pos[:, :] = self.simulator.default_dof_pos[:] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_actions), self.device)
        
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def compute_observations(self):
        """Compute observations for all environments.
        
        The observation vector includes:
        - Commands (3): lin_vel_x, lin_vel_y, ang_vel_yaw
        - Projected gravity (3): body z-axis in world frame
        - Base angular velocity (3): scaled by obs_scales.ang_vel
        - DOF positions (num_dofs): deviation from default
        - DOF velocities (num_dofs): scaled by obs_scales.dof_vel
        - Actions (num_actions): previous action for history
        
        [NOTE]: Must update _get_noise_scale_vec() when changing observations
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
        
        # Add height measurements if enabled
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5 - 
                self.measured_heights, -1, 1.
            ) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        # Add noise if enabled
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        # Compute privileged observations for asymmetric actor-critic
        if self.num_privileged_obs is not None:
            self._compute_privileged_observations()
    
    def _compute_privileged_observations(self):
        """Compute privileged observations for critic network.
        
        Includes domain randomization parameters that are not available
        on real robots:
        - Friction coefficient
        - Added base mass
        - CoM displacement
        - External push velocities
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
        """Set noise scales for observation corruption.
        
        [NOTE]: Must be updated when observation structure changes!
        The indices here must match the observation order in compute_observations().
        
        Returns:
            Tensor of noise scales for each observation dimension
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # Match indices to observation structure
        noise_vec[:3] = 0.  # commands (no noise)
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.  # actions (no noise)
        
        if self.cfg.terrain.measure_heights:
            height_start = 9 + 3 * self.num_actions
            noise_vec[height_start:] = noise_scales.height_measurements * \
                noise_level * self.obs_scales.height_measurements
        
        return noise_vec
    
    def _reset_root_states(self, env_ids):
        """Reset base position and velocity for specified environments.
        
        Args:
            env_ids: Tensor of environment indices to reset
        """
        # Base position
        if self.simulator.custom_origins:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
            base_pos[:, :2] += torch_rand_float(
                -0.5, 0.5, (len(env_ids), 2), device=self.device
            )
        else:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
        
        # Base orientation (quaternion)
        base_quat = self.simulator.base_init_quat.reshape(1, -1).repeat(len(env_ids), 1)
        
        # Base velocities (small random initial velocities)
        base_lin_vel = torch_rand_float(-0.0, 0.0, (len(env_ids), 3), self.device)
        base_ang_vel = torch_rand_float(-0.0, 0.0, (len(env_ids), 3), self.device)
        
        self.simulator.reset_root_states(
            env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel
        )
```

### 1.3 Key Methods to Override

| Method | Purpose | Override When |
|--------|---------|---------------|
| `_reset_dofs()` | Initialize joint positions | Custom standing pose needed |
| `compute_observations()` | Build observation vector | Different sensor suite |
| `_get_noise_scale_vec()` | Define observation noise | Observation structure changes |
| `_reset_root_states()` | Initialize base pose | Custom spawn behavior |

```{warning}
If you modify `compute_observations()`, you MUST update `_get_noise_scale_vec()` to match the new observation indices. Mismatched indices cause silent training failures!
```

---

## Step 2: Create Configuration Class

### 2.1 Basic Configuration

Create `myrobot_config.py` with nested configuration classes:

```python
# legged_gym/envs/myrobot/myrobot_config.py
from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import get_simulator_suffix

class MyRobotCfg(LeggedRobotCfg):
    """Configuration for MyRobot locomotion training.
    
    Inherits from LeggedRobotCfg and overrides specific parameters.
    """
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096              # Number of parallel environments
        num_observations = 45        # Observation dimension (adjust based on your robot)
        num_privileged_obs = None    # Set to int for asymmetric training
        num_actions = 12             # Number of actuated joints
        episode_length_s = 20        # Episode duration in seconds
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]        # Initial spawn position [x, y, z] in meters
        default_joint_angles = {     # Default standing pose
            'FL_hip_joint': 0.0,
            'FL_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,
            # ... add all joints
        }
    
    class control(LeggedRobotCfg.control):
        control_type = 'P'           # Position control
        stiffness = {'joint': 20.0}  # PD stiffness [N*m/rad]
        damping = {'joint': 0.5}     # PD damping [N*m*s/rad]
        action_scale = 0.25          # Action to angle mapping
        dt = 0.02                    # Control frequency: 50 Hz
        decimation = 4               # Sim steps per control step
    
    class asset(LeggedRobotCfg.asset):
        name = "myrobot"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot.xml'
        foot_name = "foot"           # Substring to identify foot links
        penalize_contacts_on = ["thigh", "calf"]  # Penalize these contacts
        terminate_after_contacts_on = ["base"]    # Terminate on these contacts
        base_link_name = "base"
        dof_names = [                # Joint names in action order
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        dof_armature = [0.01] * 12   # Joint armature for stability
    
    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.25        # Tracking reward bandwidth
        base_height_target = 0.36    # Target standing height [m]
        
        class scales(LeggedRobotCfg.rewards.scales):
            # Penalty rewards (negative)
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
            
            # Positive rewards
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            feet_air_time = 1.0
            foot_clearance = 0.5
    
    class commands(LeggedRobotCfg.commands):
        curriculum = True            # Increase command difficulty over time
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0       # Command change interval [s]
        heading_command = True       # Use heading instead of angular velocity
        
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]  # [min, max] m/s
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
    """PPO training configuration for MyRobot."""
    
    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'simple_rl' + get_simulator_suffix()
        experiment_name = 'myrobot'
        save_interval = 200
        max_iterations = 1500
```

### 2.2 Configuration Hierarchy

The configuration system uses nested class inheritance:

```
LeggedRobotCfg
├── env          # Environment settings
├── terrain      # Terrain configuration
├── init_state   # Initial robot state
├── control      # Control parameters
├── asset        # Robot asset paths
├── rewards      # Reward function settings
├── commands     # Command velocity ranges
├── domain_rand  # Domain randomization
├── normalization # Observation/action scaling
├── noise        # Observation noise
└── sim          # Simulation parameters
```

### 2.3 Key Configuration Parameters

#### Environment Settings

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `num_envs` | Parallel environments | 4096 (GPU dependent) |
| `num_observations` | Observation dimension | Robot-specific |
| `num_actions` | Actuated joints | 12 (quadruped), 12+ (biped) |
| `episode_length_s` | Episode duration | 10-20 seconds |

#### Control Settings

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `control_type` | Control mode | 'P' (position), 'V' (velocity), 'T' (torque) |
| `stiffness` | PD stiffness | 20-100 N*m/rad |
| `damping` | PD damping | 0.5-2.0 N*m*s/rad |
| `action_scale` | Action magnitude | 0.25-0.5 |
| `dt` | Control period | 0.02s (50Hz) |
| `decimation` | Sim steps per control | 4 (for 200Hz sim) |

#### Reward Scaling Strategy

```{tip}
Start with balanced reward scales and tune based on training curves:
1. Command tracking rewards (positive): 0.5-2.0
2. Smoothness penalties (negative): -0.001 to -0.1
3. Safety penalties (negative): -1.0 to -10.0
```

---

## Step 3: Define Asset Paths for Each Simulator

### 3.1 Multi-Simulator Support

LeggedGym-Ex supports three simulators, each with different asset requirements:

| Simulator | Asset Format | Configuration Key |
|-----------|--------------|-------------------|
| IsaacGym | URDF | `asset.file` |
| IsaacLab | URDF | `asset.file` |
| Genesis | XML | `asset.xml_file` |

### 3.2 Directory Structure for Assets

Place your robot assets in the resources directory:

```
resources/robots/
└── myrobot/
    ├── myrobot.urdf        # URDF for IsaacGym/IsaacLab
    ├── myrobot.xml         # XML for Genesis
    └── meshes/             # Visual and collision meshes
        ├── base.stl
        ├── thigh.stl
        └── ...
```

### 3.3 Configuration Example

```python
class asset(LeggedRobotCfg.asset):
    name = "myrobot"
    
    # URDF path for IsaacGym and IsaacLab
    file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot.urdf'
    
    # XML path for Genesis
    xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/myrobot/myrobot.xml'
    
    # Additional Genesis-specific settings
    links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    dof_vel_limits = [30.1, 30.1, 15.7] * 4  # rad/s for each joint
```

### 3.4 URDF Requirements

```{important}
Your URDF must satisfy these requirements:
1. All joints must be revolute (no prismatic joints for legs)
2. Joint limits must be defined
3. Inertial parameters must be realistic
4. Mesh files must be referenced with relative paths
```

Example URDF structure:

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
  <!-- ... more links and joints ... -->
</robot>
```

### 3.5 Genesis XML Requirements

Genesis requires a MJCF-style XML file:

```xml
<mujoco model="myrobot">
  <compiler angle="radian"/>
  
  <asset>
    <mesh name="base_mesh" file="meshes/base.stl"/>
    <!-- ... more meshes ... -->
  </asset>
  
  <worldbody>
    <body name="base" pos="0 0 0">
      <inertial mass="10.0" diaginertia="0.1 0.1 0.05"/>
      <geom type="mesh" mesh="base_mesh"/>
      
      <body name="FL_hip" pos="0.2 0.1 0">
        <joint name="FL_hip_joint" type="hinge" axis="0 0 1" 
               range="-0.5 0.5"/>
        <!-- ... more nested bodies ... -->
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="FL_hip_motor" joint="FL_hip_joint" gear="1"/>
    <!-- ... more actuators ... -->
  </actuator>
</mujoco>
```

---

## Step 4: Register in envs/__init__.py

### 4.1 Import and Register

Add your robot to the task registry in `legged_gym/envs/__init__.py`:

```python
# legged_gym/envs/__init__.py

# ... existing imports ...

# MyRobot - Add these lines
from legged_gym.envs.myrobot.myrobot import MyRobot
from legged_gym.envs.myrobot.myrobot_config import MyRobotCfg, MyRobotCfgPPO

# ... in the registration section ...
task_registry.register("myrobot", MyRobot, MyRobotCfg(), MyRobotCfgPPO())
```

### 4.2 Registration Parameters

```python
task_registry.register(
    task_name="myrobot",      # Unique task identifier
    env_class=MyRobot,        # Environment class
    env_cfg=MyRobotCfg(),     # Environment config (instantiated)
    train_cfg=MyRobotCfgPPO() # Training config (instantiated)
)
```

### 4.3 Naming Conventions

Follow these conventions for task names:

| Pattern | Example | Use Case |
|---------|---------|----------|
| `{robot}` | `go2` | Basic locomotion |
| `{robot}_ts` | `go2_ts` | Teacher-Student |
| `{robot}_ee` | `go2_ee` | Explicit Estimator |
| `{robot}_wtw` | `go2_wtw` | Walk These Ways |
| `{robot}_deepmimic` | `g1_deepmimic` | DeepMimic |
| `{robot}_amp` | `k1_amp` | Adversarial Motion Priors |

---

## Step 5: Test with Minimal Training Run

### 5.1 Quick Verification

Before full training, verify your setup:

```bash
# Set the simulator (choose one)
export SIMULATOR=genesis
# export SIMULATOR=isaacgym
# export SIMULATOR=isaaclab

# Run a short training test
python -m legged_gym.scripts.train --task=myrobot --headless --max_iterations=10
```

### 5.2 Expected Output

A successful test should show:

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

### 5.3 Common Verification Checklist

- [ ] Environment initializes without errors
- [ ] Robot spawns at correct height
- [ ] Observations have expected shape
- [ ] Rewards are being computed
- [ ] Robot doesn't immediately fall over
- [ ] No CUDA memory errors

### 5.4 Debug Mode

For visual debugging:

```bash
# Run without headless to see the robot
python -m legged_gym.scripts.train --task=myrobot --num_envs=16
```

Enable debug visualization in config:

```python
class env(LeggedRobotCfg.env):
    debug = True
    debug_draw_height_points_around_base = True
```

---

## Step 6: Tune Rewards and Commands

### 6.1 Reward Tuning Process

1. **Start simple**: Begin with command tracking rewards only
2. **Add smoothness**: Penalize jerky motions
3. **Add safety**: Penalize collisions and limits
4. **Iterate**: Adjust scales based on observed behavior

### 6.2 Common Reward Functions

```python
# In your config
class rewards(LeggedRobotCfg.rewards):
    class scales(LeggedRobotCfg.rewards.scales):
        # Command tracking (encourage goal achievement)
        tracking_lin_vel = 1.0     # Track x/y velocity commands
        tracking_ang_vel = 0.5     # Track yaw velocity command
        
        # Smoothness (penalize erratic behavior)
        lin_vel_z = -2.0           # Penalize vertical velocity
        ang_vel_xy = -0.05         # Penalize roll/pitch rates
        dof_acc = -2.5e-7          # Penalize joint acceleration
        action_rate = -0.01        # Penalize action changes
        
        # Energy efficiency
        torques = -1e-4            # Penalize large torques
        dof_power = -2e-4          # Penalize power consumption
        
        # Safety
        collision = -1.0           # Penalize body collisions
        dof_pos_limits = -10.0     # Penalize joint limit violations
        termination = -0.0         # Penalty for termination
        
        # Gait quality
        feet_air_time = 1.0        # Reward proper swing phase
        foot_clearance = 0.5       # Reward foot lift during swing
```

### 6.3 Command Tuning

```python
class commands(LeggedRobotCfg.commands):
    curriculum = True              # Enable progressive difficulty
    max_curriculum = 1.5           # Maximum velocity multiplier
    
    class ranges(LeggedRobotCfg.commands.ranges):
        # Start conservative, curriculum will expand
        lin_vel_x = [-0.3, 0.3]    # Forward/backward
        lin_vel_y = [-0.3, 0.3]    # Lateral
        ang_vel_yaw = [-0.5, 0.5]  # Turning
```

### 6.4 Domain Randomization Tuning

```python
class domain_rand(LeggedRobotCfg.domain_rand):
    # Physics randomization
    randomize_friction = True
    friction_range = [0.5, 1.5]    # Wider for robustness
    
    randomize_base_mass = True
    added_mass_range = [-2.0, 2.0] # kg variation
    
    # External disturbances
    push_robots = True
    push_interval_s = 10
    max_push_vel_xy = 0.5          # m/s push velocity
    
    # Modeling errors
    randomize_com_displacement = True
    com_pos_x_range = [-0.02, 0.02]
    com_pos_y_range = [-0.02, 0.02]
    com_pos_z_range = [-0.01, 0.01]
```

### 6.5 Monitoring Training Progress

Use tensorboard or wandb:

```bash
# View training curves
tensorboard --logdir logs/myrobot/

# Or enable wandb in config
class runner(LeggedRobotCfgPPO.runner):
    sync_wandb = True
```

Key metrics to monitor:

| Metric | Target | Action if Poor |
|--------|--------|----------------|
| `rew_tracking_lin_vel` | > 0.8 | Increase scale or reduce velocity range |
| `rew_tracking_ang_vel` | > 0.7 | Increase scale or reduce yaw range |
| `episode_length` | Near max | Check termination conditions |
| `terrain_level` | Increasing | Training is progressing |

---

## Working Example: Clone Go2 → MyRobot

This section provides a complete working example showing how to create a new robot by cloning and modifying the Go2 implementation.

### Step-by-Step Clone Process

```bash
# 1. Create directory structure
mkdir -p legged_gym/envs/myrobot

# 2. Copy Go2 files as template
cp legged_gym/envs/go2/go2.py legged_gym/envs/myrobot/myrobot.py
cp legged_gym/envs/go2/go2_config.py legged_gym/envs/myrobot/myrobot_config.py

# 3. Create __init__.py
touch legged_gym/envs/myrobot/__init__.py
```

### Modified myrobot.py

```python
# legged_gym/envs/myrobot/myrobot.py
from legged_gym import *
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float

class MyRobot(LeggedRobot):
    """Custom quadruped robot based on Go2 template."""
    
    def _reset_dofs(self, env_ids):
        """Reset DOFs with MyRobot-specific initialization."""
        dof_pos = torch.zeros((len(env_ids), self.num_actions), 
                              dtype=torch.float, device=self.device)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), 
                              dtype=torch.float, device=self.device)
        
        # Customize for MyRobot's joint structure
        # Assuming similar FR, FL, RR, RL ordering
        hip_indices = [0, 3, 6, 9]
        thigh_indices = [1, 4, 7, 10]
        calf_indices = [2, 5, 8, 11]
        
        # Hips: small randomization around 0
        dof_pos[:, hip_indices] = self.simulator.default_dof_pos[:, hip_indices] + \
            torch_rand_float(-0.1, 0.1, (len(env_ids), 4), self.device)
        
        # Thighs: larger randomization
        dof_pos[:, thigh_indices] = self.simulator.default_dof_pos[:, thigh_indices] + \
            torch_rand_float(-0.3, 0.3, (len(env_ids), 4), self.device)
        
        # Calves: moderate randomization
        dof_pos[:, calf_indices] = self.simulator.default_dof_pos[:, calf_indices] + \
            torch_rand_float(-0.3, 0.3, (len(env_ids), 4), self.device)
        
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    # Use default compute_observations from LeggedRobot
    # or customize as needed
```

### Modified myrobot_config.py

```python
# legged_gym/envs/myrobot/myrobot_config.py
from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import get_simulator_suffix

class MyRobotCfg(LeggedRobotCfg):
    """Configuration for MyRobot."""
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45        # Adjust based on your sensor setup
        num_privileged_obs = None
        num_actions = 12             # Adjust for your robot's DOFs
        episode_length_s = 20
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35]       # Adjust spawn height for your robot
        default_joint_angles = {
            # Customize for your robot's joint names
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
        stiffness = {'joint': 25.0}   # Tune for your robot
        damping = {'joint': 0.5}
        action_scale = 0.25
        dt = 0.02
        decimation = 4
    
    class asset(LeggedRobotCfg.asset):
        name = "myrobot"
        # Point to your URDF/XML files
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
        base_height_target = 0.32    # Adjust for your robot
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

### Register and Test

```python
# Add to legged_gym/envs/__init__.py

from legged_gym.envs.myrobot.myrobot import MyRobot
from legged_gym.envs.myrobot.myrobot_config import MyRobotCfg, MyRobotCfgPPO

task_registry.register("myrobot", MyRobot, MyRobotCfg(), MyRobotCfgPPO())
```

```bash
# Test the new robot
export SIMULATOR=genesis
python -m legged_gym.scripts.train --task=myrobot --headless --max_iterations=100
```

---

## Troubleshooting

### Common Issues and Solutions

#### Robot Falls Immediately After Spawn

**Symptoms**: Robot collapses instantly, episode terminates quickly.

**Solutions**:
1. Check `init_state.pos[2]` - spawn height may be too low
2. Verify `default_joint_angles` match a stable pose
3. Increase `dof_armature` for numerical stability
4. Check URDF inertia values are realistic

```python
# Debug spawn height
class init_state(LeggedRobotCfg.init_state):
    pos = [0.0, 0.0, 0.5]  # Try higher spawn
```

#### Observation Shape Mismatch

**Symptoms**: Error like "expected shape (4096, 45) but got (4096, 48)".

**Solutions**:
1. Count observation dimensions manually
2. Ensure `num_observations` matches `obs_buf.shape[1]`
3. Check if `measure_heights` adds extra dimensions

```python
# Count observations:
# commands (3) + gravity (3) + ang_vel (3) + dof_pos (12) + dof_vel (12) + actions (12) = 45
```

#### CUDA Out of Memory

**Symptoms**: RuntimeError: CUDA out of memory.

**Solutions**:
1. Reduce `num_envs`
2. Reduce `terrain.num_rows` and `num_cols`
3. Disable `measure_heights` if not needed
4. Use smaller observation history

```python
class env(LeggedRobotCfg.env):
    num_envs = 2048  # Reduce from 4096
```

#### Training Not Progressing

**Symptoms**: Rewards plateau at low values, robot doesn't learn to walk.

**Solutions**:
1. Increase `tracking_lin_vel` reward scale
2. Reduce command velocity ranges
3. Check termination conditions aren't too strict
4. Verify domain randomization isn't too aggressive

```python
class commands(LeggedRobotCfg.commands):
    class ranges(LeggedRobotCfg.commands.ranges):
        lin_vel_x = [-0.2, 0.2]  # Start with smaller commands

class rewards(LeggedRobotCfg.rewards):
    class scales(LeggedRobotCfg.rewards.scales):
        tracking_lin_vel = 2.0   # Increase tracking reward
```

#### Joint Limit Violations

**Symptoms**: Robot jerks at joint limits, unnatural motion.

**Solutions**:
1. Check URDF joint limits match real robot
2. Adjust `soft_dof_pos_limit`
3. Increase `dof_pos_limits` penalty

```python
class rewards(LeggedRobotCfg.rewards):
    soft_dof_pos_limit = 0.9  # Penalize at 90% of limit
    class scales(LeggedRobotCfg.rewards.scales):
        dof_pos_limits = -10.0  # Increase penalty
```

#### Genesis-Specific Issues

**Symptoms**: XML loading errors, missing links.

**Solutions**:
1. Ensure `links_to_keep` includes all foot links
2. Check XML uses correct quaternion convention (wxyz)
3. Verify mesh paths are relative to XML location

```python
class asset(LeggedRobotCfg.asset):
    links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    dof_vel_limits = [30.0] * 12  # Required for Genesis
```

#### IsaacGym-Specific Issues

**Symptoms**: Reset bugs, incorrect state after reset.

**Solutions**:
1. Call `simulator.forward()` after reset
2. Check `collapse_fixed_joints` setting
3. Verify `default_dof_drive_mode` is correct

```python
# In _reset_dofs or similar
self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
self.simulator.forward()  # Required for IsaacGym
```

#### IsaacLab-Specific Issues

**Symptoms**: Domain randomization errors, device mismatch.

**Solutions**:
1. Ensure DR tensors are on CPU for IsaacLab
2. Use correct `collapse_fixed_joints` setting
3. Check `measure_heights` compatibility

```python
# IsaacLab requires CPU tensors for DR
# This is handled internally, but be aware
```

### Debugging Commands

```bash
# List all registered tasks
python -c "from legged_gym.envs import task_registry; print(task_registry.task_classes.keys())"

# Check config values
python -c "from legged_gym.envs.myrobot.myrobot_config import MyRobotCfg; cfg = MyRobotCfg(); print(f'obs: {cfg.env.num_observations}, actions: {cfg.env.num_actions}')"

# Test single environment
python -m legged_gym.scripts.train --task=myrobot --num_envs=1 --headless
```

---

## Summary Checklist

Before submitting your new robot implementation:

- [ ] Robot class inherits from `LeggedRobot`
- [ ] All required methods implemented or using defaults
- [ ] Configuration class inherits from `LeggedRobotCfg`
- [ ] Asset paths correct for all simulators
- [ ] Joint names match URDF/XML exactly
- [ ] Default pose is stable
- [ ] Task registered in `__init__.py`
- [ ] Basic training runs without errors
- [ ] Robot learns to walk (even poorly)
- [ ] No CUDA memory issues with default settings

---

## Next Steps

After successfully adding your robot:

1. **Terrain Training**: Add rough terrain for robustness
   ```python
   class terrain(LeggedRobotCfg.terrain):
       mesh_type = "trimesh"
       curriculum = True
       measure_heights = True
   ```

2. **Sim-to-Real**: Prepare for deployment
   - Remove `base_lin_vel` from observations
   - Add observation noise
   - Train with domain randomization

3. **Advanced Methods**: Implement specialized algorithms
   - Teacher-Student for privileged information
   - Explicit Estimator for velocity estimation
   - Walk These Ways for gait control

---

## References

- {doc}`../parameter_reference/legged_robot_config` - Complete parameter reference
- {doc}`../api_reference/legged_robot` - API documentation
- [IsaacGym Documentation](https://developer.nvidia.com/isaac-gym)
- [Genesis Documentation](https://genesis-world.readthedocs.io/)
- [IsaacLab Documentation](https://isaac-sim.github.io/IsaacLab/)
