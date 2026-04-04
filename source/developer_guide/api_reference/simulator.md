# Simulator API Reference

The Simulator abstraction layer provides a unified interface for running legged robot reinforcement learning across multiple physics simulators. This document covers the abstract base class and its three implementations: Genesis, IsaacGym, and IsaacLab.

## Overview

LeggedGym-Ex supports three simulators through a common abstract interface:

| Simulator | Asset Format | Key Features |
|-----------|--------------|--------------|
| Genesis | MJCF/XML | Fluid/soft material support, fast training |
| IsaacGym | URDF | Fastest training, warp camera support |
| IsaacLab | URDF | Best rendering, IsaacSim integration |

## Simulator Selection

The simulator backend is selected via the `SIMULATOR` environment variable:

```bash
# Use Genesis simulator
export SIMULATOR=genesis

# Use IsaacGym simulator  
export SIMULATOR=isaacgym

# Use IsaacLab/IsaacSim simulator
export SIMULATOR=isaaclab
```

The framework automatically imports the appropriate simulator implementation based on this environment variable during initialization.

```{note}
The simulator selection must be set before importing any legged_gym modules. The `SIMULATOR` variable is read at import time and cannot be changed during runtime.
```

## Simulator Abstract Base Class

The `Simulator` abstract base class defines the interface that all simulator implementations must follow. It is located at `legged_gym/simulator/simulator.py`.

### Constructor

```python
class Simulator(ABC):
    def __init__(self, cfg: LeggedRobotCfg, sim_params: dict, 
                 sim_device: str = "cuda:0", headless: bool = False)
```

**Parameters:**
- `cfg` (`LeggedRobotCfg`): Robot configuration containing asset, control, terrain, and domain randomization settings
- `sim_params` (`dict`): Simulation parameters including dt, substeps, and physics engine settings
- `sim_device` (`str`): Device for simulation (default: "cuda:0")
- `headless` (`bool`): Whether to run without visualization (default: False)

### Core Abstract Methods

#### step()

```python
@abstractmethod
def step(self, actions: Tensor) -> None
```

Performs a simulation step, including applying actions, stepping the physics simulation, and updating internal state buffers. The method runs for `cfg.control.decimation` physics steps per control step.

**Parameters:**
- `actions` (`Tensor`): Action tensor of shape `(num_envs, num_actions)` with values typically in `[-1, 1]`

**Example:**
```python
actions = policy(observation)
simulator.step(actions)
```

#### post_physics_step()

```python
@abstractmethod
def post_physics_step(self) -> None
```

Called after each control step to update derived quantities. This method refreshes state buffers including base position, orientation, velocities, contact forces, and terrain heights.

#### reset_idx()

```python
@abstractmethod
def reset_idx(self, env_ids: Tensor) -> None
```

Resets specified environments to their initial states.

**Parameters:**
- `env_ids` (`Tensor`): Integer tensor of environment IDs to reset

**Example:**
```python
# Reset environments that have terminated
terminated_envs = dones.nonzero(as_tuple=False).flatten()
simulator.reset_idx(terminated_envs)
```

#### reset_dofs()

```python
@abstractmethod
def reset_dofs(self, env_ids: Tensor, dof_pos: Tensor, dof_vel: Tensor) -> None
```

Resets DOF (Degree of Freedom) positions and velocities for specified environments.

**Parameters:**
- `env_ids` (`Tensor`): Environment IDs to reset
- `dof_pos` (`Tensor`): Target DOF positions, shape `(len(env_ids), num_dof)`
- `dof_vel` (`Tensor`): Target DOF velocities, shape `(len(env_ids), num_dof)`

#### reset_root_states()

```python
@abstractmethod
def reset_root_states(self, env_ids: Tensor, base_pos: Tensor, 
                      base_quat: Tensor, base_lin_vel_w: Tensor, 
                      base_ang_vel_w: Tensor) -> None
```

Resets the root (base) link states including position, orientation, and velocities.

**Parameters:**
- `env_ids` (`Tensor`): Environment IDs to reset
- `base_pos` (`Tensor`): Base positions in world frame, shape `(len(env_ids), 3)`
- `base_quat` (`Tensor`): Base orientations as quaternions (xyzw order), shape `(len(env_ids), 4)`
- `base_lin_vel_w` (`Tensor`): Base linear velocities in world frame, shape `(len(env_ids), 3)`
- `base_ang_vel_w` (`Tensor`): Base angular velocities in world frame, shape `(len(env_ids), 3)`

```{warning}
Quaternion order is **xyzw** throughout the framework, matching PyTorch/SciPy conventions. However, Genesis and IsaacLab internally use wxyz order, requiring conversion.
```

#### update_sensors()

```python
@abstractmethod
def update_sensors(self) -> None
```

Updates sensor readings such as depth cameras and lidar sensors. Called after `post_physics_step()` when sensors are configured.

#### update_terrain_curriculum()

```python
@abstractmethod
def update_terrain_curriculum(self, env_ids: Tensor, move_up: Tensor, 
                               move_down: Tensor) -> None
```

Updates terrain difficulty curriculum for specified environments.

**Parameters:**
- `env_ids` (`Tensor`): Environment IDs to update
- `move_up` (`Tensor`): Boolean tensor indicating which envs should increase difficulty
- `move_down` (`Tensor`): Boolean tensor indicating which envs should decrease difficulty

#### push_robots()

```python
@abstractmethod
def push_robots(self) -> None
```

Applies random velocity perturbations to robot bases for domain randomization. The perturbation magnitude is controlled by `cfg.domain_rand.max_push_vel_xy`.

#### push_links()

```python
@abstractmethod
def push_links(self) -> None
```

Applies random forces to robot links for domain randomization. The force magnitude is controlled by `cfg.domain_rand.max_push_force`.

#### draw_debug_vis()

```python
@abstractmethod
def draw_debug_vis(self) -> None
```

Draws debug visualizations such as height sampling points and key body positions. Only active when `cfg.env.debug_*` flags are enabled.

#### set_viewer_camera()

```python
@abstractmethod
def set_viewer_camera(self, eye: np.ndarray, target: np.ndarray) -> None
```

Sets the viewer camera position and orientation.

**Parameters:**
- `eye` (`np.ndarray`): Camera position, shape `(3,)`
- `target` (`np.ndarray`): Point the camera looks at, shape `(3,)`

### State Properties

The Simulator class provides numerous read-only properties for accessing robot states:

| Property | Shape | Description |
|----------|-------|-------------|
| `base_pos` | `(num_envs, 3)` | Base position in world frame |
| `base_quat` | `(num_envs, 4)` | Base orientation (xyzw quaternion) |
| `base_euler` | `(num_envs, 3)` | Base orientation as Euler angles |
| `base_lin_vel` | `(num_envs, 3)` | Base linear velocity in base frame |
| `base_ang_vel` | `(num_envs, 3)` | Base angular velocity in base frame |
| `dof_pos` | `(num_envs, num_dof)` | Joint positions |
| `dof_vel` | `(num_envs, num_dof)` | Joint velocities |
| `feet_pos` | `(num_envs, num_feet, 3)` | Feet positions in world frame |
| `link_contact_forces` | `(num_envs, num_bodies, 3)` | Contact forces on all bodies |
| `measured_heights` | `(num_envs, num_points)` | Terrain heights around robot |
| `torques` | `(num_envs, num_dof)` | Applied joint torques |

### Domain Randomization Properties

Normalized domain randomization parameters are available through properties prefixed with `dr_`:

| Property | Description |
|----------|-------------|
| `dr_friction_values` | Friction coefficient multipliers |
| `dr_restitution_values` | Restitution coefficient multipliers |
| `dr_added_base_mass` | Added base mass (kg) |
| `dr_base_com_bias` | Center of mass displacement |
| `dr_joint_armature` | Joint armature values |
| `dr_kp_scale` | PD controller P gain scales |
| `dr_kd_scale` | PD controller D gain scales |

## GenesisSimulator

The `GenesisSimulator` provides integration with the Genesis physics engine. It is located at `legged_gym/simulator/genesis_simulator.py`.

### Asset Path Convention

Genesis requires **MJCF/XML** format robot assets:

```python
class MyRobotCfg:
    class asset:
        # Required: XML file path for Genesis
        xml_file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/my_robot/robot.xml"
```

```{important}
The `xml_file` parameter is **required** when using Genesis. URDF files are not supported directly. Use MJCF format for robot descriptions.
```

### Key Features

- **Terrain Types**: Supports plane and heightfield terrains. Trimesh terrain is not yet validated.
- **Depth Camera**: Built-in depth camera support via Genesis sensor API
- **Domain Randomization**: Full support for friction, mass, COM, joint properties
- **Quaternion Convention**: Genesis uses **wxyz** order internally; conversion to xyzw is handled automatically

### Example Usage

```python
# Environment variable must be set before import
import os
os.environ["SIMULATOR"] = "genesis"

from legged_gym.envs import MyRobot, MyRobotCfg

cfg = MyRobotCfg()
simulator = GenesisSimulator(cfg, sim_params, device="cuda:0", headless=True)
```

## IsaacGymSimulator

The `IsaacGymSimulator` provides the fastest training performance with NVIDIA's IsaacGym Preview 4. It is located at `legged_gym/simulator/isaacgym_simulator.py`.

### Asset Path Convention

IsaacGym uses **URDF** format robot assets:

```python
class MyRobotCfg:
    class asset:
        # Required: URDF file path
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/my_robot/robot.urdf"
```

### Key Features

- **Terrain Types**: Supports plane, heightfield, and trimesh terrains
- **Depth Camera**: Two modes:
  - IsaacGym native camera sensors (slower)
  - Warp-based ray casting (faster, via `cfg.sensor.use_warp = True`)
- **Warp Camera**: High-performance depth sensing using NVIDIA Warp

### Warp Camera Support

```python
class MyRobotCfg:
    class sensor:
        add_depth = True
        use_warp = True  # Enable warp-based camera
        depth_camera_config:
            resolution = [87, 58]
            near_plane = 0.1
            far_plane = 3.0
```

### IsaacGym Reset Bug

```{warning}
After calling `reset()` in IsaacGym, you must call `simulator.forward()` once before reading rigid body states. Failure to do so results in stale state data.

```python
# Correct pattern
env_ids = terminated.nonzero().flatten()
simulator.reset_idx(env_ids)
simulator._gym.simulate(simulator._sim)  # or equivalent forward call
# Now state tensors are updated
```

This is documented in the G1 DeepMimic implementation at `g1_deepmimic.py:73`.
```

## IsaacLabSimulator

The `IsaacLabSimulator` integrates with NVIDIA IsaacSim via IsaacLab framework. It is located at `legged_gym/simulator/isaaclab_simulator.py`.

### Asset Path Convention

IsaacLab uses **URDF** format robot assets:

```python
class MyRobotCfg:
    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/my_robot/robot.urdf"
```

### Key Features

- **Terrain Types**: Supports plane and trimesh terrains
- **Rendering**: Best visual quality among all simulators
- **Articulation API**: Uses IsaacLab's Articulation class for state management

```{warning}
**Heightfield terrain is NOT implemented** for IsaacLabSimulator. Use trimesh or plane terrain types instead.

```python
# This will raise NotImplementedError
cfg.terrain.mesh_type = "heightfield"  # NOT SUPPORTED
```
```

### Domain Randomization CPU Tensors

```{important}
In IsaacLab, domain randomization methods require tensors on **CPU**, not GPU:

```python
# Correct: Pass CPU tensors
material_props = self._robot.root_physx_view.get_material_properties()
self._robot.root_physx_view.set_material_properties(
    material_props.to('cpu'), 
    all_indices  # Must be on CPU
)
```

This applies to:
- `set_material_properties()` for friction/restitution
- `set_masses()` for mass randomization
- `set_coms()` for COM displacement
```

### Contact Sensor Ordering

IsaacLab's contact sensors have different link ordering than the articulation body ordering. The simulator handles this via separate properties:

```python
feet_indices = simulator.feet_indices  # Articulation body indices
feet_contact_indices = simulator.feet_contact_indices  # Contact sensor indices
```

## Asset Path Summary

| Simulator | Format | Config Parameter | Example |
|-----------|--------|------------------|---------|
| Genesis | MJCF/XML | `xml_file` | `resources/robots/go2/go2.xml` |
| IsaacGym | URDF | `file` | `resources/robots/go2/urdf/go2.urdf` |
| IsaacLab | URDF | `file` | `resources/robots/go2/urdf/go2.urdf` |

## Common Implementation Patterns

### PD Control

All simulators implement PD control in `_compute_torques()`:

```python
def _compute_torques(self, actions):
    actions_scaled = actions * self._cfg.control.action_scale
    
    # Position control (control_type='P')
    torques = self._kp_scale * self._p_gains * \
        (actions_scaled + self._default_dof_pos - self._dof_pos) - \
        self._kd_scale * self._d_gains * self._dof_vel
    
    return torch.clip(torques, -self._torque_limits, self._torque_limits)
```

### Action Delay

Control delay randomization is implemented via an action queue:

```python
def _pre_simulator_step(self, actions):
    if self._cfg.domain_rand.randomize_ctrl_delay:
        self._action_queue[:, 1:] = self._action_queue[:, :-1].clone()
        self._action_queue[:, 0] = actions.clone()
        actions = self._action_queue[torch.arange(self._num_envs), 
                                      self._action_delay].clone()
    return actions
```

### Terrain Height Sampling

Height sampling around the robot base:

```python
def _update_surrounding_heights(self):
    # Transform height sampling points to world frame
    points = quat_apply_yaw(self._base_quat.repeat(1, self._num_height_points), 
                            self._height_points) + self._base_pos.unsqueeze(1)
    
    # Sample from heightfield
    points = (points / self._cfg.terrain.horizontal_scale).long()
    heights = self._height_samples[points[:, :, 0], points[:, :, 1]]
    
    self._measured_heights = heights * self._cfg.terrain.vertical_scale
```

## Debug Visualization

Enable debug flags in configuration to visualize internal states:

```python
class MyRobotCfg:
    class env:
        debug_draw_height_points_around_base = True
        debug_draw_height_points_around_feet = True
        debug_draw_key_body_points = True
        debug_draw_depth_images = True
```

```{note}
Debug visualization significantly slows down simulation. Only enable during debugging, not during training.
```

## Error Handling

Common exceptions and their causes:

| Error | Cause | Solution |
|-------|-------|----------|
| `NotImplementedError` (Genesis) | Trimesh terrain used | Use heightfield instead |
| `NotImplementedError` (IsaacLab) | Heightfield terrain used | Use trimesh or plane |
| `ValueError` | Unknown terrain mesh type | Use 'plane', 'heightfield', or 'trimesh' |
| `NameError` | Unknown controller type | Use 'P', 'V', or 'T' |

## Related Documentation

- {doc}`../parameter_reference/legged_robot_config` - Configuration parameters
- {doc}`../../user_guide/getting_started/installation` - Simulator installation guides
- {doc}`../known_issues/genesis` - Known issues and workarounds
