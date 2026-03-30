# `legged_robot_config.py`

This document provides detailed descriptions of all configuration parameters defined in `legged_robot_config.py`.

---

## LeggedRobotCfg

### `env` Environment Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `num_envs` | int | 4096 | Number of parallel environments |
| `num_observations` | int | 48 | Size of the observation vector |
| `num_privileged_obs` | int | None | Number of privileged observations. If not None, step() returns priviledge_obs_buf (critic obs for asymmetric training), otherwise returns None |
| `num_actions` | int | 12 | Size of the action vector |
| `send_timeouts` | bool | True | Whether to send timeout information to the algorithm |
| `episode_length_s` | float | 20 | Episode length in seconds |
| `env_spacing` | float | 2.0 | Spacing between environments in the scene, only for plane terrain |
| `fail_to_terminal_time_s` | float | 0.1 | Time before a fail state leads to environment reset (seconds) |
| `debug` | bool | False | Whether to enable debug drawings in the simulator |
| `debug_draw_height_points_around_base` | bool | False | Whether to obtain height measurements around the base |
| `debug_draw_height_points_around_feet` | bool | False | Whether to obtain height measurements around the feet (9 points around each foot) |
| `debug_draw_terrain_height_points` | bool | False | Whether to draw all height points of the terrain |
| `debug_draw_key_body_points` | bool | False | Whether to draw key body points for mimic tasks |
| `max_projected_gravity` | float | -0.1 | Maximum allowed projected gravity in z-axis direction |

---

### `terrain` Terrain Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `mesh_type` | str | 'plane' | Terrain mesh type: 'plane', 'heightfield', 'trimesh' |
| `plane_length` | float | 200.0 | Plane size [m], default is 200x200x10 |
| `horizontal_scale` | float | 0.1 | Distance between height samples in x and y direction [m] |
| `vertical_scale` | float | 0.005 | Distance between height samples in z direction [m] |
| `border_size` | float | 5 | Length of the border surrounding the terrain [m] |
| `border_height` | float | 1.0 | Height of the border surrounding the terrain [m] |
| `curriculum` | bool | False | Whether to use terrain curriculum, starting from easier terrains and gradually increasing difficulty |
| `static_friction` | float | 1.0 | Coefficient of static friction of the terrain |
| `dynamic_friction` | float | 1.0 | Coefficient of dynamic friction of the terrain |
| `restitution` | float | 0. | Coefficient of restitution of the terrain |
| `obtain_terrain_info_around_feet` | bool | False | Whether to obtain terrain height information around feet (default: 9 points around each foot) |
| `measure_heights` | bool | False | Whether to obtain height measurements |
| `measured_points_x` | list | [-0.8, ..., 0.8] | X positions of height sampling around the base (relative to robot base), 1m x 1.6m rectangular area |
| `measured_points_y` | list | [-0.5, ..., 0.5] | Y positions of height sampling around the base (relative to robot base) |
| `selected` | bool | False | Whether to select a unique terrain type and pass all arguments |
| `terrain_kwargs` | dict | None | Dictionary of arguments for selected terrain |
| `max_init_terrain_level` | int | 1 | Starting curriculum level |
| `terrain_length` | float | 6.0 | Length of each subterrain [m] (X direction) |
| `terrain_width` | float | 6.0 | Width of each subterrain [m] (Y direction) |
| `platform_size` | float | 3.0 | Size of the flat platform at the center of each subterrain [m] |
| `num_rows` | int | 4 | Number of terrain rows (levels), X direction |
| `num_cols` | int | 4 | Number of terrain columns (types), Y direction |
| `terrain_proportions` | list | [0.1, 0.1, 0.35, 0.25, 0.2] | Terrain type proportions: [smooth slope, rough slope, stairs up, stairs down, discrete] |
| `slope_treshold` | float | 0.75 | Trimesh only: slopes above this threshold will be corrected to vertical surfaces |

---

### `init_state` Initial State Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `pos` | list | [0.0, 0.0, 1.] | Initial position x, y, z [m] |
| `rot` | list | [0.0, 0.0, 0.0, 1.0] | Initial orientation quaternion x, y, z, w. Note: Gym format is xyzw, Genesis format is wxyz |
| `lin_vel` | list | [0.0, 0.0, 0.0] | Initial linear velocity x, y, z [m/s] |
| `ang_vel` | list | [0.0, 0.0, 0.0] | Initial angular velocity x, y, z [rad/s] |
| `roll_random_scale` | float | 0.0 | Roll angle randomization range |
| `pitch_random_scale` | float | 0.0 | Pitch angle randomization range |
| `yaw_random_scale` | float | 0.0 | Yaw angle randomization range |
| `default_joint_angles` | dict | {"joint_a": 0., "joint_b": 0.} | Target joint angles when action = 0.0 |

---

### `control` Control Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `control_type` | str | 'P' | Control type: 'P' (position), 'V' (velocity), 'T' (torque) |
| `stiffness` | dict | {'joint_a': 10.0, 'joint_b': 15.} | PD control stiffness [N*m/rad] |
| `damping` | dict | {'joint_a': 1.0, 'joint_b': 1.5} | PD control damping [N*m*s/rad] |
| `action_scale` | float | 0.5 | Action scale factor, target angle = actionScale * action + defaultAngle |
| `dt` | float | 0.02 | Control frequency, default 50Hz |
| `decimation` | int | 4 | Ratio of control action updates to policy updates |

---

### `asset` Asset Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `name` | str | None | Asset name |
| `file` | str | "" | URDF file path |
| `foot_name` | str | "" | Foot body name, bodies containing this substring will be considered as feet |
| `key_bodies` | list | [] | List of important bodies to be tracked in mimic tasks |
| `penalize_contacts_on` | list | [] | Penalize contacts on links containing these substrings |
| `terminate_after_contacts_on` | list | [] | Terminate episode after contacts on links containing these substrings |
| `fix_base_link` | bool | False | Whether to fix the base link to the world frame |
| `obtain_link_contact_states` | bool | False | Whether to obtain contact states of specific links, can be used for privileged policy |
| `contact_state_link_names` | list | ["thigh", "calf", "foot"] | Link names for obtaining contact states |
| `base_link_name` | str | "" | Full name of the base link |
| `self_collisions` | int | 0 | Self-collision setting: 1 to disable, 0 to enable (bitwise filter) |
| `dof_names` | list | ["joint_a", "joint_b"] | Sequence of DOFs in actions and observations |
| `dof_armature` | list | [0.0] | Armature of each DOF |
| `links_to_keep` | list | [] | Genesis only: links that are not merged due to fixed joints |
| `dof_vel_limits` | list | [] | DOF velocity limits [rad/s], obtained from URDF |
| `disable_gravity` | bool | False | IsaacGym/IsaacLab only: whether to disable gravity |
| `collapse_fixed_joints` | bool | True | IsaacGym/IsaacLab only: whether to merge bodies connected by fixed joints |
| `default_dof_drive_mode` | int | 3 | IsaacGym/IsaacLab only: DOF drive mode (0 none, 1 position target, 2 velocity target, 3 effort) |
| `replace_cylinder_with_capsule` | bool | False | IsaacGym/IsaacLab only: replace collision cylinders with capsules, improves simulation speed and stability |
| `flip_visual_attachments` | bool | False | IsaacGym/IsaacLab only: whether to flip certain .obj meshes from y-up to z-up |
| `density` | float | 0.001 | Density |
| `angular_damping` | float | 0. | Angular damping |
| `linear_damping` | float | 0. | Linear damping |
| `max_angular_velocity` | float | 1000. | Maximum angular velocity |
| `max_linear_velocity` | float | 1000. | Maximum linear velocity |
| `armature` | float | 0. | Armature |
| `thickness` | float | 0.01 | Thickness |

---

### `rewards` Reward Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `only_positive_rewards` | bool | True | Whether to only use positive rewards |
| `tracking_sigma` | float | 0.25 | Tracking reward calculation parameter: reward = exp(-error^2/sigma) |
| `soft_dof_pos_limit` | float | 1. | Joint position soft limit (percentage of URDF limits), values above this limit are penalized |
| `soft_dof_vel_limit` | float | 1. | Joint velocity soft limit |
| `soft_torque_limit` | float | 1. | Torque soft limit |
| `base_height_target` | float | 1. | Target base height |
| `foot_clearance_target` | float | 0.04 | Desired foot clearance above ground [m] |
| `foot_height_offset` | float | 0.0 | Height of the foot coordinate origin above ground [m] |
| `foot_clearance_tracking_sigma` | float | 0.01 | Sigma value for foot clearance tracking |

#### `scales` Reward Weights

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `termination` | float | -0.0 | Termination penalty weight |
| `tracking_lin_vel` | float | 0 | Linear velocity tracking reward weight |
| `tracking_ang_vel` | float | 0 | Angular velocity tracking reward weight |
| `lin_vel_z` | float | 0 | Z-direction linear velocity penalty weight |
| `ang_vel_xy` | float | 0 | XY-plane angular velocity penalty weight |
| `orientation` | float | -0. | Orientation penalty weight |
| `torques` | float | 0 | Torque penalty weight |
| `dof_vel` | float | -0. | Joint velocity penalty weight |
| `dof_acc` | float | 0 | Joint acceleration penalty weight |
| `base_height` | float | -0. | Base height penalty weight |
| `feet_air_time` | float | 0 | Feet air time reward weight |
| `collision` | float | 0 | Collision penalty weight |
| `feet_stumble` | float | -0.0 | Feet stumble penalty weight |
| `action_rate` | float | 0 | Action rate penalty weight |
| `dof_pos_stand_still` | float | -0. | Joint position penalty weight when standing still |

---

### `commands` Command Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `curriculum` | bool | False | Whether to use command curriculum |
| `max_curriculum` | float | 1. | Maximum curriculum level |
| `num_commands` | int | 4 | Number of commands, default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading |
| `resampling_time` | float | 10. | Time interval for command resampling [s] |
| `heading_command` | bool | True | If True: compute angular velocity command from heading error |
| `curriculum_threshold` | float | 0.8 | Curriculum learning threshold, increase command range when tracking reward exceeds this threshold |
| `zero_cmd_prob` | float | 0.4 | Probability of sampling zero command during resampling, encourages robot to learn standing still behavior |

#### `ranges` Command Ranges

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `lin_vel_x` | list | [-1.0, 1.0] | X-direction linear velocity range [m/s] |
| `lin_vel_y` | list | [-1.0, 1.0] | Y-direction linear velocity range [m/s] |
| `ang_vel_yaw` | list | [-1, 1] | Yaw angular velocity range [rad/s] |
| `heading` | list | [-3.14, 3.14] | Heading angle range [rad] |

---

### `domain_rand` Domain Randomization Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `randomize_friction` | bool | True | Whether to randomize rigid body friction coefficient |
| `friction_range` | list | [0.5, 1.25] | Friction coefficient randomization range |
| `randomize_base_mass` | bool | True | Whether to randomize base mass |
| `added_mass_range` | list | [-1., 1.] | Added mass range [kg] |
| `push_robots` | bool | True | Whether to apply random velocity perturbations to the base |
| `push_interval_s` | float | 15 | Time interval for pushing robots [s] |
| `max_push_vel_xy` | float | 1. | Maximum push velocity in XY plane [m/s] |
| `randomize_com_displacement` | bool | True | Whether to randomize center of mass position to simulate modeling errors |
| `com_pos_x_range` | list | [-0.01, 0.01] | CoM x-position randomization range [m] |
| `com_pos_y_range` | list | [-0.01, 0.01] | CoM y-position randomization range [m] |
| `com_pos_z_range` | list | [-0.01, 0.01] | CoM z-position randomization range [m] |
| `randomize_ctrl_delay` | bool | False | Whether to apply random delay to actions to simulate control loop latency |
| `ctrl_delay_step_range` | list | [0, 1] | Control delay step range |
| `randomize_pd_gain` | bool | False | Whether to randomize PD gains by a scale factor |
| `kp_range` | list | [0.8, 1.2] | Kp scale range |
| `kd_range` | list | [0.8, 1.2] | Kd scale range |
| `randomize_joint_armature` | bool | False | Whether to randomize joint armature (significantly slows simulation in Genesis) |
| `joint_armature_range` | list | [0.0, 0.05] | Joint armature range [N*m*s/rad] |
| `randomize_joint_friction` | bool | False | Whether to randomize joint friction |
| `joint_friction_range` | list | [0.0, 0.1] | Joint friction range |
| `randomize_joint_damping` | bool | False | Whether to randomize joint damping |
| `joint_damping_range` | list | [0.0, 1.0] | Joint damping range |
| `push_links` | bool | False | Whether to apply random push forces to robot links |
| `max_push_force` | float | 10.0 | Maximum magnitude of random push force applied to each link [N] |
| `push_links_interval_s` | float | 15.0 | Time interval between random pushes [s] |

---

### `normalization` Normalization Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `clip_observations` | float | 100. | Observation clipping range |
| `clip_actions` | float | 100. | Action clipping range |

#### `obs_scales` Observation Scales

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `lin_vel` | float | 1.0 | Linear velocity observation scale factor |
| `ang_vel` | float | 0.25 | Angular velocity observation scale factor |
| `dof_pos` | float | 1.0 | Joint position observation scale factor |
| `dof_vel` | float | 0.05 | Joint velocity observation scale factor |
| `height_measurements` | float | 5.0 | Height measurement observation scale factor |

---

### `noise` Noise Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `add_noise` | bool | True | Whether to add noise |
| `noise_level` | float | 1.0 | Noise scale factor |

#### `noise_scales` Noise Scales

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `dof_pos` | float | 0.01 | Joint position noise scale |
| `dof_vel` | float | 0.5 | Joint velocity noise scale |
| `lin_vel` | float | 0.1 | Linear velocity noise scale |
| `ang_vel` | float | 0.2 | Angular velocity noise scale |
| `gravity` | float | 0.05 | Gravity observation noise scale |
| `height_measurements` | float | 0.1 | Height measurement noise scale |

---

### `constraints` Constraints Configuration

Used for CaT (Constraints as Termination) method.

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `limits` | class | - | Constraint limits configuration |

---

### `viewer` Viewer Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `ref_env` | int | 0 | Reference environment index |
| `pos` | list | [4.0, 4.0, 2.0] | Camera position relative to robot position [m] |
| `lookat` | list | [0., 0, 0.] | Point the camera looks at (relative to robot position) [m] |
| `rendered_envs_idx` | list | [0, 1, 2, 3, 4] | Genesis only: list of environment indices to render |

---

### `sensor` Sensor Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `add_depth` | bool | False | Whether to add depth camera |
| `use_warp` | bool | False | Whether to use warp-based model |

#### `depth_camera_config` Depth Camera Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `num_sensors` | int | 1 | Number of sensors |
| `num_history` | int | 1 | History frames for depth images |
| `near_clip` | float | 0.1 | Near clipping plane |
| `far_clip` | float | 10.0 | Far clipping plane |
| `near_plane` | float | 0.1 | Near plane |
| `far_plane` | float | 10.0 | Far plane |
| `resolution` | tuple | (80, 60) | Image resolution |
| `horizontal_fov_deg` | float | 75 | Horizontal field of view [degrees] |
| `pos` | tuple | (0.3, 0.0, 0.1) | Camera position |
| `euler` | tuple | (0.0, 0.0, 0.0) | Camera Euler angles |
| `decimation` | int | 5 | Decimation factor |
| `calculate_depth` | bool | True | Warp only: whether to calculate depth |
| `segmentation_camera` | bool | False | Warp only: whether to use segmentation camera |
| `return_pointcloud` | bool | False | Warp only: whether to return point cloud |
| `pointcloud_in_world_frame` | bool | False | Warp only: whether point cloud is in world frame |

---

### `sim` Simulation Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `dt` | float | 0.005 | Simulation timestep, default 200Hz |
| `substeps` | int | 1 | Number of substeps |
| `max_collision_pairs` | int | 100 | Genesis only: maximum number of collision pairs, more collision pairs will occupy more GPU memory and slow down simulation |
| `IK_max_targets` | int | 2 | Genesis only: number of IK targets, fewer targets will reduce memory usage |
| `gravity` | list | [0., 0., -9.81] | IsaacGym only: gravity acceleration [m/s^2] |
| `up_axis` | int | 1 | IsaacGym only: up axis direction, 0 for y, 1 for z |
| `use_gpu_pipeline` | bool | True | IsaacGym only: whether to use GPU pipeline |

#### `physx` PhysX Engine Configuration (IsaacGym only)

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `use_gpu` | bool | True | Whether to use GPU |
| `num_subscenes` | int | 0 | Number of subscenes |
| `num_threads` | int | 10 | Number of threads |
| `solver_type` | int | 1 | Solver type: 0 for pgs, 1 for tgs |
| `num_position_iterations` | int | 4 | Number of position iterations |
| `num_velocity_iterations` | int | 0 | Number of velocity iterations |
| `contact_offset` | float | 0.01 | Contact offset [m] |
| `rest_offset` | float | 0.0 | Rest offset [m] |
| `bounce_threshold_velocity` | float | 0.5 | Bounce threshold velocity [m/s] |
| `max_depenetration_velocity` | float | 1.0 | Maximum depenetration velocity |
| `max_gpu_contact_pairs` | int | 2**23 | Maximum GPU contact pairs |
| `default_buffer_size_multiplier` | int | 5 | Default buffer size multiplier |
| `contact_collection` | int | 2 | Contact collection mode: 0 never, 1 last substep, 2 all substeps |

---

## LeggedRobotCfgPPO

PPO algorithm configuration class.

### Basic Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `seed` | int | 1 | Random seed |
| `runner_class_name` | str | 'OnPolicyRunner' | Runner class name |

---

### `policy` Policy Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `clip_actions` | float | LeggedRobotCfg.normalization.clip_actions | Action clipping range |
| `init_noise_std` | float | 1.0 | Initial noise standard deviation |
| `actor_hidden_dims` | list | [512, 256, 128] | Actor network hidden layer dimensions |
| `critic_hidden_dims` | list | [512, 256, 128] | Critic network hidden layer dimensions |
| `activation` | str | 'elu' | Activation function, options: elu, relu, selu, crelu, lrelu, tanh, sigmoid |

#### RNN Configuration (for ActorCriticRecurrent only)

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `rnn_type` | str | 'lstm' | RNN type |
| `rnn_hidden_size` | int | 512 | RNN hidden layer size |
| `rnn_num_layers` | int | 1 | Number of RNN layers |

---

### `algorithm` Algorithm Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `value_loss_coef` | float | 1.0 | Value loss coefficient |
| `use_clipped_value_loss` | bool | True | Whether to use clipped value loss |
| `clip_param` | float | 0.2 | PPO clipping parameter |
| `entropy_coef` | float | 0.01 | Entropy coefficient |
| `num_learning_epochs` | int | 5 | Number of learning epochs |
| `num_mini_batches` | int | 4 | Number of mini batches, mini batch size = num_envs * nsteps / nminibatches |
| `learning_rate` | float | 1.e-3 | Learning rate |
| `schedule` | str | 'adaptive' | Learning rate schedule: 'adaptive' or 'fixed' |
| `gamma` | float | 0.99 | Discount factor |
| `lam` | float | 0.95 | GAE lambda parameter |
| `desired_kl` | float | 0.01 | Target KL divergence |
| `max_grad_norm` | float | 1. | Maximum gradient clipping norm |
| `use_spo` | bool | False | Whether to use SPO (Simple Policy Optimization). Note: SPO may be incompatible with default PPO parameters, recommended: learning_rate=2.5e-4, schedule='fixed' |

---

### `runner` Runner Configuration

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `policy_class_name` | str | 'ActorCritic' | Policy class name |
| `algorithm_class_name` | str | 'PPO' | Algorithm class name |
| `num_steps_per_env` | int | 24 | Number of steps per environment per iteration |
| `max_iterations` | int | 1500 | Maximum number of policy update iterations |
| `sync_wandb` | bool | False | Whether to sync logs to wandb |
| `save_interval` | int | 50 | Check for potential saves every this many iterations |
| `experiment_name` | str | 'test' | Experiment name |
| `run_name` | str | '' | Run name |
| `resume` | bool | False | Whether to resume from checkpoint |
| `load_run` | int | -1 | Run ID to load, -1 for last run |
| `checkpoint` | int | -1 | Checkpoint ID, -1 for last saved model |
| `resume_path` | str | None | Resume path, updated from load_run and checkpoint |
