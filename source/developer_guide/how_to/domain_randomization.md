# Domain Randomization Tuning Guide

Domain randomization (DR) is a critical technique for sim-to-real transfer in reinforcement learning. By randomizing simulation parameters during training, policies learn to be robust to the inevitable discrepancies between simulation and reality. This guide covers the complete domain randomization system in LeggedGym-Ex, including available parameters, simulator-specific considerations, and proven tuning strategies.

## Understanding Domain Randomization

The fundamental challenge of sim-to-real transfer stems from the reality gap—the difference between simulated and real-world dynamics. Perfect simulation fidelity is impossible due to unmodeled physics, sensor noise, actuator delays, and environmental variations. Domain randomization addresses this by training policies across a distribution of dynamics rather than a single deterministic model.

The key insight is that a policy trained across wide parameter variations learns features and strategies that generalize beyond the specific dynamics of any single simulation instance. When deployed, the real robot falls somewhere within this training distribution, enabling the policy to adapt without explicit fine-tuning.

## Available Randomization Options

LeggedGym-Ex provides comprehensive domain randomization through the `domain_rand` configuration section. All parameters are defined in `LeggedRobotCfg.domain_rand` and can be enabled or disabled independently.

### Friction Randomization

Ground friction varies dramatically across surfaces—from polished floors (μ ≈ 0.2) to rough terrain (μ ≈ 1.5). Friction randomization trains policies to handle this variability without explicit surface identification.

```python
class domain_rand:
    randomize_friction: bool = True
    friction_range: List[float] = [0.5, 1.25]
```

**Recommended Range**: `[0.5, 1.25]` for general locomotion. Expand to `[0.2, 1.7]` for deployment across diverse surfaces.

**When Applied**: At episode reset. The friction coefficient is sampled uniformly from the specified range and applied to all links in the robot.

**Physical Interpretation**: The friction range covers typical rubber-on-concrete (μ ≈ 0.8-1.0), rubber-on-wood (μ ≈ 0.6-0.8), and more slippery surfaces. Values above 1.5 are uncommon but may represent high-friction rubber or specialized terrain.

### Restitution Randomization

Restitution (coefficient of restitution) controls the bounciness of contacts. While often overlooked, restitution affects impact dynamics and can significantly influence gait stability.

```python
class domain_rand:
    randomize_restitution: bool = False
    restitution_range: List[float] = [0.0, 0.5]
```

**Recommended Range**: `[0.0, 0.5]`. Most real-world surfaces have restitution below 0.3. Enable for robust impact handling.

**Default**: Disabled by default since most terrains exhibit low restitution. Enable when deploying on surfaces with significant bounce (gym floors, trampolines).

### Mass Randomization

Base mass randomization accounts for payload variations, battery weight differences, and modeling errors in the URDF. This is one of the most impactful randomizations.

```python
class domain_rand:
    randomize_base_mass: bool = True
    added_mass_range: List[float] = [-1.0, 1.0]
```

**Recommended Range**: `[-1.0, 1.0]` kg for medium quadrupeds like Go2 (base mass ~15kg). Scale proportionally for other robot sizes (±5-10% of base mass).

**Physical Interpretation**: The range covers typical variations from battery swaps, sensor payloads, or modeling errors. A 1kg variation on a 15kg robot represents ~6.7% mass uncertainty.

### Center of Mass Displacement

The center of mass (CoM) location is rarely known precisely. Manufacturing variations, cable routing, and component placement all affect the true CoM location.

```python
class domain_rand:
    randomize_com_displacement: bool = True
    com_pos_x_range: List[float] = [-0.01, 0.01]
    com_pos_y_range: List[float] = [-0.01, 0.01]
    com_pos_z_range: List[float] = [-0.01, 0.01]
```

**Recommended Range**: `[-0.01, 0.01]` m (±1cm) for each axis. Larger robots may use `[-0.03, 0.03]` m.

**Physical Interpretation**: A 1cm CoM offset represents significant but realistic uncertainty. For a 15kg robot, 1cm offset creates ~0.15 Nm of unexpected moment at moderate accelerations.

### PD Gain Randomization

PD controller gains directly affect how actions translate to joint torques. Randomizing these gains trains policies to be robust to actuator modeling errors and gain miscalibration.

```python
class domain_rand:
    randomize_pd_gain: bool = False
    kp_range: List[float] = [0.8, 1.2]  # Scale factor
    kd_range: List[float] = [0.8, 1.2]  # Scale factor
```

**Recommended Range**: `[0.8, 1.2]` for moderate variation (±20%). Wider ranges `[0.5, 1.5]` for more robustness.

**How It Works**: The scale factor multiplies the nominal PD gains defined in `cfg.control.stiffness` and `cfg.control.damping`. A scale of 1.0 means nominal gains; 0.8 means 20% lower gains.

**Impact**: This randomization significantly increases training difficulty. Enable only after achieving stable locomotion with other randomizations.

### External Push Perturbations

Random pushes train recovery behaviors and improve robustness to disturbances. This is essential for policies that will operate in unstructured environments.

```python
class domain_rand:
    push_robots: bool = True
    push_interval_s: int = 15
    max_push_vel_xy: float = 1.0
```

**Recommended Settings**:
- `push_interval_s = 15`: Push every 15 seconds (less frequent allows policy to stabilize)
- `max_push_vel_xy = 1.0`: Maximum push velocity of 1.0 m/s (moderate perturbation)

**Tuning Guidelines**:
- Start with infrequent pushes (`push_interval_s = 15-20`) and moderate velocities
- Increase frequency for more robust recovery behaviors
- Higher velocities (`1.5-2.0 m/s`) create more challenging scenarios

### Link Push Forces

In addition to base pushes, random forces can be applied to individual links to simulate contact disturbances and wind gusts.

```python
class domain_rand:
    push_links: bool = False
    max_push_force: float = 10.0  # Newtons
    push_links_interval_s: float = 15.0
```

**Recommended Range**: `5.0-15.0` N for quadruped robots. Forces above 20N may be unrealistically large.

**Use Case**: Enable when the robot will operate in environments with frequent contact disturbances (crowded spaces, manipulator interactions).

### Control Delay Randomization

Real systems experience delays between command generation and execution due to communication overhead, motor response times, and computation. Randomizing this delay improves robustness to latency.

```python
class domain_rand:
    randomize_ctrl_delay: bool = False
    ctrl_delay_step_range: List[int] = [0, 1]  # Number of simulation steps
```

**Recommended Range**: `[0, 3]` steps at 50Hz control (0-60ms delay). Real systems typically have 20-50ms latency.

**Impact**: Delay randomization significantly increases training difficulty. It's recommended to enable this only for deployment scenarios where latency is a known concern.

### Joint Dynamics Randomization

Joint-level parameters capture actuator modeling errors and variations between individual motors.

```python
class domain_rand:
    randomize_joint_armature: bool = False
    joint_armature_range: List[float] = [0.0, 0.05]  # N*m*s/rad
    
    randomize_joint_friction: bool = False
    joint_friction_range: List[float] = [0.0, 0.1]  # N*m
    
    randomize_joint_damping: bool = False
    joint_damping_range: List[float] = [0.0, 1.0]  # N*m*s/rad
```

**Physical Interpretation**:
- **Armature**: Added rotational inertia from motor rotors and gear trains
- **Joint Friction**: Coulomb friction opposing joint motion
- **Joint Damping**: Viscous damping proportional to joint velocity

**Recommendation**: Use system identification to determine appropriate ranges. These parameters significantly affect joint dynamics and should be calibrated from real robot data.

### Camera Randomization

For vision-based policies, camera pose randomization improves robustness to sensor mounting variations.

```python
class domain_rand:
    randomize_camera_pos: bool = False
    camera_com_displacement_range: List[float] = [0.01, 0.01, 0.01]
    
    randomize_camera_euler: bool = False
    camera_euler_range: List[float] = [0.1, 0.1, 0.1]  # radians
```

**Recommended Range**: Position offsets of 1-2cm and angular offsets of 0.1-0.2 radians (5-10 degrees).

## Simulator-Specific Considerations

LeggedGym-Ex supports three simulators with different domain randomization capabilities and performance characteristics.

### IsaacGym

IsaacGym provides the fastest training speed with full DR support. Key considerations:

- **Performance**: Highest throughput, ideal for extensive DR training
- **Friction**: Applied at episode reset, cannot be modified mid-episode
- **Joint Parameters**: Supported through PhysX DOF properties
- **Limitation**: Rigid body properties cannot be modified after environment creation for some parameters

### Genesis

Genesis offers excellent physics accuracy but has performance considerations for certain DR operations:

```{warning}
Randomizing joint armature, friction, and damping in Genesis requires batching DOF/link information, which significantly slows simulation. It is recommended to keep `randomize_joint_armature`, `randomize_joint_friction`, and `randomize_joint_damping` set to `False` when using Genesis. If these randomizations are needed, use IsaacGym or IsaacLab instead.
```

**Genesis-Specific Notes**:
- Friction and mass randomization work efficiently
- Avoid joint-level randomizations (armature, friction, damping) due to performance impact
- Use for scenarios where surface properties and mass variations are the primary concern

### IsaacLab

IsaacLab provides the most realistic rendering with good DR support but requires special handling for some operations:

**CPU Tensor Requirement**: Domain randomization tensors must be on CPU for certain operations:

```python
# From isaaclab_simulator.py
# Tensors passed to set_material_properties, set_masses, set_coms must be on CPU
all_indices = torch.arange(self._robot.root_physx_view.count, device="cpu")
self._robot.root_physx_view.set_material_properties(
    target_material_props.to('cpu'), all_indices
)
```

**IsaacLab-Specific Notes**:
- All material and mass property modifications require CPU tensors
- Supports full range of randomization options
- Use when visual realism is important or when integrating with IsaacSim features

## Progressive Randomization Strategy

Enabling all randomizations simultaneously often prevents training convergence. A progressive strategy yields better results:

### Phase 1: Establish Baseline (Iterations 0-500)

Start with minimal randomization to learn basic locomotion:

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

### Phase 2: Expand Core Randomizations (Iterations 500-1000)

Gradually expand the ranges of enabled randomizations:

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

### Phase 3: Full Randomization (Iterations 1000+)

Enable remaining randomizations for maximum robustness:

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

## Tuning Workflow

Follow this systematic approach to tune domain randomization for your specific deployment:

### Step 1: System Identification

Before tuning DR, calibrate your simulation using system identification:

1. Collect trajectory data from the real robot using sinusoidal joint commands
2. Run the system identification script to find matching parameters
3. Update default values in your robot configuration

See the [Sim-to-Real Transfer Guide](../../user_guide/blind_locomotion/sim2real_skills.md) for detailed system identification procedures.

### Step 2: Identify Deployment Conditions

Document the expected operating conditions:

- **Terrain types**: Smooth floors, rough terrain, slopes, stairs?
- **Payload variations**: Fixed payload or variable?
- **Environmental disturbances**: Crowded spaces, wind, contacts?
- **Surface conditions**: Dry, wet, dusty?

### Step 3: Configure Randomization Ranges

Set ranges based on deployment conditions:

| Condition | Recommended Configuration |
|-----------|--------------------------|
| Indoor smooth floors | `friction_range = [0.4, 0.8]`, narrow mass range |
| Mixed indoor/outdoor | `friction_range = [0.3, 1.5]`, moderate mass range |
| Rough outdoor terrain | `friction_range = [0.5, 1.7]`, wide mass range |
| Variable payloads | `added_mass_range = [-3.0, 3.0]` or wider |
| Unstructured environments | Enable push_robots with frequent intervals |

### Step 4: Validate Through Sim2Sim

Before real deployment, validate using cross-simulator transfer:

1. Train policy with configured DR in IsaacGym/Genesis
2. Test in a different simulator (MuJoCo via go2_deploy)
3. If transfer fails, expand DR ranges

### Step 5: Iterative Refinement

Use the following diagnostics to refine DR settings:

| Symptom | Likely Cause | Adjustment |
|---------|-------------|------------|
| Policy succeeds in sim but fails on real robot | DR ranges too narrow | Expand ranges by 20-50% |
| Training fails to converge | DR ranges too broad | Narrow ranges, use progressive strategy |
| Unstable oscillations on real robot | Actuator dynamics mismatch | Enable PD gain and joint randomizations |
| Policy collapses under disturbances | Insufficient push training | Increase push frequency and velocity |
| Good static behavior, poor motion | Insufficient friction randomization | Expand friction range |

## Best Practices Summary

1. **Start Conservative**: Begin with narrow randomization ranges and expand progressively
2. **Prioritize Impact**: Focus on friction, mass, and push randomizations first—these have the largest effect
3. **Match Deployment**: Configure ranges to cover expected real-world conditions
4. **Validate Systematically**: Use sim2sim transfer before real deployment
5. **Document Performance**: Track which DR configurations work for which deployment scenarios
6. **Consider Simulator**: Choose simulator based on DR needs—Genesis for basic DR, IsaacGym/IsaacLab for joint-level randomizations
7. **Use System Identification**: Calibrate simulation parameters before extensive DR training
8. **Monitor Training**: Watch reward curves—if training becomes unstable, narrow DR ranges

## Integration with Teacher-Student Training

Domain randomization is most effective when combined with the Teacher-Student framework. The teacher policy receives privileged information about randomized parameters, while the student policy learns to estimate them from observation history.

When using DR with Teacher-Student training, the randomized parameters are automatically included in the privileged observation:

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

This allows the student policy to learn implicit system identification from observable quantities (joint positions, velocities, IMU data).

## Common Pitfalls

### Enabling All Randomizations at Once

**Problem**: Training never converges or reward curves remain flat.
**Solution**: Use progressive randomization strategy outlined above.

### Unrealistic Randomization Ranges

**Problem**: Policy learns unrealistic behaviors that exploit simulation quirks.
**Solution**: Base ranges on physical measurements and manufacturer specifications.

### Ignoring Simulator Limitations

**Problem**: Joint randomizations in Genesis cause significant slowdown.
**Solution**: Use IsaacGym or IsaacLab for joint-level randomizations.

### Skipping Validation

**Problem**: Policy appears to work in training but fails on real robot.
**Solution**: Always validate through sim2sim transfer before real deployment.

## References

1. [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) - Foundational DR paper
2. [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) - Randomization strategies
3. [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251) - Teacher-Student with DR
