# PPO Algorithm Variants

This document provides API reference for the PPO (Proximal Policy Optimization) algorithm variants implemented in LeggedGym-Ex. Each variant is designed for specific locomotion challenges, from sim-to-real transfer to learning from motion demonstrations.

```{note}
All algorithm classes inherit from `PPO` base class and follow a consistent interface for `init_storage()`, `act()`, `process_env_step()`, `compute_returns()`, and `update()` methods.
```

## Base PPO Class

### Class Overview

The `PPO` class implements the Proximal Policy Optimization algorithm with support for both standard PPO and SPO (Simple Policy Optimization) modes. It serves as the foundation for all variant implementations.

**File Location**: `rsl_rl/algorithms/ppo.py`

### Initialization

```python
PPO(
    actor_critic: ActorCritic,
    num_learning_epochs: int = 1,
    num_mini_batches: int = 1,
    clip_param: float = 0.2,
    gamma: float = 0.998,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.0,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    use_clipped_value_loss: bool = True,
    schedule: str = "fixed",
    desired_kl: Optional[float] = 0.01,
    use_spo: bool = False,
    device: Union[str, torch.device] = 'cpu',
)
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actor_critic` | ActorCritic | required | The actor-critic network |
| `num_learning_epochs` | int | 1 | Number of optimization epochs per update |
| `num_mini_batches` | int | 1 | Number of mini-batches for SGD |
| `clip_param` | float | 0.2 | PPO clipping parameter (epsilon) |
| `gamma` | float | 0.998 | Discount factor |
| `lam` | float | 0.95 | GAE lambda parameter |
| `value_loss_coef` | float | 1.0 | Value function loss coefficient |
| `entropy_coef` | float | 0.0 | Entropy bonus coefficient |
| `learning_rate` | float | 1e-3 | Learning rate for optimizer |
| `max_grad_norm` | float | 1.0 | Maximum gradient norm for clipping |
| `use_clipped_value_loss` | bool | True | Whether to clip value function updates |
| `schedule` | str | "fixed" | Learning rate schedule: "fixed" or "adaptive" |
| `desired_kl` | float | 0.01 | Target KL divergence for adaptive schedule |
| `use_spo` | bool | False | Use Simple Policy Optimization instead of PPO |

### Core Methods

#### init_storage()

Initialize the rollout storage buffer for collecting trajectories.

```python
def init_storage(
    self,
    num_envs: int,
    num_transitions_per_env: int,
    actor_obs_shape: Tuple[int, ...],
    critic_obs_shape: Tuple[int, ...],
    action_shape: Tuple[int, ...],
) -> None
```

**Parameters:**
- `num_envs`: Number of parallel environments
- `num_transitions_per_env`: Number of steps to store per environment (rollout length)
- `actor_obs_shape`: Shape of actor observations
- `critic_obs_shape`: Shape of critic observations
- `action_shape`: Shape of actions

#### act()

Compute actions for given observations during rollout collection.

```python
def act(
    self,
    obs: torch.Tensor,
    critic_obs: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `obs`: Actor observations, shape `[num_envs, obs_dim]`
- `critic_obs`: Critic observations, shape `[num_envs, critic_obs_dim]`

**Returns:**
- `actions`: Sampled actions, shape `[num_envs, action_dim]`

#### process_env_step()

Process environment step results and store transitions.

```python
def process_env_step(
    self,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    infos: Dict[str, Any]
) -> None
```

**Parameters:**
- `rewards`: Rewards from environment, shape `[num_envs]`
- `dones`: Done flags, shape `[num_envs]`
- `infos`: Info dictionary, may contain 'time_outs' for bootstrapping

#### compute_returns()

Compute returns and advantages using Generalized Advantage Estimation (GAE).

```python
def compute_returns(
    self,
    last_critic_obs: torch.Tensor
) -> None
```

**Parameters:**
- `last_critic_obs`: Final critic observations for bootstrapping, shape `[num_envs, critic_obs_dim]`

#### update()

Update policy using collected experiences.

```python
def update() -> Tuple[float, float]
```

**Returns:**
- `mean_value_loss`: Average value function loss
- `mean_surrogate_loss`: Average surrogate loss

---

## PPO_TS (Teacher-Student)

The Teacher-Student variant implements distillation from a privileged teacher policy to a student policy that only uses observable information. This enables sim-to-real transfer by training the student to mimic the teacher's latent representations.

**Paper Reference**: [Rapid Locomotion via Reinforcement Learning](https://agility.csail.mit.edu/)

**File Location**: `rsl_rl/algorithms/ppo_ts.py`

### Unique Features

- **Dual Network Architecture**: Teacher uses privileged observations; student uses history-encoded observations
- **History Encoder**: Distills privileged information from observation history (supports MLP or TCN)
- **Privilege Encoder**: Encodes privileged observations into latent representations
- **Separate Optimizers**: One for RL parameters, one for history encoder

### Initialization

```python
PPO_TS(
    actor_critic: ActorCriticTS,
    # ... base PPO parameters ...
    encoder_lr: float = 1e-3,
    num_encoder_epochs: int = 1,
)
```

### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_lr` | float | 1e-3 | Learning rate for history encoder |
| `num_encoder_epochs` | int | 1 | Number of encoder training epochs per update |

### Key Methods

#### act()

Compute actions using teacher-student architecture.

```python
def act(
    self,
    obs: torch.Tensor,
    privileged_obs: torch.Tensor,
    obs_history: torch.Tensor,
    critic_obs: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `obs`: Actor observations
- `privileged_obs`: Privileged observations (ground truth state)
- `obs_history`: Observation history for encoder input
- `critic_obs`: Critic observations

#### update()

Returns encoder loss in addition to base losses.

```python
def update() -> Tuple[float, float, float]
```

**Returns:**
- `mean_value_loss`: Value function loss
- `mean_surrogate_loss`: Surrogate loss
- `mean_encoder_loss`: History encoder distillation loss

### Required Storage

Uses `RolloutStorageTS` which stores:
- `privileged_observations`: Ground truth states
- `observation_histories`: History for encoder training

---

## PPO_EE (Explicit Estimator)

The Explicit Estimator variant trains a state estimator concurrently with the policy. The estimator predicts privileged information (like base velocity, terrain heights) from observable history.

**Paper Reference**: [Concurrent Training of a Control Policy and a State Estimator](https://arxiv.org/abs/2202.05481)

**File Location**: `rsl_rl/algorithms/ppo_ee.py`

### Unique Features

- **Explicit State Estimator**: Neural network that estimates privileged states
- **Concurrent Training**: Policy and estimator trained together
- **MSE Loss**: Supervised learning for estimator predictions

### Initialization

```python
PPO_EE(
    actor_critic: ActorCriticEE,
    # ... base PPO parameters ...
    estimator_lr: float = 1e-3,
    num_estimator_epochs: int = 1,
)
```

### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator_lr` | float | 1e-3 | Learning rate for estimator network |
| `num_estimator_epochs` | int | 1 | Number of estimator training epochs |

### Key Methods

#### act()

Compute actions with estimator feature recording.

```python
def act(
    self,
    estimator_features: torch.Tensor,
    critic_obs: torch.Tensor,
    estimator_labels: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `estimator_features`: Input features for estimator (history)
- `critic_obs`: Critic observations
- `estimator_labels`: Ground truth labels for supervision

#### update()

Returns estimator loss in addition to base losses.

```python
def update() -> Tuple[float, float, float]
```

**Returns:**
- `mean_value_loss`: Value function loss
- `mean_surrogate_loss`: Surrogate loss
- `mean_estimator_loss`: State estimator MSE loss

---

## PPO_CTS (Concurrent Teacher-Student)

The Concurrent Teacher-Student variant trains teacher and student policies simultaneously in the same batch, improving sample efficiency and training stability compared to sequential teacher-student approaches.

**Paper Reference**: [CTS: Concurrent Teacher-Student Reinforcement Learning](https://clearlab-sustech.github.io/concurrentTS/)

**File Location**: `rsl_rl/algorithms/ppo_cts.py`

### Unique Features

- **Concurrent Training**: Teacher and student environments run in parallel
- **Shared Storage**: Single storage buffer with teacher/student partitions
- **Dual Surrogate Losses**: Separate losses for teacher and student policies

### Initialization

```python
PPO_CTS(
    actor_critic: ActorCriticCTS,
    # ... base PPO parameters ...
    encoder_lr: float = 1e-3,
    num_encoder_epochs: int = 1,
    num_teacher: int = 1,
)
```

### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_lr` | float | 1e-3 | Learning rate for history encoder |
| `num_encoder_epochs` | int | 1 | Number of encoder training epochs |
| `num_teacher` | int | 1 | Number of teacher environments |

### Key Methods

#### act()

Compute actions for both teacher and student environments.

```python
def act(
    self,
    obs: torch.Tensor,
    privileged_obs: torch.Tensor,
    obs_history: torch.Tensor,
    critic_obs: torch.Tensor
) -> torch.Tensor
```

The first `num_teacher` environments use teacher actions; remaining use student actions.

#### update()

Returns separate losses for teacher and student.

```python
def update() -> Tuple[float, float, float, float]
```

**Returns:**
- `mean_value_loss`: Value function loss
- `mean_teacher_surrogate_loss`: Teacher surrogate loss
- `mean_student_surrogate_loss`: Student surrogate loss
- `mean_reconstruction_loss`: Encoder reconstruction loss

---

## PPO_AMP (Adversarial Motion Priors)

The AMP variant enables learning natural locomotion from motion capture data using an adversarial discriminator. The discriminator distinguishes between policy-generated and expert motion clips.

**Paper Reference**: [AMP: Adversarial Motion Priors](https://arxiv.org/abs/2104.02180)

**File Location**: `rsl_rl/algorithms/ppo_amp.py`

### Unique Features

- **Discriminator Network**: Classifies policy vs expert motions
- **Motion Replay Buffer**: Stores expert motion clips
- **Style Reward**: Discriminator output used as additional reward signal
- **Symmetry Support**: Optional symmetry loss for symmetric gaits
- **Gradient Penalty**: Stabilizes discriminator training

### Initialization

```python
PPO_AMP(
    actor_critic: ActorCritic,
    discriminator: AMPDiscriminator,
    amp_data: ReplayBuffer,
    amp_normalizer: Optional[Normalizer],
    # ... base PPO parameters ...
    amp_replay_buffer_size: int = 100000,
    disc_lr: float = 1e-4,
    symmetry_cfg: Optional[Dict] = None,
)
```

### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `discriminator` | AMPDiscriminator | required | Motion discriminator network |
| `amp_data` | ReplayBuffer | required | Expert motion data buffer |
| `amp_normalizer` | Normalizer | None | Optional normalizer for AMP observations |
| `amp_replay_buffer_size` | int | 100000 | Size of policy motion replay buffer |
| `disc_lr` | float | 1e-4 | Discriminator learning rate |
| `symmetry_cfg` | Dict | None | Symmetry configuration |

### Key Methods

#### act()

Compute actions with AMP observation recording.

```python
def act(
    self,
    obs: torch.Tensor,
    critic_obs: torch.Tensor,
    amp_obs: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `obs`: Actor observations
- `critic_obs`: Critic observations
- `amp_obs`: AMP observations (body pose, velocity, etc.)

#### process_env_step()

Process step with AMP observation storage.

```python
def process_env_step(
    self,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    infos: Dict[str, Any],
    amp_obs: torch.Tensor
) -> None
```

#### update()

Returns extensive metrics for discriminator training.

```python
def update() -> Tuple[float, float, float, float, float, float, Optional[float]]
```

**Returns:**
- `mean_value_loss`: Value function loss
- `mean_surrogate_loss`: Surrogate loss
- `mean_amp_loss`: AMP discriminator loss
- `mean_grad_pen_loss`: Gradient penalty loss
- `mean_policy_pred`: Discriminator prediction on policy samples
- `mean_expert_pred`: Discriminator prediction on expert samples
- `mean_symmetry_loss`: Symmetry loss (if enabled)

---

## PPO_DreamWaQ

The DreamWaQ variant uses a VAE-based architecture to learn terrain imagination - predicting future states from observation history. This enables robust locomotion on unseen terrain.

**Paper Reference**: [DreamWaQ: Learning Robust Quadrupedal Locomotion](https://arxiv.org/abs/2301.10602)

**File Location**: `rsl_rl/algorithms/ppo_dreamwaq.py`

### Unique Features

- **VAE Architecture**: Variational autoencoder for terrain imagination
- **Implicit Terrain Estimation**: No explicit terrain sensors needed
- **Explicit State Prediction**: Predicts body velocities and terrain information
- **KL Divergence Regularization**: VAE latent space regularization

### Initialization

```python
PPO_DreamWaQ(
    actor_critic: ActorCriticDreamWaQ,
    # ... base PPO parameters ...
    encoder_lr: float = 1e-3,
    num_encoder_epochs: int = 1,
    vae_kld_weight: float = 1.0,
)
```

### Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_lr` | float | 1e-3 | Learning rate for VAE encoder |
| `num_encoder_epochs` | int | 1 | Number of VAE training epochs |
| `vae_kld_weight` | float | 1.0 | Weight for KL divergence loss |

### Key Methods

#### act()

Compute actions with VAE input recording.

```python
def act(
    self,
    obs: torch.Tensor,
    privileged_obs: torch.Tensor,
    obs_history: torch.Tensor,
    explicit_info_labels: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `obs`: Actor observations
- `privileged_obs`: Privileged observations for critic
- `obs_history`: Observation history for VAE
- `explicit_info_labels`: Labels for explicit state prediction

#### process_env_step()

Store next state for reconstruction loss.

```python
def process_env_step(
    self,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    infos: Dict[str, Any],
    next_state: torch.Tensor
) -> None
```

#### update()

Returns VAE-related losses.

```python
def update() -> Tuple[float, float, float, float, float]
```

**Returns:**
- `mean_value_loss`: Value function loss
- `mean_surrogate_loss`: Surrogate loss
- `mean_explicit_estimation_loss`: Explicit state prediction loss
- `mean_reconstruction_loss`: State reconstruction loss
- `mean_kld_loss`: KL divergence loss

---

## Runner Classes

Runners orchestrate the training loop, managing environment interaction, data collection, and algorithm updates.

### OnPolicyRunner

The base runner for on-policy RL training. Manages the training loop, logging, and model checkpointing.

**File Location**: `rsl_rl/runners/on_policy_runner.py`

#### Initialization

```python
OnPolicyRunner(
    env: VecEnv,
    train_cfg: Dict[str, Any],
    log_dir: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
)
```

#### Key Methods

##### learn()

Run the training loop.

```python
def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
) -> None
```

**Parameters:**
- `num_learning_iterations`: Number of training iterations
- `init_at_random_ep_len`: Randomize initial episode lengths

##### save() / load()

Checkpoint management.

```python
def save(self, path: str, infos: Optional[Dict] = None) -> None
def load(self, path: str, load_optimizer: bool = True) -> Optional[Dict]
```

##### get_inference_policy()

Get the policy function for deployment.

```python
def get_inference_policy(
    self,
    device: Optional[Union[str, torch.device]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]
```

### TSRunner

Specialized runner for Teacher-Student training. Handles observation history and privileged information.

**File Location**: `rsl_rl/runners/ts_runner.py`

#### Key Differences from Base Runner

- `get_observations()` returns tuple of `(obs, privileged_obs, obs_history, critic_obs)`
- `get_inference_policy()` returns student policy (not teacher)

### EERunner

Specialized runner for Explicit Estimator training. Manages estimator features and labels.

**File Location**: `rsl_rl/runners/ee_runner.py`

#### Key Differences from Base Runner

- `get_observations()` returns tuple of `(estimator_features, estimator_labels, privileged_obs)`
- Logs estimator loss metrics

---

## Training Flow

The following describes the training flow for on-policy PPO variants:

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING ITERATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INITIALIZATION                                               │
│     ├── runner._init_agent_and_algo()                           │
│     │   └── Create actor-critic network                         │
│     │   └── Create PPO algorithm instance                       │
│     └── runner._init_storage()                                  │
│         └── alg.init_storage() -> RolloutStorage                │
│                                                                  │
│  2. ROLLOUT COLLECTION (repeat N steps)                         │
│     ├── alg.act(obs, critic_obs) -> actions                     │
│     ├── env.step(actions) -> obs, rewards, dones, infos         │
│     └── alg.process_env_step(rewards, dones, infos)             │
│         └── storage.add_transitions(transition)                 │
│                                                                  │
│  3. RETURN COMPUTATION                                           │
│     └── alg.compute_returns(last_critic_obs)                    │
│         └── GAE: A_t = Σ (γλ)^l * δ_{t+l}                       │
│         └── Returns: R_t = A_t + V(s_t)                         │
│                                                                  │
│  4. POLICY UPDATE (repeat K epochs × M mini-batches)            │
│     ├── For each mini-batch from storage:                       │
│     │   ├── Forward pass through actor-critic                   │
│     │   ├── Compute ratio: π(a|s) / π_old(a|s)                  │
│     │   ├── Surrogate loss: max(L^CLIP, L^CLIP')                │
│     │   ├── Value loss: (V(s) - R)^2                            │
│     │   ├── Entropy bonus: -β * H(π(·|s))                       │
│     │   └── optimizer.step()                                    │
│     │                                                            │
│     └── For variants with encoders:                             │
│         ├── Compute encoder loss (MSE)                          │
│         └── encoder_optimizer.step()                            │
│                                                                  │
│  5. LOGGING & CHECKPOINTING                                      │
│     ├── runner.log(metrics)                                     │
│     │   └── TensorBoard / WandB logging                         │
│     └── runner.save() if checkpoint interval                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Equations

**PPO Clipped Objective:**
```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```
where `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`

**Generalized Advantage Estimation:**
```
A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**Total Loss:**
```
L = L^CLIP + c_1 * L^VF - c_2 * H(π)
```

---

## Algorithm Configuration Parameters

### Common PPO Parameters

All variants share these base configuration parameters under `cfg.algorithm`:

```python
class LeggedRobotCfgPPO:
    class algorithm:
        value_loss_coef = 1.0          # Value function loss weight
        use_clipped_value_loss = True  # Clip value updates
        clip_param = 0.2               # PPO clipping epsilon
        entropy_coef = 0.01            # Entropy bonus weight
        num_learning_epochs = 5        # Epochs per iteration
        num_mini_batches = 4           # Mini-batches per epoch
        learning_rate = 1.e-3          # Adam learning rate
        schedule = 'adaptive'          # LR schedule
        gamma = 0.99                   # Discount factor
        lam = 0.95                     # GAE lambda
        desired_kl = 0.01              # Target KL divergence
        max_grad_norm = 1.0            # Gradient clipping
```

### Variant-Specific Parameters

#### Teacher-Student (PPO_TS)

```python
class algorithm:
    # ... base parameters ...
    encoder_lr = 1e-3              # History encoder LR
    num_encoder_epochs = 1         # Encoder epochs per update
```

#### Explicit Estimator (PPO_EE)

```python
class algorithm:
    # ... base parameters ...
    estimator_lr = 1e-3            # Estimator LR
    num_estimator_epochs = 1        # Estimator epochs
```

#### Concurrent TS (PPO_CTS)

```python
class algorithm:
    # ... base parameters ...
    encoder_lr = 1e-3              # Encoder LR
    num_encoder_epochs = 1          # Encoder epochs
    num_teacher = 1                 # Number of teacher envs
```

#### AMP (PPO_AMP)

```python
class algorithm:
    # ... base parameters ...
    disc_lr = 1e-4                  # Discriminator LR
    amp_replay_buffer_size = 100000 # Policy buffer size
```

#### DreamWaQ (PPO_DreamWaQ)

```python
class algorithm:
    # ... base parameters ...
    encoder_lr = 1e-3              # VAE encoder LR
    num_encoder_epochs = 1          # VAE epochs
    vae_kld_weight = 1.0           # KL divergence weight
```

### Runner Parameters

Configuration under `cfg.runner`:

```python
class runner:
    policy_class_name = "ActorCritic"     # Network class
    algorithm_class_name = "PPO"          # Algorithm class
    num_steps_per_env = 24                 # Rollout length
    max_iterations = 1500                  # Total iterations
    save_interval = 50                     # Checkpoint interval
    experiment_name = "test"               # Log directory name
    run_name = ""                          # Run identifier
    resume = False                         # Resume from checkpoint
    load_run = -1                          # Run ID to load
    checkpoint = -1                        # Checkpoint ID
    sync_wandb = False                     # Enable WandB sync
```

---

## Usage Examples

### Training with Base PPO

```python
from rsl_rl.runners import OnPolicyRunner

# Initialize runner
runner = OnPolicyRunner(
    env=env,
    train_cfg=train_cfg,
    log_dir=log_dir,
    device="cuda"
)

# Train
runner.learn(num_learning_iterations=1500)

# Get inference policy
policy = runner.get_inference_policy(device="cpu")
```

### Training with Teacher-Student

```python
from rsl_rl.runners import TSRunner

# TSRunner automatically uses PPO_TS and ActorCriticTS
runner = TSRunner(
    env=env,
    train_cfg=train_cfg,
    log_dir=log_dir,
    device="cuda"
)

# Train with distillation
runner.learn(num_learning_iterations=1500)

# Get student policy for deployment
student_policy = runner.get_inference_policy()
```

### Training with AMP

```python
from rsl_rl.algorithms import PPO_AMP
from rsl_rl.modules import AMPDiscriminator
from rsl_rl.storage import ReplayBuffer

# Create discriminator
discriminator = AMPDiscriminator(input_dim=amp_obs_dim * 2)

# Create algorithm
alg = PPO_AMP(
    actor_critic=actor_critic,
    discriminator=discriminator,
    amp_data=expert_motion_buffer,
    amp_normalizer=normalizer,
    device="cuda"
)

# Standard training loop with AMP-specific losses
```

---

## Component Compatibility Matrix

| Algorithm | Actor-Critic | Storage | Runner |
|-----------|--------------|---------|--------|
| PPO | ActorCritic | RolloutStorage | OnPolicyRunner |
| PPO_TS | ActorCriticTS | RolloutStorageTS | TSRunner |
| PPO_EE | ActorCriticEE | RolloutStorageEE | EERunner |
| PPO_CTS | ActorCriticCTS | RolloutStorageCTS | CTSRunner |
| PPO_AMP | ActorCritic | RolloutStorage | AMPRunner |
| PPO_DreamWaQ | ActorCriticDreamWaQ | RolloutStorageDreamWaQ | DreamWaQRunner |

```{warning}
Using incompatible components (e.g., base RolloutStorage with PPO_TS) will cause runtime errors. Always match algorithm, storage, and runner classes according to the table above.
```

---

## References

- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **SPO**: [Simple Policy Optimization](https://arxiv.org/abs/2401.16025)
- **Teacher-Student**: [Rapid Locomotion via RL](https://agility.csail.mit.edu/)
- **Explicit Estimator**: [Concurrent Training of Control Policy and State Estimator](https://arxiv.org/abs/2202.05481)
- **CTS**: [Concurrent Teacher-Student RL](https://clearlab-sustech.github.io/concurrentTS/)
- **AMP**: [Adversarial Motion Priors](https://arxiv.org/abs/2104.02180)
- **DreamWaQ**: [DreamWaQ: Learning Robust Quadrupedal Locomotion](https://arxiv.org/abs/2301.10602)
