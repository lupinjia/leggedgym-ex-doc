# Adding a New RL Method

This guide walks through the process of implementing a new reinforcement learning method in LeggedGym-Ex. The framework uses a modular architecture where each RL method consists of four core components that must work together seamlessly.

```{note}
Before starting, review existing implementations in `rsl_rl/` to understand the patterns. The Teacher-Student (TS) variant provides a complete reference implementation.
```

## Architecture Overview

LeggedGym-Ex follows a component-based architecture for RL methods. Each method requires four interconnected components:

| Component | Purpose | Base Class | Example File |
|-----------|---------|------------|--------------|
| **Algorithm** | Core RL logic, loss computation, optimization | `BaseAlgorithm` | `algorithms/ppo_ts.py` |
| **Actor-Critic** | Neural network policy and value function | `nn.Module` | `modules/actor_critic_ts.py` |
| **Storage** | Rollout buffer for experience collection | `RolloutStorage` | `storage/rollout_storage_ts.py` |
| **Runner** | Training orchestration, logging, checkpointing | `OnPolicyRunner` | `runners/ts_runner.py` |

```{important}
All four components must be compatible with each other. Using mismatched components (e.g., `PPO_TS` algorithm with base `RolloutStorage`) will cause runtime errors.
```

## Step 1: Create Algorithm Class

The algorithm class implements the core RL logic. Extend `BaseAlgorithm` or an existing algorithm like `PPO` for PPO-based methods.

### 1.1 Basic Structure

Create a new file in `rsl_rl/algorithms/`:

```python
# rsl_rl/algorithms/ppo_custom.py
from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCriticCustom
from rsl_rl.storage import RolloutStorageCustom


class PPO_Custom(PPO):
    """Custom PPO variant with specific modifications.
    
    This class extends the base PPO algorithm with custom features.
    """
    
    actor_critic: ActorCriticCustom
    storage: Optional[RolloutStorageCustom]
    transition: RolloutStorageCustom.Transition
    
    def __init__(
        self,
        actor_critic: ActorCriticCustom,
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
        device: str = 'cpu',
        # Add custom parameters here
        custom_param: float = 0.5,
    ) -> None:
        super().__init__(
            actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device=device,
        )
        self.custom_param = custom_param
        
        # Override components if needed
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        
        # Custom optimizer setup
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), 
            lr=learning_rate
        )
        
        # Initialize transition storage
        self.transition = RolloutStorageCustom.Transition()
```

### 1.2 Override Core Methods

At minimum, override these methods:

```python
def init_storage(
    self,
    num_envs: int,
    num_transitions_per_env: int,
    actor_obs_shape: Tuple[int, ...],
    privileged_obs_shape: Tuple[int, ...],
    critic_obs_shape: Tuple[int, ...],
    action_shape: Tuple[int, ...],
) -> None:
    """Initialize storage with custom observation shapes."""
    self.storage = RolloutStorageCustom(
        num_envs, 
        num_transitions_per_env, 
        actor_obs_shape,
        privileged_obs_shape, 
        critic_obs_shape, 
        action_shape, 
        self.device
    )

def act(
    self, 
    obs: torch.Tensor, 
    privileged_obs: torch.Tensor,
    critic_obs: torch.Tensor
) -> torch.Tensor:
    """Compute actions during rollout collection.
    
    This method must:
    1. Store observations in transition
    2. Compute and store actions, values, log probs
    3. Return actions for environment stepping
    """
    # Store observations
    self.transition.observations = obs
    self.transition.privileged_observations = privileged_obs
    self.transition.critic_observations = critic_obs
    
    # Compute actions
    self.transition.actions = self.actor_critic.act(
        obs, privileged_obs
    ).detach()
    
    # Compute values and log probabilities
    self.transition.values = self.actor_critic.evaluate(
        critic_obs
    ).detach()
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
        self.transition.actions
    ).detach()
    
    # Store action distribution parameters
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()
    
    return self.transition.actions

def update(self) -> Tuple[float, ...]:
    """Update policy using collected experiences.
    
    Returns:
        Tuple of loss values for logging.
    """
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    
    # Get mini-batch generator
    generator = self.storage.mini_batch_generator(
        self.num_mini_batches, 
        self.num_learning_epochs
    )
    
    for batch in generator:
        obs_batch, privileged_obs_batch, critic_obs_batch, \
        actions_batch, values_batch, advantages_batch, returns_batch, \
        old_log_prob_batch, old_mu_batch, old_sigma_batch, \
        hidden_states_batch, masks_batch = batch
        
        # Compute losses
        loss, surrogate_loss, value_loss = self._compute_loss(
            obs_batch, privileged_obs_batch, critic_obs_batch,
            actions_batch, values_batch, advantages_batch, returns_batch,
            old_log_prob_batch, old_mu_batch, old_sigma_batch,
            hidden_states_batch, masks_batch
        )
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), 
            self.max_grad_norm
        )
        self.optimizer.step()
        
        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
    
    # Clear storage after update
    self.storage.clear()
    
    num_updates = self.num_learning_epochs * self.num_mini_batches
    return (
        mean_value_loss / num_updates,
        mean_surrogate_loss / num_updates,
    )
```

### 1.3 Algorithm Registration

Add to `rsl_rl/algorithms/__init__.py`:

```python
from .ppo_custom import PPO_Custom

__all__.append("PPO_Custom")
```

## Step 2: Create Actor-Critic Module

The actor-critic module defines the neural network architecture for your policy and value function.

### 2.1 Basic Structure

Create a new file in `rsl_rl/modules/`:

```python
# rsl_rl/modules/actor_critic_custom.py
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticCustom(nn.Module):
    """Custom actor-critic network for PPO_Custom."""
    
    is_recurrent = False  # Set to True for RNN policies
    
    def __init__(
        self,
        num_actor_obs: int,
        num_actions: int,
        num_critic_obs: int,
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        activation: str = 'elu',
        init_noise_std: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        # Get activation function
        activation_fn = self._get_activation(activation)
        
        # Build actor network
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(
                nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1])
            )
            actor_layers.append(activation_fn)
        
        actor_layers.append(
            nn.Linear(actor_hidden_dims[-1], num_actions)
        )
        self.actor = nn.Sequential(*actor_layers)
        
        # Build critic network
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(
                nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1])
            )
            critic_layers.append(activation_fn)
        
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Action noise parameter
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # Disable validation for speed
        Normal.set_default_validate_args = False
    
    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'elu': nn.ELU(),
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
        }
        return activations.get(name, nn.ELU())
```

### 2.2 Implement Required Methods

```python
def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
    """Sample actions from the policy."""
    self.update_distribution(observations)
    return self.distribution.sample()

def update_distribution(self, observations: torch.Tensor) -> None:
    """Update the action distribution."""
    mean = self.actor(observations)
    self.distribution = Normal(mean, mean * 0.0 + self.std)

def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    """Get log probability of actions."""
    return self.distribution.log_prob(actions).sum(dim=-1)

def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
    """Evaluate the value function."""
    return self.critic(critic_observations)

@property
def action_mean(self) -> torch.Tensor:
    return self.distribution.mean

@property
def action_std(self) -> torch.Tensor:
    return self.distribution.stddev

@property
def entropy(self) -> torch.Tensor:
    return self.distribution.entropy().sum(dim=-1)

def reset(self, dones: torch.Tensor = None) -> None:
    """Reset hidden states for recurrent policies."""
    pass  # No-op for non-recurrent
```

### 2.3 Module Registration

Add to `rsl_rl/modules/__init__.py`:

```python
from .actor_critic_custom import ActorCriticCustom
```

## Step 3: Create Storage Class

The storage class manages rollout buffers for experience collection. Extend `RolloutStorage` for most cases.

### 3.1 Basic Structure

Create a new file in `rsl_rl/storage/`:

```python
# rsl_rl/storage/rollout_storage_custom.py
import torch
from .rollout_storage import RolloutStorage


class RolloutStorageCustom(RolloutStorage):
    """Custom rollout storage for PPO_Custom."""
    
    class Transition(RolloutStorage.Transition):
        """Extended transition for custom observations."""
        
        def __init__(self):
            super().__init__()
            # Add custom observation fields
            self.custom_observations = None
            self.privileged_observations = None
            self.critic_observations = None
    
    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: tuple,
        privileged_obs_shape: tuple,
        critic_obs_shape: tuple,
        actions_shape: tuple,
        device: str = 'cpu'
    ):
        super().__init__(
            num_envs, 
            num_transitions_per_env, 
            obs_shape,
            privileged_obs_shape,  # Used as privileged obs
            actions_shape, 
            device
        )
        
        # Additional storage for custom observations
        self.critic_obs_shape = critic_obs_shape
        self.critic_observations = torch.zeros(
            num_transitions_per_env, 
            num_envs, 
            *critic_obs_shape, 
            device=self.device
        )
```

### 3.2 Override Storage Methods

```python
def add_transitions(self, transition: Transition) -> None:
    """Add a transition to the buffer."""
    if self.step >= self.num_transitions_per_env:
        raise AssertionError("Rollout buffer overflow")
    
    # Store all transition data
    self.observations[self.step].copy_(transition.observations)
    self.privileged_observations[self.step].copy_(
        transition.privileged_observations
    )
    self.critic_observations[self.step].copy_(
        transition.critic_observations
    )
    self.actions[self.step].copy_(transition.actions)
    self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    self.values[self.step].copy_(transition.values)
    self.actions_log_prob[self.step].copy_(
        transition.actions_log_prob.view(-1, 1)
    )
    self.mu[self.step].copy_(transition.action_mean)
    self.sigma[self.step].copy_(transition.action_sigma)
    
    self.step += 1

def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
    """Generate mini-batches for training."""
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(
        num_mini_batches * mini_batch_size, 
        requires_grad=False, 
        device=self.device
    )
    
    # Flatten all buffers
    observations = self.observations.flatten(0, 1)
    privileged_observations = self.privileged_observations.flatten(0, 1)
    critic_observations = self.critic_observations.flatten(0, 1)
    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_mu = self.mu.flatten(0, 1)
    old_sigma = self.sigma.flatten(0, 1)
    
    for epoch in range(num_epochs):
        for i in range(num_mini_batches):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            batch_idx = indices[start:end]
            
            yield (
                observations[batch_idx],
                privileged_observations[batch_idx],
                critic_observations[batch_idx],
                actions[batch_idx],
                values[batch_idx],
                advantages[batch_idx],
                returns[batch_idx],
                old_actions_log_prob[batch_idx],
                old_mu[batch_idx],
                old_sigma[batch_idx],
                (None, None),  # Hidden states for RNN
                None,  # Masks for RNN
            )
```

### 3.3 Storage Registration

Add to `rsl_rl/storage/__init__.py`:

```python
from .rollout_storage_custom import RolloutStorageCustom
```

## Step 4: Create Runner Class

The runner orchestrates the training loop, handles logging, and manages checkpoints.

### 4.1 Basic Structure

Create a new file in `rsl_rl/runners/`:

```python
# rsl_rl/runners/custom_runner.py
from typing import Optional, Union, Callable, Dict, Any, List
import torch
from collections import deque
import statistics
import time

from rsl_rl.algorithms import PPO_Custom
from rsl_rl.modules import ActorCriticCustom
from rsl_rl.env import VecEnv
from .on_policy_runner import OnPolicyRunner


class CustomRunner(OnPolicyRunner):
    """Custom runner for PPO_Custom training."""
    
    def __init__(
        self,
        env: VecEnv,
        train_cfg: Dict[str, Any],
        log_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
    
    def _init_agent_and_algo(self) -> None:
        """Initialize actor-critic and algorithm."""
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCriticCustom = actor_critic_class(
            self.env.num_obs,
            self.env.num_actions,
            self.env.num_critic_obs,  # Must be defined in env
            **self.policy_cfg
        ).to(self.device)
        
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO_Custom = alg_class(
            actor_critic, 
            device=self.device, 
            **self.alg_cfg
        )
    
    def _init_storage(self) -> None:
        """Initialize storage with correct shapes."""
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            (self.env.num_obs,),
            (self.env.num_privileged_obs,),
            (self.env.num_critic_obs,),
            (self.env.num_actions,),
        )
```

### 4.2 Override Training Loop

```python
def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
) -> None:
    """Main training loop."""
    # Pre-learn setup
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint(
            0, 
            self.env.max_episode_length, 
            (self.env.num_envs,),
            device=self.env.device
        )
    
    # Get initial observations
    obs, privileged_obs, critic_obs = self.env.get_observations()
    obs = obs.to(self.device)
    privileged_obs = privileged_obs.to(self.device)
    critic_obs = critic_obs.to(self.device)
    
    self.alg.actor_critic.train()
    
    # Episode tracking buffers
    ep_infos: List[Dict[str, Any]] = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(
        self.env.num_envs, 
        dtype=torch.float, 
        device=self.device
    )
    cur_episode_length = torch.zeros(
        self.env.num_envs, 
        dtype=torch.float, 
        device=self.device
    )
    
    # Main training loop
    tot_iter = self.current_learning_iteration + num_learning_iterations
    for it in range(self.current_learning_iteration, tot_iter):
        start = time.time()
        
        # Rollout collection
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                actions = self.alg.act(obs, privileged_obs, critic_obs)
                obs, privileged_obs, critic_obs, rewards, dones, infos = \
                    self.env.step(actions)
                
                obs = obs.to(self.device)
                privileged_obs = privileged_obs.to(self.device)
                critic_obs = critic_obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                
                self.alg.process_env_step(rewards, dones, infos)
                
                # Episode tracking
                if self.log_dir is not None:
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(
                        cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    lenbuffer.extend(
                        cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
        
        collection_time = time.time() - start
        
        # Learning step
        start = time.time()
        self.alg.compute_returns(critic_obs)
        mean_value_loss, mean_surrogate_loss = self.alg.update()
        learn_time = time.time() - start
        
        # Logging
        if self.log_dir is not None:
            self.log(locals())
        
        # Checkpointing
        if it % self.save_interval == 0:
            self.save(f"{self.log_dir}/model_{it}.pt")
        
        ep_infos.clear()
    
    self.current_learning_iteration += num_learning_iterations
    self.save(f"{self.log_dir}/model_{self.current_learning_iteration}.pt")
```

### 4.3 Implement Inference Policy

```python
def get_inference_policy(
    self,
    device: Optional[Union[str, torch.device]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get policy for inference/deployment."""
    self.alg.actor_critic.eval()
    if device is not None:
        self.alg.actor_critic.to(device)
    return self.alg.actor_critic.act
```

## Step 5: Register Runner

Register the runner in `rsl_rl/runners/__init__.py`:

```python
from .custom_runner import CustomRunner
from rsl_rl.utils.runner_registry import runner_registry

runner_registry.register("CustomRunner", CustomRunner)
```

## Step 6: Create Environment Variant (Optional)

If your method requires special environment modifications, create a variant in `legged_gym/envs/`.

### 6.1 Define Configuration

```python
# legged_gym/envs/go2/go2_custom/go2_custom_config.py
from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg, 
    LeggedRobotCfgPPO
)


class GO2CustomCfg(LeggedRobotCfg):
    """Configuration for GO2 with custom method."""
    
    class env(LeggedRobotCfg.env):
        pass
    
    class runner(LeggedRobotCfg.runner):
        runner_class_name = "CustomRunner"
        policy_class_name = "ActorCriticCustom"
        algorithm_class_name = "PPO_Custom"


class GO2CustomCfgPPO(LeggedRobotCfgPPO):
    """PPO configuration for GO2 custom method."""
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        custom_param = 0.5  # Custom parameter
```

### 6.2 Register Task

Add to `legged_gym/envs/__init__.py`:

```python
from .go2.go2_custom.go2_custom import GO2Custom
from .go2.go2_custom.go2_custom_config import GO2CustomCfg, GO2CustomCfgPPO

task_registry.register(
    "go2_custom", 
    GO2Custom, 
    GO2CustomCfg, 
    GO2CustomCfgPPO
)
```

## Component Compatibility Requirements

Understanding component compatibility is critical for implementing new methods.

### Storage-Algorithm Matching

The storage must provide all observations the algorithm requires:

| Algorithm | Required Storage Fields |
|-----------|------------------------|
| `PPO` | `observations`, `privileged_observations` |
| `PPO_TS` | `observations`, `privileged_observations`, `observation_histories`, `critic_observations` |
| `PPO_EE` | `observations`, `privileged_observations`, `estimated_states` |
| `PPO_AMP` | `observations`, `privileged_observations`, `reference_motion_obs` |

### Actor-Critic Interface Matching

The actor-critic must implement methods the algorithm calls:

| Method | Called By | Must Return |
|--------|-----------|-------------|
| `act(obs, ...)` | Algorithm.act() | Action tensor |
| `evaluate(critic_obs, ...)` | Algorithm.act() | Value tensor |
| `get_actions_log_prob(actions)` | Algorithm.update() | Log probability tensor |
| `action_mean`, `action_std` | Algorithm.act() | Distribution parameters |

### Runner-Environment Matching

The environment must provide observation shapes the runner expects:

```python
# Environment must define these properties
self.num_obs = 48
self.num_privileged_obs = 187
self.num_critic_obs = 187  # For TS methods
self.num_history_obs = 480  # For TS methods
self.num_latent_dims = 64   # For encoder-based methods
```

## Complete Example: Teacher-Student Implementation

The Teacher-Student (TS) method demonstrates all components working together:

### Algorithm (PPO_TS)

```python
class PPO_TS(PPO):
    actor_critic: ActorCriticTS
    storage: RolloutStorageTS
    
    def __init__(self, actor_critic, encoder_lr=1e-3, ...):
        super().__init__(actor_critic, ...)
        # Separate optimizer for history encoder
        self.history_encoder_optimizer = optim.Adam(
            actor_critic.history_encoder.parameters(), 
            lr=encoder_lr
        )
    
    def act(self, obs, privileged_obs, obs_history, critic_obs):
        # Store all observation types
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        self.transition.critic_observations = critic_obs
        # Compute actions with privileged encoder
        self.transition.actions = self.actor_critic.act(
            obs, privileged_obs
        ).detach()
        return self.transition.actions
    
    def update(self):
        # Standard PPO update
        mean_value_loss, mean_surrogate_loss = self._ppo_update()
        # Additional encoder distillation
        mean_encoder_loss = self._encoder_update()
        return mean_value_loss, mean_surrogate_loss, mean_encoder_loss
```

### Actor-Critic (ActorCriticTS)

```python
class ActorCriticTS(nn.Module):
    def __init__(self, num_actor_obs, num_actions, 
                 num_privilege_encoder_input, num_history_encoder_input,
                 num_latent_dims, num_critic_obs, ...):
        # Privilege encoder: privileged_obs -> latent
        self.privilege_encoder = nn.Sequential(...)
        
        # History encoder: obs_history -> latent (student)
        self.history_encoder = nn.Sequential(...)
        
        # Actor: [obs, latent] -> action
        self.actor = nn.Sequential(...)
        
        # Critic: critic_obs -> value
        self.critic = nn.Sequential(...)
    
    def act(self, observations, privilege_observations):
        latent = self.privilege_encoder(privilege_observations)
        mean = self.actor(torch.cat([observations, latent], dim=-1))
        self.distribution = Normal(mean, self.std)
        return self.distribution.sample()
    
    def act_student(self, observations, observation_history):
        # For deployment: use history encoder instead
        latent = self.history_encoder(observation_history)
        mean = self.actor(torch.cat([observations, latent], dim=-1))
        return mean
```

### Storage (RolloutStorageTS)

```python
class RolloutStorageTS(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.privileged_observations = None
            self.observation_histories = None
    
    def __init__(self, num_envs, num_transitions_per_env, 
                 obs_shape, privileged_obs_shape, 
                 obs_history_shape, critic_obs_shape, ...):
        # Additional buffers for TS-specific observations
        self.observation_histories = torch.zeros(...)
        self.critic_observations = torch.zeros(...)
```

### Runner (TSRunner)

```python
class TSRunner(OnPolicyRunner):
    def _init_agent_and_algo(self):
        actor_critic = ActorCriticTS(
            self.env.num_obs,
            self.env.num_actions,
            self.env.num_privileged_obs,
            self.env.num_history_obs,
            self.env.num_latent_dims,
            self.env.num_critic_obs,
            **self.policy_cfg
        )
        self.alg = PPO_TS(actor_critic, **self.alg_cfg)
    
    def learn(self, num_learning_iterations, ...):
        # Get all observation types
        obs, privileged_obs, obs_history, critic_obs = \
            self.env.get_observations()
        # Training loop with 4 observation types
        ...
    
    def get_inference_policy(self, device=None):
        # Return student policy for deployment
        return self.alg.actor_critic.act_student
```

## Testing Your Implementation

After implementing all components, test with:

```bash
# Quick test with few environments
python -m legged_gym.scripts.train --task go2_custom --num_envs 10 --headless

# Full training
python -m legged_gym.scripts.train --task go2_custom --headless

# Inference test
python -m legged_gym.scripts.play --task go2_custom
```

## Summary

Implementing a new RL method requires:

1. **Algorithm**: Extend `BaseAlgorithm` or `PPO`, implement `init_storage()`, `act()`, `update()`
2. **Actor-Critic**: Build neural networks, implement `act()`, `evaluate()`, distribution methods
3. **Storage**: Extend `RolloutStorage`, add custom observation buffers
4. **Runner**: Extend `OnPolicyRunner`, customize initialization and training loop
5. **Registration**: Register runner in `runners/__init__.py`
6. **Environment**: Create config variant if needed, register task

Always ensure component compatibility: the algorithm must match the storage format, and the actor-critic must implement the methods the algorithm expects. Reference existing implementations like `PPO_TS` for complete working examples.
