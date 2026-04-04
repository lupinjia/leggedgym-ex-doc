# 添加新的强化学习方法

本指南介绍在 LeggedGym-Ex 中实现新的强化学习方法的流程。该框架使用模块化架构,每个强化学习方法由四个核心组件组成,必须协同工作。

```{note}
开始前,请查看 `rsl_rl/` 中的现有实现以了解模式。教师-学生 (TS) 变体提供了一个完整的参考实现。
```

## 架构概述

LeggedGym-Ex 采用基于组件的架构来实现强化学习方法。每个方法需要四个相互关联的组件:

| 组件 | 用途 | 基类 | 示例文件 |
|------|------|------|----------|
| **算法** | 核心强化学习逻辑、损失计算、优化 | `BaseAlgorithm` | `algorithms/ppo_ts.py` |
| **演员-评论家** | 神经网络策略和价值函数 | `nn.Module` | `modules/actor_critic_ts.py` |
| **存储** | 经验收集的回合缓冲区 | `RolloutStorage` | `storage/rollout_storage_ts.py` |
| **运行器** | 训练编排、日志记录、检查点 | `OnPolicyRunner` | `runners/ts_runner.py` |

```{important}
所有四个组件必须相互兼容。使用不匹配的组件 (例如,`PPO_TS` 算法与基础 `RolloutStorage`) 会导致运行时错误。
```

## 步骤 1: 创建算法类

算法类实现核心强化学习逻辑。对于基于 PPO 的方法,请扩展 `BaseAlgorithm` 或现有算法如 `PPO`。

### 1.1 基本结构

在 `rsl_rl/algorithms/` 中创建一个新文件:

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
    """具有特定修改的自定义 PPO 变体。
    
    此类扩展了基础 PPO 算法,添加了自定义功能。
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
        # 在此处添加自定义参数
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
        
        # 如需则覆盖组件
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        
        # 自定义优化器设置
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), 
            lr=learning_rate
        )
        
        # 初始化转换存储
        self.transition = RolloutStorageCustom.Transition()
```

### 1.2 重写核心方法

至少重写这些方法:

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
    """使用自定义观测形状初始化存储。"""
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
    """在回合收集期间计算动作。
    
    此方法必须:
    1. 将观测存储在转换中
    2. 计算并存储动作、价值、对数概率
    3. 返回用于环境步进的动作
    """
    # 存储观测
    self.transition.observations = obs
    self.transition.privileged_observations = privileged_obs
    self.transition.critic_observations = critic_obs
    
    # 计算动作
    self.transition.actions = self.actor_critic.act(
        obs, privileged_obs
    ).detach()
    
    # 计算价值和对数概率
    self.transition.values = self.actor_critic.evaluate(
        critic_obs
    ).detach()
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
        self.transition.actions
    ).detach()
    
    # 存储动作分布参数
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()
    
    return self.transition.actions

def update(self) -> Tuple[float, ...]:
    """使用收集的经验更新策略。
    
    Returns:
        用于日志记录的损失值元组。
    """
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    
    # 获取小批量生成器
    generator = self.storage.mini_batch_generator(
        self.num_mini_batches, 
        self.num_learning_epochs
    )
    
    for batch in generator:
        obs_batch, privileged_obs_batch, critic_obs_batch, \
        actions_batch, values_batch, advantages_batch, returns_batch, \
        old_log_prob_batch, old_mu_batch, old_sigma_batch, \
        hidden_states_batch, masks_batch = batch
        
        # 计算损失
        loss, surrogate_loss, value_loss = self._compute_loss(
            obs_batch, privileged_obs_batch, critic_obs_batch,
            actions_batch, values_batch, advantages_batch, returns_batch,
            old_log_prob_batch, old_mu_batch, old_sigma_batch,
            hidden_states_batch, masks_batch
        )
        
        # 梯度步进
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), 
            self.max_grad_norm
        )
        self.optimizer.step()
        
        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
    
    # 更新后清除存储
    self.storage.clear()
    
    num_updates = self.num_learning_epochs * self.num_mini_batches
    return (
        mean_value_loss / num_updates,
        mean_surrogate_loss / num_updates,
    )
```

### 1.3 算法注册

添加到 `rsl_rl/algorithms/__init__.py`:

```python
from .ppo_custom import PPO_Custom

__all__.append("PPO_Custom")
```

## 步骤 2: 创建演员-评论家模块

演员-评论家模块为策略和价值函数定义神经网络架构。

### 2.1 基本结构

在 `rsl_rl/modules/` 中创建一个新文件:

```python
# rsl_rl/modules/actor_critic_custom.py
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticCustom(nn.Module):
    """PPO_Custom 的自定义演员-评论家网络。"""
    
    is_recurrent = False  # 对于 RNN 策略设置为 True
    
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
        
        # 获取激活函数
        activation_fn = self._get_activation(activation)
        
        # 构建演员网络
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
        
        # 构建评论家网络
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
        
        # 动作噪声参数
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # 为提高速度禁用验证
        Normal.set_default_validate_args = False
    
    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """按名称获取激活函数。"""
        activations = {
            'elu': nn.ELU(),
            'relu': nn.ReLU(),
            'selu': nn.SELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
        }
        return activations.get(name, nn.ELU())
```

### 2.2 实现必需方法

```python
def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
    """从策略中采样动作。"""
    self.update_distribution(observations)
    return self.distribution.sample()

def update_distribution(self, observations: torch.Tensor) -> None:
    """更新动作分布。"""
    mean = self.actor(observations)
    self.distribution = Normal(mean, mean * 0.0 + self.std)

def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    """获取动作的对数概率。"""
    return self.distribution.log_prob(actions).sum(dim=-1)

def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
    """评估价值函数。"""
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
    """为循环策略重置隐藏状态。"""
    pass  # 非循环策略无操作
```

### 2.3 模块注册

添加到 `rsl_rl/modules/__init__.py`:

```python
from .actor_critic_custom import ActorCriticCustom
```

## 步骤 3: 创建存储类

存储类管理用于经验收集的回合缓冲区。大多数情况下请扩展 `RolloutStorage`。

### 3.1 基本结构

在 `rsl_rl/storage/` 中创建一个新文件:

```python
# rsl_rl/storage/rollout_storage_custom.py
import torch
from .rollout_storage import RolloutStorage


class RolloutStorageCustom(RolloutStorage):
    """PPO_Custom 的自定义回合存储。"""
    
    class Transition(RolloutStorage.Transition):
        """用于自定义观测的扩展转换。"""
        
        def __init__(self):
            super().__init__()
            # 添加自定义观测字段
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
            privileged_obs_shape,  # 用作特权观测
            actions_shape, 
            device
        )
        
        # 用于自定义观测的额外存储
        self.critic_obs_shape = critic_obs_shape
        self.critic_observations = torch.zeros(
            num_transitions_per_env, 
            num_envs, 
            *critic_obs_shape, 
            device=self.device
        )
```

### 3.2 重写存储方法

```python
def add_transitions(self, transition: Transition) -> None:
    """将转换添加到缓冲区。"""
    if self.step >= self.num_transitions_per_env:
        raise AssertionError("回合缓冲区溢出")
    
    # 存储所有转换数据
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
    """为训练生成小批量。"""
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(
        num_mini_batches * mini_batch_size, 
        requires_grad=False, 
        device=self.device
    )
    
    # 展平所有缓冲区
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
                (None, None),  # RNN 的隐藏状态
                None,  # RNN 的掩码
            )
```

### 3.3 存储注册

添加到 `rsl_rl/storage/__init__.py`:

```python
from .rollout_storage_custom import RolloutStorageCustom
```

## 步骤 4: 创建运行器类

运行器编排训练循环、处理日志记录并管理检查点。

### 4.1 基本结构

在 `rsl_rl/runners/` 中创建一个新文件:

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
    """用于 PPO_Custom 训练的自定义运行器。"""
    
    def __init__(
        self,
        env: VecEnv,
        train_cfg: Dict[str, Any],
        log_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
    
    def _init_agent_and_algo(self) -> None:
        """初始化演员-评论家和算法。"""
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCriticCustom = actor_critic_class(
            self.env.num_obs,
            self.env.num_actions,
            self.env.num_critic_obs,  # 必须在环境中定义
            **self.policy_cfg
        ).to(self.device)
        
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO_Custom = alg_class(
            actor_critic, 
            device=self.device, 
            **self.alg_cfg
        )
    
    def _init_storage(self) -> None:
        """用正确的形状初始化存储。"""
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            (self.env.num_obs,),
            (self.env.num_privileged_obs,),
            (self.env.num_critic_obs,),
            (self.env.num_actions,),
        )
```

### 4.2 重写训练循环

```python
def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
) -> None:
    """主训练循环。"""
    # 预学习设置
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint(
            0, 
            self.env.max_episode_length, 
            (self.env.num_envs,),
            device=self.env.device
        )
    
    # 获取初始观测
    obs, privileged_obs, critic_obs = self.env.get_observations()
    obs = obs.to(self.device)
    privileged_obs = privileged_obs.to(self.device)
    critic_obs = critic_obs.to(self.device)
    
    self.alg.actor_critic.train()
    
    # 回合跟踪缓冲区
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
    
    # 主训练循环
    tot_iter = self.current_learning_iteration + num_learning_iterations
    for it in range(self.current_learning_iteration, tot_iter):
        start = time.time()
        
        # 回合收集
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
                
                # 回合跟踪
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
        
        # 学习步骤
        start = time.time()
        self.alg.compute_returns(critic_obs)
        mean_value_loss, mean_surrogate_loss = self.alg.update()
        learn_time = time.time() - start
        
        # 日志记录
        if self.log_dir is not None:
            self.log(locals())
        
        # 检查点
        if it % self.save_interval == 0:
            self.save(f"{self.log_dir}/model_{it}.pt")
        
        ep_infos.clear()
    
    self.current_learning_iteration += num_learning_iterations
    self.save(f"{self.log_dir}/model_{self.current_learning_iteration}.pt")
```

### 4.3 实现推理策略

```python
def get_inference_policy(
    self,
    device: Optional[Union[str, torch.device]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """获取用于推理/部署的策略。"""
    self.alg.actor_critic.eval()
    if device is not None:
        self.alg.actor_critic.to(device)
    return self.alg.actor_critic.act
```

## 步骤 5: 注册运行器

在 `rsl_rl/runners/__init__.py` 中注册运行器:

```python
from .custom_runner import CustomRunner
from rsl_rl.utils.runner_registry import runner_registry

runner_registry.register("CustomRunner", CustomRunner)
```

## 步骤 6: 创建环境变体 (可选)

如果你的方法需要特殊的环境修改,请在 `legged_gym/envs/` 中创建一个变体。

### 6.1 定义配置

```python
# legged_gym/envs/go2/go2_custom/go2_custom_config.py
from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg, 
    LeggedRobotCfgPPO
)


class GO2CustomCfg(LeggedRobotCfg):
    """使用自定义方法的 GO2 配置。"""
    
    class env(LeggedRobotCfg.env):
        pass
    
    class runner(LeggedRobotCfg.runner):
        runner_class_name = "CustomRunner"
        policy_class_name = "ActorCriticCustom"
        algorithm_class_name = "PPO_Custom"


class GO2CustomCfgPPO(LeggedRobotCfgPPO):
    """GO2 自定义方法的 PPO 配置。"""
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        custom_param = 0.5  # 自定义参数
```

### 6.2 注册任务

添加到 `legged_gym/envs/__init__.py`:

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

## 组件兼容性要求

理解组件兼容性对于实现新方法至关重要。

### 存储-算法匹配

存储必须提供算法需要的所有观测:

| 算法 | 必需的存储字段 |
|------|--------------|
| `PPO` | `observations`, `privileged_observations` |
| `PPO_TS` | `observations`, `privileged_observations`, `observation_histories`, `critic_observations` |
| `PPO_EE` | `observations`, `privileged_observations`, `estimated_states` |
| `PPO_AMP` | `observations`, `privileged_observations`, `reference_motion_obs` |

### 演员-评论家接口匹配

演员-评论家必须实现算法调用的方法:

| 方法 | 调用者 | 必须返回 |
|------|--------|---------|
| `act(obs, ...)` | Algorithm.act() | 动作张量 |
| `evaluate(critic_obs, ...)` | Algorithm.act() | 价值张量 |
| `get_actions_log_prob(actions)` | Algorithm.update() | 对数概率张量 |
| `action_mean`, `action_std` | Algorithm.act() | 分布参数 |

### 运行器-环境匹配

环境必须提供运行器期望的观测形状:

```python
# 环境必须定义这些属性
self.num_obs = 48
self.num_privileged_obs = 187
self.num_critic_obs = 187  # 用于 TS 方法
self.num_history_obs = 480  # 用于 TS 方法
self.num_latent_dims = 64   # 用于基于编码器的方法
```

## 完整示例: 教师-学生实现

教师-学生 (TS) 方法演示所有组件协同工作:

### 算法 (PPO_TS)

```python
class PPO_TS(PPO):
    actor_critic: ActorCriticTS
    storage: RolloutStorageTS
    
    def __init__(self, actor_critic, encoder_lr=1e-3, ...):
        super().__init__(actor_critic, ...)
        # 历史编码器的单独优化器
        self.history_encoder_optimizer = optim.Adam(
            actor_critic.history_encoder.parameters(), 
            lr=encoder_lr
        )
    
    def act(self, obs, privileged_obs, obs_history, critic_obs):
        # 存储所有观测类型
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        self.transition.critic_observations = critic_obs
        # 用特权编码器计算动作
        self.transition.actions = self.actor_critic.act(
            obs, privileged_obs
        ).detach()
        return self.transition.actions
    
    def update(self):
        # 标准 PPO 更新
        mean_value_loss, mean_surrogate_loss = self._ppo_update()
        # 额外的编码器蒸馏
        mean_encoder_loss = self._encoder_update()
        return mean_value_loss, mean_surrogate_loss, mean_encoder_loss
```

### 演员-评论家 (ActorCriticTS)

```python
class ActorCriticTS(nn.Module):
    def __init__(self, num_actor_obs, num_actions, 
                 num_privilege_encoder_input, num_history_encoder_input,
                 num_latent_dims, num_critic_obs, ...):
        # 特权编码器: privileged_obs -> 潜在
        self.privilege_encoder = nn.Sequential(...)
        
        # 历史编码器: obs_history -> 潜在 (学生)
        self.history_encoder = nn.Sequential(...)
        
        # 演员: [obs, latent] -> 动作
        self.actor = nn.Sequential(...)
        
        # 评论家: critic_obs -> 价值
        self.critic = nn.Sequential(...)
    
    def act(self, observations, privilege_observations):
        latent = self.privilege_encoder(privilege_observations)
        mean = self.actor(torch.cat([observations, latent], dim=-1))
        self.distribution = Normal(mean, self.std)
        return self.distribution.sample()
    
    def act_student(self, observations, observation_history):
        # 用于部署: 改用历史编码器
        latent = self.history_encoder(observation_history)
        mean = self.actor(torch.cat([observations, latent], dim=-1))
        return mean
```

### 存储 (RolloutStorageTS)

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
        # TS 特定观测的额外缓冲区
        self.observation_histories = torch.zeros(...)
        self.critic_observations = torch.zeros(...)
```

### 运行器 (TSRunner)

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
        # 获取所有观测类型
        obs, privileged_obs, obs_history, critic_obs = \
            self.env.get_observations()
        # 带 4 种观测类型的训练循环
        ...
    
    def get_inference_policy(self, device=None):
        # 返回用于部署的学生策略
        return self.alg.actor_critic.act_student
```

## 测试你的实现

实现所有组件后,用以下命令测试:

```bash
# 用少量环境快速测试
python -m legged_gym.scripts.train --task go2_custom --num_envs 10 --headless

# 完整训练
python -m legged_gym.scripts.train --task go2_custom --headless

# 推理测试
python -m legged_gym.scripts.play --task go2_custom
```

## 总结

实现新的强化学习方法需要:

1. **算法**: 扩展 `BaseAlgorithm` 或 `PPO`,实现 `init_storage()`, `act()`, `update()`
2. **演员-评论家**: 构建神经网络,实现 `act()`, `evaluate()`, 分布方法
3. **存储**: 扩展 `RolloutStorage`,添加自定义观测缓冲区
4. **运行器**: 扩展 `OnPolicyRunner`,自定义初始化和训练循环
5. **注册**: 在 `runners/__init__.py` 中注册运行器
6. **环境**: 如果需要则创建配置变体,注册任务

始终确保组件兼容性:算法必须匹配存储格式,演员-评论家必须实现算法期望的方法。参考 `PPO_TS` 等现有实现以获取完整的工作示例。
