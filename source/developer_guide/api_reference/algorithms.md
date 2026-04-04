# PPO算法变体

本文档提供了 LeggedGym-Ex 中实现的 PPO（近端策略优化）算法变体的 API 参考。每种变体都针对特定的运动挑战而设计，从仿真到现实的迁移到从运动演示中学习。

```{note}
所有算法类都继承自 `PPO` 基类，并遵循一致的接口，包括 `init_storage()`、`act()`、`process_env_step()`、`compute_returns()` 和 `update()` 方法。
```

## 基础 PPO 类

### 类概述

`PPO` 类实现了近端策略优化算法，支持标准 PPO 和 SPO（简单策略优化）两种模式。它作为所有变体实现的基础。

**文件位置**: `rsl_rl/algorithms/ppo.py`

### 初始化

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

### 关键参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `actor_critic` | ActorCritic | 必需 | Actor-Critic 网络 |
| `num_learning_epochs` | int | 1 | 每次更新时的优化轮数 |
| `num_mini_batches` | int | 1 | SGD 的小批量数量 |
| `clip_param` | float | 0.2 | PPO 裁剪参数（epsilon） |
| `gamma` | float | 0.998 | 折扣因子 |
| `lam` | float | 0.95 | GAE lambda 参数 |
| `value损失系数` | float | 1.0 | 价值函数损失系数 |
| `entropy_coef` | float | 0.0 | 熵奖励系数 |
| `learning_rate` | float | 1e-3 | 优化器学习率 |
| `max_grad_norm` | float | 1.0 | 梯度裁剪的最大范数 |
| `use_clipped_value_loss` | bool | True | 是否裁剪价值函数更新 |
| `schedule` | str | "fixed" | 学习率调度策略: "fixed" 或 "adaptive" |
| `desired_kl` | float | 0.01 | 自适应调度的目标 KL 散度 |
| `use_spo` | bool | False | 使用简单策略优化而非 PPO |

### 核心方法

#### init_storage()

初始化 rollout 存储缓冲区以收集轨迹。

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

**参数:**
- `num_envs`: 并行环境的数量
- `num_transitions_per_env`: 每个环境要存储的步数（rollout 长度）
- `actor_obs_shape`: Actor 观察值的形状
- `critic_obs_shape`: Critic 观察值的形状
- `action_shape`: 动作的形状

#### act()

在 rollout 收集期间为给定观察值计算动作。

```python
def act(
    self,
    obs: torch.Tensor,
    critic_obs: torch.Tensor
) -> torch.Tensor
```

**参数:**
- `obs`: Actor 观察值，形状为 `[num_envs, obs_dim]`
- `critic_obs`: Critic 观察值，形状为 `[num_envs, critic_obs_dim]`

**返回:**
- `actions`: 采样的动作，形状为 `[num_envs, action_dim]`

#### process_env_step()

处理环境步骤结果并存储转移。

```python
def process_env_step(
    self,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    infos: Dict[str, Any]
) -> None
```

**参数:**
- `rewards`: 来自环境的奖励，形状为 `[num_envs]`
- `dones`: 完成标志，形状为 `[num_envs]`
- `infos`: 信息字典，可能包含用于 bootstrapping 的 'time_outs'

#### compute_returns()

使用广义优势估计（GAE）计算回报和优势。

```python
def compute_returns(
    self,
    last_critic_obs: torch.Tensor
) -> None
```

**参数:**
- `last_critic_obs`: 用于 bootstrapping 的最终 Critic 观察值，形状为 `[num_envs, critic_obs_dim]`

#### update()

使用收集的经验更新策略。

```python
def update() -> Tuple[float, float]
```

**返回:**
- `mean_value_loss`: 平均价值函数损失
- `mean_surrogate_loss`: 平均替代损失

---

## PPO_TS (教师-学生)

教师-学生变体实现了从特权教师策略到仅使用可观察信息的学生策略的知识蒸馏。这使得学生通过学习模仿教师的潜在表示来实现仿真到现实的迁移。

**论文参考**: [Rapid Locomotion via Reinforcement Learning](https://agility.csail.mit.edu/)

**文件位置**: `rsl_rl/algorithms/ppo_ts.py`

### 独特特性

- **双网络架构**: 教师使用特权观察值；学生使用历史编码的观察值
- **历史编码器**: 从观察历史中提取特权信息（支持 MLP 或 TCN）
- **特权编码器**: 将特权观察值编码为潜在表示
- **独立优化器**: 一个用于 RL 参数，一个用于历史编码器

### 初始化

```python
PPO_TS(
    actor_critic: ActorCriticTS,
    # ... 基础 PPO 参数 ...
    encoder_lr: float = 1e-3,
    num_encoder_epochs: int = 1,
)
```

### 附加参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `encoder_lr` | float | 1e-3 | 历史编码器的学习率 |
| `num_encoder_epochs` | int | 1 | 每次更新时编码器训练的轮数 |

### 关键方法

#### act()

使用教师-学生架构计算动作。

```python
def act(
    self,
    obs: torch.Tensor,
    privileged_obs: torch.Tensor,
    obs_history: torch.Tensor,
    critic_obs: torch.Tensor
) -> torch.Tensor
```

**参数:**
- `obs`: Actor 观察值
- `privileged_obs`: 特权观察值（真实状态）
- `obs_history`: 用于编码器输入的观察历史
- `critic_obs`: Critic 观察值

#### update()

除基础损失外，还返回编码器损失。

```python
def update() -> Tuple[float, float, float]
```

**返回:**
- `mean_value_loss`: 价值函数损失
- `mean_surrogate_loss`: 替代损失
- `mean_encoder_loss`: 历史编码器蒸馏损失

### 所需存储

使用 `RolloutStorageTS`，它存储：
- `privileged_observations`: 真实状态
- `observation_histories`: 用于编码器训练的历史

---

## PPO_EE (显式估计器)

显式估计器变体训练一个状态估计器与策略同时训练。估计器从可观察的历史中预测特权信息（如基础速度、地形高度）。

**论文参考**: [Concurrent Training of a Control Policy and a State Estimator](https://arxiv.org/abs/2202.05481)

**文件位置**: `rsl_rl/algorithms/ppo_ee.py`

### 独特特性

- **显式状态估计器**: 估计特权状态的神经网络
- **并发训练**: 策略和估计器一起训练
- **MSE 损失**: 估计器预测的监督学习

### 初始化

```python
PPO_EE(
    actor_critic: ActorCriticEE,
    # ... 基础 PPO 参数 ...
    estimator_lr: float = 1e-3,
    num_estimator_epochs: int = 1,
)
```

### 附加参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `estimator_lr` | float | 1e-3 | 估计器网络的学习率 |
| `num_estimator_epochs` | int | 1 | 估计器训练的轮数 |

### 关键方法

#### act()

计算动作并记录估计器特征。

```python
def act(
    self,
    estimator_features: torch.Tensor,
    critic_obs: torch.Tensor,
    estimator_labels: torch.Tensor
) -> torch.Tensor
```

**参数:**
- `estimator_features`: 估计器的输入特征（历史）
- `critic_obs`: Critic 观察值
- `estimator_labels`: 监督用的真实标签

#### update()

除基础损失外，还返回估计器损失。

```python
def update() -> Tuple[float, float, float]
```

**返回:**
- `mean_value_loss`: 价值函数损失
- `mean_surrogate_loss`: 替代损失
- `mean_estimator_loss`: 状态估计器 MSE 损失

---

## PPO_CTS (并发教师-学生)

并发教师-学生变体在同一批次中同时训练教师和学生策略，与顺序教师-学生方法相比，提高了样本效率和训练稳定性。

**论文参考**: [CTS: Concurrent Teacher-Student Reinforcement Learning](https://clearlab-sustech.github.io/concurrentTS/)

**文件位置**: `rsl_rl/algorithms/ppo_cts.py`

### 独特特性

- **并发训练**: 教师和学生环境并行运行
- **共享存储**: 具有教师/学生分区的单一存储缓冲区
- **双重替代损失**: 分别计算教师和学生策略的损失

### 初始化

```python
PPO_CTS(
    actor_critic: ActorCriticCTS,
    # ... 基础 PPO 参数 ...
    encoder_lr: float = 1e-3,
    num_encoder_epochs: int = 1,
    num_teacher: int = 1,
)
```

### 附加参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `encoder_lr` | float | 1e-3 | 历史编码器的学习率 |
| `num_encoder_epochs` | int | 1 | 编码器训练的轮数 |
| `num_teacher` | int | 1 | 教师环境的数量 |

### 关键方法

#### act()

为教师和学生环境计算动作。

```python
def act(
    self,
    obs: torch.Tensor,
    privileged_obs: torch.Tensor,
    obs_history: torch.Tensor,
    critic_obs: torch.Tensor
) -> torch.Tensor
```

前 `num_teacher` 个环境使用教师动作；其余使用学生动作。

#### update()

分别返回教师和学生损失。

```python
def update() -> Tuple[float, float, float, float]
```

**返回:**
- `mean_value_loss`: 价值函数损失
- `mean_teacher_surrogate_loss`: 教师替代损失
- `mean_student_surrogate_loss`: 学生替代损失
- `mean_reconstruction_loss`: 编码器重建损失

---

## PPO_AMP (对抗运动先验)

AMP 变体使学习能够使用对抗判别器从动作捕捉数据中学习自然的运动方式。判别器区分策略生成的动作和专家动作片段。

**论文参考**: [AMP: Adversarial Motion Priors](https://arxiv.org/abs/2104.02180)

**文件位置**: `rsl_rl/algorithms/ppo_amp.py`

### 独特特性

- **判别器网络**: 对策略与专家动作进行分类
- **动作回放缓冲区**: 存储专家动作片段
- **风格奖励**: 判别器输出用作额外的奖励信号
- **对称性支持**: 对称步态的可选对称性损失
- **梯度惩罚**: 稳定判别器训练

### 初始化

```python
PPO_AMP(
    actor_critic: ActorCritic,
    discriminator: AMPDiscriminator,
    amp_data: ReplayBuffer,
    amp_normalizer: Optional[Normalizer],
    # ... 基础 PPO 参数 ...
    amp_replay_buffer_size: int = 100000,
    disc_lr: float = 1e-4,
    symmetry_cfg: Optional[Dict] = None,
)
```

### 附加参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `discriminator` | AMPDiscriminator | 必需 | 动作判别器网络 |
| `amp_data` | ReplayBuffer | 必需 | 专家动作数据缓冲区 |
| `amp_normalizer` | Normalizer | None | AMP 观察值的可选归一化器 |
| `amp_replay_buffer_size` | int | 100000 | 策略动作回放缓冲区的大小 |
| `disc_lr` | float | 1e-4 | 判别器学习率 |
| `symmetry_cfg` | Dict | None | 对称性配置 |

### 关键方法

#### act()

计算动作并记录 AMP 观察值。

```python
def act(
    self,
    obs: torch.Tensor,
    critic_obs: torch.Tensor,
    amp_obs: torch.Tensor
) -> torch.Tensor
```

**参数:**
- `obs`: Actor 观察值
- `critic_obs`: Critic 观察值
- `amp_obs`: AMP 观察值（身体姿态、速度等）

#### process_env_step()

处理带有 AMP 观察值存储的步骤。

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

返回判别器训练的详细指标。

```python
def update() -> Tuple[float, float, float, float, float, float, Optional[float]]
```

**返回:**
- `mean_value_loss`: 价值函数损失
- `mean_surrogate_loss`: 替代损失
- `mean_amp_loss`: AMP 判别器损失
- `mean_grad_pen_loss`: 梯度惩罚损失
- `mean_policy_pred`: 策略样本的判别器预测
- `mean_expert_pred`: 专家样本的判别器预测
- `mean_symmetry_loss`: 对称性损失（如果启用）

---

## PPO_DreamWaQ

DreamWaQ 变体使用基于 VAE 的架构来学习地形想象——从观察历史预测未来状态。这使得在未见过地形上的鲁棒运动成为可能。

**论文参考**: [DreamWaQ: Learning Robust Quadrupedal Locomotion](https://arxiv.org/abs/2301.10602)

**文件位置**: `rsl_rl/algorithms/ppo_dreamwaq.py`

### 独特特性

- **VAE 架构**: 用于地形想象的变分自编码器
- **隐式地形估计**: 不需要显式地形传感器
- **显式状态预测**: 预测身体速度和地形信息
- **KL 散度正则化**: VAE 潜在空间正则化

### 初始化

```python
PPO_DreamWaQ(
    actor_critic: ActorCriticDreamWaQ,
    # ... 基础 PPO 参数 ...
    encoder_lr: float = 1e-3,
    num_encoder_epochs: int = 1,
    vae_kld_weight: float = 1.0,
)
```

### 附加参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `encoder_lr` | float | 1e-3 | VAE 编码器的学习率 |
| `num_encoder_epochs` | int | 1 | VAE 训练的轮数 |
| `vae_kld_weight` | float | 1.0 | KL 散度损失的权重 |

### 关键方法

#### act()

计算动作并记录 VAE 输入。

```python
def act(
    self,
    obs: torch.Tensor,
    privileged_obs: torch.Tensor,
    obs_history: torch.Tensor,
    explicit_info_labels: torch.Tensor
) -> torch.Tensor
```

**参数:**
- `obs`: Actor 观察值
- `privileged_obs`: 用于 Critic 的特权观察值
- `obs_history`: 用于 VAE 的观察历史
- `explicit_info_labels`: 显式状态预测的标签

#### process_env_step()

存储用于重建损失的下一状态。

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

返回与 VAE 相关的损失。

```python
def update() -> Tuple[float, float, float, float, float]
```

**返回:**
- `mean_value_loss`: 价值函数损失
- `mean_surrogate_loss`: 替代损失
- `mean_explicit_estimation_loss`: 显式状态预测损失
- `mean_reconstruction_loss`: 状态重建损失
- `mean_kld_loss`: KL 散度损失

---

## Runner 类

Runners 编排训练循环，管理环境交互、数据收集和算法更新。

### OnPolicyRunner

用于同策略 RL 训练的基础 Runner。管理训练循环、日志记录和模型检查点。

**文件位置**: `rsl_rl/runners/on_policy_runner.py`

#### 初始化

```python
OnPolicyRunner(
    env: VecEnv,
    train_cfg: Dict[str, Any],
    log_dir: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
)
```

#### 关键方法

##### learn()

运行训练循环。

```python
def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
) -> None
```

**参数:**
- `num_learning_iterations`: 训练迭代次数
- `init_at_random_ep_len`: 随机化初始回合长度

##### save() / load()

检查点管理。

```python
def save(self, path: str, infos: Optional[Dict] = None) -> None
def load(self, path: str, load_optimizer: bool = True) -> Optional[Dict]
```

##### get_inference_policy()

获取用于部署的策略函数。

```python
def get_inference_policy(
    self,
    device: Optional[Union[str, torch.device]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]
```

### TSRunner

专门用于教师-学生训练的 Runner。处理观察历史和特权信息。

**文件位置**: `rsl_rl/runners/ts_runner.py`

#### 与基础 Runner 的关键区别

- `get_observations()` 返回元组 `(obs, privileged_obs, obs_history, critic_obs)`
- `get_inference_policy()` 返回学生策略（而非教师）

### EERunner

专门用于显式估计器训练的 Runner。管理估计器特征和标签。

**文件位置**: `rsl_rl/runners/ee_runner.py`

#### 与基础 Runner 的关键区别

- `get_observations()` 返回元组 `(estimator_features, estimator_labels, privileged_obs)`
- 记录估计器损失指标

---

## 训练流程

以下描述了同策略 PPO 变体的训练流程：

```
┌─────────────────────────────────────────────────────────────────┐
│                      训练迭代                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 初始化                                               │
│     ├── runner._init_agent_and_algo()                           │
│     │   └── 创建 Actor-Critic 网络                         │
│     │   └── 创建 PPO 算法实例                           │
│     └── runner._init_storage()                                  │
│         └── alg.init_storage() -> RolloutStorage                │
│                                                                  │
│  2. ROLLOUT 收集（重复 N 步）                         │
│     ├── alg.act(obs, critic_obs) -> actions                     │
│     ├── env.step(actions) -> obs, rewards, dones, infos         │
│     └── alg.process_env_step(rewards, dones, infos)             │
│         └── storage.add_transitions(transition)                 │
│                                                                  │
│  3. 回报计算                                         │
│     └── alg.compute_returns(last_critic_obs)                    │
│         └── GAE: A_t = Σ (γλ)^l * δ_{t+l}                       │
│         └── Returns: R_t = A_t + V(s_t)                         │
│                                                                  │
│  4. 策略更新（重复 K 轮 × M 小批量）              │
│     ├── 从存储中获取每个小批量：                       │
│     │   ├── 前向传递通过 Actor-Critic              │
│     │   ├── 计算比率: π(a|s) / π_old(a|s)                  │
│     │   ├── 替代损失: max(L^CLIP, L^CLIP')                │
│     │   ├── 价值损失: (V(s) - R)^2                            │
│     │   ├── 熵奖励: -β * H(π(·|s))                       │
│     │   └── optimizer.step()                                    │
│     │                                                            │
│     └── 对于带编码器的变体：                          │
│         ├── 计算编码器损失 (MSE)                          │
│         └── encoder_optimizer.step()                            │
│                                                                  │
│  5. 日志记录与检查点                                      │
│     ├── runner.log(metrics)                                     │
│     │   └── TensorBoard / WandB 日志记录                 │
│     └── runner.save() 如果到达检查点间隔              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 关键公式

**PPO 裁剪目标:**
```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```
其中 `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`

**广义优势估计:**
```
A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**总损失:**
```
L = L^CLIP + c_1 * L^VF - c_2 * H(π)
```

---

## 算法配置参数

### 通用 PPO 参数

所有变体共享这些基础配置参数，位于 `cfg.algorithm` 下：

```python
class LeggedRobotCfgPPO:
    class algorithm:
        value_loss_coef = 1.0          # 价值函数损失权重
        use_clipped_value_loss = True  # 裁剪价值更新
        clip_param = 0.2               # PPO 裁剪 epsilon
        entropy_coef = 0.01            # 熵奖励权重
        num_learning_epochs = 5        # 每次迭代的轮数
        num_mini_batches = 4           # 每轮的小批量数
        learning_rate = 1.e-3          # Adam 学习率
        schedule = 'adaptive'          # 学习率调度
        gamma = 0.99                   # 折扣因子
        lam = 0.95                     # GAE lambda
        desired_kl = 0.01              # 目标 KL 散度
        max_grad_norm = 1.0            # 梯度裁剪
```

### 变体特定参数

#### 教师-学生 (PPO_TS)

```python
class algorithm:
    # ... 基础参数 ...
    encoder_lr = 1e-3              # 历史编码器学习率
    num_encoder_epochs = 1         # 每次更新的编码器轮数
```

#### 显式估计器 (PPO_EE)

```python
class algorithm:
    # ... 基础参数 ...
    estimator_lr = 1e-3            # 估计器学习率
    num_estimator_epochs = 1        # 估计器轮数
```

#### 并发 TS (PPO_CTS)

```python
class algorithm:
    # ... 基础参数 ...
    encoder_lr = 1e-3              # 编码器学习率
    num_encoder_epochs = 1          # 编码器轮数
    num_teacher = 1                 # 教师环境数量
```

#### AMP (PPO_AMP)

```python
class algorithm:
    # ... 基础参数 ...
    disc_lr = 1e-4                  # 判别器学习率
    amp_replay_buffer_size = 100000 # 策略缓冲区大小
```

#### DreamWaQ (PPO_DreamWaQ)

```python
class algorithm:
    # ... 基础参数 ...
    encoder_lr = 1e-3              # VAE 编码器学习率
    num_encoder_epochs = 1          # VAE 轮数
    vae_kld_weight = 1.0           # KL 散度权重
```

### Runner 参数

位于 `cfg.runner` 下的配置：

```python
class runner:
    policy_class_name = "ActorCritic"     # 网络类
    algorithm_class_name = "PPO"          # 算法类
    num_steps_per_env = 24                 # Rollout 长度
    max_iterations = 1500                  # 总迭代次数
    save_interval = 50                     # 检查点间隔
    experiment_name = "test"               # 日志目录名
    run_name = ""                          # 运行标识符
    resume = False                         # 从检查点恢复
    load_run = -1                          # 要加载的运行 ID
    checkpoint = -1                        # 检查点 ID
    sync_wandb = False                     # 启用 WandB 同步
```

---

## 使用示例

### 使用基础 PPO 训练

```python
from rsl_rl.runners import OnPolicyRunner

# 初始化 Runner
runner = OnPolicyRunner(
    env=env,
    train_cfg=train_cfg,
    log_dir=log_dir,
    device="cuda"
)

# 训练
runner.learn(num_learning_iterations=1500)

# 获取推理策略
policy = runner.get_inference_policy(device="cpu")
```

### 使用教师-学生训练

```python
from rsl_rl.runners import TSRunner

# TSRunner 自动使用 PPO_TS 和 ActorCriticTS
runner = TSRunner(
    env=env,
    train_cfg=train_cfg,
    log_dir=log_dir,
    device="cuda"
)

# 使用蒸馏训练
runner.learn(num_learning_iterations=1500)

# 获取用于部署的学生策略
student_policy = runner.get_inference_policy()
```

### 使用 AMP 训练

```python
from rsl_rl.algorithms import PPO_AMP
from rsl_rl.modules import AMPDiscriminator
from rsl_rl.storage import ReplayBuffer

# 创建判别器
discriminator = AMPDiscriminator(input_dim=amp_obs_dim * 2)

# 创建算法
alg = PPO_AMP(
    actor_critic=actor_critic,
    discriminator=discriminator,
    amp_data=expert_motion_buffer,
    amp_normalizer=normalizer,
    device="cuda"
)

# 带 AMP 特定损失的标准训练循环
```

---

## 组件兼容性矩阵

| 算法 | Actor-Critic | 存储 | Runner |
|-----------|--------------|---------|--------|
| PPO | ActorCritic | RolloutStorage | OnPolicyRunner |
| PPO_TS | ActorCriticTS | RolloutStorageTS | TSRunner |
| PPO_EE | ActorCriticEE | RolloutStorageEE | EERunner |
| PPO_CTS | ActorCriticCTS | RolloutStorageCTS | CTSRunner |
| PPO_AMP | ActorCritic | RolloutStorage | AMPRunner |
| PPO_DreamWaQ | ActorCriticDreamWaQ | RolloutStorageDreamWaQ | DreamWaQRunner |

```{warning}
使用不兼容的组件（例如，将基础 RolloutStorage 与 PPO_TS 一起使用）将导致运行时错误。始终根据上表匹配算法、存储和 Runner 类。
```

---

## 参考文献

- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **SPO**: [Simple Policy Optimization](https://arxiv.org/abs/2401.16025)
- **教师-学生**: [Rapid Locomotion via RL](https://agility.csail.mit.edu/)
- **显式估计器**: [Concurrent Training of Control Policy and State Estimator](https://arxiv.org/abs/2202.05481)
- **CTS**: [Concurrent Teacher-Student RL](https://clearlab-sustech.github.io/concurrentTS/)
- **AMP**: [Adversarial Motion Priors](https://arxiv.org/abs/2104.02180)
- **DreamWaQ**: [DreamWaQ: Learning Robust Quadrupedal Locomotion](https://arxiv.org/abs/2301.10602)
