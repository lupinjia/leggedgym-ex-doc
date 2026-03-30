# 🧑‍🏫🧑‍🏫 师生框架 (Teacher-Student Framework)

在 [Walk These Ways](walk_these_ways.md) 中，我们知道通过调节机器人行为，我们可以让在平地上训练的策略穿越一些困难地形（如小路缘石和楼梯）。这表明了策略的泛化能力，因为它只在平地上训练，训练数据中不包含复杂地形的数据。然而，为了在复杂地形上实现更好的稳定性和可穿越性，在地形上训练是不可避免的。但是正如在 [部署到真实机器人](https://genesis-lr.readthedocs.io/en/latest/user_guide/getting_started/deploy_to_real_robot.html) 中所介绍的，仅仅使用一个获取单时间步机器人状态反馈的简单网络是不够的。实际上，如果你使用 RNN（循环神经网络, Recurrent Neural Network）或堆叠多个时间步的观测，策略将获得更好的地形适应能力。然而，我们仍然需要某种机制来有效地从历史观测中提取关键信息，并使策略更好地理解机器人和周围环境。

## [师生框架 (Teacher-Student Framework)](https://arxiv.org/abs/2010.11251)

作为基于强化学习 (Reinforcement Learning, RL) 的足式机器人控制领域的先驱之一，[苏黎世联邦理工学院的 RSL 实验室](https://rsl.ethz.ch/) 提出了一系列工作来推动四足机器人在复杂地形上的性能极限。这里介绍的是师生框架。

直观上，我们理解有了特权信息 (Privilege Information)（如 base_lin_vel、friction_ratio、added_mass、base_com_bias 等），运动控制问题可以被视为 MDP（马尔可夫决策过程, Markov Decision Process）而不是 POMDP（部分可观测马尔可夫决策过程, Partially Observable Markov Decision Process）。策略可以利用更多有用信息找到更好的解决方案。然而，在现实世界中，特权信息是不可观测的。但是，它可以被估计。**师生框架的关键见解是特权信息可以从观测历史中估计出来。** 师生框架的简化图如下所示，其中 $x_t$ 是特权信息，$o_t^H=[o_t, o_{t-1},...,o_{t-H+1}]$ 是长度为 H 步的观测历史，$l_t$ 是教师编码器编码的潜在向量，$\hat{l}_t$ 是学生编码器预测的潜在向量，$a_t$ 是动作，$s_t$ 是环境状态，$\hat{V}_t$ 是 Critic 网络估计的值。

```{figure} ../../_static/images/ts_diagram.png
```

特权编码器（或教师编码器）将特权信息编码到相同维度的潜在空间中。历史编码器（或学生编码器）从观测历史中推断特权信息的潜在向量。策略观察当前观测和潜在向量，并输出动作。

原始的师生框架是一个两阶段训练框架，其中编码器和 Actor 是耦合的。第一阶段训练教师编码器和教师策略，第二阶段训练学生编码器和学生策略。为了简化训练过程，[RMA](https://ashish-kmr.github.io/rma-legged-robots/) 提出了解耦编码器和策略，只需要在第二阶段训练学生编码器。此外，[RLvRL](https://agility.csail.mit.edu/) 提出通过同时进行强化学习和监督学习来实现单阶段训练。上面的图示与 RLvRL 的方法相同。

## 实现

我们基于 RLvRL 实现了一个单阶段师生训练框架。与标准 Actor-Critic 相比的核心修改在 `actor_critic_ts.py` 和 `ppo_ts.py` 中。在 `actor_critic_ts.py` 中，我们添加了特权编码器和历史编码器作为神经网络模块。

```python
# actor_critic_ts.py
        # Privilege encoder
        privilege_encoder_layers = []
        privilege_encoder_layers.append(
            nn.Linear(num_privilege_encoder_input, privilege_encoder_hidden_dims[0]))
        privilege_encoder_layers.append(activation)
        for l in range(len(privilege_encoder_hidden_dims)):
            if l == len(privilege_encoder_hidden_dims) - 1:
                privilege_encoder_layers.append(
                    nn.Linear(privilege_encoder_hidden_dims[l], num_latent_dims))
            else:
                privilege_encoder_layers.append(nn.Linear(
                    privilege_encoder_hidden_dims[l], privilege_encoder_hidden_dims[l + 1]))
                privilege_encoder_layers.append(activation)
        self.privilege_encoder = nn.Sequential(*privilege_encoder_layers)

        # History encoder
        self.history_encoder_type = history_encoder_type
        history_encoder_layers = []
        if history_encoder_type == "MLP":
            history_encoder_layers.append(
                nn.Linear(num_history_encoder_input, history_encoder_hidden_dims[0]))
            history_encoder_layers.append(activation)
            for l in range(len(history_encoder_hidden_dims)):
                if l == len(history_encoder_hidden_dims) - 1:
                    history_encoder_layers.append(
                        nn.Linear(history_encoder_hidden_dims[l], num_latent_dims))
                else:
                    history_encoder_layers.append(
                        nn.Linear(history_encoder_hidden_dims[l], history_encoder_hidden_dims[l + 1]))
                    history_encoder_layers.append(activation)
            self.history_encoder = nn.Sequential(*history_encoder_layers)
        elif history_encoder_type == "TCN":
            in_channels = 1
            for l in range(len(history_encoder_channel_dims)):
                out_channels = history_encoder_channel_dims[l]
                padding = history_encoder_dilation[l]*(kernel_size-1)// 2
                history_encoder_layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=history_encoder_stride[l],
                              padding=padding,
                              dilation=history_encoder_dilation[l])
                )
                in_channels = out_channels
                num_history_encoder_input = (num_history_encoder_input - 1) // history_encoder_stride[l] + 1
            history_encoder_output_layer = nn.Linear(
                num_history_encoder_input * history_encoder_channel_dims[-1], history_encoder_final_layer_dim)
            history_encoder_output_activation = activation
            history_encoder_latent_layer = nn.Linear(
                history_encoder_final_layer_dim, num_latent_dims)
            history_encoder_layers.append(nn.Flatten())
            history_encoder_layers.append(history_encoder_output_layer)
            history_encoder_layers.append(history_encoder_output_activation)
            history_encoder_layers.append(history_encoder_latent_layer)
            self.history_encoder = nn.Sequential(*history_encoder_layers)
```

在训练中，我们通过强化学习更新特权编码器，通过监督学习更新历史编码器：

```python
            loss = surrogate_loss + self.value_loss_coef * \
                value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # history encoder gradient step
            for _ in range(self.num_encoder_epochs):
                if self.actor_critic.history_encoder_type == "TCN":
                    if obs_histories_batch.dim() == 2:
                        # input shape (batch_size, obs_history_len) -> (batch_size, 1, obs_history_len)
                        obs_histories_batch = obs_histories_batch.unsqueeze(1)
                encoder_predictions = self.actor_critic.history_encoder(obs_histories_batch)
                
                with torch.no_grad(): # don't backpropagate through the encoder targets
                    encoder_targets = self.actor_critic.privilege_encoder(privileged_obs_batch)

                encoder_loss = nn.functional.mse_loss( # use mse loss
                    encoder_predictions, encoder_targets)
                self.history_encoder_optimizer.zero_grad()
                encoder_loss.backward()
                self.history_encoder_optimizer.step()
```

## 非对称 Actor-Critic (Asymmetric Actor Critic, A2C)

为了弥合仿真到现实的差距 (Sim-to-Real Gap)，我们通常会在 Actor 的观测中添加噪声。如果我们给 Critic 相同的观测，那么 Critic 必须从带噪声的观测中估计值，这很困难。相反，我们可以给 Critic 从仿真中获得的无噪声特权状态，这样它就能更精确地估计值。这种 [非对称 Actor-Critic (Asymmetric Actor Critic, A2C)](https://arxiv.org/abs/1710.06542) 架构不会阻碍仿真到现实的迁移，因为 Critic 网络只在仿真中运行。

在 `go2_ts.py` 中，我们为 `critic_obs`（Critic 网络的输入）和 `privilege_obs`（特权编码器的输入）分别使用两个缓冲区。

```python
# In go2_ts.py
    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions                                                    # num_actions
        ), dim=-1)
        
        domain_randomization_info = torch.cat((
                    (self.simulator._friction_values - 
                    self.friction_value_offset),            # 1
                    self.simulator._added_base_mass,        # 1
                    self.simulator._base_com_bias,          # 3
                    self.simulator._rand_push_vels[:, :2],  # 2
                    (self.simulator._kp_scale - 
                     self.kp_scale_offset),                 # num_actions
                    (self.simulator._kd_scale - 
                     self.kd_scale_offset),                 # num_actions
                    self.simulator._joint_armature,         # 1
                    self.simulator._joint_stiffness,        # 1
                    self.simulator._joint_damping,          # 1
            ), dim=-1)
        
        # Critic observation
        critic_obs = torch.cat((
            self.obs_buf,                 # num_observations
            domain_randomization_info,    # 34
        ), dim=-1)
        if self.cfg.asset.obtain_link_contact_states:
            critic_obs = torch.cat(
                (
                    critic_obs,                         # previous
                    self.simulator.link_contact_states,  # contact states of thighs, calfs and feet (4+4+4)=12
                ),
                dim=-1,
            )
        if self.cfg.terrain.measure_heights: # 81
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            critic_obs = torch.cat((critic_obs, heights), dim=-1)
        self.critic_obs_deque.append(critic_obs)
        self.critic_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) -
                             1) * self.noise_scale_vec

        # push obs_buf to obs_history
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
        # Privileged observation, for privileged encoder
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                (
                    domain_randomization_info,                       # 34
                    self.simulator.height_around_feet.flatten(1,2),  # 9*number of feet
                    self.simulator.normal_vector_around_feet,        # 3*number of feet
                ),
                dim=-1,
            )
            if self.cfg.asset.obtain_link_contact_states:
                self.privileged_obs_buf = torch.cat(
                    (
                        self.privileged_obs_buf,                   # previous
                        self.simulator.link_contact_states,        # contact states of thighs, calfs and feet (4+4+4)=12
                    ),
                    dim=-1,
                )
```

## TCN（时序卷积网络, Temporal Convolutional Network）的消融实验

在原始的师生论文中，作者使用 TCN 作为学生编码器的主体，并将其计算效率与 GRU（门控循环单元, Gated Recurrent Unit）进行了比较。这里我们对 TCN 进行了消融研究，以查看使用 TCN 是否能带来更低的预测损失和更快的训练速度。我们将历史长度设置为 100（对应 2 秒的历史窗口）。如下图所示，我们可以看到总奖励几乎相同，但 TCN 的编码器损失（即预测损失）更大。同时，两者的数据收集时间 (collection_time) 相同，但 TCN 的学习时间 (learning_time) 几乎是 MLP 的五倍。这些结果说明了在这种情况下 MLP 相对于 TCN 的优越性。

```{figure} ../../_static/images/ts_total_reward_comparison.png
```
```{figure} ../../_static/images/ts_encoder_loss_comparison.png
```
```{figure} ../../_static/images/ts_collection_time_com.png
```
```{figure} ../../_static/images/ts_learning_time_com.png
```

## 并发师生框架 (Concurrent Teacher-Student, CTS)

简单师生框架的最大缺点是教师策略和学生策略之间的分布偏移 (Distribution Shift)。而且由于学生策略是通过监督学习而不是强化学习训练的，它容易出现分布外 (Out-of-Distribution, OOD) 问题，这将导致学生策略中出现不良行为。

为了解决这个缺点，研究人员提出了 [CTS (Concurrent Teacher-Student, 并发师生框架)](https://clearlab-sustech.github.io/concurrentTS/)。该框架的想法是让梯度除了通过 `teacher encoder -> policy` 路径流动外，还通过 `student encoder -> policy` 路径流动。这是通过将所有环境分成两组并考虑它们的替代损失来实现的。使用这种方法，可以增强最终学生策略的鲁棒性。我们提供的 CTS 实现可以在 [ppo_cts](https://github.com/lupinjia/LeggedGym-Ex/blob/main/rsl_rl/algorithms/ppo_cts.py) 中找到。

## 训练和运行

要训练一个师生策略，输入以下命令：
```bash
python train.py --task=go2_ts --headless
```

要运行它，输入以下命令：
```bash
python play.py --task=go2_ts --load_run=session_name
```

要训练 CTS 策略，你只需要将任务名称中的 `ts` 替换为 `cts`。

## 演示

我们在 [go2_deploy](https://github.com/lupinjia/go2_deploy/tree/main) 中提供了由师生框架训练的学生策略的参考部署代码。

这里我们展示了使用历史长度为 20 的师生框架训练的策略的运行演示。

```{figure} ../../_static/images/ts_demo.gif
```

## 参考文献

1. [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251)
2. [RMA: Rapid Motor Adaptation for Legged Robots](https://ashish-kmr.github.io/rma-legged-robots/)
3. [Rapid Locomotion via Reinforcement Learning](https://agility.csail.mit.edu/)
4. [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542)
5. [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://clearlab-sustech.github.io/concurrentTS/)
