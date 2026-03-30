# ⛕ Walk These Ways

如 Margolis $\textit{et al.}^{1}$ 所述，行为的多样性 (Multiplicity of Behavior, MoB) 可以帮助机器人在不同场景下实现泛化。$\textit{Walk These Ways}$ 的基本思想是在平地上学习不同的行为，并通过高层决策（本文中为人类操作员）来调节行为。

为了将不同的行为整合到一个神经网络 (Neural Network, NN) 中，我们本质上想要实现多任务强化学习 (Multi-Task Reinforcement Learning)。本实现的关键组件包含三个部分：
1. 任务相关的观测 (Task-related Observation)
2. 任务奖励 (Task Rewards)
3. 基于奖励的课程学习 (Reward-based Curriculum)

## 任务相关的观测

为了使神经网络策略能够区分不同的行为，我们需要提供与任务相关的观测作为输入。

在我们的实现中，我们感兴趣的行为包括 `gait_period`（步态周期）、`base_height`（基座高度）、`foot_clearance`（足部间隙）、`base_pitch`（基座俯仰角）和 `gait_type`（步态类型）。对于前四种行为，我们直接将参数放入观测中。对于 `gait_type`，我们使用 `theta`（四足的相位偏移）结合 `clock_input`（时钟输入）来表示不同的步态类型：
```python
    # In go2_wtw.py
    def compute_observations(self):
        """ Computes observations
        """
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,    # cmd     3
            self.simulator.projected_gravity,                        # g       3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,   # omega   3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,                       # p_t     12
            self.simulator.dof_vel * self.obs_scales.dof_vel,        # dp_t    12
            self.actions,                                  # a_{t-1} 12
            self.clock_input,                              # clock   4
            self.gait_period,                              # gait period 1
            self.base_height_target,                       # base height target 1
            self.foot_clearance_target,                    # foot clearance target 1
            self.pitch_target,                             # pitch target 1
            self.theta,                                    # theta, gait offset, 4
        ), dim=-1)

        ...
```

这里，我们使用的步态规范方法基于 [Siekmann $\textit{et al.}$](https://arxiv.org/abs/2011.01387) 论文中的原理。基于他们的原理，我们提供了一个额外的阶跃函数实现来构建步态指示器：
```python
    # In go2_wtw.py
    def _uniped_periodic_gait(self, foot_type):
        # q_frc and q_spd
        if foot_type == "FL":
            q_frc = torch.norm(
                self.simulator.link_contact_forces[:, 
                                    self.simulator.feet_indices[0], :], dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 0, :], dim=-1).view(-1, 1) # sequence of feet_pos is FL, FR, RL, RR
            # size: num_envs; need to reshape to (num_envs, 1), or there will be error due to broadcasting
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 0].unsqueeze(1)) % 1.0
        elif foot_type == "FR":
            q_frc = torch.norm(
                self.simulator.link_contact_forces[:, 
                                    self.simulator.feet_indices[1], :], dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 1, :], dim=-1).view(-1, 1)
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 1].unsqueeze(1)) % 1.0
        elif foot_type == "RL":
            q_frc = torch.norm(
                self.simulator.link_contact_forces[:, 
                                    self.simulator.feet_indices[2], :], dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 2, :], dim=-1).view(-1, 1)
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 2].unsqueeze(1)) % 1.0
        elif foot_type == "RR":
            q_frc = torch.norm(
                self.simulator.link_contact_forces[:, 
                                    self.simulator.feet_indices[3], :], dim=-1).view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 3, :], dim=-1).view(-1, 1)
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 3].unsqueeze(1)) % 1.0
        
        phi *= 2 * torch.pi  # convert phi to radians
        
        if self.gait_function_type == "smooth":
            # coefficient
            c_swing_spd = 0  # speed is not penalized during swing phase
            c_swing_frc = -1  # force is penalized during swing phase
            c_stance_spd = -1  # speed is penalized during stance phase
            c_stance_frc = 0  # force is not penalized during stance phase
            
            # clip the value of phi to [0, 1.0]. The vonmises function in scipy may return cdf outside [0, 1.0]
            F_A_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.a_swing, 
                kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
            F_B_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_swing.cpu(), 
                kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
            F_A_stance = F_B_swing
            F_B_stance = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_stance,
                kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)

            # calc the expected C_spd and C_frc according to the formula in the paper
            exp_swing_ind = F_A_swing * (1 - F_B_swing)
            exp_stance_ind = F_A_stance * (1 - F_B_stance)
            exp_C_spd_ori = c_swing_spd * exp_swing_ind + c_stance_spd * exp_stance_ind
            exp_C_frc_ori = c_swing_frc * exp_swing_ind + c_stance_frc * exp_stance_ind

            # just the code above can't result in the same reward curve as the paper
            # a little trick is implemented to make the reward curve same as the paper
            # first let all envs get the same exp_C_frc and exp_C_spd
            exp_C_frc = -0.5 + (-0.5 - exp_C_spd_ori)
            exp_C_spd = exp_C_spd_ori
            # select the envs that are in swing phase
            is_in_swing = (phi >= self.a_swing) & (phi < self.b_swing)
            indices_in_swing = is_in_swing.nonzero(as_tuple=False).flatten()
            # update the exp_C_frc and exp_C_spd of the envs in swing phase
            exp_C_frc[indices_in_swing] = exp_C_frc_ori[indices_in_swing]
            exp_C_spd[indices_in_swing] = -0.5 + \
                (-0.5 - exp_C_frc_ori[indices_in_swing])

            # Judge if it's the standing gait
            is_standing = (self.b_swing[:] == self.a_swing).nonzero(
                as_tuple=False).flatten()
            exp_C_frc[is_standing] = 0
            exp_C_spd[is_standing] = -1
        elif self.gait_function_type == "step":
            ''' ***** Step Gait Indicator ***** '''
            exp_C_frc = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
            exp_C_spd = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
            
            swing_indices = (phi >= self.a_swing) & (phi < self.b_swing)
            swing_indices = swing_indices.nonzero(as_tuple=False).flatten()
            stance_indices = (phi >= self.b_swing) & (phi < self.b_stance)
            stance_indices = stance_indices.nonzero(as_tuple=False).flatten()
            exp_C_frc[swing_indices, :] = -1
            exp_C_spd[swing_indices, :] = 0
            exp_C_frc[stance_indices, :] = 0
            exp_C_spd[stance_indices, :] = -1

        return exp_C_spd * q_spd + exp_C_frc * q_frc, \
            exp_C_spd.type(dtype=torch.float), exp_C_frc.type(dtype=torch.float)
    
    def _reward_quad_periodic_gait(self):
        quad_reward_fl, self.exp_C_spd_fl, self.exp_C_frc_fl = self._uniped_periodic_gait(
            "FL")
        quad_reward_fr, self.exp_C_spd_fr, self.exp_C_frc_fr = self._uniped_periodic_gait(
            "FR")
        quad_reward_rl, self.exp_C_spd_rl, self.exp_C_frc_rl = self._uniped_periodic_gait(
            "RL")
        quad_reward_rr, self.exp_C_spd_rr, self.exp_C_frc_rr = self._uniped_periodic_gait(
            "RR")
        # reward for the whole body
        quad_reward = quad_reward_fl.flatten() + quad_reward_fr.flatten() + \
            quad_reward_rl.flatten() + quad_reward_rr.flatten()
        return torch.exp(quad_reward)
```

用户可以选择使用 `gait_function_type`（在 `go2_wtw_config.py` 中指定）。以下是使用 `smooth_function(kappa=20)` 和 `step_function` 的 `exp_C_frc` 曲线：
```{figure} ../../_static/images/exp_C_frc_smooth_gait.png
```
```{figure} ../../_static/images/exp_C_frc_step_gait.png
```

基于我们的实践经验，`step function` 和 `smooth function` 在步态跟踪性能上没有太大差异。

## 任务奖励

为了引导策略朝着实现期望行为的方向优化，我们需要构建包含行为参数的奖励函数。

在我们的实现中，行为任务奖励包括 `_reward_quad_periodic_gait`、`_reward_tracking_base_height`、`_reward_tracking_orientation` 和 `_reward_tracking_foot_clearance`。读者可以参考 `go2_wtw.py` 查看逐行代码。

## 基于奖励的课程学习

如 Rudin $\textit{et al.}^{3}$ 所述，适当的课程设计可以促进学习过程并帮助机器人学习更困难的行为。我们实现了一个类似于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 的基于奖励的课程学习，只有当策略在当前范围内充分掌握了行为后，才会扩大行为参数的范围：
```python
def _update_behavior_param_curriculum(self, env_ids):
        if len(env_ids) == 0:
            return
        # Widen the behavior param range according to reward values
        if torch.mean(self.episode_sums["quad_periodic_gait"][env_ids]) / \
            self.max_episode_length > 0.8 * self.reward_scales["quad_periodic_gait"]: # 0.8 for step gait, 0.5 for smooth gait
            # gait period
            self.gait_period_range[0] = max(self.gait_period_range[0] - 0.05, self.gait_period_min)
            self.gait_period_range[1] = min(self.gait_period_range[1] + 0.05, self.gait_period_max)
            # gait number
            self.num_gaits = min(self.num_gaits + 1, self.num_gait_max)

        if torch.mean(self.episode_sums["tracking_base_height"][env_ids]) / \
            self.max_episode_length > 0.9 * self.reward_scales["tracking_base_height"]:
            self.base_height_target_range[0] = max(self.base_height_target_range[0] - 0.02, self.base_height_target_min)
            self.base_height_target_range[1] = min(self.base_height_target_range[1] + 0.02, self.base_height_target_max)

        if torch.mean(self.episode_sums["tracking_foot_clearance"][env_ids]) / \
            self.max_episode_length > 0.8 * self.reward_scales["tracking_foot_clearance"]:
            self.foot_clearance_target_range[0] = max(self.foot_clearance_target_range[0] - 0.01, 
                                                      self.foot_clearance_target_min)
            self.foot_clearance_target_range[1] = min(self.foot_clearance_target_range[1] + 0.01, 
                                                      self.foot_clearance_target_max)
        
        if torch.mean(self.episode_sums["tracking_orientation"][env_ids]) / \
            self.max_episode_length > 0.9 * self.reward_scales["tracking_orientation"]:
            self.pitch_target_range[0] = max(self.pitch_target_range[0] - 0.05, self.pitch_target_min)
            self.pitch_target_range[1] = min(self.pitch_target_range[1] + 0.05, self.pitch_target_max)
```

## 训练和运行

要训练一个 Walk These Ways 策略，输入以下命令：
```bash
python train.py --task=go2_wtw --headless
```

要运行它，输入以下命令：
```bash
python play.py --task=go2_wtw --load_run=session_name
```

## 演示

我们在 [go2_deploy](https://github.com/lupinjia/go2_deploy/tree/main) 中提供了 $\textit{Walk These Ways}$ 的实现，你可以使用以下命令运行：
```bash
./go2_deploy wtw
```

演示视频如下：
<video preload="auto" controls="True" width="100%">
<source src="https://github.com/lupinjia/genesis_lr_doc/raw/refs/heads/main/source/_static/videos/wtw_demo.mp4" type="video/mp4">
</video>

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/lupinjia/genesis_lr_doc/raw/refs/heads/main/source/_static/videos/wtw_demo_real.mp4" type="video/mp4">
</video>


## 参考文献

1. [Walk These Ways](https://gmargo11.github.io/walk-these-ways/)
2. [Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition](https://arxiv.org/abs/2011.01387)
3. [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)
