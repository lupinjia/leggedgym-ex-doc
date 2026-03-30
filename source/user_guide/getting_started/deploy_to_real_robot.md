# ⬇️ 部署到真实机器人

在上一小节中，我们简要学习了如何开始训练会话并在仿真器中运行生成的策略。下一个问题是，如何将策略部署到真实机器人上，与真实世界的动力学进行交互？

## 预备知识

根据强化学习（Reinforcement Learning）的知识$^{1,2}$，我们知道我们希望找到一个智能体（agent）$\pi$，它可以最大化折扣累积奖励。智能体由一个神经网络表示，该网络基于观测值（observation）输出动作（action）。在最简单的设置中，观测值通常包括机身角速度（base angular velocities）、机身姿态（base orientation）、速度指令（velocity commands）、关节角度（joint angles）、关节角速度（joint angular velocities）和上一时间步的动作（actions of last timestep）。虽然您可能在 `legged_robot.py` 中看到过机身线速度（base linear velocities）（如下代码块），但这些信息无法通过机器人上的传感器直接获得。

```python
self.obs_buf = torch.cat((self.simulator.base_lin_vel * self.obs_scales.lin_vel,                    # 3
                            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
                            self.simulator.projected_gravity,                                         # 3
                            self.commands[:, :3] * self.commands_scale,                   # 3
                            (self.simulator.dof_pos - self.simulator.default_dof_pos) 
                                      * self.obs_scales.dof_pos, # num_dofs
                            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
                            self.actions                                                    # num_actions
                            ), dim=-1)
```

通常，安装在机器人上的传感器包括惯性测量单元（Inertial Measurement Unit, IMU）和关节编码器（joint encoders）。通过 IMU，我们可以获得机身角速度和机身姿态（在上面的代码中转换为 projected_gravity）。关节角度和关节角速度可以通过关节编码器获取。

为了获得机身线速度，我们需要基于其他传感器使用诸如卡尔曼滤波器（Kalman Filter）或估计器网络（estimator network，稍后将解释）等方法进行估计。这里我们专注于利用其他可用信息使机器人成功行走。

## 修改观测值

为了将策略部署到真实机器人，我们需要确保观测值在真实机器人上都是可获得的。因此我们需要从观测中删除 `base_lin_vel` 并训练一个新策略。

要查看奖励曲线和损失曲线，您可以使用 tensorboard 或 wandb。通过在 `python train.py` 后添加 `--sync_wandb`，训练数据将同步到云端的 wandb。比较包含 `base_lin_vel` 和不包含它的会话，您可能会发现不包含 `base_lin_vel` 的奖励曲线上升较慢，甚至在训练结束时下降。这主要是由于信息缺失造成的。

```{figure} ../../_static/images/compare_with_and_wo_lin_vel.png
```

## Mujoco 中的仿真到仿真（Sim2Sim）

在部署到真实机器人之前，最好先将策略部署到另一个仿真器。通过这种方式，您可以测试策略的鲁棒性，并避免在真实机器人上出现潜在的崩溃。要使用此功能，您需要首先按照[安装](installation.md)中的说明安装 go2_deploy。

执行 `play.py` 后，您将在训练会话目录中找到一个名为 `exported` 的文件夹。在此目录下，您将找到一个可以部署的 JIT 脚本文件（.pt）。

本质上，部署代码所做的是对齐控制策略的输入和输出。对于输入，您需要使反馈信息与训练代码中的相同。对于输出，您需要使真实电机的响应与仿真中的响应一致。

我们在 [go2_deploy](https://github.com/lupinjia/go2_deploy/tree/main) 中提供了一个 `SimpleRLController` 供您部署这个最简单的控制策略，编译后执行以下命令：

```bash
# 在 go2_deploy/build 目录下
./go2_deploy simple_rl

# 在 go2_deploy/unitree_mujoco/simulate/build 目录下
./unitree_mujoco
```

以下是运动的演示视频：

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/lupinjia/genesis_lr_doc/raw/refs/heads/main/source/_static/videos/simple_rl_demo.mp4" type="video/mp4">
</video>

您可以看到这个四足机器人可以按照我们期望的速度指令行走，这些指令通过操纵杆指定。但这个策略只是一个最简单的版本，它在受到外部干扰和复杂地形时表现不佳。

## 部署到真实机器人

要将策略部署到真实的 Go2 机器人，您需要修改程序的网络接口。通过 `ifconfig` 检查您的以太网接口：

```{figure} ../../_static/images/ifconfig_output.jpeg
```

我们可以看到这台 PC 的以太网接口是 `enp4s0`。然后我们可以通过以太网将此 PC 与 `go2` 连接并执行：

```bash
./go2_deploy simple_rl enp4s0
```

您将看到机器人以与仿真没有太大区别的方式运动。

## 参考文献

1. [Hands on RL](https://hrl.boyuai.com/)
2. [Easy RL](https://datawhalechina.github.io/easy-rl/#/)
