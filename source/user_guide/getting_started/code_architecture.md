# 🧬 代码架构

## genesis_lr 的整体架构

```{figure} ../../_static/images/code_structure.png
```

整个项目可以分为 3 个部分：仿真器（simulator）、环境（environments）和算法（algorithms）。

## 仿真器（legged_gym/simulator）

仿真器层为不同的仿真器提供统一的 API，包括 IsaacGym 和 Genesis。用户可以选择使用任意一个仿真器进行训练。

除了仿真器内置的传感器外，我们还实现了外部增强传感器，主要用于 IsaacGym。目前，我们提供基于 [warp](https://github.com/NVIDIA/warp) 的深度相机传感器，以加速 IsaacGym 中的渲染。与 IsaacGym 内置的深度相机相比，基于 warp 的深度相机在无头（headless）训练期间可以提供 2-3 倍的渲染速度提升。

## 环境（legged_gym/envs）

环境是智能体（agent）收集数据并从中获得奖励信号的地方。用户可以定义任务和动力学模型。环境采用继承风格。用户可以定义继承基类（legged_robot.py）的新环境类。环境中的配置文件（config files）负责存储环境的参数和设置。

## 算法（rsl_rl）

算法定义了训练中使用的强化学习算法。目前我们实现了 PPO（Proximal Policy Optimization，近端策略优化）、[SPO（Simple Policy Optimization，简单策略优化）](https://github.com/MyRepositories-hub/Simple-Policy-Optimization)以及基于它们的几种训练架构（显式估计器 Explicit Estimator、教师-学生 Teacher Student、并发 TS Cocurrent TS、DreamWaQ）。

## 仿真到仿真（Sim2Sim）

在将策略部署到真实机器人之前，通常需要先在另一个仿真器中进行测试。这可以验证策略的鲁棒性，并避免在真实机器人上出现意外行为。我们提供了相应的工具和接口来支持这一过程。
