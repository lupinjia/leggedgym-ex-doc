# DreamWaQ

DreamWaQ (Deep Reinforcement Learning for Walking on Any Terrain with Quadrupeds) 是一种用于四足机器人在各种地形上行走的强化学习方法。

## 核心思想

DreamWaQ 的核心思想是通过学习一个鲁棒的策略，使四足机器人能够在各种复杂地形上稳定行走，包括：
- 不平整地形
- 障碍物
- 斜坡和楼梯
- 松软地面

## 方法概述

该方法通常包括以下关键组件：

### 1. 地形感知训练
- 使用程序化生成的多样化地形
- 地形课程学习 (Terrain Curriculum Learning)
- 自适应地形难度

### 2. 鲁棒性设计
- 域随机化 (Domain Randomization)
- 传感器噪声建模
- 执行器延迟建模

### 3. 高效探索
- 使用高级探索策略
- 课程学习引导探索

## 实现

在我们的框架中，DreamWaQ 的实现可以参考相关任务文件。

## 训练和运行

要训练一个 DreamWaQ 策略，输入以下命令：
```bash
python train.py --task=go2_dreamwaq --headless
```

要运行它，输入以下命令：
```bash
python play.py --task=go2_dreamwaq --load_run=session_name
```

## 参考文献

1. [DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602)
2. [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)
