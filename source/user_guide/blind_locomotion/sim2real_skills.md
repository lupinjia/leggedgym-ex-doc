# 🔄 Sim2Real 迁移技能

本章节介绍用于实现 Sim2Real（仿真到现实迁移）的各种算法技术。

## 域随机化 (Domain Randomization)

域随机化是一种常用的 Sim2Real 技术，通过在训练过程中随机化仿真环境的物理参数（如摩擦系数、质量、质心位置等），使策略对参数扰动更加鲁棒，从而能够在真实环境中表现良好。

在我们的框架中，域随机化主要包括：
- 摩擦系数随机化
- 基座质量随机化
- 质心偏移随机化
- 关节参数随机化（刚度、阻尼、armature）
- 动作延迟随机化
- 观测噪声

## 系统辨识 (System Identification)

系统辨识是通过在仿真中匹配真实机器人的动态响应来识别系统参数的过程。这有助于：
- 获得更准确的仿真模型
- 减小仿真与现实之间的差距
- 提高策略的 Sim2Real 迁移性能

主要方法包括：
1. **参数辨识**：调整物理参数使仿真和真实的运动轨迹匹配
2. **残差学习 (Residual Learning)**：学习仿真与现实之间的残差动态
3. **Domain Randomization Distribution Learning**：学习最优的域随机化分布

## 仿真到现实的迁移策略

### 1. 硬件感知训练 (Hardware-Aware Training)

在训练阶段就考虑到真实硬件的限制：
- 动作限幅
- 动作平滑
- 延迟补偿
- 传感器噪声建模

### 2. 安全层 (Safety Layer)

在部署时添加安全保护：
- 关节限位检查
- 速度限制
- 力矩限制
- 紧急停止机制

### 3. 在线适应 (Online Adaptation)

策略在真实环境中实时适应：
- RMA (Rapid Motor Adaptation)
- 在线系统辨识
- 元学习 (Meta-Learning) 方法

## 参考文献

1. [Domain Randomization for Sim-to-Real Transfer](https://arxiv.org/abs/1703.06907)
2. [Sim-to-Real via Sim-to-Sim](https://arxiv.org/abs/1810.05687)
3. [Learning Agile Robotic Locomotion Skills by Imitating Animals](https://arxiv.org/abs/2004.00784)
