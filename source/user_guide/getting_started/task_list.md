# 📋 任务列表

这里列出了 genesis_lr 支持的任务。

- 开始训练：`python legged_gym/scripts/train.py --task=task_name`。
- 可视化训练好的模型：`python legged_gym/scripts/play.py --task=task_name`
- 更多命令行参数请参见 `legged_gym/utils/helpers.py/get_args()`

| 机器人 | 任务名称 | 描述 | 论文 |
| ----- | ---- | ----------- | -----------|
| Unitree Go2 | go2 | 在平面上训练 Go2 机器人行走策略的简单示例 | |
|       | go2_wtw | 在 Go2 上实现 Walk These Ways，支持控制机身高度（base height）、机身俯仰角（base pitch angle）、足端间隙（foot clearance）、步态周期（gait period）和步态类型（gait type） | [Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://arxiv.org/abs/2212.03238) |
|       | go2_ts  | 在 Go2 上实现教师-学生（Teacher-Student）框架，用于在复杂地形上行走 | [Rapid Locomotion via Reinforcement Learning](https://agility.csail.mit.edu/) |
|       | go2_ee  | 在 Go2 上实现显式估计器（Explicit Estimator），用于在复杂地形上行走 | [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481) |
|       | go2_dreamwaq | 在 Go2 上实现 DreamWaQ，用于在复杂地形上行走 | [DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602) |
|       | go2_cat | 在 Go2 上实现约束作为终止条件（Constraints as Terminations, CaT），用于在复杂地形上行走 | [CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning](https://constraints-as-terminations.github.io/) |
|       | go2_nav | 在 Go2 上实现端到端局部导航（End-to-end local navigation） | [Advanced Skills by Learning Locomotion and Local Navigation End-to-End](https://arxiv.org/abs/2209.12827) |
|       | go2_cts | 实现并发教师-学生（Concurrent Teacher Student, CTS）框架 | [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://clearlab-sustech.github.io/concurrentTS/) |
|       | go2_ts_depth | Go2 深度估计变体（ts_depth） | |
| Unitree G1 | g1 | 在平面上训练 G1 机器人行走策略的简单示例（仅 12 自由度，上身固定） | |
|            | g1_deepmimic | 在 Unitree G1 上实现 DeepMimic（29 自由度） | [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/index.html) |
|            | g1_motion_vis | G1 动作可视化 | |
| Limx TRON1PF | tron1pf | 在平面上训练 TRON1PF 机器人行走策略的简单示例 | |
|       | tron1pf_ee | 在 TRON1PF 上实现显式估计器（Explicit Estimator），用于在复杂地形上行走 |  |
| Limx TRON1SF | tron1sf | 在平面上训练 TRON1SF 机器人行走策略的简单示例 | |
|       | tron1sf_ee | 在 TRON1SF 上实现显式估计器（Explicit Estimator），用于在复杂地形上行走 | |
| Booster K1 | k1 | 在平面上训练 K1 机器人行走策略的简单示例 | |
|  | k1_amp | 在 Booster K1 上实现 AMP | [AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control](https://arxiv.org/abs/2104.02180) |
|  | k1_cts_amp | 在 Booster K1 上实现 CTS AMP | |
|  | k1_motion_vis | K1 动作可视化 | |
|        | k1_deepmimic | 在 Booster K1 上实现 DeepMimic（29 自由度） | [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/index.html) |
