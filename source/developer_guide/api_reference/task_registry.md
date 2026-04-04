# TaskRegistry API 参考

本文档描述了 LeggedGym-Ex 用于注册、实例化和编排任务环境及其对应强化学习 (RL) runner 的 TaskRegistry 工厂模式。TaskRegistry 提供了一个单一的集中式机制，将字符串任务名称映射到其环境类和配置对象，并暴露工厂方法来创建可运行的环境和 PPO runner。其目标是将任务连接与任务使用解耦，实现模块化组合、更清晰的测试以及跨多个任务的更轻松的实验。

注意：本文档中的所有示例都遵循传统的命名模式，其中注册的任务名称形如 `<robot>_<variant>`（参见命名约定部分）。

## 1. 注册 API

注册任务的核心 API 是：

```
task_registry.register(name, env_class, cfg, cfg_ppo)
```

- name: 标识任务的唯一字符串。工厂方法使用它来定位正确的环境和配置。
- env_class: 实现 LeggedRobot 风格任务的环境类。这必须是类引用（而非实例）。
- cfg: LeggedRobotCfg（或其子类）的实例，描述此任务的环境配置。
- cfg_ppo: LeggedRobotCfgPPO（或合适的 PPO 配置）的实例，描述此任务的训练/运行时超参数。

注册行为和保证：
- 唯一性：每个任务名称必须恰好注册一次。尝试注册已存在的名称会引发明确的错误。
- 类型安全：注册表会验证所提供参数的类型。类型不匹配会引发信息性错误，而非静默失败。
- 就绪性：注册表将提供的映射存储在内部注册表（内存字典）中，供工厂方法使用。

注册期间的错误处理：
- 如果 name 已存在：引发 ValueError(f"Task '{name}' is already registered")
- 如果 env_class 不是类或不可调用：引发 TypeError("env_class must be a class reference")
- 如果 cfg 不是配置对象（或缺少预期属性）：引发 TypeError("cfg must be a LeggedRobot config instance")
- 如果 cfg_ppo 不是配置对象：引发 TypeError("cfg_ppo must be a LeggedRobot PPO config instance")

注册调用被设计为轻量级且无副作用，仅用于填充注册表。所有复杂的初始化发生在通过工厂方法请求已注册任务时。

## 2. 工厂方法

两个主要工厂方法暴露了任务创建工作流程：

- make_env(name, args)
- make_alg_runner(env, cfg, log_dir)

概述：
- make_env(name, args): 为已注册任务生成可用的环境实例。该方法通过名称查找注册表项，克隆存储的 cfg，应用来自 args 的任何运行时覆盖，并使用结果配置构建环境类。
- make_alg_runner(env, cfg, log_dir): 生成适合给定任务训练或评估的 RL runner。runner 通常通过 Runner Registry（参见参考）创建，并与该任务的环境和 PPO 配置连接。

使用模式（示例）：
```
env = task_registry.make_env("go2", {"seed": 42, "terrain_seed": 7})
runner = task_registry.make_alg_runner(env, GO2CfgPPO(), "./logs/go2")
```

实现说明和预期行为：
- make_env 中的 name 参数必须对应先前注册的任务。
- make_env 中的 args 参数是覆盖存储的 cfg 的属性名到值的映射。覆盖是浅拷贝；高级嵌套覆盖需要在 cfg 对象内部显式解包。
- make_alg_runner 函数委托给中央 Runner Registry 来为给定任务构造适当的 PPO runner。确切的 runner 类和连接方式取决于注册的任务，但公共 API 保持稳定。

工厂中的错误处理：
- 如果 name 未知：引发 KeyError(f"Unknown task '{name}'. Please register it before creating environments.")
- 如果 env_class 无法使用给定的 cfg 实例化：引发 RuntimeError("Failed to instantiate environment with the provided configuration")
- 如果 log_dir 不可写或无法访问：引发 OSError 描述 IO 问题。
- 如果 make_alg_runner 使用不对应于注册任务的环境调用：引发 TypeError("Environment does not belong to a registered task")

## 3. 任务命名约定

任务遵循将任务命名为 `<robot>_<variant>` 的语义约定。robot 段标识平台或系列（例如，GO2、G1、K1、TRON1PF 等），而 variant 段指示算法方法、训练机制或特殊设置（例如，ts 表示教师-学生，deepmimic 表示动作模仿，amp 表示 Adapting Motion Policies 等）。

- 示例：
  - go2_ts: 带有教师-学生风格约束的 GO2。
  - g1_deepmimic: 带有 DeepMimic 参考动作处理的 G1。
  - k1_amp: 带有 AMP 风格训练的 K1。

原理：
- 命名约定以紧凑形式编码机器人任务和训练方法。它使通过简单字符串键发现和比较任务变得更加容易，并且与整个注册代码和文档中使用的结构保持一致。
- 它还有助于自动化工具，脚本可以从名称本身派生环境类、默认配置和 PPO 变体。

## 4. 注册代码示例

TaskRegistry 设计为由显式注册填充，通常位于任务初始化器附近（例如，legged_gym/envs/__init__.py）。一个代表性的代码片段如下所示：

```python
# 示例：注册带有其配置变体的 GO2 任务
from legged_gym.envs.go2 import GO2, GO2Cfg, GO2CfgPPO
from legged_gym.utils import task_registry

task_registry.register("go2", GO2, GO2Cfg(), GO2CfgPPO())
```

注册后，使用遵循前面演示的工厂模式：

```python
env = task_registry.make_env("go2", {"seed": 123, "terrain_seed": 7})
runner = task_registry.make_alg_runner(env, GO2CfgPPO(), "./logs/go2")
```

注意：
- cfg 和 cfg_ppo 参数在注册时实例化一次，可以被 make_env/make_alg_runner 重用或克隆。如果您需要任务特定的动态覆盖，请优先通过 args 参数传递给 make_env，或实现一个小包装器从存储的基础配置和覆盖组合新 cfg 对象。
- runner 的确切行为可能取决于任务；注册表委托给集中式 runner 工厂，以保持关注点分离，并在可用时支持除 PPO 之外的多种 RL 算法。

## 5. 实践中的错误处理

健壮的错误处理对于开发者友好的 API 至关重要。TaskRegistry 应该快速失败并提供可操作的错误消息：

- 重复注册：ValueError("Task 'name' is already registered")
- 未知任务访问：KeyError("Unknown task 'name'. Please register it before usage")
- 无效参数类型：带有描述性消息的 TypeError，如 "env_class must be a class reference" 或 "cfg must be a LeggedRobotCfg instance"。
- 环境构造失败：根据根本原因引发的 RuntimeError 或 OSError，异常消息中包含清晰的上下文信息。

为了帮助调试，注册表可以可选地暴露一个小型诊断方法（例如，list_registered_tasks()），返回所有注册条目的人类可读视图，包括其 env_class 名称和其 cfg 对象的类型。

## 6. 设计原理：LeggedGym-Ex 中的工厂模式

TaskRegistry 体现了工厂模式：它集中了异构任务环境及其学习器的创建逻辑。通过将注册与实例化分离，代码库获得了：

- 将任务连接与任务执行解耦，启用可插拔任务而无需更改消费代码。
- 清晰的关注点分离：TaskRegistry 管理任务元数据，而 Runner Registry 处理 RL 算法的选择和配置。
- 更轻松的测试：测试可以注册模拟任务，探测注册表行为，并在不影响真实环境类的情况下练习 make_env/make_alg_runner。
- 可扩展性：添加新任务只需注册映射，无需修改核心训练循环或 runner 编排逻辑。

此设计与 LeggedGym-Ex 的更广泛目标保持一致，其中许多机器人共享公共接口，但具有不同的配置和训练机制。TaskRegistry 提供了一个轻量级、可预测的表面，用于在一系列实验中实例化和运行任务。

## 7. 参考和延伸阅读

- 代码参考：任务注册表实现在 legged_gym.utils.task_registry 中（参见 LeggedGym-Ex/legged_gym/utils/task_registry.py 仓库）。
- Runner 集成：参见 legged_gym.utils.runner_registry.py 了解 make_alg_runner 使用的中央 runner 构造逻辑。
- 环境注册：legged_gym.envs.__init__ 包含将任务名称连接到环境类和基础配置的实际注册。

## 8. 总结

TaskRegistry API 提供了一个稳定、有文档记录的途径来注册新任务，并以解耦、可测试的方式构建环境和 PPO runner。通过强制执行命名约定、防范错误注册，并将算法特定的连接委托给专用的 Runner Registry，系统在新机器人和训练策略添加时保持可扩展性和可维护性。

  
