# TaskRegistry API Reference

This document describes the TaskRegistry factory pattern used by LeggedGym-Ex to register, instantiate, and orchestrate task environments and their corresponding reinforcement learning (RL) runners. The TaskRegistry provides a single, centralized mechanism to map a string task name to its environment class and configuration objects, and exposes factories to create ready-to-run environments and PPO runners. The goal is to decouple task wiring from task usage, enabling modular composition, clearer tests, and easier experimentation across multiple tasks.

Note: All examples in this document assume the conventional naming pattern where a registered task name looks like `<robot>_<variant>` (see Naming Conventions section).

## 1. Registration API

The core API for registering a task is:

```
task_registry.register(name, env_class, cfg, cfg_ppo)
```

- name: A unique string that identifies the task. It is used by the factory methods to locate the correct environment and configuration.
- env_class: The environment class implementing the LeggedRobot-style task. This must be a class reference (not an instance).
- cfg: An instance of a LeggedRobotCfg (or a subclass) describing the environment configuration for this task.
- cfg_ppo: An instance of a LeggedRobotCfgPPO (or a suitable PPO configuration) describing the training/run-time hyperparameters for this task.

Registration behavior and guarantees:
- Uniqueness: Each task name must be registered exactly once. Attempting to register a name that already exists raises a clear error.
- Type safety: The registry validates the types of the provided arguments. Mismatches raise informative errors rather than failing silently.
- Readiness: The registry stores the provided mappings in an internal registry (an in-memory dictionary) to be used by factory methods.

Error handling during registration:
- If name already exists: raise ValueError(f"Task '{name}' is already registered")
- If env_class is not a class or not callable: raise TypeError("env_class must be a class reference")
- If cfg is not a configuration object (or lacks expected attributes): raise TypeError("cfg must be a LeggedRobot config instance")
- If cfg_ppo is not a configuration object: raise TypeError("cfg_ppo must be a LeggedRobot PPO config instance")

The registration call is intentionally lightweight and side-effect-free beyond populating the registry. All complex initialization occurs later when the registered task is requested via the factory methods.

## 2. Factory Methods

Two primary factory methods expose the task creation workflow:

- make_env(name, args)
- make_alg_runner(env, cfg, log_dir)

Overview:
- make_env(name, args): Produces a ready-to-use environment instance for the registered task. The method looks up the registry entry by name, clones the stored cfg, applies any runtime overrides from args, and constructs the environment class with the resulting configuration.
- make_alg_runner(env, cfg, log_dir): Produces an RL runner suitable for training or evaluation of the given task. The runner is typically created via the Runner Registry (see Reference), and wired with the environment and PPO configuration for that task.

Usage pattern (illustrative):
```
env = task_registry.make_env("go2", {"seed": 42, "terrain_seed": 7})
runner = task_registry.make_alg_runner(env, GO2CfgPPO(), "./logs/go2")
```

Implementation notes and expected behavior:
- The name parameter must correspond to a previously registered task.
- The args parameter in make_env is a mapping of attribute names to values that override the stored cfg. Overrides are shallow copies; advanced nested overrides would require explicit unwrapping inside the cfg object.
- The make_alg_runner function delegates to a central Runner Registry to construct an appropriate PPO runner for the given task. The exact runner class and wiring depend on the registered task, but the public API remains stable.

Error handling in factories:
- If name is unknown: raise KeyError(f"Unknown task '{name}'. Please register it before creating environments.")
- If env_class cannot be instantiated with the given cfg: raise RuntimeError("Failed to instantiate environment with the provided configuration").
- If log_dir is not writable or inaccessible: raise OSError describing the IO issue.
- If make_alg_runner is called with an environment that does not correspond to a registered task: raise TypeError("Environment does not belong to a registered task").

## 3. Task Naming Conventions

Tasks follow the semantic convention of naming as `<robot>_<variant>`. The robot segment identifies the platform or family (for example, GO2, G1, K1, TRON1PF, etc.), while the variant segment indicates the algorithmic approach, training regime, or special setup (for example, ts for teacher-student, deepmimic for motion imitation, amp for Adapting Motion Policies, etc.).

- Examples:
  - go2_ts: GO2 with Teacher-Student style constraints.
  - g1_deepmimic: G1 with DeepMimic reference motion processing.
  - k1_amp: K1 with AMP style training.

Rationale:
- The naming convention encodes both the robot task and the training methodology in a compact form. It makes it easier to discover and compare tasks through a simple string key, and it aligns with the structure used throughout the registration code and docs.
- It also helps with automated tooling, where scripts can derive the environment class, default configs, and PPO variants from the name alone.

## 4. Example Registration Code

The TaskRegistry is designed to be filled by explicit registrations, typically somewhere near the task initializers (for example, legged_gym/envs/__init__.py). A representative snippet looks like this:

```python
# Example: registering the GO2 task with its configuration variants
from legged_gym.envs.go2 import GO2, GO2Cfg, GO2CfgPPO
from legged_gym.utils import task_registry

task_registry.register("go2", GO2, GO2Cfg(), GO2CfgPPO())
```

After registration, usage follows the factory pattern demonstrated earlier:

```python
env = task_registry.make_env("go2", {"seed": 123, "terrain_seed": 7})
runner = task_registry.make_alg_runner(env, GO2CfgPPO(), "./logs/go2")
```

Notes:
- The cfg and cfg_ppo arguments are instantiated once at registration time and may be reused or cloned as needed by make_env/make_alg_runner. If you require task-specific dynamic overrides, prefer passing them through the args parameter to make_env or implement a small wrapper to compose a new cfg object from the stored base and overrides.
- The precise behavior of the runner may depend on the task; the registry delegates to the centralized runner factory to keep concerns separated and to support a variety of RL algorithms beyond PPO when available.

## 5. Error Handling in Practice

Robust error handling is critical for a developer-friendly API. The TaskRegistry should fail fast with actionable messages:

- Duplicate registration: ValueError("Task 'name' is already registered")
- Unknown task access: KeyError("Unknown task 'name'. Please register it before usage")
- Invalid argument types: TypeError with a descriptive message like "env_class must be a class reference" or "cfg must be a LeggedRobotCfg instance".
- Environment construction failures: RuntimeError or OSError depending on the root cause, with clear contextual information included in the exception message.

To aid debugging, the registry may optionally expose a small diagnostic method (e.g., list_registered_tasks()) that returns a human-readable view of all registered entries, including their env_class names and the types of their cfg objects.

## 6. Design Rationale: The Factory Pattern in LeggedGym-Ex

TaskRegistry embodies the factory pattern: it centralizes creation logic for heterogeneous task environments and their learners. By separating registration from instantiation, the codebase gains:

- Decoupled task wiring from task execution, enabling pluggable tasks without changing consumer code.
- Clear separation of concerns: the TaskRegistry manages task metadata, while the Runner Registry handles the selection and configuration of RL algorithms.
- Easier testing: tests can register mock tasks, probe registry behavior, and exercise make_env/make_alg_runner without side effects on the real environment classes.
- Extensibility: adding a new task becomes a matter of registering the mapping, without modifying the core training loop or runner orchestration logic.

This design aligns with the broader goals of LeggedGym-Ex, where many robots share a common interface but have varied configurations and training regimes. The TaskRegistry provides a thin, predictable surface to instantiate and run tasks across a spectrum of experiments.

## 7. References and Further Reading

- Code reference: the task registry implementation lives in legged_gym.utils.task_registry (see the repository at LeggedGym-Ex/legged_gym/utils/task_registry.py).
- Runner integration: see legged_gym.utils.runner_registry.py for the central runner construction logic used by make_alg_runner.
- Environment registrations: legged_gym.envs.__init__ contains the actual registrations that wire task names to environment classes and base configurations.

## 8. Summary

The TaskRegistry API provides a stable, documented pathway to register new tasks and to construct environments and PPO runners in a decoupled, testable fashion. By enforcing a naming convention, guarding against misregistrations, and delegating algorithm-specific wiring to a dedicated Runner Registry, the system remains extensible and maintainable as new robots and training strategies are added.

 
