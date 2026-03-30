# 🧬Code Architecture

## Overall Architecture of genesis_lr

```{figure} ../../_static/images/code_structure.png
```

The whole project can be separated into 3 parts: simulator, environments and algorithms.

## Simulator (legged_gym/simulator)

The simulator layer provides unified api for different simulators, including IsaacGym and Genesis. Users can choose either simulator to use for training. 

Apart from sensors embedded in simulators, we implemented external augmented sensors, mainly for IsaacGym. Currently, we provide [warp](https://github.com/NVIDIA/warp)-based depth camera sensors to accelerate rendering in IsaacGym. Compared to the embedded depth camera in IsaacGym, warp-based depth camera can provide 2-3x faster rendering speed during headless training.

## Environments (legged_gym/envs)

Environments are where the agent collects data and receives reward signal from. Users can define the task and the dynamics. Environments follow the inheritance style. Users can define new environment classes inheriting base classes (legged_robot.py). Configuration (config) files in environments are responsible for storing parameters and settings for environments. 

## Algorithms (rsl_rl)

Algorithms define reinforment learning algorithms used in training. Currently we implement PPO (Proximal Policy Optimization), [SPO (Simple Policy Optimizatino)](https://github.com/MyRepositories-hub/Simple-Policy-Optimization) and several training architecture based on them (Explicit Estimator, Teacher Student, Cocurrent TS, DreamWaQ). 

## Sim2Sim

