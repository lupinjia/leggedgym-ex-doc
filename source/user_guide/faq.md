# Frequently Asked Questions (FAQ)

This FAQ addresses common questions about LeggedGym-Ex. If you cannot find an answer here, check the detailed documentation sections or join our community discussions.

---

## General

### What is LeggedGym-Ex?

LeggedGym-Ex is a reinforcement learning framework for training legged robot locomotion policies. It extends the original [legged_gym](https://github.com/leggedrobotics/legged_gym) framework with support for multiple simulators (Genesis, IsaacGym, IsaacSim) and implementations of 10+ published methods from recent robotics research.

### What robots are supported?

LeggedGym-Ex currently supports Unitree Go2 (quadruped), Unitree G1 (biped), LimX TRON1 (quadruped), and Booster K1 (quadruped). You can also add your own robot by following the guide in {doc}`/developer_guide/how_to/add_new_robot`.

### What is the difference between LeggedGym-Ex and the original legged_gym?

LeggedGym-Ex maintains API compatibility with legged_gym while adding multi-simulator support and implementations of advanced methods like Teacher-Student, Explicit Estimator, DreamWaQ, DeepMimic, and AMP. See the [README](https://github.com/lupinjia/LeggedGym-Ex) for a full feature comparison.

### Can I use this framework for research publications?

Yes. LeggedGym-Ex is open-source and designed for research use. When publishing, please cite both the original legged_gym paper and acknowledge the specific methods you use (DeepMimic, AMP, etc.).

### Is commercial use allowed?

Yes, the framework is released under the BSD-3-Clause license. You can use it for commercial applications, including deploying policies on commercial robots.

### Where can I get help?

You can get help through several channels: the GitHub issues page for bug reports, the Feishu group (QR code in README) for community discussions, and the documentation for detailed guides.

### How do I report bugs or request features?

Open an issue on the [GitHub repository](https://github.com/lupinjia/LeggedGym-Ex). Include your simulator version, Python version, and a minimal reproduction script when reporting bugs.

---

## Installation

### Which simulator should I use?

Choose based on your needs:

- **IsaacGym**: Fastest training, best for rapid iteration, requires Python 3.8
- **Genesis**: Good balance of speed and rendering quality, supports soft materials, requires Python 3.10+
- **IsaacLab/IsaacSim**: Best rendering quality, official NVIDIA support, requires Python 3.11+

See {doc}`getting_started/installation` for detailed setup instructions for each simulator.

### What hardware do I need?

Minimum requirements:

- **CPU**: Intel Core i7 or AMD Ryzen 7 (i9 recommended)
- **GPU**: NVIDIA RTX 3060 8GB+ (RTX 3080 10GB+ recommended)
- **RAM**: 32GB recommended
- **OS**: Ubuntu 20.04 or 22.04
- **NVIDIA Driver**: 570 or newer

Training on smaller GPUs is possible by reducing `num_envs` and terrain complexity.

### Do I need separate conda environments for different simulators?

Yes. IsaacGym requires Python 3.8, while Genesis and IsaacLab require Python 3.10+. We provide a `switch_simulator.sh` script to help you switch between environments easily.

### How do I switch between simulators?

Use the provided switch script:

```bash
source ./switch_simulator.sh
```

Or manually activate the appropriate environment and set the `SIMULATOR` environment variable. See {doc}`getting_started/quick_start` for details.

### Installation fails with "CUDA out of memory" during testing

This is normal during the first run. The test script tries to create many environments. Reduce the number of test environments:

```bash
python legged_gym/scripts/train.py --task=go2 --num_envs=100 --headless
```

### Can I use AMD or Intel GPUs?

No. LeggedGym-Ex requires NVIDIA GPUs with CUDA support. The simulators (IsaacGym, Genesis, IsaacSim) all depend on CUDA.

### Can I install on Windows or macOS?

Officially, only Ubuntu Linux is supported. Windows users can try WSL2, but this is not officially supported. macOS is not supported due to CUDA requirements.

### ImportError: libpython3.8.so.1.0 not found

Add your conda environment's lib directory to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/home/username/miniconda3/envs/lr_gym/lib:$LD_LIBRARY_PATH
```

---

## Training

### How many parallel environments do I need?

The default is 4096 environments, which works well on an RTX 3080. You can reduce this based on your GPU memory:

- **RTX 3060 12GB**: 2048-3072 environments
- **RTX 3080 10GB**: 4096 environments
- **RTX 4090 24GB**: 8192+ environments

Adjust `num_envs` in your config file or via command line: `--num_envs=2048`.

### How long should I train?

Training time depends on the task complexity:

- **Flat terrain walking**: 500-1000 iterations (2-4 hours)
- **Rough terrain (TS/EE)**: 1500-3000 iterations (6-12 hours)
- **Advanced methods (DeepMimic, AMP)**: 3000+ iterations (12+ hours)

Use TensorBoard to monitor when rewards plateau.

### What is a good reward value?

This varies by task. Generally:

- **Flat terrain**: Mean reward > 15 indicates good walking
- **Rough terrain**: Mean reward > 10 with high episode lengths (>800 steps) indicates robustness
- **Tracking rewards**: Values > 0.8 indicate good command following

Monitor `episode_length` alongside rewards. Long episodes mean the robot is staying upright.

### Training loss is NaN. What should I do?

Common causes and solutions:

1. **Learning rate too high**: Reduce `learning_rate` in the PPO config (try 3e-4)
2. **Reward scales too large**: Check that penalty rewards are properly negative
3. **Observation explosion**: Verify observation scaling factors are reasonable
4. **Gradient clipping**: Ensure `clip_param` is set to 0.2

### Training is slow. How can I speed it up?

- Use `--headless` mode to disable rendering
- Reduce `num_envs` if GPU memory is constrained
- Use IsaacGym for fastest training
- Disable `measure_heights` if not needed for your task
- Reduce `episode_length_s` for faster iteration

### How do I resume training from a checkpoint?

```bash
python -m legged_gym.scripts.train --task=go2 --resume --load_run=Sep03_16-30-16_
```

Replace the run name with your actual training session folder name.

### Can I train on CPU only?

No. LeggedGym-Ex requires a CUDA-capable GPU for training. The parallel environment simulation is GPU-accelerated.

### How do I monitor training progress?

Training metrics are automatically logged to TensorBoard:

```bash
tensorboard --logdir logs/go2/
```

Enable Weights & Biases by adding `--sync_wandb` to the training command.

### What are the most important hyperparameters to tune?

For most tasks, focus on:

1. **Reward scales**: Balance tracking rewards vs penalties
2. **Command ranges**: Start conservative, expand with curriculum
3. **Domain randomization**: Essential for sim-to-real transfer
4. **Learning rate**: 1e-3 is default, reduce if training unstable

See {doc}`/developer_guide/how_to/custom_rewards` for reward tuning guidance.

---

## Simulators

### Can I use the same code across all simulators?

Yes. LeggedGym-Ex abstracts simulator differences behind a unified API. Simply set the `SIMULATOR` environment variable and use the appropriate conda environment.

### Why does my policy behave differently in different simulators?

Physics engines handle contacts, friction, and numerical integration differently. This is expected. For critical applications, train in one simulator and validate in another (Sim2Sim testing). See {doc}`getting_started/code_architecture` for best practices.

### Genesis shows mesh terrain collision issues

Genesis has known limitations with custom mesh terrain collision detection. Use `mesh_type = "heightfield"` or `mesh_type = "plane"` instead. See {doc}`/developer_guide/known_issues/genesis` for details.

### IsaacGym window does not show the rendered world

With NVIDIA driver >= 570, set the Vulkan ICD path:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### IsaacLab domain randomization errors

IsaacLab requires domain randomization tensors to be on CPU. This is handled internally by LeggedGym-Ex, but be aware that DR may be slower in IsaacLab compared to other simulators.

### Can I render depth images for training?

Depth camera support varies by simulator. IsaacGym supports warp-based depth rendering. Genesis and IsaacLab have their own camera implementations. Check task-specific implementations like `go2_ts_depth` for examples.

---

## Deployment

### Can I deploy my trained policy to a real robot?

Yes. LeggedGym-Ex supports deployment to Unitree Go2, TRON1, and other supported robots. See {doc}`getting_started/deploy_to_real_robot` for detailed instructions.

### What do I need to change for real robot deployment?

The main change is removing `base_lin_vel` from observations since linear velocity cannot be directly measured on real robots. Train a new policy without this observation component. See the deployment guide for the full checklist.

### How do I export a policy for deployment?

After training, run:

```bash
python -m legged_gym.scripts.play --task=go2 --load_run=your_run_name
```

This creates a JIT-compiled policy file in `logs/experiment_name/exported/policy.pt`.

### What is Sim2Sim testing?

Sim2Sim (Simulation-to-Simulation) validates your policy in a different simulator before real deployment. This catches simulator-specific overfitting. We provide MuJoCo-based Sim2Sim setups for Go2 and TRON1. See {doc}`getting_started/code_architecture`.

### How do I test my policy in MuJoCo before real deployment?

Install the Sim2Sim framework (see {doc}`getting_started/installation`), export your policy, and run:

```bash
./go2_deploy simple_rl
```

This tests your policy in MuJoCo without risking the real robot.

### Can I use my own robot URDF?

Yes. Follow the guide in {doc}`/developer_guide/how_to/add_new_robot` to add your robot. You'll need:

1. Valid URDF (for IsaacGym/IsaacLab) or MJCF XML (for Genesis)
2. Joint names and default positions
3. Properly defined collision meshes

### Do I need different policies for different simulators?

Policies are generally transferable between simulators, but performance may vary due to physics differences. For best results, train and deploy using the same simulator, or use domain randomization to improve cross-simulator robustness.

### What control frequency should I use on the real robot?

The default is 50Hz (`dt = 0.02`), which matches typical robot hardware capabilities. Some robots support 100Hz or higher. Ensure your deployment code maintains consistent timing with the training configuration.

### Can I deploy to robots not officially supported?

Yes, but you'll need to write your own deployment code. The key is aligning observations and actions with your robot's sensor interface and motor controllers. Use the provided Go2/TRON1 deployment code as a reference.

### How do I handle network communication with the robot?

For Unitree Go2, use the provided `go2_deploy` framework which handles DDS communication. For other robots, implement your own communication layer based on the robot's SDK. The policy itself expects only the observation tensor as input.

---

## Quick Reference

### Essential Commands

```bash
# Training
python -m legged_gym.scripts.train --task=go2 --headless

# Visualization
python -m legged_gym.scripts.play --task=go2 --load_run=run_name

# List tasks
python -c "from legged_gym.envs import task_registry; print(list(task_registry.task_classes.keys()))"

# TensorBoard
tensorboard --logdir logs/
```

### Key Configuration Files

- Environment: `legged_gym/envs/go2/go2_config.py`
- Training: PPO settings in config file
- Rewards: `class rewards` in config
- Commands: `class commands` in config

### Useful Links

- [GitHub Repository](https://github.com/lupinjia/LeggedGym-Ex)
- [Full Documentation](https://genesis-lr-doc.readthedocs.io/)
- [legged_gym Original](https://github.com/leggedrobotics/legged_gym)
