# Troubleshooting Guide

This comprehensive guide helps you diagnose and resolve common issues when working with LeggedGym-Ex. Each section covers specific categories of problems with actionable solutions.

```{contents}
:depth: 2
:local:
```

## Installation Issues

### CUDA Version Mismatch

**Problem**: Training fails with CUDA-related errors or torch cannot detect GPU.

**Error Messages**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX MiB
AssertionError: Torch not compiled with CUDA enabled
NVIDIA driver version is incompatible with CUDA version
```

**Root Cause**: The PyTorch CUDA version must match your system's NVIDIA driver and CUDA toolkit.

**Solution**:

1. Check your NVIDIA driver version:
```bash
nvidia-smi
```

2. Verify PyTorch CUDA version:
```python
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
```

3. Install compatible PyTorch version:

For IsaacGym (Python 3.8, CUDA 12.1):
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

For Genesis (Python 3.10, CUDA 12.6):
```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
```

For IsaacLab (Python 3.11, CUDA 12.6):
```bash
# IsaacSim includes PyTorch, verify after installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Prevention**: Always check driver compatibility before creating conda environments. Minimum driver version: 570.

### IsaacGym Installation Errors

**Problem**: IsaacGym Preview 4 fails to install or import.

**Error Messages**:
```
ModuleNotFoundError: No module named 'isaacgym'
ImportError: libpython3.8.so.1.0: cannot open shared object file
OSError: Cannot load IsaacGym library
```

**Root Cause**: IsaacGym requires specific Python version and proper library paths.

**Solution**:

1. Download IsaacGym Preview 4 from NVIDIA Developer website.

2. Create Python 3.8 environment:
```bash
conda create -n lr_gym python=3.8
conda activate lr_gym
```

3. Install IsaacGym:
```bash
cd isaacgym/python
pip install -e .
```

4. Verify installation:
```bash
python -c "from isaacgym import gymapi; print('IsaacGym installed successfully')"
```

5. If library errors persist, add to `~/.bashrc`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

**Alternative**: Use conda to manage library paths:
```bash
conda install -c conda-forge gcc_linux-64 gxx_linux-64
```

### Conda Environment Conflicts

**Problem**: Package conflicts or wrong Python version in conda environment.

**Error Messages**:
```
UnsatisfiableError: The following specifications were found to be incompatible
ERROR: pip's dependency resolver does not currently take into account all the packages
AssertionError: Python version mismatch
```

**Root Cause**: Mixing pip and conda installations, or creating environments with wrong Python versions.

**Solution**:

1. Clean start - remove conflicting environment:
```bash
conda deactivate
conda env remove -n problematic_env
```

2. Create fresh environment with correct Python version:
```bash
# IsaacGym requires Python 3.8
conda create -n lr_gym python=3.8 -y

# Genesis requires Python 3.10
conda create -n lr_gen python=3.10 -y

# IsaacLab requires Python 3.11
conda create -n lr_lab python=3.11 -y
```

3. Install packages in correct order:
```bash
# Install PyTorch first
pip install torch torchvision --index-url <appropriate_url>

# Then install LeggedGym-Ex
pip install -e ".[isaacgym]"  # or [genesis], [isaaclab]
```

**Best Practice**: Never mix `conda install` and `pip install` for core packages. Use pip exclusively for PyTorch and project dependencies.

## Training Issues

### NaN/Inf Loss Values

**Problem**: Training produces NaN or Inf loss values, causing training to fail.

**Error Messages**:
```
RuntimeError: Function 'MseLossBackward0' returned nan values in its 0th output
ValueError: Detected inf or nan values in loss
```

**Root Cause**: Multiple possible causes including learning rate too high, reward scaling issues, or numerical instability.

**Diagnostic Commands**:
```python
# Check for NaN in observations
import torch
if torch.isnan(env.obs_buf).any():
    print("NaN detected in observations")
    
# Check for NaN in actions
if torch.isnan(actions).any():
    print("NaN detected in actions")

# Check reward values
print(f"Rewards - min: {env.rew_buf.min()}, max: {env.rew_buf.max()}, mean: {env.rew_buf.mean()}")
```

**Solutions**:

1. **Reduce learning rate** in config:
```python
class LeggedRobotCfgPPO:
    class algorithm:
        learning_rate = 1e-4  # Reduce from 1e-3
```

2. **Check reward scales** - ensure no extreme values:
```python
class rewards:
    class scales:
        tracking_lin_vel = 1.0  # Typical values are 0.1 to 2.0
        # Avoid extremely large scales like 100.0
```

3. **Enable gradient clipping**:
```python
class algorithm:
    max_grad_norm = 1.0  # Add gradient clipping
```

4. **Check action bounds** in config:
```python
class control:
    action_scale = 0.25  # Reduce if actions are too large
    clip_actions = 100.0  # Clip extreme actions
```

5. **Validate observation normalization**:
```bash
# Add normalization debugging
python -m legged_gym.scripts.train --task go2 --debug
```

### Out of Memory (OOM) Errors

**Problem**: Training crashes with GPU or CPU out of memory errors.

**Error Messages**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX MiB
torch.OutOfMemoryError: CUDA out of memory
MemoryError: Unable to allocate array
```

**Diagnostic Commands**:
```bash
# Check GPU memory usage
nvidia-smi

# Monitor during training
watch -n 1 nvidia-smi
```

**Solutions**:

1. **Reduce number of parallel environments**:
```bash
python -m legged_gym.scripts.train --task go2 --num_envs 2048  # Default is 4096
```

2. **Reduce batch size** in config:
```python
class LeggedRobotCfgPPO:
    class algorithm:
        num_mini_batches = 8  # Increase to reduce batch size
```

3. **Enable gradient checkpointing** (if available):
```python
class algorithm:
    use_gradient_checkpointing = True
```

4. **Clear cache between iterations**:
```python
import torch
torch.cuda.empty_cache()  # Add in training loop
```

5. **Use mixed precision training**:
```python
class algorithm:
    use_amp = True  # Automatic Mixed Precision
```

**Memory Estimation**:
- Each environment: ~10-50 MB depending on robot complexity
- For 10GB GPU: max ~2000 environments with safety margin
- For 24GB GPU: max ~4000 environments

### Training Not Progressing

**Problem**: Training runs but rewards/losses don't improve over iterations.

**Error Messages**: No explicit error, but learning curves are flat.

**Diagnostic Commands**:
```bash
# Check tensorboard logs
tensorboard --logdir logs/

# Monitor reward components
python -m legged_gym.scripts.train --task go2 --debug
```

**Root Causes and Solutions**:

1. **Reward function issues**: Check that rewards are being computed correctly.

```python
# In robot config, enable reward logging
class rewards:
    class scales:
        tracking_lin_vel = 1.0  # Ensure weight is non-zero
```

2. **Observation issues**: Verify observations contain useful information.

```python
# Add observation debugging
class env:
    debug_observations = True
```

3. **Learning rate too low**: Increase learning rate.

```python
class algorithm:
    learning_rate = 3e-4  # Try higher if stuck
```

4. **Insufficient exploration**: Increase action noise or entropy coefficient.

```python
class algorithm:
    entropy_coef = 0.01  # Increase for more exploration
```

5. **Check termination conditions**: Overly strict terminations prevent learning.

```python
class terminations:
    termination_if_close_to_ground = 0.3  # Adjust threshold
```

### Slow Training Speed

**Problem**: Training is significantly slower than expected.

**Expected Performance**:
- IsaacGym: 20,000-50,000 FPS with 4096 environments
- Genesis: 10,000-30,000 FPS with 4096 environments
- IsaacLab: 5,000-15,000 FPS with 4096 environments

**Diagnostic Commands**:
```bash
# Check FPS during training
python -m legged_gym.scripts.train --task go2 --headless

# Monitor GPU utilization
nvidia-smi dmon -s u
```

**Solutions**:

1. **Enable headless mode**:
```bash
python -m legged_gym.scripts.train --task go2 --headless
```

2. **Optimize terrain generation**:
```python
class terrain:
    mesh_type = 'plane'  # Fastest for initial testing
    # mesh_type = 'trimesh'  # Slower but more realistic
```

3. **Reduce observation size**:
```python
class env:
    num_observations = 48  # Minimize for speed
    num_privileged_obs = None  # Disable if not using TS
```

4. **Adjust simulation frequency**:
```python
class sim:
    dt = 0.02  # Control frequency
    substeps = 1  # Reduce for speed (default: 4)
    # Warning: may affect stability
```

5. **Use simpler robot models**:
```python
class asset:
    self_collisions = 0  # Disable self-collision checking
    fix_base_link = False  # Keep False for locomotion
```

## Inference Issues

### Model Not Found

**Problem**: Cannot locate trained model checkpoint for inference.

**Error Messages**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'logs/...'
RuntimeError: Cannot load model from path
```

**Root Cause**: Wrong path or model not saved.

**Solution**:

1. **List available models**:
```bash
ls -R logs/
```

2. **Check experiment directory structure**:
```
logs/
└── <experiment_name>/
    └── <datetime>/
        ├── config.json
        ├── model_<iteration>.pt
        └── model_latest.pt
```

3. **Specify correct path**:
```bash
# Using experiment name (finds latest)
python -m legged_gym.scripts.play --task go2 --resume

# Using specific run
python -m legged_gym.scripts.play --task go2 --load_run <datetime>

# Using exact path
python -m legged_gym.scripts.play --task go2 --load_run logs/go2/20250403_123456
```

4. **Verify model exists**:
```python
import torch
checkpoint = torch.load('logs/go2/.../model_1000.pt')
print(checkpoint.keys())  # Should contain 'model_state_dict', 'optimizer_state_dict'
```

### JIT Export Errors

**Problem**: Cannot export model to TorchScript format for deployment.

**Error Messages**:
```
RuntimeError: Exporting the operator 'aten::grid_sampler_2d' is not supported
RuntimeError: Cannot extract guaranteed root tensor from output
```

**Root Cause**: Model contains operations not supported by TorchScript.

**Solutions**:

1. **Use correct export command**:
```bash
python -m legged_gym.scripts.play --task go2 --export
```

2. **Check exported model**:
```python
import torch
model = torch.jit.load('logs/go2/.../exported/policy.pt')
print(model.code)
```

3. **Handle observation normalization**:
```python
# Export includes normalization if configured
# Check config:
class normalizations:
    class observations:
        clip_observations = 100.0
```

4. **For Teacher-Student models**, export student network:
```bash
python -m legged_gym.scripts.play --task go2_ts --export
# Exported model only uses student observations (no privileged info)
```

### Visualization Issues

**Problem**: Cannot visualize training or inference, or viewer crashes.

**Error Messages**:
```
RuntimeError: Failed to create window
GLFW Error: X11: Failed to open display
Segmentation fault (core dumped)
```

**Root Cause**: Display or graphics driver issues.

**Solutions**:

1. **For headless servers**, disable visualization:
```bash
python -m legged_gym.scripts.train --task go2 --headless
```

2. **Check display settings**:
```bash
echo $DISPLAY
# Should output something like :0 or :1

# If empty, set it:
export DISPLAY=:0
```

3. **For remote servers with X forwarding**:
```bash
# On local machine
xhost +

# SSH with X forwarding
ssh -X user@server

# Then run training
python -m legged_gym.scripts.train --task go2
```

4. **Use VirtualGL for remote rendering**:
```bash
vglrun python -m legged_gym.scripts.train --task go2
```

5. **Record video instead of live viewer**:
```bash
python -m legged_gym.scripts.play --task go2 --record
```

## Multi-Simulator Issues

### IsaacGym Reset Bug

**Problem**: After calling `reset()`, rigid body states are incorrect, causing abnormal terminations.

**Error Messages**:
```
# No explicit error, but unexpected behavior:
# - Robot appears in wrong position after reset
# - Reference motion tracking fails after reset
# - Termination triggers incorrectly
```

**Root Cause**: IsaacGym requires one simulation step after `reset()` to update rigid body states properly. This is a known bug in IsaacGym Preview 4.

**Solution**:

Add `simulator.forward()` call after reset:

```python
# In reset_idx or post_physics_step method:
def reset_idx(self, env_ids):
    # ... reset logic ...
    
    # BUG FIX: Call forward() to update rigid body states
    if self.cfg.simulator == 'isaacgym':
        self.simulator.forward()
    
    # Now rigid body states are correct
```

**Affected Methods**:
- `_reset_dofs()`
- `_reset_root_states()`
- `_reset_dofs_from_reference_motion()`
- `_reset_root_states_from_reference_motion()`

**Example from codebase** (see `g1_deepmimic.py:73`):
```python
# BUG: IsaacGym requires 1 step after resetting to get the correct rigid body states
# When enabling reference motion termination, the rigid body state does not update
# after this reset, which causes the termination abnormally.
# The dof state and root state is reset correctly, but the rigid body state is not updated

# Solution is already implemented in DeepMimic tasks
# Apply same pattern if you encounter this issue
```

### IsaacLab CPU Tensor Requirement

**Problem**: Domain randomization functions fail with device errors in IsaacLab.

**Error Messages**:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
AssertionError: Domain randomization tensors must be on CPU for IsaacLab
```

**Root Cause**: IsaacLab's backend requires domain randomization tensors to be on CPU, unlike IsaacGym which uses GPU tensors.

**Solution**:

Move tensors to CPU before calling randomization functions:

```python
# Wrong (IsaacGym style):
self.simulator.set_material_properties(
    env_ids, 
    friction_tensors.cuda()  # Fails in IsaacLab
)

# Correct (IsaacLab compatible):
self.simulator.set_material_properties(
    env_ids,
    friction_tensors.cpu()  # Must be CPU tensors
)
```

**Affected Functions**:
- `set_material_properties()`
- `set_masses()`
- `set_coms()` (center of mass)
- `set_friction()`

**Cross-simulator Code**:

```python
def _randomize_friction(self, env_ids):
    friction = torch.rand(len(env_ids), device='cpu')  # Create on CPU
    friction = friction * (self.cfg.domain_rand.friction_range[1] - 
                          self.cfg.domain_rand.friction_range[0]) + \
               self.cfg.domain_rand.friction_range[0]
    
    # Works for all simulators
    self.simulator.set_material_properties(
        env_ids.cpu() if hasattr(env_ids, 'cpu') else env_ids,
        friction
    )
```

### Genesis XML Requirement

**Problem**: Genesis simulator fails to load robot or environment.

**Error Messages**:
```
ValueError: XML file path must be provided for Genesis simulator
FileNotFoundError: Cannot find robot XML file
genesis.sim: Failed to load URDF from XML
```

**Root Cause**: Genesis requires XML configuration files for robot and scene setup, unlike IsaacGym which uses programmatic API.

**Solution**:

1. **Ensure XML file exists**:
```bash
ls resources/robots/go2/urdf/go2.xml
```

2. **Check XML configuration**:
```python
class asset:
    file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.xml'
```

3. **Verify path resolution**:
```python
# In robot config
import os
xml_path = os.path.join(
    os.path.dirname(__file__),
    '../../../resources/robots/go2/urdf/go2.xml'
)
assert os.path.exists(xml_path), f"XML not found: {xml_path}"
```

4. **Set SIMULATOR environment variable**:
```bash
export SIMULATOR=genesis
python -m legged_gym.scripts.train --task go2
```

### Terrain Configuration Conflicts

**Problem**: Terrain generation fails with conflicting options.

**Error Messages**:
```
ValueError: Curriculum and selected terrain cannot be both True.
```

**Root Cause**: Cannot use curriculum terrain and selected terrain simultaneously.

**Solution**:

Choose one terrain mode:

```python
# Option 1: Curriculum terrain (difficulty increases over training)
class terrain:
    curriculum = True
    selected = False
    terrain_curriculum_difficulty = 0.5

# Option 2: Selected terrain (fixed terrain types)
class terrain:
    curriculum = False
    selected = True
    terrain_proportions = [0.2, 0.3, 0.5]  # Proportions for each terrain type
```

**Code Reference** (see `terrain.py:63`):
```python
if cfg.curriculum and cfg.selected:
    raise ValueError("Curriculum and selected terrain cannot be both True.")
```

### Heightfield Terrain Limitation

**Problem**: Heightfield terrain not working in IsaacLab.

**Error Messages**:
```
NotImplementedError: Heightfield terrain not implemented for IsaacLabSimulator
RuntimeError: Cannot create heightfield in IsaacLab
```

**Root Cause**: Heightfield terrain generation is not implemented for IsaacLab backend.

**Solution**:

Use trimesh terrain instead:

```python
class terrain:
    mesh_type = 'trimesh'  # Works for IsaacLab
    # mesh_type = 'heightfield'  # NOT supported in IsaacLab
```

**Simulator Compatibility Matrix**:

| Terrain Type | IsaacGym | Genesis | IsaacLab |
|-------------|----------|---------|----------|
| plane       | ✓        | ✓       | ✓        |
| heightfield | ✓        | ✓       | ✗        |
| trimesh     | ✓        | ✓       | ✓        |

## Debugging Commands Quick Reference

### Check Environment

```bash
# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Check simulator
echo $SIMULATOR

# List available tasks
python tests/test_all_tasks.py --list

# Check conda environment
conda list | grep -E "torch|genesis|isaac"
```

### Debug Training

```bash
# Run with debug output
python -m legged_gym.scripts.train --task go2 --debug

# Monitor GPU
watch -n 0.5 nvidia-smi

# Check logs
tail -f logs/go2/<experiment>/log.txt

# TensorBoard
tensorboard --logdir logs/
```

### Validate Installation

```bash
# Test specific task
python tests/test_all_tasks.py --tasks go2 --iterations 1

# Test all tasks
python tests/test_all_tasks.py

# Verify simulator backend
python -c "from legged_gym.simulator import get_simulator; print(get_simulator.__name__)"
```

## Getting Help

If you cannot resolve an issue using this guide:

1. **Check Documentation**: Full documentation at https://leggedgym-ex-doc.readthedocs.io/

2. **Search Issues**: Check existing GitHub issues for similar problems

3. **Debug Information**: When asking for help, provide:
   - Simulator type (`echo $SIMULATOR`)
   - Python version (`python --version`)
   - PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - CUDA version (`nvcc --version` or `nvidia-smi`)
   - Complete error message and stack trace
   - Minimal reproduction script

4. **Community**: Join the Feishu group (see README) for discussions

5. **Bug Reports**: Open a GitHub issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
