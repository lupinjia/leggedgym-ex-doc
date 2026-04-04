# 故障排除指南

本综合指南帮助您诊断和解决使用LeggedGym-Ex时的常见问题。每个部分涵盖特定类别的问题及可行的解决方案。

```{contents}
:depth: 2
:local:
```

## 安装问题

### CUDA版本不匹配

**问题**：训练失败并出现CUDA相关错误，或torch无法检测到GPU。

**错误信息**：
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX MiB
AssertionError: Torch not compiled with CUDA enabled
NVIDIA driver version is incompatible with CUDA version
```

**根本原因**：PyTorch CUDA版本必须与系统的NVIDIA驱动和CUDA工具包匹配。

**解决方案**：

1. 检查您的NVIDIA驱动版本：
```bash
nvidia-smi
```

2. 验证PyTorch CUDA版本：
```python
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
```

3. 安装兼容的PyTorch版本：

对于IsaacGym（Python 3.8，CUDA 12.1）：
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

对于Genesis（Python 3.10，CUDA 12.6）：
```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
```

对于IsaacLab（Python 3.11，CUDA 12.6）：
```bash
# IsaacSim包含PyTorch，安装后验证
python -c "import torch; print(torch.cuda.is_available())"
```

**预防措施**：创建conda环境前始终检查驱动兼容性。最低驱动版本：570。

### IsaacGym安装错误

**问题**：IsaacGym Preview 4无法安装或导入。

**错误信息**：
```
ModuleNotFoundError: No module named 'isaacgym'
ImportError: libpython3.8.so.1.0: cannot open shared object file
OSError: Cannot load IsaacGym library
```

**根本原因**：IsaacGym需要特定的Python版本和正确的库路径。

**解决方案**：

1. 从NVIDIA开发者网站下载IsaacGym Preview 4。

2. 创建Python 3.8环境：
```bash
conda create -n lr_gym python=3.8
conda activate lr_gym
```

3. 安装IsaacGym：
```bash
cd isaacgym/python
pip install -e .
```

4. 验证安装：
```bash
python -c "from isaacgym import gymapi; print('IsaacGym installed successfully')"
```

5. 如果库错误持续存在，添加到`~/.bashrc`：
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

**替代方案**：使用conda管理库路径：
```bash
conda install -c conda-forge gcc_linux-64 gxx_linux-64
```

### Conda环境冲突

**问题**：conda环境中的包冲突或错误的Python版本。

**错误信息**：
```
UnsatisfiableError: The following specifications were found to be incompatible
ERROR: pip's dependency resolver does not currently take into account all the packages
AssertionError: Python version mismatch
```

**根本原因**：混合使用pip和conda安装，或创建具有错误Python版本的环境。

**解决方案**：

1. 干净开始 - 移除冲突的环境：
```bash
conda deactivate
conda env remove -n problematic_env
```

2. 使用正确的Python版本创建新环境：
```bash
# IsaacGym需要Python 3.8
conda create -n lr_gym python=3.8 -y

# Genesis需要Python 3.10
conda create -n lr_gen python=3.10 -y

# IsaacLab需要Python 3.11
conda create -n lr_lab python=3.11 -y
```

3. 按正确顺序安装包：
```bash
# 首先安装PyTorch
pip install torch torchvision --index-url <appropriate_url>

# 然后安装LeggedGym-Ex
pip install -e ".[isaacgym]"  # 或 [genesis], [isaaclab]
```

**最佳实践**：对于核心包永远不要混合使用`conda install`和`pip install`。使用pip安装PyTorch和项目依赖。

## 训练问题

### NaN/Inf损失值

**问题**：训练产生NaN或Inf损失值，导致训练失败。

**错误信息**：
```
RuntimeError: Function 'MseLossBackward0' returned nan values in its 0th output
ValueError: Detected inf or nan values in loss
```

**根本原因**：多种可能的原因，包括学习率过高、奖励缩放问题或数值不稳定。

**诊断命令**：
```python
# 检查观测值中的NaN
import torch
if torch.isnan(env.obs_buf).any():
    print("NaN detected in observations")
    
# 检查动作中的NaN
if torch.isnan(actions).any():
    print("NaN detected in actions")

# 检查奖励值
print(f"Rewards - min: {env.rew_buf.min()}, max: {env.rew_buf.max()}, mean: {env.rew_buf.mean()}")
```

**解决方案**：

1. **降低学习率**：
```python
class LeggedRobotCfgPPO:
    class algorithm:
        learning_rate = 1e-4  # 从1e-3降低
```

2. **检查奖励比例** - 确保没有极端值：
```python
class rewards:
    class scales:
        tracking_lin_vel = 1.0  # 典型值为0.1到2.0
        # 避免极大的比例如100.0
```

3. **启用梯度裁剪**：
```python
class algorithm:
    max_grad_norm = 1.0  # 添加梯度裁剪
```

4. **检查动作边界**：
```python
class control:
    action_scale = 0.25  # 如果动作过大则降低
    clip_actions = 100.0  # 裁剪极端动作
```

5. **验证观测值归一化**：
```bash
# 添加归一化调试
python -m legged_gym.scripts.train --task go2 --debug
```

### 内存不足 (OOM) 错误

**问题**：训练因GPU或CPU内存不足而崩溃。

**错误信息**：
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX MiB
torch.OutOfMemoryError: CUDA out of memory
MemoryError: Unable to allocate array
```

**诊断命令**：
```bash
# 检查GPU内存使用
nvidia-smi

# 训练期间监控
watch -n 1 nvidia-smi
```

**解决方案**：

1. **减少并行环境数量**：
```bash
python -m legged_gym.scripts.train --task go2 --num_envs 2048  # 默认是4096
```

2. **减小批次大小**：
```python
class LeggedRobotCfgPPO:
    class algorithm:
        num_mini_batches = 8  # 增加以减少批次大小
```

3. **启用梯度检查点**（如果可用）：
```python
class algorithm:
    use_gradient_checkpointing = True
```

4. **迭代之间清除缓存**：
```python
import torch
torch.cuda.empty_cache()  # 在训练循环中添加
```

5. **使用混合精度训练**：
```python
class algorithm:
    use_amp = True  # 自动混合精度
```

**内存估算**：
- 每个环境：根据机器人复杂度约10-50 MB
- 对于10GB GPU：安全余量下最多约2000个环境
- 对于24GB GPU：最多约4000个环境

### 训练没有进展

**问题**：训练运行但奖励/损失在迭代中没有改善。

**错误信息**：没有明确的错误，但学习曲线是平的。

**诊断命令**：
```bash
# 检查tensorboard日志
tensorboard --logdir logs/

# 监控奖励组件
python -m legged_gym.scripts.train --task go2 --debug
```

**根本原因和解决方案**：

1. **奖励函数问题**：检查奖励是否正确计算。

```python
# 在机器人配置中，启用奖励日志记录
class rewards:
    class scales:
        tracking_lin_vel = 1.0  # 确保权重非零
```

2. **观测值问题**：验证观测值包含有用信息。

```python
# 添加观测值调试
class env:
    debug_observations = True
```

3. **学习率过低**：提高学习率。

```python
class algorithm:
    learning_rate = 3e-4  # 如果卡住尝试更高的值
```

4. **探索不足**：增加动作噪声或熵系数。

```python
class algorithm:
    entropy_coef = 0.01  # 增加以获得更多探索
```

5. **检查终止条件**：过于严格的终止会阻止学习。

```python
class terminations:
    termination_if_close_to_ground = 0.3  # 调整阈值
```

### 训练速度慢

**问题**：训练速度明显低于预期。

**预期性能**：
- IsaacGym：4096个环境下20,000-50,000 FPS
- Genesis：4096个环境下10,000-30,000 FPS
- IsaacLab：4096个环境下5,000-15,000 FPS

**诊断命令**：
```bash
# 检查训练期间的FPS
python -m legged_gym.scripts.train --task go2 --headless

# 监控GPU利用率
nvidia-smi dmon -s u
```

**解决方案**：

1. **启用无头模式**：
```bash
python -m legged_gym.scripts.train --task go2 --headless
```

2. **优化地形生成**：
```python
class terrain:
    mesh_type = 'plane'  # 初始测试最快
    # mesh_type = 'trimesh'  # 较慢但更真实
```

3. **减小观测值大小**：
```python
class env:
    num_observations = 48  # 为速度最小化
    num_privileged_obs = None  # 如果不使用TS则禁用
```

4. **调整仿真频率**：
```python
class sim:
    dt = 0.02  # 控制频率
    substeps = 1  # 为速度减少（默认：4）
    # 警告：可能影响稳定性
```

5. **使用更简单的机器人模型**：
```python
class asset:
    self_collisions = 0  # 禁用自碰撞检查
    fix_base_link = False  # 对于运动保持False
```

## 推理问题

### 模型未找到

**问题**：无法找到用于推理的训练模型检查点。

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'logs/...'
RuntimeError: Cannot load model from path
```

**根本原因**：路径错误或模型未保存。

**解决方案**：

1. **列出可用模型**：
```bash
ls -R logs/
```

2. **检查实验目录结构**：
```
logs/
└── <experiment_name>/
    └── <datetime>/
        ├── config.json
        ├── model_<iteration>.pt
        └── model_latest.pt
```

3. **指定正确路径**：
```bash
# 使用实验名称（查找最新的）
python -m legged_gym.scripts.play --task go2 --resume

# 使用特定运行
python -m legged_gym.scripts.play --task go2 --load_run <datetime>

# 使用精确路径
python -m legged_gym.scripts.play --task go2 --load_run logs/go2/20250403_123456
```

4. **验证模型存在**：
```python
import torch
checkpoint = torch.load('logs/go2/.../model_1000.pt')
print(checkpoint.keys())  # 应包含'model_state_dict', 'optimizer_state_dict'
```

### JIT导出错误

**问题**：无法将模型导出为TorchScript格式进行部署。

**错误信息**：
```
RuntimeError: Exporting the operator 'aten::grid_sampler_2d' is not supported
RuntimeError: Cannot extract guaranteed root tensor from output
```

**根本原因**：模型包含TorchScript不支持的操作。

**解决方案**：

1. **使用正确的导出命令**：
```bash
python -m legged_gym.scripts.play --task go2 --export
```

2. **检查导出的模型**：
```python
import torch
model = torch.jit.load('logs/go2/.../exported/policy.pt')
print(model.code)
```

3. **处理观测值归一化**：
```python
# 导出包含归一化（如果配置）
# 检查配置：
class normalizations:
    class observations:
        clip_observations = 100.0
```

4. **对于教师-学生模型**，导出学生网络：
```bash
python -m legged_gym.scripts.play --task go2_ts --export
# 导出的模型仅使用学生观测值（无特权信息）
```

### 可视化问题

**问题**：无法可视化训练或推理，或查看器崩溃。

**错误信息**：
```
RuntimeError: Failed to create window
GLFW Error: X11: Failed to open display
Segmentation fault (core dumped)
```

**根本原因**：显示或图形驱动问题。

**解决方案**：

1. **对于无头服务器**，禁用可视化：
```bash
python -m legged_gym.scripts.train --task go2 --headless
```

2. **检查显示设置**：
```bash
echo $DISPLAY
# 应输出类似:0或:1的内容

# 如果为空，设置它：
export DISPLAY=:0
```

3. **对于带X转发的远程服务器**：
```bash
# 在本地机器上
xhost +

# 使用X转发SSH
ssh -X user@server

# 然后运行训练
python -m legged_gym.scripts.train --task go2
```

4. **使用VirtualGL进行远程渲染**：
```bash
vglrun python -m legged_gym.scripts.train --task go2
```

5. **录制视频而不是实时查看器**：
```bash
python -m legged_gym.scripts.play --task go2 --record
```

## 多仿真器问题

### IsaacGym重置错误

**问题**：调用`reset()`后，刚体状态不正确，导致异常终止。

**错误信息**：
```
# 没有明确的错误，但意外行为：
# - 重置后机器人出现在错误位置
# - 重置后参考运动追踪失败
# - 终止触发不正确
```

**根本原因**：IsaacGym在`reset()`后需要一个仿真步骤才能正确更新刚体状态。这是IsaacGym Preview 4中的已知错误。

**解决方案**：

添加`simulator.forward()`调用在reset之后：

```python
# 在reset_idx或post_physics_step方法中：
def reset_idx(self, env_ids):
    # ... 重置逻辑 ...
    
    # BUG修复：调用forward()更新刚体状态
    if self.cfg.simulator == 'isaacgym':
        self.simulator.forward()
    
    # 现在刚体状态正确
```

**受影响的方法**：
- `_reset_dofs()`
- `_reset_root_states()`
- `_reset_dofs_from_reference_motion()`
- `_reset_root_states_from_reference_motion()`

**代码库示例**（参见`g1_deepmimic.py:73`）：
```python
# BUG: IsaacGym需要在重置后1步才能获得正确的刚体状态
# 启用参考运动终止时，刚体状态在此重置后不更新
# 导致终止异常。
# 自由度状态和根状态重置正确，但刚体状态未更新

# DeepMimic任务中已实现解决方案
# 如果遇到此问题应用相同的模式
```

### IsaacLab CPU张量要求

**问题**：域随机化函数在IsaacLab中出现设备错误。

**错误信息**：
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
AssertionError: Domain randomization tensors must be on CPU for IsaacLab
```

**根本原因**：IsaacLab的后端要求域随机化张量在CPU上，而不像IsaacGym使用GPU张量。

**解决方案**：

在调用随机化函数前将张量移到CPU：

```python
# 错误（IsaacGym风格）：
self.simulator.set_material_properties(
    env_ids, 
    friction_tensors.cuda()  # 在IsaacLab中失败
)

# 正确（兼容IsaacLab）：
self.simulator.set_material_properties(
    env_ids,
    friction_tensors.cpu()  # 必须是CPU张量
)
```

**受影响的函数**：
- `set_material_properties()`
- `set_masses()`
- `set_coms()`（质心）
- `set_friction()`

**跨仿真器代码**：

```python
def _randomize_friction(self, env_ids):
    friction = torch.rand(len(env_ids), device='cpu')  # 在CPU上创建
    friction = friction * (self.cfg.domain_rand.friction_range[1] - 
                          self.cfg.domain_rand.friction_range[0]) + \
               self.cfg.domain_rand.friction_range[0]
    
    # 适用于所有仿真器
    self.simulator.set_material_properties(
        env_ids.cpu() if hasattr(env_ids, 'cpu') else env_ids,
        friction
    )
```

### Genesis XML要求

**问题**：Genesis仿真器无法加载机器人或环境。

**错误信息**：
```
ValueError: XML file path must be provided for Genesis simulator
FileNotFoundError: Cannot find robot XML file
genesis.sim: Failed to load URDF from XML
```

**根本原因**：Genesis需要机器人的XML配置文件和场景设置，不像IsaacGym使用程序化API。

**解决方案**：

1. **确保XML文件存在**：
```bash
ls resources/robots/go2/urdf/go2.xml
```

2. **检查XML配置**：
```python
class asset:
    file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.xml'
```

3. **验证路径解析**：
```python
# 在机器人配置中
import os
xml_path = os.path.join(
    os.path.dirname(__file__),
    '../../../resources/robots/go2/urdf/go2.xml'
)
assert os.path.exists(xml_path), f"XML not found: {xml_path}"
```

4. **设置SIMULATOR环境变量**：
```bash
export SIMULATOR=genesis
python -m legged_gym.scripts.train --task go2
```

### 地形配置冲突

**问题**：地形生成失败并出现冲突选项。

**错误信息**：
```
ValueError: Curriculum and selected terrain cannot be both True.
```

**根本原因**：不能同时使用课程地形和选定地形。

**解决方案**：

选择一种地形模式：

```python
# 选项1：课程地形（难度随训练增加）
class terrain:
    curriculum = True
    selected = False
    terrain_curriculum_difficulty = 0.5

# 选项2：选定地形（固定地形类型）
class terrain:
    curriculum = False
    selected = True
    terrain_proportions = [0.2, 0.3, 0.5]  # 每种地形类型的比例
```

**代码参考**（参见`terrain.py:63`）：
```python
if cfg.curriculum and cfg.selected:
    raise ValueError("Curriculum and selected terrain cannot be both True.")
```

### 高程场地形限制

**问题**：高程场地形在IsaacLab中不起作用。

**错误信息**：
```
NotImplementedError: Heightfield terrain not implemented for IsaacLabSimulator
RuntimeError: Cannot create heightfield in IsaacLab
```

**根本原因**：未为IsaacLab后端实现高程场地形生成。

**解决方案**：

改用三角网格地形：

```python
class terrain:
    mesh_type = 'trimesh'  # 适用于IsaacLab
    # mesh_type = 'heightfield'  # IsaacLab不支持
```

**仿真器兼容性矩阵**：

| 地形类型 | IsaacGym | Genesis | IsaacLab |
|---------|----------|---------|----------|
| plane       | ✓        | ✓       | ✓        |
| heightfield | ✓        | ✓       | ✗        |
| trimesh     | ✓        | ✓       | ✓        |

## 调试命令快速参考

### 检查环境

```bash
# 验证CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# 检查仿真器
echo $SIMULATOR

# 列出可用任务
python tests/test_all_tasks.py --list

# 检查conda环境
conda list | grep -E "torch|genesis|isaac"
```

### 调试训练

```bash
# 使用调试输出运行
python -m legged_gym.scripts.train --task go2 --debug

# 监控GPU
watch -n 0.5 nvidia-smi

# 检查日志
tail -f logs/go2/<experiment>/log.txt

# TensorBoard
tensorboard --logdir logs/
```

### 验证安装

```bash
# 测试特定任务
python tests/test_all_tasks.py --tasks go2 --iterations 1

# 测试所有任务
python tests/test_all_tasks.py

# 验证仿真器后端
python -c "from legged_gym.simulator import get_simulator; print(get_simulator.__name__)"
```

## 获取帮助

如果您无法使用本指南解决问题：

1. **查看文档**：完整文档位于https://leggedgym-ex-doc.readthedocs.io/

2. **搜索问题**：查看GitHub上的现有问题以了解类似问题

3. **调试信息**：请求帮助时，请提供：
   - 仿真器类型（`echo $SIMULATOR`）
   - Python版本（`python --version`）
   - PyTorch版本（`python -c "import torch; print(torch.__version__)"`）
   - CUDA版本（`nvcc --version`或`nvidia-smi`）
   - 完整的错误消息和堆栈跟踪
   - 最小复现脚本

4. **社区**：加入飞书群组（参见README）进行讨论

5. **错误报告**：提交GitHub issue并包含：
   - 清晰描述
   - 复现步骤
   - 预期与实际行为
   - 系统信息
