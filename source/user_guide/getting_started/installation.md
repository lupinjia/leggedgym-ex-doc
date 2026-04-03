# 🛠️ 安装

## 前提条件

下表显示了运行此框架推荐的（已测试的）计算机配置要求。

| 组件 | 推荐（已测试）配置 |
|-----------|-------------|
|    CPU    |Intel Core i9|
|    GPU    |RTX 3080 10GB|
|     操作系统    | Ubuntu 22.04|
|   Python  |     >=3.8   |
|Nvidia 驱动|   >=570   |

LeggedGym-Ex 将两个仿真器整合到一个框架中。您可以选择使用任意一个仿真器，由于 Python 版本的限制，每个仿真器需要单独的 Conda 环境。以下是两个仿真器推荐（已测试）的环境设置：

| 组件 |  IsaacGym   |   Genesis   | IsaacSim |
|-----------|-------------|-------------|----------|
|  Python   |    3.8      |    >=3.10   |  >=3.10  |
|  Nvidia 驱动 |   570  |     570     |   570    |
|  PyTorch  | 2.4.1+cu121 | 2.8.0+cu126 | 2.7.0+cu128|

## 直接安装

### IsaacGym

```bash
# 1. 创建 Python 3.8 的 Conda 环境
conda create -n lr_gym python=3.8
conda activate lr_gym
# 2. 安装 PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# 3. 在 /home/username 下下载 IsaacGym Preview4
cd /home/username
wget https://developer.nvidia.com/isaac-gym-preview-4 \
    && tar -xf isaac-gym-preview-4 \
    && rm isaac-gym-preview-4
# 将 np.float 替换为 np.float32 以解决兼容性问题
find isaacgym/python -type f -name "*.py" -exec sed -i 's/np\.float/np.float32/g' {} +
# 在此环境中安装 isaacgym
cd isaacgym/python && pip install -e . && cd ../..
# 4. 安装带有 isaacgym 的 LeggedGym-Ex
git clone https://github.com/lupinjia/LeggedGym-Ex.git
cd LeggedGym-Ex && pip install -e ".[isaacgym]"
# 5. 测试安装
python legged_gym/scripts/train.py --task=go2 --num_envs=100
```
如果出现如下窗口，则安装成功。

```{figure} ../../_static/images/isaacgym_installation_success.png
```

### Genesis

```bash
# 1. 创建 Python 3.10 的 Conda 环境
conda create -n lr_gen python=3.10
conda activate lr_gen
# 2. 安装 PyTorch
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
# 3. 安装带有 genesis 的 LeggedGym-Ex
git clone https://github.com/lupinjia/LeggedGym-Ex.git
cd LeggedGym-Ex && pip install -e ".[genesis]"
# 4. 测试安装
export SIMULATOR=genesis
python legged_gym/scripts/train.py --task=go2 --num_envs=100
```
如果出现如下窗口，则安装成功。

```{figure} ../../_static/images/genesis_installation_success.png
```

### IsaacSim

```bash
# 1. 创建 Python 3.11 的 Conda 环境
conda create -n lr_lab python=3.11
conda activate lr_lab
# 2. 安装 IsaacSim
pip install --upgrade "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
# 3. 验证 IsaacSim 安装
isaacsim # 如果出现窗口，则表示安装成功
# 4. 克隆并安装 isaaclab
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab && git checkout v2.3.2
./isaaclab.sh --install none # 不安装 RL 库，以使用我们自定义的 RL 库
# 5. 验证 IsaacLab 安装
python scripts/tutorials/00_sim/create_empty.py # 如果出现带有黑色场景的窗口，则表示安装成功
# 6. 安装带有 IsaacLab 的 LeggedGym-Ex
git clone https://github.com/lupinjia/LeggedGym-Ex.git
cd LeggedGym-Ex && pip install -e ".[isaaclab]"
# 7. 测试安装
export SIMULATOR=isaaclab
python legged_gym/scripts/train.py --task=go2 --num_envs=100
```
如果出现如下窗口，则安装成功。

```{figure} ../../_static/images/isaacsim_installation_success.png
```


最后，您需要注册一个 wandb 账号并设置环境变量：
```bash
export WANDB_API_KEY=<your_api_key>
```

## 已知问题

### IsaacGym窗口不显示渲染的物理世界

当你的电脑中Nvidia显卡驱动版本>=570时你可能会遇到这个问题. 在命令行中使用`export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`可以解决这个问题.

IsaacGym使用Vulkan来渲染图形. 在一个较新的驱动下, 系统可能找不到正确的Vulkan ICD (Installable Client Driver)配置文件位置.

### ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory

将当前conda环境的库路径添加到`LD_LIBRARY_PATH`. 例如, `export LD_LIBRARY_PATH=/home/user_name/miniconda3/envs/lr_gym/lib`

## 可选安装

### Unitree Go2 仿真到仿真（Sim2Sim）

将策略部署到另一个仿真器可以有效测试策略的鲁棒性。此外，用于 Sim2Sim 的代码通常可以直接部署到真实机器人上。为避免在真实机器人上出现潜在的崩溃，最好先在仿真中测试部署代码。

由于部署代码通常用 C++ 编写，因此支持 C++ 接口的仿真器是理想的选择。我们在 mujoco 中提供了一个基于 [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)、[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) 和 [LibTorch](https://pytorch.org/) 的 Sim2Sim 框架。

对于仿真器，您可以按照 README.md 中的说明安装 [unitree_mujoco](https://github.com/lupinjia/unitree_mujoco)。

对于部署代码，您可以参考 [go2_deploy](https://github.com/lupinjia/go2_deploy) 和 [go2_deploy_python](https://github.com/lupinjia/go2_deploy_python)。

以下是 unitree_mujoco 的界面。我们通过 DDS 实现了深度图像的访问和发布。

```{figure} ../../_static/images/unitree_mujoco_demo.gif
```

### TRON1_PF Sim2Sim

对于 TRON1_PF 的 Sim2Sim，您可以安装我们提供的 [tron1-mujoco-sim](https://github.com/limxdynamics/tron1-mujoco-sim) 和 [tron1-rl-deploy-python](https://github.com/lupinjia/tron1-rl-deploy-python)。

```{figure} ../../_static/images/tron1_pf_ee_demo.gif
```
