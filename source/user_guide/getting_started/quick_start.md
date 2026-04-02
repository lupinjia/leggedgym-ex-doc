# 🚀 快速开始

默认情况下，我们通过识别 Python 版本和环境变量来切换不同的仿真器：

```python
# 在 legged_gym/__init__.py 中
if sys.version_info[1] >= 10: # >=3.10 用于 genesis 和 isaacsim
    simulator_type = os.getenv("SIMULATOR")
    if simulator_type == "genesis":
        SIMULATOR = "genesis"
    elif simulator_type == "isaaclab":
        SIMULATOR = "isaaclab"
    else:
        raise ValueError("不支持的 SIMULATOR 类型。请将 SIMULATOR 环境变量设置为 'genesis' 或 'isaaclab'。")
elif sys.version_info[1] <= 8 and sys.version_info[1] >= 6: # >=3.6 且 <3.9 用于 isaacgym
    SIMULATOR = "isaacgym"
```

要使用 IsaacGym，您需要一个 Python 3.8 的虚拟环境。Python 3.10 适用于 Genesis 和 IsaacSim（IsaacLab），因此我们添加了一个额外的环境变量（`SIMULATOR`）来确定仿真器。

您可以定义自定义策略来在不同仿真器之间切换。

## 切换仿真器环境

我们提供了一个便捷的脚本 `switch_simulator.sh`，帮助您快速切换不同的仿真器环境。

### 使用方法

在终端中运行脚本：

```bash
source ./switch_simulator.sh
```

您将看到可用的仿真器环境：

```
==========================================
       仿真器切换工具
==========================================

可用的 Conda 环境：
  [✓] lr_gym (IsaacGym)
  [✓] lr_gen (Genesis)
  [✓] lr_lab (IsaacLab)

可用选项：isaacgym、genesis、isaaclab
请输入要切换的仿真器名称（或输入 'exit' 退出）：
```

输入以下选项之一：
- `isaacgym` - 切换到 IsaacGym 环境（Python 3.8）
- `genesis` - 切换到 Genesis 环境（Python 3.10）
- `isaaclab` - 切换到 IsaacLab 环境（Python 3.11）

### 脚本功能

脚本会自动执行以下操作：
1. 检测可用的 Conda 环境
2. 激活对应的环境
3. 设置必要的环境变量：
   - **IsaacGym**：导出 `LD_LIBRARY_PATH` 用于 IsaacGym 库
   - **Genesis**：导出 `SIMULATOR=genesis`
   - **IsaacLab**：导出 `SIMULATOR=isaaclab`

### 手动切换环境

如果您希望手动切换环境，请使用以下命令：

**IsaacGym：**
```bash
conda activate lr_gym
export LD_LIBRARY_PATH=/home/username/miniconda3/envs/lr_gym/lib:$LD_LIBRARY_PATH
```

**Genesis：**
```bash
conda activate lr_gen
export SIMULATOR=genesis
```

**IsaacLab：**
```bash
conda activate lr_lab
export SIMULATOR=isaaclab
```

## 在平面上训练 Go2 策略

在 `legged_gym/envs` 目录下，我们可以看到多个针对不同机器人的文件夹。要为 Go2 机器人在平面上训练一个运动策略，我们可以参考 `legged_gym/envs/go2/go2.py` 中的 `go2` 环境，它继承自 `legged_gym/envs/base/legged_robot.py` 中的 `LeggedRobot` 类。

运行以下命令，您将在终端中看到日志信息不断输出，显示奖励值和一些训练数据：

```bash
cd legged_gym/scripts
python train.py --task=go2 --headless
```

```{figure} ../../_static/images/log_info_in_terminal.png
```

## 在 Genesis 中运行训练好的 Go2 策略

训练结束后（1000 次迭代），您将看到一个以训练开始日期和时间命名的新文件夹（格式为 `date_time_`，例如 `Sep03_16-30-16_`）。该文件夹包含本次训练的检查点（checkpoints），位于 `logs/experiment_name/` 下，其中 `experiment_name` 在 `go2_config.py` 中指定。您可以运行以下命令，将看到仿真器场景中机器人在平面上行走：

```bash
python play.py --task=go2 --load_run=train_session_name
```

```{figure} ../../_static/images/play_in_genesis.png
```
:::{note}
如果您使用 IsaacGym 或 IsaacSim 仿真器，流程相同，只是仿真器窗口不同。
:::

关于命令行参数的更多信息，请使用 `python play.py --help`：
```{figure} ../../_static/images/cli_params.png
```
