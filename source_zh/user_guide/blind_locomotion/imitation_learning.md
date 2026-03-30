# 模仿学习 (Imitation Learning)

## DeepMimic

我们已经为 Unitree G1 机器人实现了 [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html)。要使用它，你可以按照以下步骤操作

### 1. 准备重定向数据 (Retargetted Data)

我们已经验证了来自 [GMR](https://github.com/YanjieZe/GMR) 的重定向数据，你可以按照其说明为 g1_29dof 生成重定向的参考动作。

然后你应该将重定向的数据粘贴到 `LeggedGymEx/resources/reference_motion/`。

### 2. 处理重定向数据

要在我们的框架中使用重定向数据，你应该使用 `legged_gym/scripts/process_reference_motion.py` 处理它：

**处理单个动作文件：**
```python
python legged_gym/scripts/process_reference_motion.py --task=g1_motion_vis --motion_file=name_of_your_refenrece_motion.pkl

# for example
python legged_gym/scripts/process_reference_motion.py --task=g1_motion_vis --motion_file=unitree_g1/02_01_walk_stageii_60hz.pkl
```

**处理目录中的所有动作文件：**
```python
python legged_gym/scripts/process_reference_motion.py --task=g1_motion_vis --motion_file=path/to/motion_directory --motion_out_dir=output_directory

# for example
python legged_gym/scripts/process_reference_motion.py --task=g1_motion_vis --motion_file=unitree_g1 --motion_out_dir=processed/unitree_g1
```

处理时，程序将在仿真器中可视化动作。默认情况下，我们使用 60Hz 的参考动作，策略的控制频率为 50Hz。

处理后的动作将保存为 .pkl 文件在 `LeggedGymEx/resources/reference_motion/` 下。文件名中的仿真器名称表示生成它的仿真器。如果你指定了 `--motion_out_dir`，处理后的文件将保存到 `LeggedGymEx/resources/reference_motion/{motion_out_dir}/`。

### 3. 训练策略

然后你可以通过执行 `python legged_gym/scripts/train.py --task=g1_motion_vis --headless --motion_file=name_of_your_processed_motion.pkl` 开始训练（例如：`python legged_gym/scripts/train.py --task=g1_motion_vis --headless --motion_file=unitree_g1/02_01_walk_stageii_60hz_isaacgym.pkl`）。

:::{note}
由于 IsaacGym/Genesis/IsaacLab 中的连杆顺序不同，请确保训练时使用与生成参考动作相同的仿真器生成的参考动作。
:::

训练结束后，你可以使用 `python legged_gym/scripts/play.py --task=g1_motion_vis --motion_file=name_of_your_processed_motion.pkl --load_run=loaded_training_session` 查看结果。以下是一些演示：

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/lupinjia/genesis_lr_doc/raw/refs/heads/main/source/_static/videos/g1_mimic_walk_isaaclab.mp4" type="video/mp4">
</video>

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/lupinjia/genesis_lr_doc/raw/refs/heads/main/source/_static/videos/g1_mimic_run_isaacgym.mp4" type="video/mp4">
</video>

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/lupinjia/genesis_lr_doc/raw/refs/heads/main/source/_static/videos/g1_mimic_dance_isaacgym.mp4" type="video/mp4">
</video>
