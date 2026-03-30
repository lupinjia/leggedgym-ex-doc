# 🚀Quick Start

By default, we switch between different simulators through identifying python version and environment variables:

```python
# In legged_gym/__init__.py
if sys.version_info[1] >= 10: # >=3.10 for genesis and isaacsim
    simulator_type = os.getenv("SIMULATOR")
    if simulator_type == "genesis":
        SIMULATOR = "genesis"
    elif simulator_type == "isaaclab":
        SIMULATOR = "isaaclab"
    else:
        raise ValueError("Unsupported SIMULATOR type. Please set the SIMULATOR environment variable to 'genesis' or 'isaaclab'.")
elif sys.version_info[1] <= 8 and sys.version_info[1] >= 6: # >=3.6 and <3.9 for isaacgym
    SIMULATOR = "isaacgym"
```

To use IsaacGym, you need a virtual environment with python3.8. Python3.10 is viable for both Genesis and IsaacSim (IsaacLab), so we add an additional environment variable (`SIMULATOR`) to determine the simulator. 

You can define customized strategies for switching between different simulators.

## Train a go2 policy on the plane

Under the directory of `legged_gym/envs`, we can see multiple folders for different robots. To train a locomotion policy for go2 robot on the flat ground, we can refer to the `go2` environment in `legged_gym/envs/go2/go2.py` which inherits the `LeggedRobot` class from `legged_gym/envs/base/legged_robot.py`.

Run the following command and you will see log information pumping in the terminal, showing reward values and some training data:

```bash
cd legged_gym/scipts
python train.py --task=go2 --headless
```

```{figure} ../../_static/images/log_info_in_terminal.png
```

## Play the trained go2 policy in Genesis

After the training is over (1000 iterations), you will see a new folder named by the date and time when the training begins (in the format of `date_time_`, such as `Sep03_16-30-16_`). This folder contains the resulted checkpoints of this training session and is located under `logs/experiment_name/`, where `experiment_name` is specified in `go2_config.py`. You can run the following command and will see a simulator scene showing the robots walking on the plane:

```bash
python play.py --task=go2 --load_run=train_session_name
```

```{figure} ../../_static/images/play_in_genesis.png
```
:::{note}
If you use IsaacGym or IsaacSim simulator, the pipeline is the same other than that the simulator window is different.
:::

For more information about the command line arguments, please use `python play.py --help`:
```{figure} ../../_static/images/cli_params.png
```