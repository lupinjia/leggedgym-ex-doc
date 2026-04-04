# 📋 Task List

Here we list the tasks supported by genesis_lr. 

- To start the training: `python legged_gym/scripts/train.py --task=task_name`.
- To visualize the trained model: `python legged_gym/scripts/play.py --task=task_name` 
- More command line parameters can be seen in `legged_gym/utils/helpers.py/get_args()`

| Robot | Task Name | Description | Paper |
| ----- | ---- | ----------- | -----------|
| Unitree Go2 | go2 | A simple example to train a go2 policy walking on plane | |
|       | go2_wtw | Implementation of Walk These Ways on go2, supporting control of base height, base pitch angle, foot clearance, gait period and gait type | [Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://arxiv.org/abs/2212.03238) |
|       | go2_ts  | Implementation of Teacher-Student on go2, for walking on tough terrains | [Rapid Locomotion via Reinforcement Learning](https://agility.csail.mit.edu/) |
|       | go2_ee  | Implementation of Explciit Estimator on go2, for walking on tough terrains | [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481) |
|       | go2_dreamwaq | Implementation of DreamWaQ on go2, for walking on tough terrains | [DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602) |
|       | go2_cat | Constraints as Terminations on go2, for walking on tough terrains | [CaT: Constraints as Terminations for Legged Locomotion Reinforcement Learning](https://constraints-as-terminations.github.io/) |
|       | go2_nav | End-to-end local navigation on go2 | [Advanced Skills by Learning Locomotion and Local Navigation End-to-End](https://arxiv.org/abs/2209.12827) |
|       | go2_cts | Implementation of Concurrent Teacher Student framework | [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://clearlab-sustech.github.io/concurrentTS/) |
|       | go2_ts_depth | Depth estimation variant for go2 (ts_depth) | |
| Unitree G1 | g1 | A simple example to train a g1 policy walking on plane (only 12 dof, upper body fixed) | |
|            | g1_deepmimic | Implementation of DeepMimic on Unitree G1 (29 dof) | [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/index.html) |
|            | g1_motion_vis | G1 motion visualization | |
| Limx TRON1PF | tron1pf | A simple example to train a tron1pf policy walking on plane | |
|       | tron1pf_ee | Implementation of Explciit Estimator on tron1pf, for walking on tough terrains |  |
| Limx TRON1SF | tron1sf | A simple example to train a tron1sf policy walking on plane | |
| Booster K1 | k1 | A simple example to train a k1 policy walking on plane | |
|  | k1_amp | AMP on Booster K1 | [AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control](https://arxiv.org/abs/2104.02180) |
|  | k1_cts_amp | CTS AMP on Booster K1 | |
|  | k1_motion_vis | K1 Motion Visualization | |
|  | k1_deepmimic | DeepMimic on Booster K1 (29 dof) | [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/index.html) |
