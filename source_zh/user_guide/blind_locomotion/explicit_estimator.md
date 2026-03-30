# ⏱️ 显式估计器 (Explicit Estimator)

在 [师生框架 (teacher_student framework)](teacher_student.md) 中，教师编码器输出的潜在向量是特权信息 $x_t$ 的编码信息。该潜在向量是特权信息的一种表示，没有明确的物理意义。然而，根据我们在 [部署到真实机器人 (Deploy to Real Robot)](https://genesis-lr.readthedocs.io/en/latest/user_guide/getting_started/deploy_to_real_robot.html) 中的经验，`base_lin_vel`（基座线速度）的存在可以显著提高智能体的性能。那么问题是如何在真实机器人上获得 `base_lin_vel`。

从基于模型的控制的角度来看，卡尔曼滤波器 (Kalman Filter) 可以帮助我们利用机器人状态的反馈来估计 `base_lin_vel`。然而，这类方法通常依赖于某些假设，限制了它们的通用性。借助神经网络的强大能力，我们如何处理这个问题？[EstimatorNet](https://arxiv.org/abs/2202.05481) 被提出来解决这个问题。

## 框架分析

基本上，EstimatorNet 的图示形式与 [师生框架 (teacher-student framework)](teacher_student.md) 相似。最大的区别是 **EstimatorNet 中的编码器输出被训练来逼近显式的物理值，例如基座线速度 (base linear velocity)、足部接触概率 (foot contact probability)、足部高度 (foot height) 等**。EstimatorNet 的图示如下所示，其中 $e_t$ 是显式向量的真实值，$o_t^H=[o_t, o_{t-1},...,o_{t-H+1}]$ 是长度为 H 步的观测历史，$\hat{e}_t$ 是估计器网络预测的显式向量，$a_t$ 是动作，$s_t$ 是环境状态，$\hat{V}_t$ 是 Critic 网络估计的值。

```{figure} ../../_static/images/estimatornet_diagram.png
```

## 实现

该方法的实现与 [师生框架 (teacher_student framework)](teacher_student.md) 类似。读者可以查看后缀为 `ee` 的文件来找到实现。

要训练一个显式估计器策略，输入以下命令：
```bash
python train.py --task=go2_ee --headless
```

要运行它，输入以下命令：
```bash
python play.py --task=go2_ee --load_run=session_name
```

## 演示

我们在 [go2_deploy_python](https://github.com/lupinjia/go2_deploy_python) 和 [tron1_rl_deploy_python](https://github.com/lupinjia/tron1-rl-deploy-python) 中提供了显式估计器策略的参考部署代码。

Unitree Go2：

```{figure} ../../_static/images/ee_demo.gif
```

TRON1_PF：

```{figure} ../../_static/images/tron1_pf_ee_demo.gif
```


## 参考文献
1. [Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481)
