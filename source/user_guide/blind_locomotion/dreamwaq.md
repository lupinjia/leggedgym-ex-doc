# DreamWaQ: Implicit Terrain Imagination

Training blind locomotion policies that can traverse complex terrain without explicit terrain sensing has been a long-standing challenge in legged robotics. While methods like Teacher-Student framework and Explicit Estimator rely on privileged terrain information during training, they require additional mechanisms to handle the sim-to-real transfer of terrain-aware behaviors. DreamWaQ takes a different approach by combining implicit terrain representation with explicit state estimation through a hybrid variational autoencoder architecture.

## DreamWaQ Framework

The key insight of DreamWaQ is that terrain information can be implicitly encoded in the observation history, and a well-designed encoder can extract both implicit terrain representations and explicit state estimates from this history. Rather than separately training a history encoder to mimic a privilege encoder (as in Teacher-Student), DreamWaQ uses a Variational Autoencoder (VAE) that jointly learns to:

1. **Estimate explicit states** such as base linear velocity, foot contact states, and foot height information
2. **Encode implicit terrain information** into a latent vector that captures terrain geometry and properties
3. **Reconstruct future states** to ensure the learned representation is predictive and meaningful

The hybrid architecture consists of two parts: an **implicit encoder** that outputs a latent vector representing terrain information, and an **explicit estimator** that outputs a vector containing directly interpretable state estimates. Both are trained jointly through supervised learning while the policy is trained through reinforcement learning.

The diagram below illustrates the DreamWaQ architecture where observation history is encoded into an implicit latent vector and explicit state estimates. The actor receives both current observation and the encoded outputs, while the critic uses privileged observations including ground-truth terrain information.

## Implementation

DreamWaQ is implemented through three core components: the VAE module for encoding, the actor-critic architecture, and the specialized PPO algorithm. The VAE serves as the heart of the method, encoding observation history into meaningful representations.

### Variational Autoencoder (VAE)

The VAE encodes observation history into two distinct outputs: a latent vector (implicit terrain representation) and an explicit vector (state estimates). The encoder architecture uses MLP layers with reparameterization for sampling.

The VAE encoder structure consists of multiple linear layers with activation functions. The final layer outputs both latent dimensions and explicit dimensions (with mean and variance for each), enabling probabilistic sampling during training.

The VAE outputs distribution parameters for both implicit and explicit variables, enabling sampling during training and mean-value inference during deployment. The forward method encodes observation history into distribution parameters, samples from these distributions using the reparameterization trick, and returns both the samples and parameters. During inference, mean values are used for deterministic outputs.

### Actor-Critic Architecture

The actor-critic for DreamWaQ extends the standard architecture to incorporate VAE outputs. The actor receives both current observation and encoded history (implicit plus explicit), while the critic uses privileged observations with ground-truth terrain data.

The actor input dimension combines observations, explicit state estimates, and latent dimensions. The critic uses privileged observations which may include terrain height measurements, contact states, and other simulation-specific information. The VAE is initialized with history input size, latent dimensions, explicit dimensions, and decoder output size.

During action sampling, the VAE encodes history into latent variables by sampling from the learned distributions. These latent variables are concatenated with current observations and fed into the actor network to produce action distributions.

### PPO with DreamWaQ

The training algorithm combines standard PPO updates with VAE training through supervised learning. The VAE is trained with three objectives:

1. **Explicit estimation loss**: MSE between predicted and true explicit states (velocity, contacts, foot height)
2. **Reconstruction loss**: MSE between decoded next state and actual next state
3. **KL divergence loss**: Regularization to keep latent distributions close to standard normal

The explicit estimation loss uses mean squared error between predicted explicit states and ground truth labels, masked by termination flags to ignore completed episodes. The reconstruction loss similarly uses MSE between decoder outputs and next states, also masked by termination flags. The KL divergence loss regularizes the latent distribution toward a standard normal distribution to ensure meaningful latent representations.

The full VAE loss combines all three terms with a configurable weight on the KL divergence term. The total loss equals explicit estimation loss plus reconstruction loss plus the weighted KL divergence loss.

The full update procedure alternates between PPO gradient steps and VAE gradient steps. First, the PPO loss is computed and backpropagated through policy and value networks. Then, the VAE undergoes multiple epochs of gradient updates to minimize the VAE loss components.

## Configuration Parameters

DreamWaQ introduces several configuration parameters specific to the VAE architecture and training process. These are defined in the environment and algorithm configurations.

### Environment Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| frame_stack | int | 5 | Number of observation frames to stack for history encoder input |
| num_history_obs | int | 225 | Total dimension of stacked observations |
| num_latent_dims | int | 16 | Dimension of implicit terrain latent vector |
| num_explicit_dims | int | 24 | Dimension of explicit state estimates |
| num_decoder_output | int | 45 | Output dimension of decoder (next state prediction) |
| c_frame_stack | int | 5 | Frame stack size for critic observations |
| num_privileged_obs | int | computed | Total privileged observation dimension |

### Policy Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| encoder_hidden_dims | list | [256, 128] | Hidden layer dimensions for VAE encoder |
| decoder_hidden_dims | list | [256, 128] | Hidden layer dimensions for VAE decoder |
| critic_hidden_dims | list | [1024, 256, 128] | Hidden layer dimensions for critic network |

### Algorithm Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| encoder_lr | float | 2.0e-4 | Learning rate for VAE encoder/decoder |
| num_encoder_epochs | int | 1 | Number of gradient steps for VAE per PPO update |
| vae_kld_weight | float | 2.0 | Weight for KL divergence loss in VAE objective |

### Runner Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| policy_class_name | str | ActorCriticDreamWaQ | Policy class for DreamWaQ |
| algorithm_class_name | str | PPO_DreamWaQ | Algorithm class for DreamWaQ |
| runner_class_name | str | DreamWaQRunner | Runner class for DreamWaQ |

## Training Workflow

Training a DreamWaQ policy follows the standard legged_gym workflow with task-specific configuration.

### Basic Training

To train a DreamWaQ policy on the Go2 robot, use the following command:

```bash
python -m legged_gym.scripts.train --task=go2_dreamwaq --headless
```

This will start training with the default configuration defined in go2_dreamwaq_config.py. The training process will log progress to TensorBoard and save checkpoints at regular intervals.

### Configuration Customization

You can customize the training by modifying the configuration file or passing command-line arguments. Key parameters to tune include:

- num_latent_dims: Controls the capacity of implicit terrain representation
- num_explicit_dims: Determines what explicit states are estimated
- vae_kld_weight: Balances reconstruction quality versus latent regularization
- encoder_lr: Learning rate for the VAE (often needs to be lower than policy LR)

### Monitoring Training

During training, the following metrics are logged:

- Total reward and individual reward components
- Value loss and surrogate loss (PPO metrics)
- Explicit estimation loss (VAE explicit prediction accuracy)
- Reconstruction loss (VAE next state prediction accuracy)
- KL divergence loss (VAE latent regularization)

These metrics help diagnose training issues. High KL divergence may indicate the latent space is not well-regularized. High reconstruction loss suggests the VAE struggles to predict future states. High explicit estimation loss means explicit state predictions are inaccurate.

## Differences from Standard PPO

DreamWaQ extends standard PPO with several key differences that enable implicit terrain imagination:

### Architecture Differences

1. **VAE Module**: DreamWaQ adds a Variational Autoencoder that encodes observation history into latent representations. This is the core innovation that enables terrain imagination without explicit terrain sensing.

2. **Dual Output Encoder**: Unlike standard encoders that output a single representation, the DreamWaQ VAE outputs both implicit (latent) and explicit (interpretable) representations simultaneously.

3. **Expanded Actor Input**: The actor network receives not just current observations but also the encoded history (both implicit and explicit components), enabling context-aware action selection.

### Training Differences

1. **Supervised Learning Component**: DreamWaQ adds supervised learning objectives for the VAE alongside the standard RL objective. The VAE learns to predict explicit states and reconstruct future states through supervised losses.

2. **Multiple Optimizers**: Standard PPO uses a single optimizer for all parameters. DreamWaQ uses separate optimizers for RL parameters (policy and value) and VAE parameters, allowing different learning rates and update frequencies.

3. **Additional Loss Terms**: The training objective includes three VAE-specific losses (explicit estimation, reconstruction, KL divergence) in addition to the standard PPO losses (surrogate, value, entropy).

### Observation Differences

1. **History Buffer**: DreamWaQ maintains a buffer of past observations (obs_history) which is fed into the VAE. This temporal information is crucial for inferring terrain properties.

2. **Explicit Labels**: The environment computes explicit label targets (base velocity, contact states, foot heights) that supervise the explicit estimator during training.

3. **Next State Prediction**: The VAE decoder learns to predict the next observation state, requiring the environment to provide next_state_buf for training.

## Inference and Deployment

Once trained, the DreamWaQ policy can be deployed for inference on both simulation and real robots.

### Running Trained Policy

To run a trained policy, use the play script:

```bash
python -m legged_gym.scripts.play --task=go2_dreamwaq --load_run=session_name
```

This will load the trained checkpoint and run the policy in the simulation environment.

### Exporting for Real Robot

DreamWaQ policies can be exported for real robot deployment. During inference, the VAE uses deterministic mean values rather than sampling, providing consistent behavior. The observation history buffer must be maintained on the real robot to feed into the VAE encoder.

To export a trained policy:

```bash
python -m legged_gym.scripts.play --task=go2_dreamwaq --export_policy
```

The exported policy will be saved to the logs directory and can be loaded onto the real robot for deployment.

## References

1. DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning
2. Learning Quadrupedal Locomotion over Challenging Terrain (Teacher-Student framework)
3. Rapid Motor Adaptation for Legged Robots (RMA)
