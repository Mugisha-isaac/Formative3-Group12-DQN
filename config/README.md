# Configuration Files

This directory contains configuration files for the DQN Atari project.

## Files

- **config.example.yaml** - Example configuration file with all available options
  - Copy this file to `config.yaml` to create a custom configuration
  - Modify values according to your training needs
  - The YAML format allows for easy version control and reproducibility

## Usage

1. Copy the example configuration:
   ```bash
   cp config/config.example.yaml config/config.yaml
   ```

2. Edit the configuration file with your desired parameters:
   ```yaml
   training:
     total_timesteps: 1000000
     learning_rate: 0.00005
   ```

3. Load and use the configuration in your training script:
   ```python
   import yaml
   
   with open('config/config.yaml') as f:
       config = yaml.safe_load(f)
   ```

## Configuration Parameters

### Environment
- `id`: Gymnasium environment ID (e.g., "ALE/Breakout-v5")
- `render_mode`: Visualization mode ("human" or null)
- `seed`: Random seed for reproducibility

### Training
- `total_timesteps`: Total training timesteps
- `learning_rate`: Learning rate for the optimizer
- `batch_size`: Batch size for training
- `gamma`: Discount factor
- `exploration`: Epsilon-greedy exploration settings
- `network`: Neural network architecture configuration
- `replay_buffer_size`: Size of the experience replay buffer
- `learning_starts`: Timesteps before training begins
- `train_freq`: How often to update the network
- `target_update_interval`: How often to update target network
- `max_grad_norm`: Maximum gradient norm for clipping
- `gradient_steps`: Number of gradient steps per update

### Logging
- `experiment_name`: Name of the experiment for logging
- `log_dir`: Directory for saving logs
- `save_dir`: Directory for saving models
- `save_freq`: Frequency of model checkpointing
- `tensorboard_log`: TensorBoard log directory

### Hardware
- `device`: Computing device ("auto", "cpu", "cuda")
- `n_envs`: Number of parallel environments
- `n_steps`: Number of timesteps per environment

### Evaluation
- `eval_freq`: Evaluation frequency
- `n_eval_episodes`: Number of evaluation episodes
- `render_eval`: Whether to render evaluation episodes
