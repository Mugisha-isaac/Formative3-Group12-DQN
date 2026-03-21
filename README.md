# Formative 3: Deep Q Learning for Atari

This repository contains the implementation, training workflow, and evaluation artifacts for the ALU Formative 3 assignment on Deep Q-Networks (DQN).

## Objective

Build and evaluate a DQN agent using Stable Baselines3 and Gymnasium Atari environments, then analyze the impact of hyperparameter tuning and policy architecture choices.

## Repository Structure

- `train.py` - Trains the DQN agent, saves checkpoints and final model, and performs post-training evaluation.
- `play.py` - Loads a trained model and runs rendered gameplay episodes using greedy action selection.
- `train.ipynb` - Notebook-based workflow that runs 10 hyperparameter experiments, compares CnnPolicy vs MlpPolicy, and generates a comprehensive results summary table.
- `play.ipynb` - Interactive notebook for downloading and testing the trained model, displaying episode-by-episode gameplay results and performance statistics.
- `logs_archive/` - Archived logs from experiment runs.
- `models/` - Archived trained models.
- `docs/` - Documentation assets (including experiment screenshot evidence).

## Assignment Alignment

This project addresses the required tasks:

1. Train a DQN agent in an Atari environment.
2. Compare `MlpPolicy` and `CnnPolicy`.
3. Execute and document 10 hyperparameter experiments.
4. Track reward and episode-length behavior.
5. Evaluate gameplay using a trained model with greedy policy.

## Technology Stack

- Python 3.10+
- Stable Baselines3 (DQN)
- Gymnasium + Atari wrappers
- ALE-Py and AutoROM
- TensorBoard

## Setup

Install dependencies from the provided requirements file:

```powershell
python -m pip install -r requirements.txt
AutoROM --accept-license -q
```

Note: `train.py` and `play.py` currently include runtime package installation for convenience in notebook/Colab workflows.

## Run Instructions

### Train

```powershell
python train.py
```

Current default training configuration in `train.py`:

- Environment: `ALE/Pong-v5`
- Policy: `CnnPolicy`
- Timesteps: `500000`
- Model output: `./dqn_model_exp1.zip` (derived from `EXPERIMENT_NAME`)
- Logs: `./logs/<experiment_name>/`

### Play

```powershell
python play.py
```

Ensure playback settings match training settings:

- `MODEL_PATH` must point to the model produced by training.
- `ENV_ID` in `play.py` must match the environment used in `train.py`.

## Hyperparameter Tuning Protocol

For each experiment, vary one or more of the following parameters in `train.py`:

- `LEARNING_RATE`
- `GAMMA`
- `BATCH_SIZE`
- `EXPLORATION_INITIAL_EPS`
- `EXPLORATION_FINAL_EPS`
- `EXPLORATION_FRACTION`

Recommended process:

1. Set a unique `EXPERIMENT_NAME`.
2. Update hyperparameters.
3. Run training.
4. Record mean reward, episode behavior, and stability observations.
5. Repeat until 10 configurations are completed.

## Policy Comparison Guidance

To compare policy architectures fairly:

1. Keep environment and core hyperparameters constant.
2. Run one experiment with `MlpPolicy`.
3. Run one experiment with `CnnPolicy`.
4. Compare reward level, convergence behavior, and training stability.

## Results Template

| Experiment | Policy | lr | gamma | batch_size | eps_start | eps_end | eps_fraction | Mean Reward | Noted Behavior |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| exp1 | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | ... | ... |
| exp2 | CnnPolicy | ... | ... | ... | ... | ... | ... | ... | ... |
| exp3 | MlpPolicy | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| exp10 | CnnPolicy | ... | ... | ... | ... | ... | ... | ... | ... |

## Notebook Workflows

### Training Notebook (`train.ipynb`)

The training notebook executes the complete experimentation pipeline:

- **10 Hyperparameter Experiments**: Runs all configured experiment variations sequentially
  - Experiments vary learning rate, gamma, batch size, and exploration parameters
  - Each experiment includes model training and post-training evaluation (3 episodes)
  
- **Policy Comparison Analysis**: Compares CnnPolicy vs MlpPolicy performance
  - Both tested with identical baseline hyperparameters
  - Mean reward calculated for each policy architecture
  
- **Results Summary Table**: Displays comprehensive results with:
  - Experiment name, hyperparameter values, mean reward, and standard deviation
  - Behavioral analysis for each configuration (e.g., "best performing config", "unstable learning")
  - Automatic identification of best-performing model
  
- **Output**: Best model saved as `dqn_model.zip` for deployment

### Play Notebook (`play.ipynb`)

The play notebook demonstrates trained model evaluation:

- **Model Download**: Automatically retrieves trained model (`dqn_model_exp10.zip`) from GitHub
  
- **Gameplay Episodes**: Runs 5 test episodes with greedy action selection (deterministic policy)
  - Each episode displays reward and episode length
  - Real-time rendering of agent gameplay
  
- **Performance Metrics**: Aggregated statistics:
  - Mean Reward: Average across all episodes
  - Best Episode: Highest single-episode reward
  - Mean Length: Average episode duration in steps

## Isaac Mugisha Experiment Evidence

The assignment experiment screenshots are stored in `docs/` and referenced below:

### Hyperparameter Experiments Results

Screenshot showing the results summary table from `train.ipynb` with all 10 hyperparameter experiments and their performance metrics.

![Isaac Mugisha Experiment Results](docs/isaac-mugisha-experiments.png)

### Policy Comparison Results

Screenshot showing the policy comparison analysis (CnnPolicy vs MlpPolicy) from `train.ipynb` with mean reward outcomes for each architecture.

![Isaac Mugisha Policy Comparison](docs/isaac-mugisha-experiments-policy-comparison.png)

### Play Script Outcome

Screenshot from `play.ipynb` displaying the gameplay test results, showing episode-by-episode rewards, lengths, and aggregated performance statistics (Mean Reward, Best Episode, Mean Length).

![Play Script Outcome](docs/play-outcome.png)

## Troubleshooting

- ROM/license setup issues: run `AutoROM --accept-license -q`.
- Model/environment mismatch in gameplay: verify `MODEL_PATH` and `ENV_ID` consistency.
- Slow training on local hardware: reduce `TOTAL_TIMESTEPS` for quick validation runs.
- Remote rendering limitations: run gameplay on a local machine with display support.

## Submission Readiness Checklist

- 10 hyperparameter configurations completed and documented.
- Policy comparison completed (`MlpPolicy` vs `CnnPolicy`).
- Best-performing configuration identified and justified.
- Gameplay demonstration prepared using best model.
