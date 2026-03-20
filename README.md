# Formative 3 - Deep Q Learning (Atari)

Professional project documentation for the ALU Formative 3 assignment using Stable Baselines3 + Gymnasium.

## Project Summary

This project trains and evaluates a Deep Q-Network (DQN) agent on Atari environments.
It contains:

- `train.py`: trains a DQN policy, logs training metrics, saves checkpoints, and evaluates greedy policy performance.
- `play.py`: loads a trained model and runs gameplay episodes with rendering for demonstration.
- `train.ipynb`: notebook workflow version for iterative experimentation.
- `logs_archive/`: archived logs from multiple experiments (`exp1` to `exp10`).
- `models/`: archived trained models from experiments.

## Assignment Requirements Coverage

This repository is structured to satisfy the assignment tasks:

1. Define and train a DQN agent with Stable Baselines3.
2. Compare policy architectures (`MlpPolicy` vs `CnnPolicy`).
3. Tune hyperparameters across 10 experiments.
4. Log reward trends and episode lengths.
5. Save and replay the best model with greedy policy in `play.py`.
6. Present results clearly (table + gameplay demo).

## Tech Stack

- Python 3.10+
- Stable Baselines3 (DQN)
- Gymnasium Atari environments
- ALE-Py + AutoROM
- TensorBoard (for training metrics)

## Environment and Dependencies

The scripts auto-install required packages at runtime. This is convenient for Colab, but local users can also pre-install dependencies:

```powershell
python -m pip install "numpy<2.0" "stable-baselines3[extra]>=2.3.0" "gymnasium[atari,accept-rom-license]>=0.29.0" "ale-py>=0.9.0" "shimmy[atari]>=0.2.1" "autorom[accept-rom-license]>=0.6.1" "tensorboard>=2.14.0"
AutoROM --accept-license -q
```

## How to Run

### 1) Train the agent

```powershell
python train.py
```

What `train.py` currently does:

- Uses `ENV_ID = "ALE/Pong-v5"`
- Uses `POLICY = "CnnPolicy"`
- Trains for `500,000` timesteps
- Saves model as `./dqn_model_exp1.zip` (based on `EXPERIMENT_NAME`)
- Logs monitor/eval/checkpoint data under `./logs/<experiment_name>/`
- Runs post-training greedy evaluation over 10 episodes and prints mean reward

### 2) Play with a trained model

```powershell
python play.py
```

What `play.py` currently expects:

- `MODEL_PATH = "./dqn_model.zip"`
- `ENV_ID = "ALE/Breakout-v5"`
- Renders gameplay for `N_EPISODES = 5`

Important: for successful playback, set `MODEL_PATH` and `ENV_ID` to match the model trained in `train.py`.

Example alignment after training `exp1` on Pong:

- `MODEL_PATH = "./dqn_model_exp1.zip"`
- `ENV_ID = "ALE/Pong-v5"`

## Hyperparameter Tuning Workflow (10 Experiments)

The assignment requires 10 different hyperparameter combinations per member. In `train.py`, vary these values for each run:

- `LEARNING_RATE`
- `GAMMA`
- `BATCH_SIZE`
- `EXPLORATION_INITIAL_EPS`
- `EXPLORATION_FINAL_EPS`
- `EXPLORATION_FRACTION`

Recommended process:

1. Set `EXPERIMENT_NAME` (for example, `exp1`, `exp2`, ... `exp10`).
2. Adjust one or more hyperparameters.
3. Run `python train.py`.
4. Record:
   - mean reward
   - reward trend shape
   - episode length behavior
   - stability/convergence observations
5. Repeat for all 10 experiment configurations.

## Policy Comparison (MLP vs CNN)

To compare policies as requested:

- Keep the same environment and core hyperparameters.
- Run once with `POLICY = "MlpPolicy"`.
- Run once with `POLICY = "CnnPolicy"`.
- Compare final mean reward, learning stability, and training speed.

Note: Atari image observations generally favor `CnnPolicy`.

## Suggested Results Table Template

Use this in your report/README submission notes:

| Experiment | Policy | lr | gamma | batch_size | eps_start | eps_end | eps_fraction | Mean Reward | Noted Behavior |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| exp1 | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | ... | ... |
| exp2 | CnnPolicy | ... | ... | ... | ... | ... | ... | ... | ... |
| exp3 | MlpPolicy | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| exp10 | CnnPolicy | ... | ... | ... | ... | ... | ... | ... | ... |

## Logs, Models, and Artifacts

Current workspace already contains:

- `logs_archive/exp1` to `logs_archive/exp10`
- `models/dqn_model_exp1` to `models/dqn_model_exp10`

These archived artifacts are useful for:

- performance comparison across experiments
- selecting best model for final gameplay demo
- backing up evidence for presentation and grading rubric

## Presentation Checklist

Before group presentation:

- Confirm 10 hyperparameter experiments are documented.
- Highlight which changes improved and harmed performance.
- Explain best final configuration and why.
- Run `play.py` using the best model in the same environment.
- Prepare a short gameplay clip or live demo.

## Troubleshooting

- ROM/license issues: rerun `AutoROM --accept-license -q`.
- Mismatch errors in `play.py`: verify `MODEL_PATH` and `ENV_ID` match training settings.
- Slow local training: reduce `TOTAL_TIMESTEPS` for quick tests, then restore for final runs.
- Rendering issues on remote servers: use local machine or disable GUI and record metrics only.

## Authoring Notes

- Assignment: Formative 3 - Deep Q Learning
- Focus: decision quality, experiment clarity, and reproducible evaluation

---

If you want, this README can be extended with:

- exact best-experiment metrics from your archived logs
- embedded reward plots
- a short final "Best Configuration" executive summary section
