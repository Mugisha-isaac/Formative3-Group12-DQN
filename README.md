# Formative 3: Deep Q Learning for Atari

This repository contains the implementation, training workflow, and evaluation artifacts for the ALU Formative 3 assignment on Deep Q-Networks (DQN).

## Objective

Build and evaluate a DQN agent using Stable Baselines3 and Gymnasium Atari environments, then analyze the impact of hyperparameter tuning and policy architecture choices.

## Repository Structure

```
.
├── src/
│   ├── train.py              # DQN training script with checkpoints & evaluation
│   └── play.py               # Model inference & rendered gameplay
├── notebooks/
│   ├── train.ipynb           # Hyperparameter experiments (10 configs) & policy comparison
│   └── play.ipynb            # Interactive model testing & gameplay visualization
├── docs/
│   ├── experiments.png       # Training results & hyperparameter comparison table
│   ├── policy-comparison.png # CnnPolicy vs MlpPolicy performance comparison
│   ├── play-outcome.png      # Training summary & final metrics
│   ├── sid-experiments/      # Screenshots: extended runs exp11–exp20 (Sid)
│   │   ├── terminal-run-header-exp11-12.png
│   │   ├── terminal-exp14-17.png
│   │   ├── terminal-exp18-20-best.png
│   │   └── summary-tables-exp11-20.png
│   └── gameplay/
│       ├── game-1.png        # Trained agent gameplay screenshot 1
│       ├── game-2.png        # Trained agent gameplay screenshot 2
│       └── game-3.png        # Trained agent gameplay screenshot 3
├── run_my_experiments.sh     # Run only exp11–exp20 (see Extended experiments)
├── run_play.sh               # Play trained agent (CLI + optional --gui)
├── run_play_gui.sh           # Same as run_play.sh --gui (live window)
├── results/
│   ├── logs/                 # Experiment logs & TensorBoard events
│   └── models/               # Trained model checkpoints
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .venv/                    # Virtual environment
```

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

Note: Install dependencies once with `requirements.txt` before running `train.py` or `play.py`.

## Run Instructions

### Train

Run from the `src/` directory so paths resolve to `../results/` (or adjust paths accordingly):

```powershell
cd src
python train.py
```

To run **only exp11–exp20** (extended infrastructure sweep), use `./run_my_experiments.sh` from the repo root or set `RUN_EXTENDED_EXPERIMENTS_ONLY=1`.

Current defaults in `src/train.py` (verify in file if you change them):

- Environment: `ALE/Breakout-v5`
- Policy: `CnnPolicy`
- Timesteps: `TOTAL_TIMESTEPS` (default `5000`; override via environment variable, e.g. `500000` for longer runs)
- Model output: `./results/models/dqn_model_<experiment_name>.zip`; best run copied to `./results/models/dqn_model.zip`
- Logs: `./results/logs/<experiment_name>/`

### Play

```powershell
cd src
python play.py
```

Optional **live window**: `python play.py --gui` or `./run_play_gui.sh` from the repo root.

Ensure playback settings match training:

- Default model is `results/models/dqn_model.zip` (best checkpoint from training). Override with `MODEL_NAME` (filename without `.zip`).
- `ENV_ID` in `src/play.py` must match the environment used in `src/train.py`.

## Hyperparameter Tuning Protocol

For each experiment, vary one or more of the following parameters in `src/train.py`:

- `LEARNING_RATE`
- `GAMMA`
- `BATCH_SIZE`
- `EXPLORATION_INITIAL_EPS`
- `EXPLORATION_FINAL_EPS`
- `EXPLORATION_FRACTION`

Recommended process:

1. Set a unique `EXPERIMENT_NAME`.
2. Update hyperparameters in `src/train.py`.
3. Run training.
4. Record mean reward, episode behavior, and stability observations.
5. Repeat until 10 configurations are completed.

## Policy Comparison Guidance

To compare policy architectures fairly:

1. Keep environment and core hyperparameters constant.
2. Run one experiment with `MlpPolicy`.
3. Run one experiment with `CnnPolicy`.

## Results & Evidence

### Training & Hyperparameter Experiments

Comprehensive results from 10 hyperparameter configurations:

![Experiments Summary](docs/experiments.png)

### Policy Architecture Comparison

Performance comparison between CnnPolicy and MlpPolicy architectures:

![Policy Comparison](docs/policy-comparison.png)

### Training Summary

Final training metrics and model performance:

![Play Outcome](docs/play-outcome.png)

### Trained Agent Gameplay

Screenshots demonstrating the trained DQN agent playing Pong with greedy action selection:

| Game 1 | Game 2 | Game 3 |
|--------|--------|--------|
| ![Game 1](docs/gameplay/game-1.png) | ![Game 2](docs/gameplay/game-2.png) | ![Game 3](docs/gameplay/game-3.png) |

4. Compare reward level, convergence behavior, and training stability.

## Hyperparameter Tuning Deep Dive

This section captures the 10 hyperparameter experiments using the shared `train.py`, with a focus on `lr`, `gamma`, and `batch_size`.

### Experiment Table (10 Rows)

| Experiment | Policy | lr | gamma | batch_size | eps_start | eps_end | eps_fraction | Mean Reward (+/- std) | Noted Behavior |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp1 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.67 +/- 0.94 | stable training |
| exp2 | CnnPolicy | 0.0005 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.33 +/- 0.47 | stable training |
| exp3 | CnnPolicy | 0.0010 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 1.00 +/- 0.82 | high lr - unstable learning |
| exp4 | CnnPolicy | 0.0001 | 0.95 | 32 | 1.0 | 0.05 | 0.30 | 1.00 +/- 0.82 | low gamma - short-sighted |
| exp5 | CnnPolicy | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 0.30 | 0.67 +/- 0.94 | high gamma did not improve reward |
| exp6 | CnnPolicy | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 0.30 | 1.33 +/- 1.89 | best performing config |
| exp7 | CnnPolicy | 0.0001 | 0.99 | 128 | 1.0 | 0.05 | 0.30 | 1.33 +/- 1.89 | best performing config (tie) |
| exp8 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.10 | 0.50 | 0.00 +/- 0.00 | high eps_end - more exploration |
| exp9 | CnnPolicy | 0.0002 | 0.98 | 64 | 1.0 | 0.05 | 0.40 | 0.67 +/- 0.94 | stable training |
| exp10 | CnnPolicy | 0.0003 | 0.99 | 32 | 0.8 | 0.05 | 0.30 | 0.00 +/- 0.00 | low exploration start underperformed |

### Summary of Insights

- Best configuration from this run: `exp6` (`lr=1e-4`, `gamma=0.99`, `batch_size=64`) with mean reward `1.33`.
- Tie note: `exp7` reached the same mean reward (`1.33`), but `exp6` was selected as best model by first-best selection order in `train.py`.
- What helped performance:
  - Keeping `gamma` around `0.99` avoided short-sighted behavior seen at `0.95`.
  - Mid-to-large batch (`64` and `128`) produced the highest mean reward in this specific run.
  - Avoiding aggressive exploration (`eps_end=0.10`) helped final exploitation.
- What hurt performance:
  - Very high learning rate (`1e-3`) remained unstable.
  - More persistent exploration (`eps_end=0.10`, `eps_fraction=0.50`) collapsed reward to zero in this run.
  - Starting epsilon at `0.8` (exp10) underperformed versus `1.0` in this setup.

Run profile note: this completed run used `TOTAL_TIMESTEPS=5000` (configurable via environment variable in `train.py`) to fit local compute/runtime constraints while still completing all 10 required experiments.

### Extended experiments (exp11–exp20) — Sid

Additional runs vary **replay buffer**, **target update interval**, **learning_starts**, **train frequency**, **max gradient norm**, and **gradient steps**, with the same core settings as `exp1` (`lr=1e-4`, `gamma=0.99`, `batch=32`, ε from `1.0` → `0.05` over `0.3` of training). Environment: **`ALE/Breakout-v5`**, policy: **`CnnPolicy`**.

**How to run only exp11–exp20** (skip exp1–exp10):

```powershell
./run_my_experiments.sh
```

Equivalent: `RUN_EXTENDED_EXPERIMENTS_ONLY=1` and `MEMBER_NAME` set in `run_my_experiments.sh` / environment; see `src/train.py` (`EXPERIMENTS` entries for `exp11`–`exp20`).

**Run profile (documented below):** `TOTAL_TIMESTEPS=5000` per experiment.

#### Results summary (exp11–exp20)

| Experiment | lr | gamma | batch | eps_start | eps_end | eps_frac | Mean reward (± std) | Noted behavior |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp11 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.00 ± 0.00 | stable training |
| exp12 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.67 ± 0.94 | large replay buffer |
| exp13 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.67 ± 0.94 | target network schedule varied |
| exp14 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.67 ± 0.94 | target network schedule varied |
| exp15 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 1.33 ± 1.89 | best performing config |
| exp16 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.67 ± 0.94 | learning start delay varied |
| exp17 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.67 ± 0.94 | train frequency varied |
| exp18 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.00 ± 0.00 | train frequency varied |
| exp19 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.67 ± 0.94 | gradient clipping varied |
| exp20 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.30 | 0.33 ± 0.47 | multi-step gradient updates |

#### Infrastructure detail (what differed per row)

| Exp | buffer | tgt_upd | l_start | tr_freq | gr_st | mx_gn | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp11 | 10000 | 500 | 2000 | 4 | 1 | 10.0 | 0.00 |
| exp12 | 50000 | 500 | 2000 | 4 | 1 | 10.0 | 0.67 |
| exp13 | 5000 | 250 | 2000 | 4 | 1 | 10.0 | 0.67 |
| exp14 | 5000 | 1000 | 2000 | 4 | 1 | 10.0 | 0.67 |
| exp15 | 5000 | 500 | 1000 | 4 | 1 | 10.0 | 1.33 |
| exp16 | 5000 | 500 | 5000 | 4 | 1 | 10.0 | 0.67 |
| exp17 | 5000 | 500 | 2000 | 1 | 1 | 10.0 | 0.67 |
| exp18 | 5000 | 500 | 2000 | 8 | 1 | 10.0 | 0.00 |
| exp19 | 5000 | 500 | 2000 | 4 | 1 | 1.0 | 0.67 |
| exp20 | 5000 | 500 | 2000 | 4 | 4 | 10.0 | 0.33 |

**Best model in this extended run:** `exp15` (mean reward **1.33**), written to `results/models/dqn_model.zip` by `train.py` when this sweep finishes.

#### Terminal & table screenshots (Sid)

Run header and early experiments (exp11–exp12, start of exp13):

![Extended run header and exp11–exp12](docs/sid-experiments/terminal-run-header-exp11-12.png)

Mid sweep (exp14–exp17):

![Terminal output exp14–exp17](docs/sid-experiments/terminal-exp14-17.png)

Late sweep and best-model copy (exp18–exp20, `dqn_model.zip` from `exp15`):

![Terminal output exp18–exp20 and best model](docs/sid-experiments/terminal-exp18-20-best.png)

Full printed summary (results + infrastructure tables):

![Summary tables exp11–exp20](docs/sid-experiments/summary-tables-exp11-20.png)

### Final Saved Model

- The best model is saved as `dqn_model.zip`.
- A copy is also saved at `models/dqn_model.zip` so `play.py` and `train.py` use a single shared best-model artifact.

### Screenshot Evidence

Please include and keep updated screenshots in `docs/` for submission evidence:

- Hyperparameter table screenshot (10 experiments)
- Best config summary screenshot
- Play script outcome screenshot

Current repository screenshots:

![Experiment Results](docs/isaac-mugisha-experiments.png)

![Policy Comparison](docs/isaac-mugisha-experiments-policy-comparison.png)

![Play Outcome](docs/play-outcome.png)

Extended experiments (exp11–exp20) — Sid: see **Extended experiments (exp11–exp20) — Sid** above and image files under `docs/sid-experiments/`.

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

- **Model Download**: Uses local `models/dqn_model.zip` first, then falls back to downloading `models/dqn_model.zip` from GitHub when missing.
  
- **Gameplay Episodes**: Runs 5 test episodes with greedy action selection (deterministic policy)
  - Each episode displays reward and episode length
  - Real-time rendering of agent gameplay
  
- **Performance Metrics**: Aggregated statistics:
  - Mean Reward: Average across all episodes
  - Best Episode: Highest single-episode reward
  - Mean Length: Average episode duration in steps

## Troubleshooting

- ROM/license setup issues: run `.venv\Scripts\AutoROM.exe --accept-license`.
- Model/environment mismatch in gameplay: verify `MODEL_PATH` and `ENV_ID` consistency.
- Slow training on local hardware: reduce `TOTAL_TIMESTEPS` for quick validation runs.
- Remote rendering limitations: run gameplay on a local machine with display support.
