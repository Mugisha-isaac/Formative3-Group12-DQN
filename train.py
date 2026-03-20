"""
train.py — DQN Pong Training Script
Run on Google Colab: !python train.py
"""

import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_install("numpy<2.0")
_install("stable-baselines3[extra]>=2.3.0")
_install("gymnasium[atari,accept-rom-license]>=0.29.0")
_install("ale-py>=0.9.0")
_install("shimmy[atari]>=0.2.1")
_install("autorom[accept-rom-license]>=0.6.1")
_install("tensorboard>=2.14.0")

subprocess.call(["AutoROM", "--accept-license", "-q"])

import os
import warnings
import numpy as np
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT_NAME         = "exp1"
MEMBER_NAME             = "Member A"

LEARNING_RATE           = 1e-4
GAMMA                   = 0.99
BATCH_SIZE              = 32
EXPLORATION_FRACTION    = 0.10
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS   = 0.01

ENV_ID                 = "ALE/Pong-v5"
POLICY                 = "CnnPolicy"
TOTAL_TIMESTEPS        = 500_000
N_STACK                = 4
BUFFER_SIZE            = 100_000
TARGET_UPDATE_INTERVAL = 1000
LEARNING_STARTS        = 50_000
TRAIN_FREQ             = 4

MODEL_SAVE_PATH = f"./dqn_model_{EXPERIMENT_NAME}"
LOG_DIR         = f"./logs/{EXPERIMENT_NAME}/"
os.makedirs(LOG_DIR, exist_ok=True)


def make_env(env_id, log_dir):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = AtariWrapper(env)
        env = Monitor(env, log_dir)
        return env
    return _init

def build_env(env_id, log_dir):
    env = DummyVecEnv([make_env(env_id, log_dir)])
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecTransposeImage(env)
    return env


def main():
    print("=" * 55)
    print(f"  Experiment : {EXPERIMENT_NAME}  ({MEMBER_NAME})")
    print(f"  lr={LEARNING_RATE}, gamma={GAMMA}, batch={BATCH_SIZE}")
    print(f"  ε: {EXPLORATION_INITIAL_EPS} → {EXPLORATION_FINAL_EPS} over {EXPLORATION_FRACTION} of training")
    print("=" * 55)

    train_env = build_env(ENV_ID, LOG_DIR)
    eval_env  = build_env(ENV_ID, LOG_DIR)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_PATH + "_best",
        log_path=LOG_DIR,
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=LOG_DIR + "checkpoints/",
        name_prefix=f"dqn_pong_{EXPERIMENT_NAME}",
    )

    model = DQN(
        policy=POLICY,
        env=train_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_initial_eps=EXPLORATION_INITIAL_EPS,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        learning_starts=LEARNING_STARTS,
        train_freq=TRAIN_FREQ,
        optimize_memory_usage=False,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    print(f"\n  Training for {TOTAL_TIMESTEPS:,} steps ...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,
        tb_log_name=f"DQN_Pong_{EXPERIMENT_NAME}",
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\n  Model saved → {MODEL_SAVE_PATH}.zip")

    print("\n  Evaluating over 10 episodes (greedy policy) ...")
    obs = eval_env.reset()
    episode_rewards = []
    episode_reward  = 0.0
    episodes_done   = 0

    while episodes_done < 10:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]
        if done[0]:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            episodes_done += 1
            obs = eval_env.reset()

    mean_r = np.mean(episode_rewards)
    std_r  = np.std(episode_rewards)

    print("\n" + "=" * 55)
    print(f"  {MEMBER_NAME} | {EXPERIMENT_NAME}")
    print(f"  lr={LEARNING_RATE}, gamma={GAMMA}, batch={BATCH_SIZE}, "
          f"ε_start={EXPLORATION_INITIAL_EPS}, ε_end={EXPLORATION_FINAL_EPS}, "
          f"ε_fraction={EXPLORATION_FRACTION}")
    print("─" * 55)
    print(f"  Mean reward : {mean_r:.2f} ± {std_r:.2f}")
    print(f"  All episodes: {[round(r, 1) for r in episode_rewards]}")
    print("=" * 55)
    print("\n  Copy this row into your results table:")
    print(f"  {MEMBER_NAME} | {EXPERIMENT_NAME} | "
          f"lr={LEARNING_RATE}, gamma={GAMMA}, batch={BATCH_SIZE}, "
          f"ε_start={EXPLORATION_INITIAL_EPS}, ε_end={EXPLORATION_FINAL_EPS}, "
          f"ε_fraction={EXPLORATION_FRACTION} | Mean reward: {mean_r:.2f}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
