"""
play.py — Load a trained DQN model and watch it play Pong
Run this on your laptop after downloading dqn_model_expN.zip from Colab.

Setup (one time):
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
    pip install stable-baselines3[extra] gymnasium[atari,accept-rom-license] ale-py pygame
    AutoROM --accept-license
"""

import time
import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# ─────────────────────────────────────────────
# 👇 Point this to the .zip you downloaded from Colab
MODEL_PATH = "./dqn_model_exp1.zip"

N_EPISODES  = 5       # how many games to watch
STEP_DELAY  = 0.02    # slow down rendering (seconds per frame)
ENV_ID      = "ALE/Pong-v5"
N_STACK     = 4
# ─────────────────────────────────────────────


def build_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="human")   # opens a GUI window
        env = AtariWrapper(env)
        return env
    env = DummyVecEnv([_init])
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


def play():
    print(f"Loading model: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)

    env = build_env()
    all_rewards = []

    for ep in range(1, N_EPISODES + 1):
        obs        = env.reset()
        ep_reward  = 0.0
        ep_length  = 0
        done       = False

        print(f"\nEpisode {ep}/{N_EPISODES} ...")

        while not done:
            # deterministic=True → GreedyQPolicy (argmax Q-value)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _ = env.step(action)
            ep_reward += float(reward[0])
            ep_length += 1
            done = bool(done_arr[0])
            env.render()
            time.sleep(STEP_DELAY)

        all_rewards.append(ep_reward)
        print(f"  Reward: {ep_reward:.1f}  |  Length: {ep_length} steps")

    env.close()

    print("\n" + "=" * 40)
    print(f"  Mean reward : {np.mean(all_rewards):.2f}")
    print(f"  Best episode: {np.max(all_rewards):.2f}")
    print("=" * 40)


if __name__ == "__main__":
    play()
