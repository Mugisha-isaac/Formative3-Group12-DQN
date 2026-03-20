import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_install("stable-baselines3[extra]>=2.3.0")
_install("gymnasium[atari,accept-rom-license]>=0.29.0")
_install("ale-py>=0.9.0")
_install("shimmy[atari]>=0.2.1")
_install("autorom[accept-rom-license]>=0.6.1")

subprocess.call(["AutoROM", "--accept-license", "-q"])

import time
import warnings
import numpy as np
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = "./dqn_model.zip"
ENV_ID     = "ALE/Breakout-v5"
N_STACK    = 4
N_EPISODES = 5
STEP_DELAY = 0.02


def build_env():
    def _init():
        env = gym.make(ENV_ID, render_mode="human")
        env = AtariWrapper(env)
        return env
    env = DummyVecEnv([_init])
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecTransposeImage(env)
    return env


def play():
    print(f"Loading model: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)

    env = build_env()
    all_rewards = []
    all_lengths = []

    for ep in range(1, N_EPISODES + 1):
        obs        = env.reset()
        ep_reward  = 0.0
        ep_length  = 0
        done       = False

        print(f"\nEpisode {ep}/{N_EPISODES} ...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, _ = env.step(action)
            ep_reward += float(reward[0])
            ep_length += 1
            done = bool(done_arr[0])
            env.render()
            time.sleep(STEP_DELAY)

        all_rewards.append(ep_reward)
        all_lengths.append(ep_length)
        print(f"  Reward: {ep_reward:.1f}  |  Length: {ep_length} steps")

    env.close()

    print("\n" + "=" * 40)
    print(f"  Mean reward : {np.mean(all_rewards):.2f}")
    print(f"  Best episode: {np.max(all_rewards):.2f}")
    print(f"  Mean length : {np.mean(all_lengths):.0f} steps")
    print("=" * 40)


if __name__ == "__main__":
    play()
