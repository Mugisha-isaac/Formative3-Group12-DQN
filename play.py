import subprocess, sys, os
import time
import warnings
import numpy as np
import urllib.request
import shutil


# Install dependencies
def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


print("Installing dependencies...")
_install("stable-baselines3[extra]>=2.3.0")
_install("gymnasium[atari,accept-rom-license]>=0.29.0")
_install("ale-py>=0.9.0")
_install("shimmy[atari]>=0.2.1")
_install("autorom[accept-rom-license]>=0.6.1")

print("Setting up AutoROM...")
subprocess.call(["AutoROM", "--accept-license", "-q"])

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    DummyVecEnv,
    VecTransposeImage,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# GitHub model configuration
GITHUB_REPO = "Mugisha-isaac/Formative3-Group12-DQN"
GITHUB_BRANCH = "isaacm"
MODEL_NAME = "dqn_model_exp10"
GITHUB_RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/models/{MODEL_NAME}"

MODEL_DIR = f"./{MODEL_NAME}"
MODEL_PATH = f"{MODEL_DIR}/policy.pth"
ENV_ID = "ALE/Breakout-v5"
N_STACK = 4
N_EPISODES = 5
STEP_DELAY = 0.02


def download_model_from_github():
    """Download the DQN model files from GitHub"""

    if os.path.exists(MODEL_DIR):
        print(f"Model directory '{MODEL_DIR}' already exists. Skipping download...")
        return

    print(f"Downloading model from GitHub: {GITHUB_RAW_URL}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Files to download from the GitHub repo
    files_to_download = [
        "policy.pth",
        "policy.optimizer.pth",
        "pytorch_variables.pth",
        "_stable_baselines3_version",
        "system_info.txt",
    ]

    for file_name in files_to_download:
        file_url = f"{GITHUB_RAW_URL}/{file_name}"
        file_path = f"{MODEL_DIR}/{file_name}"
        try:
            print(f"  Downloading {file_name}")
            urllib.request.urlretrieve(file_url, file_path)
        except Exception as e:
            print(f"  Warning: Failed to download {file_name}: {e}")

    # Download the data directory
    try:
        print("  Downloading data files")
        data_dir = f"{MODEL_DIR}/data"
        os.makedirs(data_dir, exist_ok=True)
        data_files = ["action_count.npy", "action_steps.npy", "action_rewards.npy"]
        for data_file in data_files:
            data_url = f"{GITHUB_RAW_URL}/data/{data_file}"
            data_path = f"{data_dir}/{data_file}"
            try:
                urllib.request.urlretrieve(data_url, data_path)
            except Exception:
                pass
    except Exception as e:
        print(f"  Warning: Failed to download data files: {e}")

    print(f"Model downloaded to '{MODEL_DIR}'")


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
    # Download model from GitHub if not already present
    download_model_from_github()

    print(f"\nLoading model from: {MODEL_DIR}")
    model = DQN.load(MODEL_DIR)

    env = build_env()
    all_rewards = []
    all_lengths = []

    for ep in range(1, N_EPISODES + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_length = 0
        done = False

        print(f"\nEpisode {ep}/{N_EPISODES}")

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

    print("\n" + "-" * 40)
    print(f"  Mean Reward:  {np.mean(all_rewards):.2f}")
    print(f"  Best Episode: {np.max(all_rewards):.2f}")
    print(f"  Mean Length:  {np.mean(all_lengths):.0f} steps")
    print("-" * 40)


if __name__ == "__main__":
    play()
