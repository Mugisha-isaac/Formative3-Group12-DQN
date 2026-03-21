import subprocess
import sys


def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


_install("numpy<2.0")
_install("stable-baselines3[extra]>=2.3.0")
_install("gymnasium[atari,accept-rom-license]>=0.29.0")
_install("ale-py>=0.9.0")
_install("shimmy[atari]>=0.2.1")
_install("autorom[accept-rom-license]>=0.6.1")
_install("opencv-python-headless")

subprocess.call(["AutoROM", "--accept-license", "-q"])

import os
import warnings
import urllib.request
import numpy as np
import ale_py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, HTML
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

GITHUB_REPO = "Mugisha-isaac/Formative3-Group12-DQN"
GITHUB_BRANCH = "isaacm"
MODEL_NAME = "dqn_model_exp2"
MODEL_ZIP = f"{MODEL_NAME}.zip"
MODEL_PATH = f"../results/models/{MODEL_ZIP}"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}/raw/{GITHUB_BRANCH}/models/{MODEL_ZIP}"

ENV_ID = "ALE/Breakout-v5"
N_STACK = 4
N_EPISODES = 3


def download_model() -> None:
    import os

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists: {MODEL_PATH}")
        return
    print(f"Downloading {MODEL_ZIP} from GitHub ...")
    try:
        urllib.request.urlretrieve(GITHUB_URL, MODEL_PATH)
        print(f"Downloaded -> {MODEL_PATH}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Ensure the .zip file is uploaded to GitHub under the models/ folder.")
        raise


def make_play_env(seed: int = 42) -> VecFrameStack:
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": "rgb_array"},
    )
    return VecFrameStack(env, n_stack=N_STACK)


def render_episode(frames: list, episode: int) -> None:
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.axis("off")
    img_plot = ax.imshow(frames[0])

    def update(frame_idx):
        img_plot.set_data(frames[frame_idx])
        return [img_plot]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=33, blit=True
    )
    display(HTML(ani.to_jshtml()))
    plt.close(fig)


def play() -> None:
    download_model()

    print(f"\nLoading model: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH, device="cpu")

    env = make_play_env()
    all_rewards = []
    all_lengths = []

    print("\n" + "=" * 55)
    print("DQN PLAY MODE — ATARI BREAKOUT")
    print(f"Model   : {GITHUB_URL}")
    print(f"Episodes: {N_EPISODES}")
    print(f"Policy  : Greedy (deterministic=True)")
    print("=" * 55)

    for ep in range(1, N_EPISODES + 1):
        obs = env.reset()
        done = np.array([False])
        ep_reward = 0.0
        ep_length = 0
        frames = []

        print(f"\nEpisode {ep}/{N_EPISODES}")

        while not bool(np.any(done)):
            frames.append(env.env_method("render")[0])
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += float(reward[0])
            ep_length += 1

        all_rewards.append(ep_reward)
        all_lengths.append(ep_length)
        print(f"  Reward: {ep_reward:.1f}  |  Length: {ep_length} steps")
        render_episode(frames, ep)

    env.close()

    print("\n" + "=" * 55)
    print(f"  Mean Reward : {np.mean(all_rewards):.2f}")
    print(f"  Best Episode: {np.max(all_rewards):.2f}")
    print(f"  Mean Length : {np.mean(all_lengths):.0f} steps")
    print("=" * 55)


if __name__ == "__main__":
    play()
