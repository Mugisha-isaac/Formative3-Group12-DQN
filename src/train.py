"""Run DQN hyperparameter experiments (exp1–exp10 + optional exp11–exp20) and save the best model.

exp1–exp10: lr / gamma / batch / exploration sweeps (group baseline).
exp11–exp20: replay / target network / train schedule / gradient settings (same lr–eps baseline as exp1).

Dependencies should be installed once via requirements.txt before running this file.
"""

import os
import gc
import csv
import warnings
import numpy as np
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    DummyVecEnv,
    VecTransposeImage,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MEMBER_NAME = os.getenv("MEMBER_NAME", "Gikundiro Liliane")

EXPERIMENTS = [
    {
        "name": "exp1",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
    },
    {
        "name": "exp2",
        "lr": 5e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
    },
    {
        "name": "exp3",
        "lr": 1e-3,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
    },
    {
        "name": "exp4",
        "lr": 1e-4,
        "gamma": 0.95,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
    },
    {
        "name": "exp5",
        "lr": 1e-4,
        "gamma": 0.999,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
    },
    {
        "name": "exp6",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 64,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
    },
    {
        "name": "exp7",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 128,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
    },
    {
        "name": "exp8",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.10,
        "eps_fraction": 0.50,
    },
    {
        "name": "exp9",
        "lr": 2e-4,
        "gamma": 0.98,
        "batch": 64,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.40,
    },
    {
        "name": "exp10",
        "lr": 3e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 0.8,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
    },
    # --- exp11–exp20: infrastructure / training dynamics (baseline = exp1 core hparams) ---
    {
        "name": "exp11",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
        "buffer_size": 10_000,
    },
    {
        "name": "exp12",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
        "buffer_size": 50_000,
    },
    {
        "name": "exp13",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
        "target_update_interval": 250,
    },
    {
        "name": "exp14",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
        "target_update_interval": 1000,
    },
    {
        "name": "exp15",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
        "learning_starts": 1_000,
    },
    {
        "name": "exp16",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
        "learning_starts": 5_000,
    },
    {
        "name": "exp17",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
        "train_freq": 1,
    },
    {
        "name": "exp18",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
        "train_freq": 8,
    },
    {
        "name": "exp19",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
        "max_grad_norm": 1.0,
    },
    {
        "name": "exp20",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch": 32,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_fraction": 0.30,
        "gradient_steps": 4,
    },
]

ENV_ID = "ALE/Breakout-v5"
POLICY = "CnnPolicy"
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "5000"))
N_STACK = 4
BUFFER_SIZE = 5_000
TARGET_UPDATE_INTERVAL = 500
LEARNING_STARTS = 2_000
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
MAX_GRAD_NORM = 10.0
RUN_POLICY_COMPARISON = os.getenv("RUN_POLICY_COMPARISON", "0") == "1"
# Run only exp11–exp20 (skip exp1–exp10) when set to 1 — useful for your extra runs without redoing group configs.
RUN_EXTENDED_EXPERIMENTS_ONLY = os.getenv("RUN_EXTENDED_EXPERIMENTS_ONLY", "0") == "1"


class RewardLengthLogger(BaseCallback):
    def __init__(self, log_path):
        super().__init__(verbose=0)
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0
        self.current_length = 0

    def _on_step(self):
        self.current_reward += self.locals["rewards"][0]
        self.current_length += 1
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.current_reward = 0
            self.current_length = 0
        return True

    def save_log(self):
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length"])
            for i, (r, l) in enumerate(zip(self.episode_rewards, self.episode_lengths)):
                writer.writerow([i + 1, r, l])


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


def run_experiment(exp):
    name = exp["name"]
    lr = exp["lr"]
    gamma = exp["gamma"]
    batch = exp["batch"]
    eps_start = exp["eps_start"]
    eps_end = exp["eps_end"]
    eps_fraction = exp["eps_fraction"]
    buffer_size = exp.get("buffer_size", BUFFER_SIZE)
    target_update_interval = exp.get("target_update_interval", TARGET_UPDATE_INTERVAL)
    learning_starts = exp.get("learning_starts", LEARNING_STARTS)
    train_freq = exp.get("train_freq", TRAIN_FREQ)
    gradient_steps = exp.get("gradient_steps", GRADIENT_STEPS)
    max_grad_norm = exp.get("max_grad_norm", MAX_GRAD_NORM)
    log_dir = f"../results/logs/{name}/"
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 55)
    print(f"  {MEMBER_NAME} | {name}")
    print(f"  lr={lr}, gamma={gamma}, batch={batch}")
    print(f"  epsilon: {eps_start} -> {eps_end} over {eps_fraction} of training")
    print(
        f"  buffer={buffer_size}, target_update={target_update_interval}, "
        f"learning_starts={learning_starts}, train_freq={train_freq}, "
        f"gradient_steps={gradient_steps}, max_grad_norm={max_grad_norm}"
    )
    print("=" * 55)

    train_env = build_env(ENV_ID, log_dir)

    logger = RewardLengthLogger(log_path=f"{log_dir}reward_length_log.csv")

    model = DQN(
        policy=POLICY,
        env=train_env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch,
        gamma=gamma,
        exploration_fraction=eps_fraction,
        exploration_initial_eps=eps_start,
        exploration_final_eps=eps_end,
        target_update_interval=target_update_interval,
        learning_starts=learning_starts,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        max_grad_norm=max_grad_norm,
        optimize_memory_usage=False,
        tensorboard_log=log_dir,
        verbose=0,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=logger,
        tb_log_name=f"DQN_{name}",
    )

    logger.save_log()

    if logger.episode_rewards:
        print(
            f"  Reward trend  - first episode: {logger.episode_rewards[0]:.1f}  |  last episode: {logger.episode_rewards[-1]:.1f}"
        )
        print(
            f"  Length trend  - first episode: {logger.episode_lengths[0]}  |  last episode: {logger.episode_lengths[-1]}"
        )

    model.save(f"../results/models/dqn_model_{name}")

    eval_env = build_env(ENV_ID, log_dir)
    obs = eval_env.reset()
    episode_rewards = []
    episode_reward = 0.0
    episodes_done = 0

    while episodes_done < 3:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward[0]
        if done[0]:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            episodes_done += 1
            obs = eval_env.reset()

    mean_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)

    train_env.close()
    eval_env.close()
    del model, train_env, eval_env, logger
    gc.collect()

    return {
        "name": name,
        "lr": lr,
        "gamma": gamma,
        "batch": batch,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_fraction": eps_fraction,
        "buffer_size": buffer_size,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "max_grad_norm": max_grad_norm,
        "mean_reward": mean_r,
        "std_reward": std_r,
    }


def policy_comparison():
    print("\n" + "=" * 55)
    print("  POLICY COMPARISON: CnnPolicy vs MlpPolicy")
    print("=" * 55)

    log_dir = "../results/logs/policy_comparison/"
    os.makedirs(log_dir, exist_ok=True)

    results = {}
    for policy in ["CnnPolicy", "MlpPolicy"]:
        print(f"\n  Training with {policy} ...")
        train_env = build_env(ENV_ID, log_dir)

        model = DQN(
            policy=policy,
            env=train_env,
            learning_rate=1e-4,
            buffer_size=BUFFER_SIZE,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.30,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            target_update_interval=TARGET_UPDATE_INTERVAL,
            learning_starts=LEARNING_STARTS,
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
            optimize_memory_usage=False,
            verbose=0,
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        eval_env = build_env(ENV_ID, log_dir)
        obs = eval_env.reset()
        ep_rewards = []
        ep_reward = 0.0
        ep_done = 0

        while ep_done < 3:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward[0]
            if done[0]:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_done += 1
                obs = eval_env.reset()

        results[policy] = np.mean(ep_rewards)
        print(f"  {policy} mean reward: {results[policy]:.2f}")

        train_env.close()
        eval_env.close()
        del model, train_env, eval_env
        gc.collect()

    print("\n  Winner:", max(results, key=results.get))
    print("=" * 55)


def _behavior_note(r, best_reward):
    if r["mean_reward"] >= best_reward:
        return "best performing config"
    if r["lr"] > 5e-4:
        return "high lr - unstable learning"
    if r["gamma"] < 0.97:
        return "low gamma - short-sighted"
    if r["batch"] >= 128:
        return "large batch - slow updates"
    if r["eps_end"] >= 0.10:
        return "high eps_end - more exploration"
    if r["buffer_size"] >= 50_000:
        return "large replay buffer"
    if r["target_update_interval"] != TARGET_UPDATE_INTERVAL:
        return "target network schedule varied"
    if r["learning_starts"] != LEARNING_STARTS:
        return "learning start delay varied"
    if r["train_freq"] != TRAIN_FREQ:
        return "train frequency varied"
    if r["gradient_steps"] != GRADIENT_STEPS:
        return "multi-step gradient updates"
    if r["max_grad_norm"] != MAX_GRAD_NORM:
        return "gradient clipping varied"
    return "stable training"


def main():
    to_run = EXPERIMENTS[10:] if RUN_EXTENDED_EXPERIMENTS_ONLY else EXPERIMENTS
    n_exp = len(to_run)
    print("\n  Running", n_exp, "experiments for", MEMBER_NAME)
    if RUN_EXTENDED_EXPERIMENTS_ONLY:
        print("  (RUN_EXTENDED_EXPERIMENTS_ONLY=1: exp11–exp20 only)")
    print("  Environment:", ENV_ID)
    print("  Policy:", POLICY)
    print("  Total timesteps per experiment:", TOTAL_TIMESTEPS)
    print()

    results = []
    best_reward = -float("inf")
    best_model_name = None

    for exp in to_run:
        result = run_experiment(exp)
        results.append(result)
        print(
            f"  {result['name']} done - Mean reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}\n"
        )

        if result["mean_reward"] > best_reward:
            best_reward = result["mean_reward"]
            best_model_name = result["name"]

    import shutil

    if best_model_name:
        best_src = f"../results/models/dqn_model_{best_model_name}.zip"
        best_dst = "../results/models/dqn_model.zip"
        shutil.copy(best_src, best_dst)
        print(
            f"  Best model ({best_model_name}, reward={best_reward:.2f}) saved as ../results/models/dqn_model.zip"
        )

    print("\n" + "=" * 55)
    print(f"  RESULTS SUMMARY - {MEMBER_NAME}")
    print("=" * 55)
    print(
        f"  {'Exp':<6} {'lr':<8} {'gamma':<7} {'batch':<7} {'eps_start':<10} {'eps_end':<9} {'eps_frac':<10} {'Mean Reward':<12} {'Noted Behavior'}"
    )
    print("-" * 90)
    for r in results:
        note = _behavior_note(r, best_reward)
        print(
            f"  {r['name']:<6} {r['lr']:<8} {r['gamma']:<7} {r['batch']:<7} {r['eps_start']:<10} {r['eps_end']:<9} {r['eps_fraction']:<10} {r['mean_reward']:<12.2f} {note}"
        )
    print("=" * 90)

    extended_rows = []
    for r in results:
        if not r["name"].startswith("exp"):
            continue
        try:
            if int(r["name"].replace("exp", "")) >= 11:
                extended_rows.append(r)
        except ValueError:
            continue
    if extended_rows:
        print("\n" + "=" * 115)
        print(
            "  INFRASTRUCTURE DETAIL (exp11–exp20): buffer, target update, learning_starts, "
            "train_freq, grad_steps, max_grad_norm"
        )
        print("=" * 115)
        hdr = (
            f"  {'Exp':<6} {'buffer':<9} {'tgt_upd':<8} {'l_start':<8} {'tr_freq':<8} {'gr_st':<6} "
            f"{'mx_gn':<8} {'Mean':<8} {'Note'}"
        )
        print(hdr)
        print("-" * 115)
        for r in extended_rows:
            note = _behavior_note(r, best_reward)
            print(
                f"  {r['name']:<6} {r['buffer_size']:<9} {r['target_update_interval']:<8} "
                f"{r['learning_starts']:<8} {r['train_freq']:<8} {r['gradient_steps']:<6} "
                f"{r['max_grad_norm']:<8.1f} {r['mean_reward']:<8.2f} {note}"
            )
        print("=" * 115)

    if RUN_POLICY_COMPARISON:
        policy_comparison()


if __name__ == "__main__":
    main()
