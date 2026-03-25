import os
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from mpc_env import VolaDroneEnv
from common import *

# --- Configuration ---
# 1. Point to your best model or a specific checkpoint
MODEL_PATH = "./results/best_model/best_model.zip"
# TRACK_TO_EVAL = "race_uzh_19g"
TRACK_TO_EVAL = "3d_square"
N_EPISODES = 3
MAX_STEPS = 1500


def make_eval_env(track):
    """Ensure this matches the wrappers used in train_sac_parallel.py"""
    env = VolaDroneEnv(
        track,
        normalize_obs=True,  # MUST match training
        render_mode="human",  # Enable the 3D plotting we refactored earlier
    )
    env = TimeLimit(env, max_episode_steps=MAX_STEPS)
    env = Monitor(env)
    return env


def main():
    # 1. Create Environment
    print(f"Initializing Environment: {TRACK_TO_EVAL}")
    env = make_eval_env(TRACK_TO_EVAL)

    # 2. Load the Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from: {MODEL_PATH}")
    # device="auto" ensures it uses your GPU if available
    model = SAC.load(MODEL_PATH, env=env, device="auto")

    # 3. Evaluation Loop
    for ep in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        print(f"\n--- Starting Episode {ep+1} ---")

        while not done:
            # deterministic=True is critical for evaluation
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1
            done = terminated or truncated

            env.render()  # remove if no render method
            # input()

            # Optional: Print alpha values if they are in your info dict
            # print(f"Step {step_count} | alpha0: {env.unwrapped.alpha0:.2f}", end='\r')

        print(f"\nEpisode {ep+1} Finished!")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Steps: {step_count}")
        print(f"Final Solver Status: {info.get('solver_status', 'N/A')}")

    env.close()
    print("\nEvaluation Complete.")


if __name__ == "__main__":
    main()
