# train_sac_parallel.py
import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from gymnasium.wrappers import TimeLimit

from common import *
from mpc_env import VolaDroneEnv
from acados_settings import create_ocp

SAVE_DIR = "./results"


def make_env(track, rank, seed=0):
    """
    Create a single environment instance.

    Args:
        track: Track name
        rank: Environment ID (for seeding)
        seed: Base random seed
    """

    def _init():
        max_steps = 1500
        ocp, cbf_func = create_ocp(tube_degree)
        env = VolaDroneEnv(
            track, normalize_obs=True, render_mode=None, max_step=max_steps
        )
        env.reset(seed=seed + rank)
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = Monitor(env)  # Log episode rewards/lengths
        return env

    return _init


if __name__ == "__main__":
    # Configuration
    # tracks = ["straight_line", "race_uzh_19g"]
    tracks = ["straight_line", "3d_square", "3d_loop"]
    n_envs = 3  # Number of parallel environments
    total_timesteps = 500_000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Tracks: {tracks}")
    print(f"Parallel environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create vectorized environment
    # Each env gets a different track (cycles through list)
    print("Creating parallel environments...")
    env_fns = []
    for i in range(n_envs):
        track = tracks[i % len(tracks)]
        print(f"  Env {i}: {track}")
        env_fns.append(make_env(track, rank=i, seed=42))

    env = SubprocVecEnv(env_fns)

    # Create evaluation environment (single env)
    print(f"\nCreating evaluation environment: {tracks[0]}")
    ocp, cbf_func = create_ocp(tube_degree)
    eval_env = Monitor(VolaDroneEnv(tracks[0], normalize_obs=True))

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000 // n_envs,  # Adjust for parallel envs
        save_path=os.path.join(SAVE_DIR, "./checkpoints/"),
        name_prefix="sac_alpha",
        save_replay_buffer=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, "./best_model/"),
        log_path=os.path.join(SAVE_DIR, "./eval_logs/"),
        eval_freq=10_000 // n_envs,  # Adjust for parallel envs
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Create SAC model
    print("\nInitializing SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        device=device,
        tensorboard_log=os.path.join(SAVE_DIR, "./tensorboard_logs/"),
    )

    print(f"\nModel architecture:")
    print(f"  Policy: MlpPolicy with [256, 256] hidden layers")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Train
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        tb_log_name="sac_alpha_multitrack",
        progress_bar=True,
    )

    # Save final model
    print("\nSaving final model...")
    model.save(os.path.join(SAVE_DIR, "sac_alpha_model_final"))

    # Close environments
    env.close()
    eval_env.close()

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final model saved: sac_alpha_model_final.zip")
    print(f"Best model saved")
    print(f"Checkpoints saved")
    print(f"\nView training progress:")
    print(f"  tensorboard --logdir {os.path.join(SAVE_DIR, './tensorboard_logs/')}")
    print(f"{'='*60}\n")
