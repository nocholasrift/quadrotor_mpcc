from tqdm import tqdm
import numpy as np

from acados_settings import create_ocp
from mpc_env import VolaDroneEnv


def collect_normalization_statistics(
    track="7gates", num_episodes=10, steps_per_episode=200
):
    ocp, cbf_func = create_ocp()
    env = VolaDroneEnv(ocp, track, cbf_func=cbf_func, normalize_obs=False)

    obs_buffer = []

    print(f"Collecting {num_episodes} episodes of data...")
    for ep in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        obs_buffer.append(obs)

        for step in range(steps_per_episode):
            # Random actions
            obs, reward, done, truncated, info = env.step()

            if done or truncated:
                break

            obs_buffer.append(obs)

    # Compute statistics
    obs_array = np.array(obs_buffer)
    obs_mean = np.mean(obs_array, axis=0)
    obs_std = np.std(obs_array, axis=0) + 1e-8  # Avoid division by zero

    print("\nObservation Statistics:")
    print(f"Shape: {obs_array.shape}")
    print(f"Mean: {obs_mean}")
    print(f"Std:  {obs_std}")

    # Save to file
    np.savez(
        f"stats/{track}_normalization_stats.npz",
        obs_mean=obs_mean,
        obs_std=obs_std,
        track=track,
        num_episodes=num_episodes,
    )
    print("\nSaved to obs_normalization_stats.npz")

    return obs_mean, obs_std


if __name__ == "__main__":
    track = "race_uzh_19g"
    collect_normalization_statistics(
        track=track, num_episodes=10, steps_per_episode=500
    )
