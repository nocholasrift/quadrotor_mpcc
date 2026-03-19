from tqdm import tqdm
import numpy as np

from acados_settings import create_ocp
from mpc_env import VolaDroneEnv


def collect_normalization_statistics(
    tracks=["7gates"], num_episodes=10, steps_per_episode=1000
):

    obs_buffer = []

    print(f"Collecting {num_episodes} episodes of data...")

    for track in tracks:
        env = VolaDroneEnv(track, normalize_obs=False)

        for ep in tqdm(range(num_episodes)):
            obs, _ = env.reset()
            obs_buffer.append(obs)

            for step in range(steps_per_episode):
                # Random actions
                obs, reward, done, truncated, info = env.step()

                if done:
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
    fname_prefix = tracks[0] if len(tracks) == 1 else "all"
    np.savez(
        f"stats/{fname_prefix}_normalization_stats.npz",
        obs_mean=obs_mean,
        obs_std=obs_std,
        track=track,
        num_episodes=num_episodes,
    )
    print("\nSaved to obs_normalization_stats.npz")

    return obs_mean, obs_std


if __name__ == "__main__":
    # track = "race_uzh_19g"
    tracks = ["straight_line", "race_uzh_19g"]
    collect_normalization_statistics(
        tracks=tracks, num_episodes=10, steps_per_episode=500
    )
