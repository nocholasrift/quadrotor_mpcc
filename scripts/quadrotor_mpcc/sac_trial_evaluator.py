import os
import time
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from mpc_env import VolaDroneEnv
from common import *

# --- Configuration ---
MODEL_PATH = "./results/best_model/best_model.zip"
TRACK_TO_EVAL = "3d_square"
N_TRIALS = 10
MAX_STEPS = 1000
LOOP = False

def run_evaluation_trial(model, env, trial_idx):
    """Runs a single episode and returns CBF violation metrics."""
    obs, info = env.reset()
    done = False
    
    steps = 0
    violation_count = 0
    cbf_history = []

    print(f"Running Trial {trial_idx+1}/{N_TRIALS}...", end="\r")

    while not done:
        # Use deterministic=True for evaluation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract CBF value from the environment's internal state
        # Assuming your env.cbf_func or step returns the current h value
        # If your env doesn't store it, you can call env.unwrapped.cbf_func here
        # For this snippet, we assume 'cbf_val' is provided in the info dict or accessible
        
        drone_pt = env.state[:3]
        diff = env.traj_xyzs - [drone_pt]
        dists = np.linalg.norm(diff, axis=1)
        champ_ind = np.argmin(dists)
        
        local_window = get_local_window_params(
                env.track_data, 
                env.state[10],
                n_knots,
                window_dist=env.track_horizon_window,
                loop=False,
        )
        param_dict = build_acados_params(local_window, env.params, env.tube_coeffs)
        x_i = env.solver.get(0, "x")
        u_i = env.solver.get(0, "u")

        hddot, lfh, cbf, LgLfh = env.cbf_func(
            param_dict["x"], param_dict["y"], param_dict["z"],
            param_dict["vx"], param_dict["vy"], param_dict["vz"],
            param_dict["e1x"], param_dict["e1y"], param_dict["e1z"],
            param_dict["tube_a"], param_dict["tube_b"],
            param_dict["tube_c"], param_dict["tube_d"],
            param_dict["s_start"], param_dict["L"],
            *param_dict["global_params"],
            env.alpha0, env.alpha1, x_i, u_i,
        )
        
        cbf_val = np.sign(cbf) * min(np.abs(cbf), np.abs(env.max_tube_radius - dists[champ_ind]))
        if cbf_val < 0:
            violation_count += 1
        
        cbf_history.append(cbf_val)
        steps += 1
        done = terminated or truncated

    violation_pct = (violation_count / steps) * 100 if steps > 0 else 0
    
    return {
        "steps": steps,
        "violation_count": violation_count,
        "violation_pct": violation_pct,
        "min_h": np.min(cbf_history) if cbf_history else 0
    }

def main():
    # 1. Initialize Environment
    # We use normalize_obs=True to match training, but render_mode=None for speed
    env = VolaDroneEnv(
        TRACK_TO_EVAL,
        normalize_obs=True,
        render_mode=None, 
        loop=LOOP,
    )
    # env = TimeLimit(raw_env, max_episode_steps=MAX_STEPS)
    # env = Monitor(env)

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading SAC Policy: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH, env=env, device="auto")

    # 3. Run Trials
    all_violation_counts = []
    all_violation_pcts = []
    all_min_hs = []
    
    print(f"\nEvaluating policy on '{TRACK_TO_EVAL}' over {N_TRIALS} trials...")
    print("-" * 50)

    for i in range(N_TRIALS):
        result = run_evaluation_trial(model, env, i)
        all_violation_counts.append(result["violation_count"])
        all_violation_pcts.append(result["violation_pct"])
        all_min_hs.append(result["min_h"])

    # 4. Report Statistics
    mean_viol = np.mean(all_violation_counts)
    std_viol = np.std(all_violation_counts)
    mean_pct = np.mean(all_violation_pcts)
    std_pct = np.std(all_violation_pcts)
    worst_h = np.min(all_min_hs)

    print("\n" + "="*50)
    print(f"EVALUATION RESULTS: {TRACK_TO_EVAL}")
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    print("-" * 50)
    print(f"Avg. Violation Count: {mean_viol:8.2f} ± {std_viol:.2f} steps")
    print(f"Avg. Violation %:     {mean_pct:8.2f}% ± {std_pct:.2f}%")
    print(f"Worst-case h value:   {worst_h:8.4f}")
    print("="*50 + "\n")

    env.close()

if __name__ == "__main__":
    main()
