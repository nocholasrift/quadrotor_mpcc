import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from common import *
from tube_gen import *
from mpc_env import VolaDroneEnv

# ==========================================
# STYLE CONFIGURATION (Everything Bolded)
# ==========================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    # These parameters handle the non-math text bolding
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.weight": "bold",
    "axes.titlesize": 22,
    "axes.labelsize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 18,
    "figure.titlesize": 28,
    # This preamble ensures bold math is available
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}"
})

CELL_FONT_SIZE = 25
# ==========================================

def run_single(env, alpha0, alpha1, max_steps=2000):
    """Run one episode with fixed alphas, collect CBF metrics."""
    env.reset()
    env.alpha0 = alpha0
    env.alpha1 = alpha1

    cbf_vals = []
    steps = 0

    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step()
        steps += 1

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
        cbf_vals.append(cbf_val)

        if terminated or truncated:
            break

    cbf_arr = np.array(cbf_vals)
    violations = cbf_arr < 0
    violation_count = int(np.sum(violations))

    return {
        "alpha0": alpha0,
        "alpha1": alpha1,
        "steps": steps,
        "violation_count": violation_count,
        "violation_pct": float((violation_count / steps) * 100) if steps > 0 else 0.0,
        "violation_area": float(-np.sum(cbf_arr[violations])),
        "min_cbf": float(np.min(cbf_arr)),
        "mean_cbf": float(np.mean(cbf_arr)),
    }

def get_save_path(base="alpha_sweep_data", ext=".npz", overwrite=False):
    filepath = f"{base}{ext}"
    if not os.path.exists(filepath): return filepath
    if overwrite: return filepath

    print(f"\n'{filepath}' already exists.")
    print("  [1] Auto-increment filename (default)\n  [2] Overwrite\n  [3] Timestamp")
    choice = input("Choice [1/2/3]: ").strip()

    if choice == "2": return filepath
    elif choice == "3":
        return f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
    else:
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"): counter += 1
        return f"{base}_{counter}{ext}"

def run_sweep(track, alpha_vals, n_runs):
    env = VolaDroneEnv(track, render_mode=None, normalize_obs=False, loop=False)
    combos = list(product(alpha_vals, alpha_vals))
    metrics = ["violation_count", "violation_pct", "violation_area", "min_cbf", "mean_cbf"]
    all_results = {combo: {m: [] for m in metrics} for combo in combos}

    total_runs = len(combos) * n_runs
    run_idx = 0

    print(f"Running sweep on '{track}'...\n")
    print(f"{'alpha0':>8} {'alpha1':>8} | {'steps':>6} {'#viol':>6} {'%viol':>8} {'area':>8}")
    print("-" * 60)

    for a0, a1 in combos:
        for run in range(n_runs):
            run_idx += 1
            r = run_single(env, a0, a1)
            for m in metrics:
                all_results[(a0, a1)][m].append(r[m])
            
            print(f"{a0:8.1f} {a1:8.1f} | {r['steps']:6d} {r['violation_count']:6d} "
                  f"{r['violation_pct']:7.2f}% {r['violation_area']:8.4f}  [{run_idx}/{total_runs}]")
    return all_results, alpha_vals, track

def save_results(filepath, all_results, alpha_vals, track, n_runs):
    metrics = ["violation_count", "violation_pct", "violation_area", "min_cbf", "mean_cbf"]
    combos = list(product(alpha_vals, alpha_vals))
    data = {m: np.array([all_results[tuple(c)][m] for c in combos]) for m in metrics}
    np.savez(filepath, combos=np.array(combos), alpha_vals=np.array(alpha_vals), 
             track=str(track), n_runs=int(n_runs), **data)

def load_results(filepath):
    d = np.load(filepath, allow_pickle=True)
    metrics = ["violation_count", "violation_pct", "violation_area", "min_cbf", "mean_cbf"]
    combos = [tuple(c) for c in d["combos"]]
    all_results = {combo: {m: d[m][i].tolist() for m in metrics} for i, combo in enumerate(combos)}
    return all_results, d["alpha_vals"].tolist(), str(d["track"]), int(d["n_runs"])

def plot_results(all_results, alpha_vals, track, n_runs, selected_metrics=None):
    # Titles using \mathbf and \bm for bold math
    all_metrics = {
        "violation_count": (r"\textbf{Violation Count ($Steps$)}", "Reds", ".1f"),
        "violation_pct": (r"\textbf{Violation Percentage ($\%$)}", "Reds", ".2f"),
        "violation_area": (r"\textbf{Total Violation Area}", "Oranges", ".3f"),
        "min_cbf": (r"\textbf{Worst-Case CBF ($\min h$)}", "RdYlGn", ".3f"),
        "mean_cbf": (r"\textbf{Mean CBF ($\bar{h}$)}", "Greens", ".3f"),
    }

    if not selected_metrics: selected_metrics = list(all_metrics.keys())
    plot_configs = [(m, *all_metrics[m]) for m in selected_metrics if m in all_metrics]
    num_plots = len(plot_configs)
    if num_plots == 0: return

    ncols = 2 if num_plots > 1 else 1
    nrows = (num_plots + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols + 1, 5 * nrows + 2), squeeze=False)
    
    # Bolding the main title
    # fig.suptitle(rf"\textbf{{``{track}''}} " + r"\textbf{HOCBF Gain Sweep} ")

    n = len(alpha_vals)
    # Bold numbers in ticks using \bm
    labels = [rf"$\bm{{{v}}}$" for v in alpha_vals]

    for idx, (metric, title, cmap, fmt) in enumerate(plot_configs):
        ax = axes.flat[idx]
        data, std = np.zeros((n, n)), np.zeros((n, n))
        for (a0, a1), vals in all_results.items():
            i, j = alpha_vals.index(a1), alpha_vals.index(a0)
            data[i, j] = np.mean(vals[metric])
            std[i, j] = np.std(vals[metric])

        im = ax.imshow(data, cmap=cmap, aspect="equal", origin="lower")
        ax.set_title(title, pad=15)
        
        # Applying bold labels
        ax.set_xticks(range(n)); ax.set_xticklabels(labels)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)

        # Bold Alpha labels
        ax.set_xlabel(r"$\bm{\alpha_0}$")
        ax.set_ylabel(r"$\bm{\alpha_1}$")


        mid = (data.max() + data.min()) / 2
        for i, j in product(range(n), range(n)):
            color = "white" if data[i, j] > mid and cmap != "RdYlGn" else "black"
            
            # Format the values
            mean_str = f"{data[i, j]:{fmt}}"
            std_str = f"{std[i, j]:{fmt}}"
            
            # Build LaTeX strings (avoid f-string with \pm)
            if metric == "violation_pct":
                label_text = r"$\bm{" + mean_str + r"\%}$"# + "\n" + r"$\bm{\pm " + std_str + r"}$"
            else:
                label_text = r"$\bm{" + mean_str + r"}$"# + "\n" + r"$\bm{\pm " + std_str + r"}$"
            
            ax.text(j, i, label_text, ha="center", va="center", color=color, 
                    fontsize=CELL_FONT_SIZE)        

        fig.colorbar(im, ax=ax, shrink=0.8)

    for j in range(idx + 1, nrows * ncols): axes.flat[j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--track", type=str, default="3d_square")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 2.5, 5.0])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metrics", nargs="+", 
                        choices=["violation_count", "violation_pct", "violation_area", "min_cbf", "mean_cbf"])
    
    args = parser.parse_args()

    if args.data:
        all_results, alpha_vals, track, n_runs = load_results(args.data)
    else:
        all_results, alpha_vals, track = run_sweep(args.track, args.alphas, args.n_runs)
        n_runs = args.n_runs
        save_path = get_save_path(base=args.track, overwrite=args.overwrite)
        save_results(save_path, all_results, alpha_vals, track, n_runs)

    plot_results(all_results, alpha_vals, track, n_runs, selected_metrics=args.metrics)

if __name__ == "__main__":
    main()
