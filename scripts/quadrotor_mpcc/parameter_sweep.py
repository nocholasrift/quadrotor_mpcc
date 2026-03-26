import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from common import *
from mpc_env import VolaDroneEnv


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
        min_dist = np.min(dists)
        cbf_val = env.max_tube_radius - min_dist
        cbf_vals.append(cbf_val)

        if terminated or truncated:
            break

    cbf_arr = np.array(cbf_vals)
    violations = cbf_arr < 0

    return {
        "alpha0": alpha0,
        "alpha1": alpha1,
        "steps": steps,
        "violation_count": int(np.sum(violations)),
        "violation_area": float(-np.sum(cbf_arr[violations])),
        "min_cbf": float(np.min(cbf_arr)),
        "mean_cbf": float(np.mean(cbf_arr)),
    }


def get_save_path(base="alpha_sweep_data", ext=".npz", overwrite=False):
    """Determine save path, prompting user if file exists."""
    filepath = f"{base}{ext}"

    if not os.path.exists(filepath):
        return filepath

    if overwrite:
        print(f"Overwriting: {filepath}")
        return filepath

    # Prompt user
    print(f"\n'{filepath}' already exists.")
    print("  [1] Auto-increment filename (default)")
    print("  [2] Overwrite existing file")
    print("  [3] Save with timestamp")
    choice = input("Choice [1/2/3]: ").strip()

    if choice == "2":
        print(f"Overwriting: {filepath}")
        return filepath
    elif choice == "3":
        ts = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"{base}_{ts}{ext}"
        print(f"Saving as: {filepath}")
        return filepath
    else:
        # Default: auto-increment
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        filepath = f"{base}_{counter}{ext}"
        print(f"Saving as: {filepath}")
        return filepath


def run_sweep(track, alpha_vals, n_runs):
    """Run full parameter sweep, return results dict."""
    env = VolaDroneEnv(track, render_mode=None, normalize_obs=False, loop=False)

    combos = list(product(alpha_vals, alpha_vals))
    metrics = ["violation_count", "violation_area", "min_cbf", "mean_cbf"]

    all_results = {combo: {m: [] for m in metrics} for combo in combos}

    total_runs = len(combos) * n_runs
    run_idx = 0

    print(f"Running {len(combos)} combinations × {n_runs} runs = {total_runs} total on '{track}'...\n")
    print(f"{'alpha0':>8} {'alpha1':>8} {'run':>4} | {'steps':>6} {'#viol':>6} {'area':>8} {'min_cbf':>8} {'mean_cbf':>9}")
    print("-" * 80)

    for a0, a1 in combos:
        for run in range(n_runs):
            run_idx += 1
            r = run_single(env, a0, a1)
            for m in metrics:
                all_results[(a0, a1)][m].append(r[m])
            print(
                f"{a0:8.1f} {a1:8.1f} {run+1:4d} | "
                f"{r['steps']:6d} {r['violation_count']:6d} "
                f"{r['violation_area']:8.4f} {r['min_cbf']:8.4f} {r['mean_cbf']:9.4f}"
                f"  [{run_idx}/{total_runs}]"
            )

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'alpha0':>8} {'alpha1':>8} | {'#viol':>14} {'area':>14} {'min_cbf':>14} {'mean_cbf':>14}")
    print("-" * 90)
    for a0, a1 in combos:
        d = all_results[(a0, a1)]
        print(
            f"{a0:8.1f} {a1:8.1f} | "
            f"{np.mean(d['violation_count']):6.1f}±{np.std(d['violation_count']):4.1f}  "
            f"{np.mean(d['violation_area']):6.3f}±{np.std(d['violation_area']):5.3f}  "
            f"{np.mean(d['min_cbf']):6.3f}±{np.std(d['min_cbf']):5.3f}  "
            f"{np.mean(d['mean_cbf']):6.3f}±{np.std(d['mean_cbf']):5.3f}"
        )

    return all_results, alpha_vals, track


def save_results(filepath, all_results, alpha_vals, track, n_runs):
    """Save sweep results to .npz file."""
    metrics = ["violation_count", "violation_area", "min_cbf", "mean_cbf"]
    combos = list(product(alpha_vals, alpha_vals))

    combo_arr = np.array(combos)
    data = {}
    for m in metrics:
        data[m] = np.array([all_results[tuple(c)][m] for c in combos])

    np.savez(
        filepath,
        combos=combo_arr,
        alpha_vals=np.array(alpha_vals),
        track=str(track),
        n_runs=int(n_runs),
        **data,
    )
    print(f"Data saved to: {filepath}")


def load_results(filepath):
    """Load sweep results from .npz file."""
    d = np.load(filepath, allow_pickle=True)
    alpha_vals = d["alpha_vals"].tolist()
    track = str(d["track"])
    n_runs = int(d["n_runs"])
    combos = [tuple(c) for c in d["combos"]]

    metrics = ["violation_count", "violation_area", "min_cbf", "mean_cbf"]
    all_results = {}
    for i, combo in enumerate(combos):
        all_results[combo] = {m: d[m][i].tolist() for m in metrics}

    print(f"Loaded {len(combos)} combos × {n_runs} runs from '{filepath}' (track: {track})")
    return all_results, alpha_vals, track, n_runs


def plot_results(all_results, alpha_vals, track, n_runs):
    """Generate 2x2 heatmap figure."""
    metrics = ["violation_count", "violation_area", "min_cbf", "mean_cbf"]
    n = len(alpha_vals)

    grids_mean = {m: np.zeros((n, n)) for m in metrics}
    grids_std = {m: np.zeros((n, n)) for m in metrics}

    for (a0, a1), data in all_results.items():
        i = alpha_vals.index(a1)
        j = alpha_vals.index(a0)
        for m in metrics:
            grids_mean[m][i, j] = np.mean(data[m])
            grids_std[m][i, j] = np.std(data[m])

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"Alpha Sweep on '{track}' — CBF Metrics (mean ± std, n={n_runs})",
        fontsize=14, fontweight="bold",
    )

    labels_0 = [str(v) for v in alpha_vals]
    labels_1 = [str(v) for v in alpha_vals]

    plot_configs = [
        ("violation_count", "Violation Count (CBF < 0)", "Reds", ".1f", ".1f"),
        ("violation_area", "Total Violation Area (∫|min(cbf,0)|dt)", "Oranges", ".3f", ".3f"),
        ("min_cbf", "Worst-Case CBF (min over run)", "RdYlGn", ".3f", ".3f"),
        ("mean_cbf", "Mean CBF", "Greens", ".3f", ".3f"),
    ]

    for ax, (metric, title, cmap, mean_fmt, std_fmt) in zip(axes.flat, plot_configs):
        data = grids_mean[metric]
        std = grids_std[metric]
        im = ax.imshow(data, cmap=cmap, aspect="equal", origin="lower")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels_0)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels_1)
        ax.set_xlabel("alpha0")
        ax.set_ylabel("alpha1")
        ax.set_title(title)

        mid = (data.max() + data.min()) / 2
        for i in range(n):
            for j in range(n):
                color = "white" if data[i, j] > mid else "black"
                ax.text(
                    j, i,
                    f"{data[i, j]:{mean_fmt}}\n±{std[i, j]:{std_fmt}}",
                    ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold",
                )
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out_path = "alpha_sweep_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Alpha parameter sweep for CBF analysis")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to .npz file to load and re-plot (skips simulation)",
    )
    parser.add_argument("--track", type=str, default="3d_square")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[1.0, 5.0, 10.0],
        help="Alpha values to sweep",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing data file without prompting",
    )
    args = parser.parse_args()

    if args.data:
        all_results, alpha_vals, track, n_runs = load_results(args.data)
    else:
        all_results, alpha_vals, track = run_sweep(args.track, args.alphas, args.n_runs)
        n_runs = args.n_runs
        save_path = get_save_path(overwrite=args.overwrite)
        save_results(save_path, all_results, alpha_vals, track, n_runs)

    plot_results(all_results, alpha_vals, track, n_runs)


if __name__ == "__main__":
    main()
