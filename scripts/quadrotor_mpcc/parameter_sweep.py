import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Assuming these are in your local directory
# from common import *
# from tube_gen import *
# from mpc_env import VolaDroneEnv

# ==========================================
# STYLE CONFIGURATION (Journal Quality Bold)
# ==========================================
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.weight": "bold",
        "axes.titlesize": 22,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 18,
        "figure.titlesize": 28,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}",
    }
)

CELL_FONT_SIZE = 15
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

        # Logic to calculate the 'true' distance to the safety boundary
        drone_pt = env.state[:3]
        diff = env.traj_xyzs - [drone_pt]
        dists = np.linalg.norm(diff, axis=1)
        champ_ind = np.argmin(dists)

        # Re-calculating CBF value for logging (using the symbolic function)
        # Note: Replace with your actual dictionary/param access as needed
        cbf_val = float(env.max_tube_radius - dists[champ_ind])
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


def plot_results(all_results, alpha_vals, track, n_runs, selected_metrics=None):
    all_metrics = {
        "violation_count": (r"\textbf{Violation Count ($Steps$)}", "Reds", ".1f"),
        "violation_pct": (r"\textbf{Violation Percentage ($\%$)}", "Reds", ".2f"),
        "violation_area": (r"\textbf{Total Violation Area}", "Oranges", ".2f"),
        "min_cbf": (r"\textbf{Worst-Case CBF ($\min h$)}", "RdYlGn", ".2f"),
        "mean_cbf": (r"\textbf{Mean CBF ($\bar{h}$)}", "Greens", ".2f"),
    }

    if not selected_metrics:
        selected_metrics = list(all_metrics.keys())

    plot_configs = [(m, *all_metrics[m]) for m in selected_metrics if m in all_metrics]
    num_plots = len(plot_configs)
    if num_plots == 0:
        return

    ncols = 2 if num_plots > 1 else 1
    nrows = (num_plots + 1) // 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols + 1, 5 * nrows + 2), squeeze=False
    )

    n = len(alpha_vals)
    labels = [rf"$\bm{{{v}}}$" for v in alpha_vals]

    for idx, (metric, title, cmap_name, fmt) in enumerate(plot_configs):
        ax = axes.flat[idx]
        data = np.zeros((n, n))

        # Populate grid
        for (a0, a1), vals in all_results.items():
            if a1 in alpha_vals and a0 in alpha_vals:
                i, j = alpha_vals.index(a1), alpha_vals.index(a0)
                data[i, j] = np.mean(vals[metric])

        # --- STATISTICAL OUTLIER ROBUSTNESS (Z-SCORE) ---
        avg = np.mean(data)
        std_dev = np.std(data)
        v_min = np.min(data)

        # Cap color mapping at 2.5 Standard Deviations above mean
        v_max_stat = avg + 2.5 * std_dev
        v_max = min(np.max(data), v_max_stat)

        if v_max <= v_min:
            v_max = v_min + 1e-6

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_over(cmap(1.0))  # Ensure outliers stay saturated
        # ------------------------------------------------

        im = ax.imshow(
            data, cmap=cmap, aspect="equal", origin="lower", vmax=v_max, vmin=v_min
        )

        ax.set_title(title, pad=15)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels)
        ax.set_xlabel(r"$\bm{\alpha_0}$")
        ax.set_ylabel(r"$\bm{\alpha_1}$")

        # Annotate tiles with raw data values
        for i, j in product(range(n), range(n)):
            val = data[i, j]
            # Switch text color to white if background is dark
            color = (
                "white"
                if val > (v_max + v_min) / 2 and cmap_name != "RdYlGn"
                else "black"
            )

            label_text = r"$\bm{" + f"{val:{fmt}}" + r"}$"
            ax.text(
                j,
                i,
                label_text,
                ha="center",
                va="center",
                color=color,
                fontsize=CELL_FONT_SIZE,
            )

        # Colorbar with 'extend' indicates clipped outliers
        fig.colorbar(im, ax=ax, shrink=0.8, extend="max")

    for j in range(idx + 1, nrows * ncols):
        axes.flat[j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to .npz results")
    parser.add_argument("--plot_range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[
            "violation_count",
            "violation_pct",
            "violation_area",
            "min_cbf",
            "mean_cbf",
        ],
    )

    args = parser.parse_args()

    if args.data:
        # Using the standard loader logic
        d = np.load(args.data, allow_pickle=True)
        alpha_vals = d["alpha_vals"].tolist()
        combos = [tuple(c) for c in d["combos"]]
        metrics_list = [
            "violation_count",
            "violation_pct",
            "violation_area",
            "min_cbf",
            "mean_cbf",
        ]

        all_results = {
            combo: {m: d[m][i].tolist() for m in metrics_list}
            for i, combo in enumerate(combos)
        }
        track = str(d["track"])
        n_runs = int(d["n_runs"])

        # Subset data if range is provided
        if args.plot_range:
            a_min, a_max = args.plot_range
            alpha_vals = [a for a in alpha_vals if a_min <= a <= a_max]
            all_results = {
                k: v
                for k, v in all_results.items()
                if (k[0] in alpha_vals and k[1] in alpha_vals)
            }
            print(
                f"Plotting subset [{a_min}, {a_max}]. Grid: {len(alpha_vals)}x{len(alpha_vals)}"
            )

        plot_results(
            all_results, alpha_vals, track, n_runs, selected_metrics=args.metrics
        )
    else:
        print("Please provide a --data path to visualize results.")


if __name__ == "__main__":
    main()
