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


def run_single(env, alpha0, alpha1, max_steps=350):
    """Run one episode with fixed alphas, collect safety, success, and completion."""
    env.reset()
    env.alpha0 = alpha0
    env.alpha1 = alpha1

    cbf_vals = []
    steps = 0
    success = False
    total_s = env.track_data["s"][-1]

    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step()
        steps += 1

        drone_pt = env.state[:3]
        diff = env.traj_xyzs - [drone_pt]
        dists = np.linalg.norm(diff, axis=1)
        champ_ind = np.argmin(dists)

        # Symbolic HOCBF Calculation
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

        _, _, cbf, _, _ = env.cbf_func(
            param_dict["x"],
            param_dict["y"],
            param_dict["z"],
            param_dict["vx"],
            param_dict["vy"],
            param_dict["vz"],
            param_dict["e1x"],
            param_dict["e1y"],
            param_dict["e1z"],
            param_dict["tube_a"],
            param_dict["tube_b"],
            param_dict["tube_c"],
            param_dict["tube_d"],
            param_dict["s_start"],
            param_dict["L"],
            *param_dict["global_params"],
            env.alpha0,
            env.alpha1,
            x_i,
            u_i,
        )

        cbf_val = min(float(cbf), float(env.max_tube_radius - dists[champ_ind]))
        cbf_vals.append(cbf_val)

        if terminated:
            success = True
            break
        if truncated or dists[champ_ind] > 3 * env.max_tube_radius:
            success = False
            break

    completion_pct = (env.state[10] / total_s) * 100.0
    cbf_arr = np.array(cbf_vals)
    violations = cbf_arr < 0
    violation_count = int(np.sum(violations))

    return {
        "alpha0": alpha0,
        "alpha1": alpha1,
        "steps": steps,
        "success": 1.0 if success else 0.0,
        "completion_pct": float(completion_pct),
        "violation_count": violation_count,
        "violation_pct": float((violation_count / steps) * 100) if steps > 0 else 0.0,
        "violation_area": float(-np.sum(cbf_arr[violations])),
        "min_cbf": float(np.min(cbf_arr)),
        "mean_cbf": float(np.mean(cbf_arr)),
    }


def run_sweep(track, alpha_vals, n_runs):
    env = VolaDroneEnv(track, render_mode=None, normalize_obs=False, loop=False)
    combos = list(product(alpha_vals, alpha_vals))
    metrics = [
        "success",
        "completion_pct",
        "violation_count",
        "violation_pct",
        "violation_area",
        "min_cbf",
        "mean_cbf",
    ]
    all_results = {combo: {m: [] for m in metrics} for combo in combos}

    total_runs = len(combos) * n_runs
    print(f"Running sweep on '{track}'...\n")

    for i, (a0, a1) in enumerate(combos):
        for run in range(n_runs):
            r = run_single(env, a0, a1)
            for m in metrics:
                all_results[(a0, a1)][m].append(r[m])
            print(
                f"[{i*n_runs + run + 1}/{total_runs}] a0: {a0:4.1f} a1: {a1:4.1f} | Comp: {r['completion_pct']:5.1f}% | %Viol: {r['violation_pct']:5.2f}%"
            )
    return all_results, alpha_vals, track


def plot_results(all_results, alpha_vals, track, n_runs, selected_metrics=None):
    all_metrics = {
        "violation_pct": (r"\textbf{Violation Percentage ($\%$)}", "Reds", ".2f"),
        "completion_pct": (r"\textbf{Completion ($\%$)}", "YlGnBu", ".1f"),
        "success": (r"\textbf{Success Rate ($\%$)}", "YlGn", ".0f"),
        "violation_count": (r"\textbf{Violation Count ($Steps$)}", "Reds", ".1f"),
        "violation_area": (r"\textbf{Total Violation Area}", "Oranges", ".2f"),
        "min_cbf": (r"\textbf{Worst-Case CBF ($\min h$)}", "RdYlGn", ".2f"),
        "mean_cbf": (r"\textbf{Mean CBF ($\bar{h}$)}", "Greens", ".2f"),
    }

    # Default to Violation % and Completion %
    if not selected_metrics:
        selected_metrics = ["violation_pct", "completion_pct"]

    plot_configs = [(m, *all_metrics[m]) for m in selected_metrics if m in all_metrics]
    n = len(alpha_vals)
    ncols = 2 if len(plot_configs) > 1 else 1
    nrows = (len(plot_configs) + 1) // 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols + 1, 5 * nrows + 2), squeeze=False
    )
    labels = [rf"$\bm{{{v}}}$" for v in alpha_vals]

    for idx, (metric, title, cmap_name, fmt) in enumerate(plot_configs):
        ax = axes.flat[idx]
        data = np.zeros((n, n))
        for (a0, a1), vals in all_results.items():
            i, j = alpha_vals.index(a1), alpha_vals.index(a0)
            data[i, j] = np.mean(vals[metric])

        # Logic for Colorbar scaling
        v_min, v_max = np.min(data), np.max(data)
        if metric in ["success", "completion_pct", "violation_pct"]:
            v_min, v_max = (
                (0, 100) if metric != "violation_pct" else (0, max(1.0, v_max))
            )
        else:
            avg, std = np.mean(data), np.std(data)
            v_max = min(v_max, avg + 2.5 * std)

        if v_max <= v_min:
            v_max = v_min + 1e-6
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_over(cmap(1.0))

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

        for i, j in product(range(n), range(n)):
            val = data[i, j]
            color = (
                "white"
                if val > (v_max + v_min) / 2 and cmap_name != "RdYlGn"
                else "black"
            )
            if metric in ["success", "completion_pct"]:
                color = "white" if val < 40 else "black"

            label_text = (
                r"$\bm{" + f"{val:{fmt}}" + (r"\%" if "%" in title else "") + r"}$"
            )
            ax.text(
                j,
                i,
                label_text,
                ha="center",
                va="center",
                color=color,
                fontsize=CELL_FONT_SIZE,
            )

        fig.colorbar(
            im,
            ax=ax,
            shrink=0.8,
            extend="max" if metric not in ["success", "completion_pct"] else "neither",
        )

    for j in range(idx + 1, nrows * ncols):
        axes.flat[j].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Auto-save for LaTeX
    save_name = f"gain_sweep_{track}.pdf"
    plt.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"Figure saved to: {save_name}")
    plt.show()


def get_save_path(base, overwrite=False):
    filepath = f"{base}.npz"
    if not os.path.exists(filepath) or overwrite:
        return filepath
    return f"{base}_{int(time.time())}.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--track", type=str, default="3d_square")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument(
        # "--alphas", type=float, nargs="+", default=[0.1, 2.0, 4.0, 6.0, 8.0, 10.0]
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 2.0],
    )
    parser.add_argument("--plot_range", type=float, nargs=2)
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.data:
        # Load results with a backward-compatible check for new metrics
        d = np.load(args.data, allow_pickle=True)
        alpha_vals = d["alpha_vals"].tolist()
        combos = [tuple(c) for c in d["combos"]]
        all_results = {
            combo: {
                m: d[m][i].tolist()
                for m in d.files
                if m not in ["combos", "alpha_vals", "track", "n_runs"]
            }
            for i, combo in enumerate(combos)
        }
        track, n_runs = str(d["track"]), int(d["n_runs"])
    else:
        all_results, alpha_vals, track = run_sweep(args.track, args.alphas, args.n_runs)
        save_path = get_save_path(args.track, args.overwrite)
        # Flatten dictionary for saving
        save_dict = {
            m: np.array(
                [
                    all_results[tuple(c)][m]
                    for c in list(product(alpha_vals, alpha_vals))
                ]
            )
            for m in all_results[list(all_results.keys())[0]].keys()
        }
        np.savez(
            save_path,
            combos=np.array(list(product(alpha_vals, alpha_vals))),
            alpha_vals=np.array(alpha_vals),
            track=track,
            n_runs=args.n_runs,
            **save_dict,
        )

    if args.plot_range:
        a_min, a_max = args.plot_range
        alpha_vals = [a for a in alpha_vals if a_min <= a <= a_max]
        all_results = {
            k: v
            for k, v in all_results.items()
            if (k[0] in alpha_vals and k[1] in alpha_vals)
        }

    plot_results(
        all_results, alpha_vals, track, args.n_runs, selected_metrics=args.metrics
    )


if __name__ == "__main__":
    main()
