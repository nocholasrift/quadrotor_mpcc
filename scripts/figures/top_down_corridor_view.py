import os
import sys
mpcc_path = os.path.abspath("../quadrotor_mpcc")
if mpcc_path not in sys.path:
    sys.path.append(mpcc_path)

from common import *
from tube_gen import *
from mpc_env import VolaDroneEnv
from load_env import load_pcl_from_env

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import cKDTree

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

TRACK   = "short_line"
S_VALUE = 2.0

ELLIPSE_SLICES  = 50
N_CENTER_PTS    = 200
N_ELLIPSE_DRAWS = 20

FRAME_SLICES = 3
FRAME_LENGTH = 0.6

BG      = "white"
C_TRACK = "#1C1C2E"
C_TUBE  = "#4A90D9"
C_PCL   = "viridis"
C_ROBOT = "darkorange"
C_CENTER = "#E05252"

C_T  = "#E05252"
C_E1 = "#52A852"
C_E2 = "#5271E0"

# ==============================================================================
#  ELLIPSE HELPERS
# ==============================================================================

def get_ellipse_parameters(P, pp, p0=np.array([0, 0])):
    a = P[0, 0]
    b = 2 * P[0, 1]
    c = P[1, 1]
    d = pp[0]
    ee = pp[1]
    f = -1

    disc = b**2 - 4 * a * c
    inner = a * ee**2 + c * d**2 - b * d * ee + disc * f

    aell = -np.sqrt(
        2 * inner * ((a + c) + np.sqrt((a - c)**2 + b**2))
    ) / disc
    bell = -np.sqrt(
        2 * inner * ((a + c) - np.sqrt((a - c)**2 + b**2))
    ) / disc

    xc = (2 * c * d - b * ee) / disc + p0[0]
    yc = (2 * a * ee - b * d) / disc + p0[1]

    height = 2 * aell
    width = 2 * bell
    pc = np.array([xc, yc])
    theta = 0.5 * np.arctan2(-b, c - a) + np.pi / 2

    return pc, width, height, theta

def get_ellipse_points(width, height, angle, thetas):
    def r_ellipse(theta, a, b):
        return a * b / np.sqrt((a * np.sin(theta))**2 + (b * np.cos(theta))**2)

    a = width / 2
    b = height / 2

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    pts = []
    for theta in thetas:
        r = r_ellipse(theta - angle, a, b)
        rot = r * np.array([np.cos(theta - angle), np.sin(theta - angle)])
        pts.append(R @ rot)

    return np.array(pts)

# ==============================================================================
#  HELPERS
# ==============================================================================

def eval_tube_at_s(local_window, tube_coeffs, s_query):
    s_local = local_window["s"]
    xi = (s_query - s_local[0]) / (s_local[-1] - s_local[0] + 1e-12)
    poly_deg = tube_coeffs.shape[1] - 1
    Phi = get_cheby_basis(xi, poly_deg)

    a_vals = Phi @ tube_coeffs[0]
    b_vals = Phi @ tube_coeffs[1]
    c_vals = Phi @ tube_coeffs[2]
    d_vals = Phi @ tube_coeffs[3]
    return a_vals, b_vals, c_vals, d_vals

def get_frame_at_s(local_window, s_query):
    s_local = local_window["s"]

    pos = np.stack([
        np.interp(s_query, s_local, local_window["x"]),
        np.interp(s_query, s_local, local_window["y"]),
    ], axis=1)

    vx  = np.interp(s_query, s_local, local_window["vx"])
    vy  = np.interp(s_query, s_local, local_window["vy"])
    vz  = np.interp(s_query, s_local, local_window["vz"])
    e1x = np.interp(s_query, s_local, local_window["e1x"])
    e1y = np.interp(s_query, s_local, local_window["e1y"])
    e1z = np.interp(s_query, s_local, local_window["e1z"])

    T = np.stack([vx, vy, vz], axis=1)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12

    e1 = np.stack([e1x, e1y, e1z], axis=1)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12

    e2 = np.cross(T, e1)

    e1_xy = e1[:, :2]
    e2_xy = e2[:, :2]
    T_xy  = T[:, :2]

    return pos, T_xy, e1_xy, e2_xy

# ==============================================================================
#  DRAWING FUNCTIONS
# ==============================================================================

def draw_obstacle_projections_topdown(ax, pcl, local_window, max_dist=2.0):
    s_local = local_window["s"]
    s_fine = np.linspace(s_local[0], s_local[-1], 500)

    cx = np.interp(s_fine, s_local, local_window["x"])
    cy = np.interp(s_fine, s_local, local_window["y"])
    cz = np.interp(s_fine, s_local, local_window["z"])
    curve_fine = np.stack([cx, cy, cz], axis=1)

    tree = cKDTree(curve_fine)
    projected_s_vals = []

    for obs_pt in pcl:
        dist, idx = tree.query(obs_pt)

        if dist > max_dist:
            continue
        if s_fine[idx] < s_local[0] or s_fine[idx] > s_local[-1]:
            continue

        ax.scatter(cx[idx], cy[idx], s=40, color="black", zorder=6)
        projected_s_vals.append(s_fine[idx])

    return np.unique(projected_s_vals)

def draw_ellipses_topdown(ax, local_window, tube_coeffs, s_query, n_ellipse_pts=50,
                          fill_alpha=0.12, edge_alpha=0.85):
    if len(s_query) == 0:
        return

    pos, _, e1_xy, e2_xy = get_frame_at_s(local_window, s_query)
    a_vals, b_vals, c_vals, d_vals = eval_tube_at_s(local_window, tube_coeffs, s_query)
    thetas = np.linspace(0, 2 * np.pi, n_ellipse_pts)

    for i in range(len(s_query)):
        P = np.array([[a_vals[i], 0], [0, b_vals[i]]])
        pp = np.array([c_vals[i], d_vals[i]])

        try:
            pc, width, height, angle = get_ellipse_parameters(P, pp)
        except Exception as e:
            print(f"Skipping ellipse {i}: {e}")
            continue

        pts_2d = get_ellipse_points(width, height, angle, thetas) + pc

        w1 = pts_2d[:, 0]
        w2 = pts_2d[:, 1]
        world_xy = pos[i] + w1[:, None] * e1_xy[i] + w2[:, None] * e2_xy[i]

        patch = Polygon(
            world_xy,
            closed=True,
            facecolor=C_TUBE,
            edgecolor=C_TUBE,
            alpha=fill_alpha,
            linewidth=0
        )
        ax.add_patch(patch)

        ax.plot(
            world_xy[:, 0], world_xy[:, 1],
            color=C_TUBE, lw=1.8, alpha=edge_alpha, zorder=4
        )

def draw_ellipse_center_line(ax, local_window, tube_coeffs, n_pts=200):
    s_local = local_window["s"]
    s_query = np.linspace(s_local[0], s_local[-1], n_pts)

    pos, _, e1_xy, e2_xy = get_frame_at_s(local_window, s_query)
    a_vals, b_vals, c_vals, d_vals = eval_tube_at_s(local_window, tube_coeffs, s_query)

    centers_world = []

    for i in range(n_pts):
        P = np.array([[a_vals[i], 0], [0, b_vals[i]]])
        pp = np.array([c_vals[i], d_vals[i]])

        try:
            pc, _, _, _ = get_ellipse_parameters(P, pp)
            center_world = pos[i] + pc[0] * e1_xy[i] + pc[1] * e2_xy[i]
        except Exception:
            center_world = pos[i]

        centers_world.append(center_world)

    centers_world = np.array(centers_world)

    ax.plot(
        centers_world[:, 0], centers_world[:, 1],
        color=C_CENTER, lw=4.5, ls="--", alpha=0.95, zorder=5
    )

def draw_rmf_frames_topdown(ax, local_window, n_frames=3, length=0.6):
    s_local = local_window["s"]
    s_query = np.linspace(s_local[0], s_local[-1], n_frames)

    pos, T_xy, e1_xy, e2_xy = get_frame_at_s(local_window, s_query)

    for i in range(n_frames):
        for vec, color in zip([T_xy, e1_xy, e2_xy], [C_T, C_E1, C_E2]):
            v = vec[i]
            vnorm = np.linalg.norm(v)
            if vnorm < 1e-10:
                continue
            v = length * v / vnorm

            ax.arrow(
                pos[i, 0], pos[i, 1],
                v[0], v[1],
                color=color,
                width=.05,
                head_width=0.10,
                head_length=0.12,
                length_includes_head=True,
                zorder=7,
                alpha=0.95
            )

# ==============================================================================
#  Extract data from env
# ==============================================================================

def extract_snapshot(env, n_knots_plot=200):
    env.reset()
    while env.state[10] < S_VALUE:
        env.step()

    s_now = env.state[10]
    robot_pos = env.state[:3]
    tube_coeffs = env.tube_coeffs.copy()
    track_data = env.track_data
    pcl = env.pcl

    local_window_plot = get_local_window_params(
        track_data, s_now, n_knots_plot,
        window_dist=env.track_horizon_window
    )

    return {
        "s_now":        s_now,
        "robot_pos":    robot_pos,
        "tube_coeffs":  tube_coeffs,
        "track_data":   track_data,
        "pcl":          pcl,
        "local_window": local_window_plot,
    }

# ==============================================================================
#  Plot
# ==============================================================================

def plot_topdown(snap):
    track_data   = snap["track_data"]
    local_window = snap["local_window"]
    tube_coeffs  = snap["tube_coeffs"]
    pcl          = snap["pcl"]
    demo_pcl     = snap["demo_pcl"]
    robot_pos    = snap["robot_pos"]
    s_now        = snap["s_now"]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    # Point cloud
    if len(pcl) > 0:
        ax.scatter(
            pcl[:, 0], pcl[:, 1],
            c=pcl[:, 2], cmap=C_PCL,
            s=6, alpha=0.2, zorder=1
        )

    # Tube scatter points
    corridor_pts = get_corridor_pts(ax, local_window, tube_coeffs)
    ax.scatter(
        corridor_pts[:, 0], corridor_pts[:, 1],
        s=3, alpha=0.2, color=C_TUBE, zorder=2
    )

    # Full track
    ax.plot(
        track_data["x"], track_data["y"],
        color=C_TRACK, lw=3.0, alpha=0.8, zorder=3
    )

    # Local window
    ax.plot(
        local_window["x"], local_window["y"],
        color="#0000FF", lw=4.0, linestyle="dashed", alpha=0.8, zorder=4
    )

    # Obstacle projection points
    projected_s_vals = []
    if len(demo_pcl) > 0:
        projected_s_vals = draw_obstacle_projections_topdown(
            ax, demo_pcl, local_window, max_dist=2.0
        )

    # Ellipse slices
    # if len(projected_s_vals) > 0:
    #     draw_ellipses_topdown(
    #         ax, local_window, tube_coeffs,
    #         s_query=projected_s_vals,
    #         n_ellipse_pts=ELLIPSE_SLICES
    #     )
    # else:
    #     fallback_s = np.linspace(
    #         local_window["s"][0], local_window["s"][-1], N_ELLIPSE_DRAWS
    #     )
    #     draw_ellipses_topdown(
    #         ax, local_window, tube_coeffs,
    #         s_query=fallback_s,
    #         n_ellipse_pts=ELLIPSE_SLICES
    #     )

    # Center swept-volume line
    draw_ellipse_center_line(ax, local_window, tube_coeffs, n_pts=N_CENTER_PTS)

    # RMF frames back in
    draw_rmf_frames_topdown(
        ax, local_window,
        n_frames=FRAME_SLICES,
        length=FRAME_LENGTH
    )

    # Robot
    # ax.scatter(
    #     robot_pos[0], robot_pos[1],
    #     s=150, color=C_ROBOT, zorder=8
    # )

    ax.set_xlim(-2, 2)
    ax.set_ylim(s_now - 2, s_now + 6)

    plt.tight_layout()
    save_name = f"tube_topdown_s{s_now:.1f}.png"
    plt.savefig(save_name, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved {save_name}")
    plt.show()

# ==============================================================================
#  Main
# ==============================================================================

def main():
    env = VolaDroneEnv(TRACK, normalize_obs=False, pcl_density=200)

    demo_pcl = load_pcl_from_env(
        f"../../resources/envs/{TRACK}.yaml", 0.1
    )

    snap = extract_snapshot(env, n_knots_plot=N_CENTER_PTS)
    snap["demo_pcl"] = demo_pcl
    plot_topdown(snap)

if __name__ == "__main__":
    main()
