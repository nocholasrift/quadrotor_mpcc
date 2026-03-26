import os
import sys
mpcc_path = os.path.abspath("../quadrotor_mpcc")
if mpcc_path not in sys.path:
    sys.path.append(mpcc_path)

from common import *
from tube_gen import *
from mpc_env import VolaDroneEnv

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

TRACK   = "short_line"
S_VALUE = 2.0          # step the env forward until robot reaches this s

FRAME_LENGTH = 0.6
FRAME_SLICES = 3
ELLIPSE_SLICES = 50

VIEW_ELEV = 46
VIEW_AZIM = -55

BG      = "white"
C_TRACK = "#1C1C2E"
C_TUBE  = "#4A90D9"
C_PCL   = "viridis"
C_ROBOT = "darkorange"

# Match frame_viz.py palette exactly
C_T  = "#E05252"   # tangent   — red
C_E1 = "#52A852"   # e1        — green
C_E2 = "#5271E0"   # e2        — blue

def draw_obstacle_projections(ax, pcl, local_window, plane_size=0.6, alpha=0.25):
    """
    For each obstacle point in the pcl, find its projection onto the curve
    and draw:
      - dashed line from obstacle to curve
      - shaded normal plane (e1-e2) at the projected point
      - the frame axes at that point
    """
    s_local = local_window["s"]

    # Build curve positions for nearest-point search
    curve_pts = np.stack([
        local_window["x"],
        local_window["y"],
        local_window["z"],
    ], axis=1)

    # Interpolation grids — use a fine grid for accurate projection
    s_fine  = np.linspace(s_local[0], s_local[-1], 500)
    cx = np.interp(s_fine, s_local, local_window["x"])
    cy = np.interp(s_fine, s_local, local_window["y"])
    cz = np.interp(s_fine, s_local, local_window["z"])
    curve_fine = np.stack([cx, cy, cz], axis=1)

    vx  = np.interp(s_fine, s_local, local_window["vx"])
    vy  = np.interp(s_fine, s_local, local_window["vy"])
    vz  = np.interp(s_fine, s_local, local_window["vz"])
    e1x = np.interp(s_fine, s_local, local_window["e1x"])
    e1y = np.interp(s_fine, s_local, local_window["e1y"])
    e1z = np.interp(s_fine, s_local, local_window["e1z"])

    T  = np.stack([vx, vy, vz], axis=1)
    T  /= np.linalg.norm(T,  axis=1, keepdims=True) + 1e-12
    e1 = np.stack([e1x, e1y, e1z], axis=1)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = np.cross(T, e1)

    # Filter pcl to points inside the local window s range
    # (only plot projections for obstacles in this window)
    s_min = s_local[0]
    s_max = s_local[-1]

    # Use cKDTree for fast nearest-point on curve
    from scipy.spatial import cKDTree
    tree = cKDTree(curve_fine)

    for obs_pt in pcl:
        dist, idx = tree.query(obs_pt)

        # Skip if projected point is outside the window or too far
        if s_fine[idx] < s_min or s_fine[idx] > s_max:
            continue
        if dist > 2.0:   # tune: max distance to bother drawing
            continue

        proj_pt = curve_fine[idx]
        t_i  = T[idx]
        e1_i = e1[idx]
        e2_i = e2[idx]

        # --- Dashed projection line: obstacle → curve ---
        # ax.plot([obs_pt[0], proj_pt[0]],
        #         [obs_pt[1], proj_pt[1]],
        #         [obs_pt[2], proj_pt[2]],
        #         color="gray", lw=1.0, ls="--", alpha=0.7, zorder=3)

        # --- Normal plane (e1-e2 plane) as a shaded quad ---
        # Four corners of the plane patch centered at proj_pt
        corners = np.array([
            proj_pt + plane_size * ( e1_i + e2_i),
            proj_pt + plane_size * (-e1_i + e2_i),
            proj_pt + plane_size * (-e1_i - e2_i),
            proj_pt + plane_size * ( e1_i - e2_i),
        ])
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        poly = Poly3DCollection([corners],
                                alpha=alpha,
                                facecolor=C_TUBE,
                                edgecolor=C_TUBE,
                                linewidth=0.5)
        ax.add_collection3d(poly)

        # --- Frame axes at projected point ---
        # ax.quiver(*proj_pt, *e1_i, color=C_E1, length=FRAME_LENGTH,
        #           normalize=True, linewidth=1.5, arrow_length_ratio=0.3)
        # ax.quiver(*proj_pt, *e2_i, color=C_E2, length=FRAME_LENGTH,
        #           normalize=True, linewidth=1.5, arrow_length_ratio=0.3)
        # ax.quiver(*proj_pt, *t_i,  color=C_T,  length=FRAME_LENGTH,
        #           normalize=True, linewidth=1.5, arrow_length_ratio=0.3)

        # --- Dot at projected point on curve ---
        ax.scatter(*proj_pt, s=40, color="black", zorder=6, depthshade=False)

def draw_rmf_frames(ax, local_window, n_frames=10, length=0.5):
    """Draw RMF triads at sparse points along the local window."""
    s_local = local_window["s"]
    s_query = np.linspace(s_local[0], s_local[-1], n_frames)

    # Interpolate position and frame vectors
    pos = np.stack([
        np.interp(s_query, s_local, local_window["x"]),
        np.interp(s_query, s_local, local_window["y"]),
        np.interp(s_query, s_local, local_window["z"]),
    ], axis=1)

    vx  = np.interp(s_query, s_local, local_window["vx"])
    vy  = np.interp(s_query, s_local, local_window["vy"])
    vz  = np.interp(s_query, s_local, local_window["vz"])
    e1x = np.interp(s_query, s_local, local_window["e1x"])
    e1y = np.interp(s_query, s_local, local_window["e1y"])
    e1z = np.interp(s_query, s_local, local_window["e1z"])

    T  = np.stack([vx, vy, vz], axis=1)
    T  /= np.linalg.norm(T,  axis=1, keepdims=True) + 1e-12
    e1 = np.stack([e1x, e1y, e1z], axis=1)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = np.cross(T, e1)

    for i in range(n_frames):
        for vec, color in zip([T, e1, e2], [C_T, C_E1, C_E2]):
            ax.quiver(*pos[i], *vec[i],
                      color=color, length=length,
                      normalize=True, linewidth=4.0,
                      arrow_length_ratio=0.4)

# ==============================================================================
#  Extract data from env
# ==============================================================================

def extract_snapshot(env, n_knots_plot=10):
    """
    Step the env (no action) until s >= S_VALUE, then pull out
    everything we need for plotting.
    """
    env.reset()
    while env.state[10] < S_VALUE:
        env.step()

    s_now        = env.state[10]
    robot_pos    = env.state[:3]
    tube_coeffs  = env.tube_coeffs.copy()
    track_data   = env.track_data
    pcl          = env.pcl

    local_window = get_local_window_params(
        track_data, s_now, n_knots,
        window_dist=env.track_horizon_window
    )

    local_window_plot = get_local_window_params(
        track_data, s_now, n_knots_plot,
        window_dist=env.track_horizon_window
    )

    return {
        "s_now":       s_now,
        "robot_pos":   robot_pos,
        "tube_coeffs": tube_coeffs,
        "track_data":  track_data,
        "pcl":         pcl,
        "local_window": local_window_plot,
    }

# ==============================================================================
#  Plot
# ==============================================================================

def plot_snapshot(snap):
    track_data   = snap["track_data"]
    local_window = snap["local_window"]
    tube_coeffs  = snap["tube_coeffs"]
    pcl          = snap["pcl"]
    robot_pos    = snap["robot_pos"]
    s_now        = snap["s_now"]

    fig = plt.figure(figsize=(12, 8), facecolor=BG)
    # fig.suptitle(f"Tube Generation  |  s = {s_now:.2f} / {track_data['L']:.2f}",
    #              fontsize=13, fontweight="bold")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    ax.set_axis_off()


    # Point cloud
    if len(pcl) > 0:
        ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2],
                   c=pcl[:,2], cmap=C_PCL, s=6, alpha=0.2)

        # Normal planes at projected obstacle positions
        # draw_obstacle_projections(ax, pcl, local_window,
        #                           plane_size=1.0, alpha=0.2)

    # Tube — reuse corridor_pts logic directly from mpc_env
    corridor_pts = get_corridor_pts(ax, local_window, tube_coeffs)
    ax.scatter(corridor_pts[:, 0], corridor_pts[:, 1], corridor_pts[:, 2],
               s=3, alpha=0.2, color=C_TUBE)


    # Full track
    ax.plot(track_data["x"], track_data["y"], track_data["z"],
            color=C_TRACK, lw=3.0, alpha=0.8, label="Track")

    # Local window
    ax.plot(local_window["x"], local_window["y"], local_window["z"],
            color="#0000FF", lw=4.0, label="Local window", linestyle='dashed', alpha=0.8)

    # Robot
    ax.scatter(*robot_pos, s=120, color=C_ROBOT,
                depthshade=False, label=f"Robot s={s_now:.2f}")

    draw_rmf_frames(ax, local_window, n_frames=FRAME_SLICES, length=FRAME_LENGTH)

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    # Limits — tune these
    ax.set_xlim(-2, 2)
    ax.set_ylim(s_now - 2, s_now + 6)
    ax.set_zlim(0, 3)
    ax.set_box_aspect([4, 8, 3])

    # ax.legend(fontsize=9, loc="upper left",
    #           framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    plt.savefig(f"tube_snapshot_s{s_now:.1f}.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    print(f"Saved tube_snapshot_s{s_now:.1f}.png")
    plt.show()

# ==============================================================================
#  Main
# ==============================================================================

def main():
    env  = VolaDroneEnv(TRACK, normalize_obs=False,pcl_density=200)
    snap = extract_snapshot(env, n_knots_plot=ELLIPSE_SLICES)
    plot_snapshot(snap)

if __name__ == "__main__":
    main()
