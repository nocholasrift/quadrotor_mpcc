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
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
C_ELLIPSE = "#FF5A5F"
C_PCL   = "viridis"
C_ROBOT = "darkorange"

# Match frame_viz.py palette exactly
C_T  = "#E05252"   # tangent   — red
C_E1 = "#52A852"   # e1        — green
C_E2 = "#5271E0"   # e2        — blue

# ==============================================================================
#  DRAWING FUNCTIONS
# ==============================================================================

def draw_obstacle_projections(ax, pcl, local_window, plane_size=0.6, alpha=0.25):
    s_local = local_window["s"]

    curve_pts = np.stack([
        local_window["x"], local_window["y"], local_window["z"],
    ], axis=1)

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

    s_min = s_local[0]
    s_max = s_local[-1]

    from scipy.spatial import cKDTree
    tree = cKDTree(curve_fine)

    projected_s_vals = []

    for obs_pt in pcl:
        dist, idx = tree.query(obs_pt)

        if s_fine[idx] < s_min or s_fine[idx] > s_max:
            continue
        if dist > 2.0:
            continue

        proj_pt = curve_fine[idx]
        e1_i = e1[idx]
        e2_i = e2[idx]

        corners = np.array([
            proj_pt + plane_size * ( e1_i + e2_i),
            proj_pt + plane_size * (-e1_i + e2_i),
            proj_pt + plane_size * (-e1_i - e2_i),
            proj_pt + plane_size * ( e1_i - e2_i),
        ])
        poly = Poly3DCollection([corners], alpha=alpha, facecolor=C_TUBE, edgecolor=C_TUBE, linewidth=0.5)
        ax.add_collection3d(poly)
        ax.scatter(*proj_pt, s=40, color="black", zorder=6, depthshade=False)
        
        # Save the s value where this projection occurred
        projected_s_vals.append(s_fine[idx])

    # Return unique s values to avoid drawing duplicate ellipses at the exact same spot
    return np.unique(projected_s_vals)

def draw_rmf_frames(ax, local_window, n_frames=10, length=0.5):
    s_local = local_window["s"]
    s_query = np.linspace(s_local[0], s_local[-1], n_frames)

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
            ax.quiver(*pos[i], *vec[i], color=color, length=length, normalize=True, linewidth=4.0, arrow_length_ratio=0.4)

def draw_tube_ellipses(ax, local_window, tube_coeffs, s_query, n_ellipse_pts=50):
    """Draws the 2D ellipse cross-sections at the specified s_query locations."""
    if len(s_query) == 0:
        return

    s_local = local_window["s"]
    n_frames = len(s_query)

    # Interpolate position and frame vectors
    pos = np.stack([
        np.interp(s_query, s_local, local_window["x"]),
        np.interp(s_query, s_local, local_window["y"]),
        np.interp(s_query, s_local, local_window["z"]),
    ], axis=1)

    e1x = np.interp(s_query, s_local, local_window["e1x"])
    e1y = np.interp(s_query, s_local, local_window["e1y"])
    e1z = np.interp(s_query, s_local, local_window["e1z"])
    vx  = np.interp(s_query, s_local, local_window["vx"])
    vy  = np.interp(s_query, s_local, local_window["vy"])
    vz  = np.interp(s_query, s_local, local_window["vz"])

    T  = np.stack([vx, vy, vz], axis=1)
    T  /= np.linalg.norm(T,  axis=1, keepdims=True) + 1e-12
    e1 = np.stack([e1x, e1y, e1z], axis=1)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = np.cross(T, e1)

    # Evaluate tube coefficients at the slices
    xi = (s_query - s_local[0]) / (s_local[-1] - s_local[0] + 1e-12)
    poly_deg = tube_coeffs.shape[1] - 1
    Phi = get_cheby_basis(xi, poly_deg)

    a_vals = Phi @ tube_coeffs[0]
    b_vals = Phi @ tube_coeffs[1]
    c_vals = Phi @ tube_coeffs[2]
    d_vals = Phi @ tube_coeffs[3]

    thetas = np.linspace(0, 2 * np.pi, n_ellipse_pts)

    for i in range(n_frames):
        P = np.array([[a_vals[i], 0], [0, b_vals[i]]])
        pp = np.array([c_vals[i], d_vals[i]])

        try:
            pc, width, height, angle = get_ellipse_parameters(P, pp)
        except Exception as e:
            print(f"Skipping ellipse {i} due to math error: {e}")
            continue

        # Get 2D points and shift by center offset
        ellipse_2d = get_ellipse_points(width, height, angle, thetas).T + pc

        w1 = ellipse_2d[:, 0]
        w2 = ellipse_2d[:, 1]

        # Transform 2D (w1, w2) points into 3D space using the local frame
        ellipse_3d = pos[i] + w1[:, None] * e1[i] + w2[:, None] * e2[i]

        # Draw the thick boundary line
        ax.plot(ellipse_3d[:, 0], ellipse_3d[:, 1], ellipse_3d[:, 2], 
                color=C_ELLIPSE, lw=2.5, alpha=0.9, zorder=4)

        # Add a translucent fill to make it look like a solid cross-section
        poly = Poly3DCollection([ellipse_3d], alpha=0.15, facecolor=C_ELLIPSE, zorder=3)
        ax.add_collection3d(poly)

# ==============================================================================
#  Extract data from env
# ==============================================================================

def extract_snapshot(env, n_knots_plot=10):
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
    demo_pcl     = snap["demo_pcl"]
    robot_pos    = snap["robot_pos"]
    s_now        = snap["s_now"]

    fig = plt.figure(figsize=(12, 8), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    ax.set_axis_off()

    projected_s_vals = []

    # Point cloud
    if len(pcl) > 0:
        ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2],
                   c=pcl[:,2], cmap=C_PCL, s=6, alpha=0.05)

        # Normal planes at projected obstacle positions
        projected_s_vals = draw_obstacle_projections(ax, demo_pcl, local_window,
                                                     plane_size=1.5, alpha=0.2)

    # Tube scatter points
    # corridor_pts = get_corridor_pts(ax, local_window, tube_coeffs)
    # ax.scatter(corridor_pts[:, 0], corridor_pts[:, 1], corridor_pts[:, 2],
    #            s=3, alpha=0.2, color=C_TUBE)
    
    # Full track
    ax.plot(track_data["x"], track_data["y"], track_data["z"],
            color=C_TRACK, lw=3.0, alpha=0.8, label="Track")

    # Local window
    ax.plot(local_window["x"], local_window["y"], local_window["z"],
            color="#0000FF", lw=4.0, label="Local window", linestyle='dashed', alpha=0.8)

    # Draw RMF frames
    draw_rmf_frames(ax, local_window, n_frames=FRAME_SLICES, length=FRAME_LENGTH)
    
    # # Draw Ellipse Cross-Sections at the exact obstacle projection locations
    # if len(projected_s_vals) > 0:
    #     draw_tube_ellipses(ax, local_window, tube_coeffs, s_query=projected_s_vals, n_ellipse_pts=ELLIPSE_SLICES)
    # else:
    #     # Fallback if no obstacles: draw evenly spaced ellipses
    #     fallback_s = np.linspace(local_window["s"][0], local_window["s"][-1], FRAME_SLICES)
    #     draw_tube_ellipses(ax, local_window, tube_coeffs, s_query=fallback_s, n_ellipse_pts=ELLIPSE_SLICES)

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    ax.set_xlim(-2, 2)
    ax.set_ylim(s_now - 2, s_now + 6)
    ax.set_zlim(0, 3)
    ax.set_box_aspect([4, 8, 3])

    plt.tight_layout()
    plt.savefig(f"tube_snapshot_s{s_now:.1f}.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    print(f"Saved tube_snapshot_s{s_now:.1f}.png")
    plt.show()

# ==============================================================================
#  Main
# ==============================================================================

def main():
    env  = VolaDroneEnv(TRACK, normalize_obs=False, pcl_density=200)

    demo_pcl = load_pcl_from_env(
        f"../../resources/envs/{TRACK}.yaml", 0.1
    )

    snap = extract_snapshot(env, n_knots_plot=ELLIPSE_SLICES)
    snap["demo_pcl"] = demo_pcl
    plot_snapshot(snap)

if __name__ == "__main__":
    main()
