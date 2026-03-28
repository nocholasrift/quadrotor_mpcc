import os
import sys

mpcc_path = os.path.abspath("../quadrotor_mpcc")
if mpcc_path not in sys.path:
    sys.path.append(mpcc_path)

from common import *
from tube_gen import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ==============================================================================
#  CONFIGURATION — toggle here
# ==============================================================================

# FRAME_MODE = "RMF"       # "RMF" or "FS"
FRAME_MODE = "RMF"       # "RMF" or "FS"

y_mult = 1.0
z_mult = 1.0

N_CURVE  = 300          # dense curve points
N_FRAMES = 20           # sparse frame triad locations
N_TUBE   = 300          # dense tube ring locations

# Ellipse semi-axes — unequal ratio makes the flip visually dramatic
TUBE_OFFSET_E1   = 0.1
TUBE_OFFSET_E2   = 0.1
TUBE_RADIUS_E1   = 0.5  # semi-axis along e1 / N  (tall dimension)
TUBE_RADIUS_E2   = 0.50  # semi-axis along e2 / B  (flat dimension)
TUBE_N_ANGLES    = 60
TUBE_ALPHA       = 0.3
TUBE_N_LONG_LINES = 12

FRAME_LENGTH     = 0.3
LABEL_OFFSET     = 1.45
if FRAME_MODE == "FS":
    LABEL_NUDGE = {
        "T":  np.array([0.0,  -0.1,  0.0]),
        "N":  np.array([0.0,  0.05, -0.15]),  # lower N a smidge
        "B":  np.array([0.0,  0.0,  0.1]),
    }
else:
    LABEL_NUDGE = {
        "T":  np.array([0.0,  -0.15,  0.0]),
        "e1":  np.array([0.0,  -0.1, 0.1]),  # lower N a smidge
        "e2":  np.array([0.0,  0.0,  0.0]),
    }

# LABEL_INDICES    = None  # None = auto (first, mid, last); or e.g. [0, 5, 11]
LABEL_INDICES    = [N_FRAMES-1]

# Window centered on the inflection at t = pi
HALF_WINDOW = 1.3        # show +/- this many radians around pi

VIEW_ELEV = -8
VIEW_AZIM = -97

# ==============================================================================
#  Palette
# ==============================================================================

C_CURVE = "#1C1C2E"
C_TUBE  = "#4A90D9"
C_T     = "#E05252"
C_E1    = "#52A852"
C_E2    = "#5271E0"
BG      = "white"

# ==============================================================================
#  Curve definition
# ==============================================================================

def curve(t):
    x = t
    y = y_mult * np.sin(t)
    z = z_mult * np.sin(2 * t)
    return np.stack([x, y, z], axis=-1)


def curve_d1(t):
    dx = np.ones_like(t)
    dy = y_mult * np.cos(t)
    dz = 2 * z_mult * np.cos(2 * t)
    return np.stack([dx, dy, dz], axis=-1)


def curve_d2(t):
    dx = np.zeros_like(t)
    dy = -y_mult * np.sin(t)
    dz = -4 * z_mult * np.sin(2 * t)
    return np.stack([dx, dy, dz], axis=-1)


def normalize(v):
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norms + 1e-12)

# ==============================================================================
#  Frame computation
# ==============================================================================

def get_rmf_basis(t_vals):
    vels = curve_d1(t_vals)
    vxref, vyref, vzref = interpolLUT(vels[:, 0], vels[:, 1], vels[:, 2], t_vals)
    T, e1, e2 = getRMFBasis(vxref, vyref, vzref, t_vals, np.array([1.0, 1.0, 1.0]))
    return T, e1, e2


def get_fs_basis(t_vals):
    """
    Frenet-Serret frame. N flips sign at the inflection point (t=pi)
    where curvature passes through zero — that discontinuity is the
    whole point of this figure.
    """
    d1 = curve_d1(t_vals)
    d2 = curve_d2(t_vals)
    T  = normalize(d1)
    d2_perp = d2 - np.sum(d2 * T, axis=-1, keepdims=True) * T
    N  = normalize(d2_perp)   # flips sign at inflection!
    B  = normalize(np.cross(T, N))
    return T, N, B


def get_basis(t_vals, mode):
    if mode == "RMF":
        return get_rmf_basis(t_vals)
    elif mode == "FS":
        return get_fs_basis(t_vals)
    else:
        raise ValueError(f"Unknown FRAME_MODE '{mode}'. Use 'RMF' or 'FS'.")

# ==============================================================================
#  Drawing helpers
# ==============================================================================

def draw_tube_mesh(ax, p, e1, e2, t_vals,
                   r1=TUBE_RADIUS_E1,
                   r2=TUBE_RADIUS_E2,
                   n_angles=TUBE_N_ANGLES,
                   alpha=TUBE_ALPHA,
                   color=C_TUBE):
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)

    # Index of the two rings closest to the inflection on each side
    before_inf = np.where(t_vals < np.pi)[0][-1]   # last ring before pi
    after_inf  = np.where(t_vals > np.pi)[0][0]    # first ring after pi

    rings = []
    for i in range(len(p)):
        center = (p[i]
                  + TUBE_OFFSET_E1 * e1[i]
                  + TUBE_OFFSET_E2 * e2[i])
        ring = (center
                + r1 * cos_a[:, None] * e1[i]
                + r2 * sin_a[:, None] * e2[i])
        rings.append(ring)
        closed = np.vstack([ring, ring[0]])

        # Highlight the two rings bracketing the inflection (FS only)
        if i in (before_inf, after_inf):
            ax.plot(closed[:, 0], closed[:, 1], closed[:, 2],
                    color="black", alpha=0.5, lw=2.0)
        else:
            ax.plot(closed[:, 0], closed[:, 1], closed[:, 2],
                    color=color, alpha=alpha, lw=0.6)

    rings = np.array(rings)
    step  = max(1, n_angles // TUBE_N_LONG_LINES)
    for j in range(0, n_angles, step):
        ax.plot(rings[:, j, 0], rings[:, j, 1], rings[:, j, 2],
                color=color, alpha=alpha * 0.7, lw=0.5)

def draw_frames(ax, p, T, e1, e2, mode, length=FRAME_LENGTH):
    if mode == "RMF":
        labels = [r"$\mathbf{T}$", r"$\mathbf{e}_1$", r"$\mathbf{e}_2$"]
    else:
        labels = [r"$\mathbf{T}$", r"$\mathbf{N}$", r"$\mathbf{B}$"]

    n = len(p)
    labeled = set(LABEL_INDICES) if LABEL_INDICES is not None else {0, n // 2, n - 1}
    nudges = list(LABEL_NUDGE.values())  # order matches [T, e1/N, e2/B]

    for i in range(n):
        for vec, color in zip([T, e1, e2], [C_T, C_E1, C_E2]):
            ax.quiver(*p[i], *vec[i],
                      color=color, length=length,
                      normalize=True, linewidth=3.0,
                      arrow_length_ratio=0.28)


        # if i in labeled:
        #     for vec, color, lbl, nudge in zip([T, e1, e2],
        #                                 [C_T, C_E1, C_E2], labels, nudges):
        #         tip = p[i] + normalize(vec[i:i+1])[0] * length * LABEL_OFFSET + nudge
        #         ax.text(*tip, lbl, color=color,
        #                 fontsize=20, ha="center", va="center",
        #                 fontweight="bold")


def set_equal_axes(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    mid  = (mins + maxs) / 2
    ranges = maxs - mins

    # Give each axis just enough room for the data + small padding
    pad = 0.15
    half = ranges.max() / 2 * (1 + pad)  # still equal scale
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    # Match the visual box proportions to the actual data ranges
    # so matplotlib doesn't pad with whitespace to make a cube
    ax.set_box_aspect(ranges / ranges.max())


def style_ax(ax, title):
    ax.set_facecolor(BG)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#dddddd")
    ax.xaxis._axinfo["grid"]["color"] = "#eeeeee"
    ax.yaxis._axinfo["grid"]["color"] = "#eeeeee"
    ax.zaxis._axinfo["grid"]["color"] = "#eeeeee"
    ax.tick_params(labelsize=7, pad=1)
    ax.set_xlabel("$x$", fontsize=10, labelpad=4)
    ax.set_ylabel("$y$", fontsize=10, labelpad=4)
    ax.set_zlabel("$z$", fontsize=10, labelpad=4)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


def make_legend(ax, mode):
    n_label = (r"$\mathbf{e}_1$ — normal" if mode == "RMF"
               else r"$\mathbf{N}$ — principal normal")
    b_label = (r"$\mathbf{e}_2$ — binormal" if mode == "RMF"
               else r"$\mathbf{B}$ — binormal")
    patches = [
        mpatches.Patch(color=C_T,    label=r"$\mathbf{T}$ — tangent"),
        mpatches.Patch(color=C_E1,   label=n_label),
        mpatches.Patch(color=C_E2,   label=b_label),
        mpatches.Patch(color=C_TUBE, alpha=0.6, label="Swept elliptical tube"),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='darkorange', markersize=7,
                   label="Inflection point"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left",
              framealpha=0.92, edgecolor="#cccccc")

# ==============================================================================
#  Main
# ==============================================================================

def main():
    # Centered on the inflection at t = pi
    t_start = np.pi - HALF_WINDOW
    t_end   = np.pi + HALF_WINDOW

    # Dense centerline
    t   = np.linspace(t_start, t_end, N_CURVE)
    pts = curve(t)

    # Sparse — frame triads
    t_sparse  = np.linspace(t_start, t_end, N_FRAMES)
    p         = curve(t_sparse)
    T, e1, e2 = get_basis(t_sparse, FRAME_MODE)

    # Dense — tube rings
    t_tube              = np.linspace(t_start, t_end, N_TUBE)
    p_tube              = curve(t_tube)
    _, e1_tube, e2_tube = get_basis(t_tube, FRAME_MODE)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5, 7), facecolor=BG)

    title_str = (r"Rotation-Minimizing Frame (Bishop) along an $S$-curve"
                 if FRAME_MODE == "RMF"
                 else r"Frenet-Serret Frame along an $S$-curve")
    # fig.suptitle(title_str, fontsize=13, fontweight="bold", y=0.98)

    ax = fig.add_subplot(111, projection="3d")
    subtitle = ("RMF — elliptical tube twists smoothly, no discontinuity"
                if FRAME_MODE == "RMF"
                else r"Frenet-Serret — ellipse snaps $180°$ at inflection point")
    # style_ax(ax, subtitle)

    # Centerline
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            color=C_CURVE, lw=2.2, zorder=5)

    # Inflection marker — right at t = pi
    idx_inf = np.argmin(np.abs(t - np.pi))
    p_inf   = pts[idx_inf]
    ax.scatter(*p_inf, s=80, color="darkorange", zorder=6, depthshade=False)

    # Dense elliptical tube
    draw_tube_mesh(ax, p_tube, e1_tube, e2_tube, t_tube)

    # Sparse frames
    draw_frames(ax, p, T, e1, e2, FRAME_MODE)

    set_equal_axes(ax, pts)
    # make_legend(ax, FRAME_MODE)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM, roll = 0)
    ax.set_axis_off()

    plt.tight_layout()

    out_name = f"frame_viz_{FRAME_MODE.lower()}"
    # plt.savefig(f"{out_name}.pdf", dpi=300, bbox_inches="tight", facecolor=BG)
    # plt.savefig(f"{out_name}.png", dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out_name}.pdf / .png")
    plt.show()


if __name__ == "__main__":
    main()
