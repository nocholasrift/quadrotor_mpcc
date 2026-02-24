#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# reference : "Towards Time-optimal Tunnel-following for Quadrotors", Jon Arrizabalaga et al.

import yaml
import numpy as np
import casadi as ca
import os
from pathlib import Path
from typing import Union
from tube_gen import *

"""Global variables"""

# track="crazyflie_arclen_traj.txt"


def getTrack(track):
    # script_dir = Path(__file__).resolve().parent
    # track_dir = script_dir.parents[1] / "resources" / "tracks"
    fname = track + ".txt"
    track_dir = Path(os.environ["QUAD_RACE_RESOURCES"])
    track_file = track_dir / "trajectory" / fname

    array = np.loadtxt(track_file, skiprows=1)
    sref = array[0:, 0]
    xref = array[0:, 1]
    yref = array[0:, 2]
    zref = array[0:, 3]
    vxref = array[0:, 4]
    vyref = array[0:, 5]
    vzref = array[0:, 6]

    return sref, xref, yref, zref, vxref, vyref, vzref


def load_gates(track):
    # Resolve path
    fname = track + ".yaml"
    resource_dir = Path(os.environ["QUAD_RACE_RESOURCES"])
    track_file = resource_dir / "racetrack" / fname

    # Load YAML
    with open(track_file, "r") as f:
        config = yaml.safe_load(f)

    gates = []

    # Respect the defined order
    for gate_name in config["orders"]:
        gate_cfg = config[gate_name]

        gate = {
            "name": gate_name,
            "type": gate_cfg["type"],
            "position": gate_cfg["position"],  # [x, y, z]
            "rpy": gate_cfg["rpy"],  # degrees
            "width": gate_cfg["width"],
            "height": gate_cfg["height"],
            "marginW": gate_cfg["marginW"],
            "marginH": gate_cfg["marginH"],
            "length": gate_cfg["length"],
            "midpoints": gate_cfg["midpoints"],
            "stationary": gate_cfg["stationary"],
        }

        gates.append(gate)

    return gates


def rpy_to_rot(rpy_deg):
    roll, pitch, yaw = np.deg2rad(rpy_deg)

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )

    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    return Rz @ Ry @ Rx


def draw_gates(gate_data, ax):

    for gate in gate_data:

        width = gate["width"]
        height = gate["height"]
        pos = np.array(gate["position"])
        R = rpy_to_rot(gate["rpy"])

        # Rectangle in local frame (centered at origin)
        corners = np.array(
            [
                [-width / 2, -height / 2, 0],
                [width / 2, -height / 2, 0],
                [width / 2, height / 2, 0],
                [-width / 2, height / 2, 0],
                [-width / 2, -height / 2, 0],  # close loop
            ]
        )

        # Rotate + translate
        world_corners = (R @ corners.T).T + pos

        ax.plot(
            world_corners[:, 0],
            world_corners[:, 1],
            world_corners[:, 2],
            "b-",
            linewidth=2,
        )


def interpolLUT(x, y, z, s):

    if isinstance(s, np.ndarray):
        s_grid = s.tolist()
    else:
        s_grid = s

    # Data must be lists of floats
    x_data = x.tolist() if isinstance(x, np.ndarray) else x
    y_data = y.tolist() if isinstance(y, np.ndarray) else y
    z_data = z.tolist() if isinstance(z, np.ndarray) else z

    x_ref_curve = ca.interpolant("x_ref", "bspline", [s_grid], x_data)
    y_ref_curve = ca.interpolant("y_ref", "bspline", [s_grid], y_data)
    z_ref_curve = ca.interpolant("z_ref", "bspline", [s_grid], z_data)

    return x_ref_curve, y_ref_curve, z_ref_curve


def getFrenSerretBasis(x_func, y_func, z_func, s_vals):
    # Symbolic variable
    s_sym = ca.SX.sym("s")

    # symbolic expressions
    x_s = x_func(s_sym)
    y_s = y_func(s_sym)
    z_s = z_func(s_sym)

    # first derivatives
    # dx_ds = ca.jacobian(x_s, s_sym)
    # dy_ds = ca.jacobian(y_s, s_sym)
    # dz_ds = ca.jacobian(z_s, s_sym)

    # second derivatives
    # d2x_ds2 = ca.jacobian(dx_ds, s_sym)
    # d2y_ds2 = ca.jacobian(dy_ds, s_sym)
    # d2z_ds2 = ca.jacobian(dz_ds, s_sym)

    [d2x_ds2, dx_ds] = ca.hessian(x_s, s_sym)
    [d2y_ds2, dy_ds] = ca.hessian(y_s, s_sym)
    [d2z_ds2, dz_ds] = ca.hessian(z_s, s_sym)

    # Create a CasADi function to evaluate at numeric s
    fs = ca.Function(
        "frenet_basis", [s_sym], [dx_ds, dy_ds, dz_ds, d2x_ds2, d2y_ds2, d2z_ds2]
    )

    N = len(s_vals)
    ts = np.zeros((N, 3))
    ns = np.zeros((N, 3))
    bs = np.zeros((N, 3))

    for i, s_val in enumerate(s_vals):
        dx_val, dy_val, dz_val, d2x_val, d2y_val, d2z_val = fs(s_val)

        t = np.array([dx_val, dy_val, dz_val]).flatten()
        dt_ds = np.array([d2x_val, d2y_val, d2z_val]).flatten()

        # normalize tangent
        t_norm = t

        # print(i, "\t", dt_ds)
        # normal vector
        n = dt_ds
        n /= np.linalg.norm(n) + 1e-8

        # binormal vector
        b = np.cross(t_norm, n)

        ts[i, :] = t_norm
        ns[i, :] = n
        bs[i, :] = b

    return ts, ns, bs

def casadi_chebyshev_basis(xi, degree):
    T = [0] * (degree + 1)
    x = 2 * xi - 1

    T[0] = 1
    if degree >= 1:

        T[1] = x

    for k in range(2, degree+1):
        T[k] = 2 * x * T[k-1] - T[k-2]

    return T

def getRMFBasis(vx, vy, vz, s):

    e1 = np.zeros((len(s), 3))
    e2 = np.zeros((len(s), 3))
    T = np.zeros((len(s), 3))
    ref = np.array([0.0, 0.0, 1.0])

    # initialize first frame
    T[0] = np.array([vx(s[0]), vy(s[0]), vz(s[0])]).reshape(
        3,
    )
    T[0] /= np.linalg.norm(T[0])
    if abs(1 - np.dot(T[0], ref)) < 1e-8:
        ref = np.array([0.0, 1.0, 0.0])

    # T is nearly 1 in magnitude already
    e1[0] = np.cross(T[0], ref)
    e1[0] /= np.linalg.norm(e1[0])
    e2[0] = np.cross(T[0], e1[0])

    # parallel transport
    for ind, s_val in enumerate(s[1:]):
        prev_T = T[ind]
        T[ind + 1] = np.array([vx(s_val), vy(s_val), vz(s_val)]).reshape(
            3,
        )
        T[ind + 1] /= np.linalg.norm(T[ind + 1])
        v = np.cross(prev_T, T[ind + 1])
        c = np.dot(prev_T, T[ind + 1])

        # straight line
        if np.linalg.norm(v) < 1e-8:
            e1[ind + 1] = e1[ind]
        else:
            v_orig = v
            v /= np.linalg.norm(v)
            theta = np.arctan2(np.linalg.norm(np.cross(prev_T, T[ind + 1])), c)

            # rotation
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

            e1[ind + 1] = R @ e1[ind]

        e2[ind + 1] = np.cross(T[ind + 1], e1[ind + 1])

        e1[ind + 1] /= np.linalg.norm(e1[ind + 1])
        e2[ind + 1] /= np.linalg.norm(e2[ind + 1])

    return T, e1, e2


def compute_corridor_fs(p_traj, ns, bs, radius=0.5, n_points=20):
    corridor_pts = []
    angles = np.linspace(0, 2 * np.pi, n_points)
    for i in range(p_traj.shape[0]):
        circle_pts = p_traj[i] + radius * (
            np.outer(np.cos(angles), ns[i]) + np.outer(np.sin(angles), bs[i])
        )
        corridor_pts.append(circle_pts)
    return corridor_pts


def get_local_window_params(track_data, s_global_now, num_knots, window_dist=4.0):
    s_orig = track_data["s"]
    s_end = min(s_global_now + window_dist, s_orig[-1])

    # Create the evaluation points for the knots
    # We sample knots over the window_dist
    s_query = np.linspace(s_global_now, s_end, num_knots)

    new_knots = {}
    for key in [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "e1x",
        "e1y",
        "e1z",
        "e2x",
        "e2y",
        "e2z",
    ]:
        # We use 'period' if your track is a loop, otherwise let it clamp
        new_knots[key] = np.interp(s_query, s_orig, track_data[key])

    # Construct parameter vector
    # IMPORTANT: The last parameter is the LENGTH of this local segment
    local_window = {
        "s": s_query,
        "x": new_knots["x"],
        "y": new_knots["y"],
        "z": new_knots["z"],
        "vx": new_knots["vx"],
        "vy": new_knots["vy"],
        "vz": new_knots["vz"],
        "e1x": new_knots["e1x"],
        "e1y": new_knots["e1y"],
        "e1z": new_knots["e1z"],
        "e2x": new_knots["e2x"],
        "e2y": new_knots["e2y"],
        "e2z": new_knots["e2z"],
        "L": s_end - s_global_now,
    }

    return local_window

def build_acados_params(local_window, global_params, tube_coeffs):
    return {
        "x":            local_window["x"],
        "y":            local_window["y"],
        "z":            local_window["z"],
        "vx":           local_window["vx"],
        "vy":           local_window["vy"],
        "vz":           local_window["vz"],
        "e1x":          local_window["e1x"],
        "e1y":          local_window["e1y"],
        "e1z":          local_window["e1z"],
        "tube_a":       tube_coeffs[0, :],
        "tube_b":       tube_coeffs[1, :],
        "tube_c":       tube_coeffs[2, :],
        "tube_d":       tube_coeffs[3, :],
        "L":            np.array([local_window["L"]]),
        "global_params": np.atleast_1d(global_params),
    }

def dict_to_list(d):
    return np.concatenate(list(d.values()))


def draw_horizon(ax, track_data, state):
    window = get_local_window_params(track_data, state[10], n_knots)
    s_end = state[10] + 4.0
    # print(state[10])
    knots = np.linspace(state[10], s_end, n_knots)

    n = n_knots
    x = window["x"].tolist()
    y = window["y"].tolist()
    z = window["z"].tolist()
    vx = window["vx"].tolist()
    vy = window["vy"].tolist()
    vz = window["vz"].tolist()
    e1x = window["e1x"].tolist()
    e1y = window["e1y"].tolist()
    e1z = window["e1z"].tolist()

    xref, yref, zref = interpolLUT(x, y, z, knots)
    vxref, vyref, vzref = interpolLUT(vx, vy, vz, knots)
    e1xref, e1yref, e1zref = interpolLUT(e1x, e1y, e1z, knots)

    ss = np.linspace(state[10], s_end, n_knots)
    p = np.zeros((n_knots, 3))
    t = np.zeros((n_knots, 3))
    e1 = np.zeros((n_knots, 3))
    e2 = np.zeros((n_knots, 3))
    for ind, s in enumerate(ss):
        p[ind, :] = np.array([xref(s), yref(s), zref(s)]).reshape((3,))
        t[ind, :] = np.array([vxref(s), vyref(s), vzref(s)]).reshape((3,))
        e1[ind, :] = np.array([e1xref(s), e1yref(s), e1zref(s)]).reshape((3,))
        e2[ind, :] = np.cross(e1[ind], t[ind])
        # print(np.linalg.norm(v), np.linalg.norm(e1), np.linalg.norm(e2))

    return p, t, e1, e2


def get_corridor_pts(
    ax, track_data, coeffs, alpha=0.2
):
    n_sweep = 100
    xi_eval = np.linspace(0, 1, n_sweep)
    P_eval = np.zeros((n_sweep, 2, 2))
    pp_eval = np.zeros((n_sweep, 2))

    poly_deg = len(coeffs[0]) - 1
    Phi_sweep = get_cheby_basis(xi_eval, poly_deg)
    a, b, c, d = coeffs

    a_sweep = Phi_sweep @ a
    b_sweep = Phi_sweep @ b
    c_sweep = Phi_sweep @ c
    d_sweep = Phi_sweep @ d

    s_sweep = track_data["s"]
    s_sweep = s_sweep - s_sweep[0]

    # print(s_sweep)
    L_path = s_sweep[-1]
    # L_path = track_data["L"]

    x = track_data["x"]
    y = track_data["y"]
    z = track_data["z"]
    traj = np.vstack([x, y, z]).T

    vx = track_data["vx"]
    vy = track_data["vy"]
    vz = track_data["vz"]
    tan = np.vstack([vx, vy, vz]).T
    tan /= np.linalg.norm(tan, axis=1)[:, np.newaxis]

    e1x = track_data["e1x"]
    e1y = track_data["e1y"]
    e1z = track_data["e1z"]
    e1 = np.vstack([e1x, e1y, e1z]).T
    e1 /= np.linalg.norm(e1, axis=1)[:, np.newaxis]

    e2 = np.cross(tan, e1)

    n_angles = 50
    angles = np.linspace(0, 2 * np.pi, n_angles)

    ellipse_pts_world = []

    for i in range(0, n_sweep, 5):
        P_eval[i] = np.array([[a_sweep[i], 0], [0, b_sweep[i]]])
        pp_eval[i] = np.array([c_sweep[i], d_sweep[i]])
        pc, width, height, angle = get_ellipse_parameters(P=P_eval[i], pp=pp_eval[i])

        ellipse_params = np.array([width, height, angle, pc[0], pc[1]])
        for j in range(n_angles):
            ellipse_pts = get_ellipse_points(
                width=ellipse_params[0],
                height=ellipse_params[1],
                angle=ellipse_params[2],
                theta=angles[j]
            )

            ind = np.argmin(np.abs(s_sweep - xi_eval[i] * L_path))
            w = ellipse_pts[:] + ellipse_params[-2:]
            ellipse_pts_world.append(traj[ind] + w[0] * e1[ind] + w[1] * e2[ind])


    all_pts = np.array(ellipse_pts_world)
    return all_pts
    # ax.scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2], alpha=alpha, s=5)

def action_unnormalize(val, min, max):
    return (val + 1.0) * (max - min) / 2.0 + min


def resample_path(data_array, N):
    """
    Resamples a 1D array to exactly N samples using linear interpolation.
    """
    # Current number of points
    M = len(data_array)

    # Original indices [0, 1, ..., M-1]
    original_indices = np.linspace(0, M - 1, M)

    # New indices [0, ..., M-1] but with exactly N points
    new_indices = np.linspace(0, M - 1, N)

    # Interpolate to get values at the new indices
    return np.interp(new_indices, original_indices, data_array)


# CrazyFlie 2.1 physical parameters
g0 = 9.80665  # [m.s^2] gravitational accerelation
mq = 31e-3  # [kg] total mass (with Lighthouse deck)
Ix = 1.395e-5  # [kg.m^2] Inertial moment around x-axis
Iy = 1.395e-5  # [kg.m^2] Inertial moment around y-axis
Iz = 2.173e-5  # [kg.m^2] Inertia moment around z-axis
Cd = 7.9379e-06  # [N/krpm^2] Drag coefficient
Ct = 3.25e-4  # [N/krpm^2] Thrust coefficient
dq = 92e-3  # [m] distance between motors' center
l = dq / 2  # [m] distance between motors' center and the axis of rotation
n_knots = 10


min_alpha = 0.1
max_alpha = 10.0

min_alpha_dot = -3.0
max_alpha_dot = 3.0
