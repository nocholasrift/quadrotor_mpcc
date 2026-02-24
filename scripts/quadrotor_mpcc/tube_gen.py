# all credit to jon arrizabalaga @ https://github.com/jonarriza96/corrgen
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from scipy.spatial.distance import cdist

import time


def poly_basis(xi, k, basis, d):
    if basis == "n":  # nominal
        return xi**k
    elif basis == "c":  # chebyshev
        return np.cos(k * np.arccos(2 * xi - 1))
    elif basis == "b":  # bernstein
        return np.math.comb(d, k) * xi**k * (1 - xi) ** (d - k)


def polynomial(xi, coeffs, degree):
    p_basis = "c"
    a = 0
    b = 0
    c = 0
    d = 0
    for k in range(degree + 1):
        a += coeffs["a"][k] * poly_basis(xi=xi, k=k, basis=p_basis, d=degree)
        b += coeffs["b"][k] * poly_basis(xi=xi, k=k, basis=p_basis, d=degree)
        c += coeffs["c"][k] * poly_basis(xi=xi, k=k, basis=p_basis, d=degree)
        d += coeffs["d"][k] * poly_basis(xi=xi, k=k, basis=p_basis, d=degree)

    return a, b, c, d

def get_cheby_basis(xi_vec, degree):
    # xi_vec shape: (N,) -> Φ shape: (N, degree + 1)
    N = xi_vec.shape[0]
    K = np.arange(degree + 1)
    # xi_clip = np.clip(xi_vec, 0.0, 1.0)
    # Perform basis calculation for all points and all degrees at once
    return np.cos(K * np.arccos(2 * xi_vec[:, np.newaxis] - 1))

def get_free_tube(poly_deg, max_radius):
    coeffs = np.zeros((4, poly_deg+1))
    coeffs[0, 0] = 1 / max_radius**2
    coeffs[1, 0] = 1 / max_radius**2

    return coeffs

def NLP(poly_deg, occ_points, traj_len, max_radius, LP):

    n_sweep = 100

    # define variables X = [[a c], [c b]]
    a = cp.Variable(poly_deg + 1)
    b = cp.Variable(poly_deg + 1)
    c = cp.Variable(poly_deg + 1)
    d = cp.Variable(poly_deg + 1)
    coeffs = {"a": a, "b": b, "c": c, "d": d}
    # coeffs = {"a": a, "b": b}

    # occ_param = cp.Parameter((max_occ_points, 3))
    xi_sweep = np.linspace(0, 1, n_sweep)
    Phi_sweep = get_cheby_basis(xi_sweep, poly_deg)

    a_sweep = Phi_sweep @ a
    b_sweep = Phi_sweep @ b
    c_sweep = Phi_sweep @ c
    d_sweep = Phi_sweep @ d
    
    cost = cp.sum(a_sweep + b_sweep)
    min_bound = 1 / (max_radius**2)
    constraints = [a_sweep >= min_bound, b_sweep >= min_bound]
    # constraints = [c_sweep == 0, d_sweep == 0]

    cage_pts = get_cage(xi_sweep * traj_len, max_radius)
    occ_points = np.vstack([occ_points, cage_pts])

    xi_occ = occ_points[:, 0] / traj_len
    w1 = occ_points[:, 1]
    w2 = occ_points[:, 2]

    Phi_occ = get_cheby_basis(xi_occ, poly_deg)
    a_occ = Phi_occ @ a
    b_occ = Phi_occ @ b
    c_occ = Phi_occ @ c
    d_occ = Phi_occ @ d

    # Vectorized ellipse constraint: a*w1^2 + b*w2^2 + c*w1 + d*w2 >= 1
    # We use cp.multiply for element-wise multiplication of expressions
    occ_expr = (cp.multiply(np.square(w1), a_occ) + 
                cp.multiply(np.square(w2), b_occ) + 
                cp.multiply(w1, c_occ) + 
                cp.multiply(w2, d_occ))
    # occ_expr = (cp.multiply(np.square(w1), a_occ) + 
    #             cp.multiply(np.square(w2), b_occ))

    # constraints += [d_occ == 0]
    # constraints += [c_occ == 0]
    # constraints += [c_sweep <= 2*max_radius]
    # constraints += [c_sweep >= -2*max_radius]
    # constraints += [d_sweep <= 2*max_radius]
    # constraints += [d_sweep >= -2*max_radius]

    constraints += [occ_expr >= 1]

    # define problem
    prob = cp.Problem(cp.Minimize(cost), constraints)

    return prob, [a, b, c, d]
    # return prob, [a, b]

def get_cage(xi_sweep, max_radius):
    n_wrap = len(xi_sweep)
    top    = np.column_stack([xi_sweep,  max_radius * np.ones(n_wrap),  np.zeros(n_wrap)])
    bottom = np.column_stack([xi_sweep, -max_radius * np.ones(n_wrap),  np.zeros(n_wrap)])
    left   = np.column_stack([xi_sweep,  np.zeros(n_wrap),  max_radius * np.ones(n_wrap)])
    right  = np.column_stack([xi_sweep,  np.zeros(n_wrap), -max_radius * np.ones(n_wrap)])
    
    return np.vstack([top, bottom, left, right])

def project_cloud_to_parametric_path(
    pcl, track_data, track_kdtree, max_radius=1, safety_check=False, prune=True
):
    s = track_data["s"]

    dists, indices = track_kdtree.query(pcl)
    indices = indices[dists <= max_radius]

    if indices.shape[0] == 0:
        return np.array([])

    x = track_data["x"]
    y = track_data["y"]
    z = track_data["z"]
    traj = np.vstack([x, y, z]).T[indices]

    vx = track_data["vx"]
    vy = track_data["vy"]
    vz = track_data["vz"]
    tan = np.vstack([vx, vy, vz]).T[indices]
    tan /= np.linalg.norm(tan, axis=1)[:, np.newaxis]

    e1x = track_data["e1x"]
    e1y = track_data["e1y"]
    e1z = track_data["e1z"]
    e1 = np.vstack([e1x, e1y, e1z]).T[indices]
    e1 /= np.linalg.norm(e1, axis=1)[:, np.newaxis]

    e2 = np.cross(tan, e1)

    pcl_xi = s[indices]

    offset = pcl[dists <= max_radius] - traj
    w1 = np.sum(offset * e1, axis=1)
    w2 = np.sum(offset * e2, axis=1)

    return np.column_stack((pcl_xi, w1, w2))

def add_world_boundaries(occ_cl, planar):

    x_min = min(occ_cl[:, 0])  # -1
    x_max = max(occ_cl[:, 0])  # 11
    y_min = min(occ_cl[:, 1])  # -1
    y_max = max(occ_cl[:, 1])  # 11
    z_min = min(occ_cl[:, 2])  # 0
    z_max = max(occ_cl[:, 2])  # 6
    n_side = 50

    z_side = np.linspace(z_min, z_max, n_side)
    z_side = np.repeat(z_side, n_side)

    x_side1 = np.linspace(x_min, x_max, n_side)
    y_side1 = np.linspace(y_min, y_min, n_side)
    side1 = np.vstack([x_side1, y_side1]).T
    side1 = np.hstack([np.tile(side1.T, n_side).T, z_side[:, None]])

    x_side2 = np.linspace(x_max, x_max, n_side)
    y_side2 = np.linspace(y_min, y_max, n_side)
    side2 = np.vstack([x_side2, y_side2]).T
    side2 = np.hstack([np.tile(side2.T, n_side).T, z_side[:, None]])

    x_side3 = np.linspace(x_max, x_min, n_side)
    y_side3 = np.linspace(y_max, y_max, n_side)
    side3 = np.vstack([x_side3, y_side3]).T
    side3 = np.hstack([np.tile(side3.T, n_side).T, z_side[:, None]])

    x_side4 = np.linspace(x_min, x_min, n_side)
    y_side4 = np.linspace(y_max, y_min, n_side)
    side4 = np.vstack([x_side4, y_side4]).T
    side4 = np.hstack([np.tile(side4.T, n_side).T, z_side[:, None]])

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, n_side), np.linspace(y_min, y_max, n_side)
    )
    side5 = np.vstack(
        [X.flatten(), Y.flatten(), z_min * np.ones((n_side, n_side)).flatten()]
    ).T
    side6 = np.vstack(
        [X.flatten(), Y.flatten(), z_max * np.ones((n_side, n_side)).flatten()]
    ).T

    if planar:
        bnd = np.vstack([side1[:, :2], side2[:, :2], side3[:, :2], side4[:, :2]])
    else:
        bnd = np.vstack([side1, side2, side3, side4, side5, side6])
    occ_cl = np.vstack([occ_cl, bnd])

    return occ_cl


def add_roof_floor(ref_path, occ_cl, kitti_zmax, kitti_zmin):

    cl_x_min = np.min(occ_cl[:, 0])
    cl_x_max = np.max(occ_cl[:, 0])
    cl_y_min = np.min(occ_cl[:, 1])
    cl_y_max = np.max(occ_cl[:, 1])

    rp_x_min = np.min(ref_path[:, 0])
    rp_x_max = np.max(ref_path[:, 0])
    rp_y_min = np.min(ref_path[:, 1])
    rp_y_max = np.max(ref_path[:, 1])

    x_min = min(cl_x_min, rp_x_min)
    x_max = max(cl_x_max, rp_x_max)
    y_min = min(cl_y_min, rp_y_min)
    y_max = max(cl_y_max, rp_y_max)

    n_side = 30
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, n_side), np.linspace(y_min, y_max, n_side)
    )
    roof = np.vstack(
        [
            X.flatten(),
            Y.flatten(),
            kitti_zmax * np.ones((n_side, n_side)).flatten(),
        ]
    ).T
    floor = np.vstack(
        [
            X.flatten(),
            Y.flatten(),
            kitti_zmin * np.ones((n_side, n_side)).flatten(),
        ]
    ).T
    occ_cl = np.vstack([occ_cl, roof, floor])

    return occ_cl


def get_ellipse_parameters(P, pp, p0=np.array([0, 0])):
    """Calculates parameters for an ellipse given as (x-p0)^T @ P @ (x-p0) + pp @ (x-p0) - 1 = 0

    Args:
        P (np.array): matrix of the ellipse
        pp (np.array): vector of the ellipse (allows for offsets from center point p)
        p0 (np.array): center point (without offset)
    Returns:
        pc: Actual center of the ellipse
        width: width of the ellipse (2*a from x**2/a**2 + y**2/b**2 = 1)
        height: height of the ellipse (2*b from x**2/a**2 + y**2/b**2 = 1)
        theta: angle of the major axis

    NOTE: Equations taken from wikipedia (ellipse, section"General ellipse",
    https://en.wikipedia.org/wiki/Ellipse). Check "visualization.nb" for theoretical
    analysis of the equations.
    """
    a = P[0, 0]
    b = 2 * P[0, 1]
    c = P[1, 1]
    d = pp[0]
    ee = pp[1]
    f = -1

    aell = -np.sqrt(
        2
        * (a * ee**2 + c * d**2 - b * d * ee + (b**2 - 4 * a * c) * f)
        * ((a + c) + np.sqrt((a - c) ** 2 + b**2))
    ) / (b**2 - 4 * a * c)
    bell = -np.sqrt(
        2
        * (a * ee**2 + c * d**2 - b * d * ee + (b**2 - 4 * a * c) * f)
        * ((a + c) - np.sqrt((a - c) ** 2 + b**2))
    ) / (b**2 - 4 * a * c)

    xc = (2 * c * d - b * ee) / (b**2 - 4 * a * c) + p0[0]
    yc = (2 * a * ee - b * d) / (b**2 - 4 * a * c) + p0[1]

    height = 2 * aell
    width = 2 * bell
    pc = np.array([xc, yc])
    theta = 1 / 2 * np.arctan2(-b, c - a) + np.pi / 2

    return pc, width, height, theta

def get_ellipse_points(width, height, angle, theta):
    """Computes the points in the contour of an ellipse x**2/a + y**2/b = 1
    NOTE: Implementation taken from https://math.stackexchange.com/a/4517941
    Args:
        width (float): 2*a
        height (float): 2*b
        angle (float): Angle by which the x-axis (width or a) is rotated (anticlockwise)
        theta (float): Rotation angle by which the point needs to be located

    Returns:
        pt (np.ndarray): Point in the contour of the ellipse
    """

    def r_ellipse(theta, a, b):
        return a * b / np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)

    a = width / 2
    b = height / 2

    R_ellipse = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rot = r_ellipse(theta - angle, a, b) * np.array(
        [np.cos(theta - angle), np.sin(theta - angle)]
    )

    pt = R_ellipse @ rot

    return pt


