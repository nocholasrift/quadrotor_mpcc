# Add this to your environment file or create a separate visualization utility

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common import *
from acados_settings import *
from mpc_env import VolaDroneEnv



def create_cbf_debug_function(alpha):
    # Arc length knots (same as in model)
    arc_len_knots = np.linspace(0, 1, n_knots)

    # Symbolic inputs - knot coefficients
    x_coeff = ca.MX.sym("x_coeffs", n_knots)
    y_coeff = ca.MX.sym("y_coeffs", n_knots)
    z_coeff = ca.MX.sym("z_coeffs", n_knots)

    vx_coeff = ca.MX.sym("vx_coeffs", n_knots)
    vy_coeff = ca.MX.sym("vy_coeffs", n_knots)
    vz_coeff = ca.MX.sym("vz_coeffs", n_knots)

    e1x_coeff = ca.MX.sym("e1x_coeffs", n_knots)
    e1y_coeff = ca.MX.sym("e1y_coeffs", n_knots)
    e1z_coeff = ca.MX.sym("e1z_coeffs", n_knots)

    e2x_coeff = ca.MX.sym("e2x_coeffs", n_knots)
    e2y_coeff = ca.MX.sym("e2y_coeffs", n_knots)
    e2z_coeff = ca.MX.sym("e2z_coeffs", n_knots)

    L_path = ca.MX.sym("L_path")

    def create_interp(name, coeffs):
        interp = ca.interpolant(name, "bspline", [arc_len_knots.tolist()])
        s_norm = s / L_path
        return interp(s_norm, coeffs)

    # State vector
    px = ca.MX.sym("px")
    py = ca.MX.sym("py")
    pz = ca.MX.sym("pz")
    pos = ca.vertcat(px, py, pz)

    vx = ca.MX.sym("vx")
    vy = ca.MX.sym("vy")
    vz = ca.MX.sym("vz")
    v = ca.vertcat(vx, vy, vz)

    q1 = ca.MX.sym("q1")
    q2 = ca.MX.sym("q2")
    q3 = ca.MX.sym("q3")
    q4 = ca.MX.sym("q4")
    q = ca.vertcat(q1, q2, q3, q4)

    s = ca.MX.sym("s")
    s_dot = ca.MX.sym("s_dot")

    
    # 1. Update the Control Vector to match your setup
    thrust = ca.MX.sym("thrust")
    wr = ca.MX.sym("wr")
    wp = ca.MX.sym("wp")
    wy = ca.MX.sym("wy")
    s_ddot = ca.MX.sym("s_ddot")
    u = ca.vertcat(thrust, wr, wp, wy, s_ddot)
    omg = ca.vertcat(wr, wp, wy)


    xr = create_interp("xr_dbg", x_coeff)
    yr = create_interp("yr_dbg", y_coeff)
    zr = create_interp("zr_dbg", z_coeff)
    pr = ca.vertcat(xr, yr, zr)

    # Interpolate velocity (tangent)
    xr_dot = create_interp("vxr_dbg", vx_coeff)
    yr_dot = create_interp("vyr_dbg", vy_coeff)
    zr_dot = create_interp("vzr_dbg", vz_coeff)
    tr_raw = ca.vertcat(xr_dot, yr_dot, zr_dot)
    tr = tr_raw / (ca.norm_2(tr_raw) + 1e-8)

    # Interpolate e1
    e1x = create_interp("e1x_dbg", e1x_coeff)
    e1y = create_interp("e1y_dbg", e1y_coeff)
    e1z = create_interp("e1z_dbg", e1z_coeff)
    e1_raw = ca.vertcat(e1x, e1y, e1z)
    e1 = e1_raw / (ca.norm_2(e1_raw) + 1e-8)

    e2 = ca.cross(tr, e1)

    Rq = ca.vertcat(
        ca.horzcat(
            1 - 2 * (q3**2 + q4**2),
            2 * (q2 * q3 - q1 * q4),
            2 * (q2 * q4 + q1 * q3),
        ),
        ca.horzcat(
            2 * (q2 * q3 + q1 * q4),
            1 - 2 * (q2**2 + q4**2),
            2 * (q3 * q4 - q1 * q2),
        ),
        ca.horzcat(
            2 * (q2 * q4 - q1 * q3),
            2 * (q3 * q4 + q1 * q2),
            1 - 2 * (q2**2 + q3**2),
        ),
    )


    # 3. Use your new Exponential CBF Logic
    e_tot = pos - pr
    e_l = ca.dot(tr, e_tot)
    e_perp = e_tot - e_l * tr
    
    z_b = Rq[:, 2] # Body z-axis from your Rq matrix
    
    radius = 1.0

    f_danger = .05 * ca.dot(e_perp, v) + .01 * ca.dot(e_perp, z_b)
    h_pos = radius**2 - ca.dot(e_perp, e_perp)
    cbf = h_pos * ca.exp(-f_danger)
    # cbf = h_pos / (1.0 + f_danger**2)

    # 4. Lie Derivatives (Crucial for verifying Relative Degree 1)
    # Dynamics exactly as in setup()
    qdot = 0.5 * ca.vertcat(
        (-q2 * wr - q3 * wp - q4 * wy),
        (q1 * wr + q3 * wy - q4 * wp),
        (q1 * wp - q2 * wy + q4 * wr),
        (q1 * wy + q2 * wp - q3 * wr)
    )
    vdot = -ca.vertcat(0, 0, g0) + (1 / mq) * Rq @ ca.vertcat(0, 0, thrust)
    f_expl = ca.vertcat(v, vdot, qdot, s_dot, s_ddot)
    
    # State Vector x
    x = ca.vertcat(pos, v, q, s, s_dot)
    
    grad_h = ca.jacobian(cbf, x)
    Lfh = grad_h @ f_expl # Note: f_expl contains u, but usually we use drift f(x)
    # To get proper Lfh/Lgh separation:
    F_drift = ca.vertcat(v, -ca.vertcat(0, 0, g0), ca.MX.zeros(4,1), s_dot, 0)
    G_input = ca.jacobian(f_expl, u)
    
    Lfh = grad_h @ F_drift
    Lgh = grad_h @ G_input # This matrix will show sensitivity to thrust and omega
    cbf_const = Lfh + Lgh@u + alpha * cbf

    # 5. Updated Function Returns
    # We return the key values you need to plot
    return ca.Function("cbf_debug", 
        [x_coeff, y_coeff, z_coeff, vx_coeff, vy_coeff, vz_coeff, 
         e1x_coeff, e1y_coeff, e1z_coeff, e2x_coeff, e2y_coeff, e2z_coeff, 
         L_path, x, u],
        [cbf_const, Lfh, Lgh, cbf, h_pos, f_danger, e_perp, tr, pr],
        ["x_c", "y_c", "z_c", "vx_c", "vy_c", "vz_c", "e1x_c", "e1y_c", "e1z_c", "e2x_c", "e2y_c", "e2z_c", "L", "x", "u"],
        ["cbf_const", "Lfh", "Lgh", "cbf", "h_pos", "f_danger", "e_perp", "tr", "pr"]
    )

# def create_cbf_debug_function():
#     """
#     Creates a CasADi function that evaluates CBF quantities and RMF frame
#     using the exact same interpolation logic as the MPC model.
#     """
#     # Arc length knots (same as in model)
#     arc_len_knots = np.linspace(0, 1, n_knots)
#
#     # Symbolic inputs - knot coefficients
#     x_coeff = ca.MX.sym("x_coeffs", n_knots)
#     y_coeff = ca.MX.sym("y_coeffs", n_knots)
#     z_coeff = ca.MX.sym("z_coeffs", n_knots)
#
#     vx_coeff = ca.MX.sym("vx_coeffs", n_knots)
#     vy_coeff = ca.MX.sym("vy_coeffs", n_knots)
#     vz_coeff = ca.MX.sym("vz_coeffs", n_knots)
#
#     e1x_coeff = ca.MX.sym("e1x_coeffs", n_knots)
#     e1y_coeff = ca.MX.sym("e1y_coeffs", n_knots)
#     e1z_coeff = ca.MX.sym("e1z_coeffs", n_knots)
#
#     e2x_coeff = ca.MX.sym("e2x_coeffs", n_knots)
#     e2y_coeff = ca.MX.sym("e2y_coeffs", n_knots)
#     e2z_coeff = ca.MX.sym("e2z_coeffs", n_knots)
#
#     L_path = ca.MX.sym("L_path")
#
#     # State vector
#     px = ca.MX.sym("px")
#     py = ca.MX.sym("py")
#     pz = ca.MX.sym("pz")
#     pos = ca.vertcat(px, py, pz)
#
#     vx = ca.MX.sym("vx")
#     vy = ca.MX.sym("vy")
#     vz = ca.MX.sym("vz")
#     v = ca.vertcat(vx, vy, vz)
#
#     q1 = ca.MX.sym("q1")
#     q2 = ca.MX.sym("q2")
#     q3 = ca.MX.sym("q3")
#     q4 = ca.MX.sym("q4")
#     q = ca.vertcat(q1, q2, q3, q4)
#
#     s = ca.MX.sym("s")
#     s_dot = ca.MX.sym("s_dot")
#
#     x = ca.vertcat(pos, v, q, s, s_dot)
#
#     # Control vector
#     thrust = ca.MX.sym("thrust")
#     wr = ca.MX.sym("wr")
#     wp = ca.MX.sym("wp")
#     wy = ca.MX.sym("wy")
#     s_ddot = ca.MX.sym("s_ddot")
#     u = ca.vertcat(thrust, wr, wp, wy, s_ddot)
#
#     # Create interpolants (exactly as in model)
#     def create_interp(name, coeffs):
#         interp = ca.interpolant(name, "bspline", [arc_len_knots.tolist()])
#         s_norm = s / L_path
#         return interp(s_norm, coeffs)
#
#     # Interpolate position
#     xr = create_interp("xr_dbg", x_coeff)
#     yr = create_interp("yr_dbg", y_coeff)
#     zr = create_interp("zr_dbg", z_coeff)
#     pr = ca.vertcat(xr, yr, zr)
#
#     # Interpolate velocity (tangent)
#     xr_dot = create_interp("vxr_dbg", vx_coeff)
#     yr_dot = create_interp("vyr_dbg", vy_coeff)
#     zr_dot = create_interp("vzr_dbg", vz_coeff)
#     tr_raw = ca.vertcat(xr_dot, yr_dot, zr_dot)
#     tr = tr_raw / (ca.norm_2(tr_raw) + 1e-8)
#
#     # Interpolate e1
#     e1x = create_interp("e1x_dbg", e1x_coeff)
#     e1y = create_interp("e1y_dbg", e1y_coeff)
#     e1z = create_interp("e1z_dbg", e1z_coeff)
#     e1_raw = ca.vertcat(e1x, e1y, e1z)
#     e1 = e1_raw / (ca.norm_2(e1_raw) + 1e-8)
#
#     # Interpolate e2
#     # e2x = create_interp("e2x_dbg", e2x_coeff)
#     # e2y = create_interp("e2y_dbg", e2y_coeff)
#     # e2z = create_interp("e2z_dbg", e2z_coeff)
#     # e2_raw = ca.vertcat(e2x, e2y, e2z)
#     # e2 = e2_raw / (ca.norm_2(e2_raw) + 1e-8)
#     e2 = ca.cross(tr, e1)
#
#     # Rotation matrix
#     Rq = ca.vertcat(
#         ca.horzcat(
#             1 - 2 * (q3**2 + q4**2),
#             2 * (q2 * q3 - q1 * q4),
#             2 * (q2 * q4 + q1 * q3),
#         ),
#         ca.horzcat(
#             2 * (q2 * q3 + q1 * q4),
#             1 - 2 * (q2**2 + q4**2),
#             2 * (q3 * q4 - q1 * q2),
#         ),
#         ca.horzcat(
#             2 * (q2 * q4 - q1 * q3),
#             2 * (q3 * q4 + q1 * q2),
#             1 - 2 * (q2**2 + q3**2),
#         ),
#     )
#
#     # CBF computation (exactly as in model)
#     e_tot = pos - pr
#     e_l = ca.dot(tr, e_tot)
#     e_l_vec = e_l * tr
#
#     b1 = ca.dot(e1, e_tot)
#     b2 = ca.dot(e2, e_tot)
#
#     e_perp = e_tot - e_l_vec
#     dist_to_path = ca.norm_2(e_perp)
#
#     radius = 0.5
#     h = radius - dist_to_path
#
#     perp_dir = e_perp / (
#         ca.norm_2(e_perp) + 1e-8
#     )  # Unit vector pointing away from path
#
#     thrust_dir = Rq[:, 2]
#     thrust_perp = (
#         thrust_dir - ca.dot(thrust_dir, tr) * tr
#     )  # Remove tangential component
#     thrust_toward_boundary = ca.dot(thrust_perp, perp_dir)
#
#     v_perp = v - ca.dot(v, tr) * tr  # Remove tangential component
#     v_toward_boundary = ca.dot(v_perp, perp_dir)
#
#     beta = 0.05
#     p = thrust_toward_boundary + beta * v_toward_boundary
#     p_clamped = ca.fmax(ca.fmin(p, 1.0), -1.0)
#     cbf = h * ca.exp(-p_clamped)
#     # radius = 0.5
#     # h = 1 - (b1 / radius) ** 2 - (b2 / radius) ** 2
#     #
#     # thrust_dir = Rq[:, 2]
#     contour_mag = ca.sqrt(b1**2 + b2**2 + 1e-8)
#     boundary_dir_3d = (b1 * e1 + b2 * e2) / contour_mag
#     #
#     # thrust_toward_boundary = ca.dot(thrust_dir, boundary_dir_3d)
#     # v_toward_boundary = ca.dot(v, boundary_dir_3d)
#     #
#     # p = thrust_toward_boundary + 0.05 * v_toward_boundary
#     # p_sat = ca.fmax(ca.fmin(p, 1.0), -1.0)
#     # cbf = h * ca.exp(-p_sat)
#
#     # Lie derivatives
#     gravity = ca.vertcat(0, 0, g0)
#     F = ca.vertcat(v, -gravity, ca.MX.zeros(4, 1), s_dot, 0)
#     G = ca.MX.zeros(12, 5)
#     G[3:6, 0] = (1 / mq) * Rq[:, 2]
#     G[6:10, 1:4] = 0.5 * ca.vertcat(
#         ca.horzcat(-q2, -q3, -q4),
#         ca.horzcat(q1, -q4, q3),
#         ca.horzcat(q4, q1, -q2),
#         ca.horzcat(-q3, q2, q1),
#     )
#     G[11, 4] = 1
#
#     grad_h = ca.jacobian(cbf, x)
#     Lfh = grad_h @ F
#     Lgh = grad_h @ G
#
#     # Diagnostics
#     e1_norm = ca.norm_2(e1)
#     e2_norm = ca.norm_2(e2)
#     e1_dot_e2 = ca.dot(e1, e2)
#     e1_dot_tr = ca.dot(e1, tr / (ca.norm_2(tr) + 1e-8))
#     e2_dot_tr = ca.dot(e2, tr / (ca.norm_2(tr) + 1e-8))
#
#     # Tracking errors
#     e_l = ca.dot(tr, e_tot)
#     e_l_vec = e_l * tr
#     e_c = e_tot - e_l_vec
#
#     # Create the function
#     debug_func = ca.Function(
#         "cbf_debug",
#         [
#             x_coeff,
#             y_coeff,
#             z_coeff,
#             vx_coeff,
#             vy_coeff,
#             vz_coeff,
#             e1x_coeff,
#             e1y_coeff,
#             e1z_coeff,
#             e2x_coeff,
#             e2y_coeff,
#             e2z_coeff,
#             L_path,
#             x,
#             u,
#         ],
#         [
#             Lfh,
#             Lgh,
#             cbf,  # CBF quantities
#             b1,
#             b2,
#             h,
#             p,  # Intermediate values
#             thrust_toward_boundary,
#             v_toward_boundary,  # Components
#             e1,
#             e2,
#             tr,
#             pr,  # Frame and reference
#             e1_norm,
#             e2_norm,
#             e1_dot_e2,
#             e1_dot_tr,
#             e2_dot_tr,  # Orthonormality
#             e_c,
#             e_l,  # Tracking errors
#             boundary_dir_3d,  # Boundary direction
#         ],
#         [
#             "x_c",
#             "y_c",
#             "z_c",
#             "vx_c",
#             "vy_c",
#             "vz_c",
#             "e1x_c",
#             "e1y_c",
#             "e1z_c",
#             "e2x_c",
#             "e2y_c",
#             "e2z_c",
#             "L_path",
#             "x",
#             "u",
#         ],
#         [
#             "Lfh",
#             "Lgh",
#             "cbf",
#             "b1",
#             "b2",
#             "h",
#             "p",
#             "thrust_tb",
#             "v_tb",
#             "e1",
#             "e2",
#             "tr",
#             "pr",
#             "e1_norm",
#             "e2_norm",
#             "e1_dot_e2",
#             "e1_dot_tr",
#             "e2_dot_tr",
#             "e_c",
#             "e_l",
#             "boundary_dir",
#         ],
#     )
#
#     return debug_func


# def visualize_cbf_diagnostics(env, num_steps=50):
#     """
#     Visualize CBF and RMF frame evolution along the trajectory.
#
#     Args:
#         env: VolaDroneEnv instance
#         num_steps: Number of steps to simulate and visualize
#     """
#     cbf_debug = create_cbf_debug_function()
#
#     # Storage for trajectory
#     history = {
#         "pos": [],
#         "s": [],
#         "cbf": [],
#         "h": [],
#         "p": [],
#         "Lfh": [],
#         "b1": [],
#         "b2": [],
#         "thrust_tb": [],
#         "v_tb": [],
#         "e1_norm": [],
#         "e2_norm": [],
#         "e1_dot_e2": [],
#         "e_c_norm": [],
#         "e_l": [],
#         "e1": [],
#         "e2": [],
#         "tr": [],
#         "pr": [],
#         "boundary_dir": [],
#     }
#
#     env.reset()
#
#     for step in range(num_steps):
#         # Get current state and control
#         s_global = env.state[10]
#         local_p = env._get_local_window_params(s_global)
#
#         # Solve MPC
#         _, _, done, _, _ = env.step(np.array([0, 0]))
#         if done:
#             break
#
#         u = env.solver.get(0, "u")
#         x_state = env.state
#
#         # Prepare local state (s normalized to window)
#         local_state = x_state.copy()
#         local_state[10] = x_state[10] - s_global  # Local s within window
#
#         # Evaluate debug function
#         n = n_knots
#         res_tuple = cbf_debug(
#             local_p[0:n],
#             local_p[n : 2 * n],
#             local_p[2 * n : 3 * n],
#             local_p[3 * n : 4 * n],
#             local_p[4 * n : 5 * n],
#             local_p[5 * n : 6 * n],
#             local_p[6 * n : 7 * n],
#             local_p[7 * n : 8 * n],
#             local_p[8 * n : 9 * n],
#             local_p[9 * n : 10 * n],
#             local_p[10 * n : 11 * n],
#             local_p[11 * n : 12 * n],
#             local_p[-2],  # L_path
#             local_state,
#             u,
#         )
#
#         Lfh_val, Lgh_val, cbf_val, h_pos_val, f_danger_val, e_perp_val, tr_val, pr_val = res
#
#         history["pos"].append(env.state[:3].copy())
#         history["s"].append(s_global)
#         history["cbf"].append(float(cbf_val))
#         history["h"].append(float(h_val))
#         history["p"].append(float(p_val))
#         history["Lfh"].append(float(Lfh_val))
#         history["b1"].append(float(b1_val))
#         history["b2"].append(float(b2_val))
#         history["thrust_tb"].append(float(thrust_tb_val))
#         history["v_tb"].append(float(v_tb_val))
#         history["e1_norm"].append(float(e1_norm_val))
#         history["e2_norm"].append(float(e2_norm_val))
#         history["e1_dot_e2"].append(float(e1_dot_e2_val))
#         history["e_c_norm"].append(float(np.linalg.norm(e_c_val)))
#         history["e_l"].append(float(e_l_val))
#         history["e1"].append(np.array(e1_val).flatten())
#         history["e2"].append(np.array(e2_val).flatten())
#         history["tr"].append(np.array(tr_val).flatten())
#         history["pr"].append(np.array(pr_val).flatten())
#         history["boundary_dir"].append(np.array(boundary_dir_val).flatten())
#         history["Lgh_thrust"].append(float(Lgh_val[0]))
#         history["Lgh_omega_norm"].append(float(np.linalg.norm(Lgh_val[1:4])))
#
#         if step % 10 == 0:
#             print(f"\n=== Step {step}, s={s_global:.2f} ===")
#             print(
#                 f"  CBF: {history['cbf'][-1]:.4f}, h: {history['h'][-1]:.4f}, p: {history['p'][-1]:.4f}"
#             )
#             print(f"  Lfh: {history['Lfh'][-1]:.4e}")
#             print(f"  b1: {history['b1'][-1]:.3f}, b2: {history['b2'][-1]:.3f}")
#             print(
#                 f"  thrust→boundary: {history['thrust_tb'][-1]:.3f}, v→boundary: {history['v_tb'][-1]:.3f}"
#             )
#             print(
#                 f"  ||e1||: {history['e1_norm'][-1]:.4f}, ||e2||: {history['e2_norm'][-1]:.4f}"
#             )
#             print(f"  e1·e2: {history['e1_dot_e2'][-1]:.4f}")
#             print(
#                 f"  e_c: {history['e_c_norm'][-1]:.3f}, e_l: {history['e_l'][-1]:.3f}"
#             )
#
#     # Convert to arrays
#     for key in history:
#         if key in ["e1", "e2", "tr", "pr", "pos", "boundary_dir"]:
#             history[key] = np.array(history[key])
#         else:
#             history[key] = np.array(history[key])
#
#     # Create visualization
#     fig = plt.figure(figsize=(18, 12))
#
#     # Plot 1: 3D trajectory with RMF frame
#     ax1 = fig.add_subplot(3, 3, 1, projection="3d")
#     ax1.plot(
#         history["pos"][:, 0],
#         history["pos"][:, 1],
#         history["pos"][:, 2],
#         "b-",
#         linewidth=2,
#         label="Actual trajectory",
#     )
#     ax1.plot(
#         history["pr"][:, 0],
#         history["pr"][:, 1],
#         history["pr"][:, 2],
#         "r--",
#         linewidth=1,
#         alpha=0.5,
#         label="Reference",
#     )
#
#     # Plot RMF vectors at subset of points
#     skip = max(1, len(history["pos"]) // 10)
#     scale = 0.3
#     for i in range(0, len(history["pos"]), skip):
#         # e1 (red)
#         ax1.quiver(
#             history["pr"][i, 0],
#             history["pr"][i, 1],
#             history["pr"][i, 2],
#             history["e1"][i, 0],
#             history["e1"][i, 1],
#             history["e1"][i, 2],
#             color="red",
#             length=scale,
#             normalize=True,
#             arrow_length_ratio=0.3,
#         )
#         # e2 (green)
#         ax1.quiver(
#             history["pr"][i, 0],
#             history["pr"][i, 1],
#             history["pr"][i, 2],
#             history["e2"][i, 0],
#             history["e2"][i, 1],
#             history["e2"][i, 2],
#             color="green",
#             length=scale,
#             normalize=True,
#             arrow_length_ratio=0.3,
#         )
#
#     ax1.set_xlabel("X")
#     ax1.set_ylabel("Y")
#     ax1.set_zlabel("Z")
#     ax1.set_title("Trajectory with RMF Frame")
#     ax1.legend()
#
#     # Plot 2: CBF evolution
#     ax2 = fig.add_subplot(3, 3, 2)
#     ax2.plot(history["s"], history["cbf"], "b-", label="CBF (augmented)", linewidth=2)
#     ax2.plot(history["s"], history["h"], "r--", label="h (base)", linewidth=1.5)
#     ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
#     ax2.set_xlabel("s")
#     ax2.set_ylabel("Value")
#     ax2.set_title("CBF Evolution")
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
#
#     # Plot 3: Lfh (THIS IS KEY - check if it's blowing up)
#     ax3 = fig.add_subplot(3, 3, 3)
#     ax3.plot(history["s"], history["Lfh"], "r-", linewidth=2)
#     ax3.set_xlabel("s")
#     ax3.set_ylabel("Lfh")
#     ax3.set_title("Lie Derivative Lfh")
#     ax3.grid(True, alpha=0.3)
#     ax3.axhline(y=0, color="k", linestyle="--", alpha=0.5)
#
#     # Plot 4: Exponent term p
#     ax4 = fig.add_subplot(3, 3, 4)
#     ax4.plot(history["s"], history["p"], "g-", label="p (total)", linewidth=2)
#     ax4.plot(
#         history["s"],
#         history["thrust_tb"],
#         "r--",
#         label="thrust→boundary",
#         linewidth=1.5,
#     )
#     ax4.plot(history["s"], history["v_tb"], "b--", label="v→boundary", linewidth=1.5)
#     ax4.set_xlabel("s")
#     ax4.set_ylabel("Value")
#     ax4.set_title("CBF Exponent Components")
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
#     ax4.axhline(y=0, color="k", linestyle="--", alpha=0.5)
#
#     # Plot 5: Tube coordinates
#     ax5 = fig.add_subplot(3, 3, 5)
#     ax5.plot(history["s"], history["b1"], "r-", label="b1", linewidth=2)
#     ax5.plot(history["s"], history["b2"], "b-", label="b2", linewidth=2)
#     ax5.axhline(y=0.5, color="r", linestyle=":", alpha=0.5, label="radius")
#     ax5.axhline(y=-0.5, color="r", linestyle=":", alpha=0.5)
#     ax5.set_xlabel("s")
#     ax5.set_ylabel("Distance [m]")
#     ax5.set_title("Tube Coordinates (b1, b2)")
#     ax5.legend()
#     ax5.grid(True, alpha=0.3)
#
#     # Plot 6: Orthonormality
#     ax6 = fig.add_subplot(3, 3, 6)
#     ax6.plot(history["s"], history["e1_norm"], "r-", label="||e1||", linewidth=2)
#     ax6.plot(history["s"], history["e2_norm"], "g-", label="||e2||", linewidth=2)
#     ax6.plot(
#         history["s"],
#         np.abs(history["e1_dot_e2"]),
#         "k--",
#         label="|e1·e2|",
#         linewidth=1.5,
#     )
#     ax6.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
#     ax6.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
#     ax6.set_xlabel("s")
#     ax6.set_ylabel("Value")
#     ax6.set_title("RMF Orthonormality")
#     ax6.legend()
#     ax6.grid(True, alpha=0.3)
#
#     # Plot 7: Tracking errors
#     ax7 = fig.add_subplot(3, 3, 7)
#     ax7.plot(
#         history["s"],
#         history["e_c_norm"],
#         "r-",
#         label="||e_c|| (contouring)",
#         linewidth=2,
#     )
#     ax7.plot(
#         history["s"], np.abs(history["e_l"]), "b-", label="|e_l| (lag)", linewidth=2
#     )
#     ax7.set_xlabel("s")
#     ax7.set_ylabel("Error [m]")
#     ax7.set_title("Tracking Errors")
#     ax7.legend()
#     ax7.grid(True, alpha=0.3)
#
#     # Plot 8: CBF constraint violation (Lfh + alpha*cbf should be ≥ 0)
#     alpha0 = 0.5  # Your current alpha value
#     ax8 = fig.add_subplot(3, 3, 8)
#     constraint_val = history["Lfh"] + alpha0 * history["cbf"]
#     ax8.plot(history["s"], constraint_val, "b-", linewidth=2)
#     ax8.axhline(y=0, color="r", linestyle="--", linewidth=2, label="Constraint limit")
#     ax8.fill_between(
#         history["s"],
#         0,
#         np.minimum(constraint_val, 0),
#         color="red",
#         alpha=0.3,
#         label="Violations",
#     )
#     ax8.set_xlabel("s")
#     ax8.set_ylabel("Lfh + α₀·CBF")
#     ax8.set_title("CBF Constraint Satisfaction")
#     ax8.legend()
#     ax8.grid(True, alpha=0.3)
#
#     # Plot 9: Summary statistics
#     ax9 = fig.add_subplot(3, 3, 9)
#     ax9.axis("off")
#     stats_text = f"""
#     Summary Statistics:
#
#     CBF range: [{history['cbf'].min():.3f}, {history['cbf'].max():.3f}]
#     h range: [{history['h'].min():.3f}, {history['h'].max():.3f}]
#     Lfh range: [{history['Lfh'].min():.2e}, {history['Lfh'].max():.2e}]
#
#     Max |Lfh|: {np.abs(history['Lfh']).max():.2e}
#
#     Constraint violations: {np.sum(constraint_val < 0)} / {len(constraint_val)}
#     Min constraint value: {constraint_val.min():.3f}
#
#     RMF orthonormality:
#       max ||e1|| error: {np.abs(history['e1_norm'] - 1).max():.4f}
#       max ||e2|| error: {np.abs(history['e2_norm'] - 1).max():.4f}
#       max |e1·e2|: {np.abs(history['e1_dot_e2']).max():.4f}
#
#     Tracking:
#       mean e_c: {history['e_c_norm'].mean():.3f} m
#       mean |e_l|: {np.abs(history['e_l']).mean():.3f} m
#     """
#     ax9.text(
#         0.1,
#         0.5,
#         stats_text,
#         transform=ax9.transAxes,
#         fontsize=10,
#         verticalalignment="center",
#         family="monospace",
#     )
#
#     plt.tight_layout()
#     plt.show()
#
#     return history

def visualize_cbf_diagnostics(env, num_steps=50):
    """
    Visualize Relative Degree 1 CBF and Input Sensitivities.
    """
    cbf_debug = create_cbf_debug_function(env.alpha0)

    # Enhanced history for Relative Degree 1 analysis
    history = {
        "s": [], "cbf": [], "h_pos": [], "f_danger": [],
        "Lfh": [], "Lgh_thrust": [], "Lgh_omega_mag": [],
        "pos": [], "pr": [], "e_perp_norm": [],
        "u_thrust": [], "u_omega_mag": [], "cbf_const": []
    }

    env.reset()

    for step in range(num_steps):
        s_global = env.state[10]
        local_p = get_local_window_params(env.track_data, s_global)

        # Step the environment
        _, _, done, _, _ = env.step(np.array([0, 0])) # Assuming auto-MPC step
        if done: break

        u_val = env.solver.get(0, "u")
        x_state = env.state

        # Align s to local window
        local_state = x_state.copy()
        local_state[10] = x_state[10] - s_global 

        # Evaluate the debug function
        n = n_knots
        res = cbf_debug(
            local_p[0:n], local_p[n:2*n], local_p[2*n:3*n],     # pos coeffs
            local_p[3*n:4*n], local_p[4*n:5*n], local_p[5*n:6*n], # vel coeffs
            local_p[6*n:7*n], local_p[7*n:8*n], local_p[8*n:9*n], # e1 coeffs
            local_p[9*n:10*n], local_p[10*n:11*n], local_p[11*n:12*n], # e2 coeffs
            local_p[-3], # L_path (check index based on your p vector)
            local_state,
            u_val
        )

        cbf_const, Lfh_v, Lgh_v, cbf_v, h_pos_v, f_dang_v, e_perp_v, tr_v, pr_v = res

        # Store primary metrics
        history["s"].append(s_global)
        history["cbf"].append(float(cbf_v))
        history["h_pos"].append(float(h_pos_v))
        history["f_danger"].append(float(f_dang_v))
        history["Lfh"].append(float(Lfh_v))
        
        # Input Sensitivities (Lgh columns)
        # Column 0: Thrust, Columns 1-3: Omega
        lgh_array = np.array(Lgh_v).flatten()
        history["Lgh_thrust"].append(lgh_array[0] * u_val[0])
        # history["Lgh_omega_mag"].append(np.linalg.norm(lgh_array[1:4]))
        history["Lgh_omega_mag"].append(lgh_array[1:4] @ u_val[1:4])
        
        # Actual applied inputs
        history["u_thrust"].append(u_val[0])
        history["u_omega_mag"].append(np.linalg.norm(u_val[1:4]))
        
        history["pos"].append(x_state[:3])
        history["pr"].append(np.array(pr_v).flatten())
        history["e_perp_norm"].append(np.linalg.norm(e_perp_v))

        cbf_const = np.array(cbf_const).item()
        history["cbf_const"].append(cbf_const)

    # --- Plotting ---
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    s = history["s"]

    # 1. CBF vs Geometric Safety
    axs[0, 0].plot(s, history["h_pos"], 'r--', label="Geometric Safety (h_pos)")
    axs[0, 0].plot(s, history["cbf"], 'b-', label="CBF (Multiplicative)")
    axs[0, 0].set_title("Safety Values")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Danger Term f(v, q)
    axs[0, 1].plot(s, history["f_danger"], 'g-')
    axs[0, 1].set_title("Danger Metric f(v, q)\n(Positive = Danger High)")
    axs[0, 1].grid(True)

    # 3. Lie Derivative Drift (Lfh)
    axs[0, 2].plot(s, history["Lfh"], 'm-')
    axs[0, 2].set_title("Lfh")
    axs[0, 2].grid(True)

    # 4. Input Sensitivity Comparison (Lgh)
    # This shows which actuator the CBF "prefers" to use
    axs[1, 0].plot(s, history["Lgh_thrust"], label="LgH_thrust (Mag)")
    axs[1, 0].plot(s, history["Lgh_omega_mag"], label="LgH_omega (Norm)")
    axs[1, 0].set_title("Input Sensitivity (Authority)")
    axs[1, 0].set_yscale('log') # Useful as magnitudes can vary wildly
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 5. Applied Control Comparison
    axs[1, 1].plot(s, history["u_thrust"], color='orange', label="Thrust")
    ax_twin = axs[1, 1].twinx()
    ax_twin.plot(s, history["u_omega_mag"], color='cyan', label="Omega Mag")
    axs[1, 1].set_title("Applied Controls")
    axs[1, 1].legend(loc='upper left')
    ax_twin.legend(loc='upper right')

    # 6. Distance to Path
    axs[1, 2].plot(s, history["cbf_const"], 'k-')
    axs[1, 2].axhline(y=0.0, color='r', linestyle=':', label="CBF Constraint")
    axs[1, 2].set_title("CBF Constraint")
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.show()


# Usage in your main:
if __name__ == "__main__":
    ocp, cbf_func = create_ocp()
    env = VolaDroneEnv(ocp, "12gates", render_mode="human", normalize_obs=False, cbf_func=cbf_func)

    # Run visualization
    history = visualize_cbf_diagnostics(env, num_steps=500)

    # Check for issues
    if np.abs(history["Lfh"]).max() > 1e3:
        print("\n⚠️  WARNING: Lfh is very large! Check the exponent term p.")
        print(f"Max |p|: {np.abs(history['p']).max():.3f}")
        print(f"Max thrust→boundary: {np.abs(history['thrust_tb']).max():.3f}")
        print(f"Max v→boundary: {np.abs(history['v_tb']).max():.3f}")
