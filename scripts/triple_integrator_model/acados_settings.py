#!/usr/bin/env python3

import sys
import time
import yaml
import argparse
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from sys_dynamics import SysDyn
from common import getTrack, n_knots
from mpl_toolkits.mplot3d import Axes3D
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, ACADOS_INFTY


def create_ocp():

    ocp = AcadosOcp()

    # set model
    # model = export_mpcc_ode_model(list(ss), list(xs), list(ys))
    dynamics = SysDyn()
    [model, cbf_func] = dynamics.setup()
    ocp.model = model

    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    nparams = model.p.rows()
    N = 20

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.dims.N = N
    ocp.parameter_values = np.zeros((nparams,))

    ocp.model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    con_upper_bounds = np.array([ACADOS_INFTY, 0])
    con_lower_bounds = np.array([0, -ACADOS_INFTY])

    # hard constraint
    ocp.constraints.uh_0 = con_upper_bounds
    ocp.constraints.lh_0 = con_lower_bounds
    ocp.constraints.uh = con_upper_bounds
    ocp.constraints.lh = con_lower_bounds
    ocp.constraints.uh_e = np.array([con_upper_bounds[-1]])
    ocp.constraints.lh_e = np.array([con_lower_bounds[-1]])

    ocp.constraints.lbu = np.array([-50, -50, -50, -7])
    ocp.constraints.ubu = np.array([50, 50, 50, 7])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # theta can be whatever
    ocp.constraints.lbx = np.array(
        [-ACADOS_INFTY, -ACADOS_INFTY, -ACADOS_INFTY, -3, -3, -3, -8, -8, -7, 0, 0]
    )
    ocp.constraints.ubx = np.array(
        [ACADOS_INFTY, ACADOS_INFTY, ACADOS_INFTY, 3, 3, 3, 8, 8, 5, ACADOS_INFTY, 3.1]
    )
    ocp.constraints.idxbx = np.array(range(nx))  # Covers all state indices

    ocp.constraints.x0 = np.zeros(nx)

    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N
    ocp.solver_options.shooting_nodes = np.linspace(0, Tf, N + 1)

    # Partial is slightly slower but more stable allegedly than full condensing.
    # ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.hessian_approx = "EXACT"
    # ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.nlp_solver_type = "SQP"
    # # sometimes solver failed due to NaNs, regularizing Hessian helped
    # ocp.solver_options.regularize_method = "MIRROR"
    # # ocp.solver_options.tol = 1e-4

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # sometimes solver failed due to NaNs, regularizing Hessian helped
    ocp.solver_options.regularize_method = "MIRROR"
    # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.globalization_line_search_use_sufficient_descent = True
    # ocp.solver_options.levenberg_marquardt = 1e-4
    # ocp.solver_options.warm_start_first_qp = 1

    # ocp.solver_options.alpha_min = 0.05  # Default is 0.1, reduce if flickering
    # ocp.solver_options.alpha_reduction = 0.5  # Reduce aggressive steps

    # used these previously and they didn't help anything too much
    # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    # ocp.solver_options.nlp_solver_max_iter = 100
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.hpipm_mode = "ROBUST"
    ocp.solver_options.hpipm_mode = "SPEED"
    # ocp.solver_options.qp_solver_iter_max = 100
    # ocp.solver_options.globalization_line_search_use_sufficient_descent = True

    return ocp, cbf_func


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


def get_closest_idx(s_ref, x_ref, y_ref, z_ref, state):
    p = state[:3]

    # N, 3 matrix
    p_ref = np.stack([x_ref, y_ref, z_ref], axis=1)

    dist_sq = np.sum((p_ref - p) ** 2, axis=1)
    idx_min = np.argmin(dist_sq)

    return idx_min


def run_simulation(ocp_solver, integrator, e_tot, ec_func, sim_steps=1000):
    """
    initial_state: np.array [11,] (px, py, pz, vx, vy, vz, ax, ay, az, s, s_dot)
    params_vector: np.array [n_params] (The weights and path coeffs you defined)
    """

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    N = ocp_solver.acados_ocp.dims.N

    # track="trefoil_track.txt"
    track = "crazyflie_arclen_traj.txt"
    [s_ref, x_ref, y_ref, z_ref, vx_ref, vy_ref, vz_ref] = getTrack(track)

    x_ref_sampled = resample_path(x_ref, n_knots)
    y_ref_sampled = resample_path(y_ref, n_knots)
    z_ref_sampled = resample_path(z_ref, n_knots)
    vx_ref_sampled = resample_path(vx_ref, n_knots)
    vy_ref_sampled = resample_path(vy_ref, n_knots)
    vz_ref_sampled = resample_path(vz_ref, n_knots)
    s_ref_sampled = resample_path(s_ref, n_knots)

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        x_ref_sampled, y_ref_sampled, z_ref_sampled, "k--", alpha=0.3, label="Track"
    )

    # Create the objects we will update
    (drone_dot,) = ax.plot([], [], [], "ro", markersize=8, label="Crazyflie")
    (horizon_line,) = ax.plot([], [], [], "b-", alpha=0.5, label="MPC Horizon")
    (path_trail,) = ax.plot([], [], [], "g-", alpha=0.2)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()

    p_ref = np.stack([x_ref_sampled, y_ref_sampled, z_ref_sampled], axis=1)

    initial_state = np.zeros(nx)
    initial_state[0] = x_ref_sampled[0]
    initial_state[1] = y_ref_sampled[0]
    initial_state[2] = z_ref_sampled[0]

    # Initialize buffers for logging
    state_history = np.zeros((sim_steps + 1, nx))
    control_history = np.zeros((sim_steps, nu))

    current_x = initial_state
    state_history[0, :] = current_x

    Q_c = 20
    Q_l = 100
    Q_j = 1
    Q_sdd = 1
    Q_s = 1.0
    L_path = s_ref_sampled[-1]

    params_vector = np.concatenate(
        [
            x_ref_sampled,
            y_ref_sampled,
            z_ref_sampled,
            vx_ref_sampled,
            vy_ref_sampled,
            vz_ref_sampled,
            [Q_c, Q_l, Q_j, Q_sdd, Q_s],
            [L_path],
        ]
    )

    print(f"Starting simulation for {sim_steps} steps...")

    for i in range(sim_steps):
        # 1. Set current state as initial constraint for OCP
        ocp_solver.set(0, "lbx", current_x)
        ocp_solver.set(0, "ubx", current_x)

        # 2. Update parameters across the entire horizon
        # (Very important for MPCC to know the path ahead!)
        for stage in range(N + 1):
            ocp_solver.set(stage, "p", params_vector)

        # 3. Solve the OCP
        start = time.time()
        status = ocp_solver.solve()
        print("solved in", (time.time() - start))
        if status != 0:
            print(
                f"Step {i}: OCP solver failed with status {status}. High-five a wall?"
            )
            break

        # x_pred is (N+1, nx)
        x_pred = np.array([ocp_solver.get(j, "x") for j in range(N + 1)])

        # 2. Update the Plot Objects
        drone_dot.set_data([current_x[0]], [current_x[1]])
        drone_dot.set_3d_properties([current_x[2]])

        horizon_line.set_data(x_pred[:, 0], x_pred[:, 1])
        horizon_line.set_3d_properties(x_pred[:, 2])

        # Optionally update the trail of where the drone has been
        path_trail.set_data(state_history[: i + 1, 0], state_history[: i + 1, 1])
        path_trail.set_3d_properties(state_history[: i + 1, 2])

        # 3. Force Matplotlib to draw the update
        plt.draw()
        plt.pause(0.01)  # Small pause to allow GUI thread to catch up

        # 4. Get the first optimal control action
        u_opt = ocp_solver.get(0, "u")
        control_history[i, :] = u_opt

        # 5. Integrate dynamics to get the next state
        integrator.set("x", current_x)
        integrator.set("u", u_opt)

        # We must also give the parameters to the integrator!
        integrator.set("p", params_vector)

        status_sim = integrator.solve()
        if status_sim != 0:
            raise Exception(f"Integrator failed at step {i}")

        current_x = integrator.get("x")
        state_history[i + 1, :] = current_x

        # Optional: Print progress every 10 steps
        idx = get_closest_idx(
            s_ref_sampled, x_ref_sampled, y_ref_sampled, z_ref_sampled, current_x
        )
        if idx > len(s_ref_sampled) - 5:
            break
        # error = np.linalg.norm(p_ref[idx] - current_x[:3])
        # e_tot_val = e_tot(x_ref_sampled, y_ref_sampled, z_ref_sampled, L_path, current_x)
        # ec_s = ec_func(x_ref_sampled, y_ref_sampled, z_ref_sampled, L_path, current_x)
        # print(f"Step {i}: s = {current_x[9]:.2f}, s_dot = {current_x[10]:.2f}, error = {error:.2f}")
        # print(f"\tec: {ec_s}\tec_norm: {np.linalg.norm(ec_s)}")
        # print(f"\te_tot: {e_tot_val}\tetot_norm: {np.linalg.norm(e_tot_val)}")

    plt.ioff()
    plt.show()
    return state_history, control_history, x_ref, y_ref, z_ref


if __name__ == "__main__":
    # ocp = create_ocp()
    # acados_ocp_solver = AcadosOcpSolver(ocp)

    # parser = argparse.ArgumentParser(description="test BARN navigation challenge")
    # parser.add_argument("--yaml", type=str, default="")
    # parser.add_argument("--output_dir", type=str, default="")
    # parser.add_argument("--casadi_dir", type=str, default="")

    # args = parser.parse_args()
    #
    # ocp = create_ocp(args.yaml, args.casadi_dir)
    # if args.output_dir != "":
    #     ocp.code_export_directory = args.output_dir

    # ocp = create_ocp_dyna_obs(args.yaml)
    ocp, tr_func, ec_func = create_ocp()
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

    states, controls = run_simulation(
        acados_ocp_solver, acados_integrator, tr_func, ec_func
    )
