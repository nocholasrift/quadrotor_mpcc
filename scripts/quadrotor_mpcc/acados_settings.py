#!/usr/bin/env python3

import sys
import time
import yaml
import argparse
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from sys_dynamics import SysDyn
from common import *
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

    # con_lower_bounds = np.array([-ACADOS_INFTY, -ACADOS_INFTY, -ACADOS_INFTY])
    # con_upper_bounds = np.array([0, ACADOS_INFTY, 0])
    con_lower_bounds = np.array([0, -ACADOS_INFTY])
    con_upper_bounds = np.array([ACADOS_INFTY, 0])
    # con_lower_bounds = np.array([-ACADOS_INFTY])
    # con_upper_bounds = np.array([0])

    # hard constraint
    ocp.constraints.uh_0 = con_upper_bounds
    ocp.constraints.lh_0 = con_lower_bounds
    ocp.constraints.uh = con_upper_bounds
    ocp.constraints.lh = con_lower_bounds
    ocp.constraints.uh_e = np.array([con_upper_bounds[-1]])
    ocp.constraints.lh_e = np.array([con_lower_bounds[-1]])

    # soft constraints
    # nsh = 1
    # ocp.constraints.lsh_0 = np.zeros((nsh,))
    # ocp.constraints.ush_0 = np.zeros((nsh,))
    # ocp.constraints.idxsh_0 = np.array([0])
    #
    # ocp.constraints.lsh = np.zeros((nsh,))
    # ocp.constraints.ush = np.zeros((nsh,))
    # ocp.constraints.idxsh = np.array([0])
    #
    # grad_cost = 1e4
    # hess_cost = 1e2
    #
    # ocp.cost.Zl_0 = hess_cost * np.ones((nsh,))
    # ocp.cost.Zu_0 = hess_cost * np.ones((nsh,))
    # ocp.cost.zl_0 = grad_cost * np.ones((nsh,))
    # ocp.cost.zu_0 = grad_cost * np.ones((nsh,))
    #
    # ocp.cost.Zl = hess_cost * np.ones((nsh,))
    # ocp.cost.Zu = hess_cost * np.ones((nsh,))
    # ocp.cost.zl = grad_cost * np.ones((nsh,))
    # ocp.cost.zu = grad_cost * np.ones((nsh,))

    # grad_cost = 1e4
    # hess_cost = 1e2
    # num_cbfs = 1
    # ocp.cost.Zl_0 = hess_cost * np.ones((num_cbfs,))
    # ocp.cost.Zu_0 = hess_cost * np.ones((num_cbfs,))
    # ocp.cost.zl_0 = grad_cost * np.ones((num_cbfs,))
    # ocp.cost.zu_0 = grad_cost * np.ones((num_cbfs,))
    #
    # ocp.cost.Zl = hess_cost * np.ones((num_cbfs,))
    # ocp.cost.Zu = hess_cost * np.ones((num_cbfs,))
    # ocp.cost.zl = grad_cost * np.ones((num_cbfs,))
    # ocp.cost.zu = grad_cost * np.ones((num_cbfs,))
    #
    ocp.constraints.lbu = np.array([0.01, -7.0, -7.0, -4.5, 0])
    ocp.constraints.ubu = np.array([0.40, 7.0, 7.0, 4.5, 3.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.x0 = np.zeros(nx)
    ocp.constraints.x0[6] = 1

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
    # ocp.solver_options.globalization_line_search_use_sufficient_descent = True
    # ocp.solver_options.levenberg_marquardt = 1e-4
    # ocp.solver_options.warm_start_first_qp = 1

    # ocp.solver_options.alpha_min = 0.05  # Default is 0.1, reduce if flickering
    # ocp.solver_options.alpha_reduction = 0.5  # Reduce aggressive steps

    # used these previously and they didn't help anything too much
    # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    # ocp.solver_options.nlp_solver_max_iter = 100
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.hpipm_mode = "ROBUST"
    # ocp.solver_options.hpipm_mode = "SPEED"
    # ocp.solver_options.qp_solver_iter_max = 100
    # ocp.solver_options.globalization_line_search_use_sufficient_descent = True

    return ocp, cbf_func


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
    ocp, _ = create_ocp()
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)
