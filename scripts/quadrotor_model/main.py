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

import numpy as np
from time import time
import casadi as ca
from common import *
from acados_settings import AcadosCustomOcp
from visualize_mpl import animOptVars

def plan_ocp(ocp_wrapper: AcadosCustomOcp):
    '''Motion control problem of drone trajectory tracking'''

    # dimensions
    nx = ocp_wrapper.nx
    nu = ocp_wrapper.nu

    # initialize iteration variables
    t0 = 0
    mpc_iter = 0
    cost = 0
    times = []

    # Initialize data collection lists (More robust than iterative dstack)
    # We store the initial state horizon and control
    state_history = [np.copy(ocp_wrapper.zeta_N)] 
    control_history = [ocp_wrapper.u_N[:, 0].reshape((nu, 1))]
    misc_history = [np.array([0, 0]).reshape(2, 1)]

    # Reference state for the loop
    zeta_N = ocp_wrapper.zeta_N

    # Control loop entry point
    for i in range(Nsim):
        t1 = time()

        # Update reference and check if track is complete
        end = ocp_wrapper.cost_update_ref(zeta_N[:, 0], U_HOV)
        if (end):
            print("Track complete !")
            break

        # Solve the OCP
        ocp_wrapper.solve_and_sim()

        # Update time and log solver performance
        t0 = round(t0 + T_del, 3)
        t2 = time()
        times.append(t2 - t1)
        mpc_iter += 1

        # Retrieve current solution horizon
        zeta_N = ocp_wrapper.zeta_N

        # --- Data Collection ---
        # Append 2D slices to our lists
        state_history.append(np.copy(ocp_wrapper.zeta_N))
        control_history.append(ocp_wrapper.u_N[:, 0].reshape((nu, 1)))
        misc_history.append(np.array([t0, cost]).reshape(2, 1))

        print(f'\n Soln. {mpc_iter} Sim: {(np.round(zeta_N[:, 0], 2).T)} at {round(t0, 2)} s\t')

    # --- Post-Processing ---
    # Convert lists to homogeneous NumPy arrays
    # state_steps shape: (nx, N+1, mpc_iter)
    state_steps = np.stack(state_history, axis=-1)
    
    # control_steps shape: (nu, mpc_iter)
    control_steps = np.concatenate(control_history, axis=1)
    
    # misc_step shape: (2, mpc_iter)
    misc_step = np.concatenate(misc_history, axis=1)

    # Statistics
    if times:
        sqp_max_sec = round(max(times), 3)
        sqp_avg_sec = round(sum(times) / len(times), 3)
        avg_n = round(np.abs(state_steps[1, 0, :]).mean(), 4)
        avg_b = round(np.abs(state_steps[2, 0, :]).mean(), 4)

        print(f'Max. solver time\t\t: {sqp_max_sec * 1000} ms')
        print(f'Avg. solver time\t\t: {sqp_avg_sec * 1000} ms')
        print(f'Avg. lateral deviation n\t: {avg_n} m')
        print(f'Avg. vertical deviation b\t: {avg_b} m')

    return misc_step, state_steps, control_steps

if __name__ == '__main__':

    custom_ocp = AcadosCustomOcp()
    custom_ocp.setup_acados_ocp()
    traj_sample, traj_ST, traj_U = plan_ocp(custom_ocp)
    #  animated plot
    animOptVars(traj_sample, traj_ST, traj_U)
