import os
import sys

import time
#
# import motioncapture
# import cflib.crtp
#
# from threading import Thread
# from cflib.crazyflie import Crazyflie
# from cflib.crazyflie.mem import MemoryElement
# from cflib.crazyflie.mem import Poly4D
# from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
# from cflib.utils import uri_helper
# from cflib.utils.reset_estimator import reset_estimator
#
from motion_commander import * 
from connect_log_param import *

# super dirty way of doing things but alas... Desparate times.
mpcc_path = os.path.abspath("..") 
if mpcc_path not in sys.path:
    sys.path.append(mpcc_path)

from common import *
from tube_gen import *
from acados_settings import create_ocp, resample_path
from acados_template import AcadosOcpSolver

pos_data = None
vel_data = None
ori_data = None

def p_callback(timestamp, data, logconf):
    global pos_data
    pos_data = data

def v_callback(timestamp, data, logconf):
    global vel_data
    vel_data = data

def ori_callback(timestamp, data, logconf):
    global ori_data
    ori_data = data

def start_logging(logs):
    for log in logs:
        log.start()

def stop_logging(logs):
    for log in logs:
        log.stop()

def spin(cf, p_log, v_log, ori_log, solver, ocp):
    global pos_data, vel_data, ori_data
    commander = cf.high_level_commander
    ll_commander = cf.commander

    current_state = np.zeros(12)
    current_state[6] = 1

    T_horizon = ocp.solver_options.tf 
    n_steps = ocp.dims.N

    dt = T_horizon / float(n_steps)

    start_logging([p_log, v_log, ori_log])
    track_data = setup_track("short_line")

    params = np.array(
        [3.0, 400.0, 1, 5.0, 1.0, 0.3],  # Q_c, Q_l, Q_t, Q_w, Q_sdd, Q_s
    )

    tube_coeffs = get_free_tube(5, 5.0)

    while ori_data == None or vel_data == None or pos_data == None:
        print("waiting for data...")
        print(ori_data)
        print(vel_data)
        print(pos_data)
        time.sleep(0.1)

    commander.takeoff(1.2, 3.0)
    time.sleep(5.0)

    max_itr = 100
    for i in range(max_itr):
        start_t = time.time()
        s_global_now = current_state[10]
        local_window = get_local_window_params(track_data, s_global_now, n_knots, 4.0)
        param_dict = build_acados_params(local_window, params, tube_coeffs)
        local_p = dict_to_list(param_dict)
        local_p = np.concatenate([local_p, [5.0, 5.0]])

        stop_logging([p_log, v_log, ori_log])

        # current_state[0:3] = [pos_data['stateEstimate.x'], pos_data['stateEstimate.y'], pos_data['stateEstimate.z']]
        # current_state[3:6] = [vel_data['stateEstimate.vx'], vel_data['stateEstimate.vy'], vel_data['stateEstimate.vz']]
        # current_state[6:10] = [ori_data['stateEstimate.qw'], ori_data['stateEstimate.qx'], 
        #                        ori_data['stateEstimate.qy'], ori_data['stateEstimate.qz']]


        local_state = current_state.copy()
        local_state[10] = 0.0

        # Set integrator inputs
        solver.set(0, "lbx", local_state)
        solver.set(0, "ubx", local_state)

        for stage in range(n_steps + 1):
            solver.set(stage, "p", local_p)
            if stage < n_steps and i != 0:
                prev_x = solver.get(stage + 1, "x")
                solver.set(stage, "x", prev_x)

        next_local_state = solver.get(1, "x")
        global_s_next = s_global_now + next_local_state[10]
        current_state = next_local_state.copy()
        current_state[10] = global_s_next
        current_state[11] = next_local_state[11]

        status = solver.solve()
        u = solver.get(0, "u")
        x = solver.get(1, "x")
        # current_state = solver.

        start_logging([p_log, v_log, ori_log])

        vel = x[3:6]
        print("iter:", i, "\t", vel)
        ref = np.array([local_window["x"][0], local_window["y"][0], local_window["z"][0]])
        print("\tref:", ref)
        print("\tref:", current_state[:3])
        ll_commander.send_velocity_world_setpoint(vel[0], vel[1], vel[2], 0)


        if status != 0:
            break

        duration = time.time() - start_t
        sleep_t = max(0.0, dt - duration)
        time.sleep(sleep_t)
    
    stop_logging([p_log, v_log, ori_log])

    commander.land(0.0, 5.0)
    time.sleep(5.0)
    commander.stop()



def main():
    cflib.crtp.init_drivers()

    lg_p_stab = LogConfig(name='pos', period_in_ms=10)
    lg_p_stab.add_variable('stateEstimate.x', 'float')
    lg_p_stab.add_variable('stateEstimate.y', 'float')
    lg_p_stab.add_variable('stateEstimate.z', 'float')

    lg_v_stab = LogConfig(name='vel', period_in_ms=10)
    lg_v_stab.add_variable('stateEstimate.vx', 'float')
    lg_v_stab.add_variable('stateEstimate.vy', 'float')
    lg_v_stab.add_variable('stateEstimate.vz', 'float')

    lg_ori_stab = LogConfig(name='orientation', period_in_ms=10)
    lg_ori_stab.add_variable('stateEstimate.qx', 'float')
    lg_ori_stab.add_variable('stateEstimate.qy', 'float')
    lg_ori_stab.add_variable('stateEstimate.qz', 'float')
    lg_ori_stab.add_variable('stateEstimate.qw', 'float')

    ## Acados
    tube_degree = 5
    ocp, cbf_func = create_ocp(tube_degree)
    # solver = AcadosOcpSolver(ocp, generate=False, build=False, json_file="../acados_ocp.json")
    solver = AcadosOcpSolver(ocp, json_file="../acados_ocp.json")

    # Connect to the mocap system
    mocap_wrapper = MocapWrapper(rigid_body_name)

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        # Set up a callback to handle data from the mocap system
        mocap_wrapper.on_pose = lambda pose: send_extpose_quat(cf, pose[0], pose[1], pose[2], pose[3])

        adjust_orientation_sensitivity(cf)
        activate_kalman_estimator(cf)
        # activate_mellinger_controller(cf)
        reset_estimator(cf)

        cf.log.add_config(lg_p_stab)
        lg_p_stab.data_received_cb.add_callback(p_callback)
        cf.log.add_config(lg_v_stab)
        lg_v_stab.data_received_cb.add_callback(v_callback)
        cf.log.add_config(lg_ori_stab)
        lg_ori_stab.data_received_cb.add_callback(ori_callback)

        # Arm the Crazyflie
        cf.platform.send_arming_request(True)
        time.sleep(1.0)

        spin(cf, lg_p_stab, lg_v_stab, lg_ori_stab, solver, ocp)

    mocap_wrapper.close()

if __name__ == "__main__":
    main()
