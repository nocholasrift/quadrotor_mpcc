import os
import sys

# import time
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

def spin(cf, p_log, v_log, ori_log, solver):
    commander = cf.high_level_commander
    ll_commander = cf.commander

    current_state = np.zeros(11)
    current_state[6] = 1

    commander.takeoff(1.0, 3.0)
    time.sleep(5.0)

    start_logging(p_log, v_log, ori_log)

    track_data = self._setup_track("straight_line")

    max_itr = 100
    for i in range(max_itr):
        s_global_now = current_state[10]
        local_p = get_local_window_params(track_data, s_global_now)
        local_p = np.concatenate([local_p, 5.0])

        stop_logging(p_log, v_log, ori_log)

        current_state[0:3] = [pos_data['stateEstimate.x'], pos_data['stateEstimate.y'], pos_data['stateEstimate.z']]
        current_state[3:6] = [vel_data['stateEstimate.vx'], vel_data['stateEstimate.vy'], vel_data['stateEstimate.vz']]

        start_logging(p_log, v_log, ori_log)


        local_state = current_state.copy()
        local_state[10] = 0.0

        # Set integrator inputs
        solver.set(0, "lbx", local_state)
        solver.set(0, "ubx", local_state)

        for stage in range(self.N + 1):
            # self.solver.set(stage, "p", self.params)
            self.solver.set(stage, "p", local_p)
            if stage < N:
                prev_x = solver.get(stage + 1, "x")
                self.solver.set(stage, "x", prev_x)

        next_local_state = solver.get(1, "x")
        global_s_next = s_global_now + next_local_state[10]
        current_state[10] = global_s_next
        current_state[11] = next_local_state[11]

        status = solver.solve()
        u = solver.get(0, "u")

        if status != 0:
            break

        time.sleep(0.01)
    
    stop_logging(p_log, v_log, ori_log)

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
    ocp, cbf_func = create_ocp()
    solver = AcadosOcpSolver(ocp, generate=False, build=False, json_file="../acados_ocp.json")

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

        spin(cf, lg_p_stab, lg_v_stab, lg_ori_stab, solver)

    mocap_wrapper.close()

if __name__ == "__main__":
    main()
