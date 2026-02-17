#!/usr/bin/env python3

import time
import rospy
import numpy as np

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from trajectory_msgs.msg import JointTrajectoryPoint

from acados_template import AcadosOcpSolver, AcadosSimSolver
from common import getTrack
from acados_settings import create_ocp


class MPCCNode:

    def __init__(self):

        rospy.init_node("mpcc_controller")

        # ----------------------------
        # Build OCP
        # ----------------------------
        ocp, self.e_tot, self.ec_func = create_ocp()
        self.ocp_solver = AcadosOcpSolver(ocp)
        self.integrator = AcadosSimSolver(ocp)

        self.N = ocp.dims.N
        self.nx = ocp.model.x.rows()
        self.nu = ocp.model.u.rows()

        self.dt = 1.0 / self.N
        self.x_next = np.zeros(self.nx)

        # ----------------------------
        # Load and resample trajectory
        # ----------------------------
        track = "crazyflie_arclen_traj.txt"
        s_ref, x_ref, y_ref, z_ref, vx_ref, vy_ref, vz_ref = getTrack(track)

        n_knots = 20
        self.x_ref = np.interp(
            np.linspace(0, len(x_ref) - 1, n_knots), np.arange(len(x_ref)), x_ref
        )
        self.y_ref = np.interp(
            np.linspace(0, len(y_ref) - 1, n_knots), np.arange(len(y_ref)), y_ref
        )
        self.z_ref = np.interp(
            np.linspace(0, len(z_ref) - 1, n_knots), np.arange(len(z_ref)), z_ref
        )

        self.s_ref = np.interp(
            np.linspace(0, len(s_ref) - 1, n_knots), np.arange(len(s_ref)), s_ref
        )

        self.vx_ref = np.interp(
            np.linspace(0, len(vx_ref) - 1, n_knots), np.arange(len(vx_ref)), vx_ref
        )

        self.vy_ref = np.interp(
            np.linspace(0, len(vy_ref) - 1, n_knots), np.arange(len(vy_ref)), vy_ref
        )

        self.vz_ref = np.interp(
            np.linspace(0, len(vz_ref) - 1, n_knots), np.arange(len(vz_ref)), vz_ref
        )

        self.L_path = self.s_ref[-1]

        # ----------------------------
        # ROS I/O
        # ----------------------------
        self.odom_sub = rospy.Subscriber(
            "/odometry", Odometry, self.odom_callback, queue_size=1
        )

        self.imu_sub = rospy.Subscriber("/imu", Imu, self.imu_callback, queue_size=1)

        self.cmd_pub = rospy.Publisher(
            "/cmd_full_state", JointTrajectoryPoint, queue_size=1
        )

        self.current_state = np.zeros(self.nx - 2)
        self.shift_applied = False
        self.odom_received = False
        self.imu_received = False

        self.timer = rospy.Timer(rospy.Duration(self.dt), self.control_loop)

        rospy.loginfo("MPCC ROS node ready.")

    # --------------------------------------------------
    def odom_callback(self, msg):
        # Update position and velocity from odometry
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        self.current_state[2] = msg.pose.pose.position.z

        self.current_state[3] = msg.twist.twist.linear.x
        self.current_state[4] = msg.twist.twist.linear.y
        self.current_state[5] = msg.twist.twist.linear.z

        if not self.shift_applied:
            ref_start = np.array([self.x_ref[0], self.y_ref[0], self.z_ref[0]])
            shift = self.current_state[:3] - ref_start
            self.x_ref += shift[0]
            self.y_ref += shift[1]
            self.z_ref += shift[2]

            self.x_next[:3] = self.current_state[:3]

            self.shift_applied = True

        self.odom_received = True

    def quaternion_to_rotation_matrix(self, q):
        """Converts quaternion to rotation matrix (global to body frame)"""
        q0, q1, q2, q3 = q
        R = np.array(
            [
                [
                    1 - 2 * (q2**2 + q3**2),
                    2 * (q1 * q2 - q0 * q3),
                    2 * (q1 * q3 + q0 * q2),
                ],
                [
                    2 * (q1 * q2 + q0 * q3),
                    1 - 2 * (q1**2 + q3**2),
                    2 * (q2 * q3 - q0 * q1),
                ],
                [
                    2 * (q1 * q3 - q0 * q2),
                    2 * (q2 * q3 + q0 * q1),
                    1 - 2 * (q1**2 + q2**2),
                ],
            ]
        )
        return R

    # --------------------------------------------------
    def imu_callback(self, msg):
        # Get IMU linear acceleration in body frame
        acc_body = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]
        )

        # Get IMU orientation (quaternion)
        # q = np.array(
        #     [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        # )

        # Convert quaternion to rotation matrix (global to body)
        # R = self.quaternion_to_rotation_matrix(q)

        # Convert body frame accelerations to global frame
        # acc_global = np.dot(R, acc_body)

        # Store global accelerations (subtract gravity if needed)
        # self.current_state[6:9] = acc_global - np.array([0, 0, 9.81])
        self.current_state[6:9] = acc_body - np.array([0, 0, 9.81])

        self.imu_received = True

    # --------------------------------------------------
    def control_loop(self, event):

        # Wait until both odometry and IMU data have been received
        if not self.odom_received or not self.imu_received or not self.shift_applied:
            return

        # Build parameter vector for the current trajectory
        Q_c = 20
        Q_l = 100
        Q_j = 1
        Q_sdd = 1
        Q_s = 2.0

        # Use the current state as the "real" state
        # We don’t use a reference trajectory to update our state, this is closed-loop!
        params_vector = np.concatenate(
            [
                self.x_ref,
                self.y_ref,
                self.z_ref,
                self.vx_ref,
                self.vy_ref,
                self.vz_ref,
                [Q_c, Q_l, Q_j, Q_sdd, Q_s],  # Cost parameters
                [self.L_path],  # Path length
            ]
        )

        x0 = np.concatenate([self.current_state, self.x_next[-2:]])
        #
        # print(x0)
        # x0 = self.x_next
        # 1. Set initial state in MPC solver
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # 2. Update parameters for the whole horizon
        for stage in range(self.N + 1):
            self.ocp_solver.set(stage, "p", params_vector)

        # 3. Solve the MPC problem
        start = time.time()
        status = self.ocp_solver.solve()
        print("time:", time.time() - start)

        if status != 0:
            rospy.logwarn("MPC solve failed")
            return

        # Get the optimal state and control input from MPC solver
        self.x_next = self.ocp_solver.get(1, "x")
        u_opt = self.ocp_solver.get(0, "u")

        # Here you can send the control commands to actuators
        msg = JointTrajectoryPoint()
        msg.positions = self.x_next[0:3].tolist()  # Position
        msg.velocities = self.x_next[3:6].tolist()  # Velocity
        msg.accelerations = self.x_next[6:9].tolist()  # Acceleration (if relevant)
        msg.effort = u_opt[0:3].tolist()  # Control effort (thrust, etc.)

        # Publish control commands
        self.cmd_pub.publish(msg)


if __name__ == "__main__":
    try:
        node = MPCCNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
