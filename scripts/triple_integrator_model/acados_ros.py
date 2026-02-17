#!/usr/bin/env python3

import time
import rospy
import numpy as np

from nav_msgs.msg import Odometry
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

        # ----------------------------
        # Load and resample trajectory
        # ----------------------------
        track = "straight_line"
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

        self.cmd_pub = rospy.Publisher(
            "/cmd_full_state", JointTrajectoryPoint, queue_size=1
        )

        self.current_state = np.zeros(self.nx)
        self.shift_applied = False
        self.odom_received = False

        self.timer = rospy.Timer(rospy.Duration(self.dt), self.control_loop)

        rospy.loginfo("MPCC ROS node ready.")

    # --------------------------------------------------
    def odom_callback(self, msg):

        if not self.shift_applied:

            # Compute shift from first odometry
            odom_pos = np.array(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ]
            )

            ref_start = np.array([self.x_ref[0], self.y_ref[0], self.z_ref[0]])

            shift = odom_pos - ref_start

            self.x_ref += shift[0]
            self.y_ref += shift[1]
            self.z_ref += shift[2]

            rospy.loginfo(f"Applied trajectory shift: {shift}")

            # Initialize state at odom
            self.current_state[0:3] = odom_pos

            self.shift_applied = True
            self.current_state[0] = self.x_ref[0]
            self.current_state[1] = self.y_ref[0]
            self.current_state[2] = self.z_ref[0]

        # Always update position + velocity
        # self.current_state[0] = msg.pose.pose.position.x
        # self.current_state[1] = msg.pose.pose.position.y
        # self.current_state[2] = msg.pose.pose.position.z
        #
        # self.current_state[3] = msg.twist.twist.linear.x
        # self.current_state[4] = msg.twist.twist.linear.y
        # self.current_state[5] = msg.twist.twist.linear.z

        self.odom_received = True

    # --------------------------------------------------
    def control_loop(self, event):

        if not self.odom_received or not self.shift_applied:
            return

        # Build parameter vector
        Q_c = 20
        Q_l = 100
        Q_j = 1
        Q_sdd = 1
        Q_s = 2.0

        params_vector = np.concatenate(
            [
                self.x_ref,
                self.y_ref,
                self.z_ref,
                self.vx_ref,
                self.vy_ref,
                self.vz_ref,
                [Q_c, Q_l, Q_j, Q_sdd, Q_s],
                [self.L_path],
            ]
        )

        # 1. Set initial state
        self.ocp_solver.set(0, "lbx", self.current_state)
        self.ocp_solver.set(0, "ubx", self.current_state)

        # 2. Update parameters across horizon
        for stage in range(self.N + 1):
            self.ocp_solver.set(stage, "p", params_vector)

        # 3. Solve
        start = time.time()
        status = self.ocp_solver.solve()
        print("time:", time.time() - start)

        if status != 0:
            rospy.logwarn("MPC solve failed")
            return

        x_next = self.ocp_solver.get(1, "x")
        u_opt = self.ocp_solver.get(0, "u")

        msg = JointTrajectoryPoint()
        msg.positions = x_next[0:3].tolist()
        msg.velocities = x_next[3:6].tolist()
        msg.accelerations = x_next[6:9].tolist()
        msg.effort = u_opt[0:3].tolist()

        self.current_state = x_next

        self.cmd_pub.publish(msg)


if __name__ == "__main__":
    try:
        node = MPCCNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
