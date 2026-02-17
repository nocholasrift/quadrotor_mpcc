import time
import matplotlib.pyplot as plt
from common import *
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from acados_settings import create_ocp, resample_path
from acados_template import AcadosOcpSolver, AcadosSimSolver

cbf_func = None


class VolaDroneEnv(gym.Env):
    def __init__(self, ocp, track, render_mode=None):
        super().__init__()
        self.solver = AcadosOcpSolver(ocp)
        self.integrator = AcadosSimSolver(ocp)
        self.N = ocp.dims.N
        self.nx = ocp.model.x.rows()  # [px, py, pz, vx, vy, vz, ax, ay, az, s, s_dot]
        self.nu = ocp.model.u.rows()  # [jx, jy, jz, s_ddot]
        self.alpha0 = 10.0
        self.alpha1 = 10.0
        self.alpha2 = 10.0

        # Load Track metadata
        self.track_data = self._setup_track(track)
        self.gate_data = self._setup_gates(track)
        self.render_mode = render_mode
        self.state = None

        # Plotting objects for rendering
        self.fig = None

    def _setup_track(self, track):
        [s, x, y, z, vx, vy, vz] = getTrack(track)
        (
            vxref,
            vyref,
            vzref,
        ) = interpolLUT(vx, vy, vz, s)
        T, e1, e2 = getRMFBasis(vxref, vyref, vzref, s)

        n = 50
        return {
            "s": resample_path(s, n),
            "x": resample_path(x, n),
            "y": resample_path(y, n),
            "z": resample_path(z, n),
            "vx": resample_path(vx, n),
            "vy": resample_path(vy, n),
            "vz": resample_path(vz, n),
            "e1x": resample_path(e1[0], n),
            "e1y": resample_path(e1[1], n),
            "e1z": resample_path(e1[2], n),
            "e2x": resample_path(e2[0], n),
            "e2y": resample_path(e2[1], n),
            "e2z": resample_path(e2[2], n),
            "L": s[-1],
        }

    def _setup_gates(self, track):
        return load_gates(track)

    def _get_local_window_params(self, s_global_now, window_dist=4.0):
        s_orig = self.track_data["s"]
        s_end = s_global_now + window_dist

        # Create the evaluation points for the knots
        # We sample knots over the window_dist
        s_query = np.linspace(s_global_now, s_end, n_knots)

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
            new_knots[key] = np.interp(s_query, s_orig, self.track_data[key])

        # Construct parameter vector
        # IMPORTANT: The last parameter is the LENGTH of this local segment
        local_p = np.concatenate(
            [
                new_knots["x"],
                new_knots["y"],
                new_knots["z"],
                new_knots["vx"],
                new_knots["vy"],
                new_knots["vz"],
                new_knots["e1x"],
                new_knots["e1y"],
                new_knots["e1z"],
                new_knots["e2x"],
                new_knots["e2y"],
                new_knots["e2z"],
                [20.0, 100.0, 1.0, 1.0, 1.0],  # Q_c, Q_l, Q_j, Q_sdd, Q_s
                [window_dist],  # This MUST match the distance used in s_query
            ]
        )
        return local_p

    def reset(self, seed=None):
        super().reset(seed=seed)
        # 1. Reset state to track start
        self.state = np.zeros(self.nx)
        self.state[0] = self.track_data["x"][0]
        self.state[1] = self.track_data["y"][0]
        self.state[2] = self.track_data["z"][0]
        # self.state[10] = 0.2  # Initial forward kick for MPCC progress

        # 2. Re-calculate parameter vector for the solver
        self.params = np.concatenate(
            [
                self.track_data["x"],
                self.track_data["y"],
                self.track_data["z"],
                self.track_data["vx"],
                self.track_data["vy"],
                self.track_data["vz"],
                self.track_data["e1x"],
                self.track_data["e1y"],
                self.track_data["e1z"],
                self.track_data["e2x"],
                self.track_data["e2y"],
                self.track_data["e2z"],
                [20.0, 100.0, 1.0, 1.0, 1.0],  # Q_c, Q_l, Q_j, Q_sdd, Q_s
                [self.track_data["L"]],
            ]
        )
        return self.state, {}

    def step(self, action):
        s_global_now = self.state[9]
        local_p = self._get_local_window_params(s_global_now)

        unnormed_action = action_unnormalize(action, min_alpha_dot, max_alpha_dot)
        self.alpha0 += unnormed_action[0]
        self.alpha1 += unnormed_action[1]
        self.alpha2 += unnormed_action[2]
        alphas = np.array([self.alpha0, self.alpha1, self.alpha2])

        local_p = np.concatenate([local_p, alphas])

        local_state = self.state.copy()
        local_state[9] = 0.0

        # Set integrator inputs
        self.solver.set(0, "lbx", local_state)
        self.solver.set(0, "ubx", local_state)
        # self.solver.set(0, "lbx", self.state)
        # self.solver.set(0, "ubx", self.state)

        for stage in range(self.N + 1):
            # self.solver.set(stage, "p", self.params)
            self.solver.set(stage, "p", local_p)
            if stage < self.N:
                prev_x = self.solver.get(stage + 1, "x")
                # We must subtract the progress made in the last step
                # to keep the horizon consistent with s=0 at start
                prev_x[9] -= (
                    next_s_from_prev_step if "next_s_from_prev_step" in locals() else 0
                )
                self.solver.set(stage, "x", prev_x)

        # Evolve physics
        start = time.time()
        self.solver.solve()
        print("solve time:", time.time() - start)

        next_local_state = self.solver.get(1, "x")
        global_s_next = s_global_now + next_local_state[9]
        self.state = next_local_state.copy()
        self.state[9] = global_s_next

        u = self.solver.get(0, "u")
        gym_obs = self._get_obs(local_p, local_state, u, s_global_now)

        # Termination: Finished 99% of the track
        terminated = self.state[9] >= self.track_data["L"] * 0.99

        # Truncation: Drone flew way off course (safety check)
        # Using 5 meters from start as a simple failure condition
        truncated = np.linalg.norm(self.state[:3]) > 100.0

        return gym_obs, 0.0, terminated, truncated, {}

    def _get_obs(self, local_p, local_state, u_applied, s_global_now):
        n = n_knots
        N = self.N  # Total horizon steps (e.g., 20)
        M = 5  # Number of samples we want
        stride = N // M  # e.g., 4

        horizon_obs = []

        # We loop through the horizon at the specified stride
        for i in range(0, N, stride):
            # 1. Get the predicted state and control at this horizon step
            x_i = self.solver.get(i, "x")
            u_i = self.solver.get(i, "u") if i < N else u_applied

            # 2. Call the CasADi function for this specific state
            # Note the corrected indices for the spline coefficients!
            safety_terms = cbf_func(
                local_p[0:n],  # x
                local_p[n : 2 * n],  # y
                local_p[2 * n : 3 * n],  # z
                local_p[3 * n : 4 * n],  # vx
                local_p[4 * n : 5 * n],  # vy
                local_p[5 * n : 6 * n],  # vz
                local_p[6 * n : 7 * n],  # e1x
                local_p[7 * n : 8 * n],  # e1y
                local_p[8 * n : 9 * n],  # e1z
                local_p[9 * n : 10 * n],  # e2x
                local_p[10 * n : 11 * n],  # e2y
                local_p[11 * n : 12 * n],  # e2z
                local_p[-1],  # L_window (usually last element)
                x_i,  # Use the horizon state
            )

            # 3. Extract and compute the Lie term
            lf3h = float(safety_terms[0])
            lglf2h = np.array(safety_terms[1]).flatten()  # (4,)
            lf2h = float(safety_terms[2])
            lfh = float(safety_terms[3])
            cbf = float(safety_terms[4])

            lglf2h_u = np.dot(lglf2h, u_i)

            # 4. Store this "safety snapshot"
            horizon_obs.extend([lf3h, float(lglf2h_u), lf2h, lfh, cbf])

        # Add the current global position at the very end
        full_obs = np.array(horizon_obs + [float(s_global_now)], dtype=np.float32)
        return full_obs

    def render(self):
        if self.render_mode == "human":
            if self.fig is None:
                plt.ion()
                self.fig = plt.figure(figsize=(8, 6))
                self.ax = self.fig.add_subplot(111, projection="3d")
                self.ax.plot(
                    self.track_data["x"],
                    self.track_data["y"],
                    self.track_data["z"],
                    "k--",
                    alpha=0.5,
                )

                draw_gates(self.gate_data, self.ax)
                draw_corridor(
                    self.ax,
                    self.track_data["x"],
                    self.track_data["y"],
                    self.track_data["z"],
                    self.track_data["vx"],
                    self.track_data["vy"],
                    self.track_data["vz"],
                    np.linspace(0, self.track_data["L"], len(self.track_data["x"])),
                    radius=0.5,
                    color="c",
                    alpha=0.2,
                )

                (self.drone_marker,) = self.ax.plot([], [], [], "ro", markersize=10)
                (self.trail,) = self.ax.plot([], [], [], "g-", alpha=0.3)
                (self.horizon_line,) = self.ax.plot([], [], [], "b-", alpha=0.5)

                self.history = []

            self.history.append(self.state[:3].copy())
            hist = np.array(self.history)

            self.drone_marker.set_data([self.state[0]], [self.state[1]])
            self.drone_marker.set_3d_properties([self.state[2]])
            self.trail.set_data(hist[:, 0], hist[:, 1])
            self.trail.set_3d_properties(hist[:, 2])

            horizon_states = []

            for i in range(self.N + 1):
                try:
                    x_pred = self.solver.get(i, "x")
                    horizon_states.append(x_pred[:3])
                except:
                    break

            if len(horizon_states) > 0:
                horizon_states = np.array(horizon_states)

                self.horizon_line.set_data(
                    horizon_states[:, 0],
                    horizon_states[:, 1],
                )
                self.horizon_line.set_3d_properties(horizon_states[:, 2])

            plt.draw()
            plt.pause(1e-4)


if __name__ == "__main__":
    ocp, cbf_func = create_ocp()

    env = VolaDroneEnv(ocp, "straight_line", render_mode="human")

    env.reset()
    for i in range(0, 1000):
        _, _, done, _, _ = env.step(np.array([0, 0, 0]))
        env.render()

        if done:
            break
        # input()
