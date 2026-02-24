import time
import click
import matplotlib.pyplot as plt
from common import *
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from scipy.spatial import cKDTree

from acados_settings import create_ocp, resample_path
from acados_template import AcadosOcpSolver, AcadosSimSolver

from load_env import load_pcl_from_env
from tube_gen import *


class VolaDroneEnv(gym.Env):
    def __init__(
        self,
        track,
        solver=None,
        integrator=None,
        render_mode=None,
        normalize_obs=True,
    ):
        super().__init__()

        try:
            self.pcl = load_pcl_from_env(f"../../resources/envs/{track}.yaml")
        except:
            self.pcl = []

        self.should_normalize_obs = normalize_obs

        self.alpha0 = 1.0
        self.alpha1 = 1.0

        # Load Track metadata
        self.track = track
        self.track_data = self._setup_track(track)
        x = self.track_data["x"]
        y = self.track_data["y"]
        z = self.track_data["z"]

        traj = np.stack([x, y, z], axis=1)

        self.track_kdtree = cKDTree(traj)
        self.track_horizon_window = 4.0
        self.max_tube_radius = 1.0
        self.tube_degree = 5
        self.tube_coeffs = np.zeros((4, self.tube_degree + 1))

        self.prev_s = 0

        self.render_mode = render_mode
        self.state = None

        # Plotting objects for rendering
        self.fig = None

        self.n_consecutive_infeasibilities = 0

        if self.should_normalize_obs:
            stats = np.load(f"stats/{track}_normalization_stats.npz")
            self.obs_mean = stats["obs_mean"]
            self.obs_std = stats["obs_std"]


        ocp, self.cbf_func = create_ocp(self.tube_degree)
        if not solver:
            self.solver = AcadosOcpSolver(ocp, build=False, generate=False)
            # self.solver = AcadosOcpSolver(ocp)
        else:
            self.solver = solver

        if not integrator:
            self.integrator = AcadosSimSolver(ocp, build=False, generate=False)
            # self.integrator = AcadosSimSolver(ocp)
        else:
            self.integrator = integrator

        self.N = ocp.dims.N
        self.M = 3
        self.nx = ocp.model.x.rows()  # [px, py, pz, vx, vy, vz, ax, ay, az, s, s_dot]
        self.nu = ocp.model.u.rows()  # [jx, jy, jz, s_ddot]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self.M + 1,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # self.reset()

    def _setup_track(self, track):
        [s, x, y, z, vx, vy, vz] = getTrack(track)
        (
            vxref,
            vyref,
            vzref,
        ) = interpolLUT(vx, vy, vz, s)
        T, e1, e2 = getRMFBasis(vxref, vyref, vzref, s)

        n = int(s[-1] * 10)
        return {
            "s": resample_path(s, n),
            "x": resample_path(x, n),
            "y": resample_path(y, n),
            "z": resample_path(z, n),
            "vx": resample_path(vx, n),
            "vy": resample_path(vy, n),
            "vz": resample_path(vz, n),
            "e1x": resample_path(e1[:, 0], n),
            "e1y": resample_path(e1[:, 1], n),
            "e1z": resample_path(e1[:, 2], n),
            "e2x": resample_path(e2[:, 0], n),
            "e2y": resample_path(e2[:, 1], n),
            "e2z": resample_path(e2[:, 2], n),
            "L": s[-1],
        }

    def _setup_gates(self, track):
        return load_gates(track)

    def normalize_obs(self, obs, mean, std):
        obs[:-1] = (obs[:-1] - mean[:-1]) / std[:-1]
        obs[-1] = 2 * (obs[-2] / self.track_data["s"][-1]) - 1.0
        # obs[-2] = 2 * (obs[-2] - min_alpha) / (max_alpha - min_alpha) - 1
        # obs[-1] = 2 * (obs[-1] - min_alpha) / (max_alpha - min_alpha) - 1

        return obs

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.nx)
        self.state[0] = self.track_data["x"][0]
        self.state[1] = self.track_data["y"][0]
        self.state[2] = self.track_data["z"][0]
        self.state[6] = 1.0

        self.n_consecutive_infeasibilities = 0
        self.prev_s = 0

        for stage in range(self.N + 1):
            self.solver.set(stage, "x", self.state)
            if stage < self.N:
                self.solver.set(stage, "u", np.zeros(self.nu))

        self.params = np.array(
                [3.0, 400.0, 1, 5.0, 1.0, 0.3],  # Q_c, Q_l, Q_t, Q_w, Q_sdd, Q_s
        )

        self.tube_coeffs = get_free_tube(self.tube_degree, self.max_tube_radius)

        local_window = get_local_window_params(self.track_data, 0, n_knots, window_dist=self.track_horizon_window)
        local_p = build_acados_params(local_window, self.params, self.tube_coeffs)
        return self._get_obs(local_p, 0), {}

    def step(self, action=None):
        s_global_now = self.state[10]
        self.prev_s = s_global_now
        local_window = get_local_window_params(self.track_data, s_global_now, n_knots, window_dist=self.track_horizon_window)

        # unnormed_action = action_unnormalize(action, min_alpha_dot, max_alpha_dot)
        # self.alpha0 += unnormed_action[0]
        if type(action) is np.ndarray:
            unnormed_action = action_unnormalize(action, min_alpha, max_alpha)
            self.alpha0 = unnormed_action[0]
            self.alpha1 = unnormed_action[1]
            # print(self.alpha0)

        start = time.time()

        if len(self.pcl) > 0:
            occ_data = project_cloud_to_parametric_path(self.pcl, self.track_data, self.track_kdtree, max_radius=self.max_tube_radius)
            occ_data[:,0] -= s_global_now
            mask = (occ_data[:,0] >= 0) & (occ_data[:,0] <= local_window["L"])
            occ_data = occ_data[mask]

            # if occ_data.shape[0] > 0:
            solver, coeffs = NLP(self.tube_degree, occ_data, local_window["L"], self.max_tube_radius, True)
            solver.solve(solver=cp.CLARABEL, verbose=False)
            a, b, c, d = coeffs
            self.tube_coeffs[0,:] = a.value
            self.tube_coeffs[1,:] = b.value
            self.tube_coeffs[2,:] = c.value
            self.tube_coeffs[3,:] = d.value
            # else:
            #     self.tube_coeffs = get_free_tube(self.tube_degree, self.max_tube_radius)

        # print(self.tube_coeffs)

        param_dict = build_acados_params(local_window, self.params, self.tube_coeffs)
        alphas = np.array([self.alpha0, self.alpha1]).reshape((2,))

        local_p = dict_to_list(param_dict)
        local_p = np.concatenate([local_p, alphas])

        local_state = self.state.copy()
        local_state[10] = 0.0

        # Set integrator inputs
        self.solver.set(0, "lbx", local_state)
        self.solver.set(0, "ubx", local_state)

        for stage in range(self.N + 1):
            # self.solver.set(stage, "p", self.params)
            self.solver.set(stage, "p", local_p)
            if stage < self.N:
                prev_x = self.solver.get(stage + 1, "x")
                # We must subtract the progress made in the last step
                # to keep the horizon consistent with s=0 at start
                prev_x[10] -= (
                    next_s_from_prev_step if "next_s_from_prev_step" in locals() else 0
                )
                self.solver.set(stage, "x", prev_x)


        # pad_amt = max_occ_points - occ_data.shape[0]
        # pad_val = [0, 10, 10]
        # padded_occ_data = np.pad(occ_data, (0, pad_amt), mode='constant', constant_values=pad_val)

        # Evolve physics
        start = time.time()
        status = self.solver.solve()

        # print("solve time:", time.time() - start)

        next_local_state = self.solver.get(1, "x")
        q_norm = np.linalg.norm(next_local_state[6:10])
        next_local_state[6:10] /= q_norm

        global_s_next = s_global_now + next_local_state[10]
        self.state = next_local_state.copy()
        self.state[10] = global_s_next

        u = self.solver.get(0, "u")
        gym_obs = self._get_obs(param_dict, global_s_next)
        # gym_obs = []

        # print(f"{s_global_now} / {self.params[-1]}")

        # Termination: Finished 99% of the track
        terminated = bool(self.state[10] >= self.track_data["L"] - self.track_horizon_window)

        # Truncation: Drone flew way off course (safety check)
        truncated = (
            bool(np.linalg.norm(self.state[:3]) > 100.0)
            or self.n_consecutive_infeasibilities >= 3
        )

        reward = self._get_reward(gym_obs, self.state[-1], terminated, status)

        # print("un-norm", gym_obs)
        if self.should_normalize_obs:
            gym_obs = self.normalize_obs(gym_obs, self.obs_mean, self.obs_std)

        # print("norm", gym_obs)
        return gym_obs, reward, terminated, truncated, {}

    def _get_obs(self, local_p, s):
        n = n_knots
        N = self.N
        M = self.M

        inds = np.linspace(1, N - 1, num=M, dtype=int)
        obs = []

        for i in inds:

            x_i = self.solver.get(int(i), "x")
            u_i = self.solver.get(int(i), "u")

            
            hddot, lfh, cbf = self.cbf_func(
                local_p["x"],
                local_p["y"],
                local_p["z"],
                local_p["vx"],
                local_p["vy"],
                local_p["vz"],
                local_p["e1x"],
                local_p["e1y"],
                local_p["e1z"],
                local_p["tube_a"],
                local_p["tube_b"],
                local_p["tube_c"],
                local_p["tube_d"],
                local_p["L"],
                *local_p["global_params"],
                self.alpha0,
                self.alpha1,
                x_i,
                u_i,
            )

            # print(LgLfh)

            # print(lfh, lgh, u_i)
            # hdot = lfh + lgh @ u_i
            cbf = np.clip(float(cbf), -5, 5)
            lfh = np.clip(float(lfh), -50, 50)
            hddot = np.clip(float(hddot), -50, 50)

            obs.extend([cbf, lfh, hddot])

        obs.append(s)
        # obs.append(self.alpha0)
        # obs.append(self.alpha1)

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.array(obs, dtype=np.float32)

    def _get_reward(self, obs, s_dot, terminated, solver_status):

        # 1. Progress reward
        progress_reward = 0.2 * np.clip(s_dot, 0, 3.0)  # Range: [0, 6]

        # 2. CBF constraint violations from observation
        cbf_violation_penalty = 0.0

        M = self.M
        alpha = self.alpha0
        for i in range(M):
            cbf = obs[2 * i]
            hdot = obs[2 * i + 1]

            # Check constraint: Lfh + alpha*cbf >= 0
            constraint_value = self.alpha0 * cbf + hdot

            if constraint_value < 0:  # Violation
                # Penalize proportional to violation magnitude
                cbf_violation_penalty += -1.0 * abs(constraint_value)

        # 3. Alpha regularization (prefer small alpha for efficiency)
        # alpha typically in [0.1, 10]
        alpha_reg = -0.02 * alpha  # Range: [-0.005, -0.5]
        # alpha_reg = 0

        # 4. Feasibility penalties (keep alpha in bounds)
        feasibility_penalty = 0.0
        if alpha < min_alpha:
            feasibility_penalty = -1.0 * (min_alpha - alpha) ** 2
        elif alpha > max_alpha:
            feasibility_penalty = -1.0 * (alpha - max_alpha) ** 2

        # 5. Terminal bonus (reached goal)
        terminal_bonus = 5.0 if terminated else 0.0

        solver_status_reward = 0
        if solver_status != 0:
            self.n_consecutive_infeasibilities += 1
            solver_status_reward -= 1.0
        else:
            self.n_consecutive_infeasibilities = 0

        if self.n_consecutive_infeasibilities >= 3:
            solver_status_reward -= 20

        if np.random.random() < 0.01:  # Log 1% of the time
            print(
                f"Reward breakdown: progress={progress_reward:.2f}, "
                f"cbf_viol={cbf_violation_penalty:.2f}, "
                f"alpha_reg={alpha_reg:.2f}, "
                f"feasibility={feasibility_penalty:.2f}"
                f"solver_reward={solver_status_reward:.2f}"
            )

        # Total reward
        reward = (
            progress_reward
            + cbf_violation_penalty
            + alpha_reg
            + feasibility_penalty
            + terminal_bonus
            + solver_status_reward
        )

        return reward

    def render(self):
        if self.render_mode == "human":
            if self.fig is None:
                plt.ion()
                self.fig = plt.figure(figsize=(8, 6))
                self.ax = self.fig.add_subplot(111, projection="3d")
                [_, x, y, z, _, _, _] = getTrack(self.track)
                self.ax.plot(
                    x,
                    y,
                    z,
                    "k--",
                    alpha=0.25,
                )

                # draw_gates(self.gate_data, self.ax)

                local_window = get_local_window_params(self.track_data, self.prev_s, n_knots, window_dist=self.track_horizon_window)
                # draw_corridor(self.ax, local_window, self.tube_coeffs, alpha=0.1)
                # draw_corridor(
                #     self.ax,
                #     self.track_data,
                #     np.linspace(0, self.track_data["L"], len(self.track_data["x"])),
                #     radius=1.0,
                #     color="c",
                #     alpha=0.2,
                # )
                self.tube_plot = self.ax.scatter([], [], [], s=3, alpha=0.3)

                if len(self.pcl) > 0:
                    self.ax.scatter(self.pcl[:, 0], self.pcl[:, 1], self.pcl[:, 2], c=self.pcl[:,2], cmap='plasma', s=2)
                    scale = np.concatenate([self.pcl.flatten(), x, y, z])
                else:
                    scale = np.concatenate([x, y, z])

                self.ax.auto_scale_xyz(scale, scale, scale)

                self.t_quiv = self.ax.quiver(
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    color="r",
                    length=0.5,
                    normalize=True,
                )

                self.e1_quiv = self.ax.quiver(
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    color="g",
                    length=0.5,
                    normalize=True,
                )

                self.e2_quiv = self.ax.quiver(
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    color="b",
                    length=0.5,
                    normalize=True,
                )

                (self.drone_marker,) = self.ax.plot([], [], [], "ro", markersize=10)
                (self.trail,) = self.ax.plot([], [], [], "g-", alpha=0.3)
                (self.horizon_line,) = self.ax.plot([], [], [], "b-", alpha=0.5)

                self.history = []

            local_window= get_local_window_params(self.track_data, self.prev_s, 100, window_dist=self.track_horizon_window)
            corridor_points = get_corridor_pts(self.ax, local_window, self.tube_coeffs)
            self.tube_plot._offsets3d = (corridor_points[:,0], corridor_points[:,1], corridor_points[:,2])

            p, t, e1, e2 = draw_horizon(self.ax, self.track_data, self.state)
            self.t_quiv.remove()
            self.t_quiv = self.ax.quiver(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                t[:, 0],
                t[:, 1],
                t[:, 2],
                color="r",
                length=0.5,
                normalize=True,
            )

            self.e1_quiv.remove()
            self.e1_quiv = self.ax.quiver(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                e1[:, 0],
                e1[:, 1],
                e1[:, 2],
                color="g",
                length=0.5,
                normalize=True,
            )

            self.e2_quiv.remove()
            self.e2_quiv = self.ax.quiver(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                e2[:, 0],
                e2[:, 1],
                e2[:, 2],
                color="b",
                length=0.5,
                normalize=True,
            )

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


def main():
    track = "straight_line"
    # track = "7gates"
    # track = "figure8"
    # track = "knotted_helix"
    # track = "12gates"
    # track = "race_uzh_19g"

    env = VolaDroneEnv(
        track, render_mode="human", normalize_obs=False
    )

    env.reset()
    for i in range(0, 2000):
        start = time.time()
        _, _, done, _, _ = env.step()
        print("step took ", time.time() - start)
        env.render()

        # input()
        if done:
            input()
            break


if __name__ == "__main__":
    main()
