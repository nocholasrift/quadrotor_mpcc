import time
import click

import matplotlib

# matplotlib.use("Qt5Agg")

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
        max_step=1000,
        loop=False,
        pcl_density=20,
    ):
        super().__init__()

        try:
            self.pcl = load_pcl_from_env(f"../../resources/envs/{track}.yaml", pcl_density)
        except:
            self.pcl = []

        self.max_step = max_step
        self.loop = loop
        self.should_normalize_obs = normalize_obs

        self.alpha0_init = 1.0
        self.alpha1_init = 10.0

        self.alpha0 = self.alpha0_init
        self.alpha1 = self.alpha1_init

        # Load Track metadata
        self.track = track
        self.track_data = setup_track(track)
        x = self.track_data["x"]
        y = self.track_data["y"]
        z = self.track_data["z"]

        traj = np.stack([x, y, z], axis=1)

        self.track_kdtree = cKDTree(traj)
        self.max_tube_radius = 1.0
        self.tube_coeffs = np.zeros((4, tube_degree + 1))

        self.prev_s = 0
        self.step_count = 0

        self.render_mode = render_mode
        self.state = None

        # Plotting objects for rendering
        self.fig = None

        self.n_consecutive_infeasibilities = 0

        if self.should_normalize_obs:
            # stats = np.load(f"stats/{track}_normalization_stats.npz")
            stats = np.load(f"stats/all_normalization_stats.npz")

            self.obs_mean = stats["obs_mean"]
            self.obs_std = stats["obs_std"]

        ocp, self.cbf_func = create_ocp(tube_degree)
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
        self.M = 10
        self.nx = ocp.model.x.rows()  # [px, py, pz, vx, vy, vz, ax, ay, az, s, s_dot]
        self.nu = ocp.model.u.rows()  # [jx, jy, jz, s_ddot]

        # 20% buffer on as far as robot can travel in horizon time
        self.track_horizon_window = max_s_dot * Tf * 1.2
        # exit(0)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 * self.M + 1,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # self.reset()

    def _setup_gates(self, track):
        return load_gates(track)

    def normalize_obs(self, obs, mean, std):
        obs[:-1] = (obs[:-1] - mean[:-1]) / std[:-1]
        # obs[-1] = 2 * (obs[-2] / self.track_data["s"][-1]) - 1.0
        # obs[-2] = 2 * (obs[-2] - min_alpha) / (max_alpha - min_alpha) - 1
        # obs[-1] = 2 * (obs[-1] - min_alpha) / (max_alpha - min_alpha) - 1

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.nx)
        self.state[0] = self.track_data["x"][0]
        self.state[1] = self.track_data["y"][0]
        self.state[2] = self.track_data["z"][0]
        self.state[6] = 1.0

        self.n_consecutive_infeasibilities = 0
        self.prev_s = 0
        self.step_count = 0

        self.alpha0 = self.alpha0_init
        self.alpha1 = self.alpha1_init

        for stage in range(self.N + 1):
            self.solver.set(stage, "x", self.state)
            if stage < self.N:
                self.solver.set(stage, "u", np.zeros(self.nu))

        self.params = np.array(
            [1.0, 100.0, 1, 5.0, 1.0, 1.0],  # Q_c, Q_l, Q_t, Q_w, Q_sdd, Q_s
        )

        self.tube_coeffs = get_free_tube(tube_degree, self.max_tube_radius)

        local_window = get_local_window_params(
            self.track_data, 0, n_knots, window_dist=self.track_horizon_window
        )
        local_p = build_acados_params(local_window, self.params, self.tube_coeffs)
        return self._get_obs(local_p, 0), {}

    def _warm_start(self, local_p):
        prev_x = [self.solver.get(i, "x") for i in range(self.N + 1)]
        prev_u = [self.solver.get(i, "u") for i in range(self.N)]

        ds = prev_x[1][10]

        for i in range(self.N - 1):
            # Move stage i+1 to stage i
            new_x = prev_x[i + 1].copy()
            new_u = prev_u[i + 1]

            self.solver.set(i, "x", new_x)
            self.solver.set(i, "u", new_u)

        x_Nm1_shifted = prev_x[-1].copy()
        u_Nm1 = prev_u[-1]

        self.solver.set(self.N - 1, "x", x_Nm1_shifted)
        self.solver.set(self.N - 1, "u", u_Nm1)

        self.integrator.set("x", x_Nm1_shifted)
        self.integrator.set("u", u_Nm1)
        self.integrator.set("p", local_p)
        self.integrator.solve()

        x_N_warm = self.integrator.get("x")
        if x_N_warm[10] >= self.track_data["s"][-1] - 1e-1:
            x_N_warm = x_Nm1_shifted
            self.solver.set(self.N - 1, "u", np.zeros(len(u_Nm1)))

        self.solver.set(self.N, "x", x_N_warm)

    def step(self, action=None):
        s_global_now = self.state[10]
        self.prev_s = s_global_now - 0.1
        local_window = get_local_window_params(
            self.track_data,
            s_global_now,
            n_knots,
            window_dist=self.track_horizon_window,
            loop=self.loop,
        )

        # unnormed_action = action_unnormalize(action, min_alpha_dot, max_alpha_dot)
        # self.alpha0 += unnormed_action[0]
        if type(action) is np.ndarray:
            unnormed_action = action_unnormalize(action, min_alpha, max_alpha)
            self.alpha0 = unnormed_action[0]
            self.alpha1 = unnormed_action[1]
            # print(self.alpha0)

        start = time.time()

        occ_data = []
        if len(self.pcl) > 0:
            occ_data = project_cloud_to_parametric_path(
                self.pcl,
                self.track_data,
                self.track_kdtree,
                max_radius=self.max_tube_radius,
            )
        if len(occ_data) > 0:
            occ_data[:, 0] -= s_global_now
            mask = (occ_data[:, 0] >= 0) & (occ_data[:, 0] <= local_window["L"])
            occ_data = occ_data[mask]

            # if occ_data.shape[0] > 0:
            solver, coeffs = NLP(
                tube_degree,
                occ_data,
                local_window["L"],
                self.max_tube_radius,
                True,
            )
            solver.solve(solver=cp.CLARABEL, verbose=False, time_limit=1.0)
            a, b, c, d = coeffs
            self.tube_coeffs[0, :] = a.value
            self.tube_coeffs[1, :] = b.value
            self.tube_coeffs[2, :] = c.value
            self.tube_coeffs[3, :] = d.value

        # print(self.tube_coeffs)

        param_dict = build_acados_params(local_window, self.params, self.tube_coeffs)
        # print("param dict:\n", param_dict)
        alphas = np.array([self.alpha0, self.alpha1]).reshape((2,))

        local_p = dict_to_list(param_dict)
        local_p = np.concatenate([local_p, alphas])

        local_state = self.state.copy()
        # local_state[10] = 0.0
        local_state[10] = max(param_dict["s_start"][0] + 1e-3, local_state[10])

        if self.step_count != 0:
            self._warm_start(local_p)

        # Set parameters and pin initial state
        # print("s_start", param_dict["s_start"])
        # print("state s", local_state[10] / self.track_data["s"][-1])
        for stage in range(self.N + 1):
            self.solver.set(stage, "p", local_p)

        # Set inputs
        self.solver.set(0, "lbx", local_state)
        self.solver.set(0, "ubx", local_state)
        # print("window:", self.track_horizon_window)
        # print("true x0", local_state)

        # for stage in range(self.N + 1):
        #     # self.solver.set(stage, "p", self.params)
        #     self.solver.set(stage, "p", local_p)
        #     start_s = 0
        #     if stage < self.N:
        #         prev_x = self.solver.get(stage + 1, "x")
        #         if stage == 0:
        #             start_s = prev_x[10]
        #         # We must subtract the progress made in the last step
        #         # to keep the horizon consistent with s=0 at start
        #         prev_x[10] -= start_s
        #         # prev_x[10] -= (
        #         #     next_s_from_prev_step if "next_s_from_prev_step" in locals() else 0
        #         # )
        #         self.solver.set(stage, "x", prev_x)

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
        # global_s_next = global_s_next % self.track_data["s"][-1]
        self.state = next_local_state.copy()
        # self.state[10] = global_s_next
        # print("xN s:", self.solver.get(self.N, "x")[10])
        xN = self.solver.get(self.N, "x")
        # print("s_dot", xN[-1])
        # print("delta s", xN[10] - local_state[10])

        u = self.solver.get(0, "u")
        gym_obs = self._get_obs(param_dict, self.state[11] / max_s_dot)
        # gym_obs = []

        # print(f"{s_global_now} / {self.params[-1]}")

        # Termination: Finished 99% of the track
        self.step_count += 1
        terminated = (
            bool(
                # self.state[10]
                # >= self.track_data["L"] - self.track_horizon_window - 0.1
                self.state[10]
                >= self.track_data["L"] - 0.1
            )
            and not self.loop
        )

        # Truncation: Drone flew way off course (safety check)
        truncated = (
            bool(np.linalg.norm(self.state[:3]) > 100.0)
            or self.step_count >= self.max_step
        )

        reward = self._get_reward(gym_obs, self.state[-2], terminated, status)

        # print("un-norm", gym_obs)
        if self.should_normalize_obs:
            gym_obs = self.normalize_obs(gym_obs, self.obs_mean, self.obs_std)

        # print("norm", gym_obs)
        return gym_obs, reward, terminated, truncated, {}

    def _get_obs(self, local_p, s_dot):
        n = n_knots
        N = self.N
        M = self.M

        inds = np.linspace(1, N - 1, num=M, dtype=int)
        # inds = [i for i in range(N)]
        obs = []

        for i in inds:

            x_i = self.solver.get(int(i), "x")
            u_i = self.solver.get(int(i), "u")

            hddot, lfh, cbf, LgLfh = self.cbf_func(
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
                local_p["s_start"],
                local_p["L"],
                *local_p["global_params"],
                self.alpha0,
                self.alpha1,
                x_i,
                u_i,
            )

            # print(i, "hddot", hddot)
            # print(i, "lfh", lfh)
            # print(i, "cbf", cbf)
            # print(i, "grad_h", grad_h)

            # print(lfh, lgh, u_i)
            # hdot = lfh + lgh @ u_i
            cbf = np.clip(float(cbf), -5, 5)
            lfh = np.clip(float(lfh), -50, 50)
            hddot = np.clip(float(hddot), -50, 50)

            obs.extend([cbf, lfh, hddot])

        obs.append(s_dot)
        # print("len: ", s)
        # obs.append(self.alpha0)
        # obs.append(self.alpha1)

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.array(obs, dtype=np.float32)

    def _get_reward(self, obs, s, terminated, solver_status):

        # 1. Progress reward
        s_dot = obs[-1]
        progress_reward = 0.2 * np.clip(s_dot, 0, max_s_dot)

        # 2. CBF constraint violations from observation
        cbf_violation_penalty = 0.0

        M = self.M
        for i in range(M):
            cbf = obs[3 * i]
            lfh = obs[3 * i + 1]
            hddot = obs[3 * i + 2]

            # Check constraint: Lfh + alpha*cbf >= 0
            constraint_value = self.alpha0 * cbf + self.alpha1 * lfh + hddot
            constraint_value = np.clip(constraint_value, -10.0, 5.0)

            if constraint_value < 0:  # Violation
                # Penalize proportional to violation magnitude
                cbf_violation_penalty += -1.0 * abs(constraint_value)

        # 3. Alpha regularization (prefer small alpha for efficiency)
        # alpha typically in [0.1, 10]
        alphas = np.array([self.alpha0, self.alpha1])
        alpha_reg = -0.05 * np.sum(alphas)
        # alpha_reg = 0

        # 4. Feasibility penalties (keep alpha in bounds)
        feasibility_penalty = 0.0
        if np.any(alphas < min_alpha):
            feasibility_penalty = -1.0 * np.min(min_alpha - alphas) ** 2
        elif np.any(alphas > max_alpha):
            feasibility_penalty = -1.0 * np.min(alphas - max_alpha) ** 2

        # 5. Terminal bonus (reached goal)
        # terminal_bonus = 5.0 if terminated else 0.0

        solver_status_reward = 0
        traj_len = self.track_data["s"][-1]
        if s < traj_len - self.track_horizon_window:

            if solver_status != 0:
                self.n_consecutive_infeasibilities += 1
                solver_status_reward -= 0.5
            else:
                self.n_consecutive_infeasibilities = 0

        # if self.n_consecutive_infeasibilities >= 3:
        #     solver_status_reward -= 1.0

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
            # + terminal_bonus
            + solver_status_reward
        )

        return reward

    def render(self):
        if self.render_mode != "human":
            return

        # --- Initialization ---
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(12, 6))

            self.ax = self.fig.add_subplot(121, projection="3d")
            self.ax_alpha = self.fig.add_subplot(122)

            [_, x_t, y_t, z_t, _, _, _] = getTrack(self.track)
            self.ax.plot(x_t, y_t, z_t, "k--", alpha=0.1)

            # --- ADDED: Obstacle PointCloud ---
            if len(self.pcl) > 0:
                sample_size = int(0.2 * len(self.pcl))
                inds = np.linspace(0, len(self.pcl), num=len(self.pcl), dtype=int)
                sampled_pcl = np.random.choice(inds, size=sample_size, replace=False)
                print(sampled_pcl)
                self.ax.scatter(
                    self.pcl[sampled_pcl, 0],
                    self.pcl[sampled_pcl, 1],
                    self.pcl[sampled_pcl, 2],
                    c=self.pcl[sampled_pcl, 2],
                    cmap="plasma",
                    s=2,
                    alpha=0.5,
                )

            self.tube_plot = self.ax.scatter([], [], [], s=3, alpha=0.10, color="blue")
            (self.drone_marker,) = self.ax.plot(
                [], [], [], "ro", markersize=8, zorder=10
            )
            (self.trail,) = self.ax.plot([], [], [], "g-", alpha=0.4)
            (self.horizon_line,) = self.ax.plot([], [], [], "b-", lw=1.5)

            self.quivers = {"t": None, "e1": None, "e2": None}
            self.alpha_history = {"0": [], "1": []}
            self.ts = []

            (self.line_a0,) = self.ax_alpha.plot([], [], label=r"$\alpha_0$")
            (self.line_a1,) = self.ax_alpha.plot([], [], label=r"$\alpha_1$")
            self.ax_alpha.set_title("Safety Parameters (Alpha)")
            self.ax_alpha.legend()
            self.ax_alpha.grid(True, alpha=0.2)

            self.history = []

        # --- Updates ---
        curr_pos = self.state[:3]
        self.history.append(curr_pos.copy())
        self.alpha_history["0"].append(self.alpha0)
        self.alpha_history["1"].append(self.alpha1)
        self.ts.append(len(self.history))

        hist = np.array(self.history)

        # Update Drone & Trail
        self.drone_marker.set_data([curr_pos[0]], [curr_pos[1]])
        self.drone_marker.set_3d_properties([curr_pos[2]])
        self.trail.set_data(hist[:, 0], hist[:, 1])
        self.trail.set_3d_properties(hist[:, 2])

        # --- RE-ADDED: Corridor/Tube Logic ---
        local_window = get_local_window_params(
            self.track_data, self.prev_s, 100, window_dist=self.track_horizon_window
        )
        corridor_points = get_corridor_pts(self.ax, local_window, self.tube_coeffs, n_sweep=20)
        self.tube_plot._offsets3d = (
            corridor_points[:, 0],
            corridor_points[:, 1],
            corridor_points[:, 2],
        )

        # Update MPC Horizon
        horizon_states = []
        for i in range(self.N + 1):
            try:
                horizon_states.append(self.solver.get(i, "x")[:3])
            except:
                break

        if horizon_states:
            h_states = np.array(horizon_states)
            self.horizon_line.set_data(h_states[:, 0], h_states[:, 1])
            self.horizon_line.set_3d_properties(h_states[:, 2])

        # Update Alpha Plot
        self.line_a0.set_data(self.ts, self.alpha_history["0"])
        self.line_a1.set_data(self.ts, self.alpha_history["1"])
        self.ax_alpha.relim()
        self.ax_alpha.autoscale_view()

        # Quiver Refresh
        # p, t, e1, e2 = draw_horizon(self.ax, self.track_data, self.state)
        # for key, data, color in zip(["t", "e1", "e2"], [t, e1, e2], ["r", "g", "b"]):
        #     if self.quivers[key] is not None:
        #         self.quivers[key].remove()
        #     self.quivers[key] = self.ax.quiver(
        #         p[:, 0],
        #         p[:, 1],
        #         p[:, 2],
        #         data[:, 0],
        #         data[:, 1],
        #         data[:, 2],
        #         color=color,
        #         length=0.5,
        #         normalize=True,
        #     )

        # --- Sensible Axis Limits ---
        # Look ahead 10 units, look behind 5 units
        self.ax.set_xlim(curr_pos[0] - 5, curr_pos[0] + 10)
        self.ax.set_ylim(curr_pos[1] - 5, curr_pos[1] + 10)
        self.ax.set_zlim(curr_pos[2] - 5, curr_pos[2] + 10)

        plt.draw()
        plt.pause(1e-4)


def main():
    loop = False
    # track = "short_line"
    # track = "straight_line"
    # track = "7gates"
    # track = "figure8"
    # track = "knotted_helix"
    # track = "12gates"
    # track = "race_uzh_19g"

    # looped tracks
    # track = "3d_loop"
    track = "3d_square"
    # loop = True

    env = VolaDroneEnv(track, render_mode="human", normalize_obs=False, loop=loop)

    env.reset()
    for i in range(0, 2000):
        start = time.time()
        _, _, done, _, _ = env.step()
        # print("step took ", time.time() - start)
        env.render()

        # input()
        if done:
            input()
            break


if __name__ == "__main__":
    main()
