import os
import time
import numpy as np
import matplotlib

matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.spatial import KDTree
from common import *

# ==========================================
# GLOBAL STYLE CONFIGURATION (CDC/IEEE LOOK)
# ==========================================
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.weight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 10,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}",
    }
)


class Renderer:
    # -------------------------------------------------------------------------
    # TUNABLE RENDER CONFIG
    # -------------------------------------------------------------------------
    RENDER_CONFIG = {
        "trail_width": 1.5,
        "collision_width": 6.0,
        "track_width": 4.0,
        "horizon_width": 2.5,
        "pcl_size": [4.0, 1.0, 1.0],
        "pcl_alpha": 0.25,
        "pcl_radius": 25.0,
        "tube_n_sweep": 10,
        # Camera Settings (Iso View only)
        "iso_elev": 67,
        "iso_azim": -135,
        "pad_x": 6.0,
        "pad_y": 6.0,
        "pad_z_low": 2.0,
        "pad_z_high": 12.0,
    }

    COLORS = {
        "track": "#888888",
        "drone": "#e63946",
        "trail": "#2a9d8f",
        "horizon": "#1d6fa4",
        "tube": "#e76f51",
        "pcl": "viridis",
        "alpha0": "#2a9d8f",
        "alpha1": "#1d6fa4",
        "cbf": "#e63946",
        "cbf_hline": "#e76f51",
        "fig_bg": "#f5f5f5",
        "ax_bg": "#ffffff",
        "pane": "#ebebeb",
    }

    def __init__(
        self,
        env,
        save_video=False,
        video_name="vola_drone_sim.mp4",
        pcl_sample_rate=0.2,
    ):
        self.env = env
        self.save_video = save_video
        R = self.RENDER_CONFIG
        C = self.COLORS

        self.env.traj_xyzs = np.column_stack(
            [
                self.env.traj_dense["x"],
                self.env.traj_dense["y"],
                self.env.traj_dense["z"],
            ]
        )

        if len(self.env.pcl) > 0:
            indices = np.random.choice(
                len(self.env.pcl),
                int(pcl_sample_rate * len(self.env.pcl)),
                replace=False,
            )
            self.sampled_pcl = self.env.pcl[indices]
            self.pcl_kdtree = KDTree(self.env.pcl)
        else:
            self.sampled_pcl = np.empty((0, 3))
            self.pcl_kdtree = None

        self.collision_flags = []
        track = self.env.track_data
        self.bounds = {
            "x": (np.min(track["x"]) - R["pad_x"], np.max(track["x"]) + R["pad_x"]),
            "y": (np.min(track["y"]) - R["pad_y"], np.max(track["y"]) + R["pad_y"]),
            "z": (
                np.min(track["z"]) - R["pad_z_low"],
                np.max(track["z"]) + R["pad_z_high"],
            ),
        }

        self.fig_geo = plt.figure("Geometry", figsize=(14, 7), facecolor=C["fig_bg"])

        gs_geo = self.fig_geo.add_gridspec(
            2,
            2,
            width_ratios=[2.2, 1],  # iso clearly larger, but not extreme
            height_ratios=[1, 1],  # ensures top + side split evenly
            wspace=0.02,
            hspace=0.02,
        )

        self.fig_geo.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

        self.ax_iso = self.fig_geo.add_subplot(
            gs_geo[:, 0], projection="3d", facecolor=C["fig_bg"]
        )
        self.ax_top = self.fig_geo.add_subplot(
            gs_geo[0, 1], projection="3d", facecolor=C["fig_bg"]
        )
        self.ax_side = self.fig_geo.add_subplot(
            gs_geo[1, 1], projection="3d", facecolor=C["fig_bg"]
        )
        # self.fig_geo = plt.figure("Geometry", figsize=(14, 7), facecolor=C["fig_bg"])
        # gs_geo = self.fig_geo.add_gridspec(2, 2, width_ratios=[2, 1.2], hspace=0.1, wspace=0.1)
        # self.fig_geo.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        #
        # self.ax_iso = self.fig_geo.add_subplot(gs_geo[:, 0], projection="3d", facecolor=C["fig_bg"])
        # self.ax_top = self.fig_geo.add_subplot(gs_geo[0, 1], projection="3d", facecolor=C["fig_bg"])
        # self.ax_side = self.fig_geo.add_subplot(gs_geo[1, 1], projection="3d", facecolor=C["fig_bg"])

        self.fig_telem = plt.figure("Telemetry", figsize=(10, 7), facecolor=C["fig_bg"])
        gs_telem = self.fig_telem.add_gridspec(2, 1, hspace=0.4)
        self.ax_alpha = self.fig_telem.add_subplot(gs_telem[0])
        self.ax_cbf = self.fig_telem.add_subplot(gs_telem[1])

        self.axes = [self.ax_iso, self.ax_top, self.ax_side]
        self.alpha_hist = {"alpha0": [], "alpha1": []}
        self.cbf_hist, self.time_steps, self.history = [], [], []

        if self.save_video:
            self.writer = FFMpegWriter(fps=20, bitrate=4000)
            self.writer.setup(self.fig_geo, video_name, dpi=120)

        self._init_artists()
        self._style_axes()

    def _init_artists(self):
        self.drone_markers, self.trails, self.collision_trails = [], [], []
        self.horizons, self.pcl_plots, self.tube_plots = [], [], []
        C, R = self.COLORS, self.RENDER_CONFIG

        for i, ax in enumerate(self.axes):
            ax.plot(
                self.env.track_data["x"],
                self.env.track_data["y"],
                self.env.track_data["z"],
                color=C["track"],
                linestyle="--",
                linewidth=R["track_width"],
                alpha=0.5,
            )

            (dm,) = ax.plot(
                [],
                [],
                [],
                "o",
                color=C["drone"],
                markersize=10 if i == 0 else 6,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=100,
            )
            (tr,) = ax.plot(
                [], [], [], color=C["trail"], linewidth=R["trail_width"], zorder=50
            )
            (ct,) = ax.plot(
                [], [], [], color="#ff0000", linewidth=R["collision_width"], zorder=55
            )
            (hz,) = ax.plot(
                [],
                [],
                [],
                color=C["horizon"],
                linewidth=R["horizon_width"],
                linestyle="--",
                alpha=0.6,
            )

            pc = ax.scatter(
                [],
                [],
                [],
                c=[],
                s=R["pcl_size"][i],
                alpha=R["pcl_alpha"],
                cmap=C["pcl"],
                zorder=5,
            )
            tp = ax.scatter(
                [], [], [], s=6 if i == 0 else 3, alpha=0.15, color=C["tube"], zorder=10
            )

            for item, lst in zip(
                [dm, tr, ct, hz, pc, tp],
                [
                    self.drone_markers,
                    self.trails,
                    self.collision_trails,
                    self.horizons,
                    self.pcl_plots,
                    self.tube_plots,
                ],
            ):
                lst.append(item)

    def _style_axes(self):
        for ax in self.axes:
            ax.xaxis.pane.set_facecolor(self.COLORS["pane"])
            ax.yaxis.pane.set_facecolor(self.COLORS["pane"])
            ax.zaxis.pane.set_facecolor(self.COLORS["pane"])
            ax.grid(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # often makes 3D side views look tiny
            ax.set_box_aspect(None)

    def _update_telemetry(self):
        C = self.COLORS
        self.ax_alpha.clear()
        self.ax_alpha.plot(
            self.time_steps,
            self.alpha_hist["alpha0"],
            color=C["alpha0"],
            label=r"$\bm{\alpha_0}$",
        )
        self.ax_alpha.plot(
            self.time_steps,
            self.alpha_hist["alpha1"],
            color=C["alpha1"],
            label=r"$\bm{\alpha_1}$",
        )
        self.ax_alpha.set_title(r"\textbf{Adaptation Parameters}")
        self.ax_alpha.legend(loc="upper right")
        self.ax_alpha.grid(True, alpha=0.2)

        self.ax_cbf.clear()
        self.ax_cbf.plot(self.time_steps, self.cbf_hist, color=C["cbf"], linewidth=2)
        self.ax_cbf.axhline(0, color=C["cbf_hline"], linestyle="--")
        self.ax_cbf.set_title(r"\textbf{Safety Margin } $\bm{h(\mathbf{x})}$")
        self.ax_cbf.set_ylabel(r"\textbf{Margin [m]}")
        self.ax_cbf.grid(True, alpha=0.2)

    def update(self):
        state = self.env.state
        curr_pos = state[:3]
        self.history.append(curr_pos.copy())
        R = self.RENDER_CONFIG

        # 1. Corridor & PCL Data Prep
        local_window = get_local_window_params(
            self.env.track_data,
            self.env.prev_s,
            100,
            window_dist=self.env.track_horizon_window,
        )
        corridor_points = get_corridor_pts(
            self.ax_iso, local_window, self.env.tube_coeffs, n_sweep=R["tube_n_sweep"]
        )
        horizon = np.array(
            [self.env.solver.get(i, "x")[:3] for i in range(self.env.N + 1)]
        )
        hist_arr = np.array(self.history)

        # 2. Update Artists axis-by-axis
        for i, ax in enumerate(self.axes):
            # Update position artists
            self.drone_markers[i].set_data([curr_pos[0]], [curr_pos[1]])
            self.drone_markers[i].set_3d_properties([curr_pos[2]])
            self.trails[i].set_data(hist_arr[:, 0], hist_arr[:, 1])
            self.trails[i].set_3d_properties(hist_arr[:, 2])
            self.horizons[i].set_data(horizon[:, 0], horizon[:, 1])
            self.horizons[i].set_3d_properties(horizon[:, 2])

            # Update corridor
            self.tube_plots[i]._offsets3d = (
                corridor_points[:, 0],
                corridor_points[:, 1],
                corridor_points[:, 2],
            )

            # Update PCL
            if len(self.sampled_pcl) > 0:
                mask = np.ones(len(self.sampled_pcl), dtype=bool)
                if i == 0:  # Iso local view
                    mask = (
                        np.sum((self.sampled_pcl - curr_pos) ** 2, axis=1)
                        < R["pcl_radius"] ** 2
                    )

                disp = self.sampled_pcl[mask]
                if len(disp) > 0:
                    self.pcl_plots[i]._offsets3d = (disp[:, 0], disp[:, 1], disp[:, 2])
                    self.pcl_plots[i].set_array(disp[:, 2])

            # SET VIEW ANGLE INDIVIDUALLY
            if i == 0:  # ISO
                ax.view_init(elev=R["iso_elev"], azim=R["iso_azim"])
            elif i == 1:  # TOP
                ax.view_init(elev=90, azim=-90)
            elif i == 2:  # SIDE
                ax.view_init(elev=0, azim=-90)

            ax.set_xlim(self.bounds["x"])
            ax.set_ylim(self.bounds["y"])
            ax.set_zlim(self.bounds["z"])

        # 3. Telemetry Update
        self.time_steps.append(len(self.time_steps))
        self.alpha_hist["alpha0"].append(self.env.alpha0)
        self.alpha_hist["alpha1"].append(self.env.alpha1)

        # CBF calculation
        drone_pt = self.env.state[:3]
        diff = self.env.traj_xyzs - [drone_pt]
        dists = np.linalg.norm(diff, axis=1)
        champ_ind = np.argmin(dists)

        local_window = get_local_window_params(
            self.env.track_data,
            self.env.state[10],
            n_knots,
            window_dist=self.env.track_horizon_window,
            loop=False,
        )
        param_dict = build_acados_params(
            local_window, self.env.params, self.env.tube_coeffs
        )
        x_i = self.env.solver.get(0, "x")
        u_i = self.env.solver.get(0, "u")

        hddot, lfh, cbf, ellipse_dist, a = self.env.cbf_func(
            param_dict["x"],
            param_dict["y"],
            param_dict["z"],
            param_dict["vx"],
            param_dict["vy"],
            param_dict["vz"],
            param_dict["e1x"],
            param_dict["e1y"],
            param_dict["e1z"],
            param_dict["tube_a"],
            param_dict["tube_b"],
            param_dict["tube_c"],
            param_dict["tube_d"],
            param_dict["s_start"],
            param_dict["L"],
            *param_dict["global_params"],
            self.env.alpha0,
            self.env.alpha1,
            x_i,
            u_i,
        )

        s_val = max(
            (self.env.state[10] - param_dict["s_start"]) / param_dict["L"], 1e-2
        )
        # print("s_val", s_val)
        a = polynomial_flat(s_val, self.env.tube_coeffs[0, :], tube_degree)
        b = polynomial_flat(s_val, self.env.tube_coeffs[1, :], tube_degree)
        c = polynomial_flat(s_val, self.env.tube_coeffs[2, :], tube_degree)
        d = polynomial_flat(s_val, self.env.tube_coeffs[3, :], tube_degree)

        E = np.diag([a, b])
        Pe = np.array([c, d]).flatten()

        e1x = self.env.traj_dense["e1x"][champ_ind]
        e1y = self.env.traj_dense["e1y"][champ_ind]
        e1z = self.env.traj_dense["e1z"][champ_ind]

        e2x = self.env.traj_dense["e2x"][champ_ind]
        e2y = self.env.traj_dense["e2y"][champ_ind]
        e2z = self.env.traj_dense["e2z"][champ_ind]

        e1 = np.array([e1x, e1y, e1z])
        e2 = np.array([e2x, e2y, e2z])

        ref_pos_x = self.env.traj_dense["x"][champ_ind]
        ref_pos_y = self.env.traj_dense["y"][champ_ind]
        ref_pos_z = self.env.traj_dense["z"][champ_ind]
        ref_p = np.array([ref_pos_x, ref_pos_y, ref_pos_z])

        manual_cbf = self.compute_cbf(drone_pt, ref_p, e1, e2, [a, b, c, d])

        # print("diff s", self.env.state[10] - param_dict["s_start"])
        # print("ellipse_dist:", ellipse_dist)
        # print("cbf:", cbf)
        # print("manual cbf:", manual_cbf)
        # print("dists[champ_ind]:", dists[champ_ind])
        # print("cbf a", polynomial_flat(s_val, self.env.tube_coeffs[0,:], tube_degree))
        # print("a", polynomial_flat(s_val, self.env.tube_coeffs[0,:], tube_degree))
        # print("b", polynomial_flat(s_val, self.env.tube_coeffs[1,:], tube_degree))
        # cbf_val = float(np.sign(cbf) * min(np.abs(cbf), np.abs(self.env.max_tube_radius - dists[champ_ind])))
        cbf_val = float(cbf)

        # self.cbf_hist.append(cbf_val)
        self.cbf_hist.append(manual_cbf)
        self._update_telemetry()

        self.fig_geo.canvas.draw()
        plt.pause(0.001)
        if self.save_video:
            self.writer.grab_frame()

    def compute_cbf(self, drone_pos, ref_pos, e1, e2, coeffs_at_s):
        """
        drone_pos: np.array([x, y, z])
        ref_pos:   np.array([xr, yr, zr]) from spline
        e1, e2:    Basis vectors for the plane
        coeffs_at_s: [a, b, c, d] (the polynomial outputs)
        """
        error_world = drone_pos - ref_pos
        w1 = np.dot(error_world, e1)
        w2 = np.dot(error_world, e2)

        # f(w) = A*w1^2 + B*w1*w2 + C*w2^2 + D*w1 + E*w2 + F
        A = coeffs_at_s[0]  # a_sweep
        B = 0.0  # P matrix is diagonal
        C = coeffs_at_s[1]  # b_sweep
        D = coeffs_at_s[2]  # c_sweep
        E = coeffs_at_s[3]  # d_sweep
        F = -1.0  # constant from your visualization
        # print("params:\n", np.array(coeffs_at_s))

        # Inside the ellipse, this value is < 0 (because F = -1)
        f_val = A * w1**2 + B * w1 * w2 + C * w2**2 + D * w1 + E * w2 + F

        # We want h = 0 at the boundary and h > 0 inside.
        # Since f_val is negative inside, h = -f_val works.
        # To make it depth-consistent (h=1 at center), divide by K.
        K = 1.0 + (D**2 / (4 * A + 1e-8)) + (E**2 / (4 * C + 1e-8))
        h = -f_val / K

        return h

    def close(self):
        if self.save_video:
            self.writer.finish()
        plt.close("all")
