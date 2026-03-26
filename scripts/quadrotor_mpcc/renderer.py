import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.spatial import KDTree
from common import *

class Renderer:
    # -------------------------------------------------------------------------
    # COLOR CONFIG
    # -------------------------------------------------------------------------
    COLORS = {
        'track':      '#888888',   # reference track dashes
        'drone':      '#e63946',   # drone marker
        'trail':      '#2a9d8f',   # flight trail
        'horizon':    '#1d6fa4',   # MPC horizon prediction
        'tube':       '#e76f51',   # corridor tube
        'pcl':        'plasma',    # point cloud colormap
        'alpha0':     '#2a9d8f',   # alpha_0 plot line
        'alpha1':     '#1d6fa4',   # alpha_1 plot line
        'cbf':        '#e63946',   # CBF plot line
        'cbf_hline':  '#e76f51',   # CBF zero-line
        'fig_bg':     '#f5f5f5',   # figure background
        'ax_bg':      '#ffffff',   # 2D axes background
        'pane':       '#ebebeb',   # 3D pane fill
        'pane_edge':  '#cccccc',   # 3D pane edge
    }

    PCL_SIZE  = [4.0, 0.5, 0.5]
    PCL_ALPHA = [0.3, 0.3, 0.3]
    COLLISION_RADIUS = 0.3

    def __init__(self, env, save_video=False, video_name="vola_drone_sim.mp4", pcl_sample_rate=0.2):
        self.env = env
        self.save_video = save_video

        # Ensure trajectory is (N,3) for distance calcs
        self.env.traj_xyzs = np.column_stack([
            self.env.traj_dense["x"],
            self.env.traj_dense["y"],
            self.env.traj_dense["z"],
        ])

        # Pre-downsample point cloud
        if len(self.env.pcl) > 0:
            indices = np.random.choice(len(self.env.pcl), int(pcl_sample_rate * len(self.env.pcl)), replace=False)
            self.sampled_pcl = self.env.pcl[indices]
            self.pcl_kdtree = KDTree(self.env.pcl)
        else:
            self.sampled_pcl = np.empty((0, 3))
            self.pcl_kdtree = None

        self.collision_flags = []

        # Calculate global bounds
        track = self.env.track_data
        self.bounds = {
            'x': (np.min(track["x"]) - 5, np.max(track["x"]) + 5),
            'y': (np.min(track["y"]) - 5, np.max(track["y"]) + 5),
            'z': (np.min(track["z"]) - 2, np.max(track["z"]) + 10)
        }

        C = self.COLORS
        plt.style.use('default')

        # ── Window 1: Geometry ────────────────────────────────────────────────
        self.fig_geo = plt.figure("Geometry", figsize=(16, 8), facecolor=C['fig_bg'])
        gs_geo = self.fig_geo.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[3, 3.2], hspace=0.05, wspace=0.05)

        self.ax_iso  = self.fig_geo.add_subplot(gs_geo[0:2, 0], projection='3d', facecolor=C['fig_bg'])
        self.ax_top  = self.fig_geo.add_subplot(gs_geo[0,   1], projection='3d', facecolor=C['fig_bg'])
        self.ax_side = self.fig_geo.add_subplot(gs_geo[1,   1], projection='3d', facecolor=C['fig_bg'])

        # ── Window 2: Telemetry ───────────────────────────────────────────────
        self.fig_telem = plt.figure("Telemetry", figsize=(11, 8), facecolor=C['fig_bg'])
        gs_telem = self.fig_telem.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)

        self.ax_alpha0 = self.fig_telem.add_subplot(gs_telem[0])
        self.ax_cbf    = self.fig_telem.add_subplot(gs_telem[1])

        self.collision_text = self.fig_telem.text(0.5, 0.97, '', ha='center', va='top', fontsize=13, fontweight='bold', color='#e63946', transform=self.fig_telem.transFigure)

        self.axes = [self.ax_iso, self.ax_top, self.ax_side]
        self.history = []
        self.alpha_hist = {"alpha0": [], "alpha1": []}
        self.cbf_hist = []
        self.time_steps = []

        if self.save_video:
            self.writer = FFMpegWriter(fps=20, bitrate=4000)
            self.writer.setup(self.fig_geo, video_name, dpi=120)

        self._init_artists()
        self._style_axes()

    def _init_artists(self):
        self.drone_markers = []
        self.trails = []
        self.collision_trails = []
        self.horizons = []
        self.pcl_plots = []
        self.tube_plots = []

        C = self.COLORS

        for i, ax in enumerate(self.axes):
            ax.plot(self.env.track_data["x"], self.env.track_data["y"], self.env.track_data["z"], color=C['track'], linestyle='--', linewidth=1.5, alpha=0.3)

            dm, = ax.plot([], [], [], 'o', color=C['drone'], markersize=12 if i == 0 else 8, markeredgecolor='white', markeredgewidth=1.5, zorder=100)
            tr, = ax.plot([], [], [], color=C['trail'], linewidth=2.5 if i == 0 else 2, alpha=0.8, zorder=50)
            ct, = ax.plot([], [], [], color='#ff0000', linewidth=3.5 if i == 0 else 2.5, alpha=0.95, zorder=55)
            hz, = ax.plot([], [], [], color=C['horizon'], linewidth=2, linestyle='--', alpha=0.6, zorder=40)
            pc = ax.scatter([], [], [], s=self.PCL_SIZE[i], alpha=self.PCL_ALPHA[i], cmap=C['pcl'], vmin=self.bounds['z'][0], vmax=self.bounds['z'][1])
            tp = ax.scatter([], [], [], s=6 if i == 0 else 3, alpha=0.15, color=C['tube'], zorder=10)

            for item, lst in zip([dm, tr, ct, hz, pc, tp], [self.drone_markers, self.trails, self.collision_trails, self.horizons, self.pcl_plots, self.tube_plots]):
                lst.append(item)

        title_style = {'fontsize': 12, 'fontweight': 'bold', 'color': '#222222', 'pad': 8}
        self.ax_iso.set_title("Isometric View", **title_style)
        self.ax_top.set_title("Top View",  **title_style)
        self.ax_side.set_title("Side View", **title_style)

    def _style_axes(self):
        for ax in self.axes:
            ax.grid(False)
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = True
            ax.xaxis.pane.set_facecolor(self.COLORS['pane'])
            ax.yaxis.pane.set_facecolor(self.COLORS['pane'])
            ax.zaxis.pane.set_facecolor(self.COLORS['pane'])
            ax.tick_params(colors='#333333', labelsize=8)
            if ax != self.ax_iso:
                ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    def _compute_cbf(self, state):
        drone_pt = state[:3]
        s_now = state[10]
        s_array = self.env.traj_dense["s"]
        idx = np.clip(np.searchsorted(s_array, s_now), 0, len(s_array) - 1)

        closest = self.env.traj_xyzs[idx]
        error = drone_pt - closest
        e1 = np.array([self.env.traj_dense["e1x"][idx], self.env.traj_dense["e1y"][idx], self.env.traj_dense["e1z"][idx]])
        e2 = np.array([self.env.traj_dense["e2x"][idx], self.env.traj_dense["e2y"][idx], self.env.traj_dense["e2z"][idx]])

        normal_dist = np.sqrt(np.dot(error, e1)**2 + np.dot(error, e2)**2)
        return float(self.env.max_tube_radius - normal_dist)

    def _plot_cbf(self):
        C = self.COLORS
        self.ax_cbf.clear()
        self.ax_cbf.set_facecolor(C['ax_bg'])
        self.ax_cbf.plot(self.time_steps, self.cbf_hist, color=C['cbf'], linewidth=2, label='Tube Margin')
        self.ax_cbf.axhline(y=0, color=C['cbf_hline'], linestyle='--', linewidth=1.5, alpha=0.7)
        self.ax_cbf.set_ylabel('Margin (m)', fontsize=10); self.ax_cbf.set_xlabel('Timestep', fontsize=10)
        self.ax_cbf.set_title('CBF Value (Safety Margin)', fontsize=12, fontweight='bold')
        self.ax_cbf.grid(True, alpha=0.3, linestyle='--')
        if len(self.cbf_hist) > 0 and min(self.cbf_hist) < 0.2:
            self.ax_cbf.fill_between(self.time_steps, 0, 0.2, alpha=0.1, color='red')

    def _update_telemetry(self, state, in_collision):
        C = self.COLORS
        self.collision_text.set_text('⚠ COLLISION DETECTED' if in_collision else '')

        # Alpha plot
        self.ax_alpha0.clear()
        self.ax_alpha0.set_facecolor(C['ax_bg'])
        self.ax_alpha0.plot(self.time_steps, self.alpha_hist["alpha0"], color=C['alpha0'], label=r"$\alpha_0$")
        self.ax_alpha0.plot(self.time_steps, self.alpha_hist["alpha1"], color=C['alpha1'], label=r"$\alpha_1$")
        self.ax_alpha0.legend(loc='upper right'); self.ax_alpha0.grid(True, alpha=0.3)
        self.ax_alpha0.set_title('Alpha Values', fontsize=12, fontweight='bold')

        # CBF plot
        self.cbf_hist.append(self._compute_cbf(state))
        self._plot_cbf()

    def _get_trail_segments(self, hist, flags):
        def build_segmented(pts, mask):
            out = []
            in_seg = False
            for pt, m in zip(pts, mask):
                if m:
                    if not in_seg and out: out.append([np.nan, np.nan, np.nan])
                    out.append(pt); in_seg = True
                else: in_seg = False
            return np.array(out) if out else np.empty((0, 3))
        flags_arr = np.array(flags, dtype=bool)
        return build_segmented(hist, ~flags_arr), build_segmented(hist, flags_arr)

    def update(self):
        state = self.env.state
        curr_pos = state[:3]
        self.history.append(curr_pos.copy())
        hist = np.array(self.history)

        if self.pcl_kdtree is not None:
            dist, _ = self.pcl_kdtree.query(curr_pos)
            in_collision = dist <= self.COLLISION_RADIUS
        else: in_collision = False
        self.collision_flags.append(in_collision)

        horizon = np.array([self.env.solver.get(i, "x")[:3] for i in range(self.env.N + 1)])
        local_window = get_local_window_params(self.env.track_data, self.env.prev_s, 100, window_dist=self.env.track_horizon_window)
        corridor_points = get_corridor_pts(self.ax_iso, local_window, self.env.tube_coeffs, n_sweep=20)

        trail_len = min(200, len(hist))
        safe_pts, coll_pts = self._get_trail_segments(hist[-trail_len:], self.collision_flags[-trail_len:])

        for i in range(3):
            self.drone_markers[i].set_data([curr_pos[0]], [curr_pos[1]]); self.drone_markers[i].set_3d_properties([curr_pos[2]])
            self.trails[i].set_data(safe_pts[:, 0], safe_pts[:, 1]); self.trails[i].set_3d_properties(safe_pts[:, 2])
            self.collision_trails[i].set_data(coll_pts[:, 0], coll_pts[:, 1]); self.collision_trails[i].set_3d_properties(coll_pts[:, 2])
            self.horizons[i].set_data(horizon[:, 0], horizon[:, 1]); self.horizons[i].set_3d_properties(horizon[:, 2])
            self.tube_plots[i]._offsets3d = (corridor_points[:, 0], corridor_points[:, 1], corridor_points[:, 2])
            if len(self.sampled_pcl) > 0:
                self.pcl_plots[i]._offsets3d = (self.sampled_pcl[:, 0], self.sampled_pcl[:, 1], self.sampled_pcl[:, 2])
                self.pcl_plots[i].set_array(self.sampled_pcl[:, 2])

        self.ax_iso.view_init(elev=25, azim=-60); self.ax_iso.set_xlim(self.bounds['x']); self.ax_iso.set_ylim(self.bounds['y']); self.ax_iso.set_zlim(self.bounds['z'])
        self.ax_top.view_init(elev=90, azim=-90); self.ax_top.set_xlim(self.bounds['x']); self.ax_top.set_ylim(self.bounds['y']); self.ax_top.set_zlim(self.bounds['z'])
        self.ax_side.view_init(elev=0, azim=-90); self.ax_side.set_xlim(self.bounds['x']); self.ax_side.set_ylim(self.bounds['y']); self.ax_side.set_zlim(self.bounds['z'])

        self.alpha_hist["alpha0"].append(self.env.alpha0); self.alpha_hist["alpha1"].append(self.env.alpha1)
        self.time_steps.append(len(self.time_steps))
        self._update_telemetry(state, in_collision)

        self.fig_geo.canvas.draw(); self.fig_telem.canvas.draw()
        plt.pause(0.0001)
        if self.save_video: self.writer.grab_frame()

    def close(self):
        if self.save_video: self.writer.finish()
        plt.close(self.fig_geo); plt.close(self.fig_telem)
