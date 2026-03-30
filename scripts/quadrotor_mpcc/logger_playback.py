import yaml
import pickle
import argparse
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from common import *
from load_env import load_pcl_from_env

# ==========================================
# 1. CORE GEOMETRY
# ==========================================
class BoxWireframe:
    def __init__(self, pos, size, shift_to_base=False):
        self.pos = np.array(pos)
        self.size = np.array(size)
        self.shift_to_base = shift_to_base

    def get_edges(self):
        dx, dy, dz = self.size
        x_mid, y_mid, z_mid = self.pos
        x = [x_mid - dx/2, x_mid + dx/2]
        y = [y_mid - dy/2, y_mid + dy/2]
        z = [z_mid, z_mid + dz] if self.shift_to_base else [z_mid - dz/2, z_mid + dz/2]
        v = np.array([[x0, y0, z0] for x0 in x for y0 in y for z0 in z])
        return [
            [v[0], v[1]], [v[1], v[3]], [v[3], v[2]], [v[2], v[0]],
            [v[4], v[5]], [v[5], v[7]], [v[7], v[6]], [v[6], v[4]],
            [v[0], v[4]], [v[1], v[5]], [v[2], v[6]], [v[3], v[7]]
        ]

# ==========================================
# 2. PLAYBACK INSPECTOR
# ==========================================
class VolaPlaybackInspector:
    def __init__(self, log_path):
        try:
            with open(log_path, 'rb') as f:
                self.log = pickle.load(f)
        except Exception as e:
            print(f"Error loading log: {e}"); sys.exit(1)

        self.track_name = self.log["track_name"]
        self.track_data = setup_track(self.track_name)
        self.steps = self.log["steps"]
        
        # Environment Loading (Matches your inspector)
        track_path = f"../../resources/envs/{self.track_name}.yaml"
        self.pcl = load_pcl_from_env(track_path, samples_per_m2=15)

        with open(track_path, 'r') as f:
            raw_yaml = yaml.safe_load(f)
        self.obstacles = [BoxWireframe(obs['position'], obs['size']) for obs in raw_yaml.get('obstacles', [])]

        plt.rcParams.update({
            "text.usetex": True, "font.family": "serif", "figure.facecolor": "white",
            "axes.facecolor": "white", "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}"
        })

    def render_snapshot(self, step_idx, use_corridor=False):
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1])
        
        ax_iso  = fig.add_subplot(gs[:, 0], projection='3d')
        ax_top  = fig.add_subplot(gs[0, 1], projection='3d')
        ax_side = fig.add_subplot(gs[1, 1], projection='3d')

        view_configs = [
            {"ax": ax_iso,  "elev": 25, "azim": -135, "type": "iso",  "title": f"Isometric: {self.track_name}"},
            {"ax": ax_top,  "elev": 90, "azim": -90,  "type": "top",  "title": "Top-Down (X-Y)"},
            {"ax": ax_side, "elev": 0,  "azim": -90,  "type": "side", "title": "Side Profile (X-Z)"}
        ]

        # Data for the current step
        curr_state = self.steps["state"][step_idx]
        s_now = curr_state[10]
        path_history = self.steps["state"][:step_idx+1, :3]
        
        # Get corridor using your common utility
        # Note: adjust window_dist if the corridor isn't visible enough
        track_horizon_window = max_s_dot * Tf * 1.2
        local_window = get_local_window_params(self.track_data, s_now, n_knots, window_dist=track_horizon_window)

        for v in view_configs:
            ax = v["ax"]
            
            # 1. Reference Track
            ax.plot(self.track_data["x"], self.track_data["y"], self.track_data["z"], color='#2a9d8f', linewidth=2.5, alpha=0.3)
            
            # 2. Actual Robot Path
            ax.plot(path_history[:, 0], path_history[:, 1], path_history[:, 2], color='#e76f51', linewidth=5, label="Actual Path")

            # 3. Wireframe Obstacles (using the BoxWireframe logic)
            edges = []
            for box in self.obstacles: edges.extend(box.get_edges())
            ax.add_collection3d(Line3DCollection(edges, colors='#333333', linewidths=1.5, alpha=0.6))

            # 4. Point Cloud (PCL)
            if self.pcl is not None:
                ax.scatter(self.pcl[:, 0], self.pcl[:, 1], self.pcl[:, 2], c=self.pcl[:, 2], cmap='viridis', s=0.8, alpha=0.15)

            # 5. Tube / Safety Corridor
            if use_corridor:
                pts = get_corridor_pts(ax, local_window, self.steps["tube_coeffs"][step_idx], n_sweep=15)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s = 6, alpha=0.15, color="blue")

            # 6. Framing Logic (Matches your inspector exactly)
            cx, cy = np.mean(self.track_data["x"]), np.mean(self.track_data["y"])
            buffer = 8.0
            if "figure8" in self.track_name: buffer *= 2.0

            if v["type"] == "iso":
                ax.set_box_aspect([1, 1, 0.6])
            elif v["type"] == "top":
                ax.set_box_aspect([1, 1, 0.1]); ax.set_zticks([])
            elif v["type"] == "side":
                ax.set_box_aspect([1, 0.1, 0.6])
                # Center Y on the drone so we don't clip it, but it's "flat" anyway
                ax.set_ylim(curr_state[1] - 0.5, curr_state[1] + 0.5)
                
                # --- FIX: Turn off overlapping labels ---
                # ax.set_xticklabels([])
                ax.set_yticklabels([])
                # ax.set_xlabel("")
                ax.set_ylabel("")
                # Keep the Z-label since it's the most important for the side profile!
                # ax.set_zlabel(rf"\textbf{{Height (m)}}", fontsize=20)

            ax.set_xlim(cx - buffer, cx + buffer)
            ax.set_ylim(cy - buffer, cy + buffer)
            ax.set_zlim(-10, 15)
            ax.tick_params(width=1.0, length=6, labelsize=25)

            ax.set_title(rf"\textbf{{{v['title']}}}")
            ax.view_init(elev=v["elev"], azim=v["azim"])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)); ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)); ax.grid(False)

        plt.subplots_adjust(wspace=-0.2, hspace=0.1, left=0.01, right=0.99, top=0.95, bottom=0.05)
        plt.show()

    def plot_data_series(self, highlight_step):
        # --- PLOT CONFIGURATION ---
        P = {
            "lw_main": 4.5,           # Thickness of the data lines
            "lw_grid": 1.5,           # Thickness of grid lines
            "lw_spines": 4.5,         # Thickness of the axes box
            "lw_tick": 2.5,         
            "font_label": 22,         # X/Y Label font size
            "font_title": 24,         # Title font size
            "font_tick": 20,          # Tick number size
            "legend_size": 25,        # Legend font size
            "snapshot_alpha": 0.4,    # Transparency of the blue snapshot line
            "color_h": '#e76f51',     # Safety margin color
            "color_a0": '#264653',    # Alpha 0 color
            "color_a1": '#2a9d8f'     # Alpha 1 color
        }

        cbf_vars = np.array(self.steps["cbf_vars"]) 
        alphas = np.array(self.steps["alphas"])
        t = np.arange(len(cbf_vars))

        def apply_style(ax, title, xlabel, ylabel):
            ax.set_title(rf"\textbf{{{title}}}", fontsize=P["font_title"], pad=15)
            ax.set_xlabel(rf"\textbf{{{xlabel}}}", fontsize=P["font_label"])
            ax.set_ylabel(rf"\textbf{{{ylabel}}}", fontsize=P["font_label"])
            
            # Thick Spines (The box around the plot)
            # for spine in ax.spines.values():
            #     spine.set_linewidth(P["lw_spines"])
            
            # Thick Ticks
            ax.tick_params(width=P["lw_tick"], length=6, labelsize=P["font_tick"])
            ax.grid(True, alpha=0.2, linewidth=P["lw_grid"])

        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(t, cbf_vars[:, 0], color=P["color_h"], lw=P["lw_main"], label=r"$h(\bm{x})$")
        ax1.axhline(0, color='black', ls='--', lw=P["lw_spines"])
        
        ax1.set_ylim([-1.1, 1.1])
        apply_style(ax1, "HOCBF Safety Margin", "Time Step", "Safety Value")
        ax1.legend(loc='best', fontsize=P["legend_size"], frameon=True)
        fig1.tight_layout()

        if np.any(np.std(alphas, axis=0) > 1e-6):
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(t, alphas[:, 0], label=r"$\alpha_0$", color=P["color_a0"], lw=P["lw_main"])
            ax2.plot(t, alphas[:, 1], label=r"$\alpha_1$", color=P["color_a1"], lw=P["lw_main"])
            
            apply_style(ax2, "Adaptive Gains", "Time Step", "Gain Magnitude")
            ax2.legend(loc='upper right', fontsize=P["legend_size"], frameon=True)
            fig2.tight_layout()
        
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=str)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--corridor", action="store_true")
    args = parser.parse_args()

    inspector = VolaPlaybackInspector(args.log)
    if args.step != None and args.step < 0:
        target_step = np.argmax(inspector.steps["cbf_vars"][:, 0]) - 10
    else:
        target_step = args.step if args.step is not None else np.argmin(inspector.steps["cbf_vars"][:, 0])
    
    inspector.render_snapshot(target_step, use_corridor=args.corridor)
    inspector.plot_data_series(target_step)
