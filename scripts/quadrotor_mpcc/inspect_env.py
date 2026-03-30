import numpy as np
import yaml
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
# 2. ENVIRONMENT INSPECTOR
# ==========================================
def render_env_inspector(track_name, show_pcl=True, show_wire=True, density=15):
    track_path = f"../../resources/envs/{track_name}.yaml"
    track_data = setup_track(track_name)
    
    pcl = None
    if show_pcl:
        try: pcl = load_pcl_from_env(track_path, density)
        except Exception as e: print(f"Warning: {e}")

    with open(track_path, 'r') as f:
        raw_yaml = yaml.safe_load(f)
    obstacles = [BoxWireframe(obs['position'], obs['size']) for obs in raw_yaml.get('obstacles', [])]

    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "figure.facecolor": "white",
        "axes.facecolor": "white", "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}"
    })

    # FIX 1: Tighter GridSpec width ratios
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1])
    
    ax_iso  = fig.add_subplot(gs[:, 0], projection='3d')
    ax_top  = fig.add_subplot(gs[0, 1], projection='3d')
    ax_side = fig.add_subplot(gs[1, 1], projection='3d')

    view_configs = [
        {"ax": ax_iso,  "elev": 25, "azim": -135, "type": "iso",  "title": f"Isometric: {track_name}"},
        {"ax": ax_top,  "elev": 90, "azim": -90,  "type": "top",  "title": "Top-Down (X-Y)"},
        {"ax": ax_side, "elev": 0,  "azim": -90,  "type": "side", "title": "Side Profile (X-Z)"}
    ]

    for v in view_configs:
        ax = v["ax"]
        ax.plot(track_data["x"], track_data["y"], track_data["z"], color='#2a9d8f', linewidth=2.5)

        if show_wire:
            edges = []
            for box in obstacles: edges.extend(box.get_edges())
            ax.add_collection3d(Line3DCollection(edges, colors='#333333', linewidths=1.5, alpha=0.6))

        if show_pcl and pcl is not None:
            ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], c=pcl[:, 2], cmap='viridis', s=0.8, alpha=0.15)

        cx, cy = np.mean(track_data["x"]), np.mean(track_data["y"])
        buffer = 8.0

        # FIX 2: Set Box Aspect to 'auto' or explicit ratios to kill the "white box" effect
        if "figure8" in track_path:
            buffer *= 2.0
        if v["type"] == "iso":
            ax.set_xlim(cx - buffer, cx + buffer)
            ax.set_ylim(cy - buffer, cy + buffer)
            ax.set_zlim(-10, 15)
            ax.set_box_aspect([1, 1, 0.6]) # Squashes the tall vertical Z empty space
        
        elif v["type"] == "top":
            ax.set_xlim(cx - buffer, cx + buffer)
            ax.set_ylim(cy - buffer, cy + buffer)
            ax.set_zlim(-10, 15)
            ax.set_box_aspect([1, 1, 0.1]) # Makes it flat
            ax.set_zticks([])
            
        elif v["type"] == "side":
            ax.set_xlim(cx - buffer, cx + buffer)
            ax.set_ylim(cy - 0.5, cy + 0.5) 
            ax.set_zlim(-10, 15)
            ax.set_box_aspect([1, 0.1, 0.6]) # Squashes the Y depth to almost nothing

        ax.set_title(rf"\textbf{{{v['title']}}}")
        ax.view_init(elev=v["elev"], azim=v["azim"])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)

    # FIX 3: Negative wspace to pull the plots together
    plt.subplots_adjust(wspace=-0.2, hspace=0.1, left=0.01, right=0.99, top=0.95, bottom=0.05)
    plt.show()

if __name__ == "__main__":
    render_env_inspector("figure8", show_pcl=True, show_wire=True)
