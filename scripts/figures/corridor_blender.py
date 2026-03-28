import bpy
import numpy as np
import sys
import os
import mathutils

# ==============================
# PATH SETUP
# ==============================
mpcc_path = os.path.abspath("../quadrotor_mpcc")
if mpcc_path not in sys.path:
    sys.path.append(mpcc_path)

from common import *
from tube_gen import *
from mpc_env import VolaDroneEnv

# ==============================
# CONFIG
# ==============================
TRACK = "short_line"
S_VALUE = 2.0
RENDER_PATH = "/Users/nickmohammad/Programs/quadrotor_mpcc/scripts/figures/tube_snapshot.png"

# ==============================
# COLOR HELPERS
# ==============================
def get_viridis_color(val):
    """Manual Viridis interpolation to avoid matplotlib dependency issues."""
    # Viridis color map sampled at 5 points
    v_map = [
        (0.267, 0.004, 0.329), # Purple
        (0.190, 0.407, 0.556), # Blue
        (0.127, 0.566, 0.550), # Teal
        (0.368, 0.788, 0.382), # Green
        (0.993, 0.906, 0.143)  # Yellow
    ]
    val = np.clip(val, 0, 1) * (len(v_map) - 1)
    idx = int(val)
    frac = val - idx
    if idx >= len(v_map) - 1: return (*v_map[-1], 1.0)
    
    c1 = v_map[idx]
    c2 = v_map[idx+1]
    return (
        c1[0] + (c2[0]-c1[0])*frac,
        c1[1] + (c2[1]-c1[1])*frac,
        c1[2] + (c2[2]-c1[2])*frac,
        1.0
    )

# ==============================
# MATERIALS
# ==============================
def make_mat(name, color, alpha=1.0, emit=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    
    rgb = color[:3]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Emission Strength"].default_value = emit
    
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
#        mat.shadow_method = 'NONE'
    return mat

def make_vertex_color_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    bsdf = nodes["Principled BSDF"]
    v_col = nodes.new(type="ShaderNodeVertexColor")
    v_col.layer_name = "Col"
    
    links.new(v_col.outputs["Color"], bsdf.inputs["Base Color"])
    return mat

# ==============================
# SCENE BUILDERS
# ==============================
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 900
    scene.render.filepath = RENDER_PATH

def setup_light():
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    bpy.context.object.data.energy = 3

def make_vertex_color_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes["Principled BSDF"]

    attr = nodes.new(type="ShaderNodeAttribute")
    attr.attribute_name = "Col"   # <-- only this matters

    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])

    return mat

# ==============================
# UPDATED PCL GENERATION (NO CHANGES NEEDED, BUT FOR CLARITY)
# ==============================
def create_instanced_pcl(name, points, radius=0.02, color=None, use_viridis=False):
    if len(points) == 0: return
    
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius)
    base = bpy.context.object
    base.name = f"{name}_instance"
    base.hide_viewport = True
    
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()

    if use_viridis:
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        
        # We store the color on the POINTS of the instancer
        color_attr = mesh.color_attributes.new(name="Col", type='BYTE_COLOR', domain='POINT')

        for i in range(len(points)):
            color_attr.data[i].color = get_viridis_color(z_norm[i])
            
        base.data.materials.append(make_vertex_color_mat(f"{name}_mat"))
    else:
        # For the tube, we just use a standard solid material
        base.data.materials.append(make_mat(f"{name}_mat", color, alpha=0.2, emit=0.5))

    obj.instance_type = 'VERTS'
    base.parent = obj
        
def setup_camera(target):
    bpy.ops.object.camera_add(location=(target[0]+5, target[1]-7, target[2]+4))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    direction = mathutils.Vector(target) - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

# ==============================
# MAIN
# ==============================
def main():
    clear_scene()
    setup_render()
    setup_light()

    # 1. Environment Simulation
    env = VolaDroneEnv(TRACK, normalize_obs=False, pcl_density=200)
    env.reset()
    while env.state[10] < S_VALUE:
        env.step()

    # 2. Generate Corridor Points (Using your specific method)
    # Note: ax is passed as None since it's likely a matplotlib handle
    corridor_pts = get_corridor_pts(
        ax=None, 
        track_data=env.track_data, 
        coeffs=env.tube_coeffs, 
        n_sweep=100, 
        n_angles=40
    )

    # 3. Create Scene Objects
    # Obstacles: Viridis based on Z
    if env.pcl is not None:
        create_instanced_pcl("Obstacles", env.pcl, radius=0.025, use_viridis=True)

    # Corridor: Transparent Blue with a slight glow
    create_instanced_pcl("Corridor", corridor_pts, radius=0.01, color=(0.0, 0.5, 1.0))

    # Robot: Red Sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=env.state[:3])
    bpy.context.object.data.materials.append(make_mat("RobotMat", (1, 0, 0)))

    # Track Line: Dark Grey
    track_pts = np.stack([env.track_data["x"], env.track_data["y"], env.track_data["z"]], axis=1)
    curve_data = bpy.data.curves.new("TrackCurve", type='CURVE')
    curve_data.dimensions = '3D'
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(track_pts)-1)
    for i, p in enumerate(track_pts):
        polyline.points[i].co = (*p, 1)
    curve_obj = bpy.data.objects.new("TrackLine", curve_data)
    curve_obj.data.bevel_depth = 0.01
    bpy.context.collection.objects.link(curve_obj)
    curve_obj.data.materials.append(make_mat("TrackMat", (0.1, 0.1, 0.1)))

    # 4. Finalize
    setup_camera(env.state[:3])
    bpy.ops.render.render(write_still=True)
    print(f"Render Complete: {RENDER_PATH}")

if __name__ == "__main__":
    main()
