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

# Assuming these modules exist in your environment
try:
    from common import *
    from tube_gen import *
    from mpc_env import VolaDroneEnv
except ImportError:
    print("Warning: Custom modules not found. Ensure paths are correct.")

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
    return mat

def make_vertex_color_mat(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in nodes: nodes.remove(n)

    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    
    # CHANGE: Use 'GEOMETRY' instead of 'INSTANCER'
    attr = nodes.new(type="ShaderNodeAttribute")
    attr.attribute_name = "Col"
    attr.attribute_type = 'GEOMETRY' 

    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat

# ==============================
# SCENE BUILDERS
# ==============================
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Clear orphaned data
    for mesh in bpy.data.meshes: bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials: bpy.data.materials.remove(mat)

def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 900
    scene.render.filepath = RENDER_PATH

    # 1. FIX BACKGROUND: Pure White
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    bg_node = nodes.get("Background")
    bg_node.inputs[0].default_value = (1, 1, 1, 1) # White
    bg_node.inputs[1].default_value = 1.0         # Strength

    # 2. FIX COLOR SPACE: Ensure whites aren't grey
    scene.view_settings.view_transform = 'Standard'

def setup_light():
    # Primary Sun Light
    bpy.ops.object.light_add(type='SUN', location=(10, -10, 10))
    sun = bpy.context.object
    sun.data.energy = 4
    sun.data.angle = 0.05 # Crisp shadows

    # Soft Fill Light to brighten shadows
    bpy.ops.object.light_add(type='POINT', location=(-5, 5, 5))
    fill = bpy.context.object
    fill.data.energy = 200

# ==============================
# OBJECT GENERATION
# ==============================
def create_instanced_pcl(name, points, radius=0.02, color=None, use_viridis=False):
    if len(points) == 0: return
    
    # 1. Create the shape we want to see (the sphere)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius)
    instance_shape = bpy.context.object
    instance_shape.name = f"{name}_instance"
    instance_shape.hide_viewport = False
    
    # 2. Create the Emitter (the set of points)
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    emitter_obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(emitter_obj)
    
    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()
    if use_viridis:
        z = points[:, 2]
        # Ensure we don't divide by zero if all points are at the same height
        z_min, z_max = z.min(), z.max()
        z_range = z_max - z_min if z_max != z_min else 1.0
        z_norm = (z - z_min) / z_range
        
        # Create the attribute
        if not mesh.color_attributes:
            color_attr = mesh.color_attributes.new(name="Col", type='BYTE_COLOR', domain='POINT')
        else:
            color_attr = mesh.color_attributes["Col"]

        for i in range(len(points)):
            color_attr.data[i].color = get_viridis_color(z_norm[i])
        
        # CRITICAL: The material must be on the instance_shape (the sphere)
        # but the 'Col' attribute is on the emitter_obj. 
        # To make this work with 'VERTS' instancing, the sphere usually 
        # inherits the data from the vertex it sits on.
        instance_shape.data.materials.append(make_vertex_color_mat(f"{name}_mat"))

    else:
        # Standard solid/transparent material
        instance_shape.data.materials.append(make_mat(f"{name}_mat", color, alpha=0.3, emit=0.2))

    # 3. Enable Instancing
    instance_shape.parent = emitter_obj
    emitter_obj.instance_type = 'VERTS'
    
    # Hide the emitter "dots" so we only see the spheres
    emitter_obj.show_instancer_for_viewport = False
    emitter_obj.show_instancer_for_render = False

def setup_camera(target):
    bpy.ops.object.camera_add(location=(target[0]+6, target[1]-8, target[2]+5))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    
    # Point camera at target
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

    # --- 1. Simulation Setup (Assumes your Env is available) ---
    env = VolaDroneEnv(TRACK, normalize_obs=False, pcl_density=200)
    env.reset()
    while env.state[10] < S_VALUE:
        env.step()

    # --- 2. Corridor Points ---
    corridor_pts = get_corridor_pts(
        ax=None, 
        track_data=env.track_data, 
        coeffs=env.tube_coeffs, 
        n_sweep=100, 
        n_angles=40
    )

    # --- 3. Create Scene Objects ---
    
    # Obstacles: Viridis coloring based on height
    if env.pcl is not None:
        create_instanced_pcl("Obstacles", env.pcl, radius=0.03, use_viridis=True)

    # Corridor: Semi-transparent Blue
    create_instanced_pcl("Corridor", corridor_pts, radius=0.015, color=(0.1, 0.4, 1.0))

    # Robot: Large Red Sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.18, location=env.state[:3])
    bpy.context.object.data.materials.append(make_mat("RobotMat", (1, 0, 0), emit=0.1))

    # Track Line: Solid Black
    track_pts = np.stack([env.track_data["x"], env.track_data["y"], env.track_data["z"]], axis=1)
    curve_data = bpy.data.curves.new("TrackCurve", type='CURVE')
    curve_data.dimensions = '3D'
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(track_pts)-1)
    for i, p in enumerate(track_pts):
        polyline.points[i].co = (*p, 1)
    curve_obj = bpy.data.objects.new("TrackLine", curve_data)
    curve_obj.data.bevel_depth = 0.02
    bpy.context.collection.objects.link(curve_obj)
    curve_obj.data.materials.append(make_mat("TrackMat", (0, 0, 0)))

    # --- 4. Finalize ---
    setup_camera(env.state[:3])
    
    print("Starting Render...")
    bpy.ops.render.render(write_still=True)
    print(f"Render Complete: {RENDER_PATH}")

if __name__ == "__main__":
    main()
