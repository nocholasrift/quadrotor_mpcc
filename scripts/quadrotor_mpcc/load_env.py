import yaml
import trimesh
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_pcl_from_env(yaml_path, samples_per_m2=20):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    pcl = []
    for obj in config["obstacles"]:
        if obj["type"] == "box":
            mesh = trimesh.creation.box(extents=obj["size"])

        translation = trimesh.transformations.translation_matrix(obj["position"])
        rotation = trimesh.transformations.euler_matrix(*obj["rotation"])
        mesh.apply_transform(translation @ rotation)

        points = mesh.sample(int(samples_per_m2 * mesh.area))
        pcl.append(points)

    pcl = np.vstack(pcl)
        
    return pcl


# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")
# pcl = load_pcl_from_env("../../resources/envs/two_cubes.yaml")
#
# ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], color='r', s=1)
# scale = pcl.flatten()
# ax.auto_scale_xyz(scale, scale, scale)
#
# plt.show()
