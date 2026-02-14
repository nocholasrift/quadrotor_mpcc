import numpy as np
from time import time, sleep
from matplotlib import pyplot as plt

from common import *
from acados_settings import AcadosCustomOcp
from sys_dynamics import SysDyn

# --- Setup MPC and system ---
custom_ocp = AcadosCustomOcp()
custom_ocp.setup_acados_ocp()

sysModel = SysDyn()
zetaMx, _, _, _, _ = sysModel.SetupOde()

# Reference trajectory
_, xref_track, yref_track, zref_track = getTrack(track)

# --- Real-time plotting setup ---
plt.ion()
fig = plt.figure(figsize=(12, 6))

# 3D Path Subplot
ax3d = fig.add_subplot(121, projection='3d')
ax3d.plot(xref_track, yref_track, zref_track, '--', c='lightgray', alpha=0.5)
drone_scatter = ax3d.scatter([], [], [], c='red', s=50)

# Velocity Vector (Quiver)
vel_vector = ax3d.quiver(0, 0, 0, 0, 0, 0, color='blue', length=0.2, normalize=False)

# 2D Speed Subplot
ax_speed = fig.add_subplot(122)
speed_data = []
speed_line, = ax_speed.plot([], [], 'b-')
ax_speed.set_xlim(0, Nsim)
ax_speed.set_ylim(0, 3) # Adjust based on expected max speed (m/s)
ax_speed.set_ylabel('Speed (m/s)')
ax_speed.set_xlabel('Step')

zeta_N = custom_ocp.zeta_N  # initial state
# --- Simulation loop snippet ---
for step in range(Nsim):

    # Update reference and stop if done
    if custom_ocp.cost_update_ref(zeta_N[:, 0], U_HOV):
        print("Track complete!")
        break

    custom_ocp.solve_and_sim()
    zeta_N = custom_ocp.zeta_N 

    current_u = custom_ocp.solver.get(0, "u")
    print(f"Throttle: {np.round(current_u, 2)} | Max: {U_MAX}")

    # 1. Extract Position (Cartesian)
    s_i = np.array([zeta_N[0, 0]]).ravel()
    n_i = np.array([zeta_N[1, 0]]).ravel()
    b_i = np.array([zeta_N[2, 0]]).ravel()
    x_i, y_i, z_i = sysModel.Fren2CartT(zetaMx, s_i, n_i, b_i)

    # 2. Extract Velocity
    # In your common.py, zeta[14:17] are vx, vy, vz (Cartesian velocities)
    vx, vy, vz = zeta_N[13:16, 0]
    current_speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # 3. Update 3D Visuals
    drone_scatter._offsets3d = (np.ravel(x_i), np.ravel(y_i), np.ravel(z_i))
    
    # Remove old quiver and draw new one
    vel_vector.remove()
    vel_vector = ax3d.quiver(x_i, y_i, z_i, vx, vy, vz, color='blue', length=0.1)

    # 4. Update Speed Graph
    speed_data.append(current_speed)
    speed_line.set_data(range(len(speed_data)), speed_data)

    fig.canvas.draw()
    fig.canvas.flush_events()



# import numpy as np
# from time import time, sleep
# from matplotlib import pyplot as plt
#
# from common import *
# from acados_settings import AcadosCustomOcp
# from sys_dynamics import SysDyn
#
# # --- Setup MPC and system ---
# custom_ocp = AcadosCustomOcp()
# custom_ocp.setup_acados_ocp()
#
# sysModel = SysDyn()
# zetaMx, _, _, _, _ = sysModel.SetupOde()
#
# # Reference trajectory
# _, xref_track, yref_track, zref_track = getTrack()
#
# # --- Real-time plotting setup ---
# plt.ion()
# fig = plt.figure(figsize=(10, 8))
# ax3d = fig.add_subplot(111, projection='3d')
#
# # Plot reference trajectory
# ax3d.plot(xref_track, yref_track, zref_track, '--', c='lightgray', alpha=0.5, label='Reference')
#
# # Initialize drone scatter
# drone_scatter = ax3d.scatter([], [], [], c='red', s=50, label='Drone')
#
# # Axis limits and labels
# ax3d.set_xlim(min(xref_track), max(xref_track))
# ax3d.set_ylim(min(yref_track), max(yref_track))
# ax3d.set_zlim(min(zref_track), max(zref_track))
# ax3d.set_xlabel('X (m)')
# ax3d.set_ylabel('Y (m)')
# ax3d.set_zlabel('Z (m)')
# ax3d.legend()
# plt.show()
#
# # --- Simulation loop ---
# zeta_N = custom_ocp.zeta_N  # initial state
# T_del = 0.002  # simulation timestep
#
# for step in range(Nsim):
#     t1 = time()
#
#     # Update reference and stop if done
#     if custom_ocp.cost_update_ref(zeta_N[:, 0], U_HOV):
#         print("Track complete!")
#         break
#
#     # Solve OCP
#     custom_ocp.solve_and_sim()
#
#     # Convert Frenet to world coordinates
#     # Always ensure s_i, n_i, b_i are 1D arrays for Fren2CartT
#     s_i = np.array([zeta_N[0, 0]]).ravel()
#     n_i = np.array([zeta_N[1, 0]]).ravel()
#     b_i = np.array([zeta_N[2, 0]]).ravel()
#     x_i, y_i, z_i = sysModel.Fren2CartT(zetaMx, s_i, n_i, b_i)
#
#     # Update drone scatter in plot
#     drone_scatter._offsets3d = (np.ravel(x_i), np.ravel(y_i), np.ravel(z_i))
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#
#     # Update state for next iteration
#     zeta_N = custom_ocp.zeta_N
#
#     # Optional: slow down visualization to match real time
#     sleep(T_del)
#
