import numpy as np
import pickle
import os
from datetime import datetime

class VolaLogger:
    def __init__(self, track_name, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.filename = os.path.join(
            log_dir, f"sim_{track_name}_{datetime.now().strftime('%m%d_%H%M')}.pkl"
        )

        self.data = {
            "track_name": track_name,
            "steps": {
                "state": [],       # [px, py, pz, vx, vy, vz, ax, ay, az, s, s_dot]
                "tube_coeffs": [], # (4, deg+1)
                "alphas": [],      # [alpha0, alpha1]
                "cbf_vars": [],    # [h, h_dot, h_ddot] for the current step
                "solver_status": []
            }
        }

    def log_step(self, state, tube_coeffs, alphas, cbf_vals, status):
        """
        cbf_vals: [h, h_dot, h_ddot] calculated at the current state/control
        """
        d = self.data["steps"]
        d["state"].append(state.copy())
        d["tube_coeffs"].append(tube_coeffs.copy())
        d["alphas"].append(alphas.copy())
        d["cbf_vars"].append(cbf_vals) # List or array
        d["solver_status"].append(status)

    def save(self):
        # Cast to numpy for easy indexing during plotting
        for key in self.data["steps"]:
            self.data["steps"][key] = np.array(self.data["steps"][key])
            
        with open(self.filename, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"[Logger] Logged {len(self.data['steps']['state'])} steps to {self.filename}")
