import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from mpc_env import VolaDroneEnv  # adjust import
from acados_settings import create_ocp
from acados_template import AcadosOcpSolver, AcadosSimSolver

ocp, cbf_func = create_ocp()
solver = AcadosOcpSolver(ocp)
integrator = AcadosSimSolver(ocp)


def make_env():
    env = VolaDroneEnv(
        ocp,
        "12gates",
        solver=solver,
        integrator=integrator,
        cbf_func=cbf_func,
        render_mode="human",
    )
    env = Monitor(env)  # logs episode rewards
    return env


# --------------------------------------------------
# 1. Create the environment EXACTLY like training
# --------------------------------------------------
env = make_env()  # DummyVecEnv([make_env])

# Optional: sanity check (run once)
# check_env(make_env())

# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------
model = SAC.load("race_uzh_19g")  # your saved model name

# --------------------------------------------------
# 3. Run one episode
# --------------------------------------------------
obs, info = env.reset()
done = False
episode_reward = 0.0

while not done:
    # deterministic=True removes exploration noise
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_reward += reward

    env.render()  # remove if no render method

print(f"Episode reward: {episode_reward}")
env.close()
