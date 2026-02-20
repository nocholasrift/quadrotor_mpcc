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
        ocp, "race_uzh_19g", solver=solver, integrator=integrator, cbf_func=cbf_func
    )
    env = Monitor(env)  # logs episode rewards
    return env


if __name__ == "__main__":

    env = DummyVecEnv([make_env])

    # Optional: sanity check (run once)
    check_env(make_env())

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=2000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model.learn(total_timesteps=200_000)

    model.save("sac_alpha_model")
