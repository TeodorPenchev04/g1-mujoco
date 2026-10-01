import numpy as np
import torch
from pathlib import Path

from algorithms.ppo import PPO
from envs.g1_env import G1Env


NUM_EPISODES = 10
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "g1_ppo.pt"


env = G1Env()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


agent = PPO(
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
)

agent.load(
    str(CHECKPOINT_PATH)
)

agent.network.eval()


for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False

    total_reward = 0.0

    velocities = []
    heights = []
    upright_values = []

    while not done:

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():

            raw_mean = (
                agent.network.actor(
                    obs_tensor
                )
            )

            action = torch.tanh(
                raw_mean
            )

        action = (
            action.squeeze(0)
            .cpu()
            .numpy()
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        done = (
            terminated
            or truncated
        )

        total_reward += reward

        velocities.append(
            info["forward_velocity"]
        )

        heights.append(
            info["height"]
        )

        upright_values.append(
            info["upright"]
        )

    print(
        f"Episode {episode + 1:2d} | "
        f"Reward {total_reward:8.2f} | "
        f"Mean vx {np.mean(velocities):5.3f} | "
        f"Height {np.mean(heights):5.3f} | "
        f"Upright {np.mean(upright_values):5.3f} | "
        f"Steps {len(velocities):4d}"
    )


env.close()