import numpy as np
import torch

from envs.g1_env import G1Env
from algorithms.ppo import PPO


env = G1Env()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

agent = PPO(
    obs_dim,
    action_dim,
    device=device,
)

agent.load(
    "checkpoints/g1_ppo.pt"
)

agent.network.eval()


NUM_EPISODES = 10


for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False
    total_reward = 0.0

    velocities = []

    while not done:

        obs_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
            device=agent.device,
        ).unsqueeze(0)

        with torch.no_grad():

            mean_action = (
                agent.network.actor(
                    obs_tensor
                )
            )

        action = (
            mean_action
            .squeeze(0)
            .cpu()
            .numpy()
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
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

    mean_velocity = np.mean(
        velocities
    )

    print(
        f"Episode {episode + 1} | "
        f"Reward: {total_reward:.2f} | "
        f"Mean vx: {mean_velocity:.3f}"
    )


env.close()