import os

import imageio
import mujoco
import numpy as np
import torch

from algorithms.ppo import PPO
from envs.g1_env import G1Env


NUM_EPISODES = 10

VIDEO_PATH = (
    "results/best_episode.mp4"
)


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
    "checkpoints/g1_ppo.pt"
)

agent.network.eval()


best_reward = -np.inf
best_actions = None


# -----------------------------------
# Evaluate deterministic episodes
# -----------------------------------

for episode in range(NUM_EPISODES):

    obs, _ = env.reset()

    done = False
    total_reward = 0.0

    episode_actions = []

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

        episode_actions.append(
            action.copy()
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        done = (
            terminated
            or truncated
        )

    print(
        f"Episode {episode + 1}: "
        f"reward = {total_reward:.2f}"
    )

    if total_reward > best_reward:

        best_reward = total_reward

        best_actions = (
            episode_actions
        )


print()
print(
    f"Best reward: {best_reward:.2f}"
)


# -----------------------------------
# Render best episode
# -----------------------------------

os.makedirs(
    "results",
    exist_ok=True,
)

obs, _ = env.reset()

renderer = mujoco.Renderer(
    env.model,
    height=480,
    width=640,
)

frames = []


for action in best_actions:

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    renderer.update_scene(
        env.data
    )

    frame = renderer.render()

    frames.append(
        frame.copy()
    )

    if terminated or truncated:
        break


fps = int(
    1.0
    / (
        env.model.opt.timestep
        * env.frame_skip
    )
)


imageio.mimsave(
    VIDEO_PATH,
    frames,
    fps=fps,
)


renderer.close()
env.close()


print(
    f"Video saved to: {VIDEO_PATH}"
)