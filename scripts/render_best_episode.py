from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch

from algorithms.ppo import PPO
from envs.g1_env import G1Env


NUM_EPISODES = 10
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "g1_ppo.pt"
RESULTS_DIR = PROJECT_ROOT / "results"
VIDEO_PATH = RESULTS_DIR / "best_episode.mp4"


# --------------------------------------------------
# Setup
# --------------------------------------------------

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


# --------------------------------------------------
# Find best deterministic episode
# --------------------------------------------------

best_reward = -np.inf
best_actions = None


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

            raw_mean = agent.network.actor(
                obs_tensor
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
        best_actions = episode_actions


print()
print(
    f"Best reward: {best_reward:.2f}"
)


# --------------------------------------------------
# Render best episode
# --------------------------------------------------

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

obs, _ = env.reset()

renderer = mujoco.Renderer(
    env.model,
    height=480,
    width=640,
)


# --------------------------------------------------
# Camera
# --------------------------------------------------

camera = mujoco.MjvCamera()

mujoco.mjv_defaultCamera(
    camera
)

# Side / slightly diagonal view
camera.azimuth = 135
camera.elevation = -12

# Distance from robot
camera.distance = 3.0


frames = []


for action in best_actions:

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    # --------------------------------------------------
    # Lock camera onto G1
    # --------------------------------------------------

    # Floating base position
    base_x = env.data.qpos[0]
    base_y = env.data.qpos[1]
    base_z = env.data.qpos[2]

    camera.lookat[:] = [
        base_x,
        base_y,
        base_z * 0.65,
    ]

    # Update rendered scene using tracking camera
    renderer.update_scene(
        env.data,
        camera=camera,
    )

    frame = renderer.render()

    frames.append(
        frame.copy()
    )

    if terminated or truncated:
        break


# --------------------------------------------------
# Save video
# --------------------------------------------------

fps = int(
    1.0
    / (
        env.model.opt.timestep
        * env.frame_skip
    )
)

imageio.mimsave(
    str(VIDEO_PATH),
    frames,
    fps=fps,
)

renderer.close()
env.close()


print()
print(
    f"Video saved to: {VIDEO_PATH}"
)