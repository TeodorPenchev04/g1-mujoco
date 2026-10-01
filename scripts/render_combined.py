from pathlib import Path
import argparse

import imageio
import mujoco
import torch

from algorithms.ppo import PPO
from envs.g1_combined_env import G1CombinedEnv


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


# ======================================================
# Arguments
# ======================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint",
    type=str,
    default="g1_speedmatched_final.pt",
    help=(
        "Checkpoint filename inside checkpoints/. "
        "Example: g1_speedmatched_150k.pt"
    ),
)

args = parser.parse_args()


CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / args.checkpoint
)

checkpoint_stem = (
    CHECKPOINT_PATH.stem
)

VIDEO_PATH = (
    RESULTS_DIR
    / f"{checkpoint_stem}_walk.mp4"
)


# ======================================================
# Environment
# ======================================================

env = G1CombinedEnv()


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


agent = PPO(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    device=device,
)


if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


agent.load(
    str(
        CHECKPOINT_PATH
    )
)

agent.network.eval()


print()
print(
    "Rendering checkpoint:"
)

print(
    CHECKPOINT_PATH
)

print()


# ======================================================
# Renderer
# ======================================================

renderer = mujoco.Renderer(
    env.model,
    height=480,
    width=640,
)


camera = mujoco.MjvCamera()

mujoco.mjv_defaultCamera(
    camera
)

camera.azimuth = 135
camera.elevation = -12
camera.distance = 3.0


# ======================================================
# Deterministic episode
# ======================================================

obs, _ = env.reset()


frames = []

total_reward = 0.0

steps = 0


vx_values = []
locomotion_values = []
imitation_values = []
pose_values = []
joint_velocity_values = []
orientation_values = []


while True:

    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():

        mean = (
            agent.network.actor(
                obs_tensor
            )
        )

        action = torch.tanh(
            mean
        )

    action = (
        action
        .squeeze(0)
        .cpu()
        .numpy()
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        action
    )

    total_reward += (
        reward
    )

    steps += 1

    vx_values.append(
        info[
            "forward_velocity"
        ]
    )

    locomotion_values.append(
        info[
            "normalized_locomotion_reward"
        ]
    )

    imitation_values.append(
        info[
            "imitation_reward"
        ]
    )

    pose_values.append(
        info[
            "pose_reward"
        ]
    )

    joint_velocity_values.append(
        info[
            "reference_joint_velocity_reward"
        ]
    )

    orientation_values.append(
        info[
            "reference_orientation_reward"
        ]
    )

    # ==================================================
    # Camera follows robot
    # ==================================================

    camera.lookat[:] = [
        env.data.qpos[0],
        env.data.qpos[1],
        env.data.qpos[2]
        * 0.65,
    ]

    renderer.update_scene(
        env.data,
        camera=camera,
    )

    frames.append(
        renderer.render().copy()
    )

    if (
        terminated
        or truncated
    ):
        break


# ======================================================
# Save video
# ======================================================

fps = int(
    round(
        1.0
        / env.control_dt
    )
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


imageio.mimsave(
    str(
        VIDEO_PATH
    ),
    frames,
    fps=fps,
)


renderer.close()
env.close()


# ======================================================
# Summary
# ======================================================

import numpy as np


print()
print(
    "Episode steps:",
    steps,
)

print(
    "Total reward:",
    f"{total_reward:.2f}",
)

print(
    "Mean vx:",
    f"{np.mean(vx_values):.3f}",
)

print(
    "Mean LocoN:",
    f"{np.mean(locomotion_values):.3f}",
)

print(
    "Mean imitation:",
    f"{np.mean(imitation_values):.3f}",
)

print(
    "Mean pose:",
    f"{np.mean(pose_values):.3f}",
)

print(
    "Mean joint velocity:",
    f"{np.mean(joint_velocity_values):.3f}",
)

print(
    "Mean orientation:",
    f"{np.mean(orientation_values):.3f}",
)

print()
print(
    "Video:",
    VIDEO_PATH,
)