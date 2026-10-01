from pathlib import Path

import imageio
import mujoco
import torch

from algorithms.ppo import PPO
from envs.g1_imitation_env import G1ImitationEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "g1_imitation_ppo.pt"
RESULTS_DIR = PROJECT_ROOT / "results"
VIDEO_PATH = RESULTS_DIR / "imitation_walk.mp4"


env = G1ImitationEnv()

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

agent.load(
    str(CHECKPOINT_PATH)
)

agent.network.eval()


# --------------------------------------------------
# Renderer
# --------------------------------------------------

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


# --------------------------------------------------
# Deterministic reference start
# --------------------------------------------------

obs, _ = env.reset(
    options={
        "reference_frame": 0
    }
)


frames = []

total_reward = 0.0
steps = 0


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

    total_reward += reward
    steps += 1

    # Camera follows robot

    camera.lookat[:] = [
        env.data.qpos[0],
        env.data.qpos[1],
        env.data.qpos[2] * 0.65,
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


fps = int(
    round(
        1.0
        / env.control_dt
    )
)


RESULTS_DIR.mkdir(parents=True, exist_ok=True)


imageio.mimsave(
    str(VIDEO_PATH),
    frames,
    fps=fps,
)


renderer.close()
env.close()


print()
print(
    f"Episode steps: {steps}"
)

print(
    f"Total reward: {total_reward:.2f}"
)

print(
    f"Video saved to: {VIDEO_PATH}"
)