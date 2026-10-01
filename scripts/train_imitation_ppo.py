import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch

from algorithms.ppo import PPO
from envs.g1_imitation_env import G1ImitationEnv


TOTAL_TIMESTEPS = 300_000
ROLLOUT_STEPS = 2048
SMOOTH_WINDOW = 20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_PATH = CHECKPOINTS_DIR / "g1_imitation_ppo.pt"
CSV_PATH = RESULTS_DIR / "imitation_training_log.csv"


CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


env = G1ImitationEnv()

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


print()
print("Device:", device)
print(
    "Observation dimension:",
    env.observation_space.shape[0],
)
print(
    "Action dimension:",
    env.action_space.shape[0],
)
print()


agent = PPO(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    device=device,
)


# --------------------------------------------------
# CSV
# --------------------------------------------------

with open(
    CSV_PATH,
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "episode",
        "global_step",

        "episode_reward",
        "smooth_reward",

        "episode_length",

        "mean_vx",
        "mean_reference_vx",

        "mean_pose_reward",
        "mean_joint_velocity_reward",

        "mean_orientation_reward",
        "mean_root_velocity_reward",
        "mean_root_angular_velocity_reward",

        "mean_height_reward",
    ])


def new_episode_stats():
    return {
        "reward": 0.0,

        "vx": [],
        "ref_vx": [],

        "pose": [],
        "joint_vel": [],

        "orientation": [],
        "root_vel": [],
        "root_ang_vel": [],

        "height": [],
    }


obs, _ = env.reset()

stats = new_episode_stats()

global_step = 0
episode_number = 0

recent_rewards = deque(
    maxlen=SMOOTH_WINDOW
)


# ======================================================
# Training
# ======================================================

while (
    global_step
    < TOTAL_TIMESTEPS
):

    observations = []
    raw_actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    for _ in range(
        ROLLOUT_STEPS
    ):

        (
            action,
            raw_action,
            log_prob,
            value,
        ) = agent.select_action(
            obs
        )

        (
            next_obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        done = (
            terminated
            or truncated
        )

        observations.append(
            obs.copy()
        )

        raw_actions.append(
            raw_action.copy()
        )

        rewards.append(
            reward
        )

        dones.append(
            float(done)
        )

        log_probs.append(
            log_prob
        )

        values.append(
            value
        )

        obs = next_obs

        global_step += 1

        # ------------------------------------------
        # Logging
        # ------------------------------------------

        stats["reward"] += reward

        stats["vx"].append(
            info[
                "forward_velocity"
            ]
        )

        stats["ref_vx"].append(
            info[
                "reference_velocity"
            ]
        )

        stats["pose"].append(
            info[
                "pose_reward"
            ]
        )

        stats["joint_vel"].append(
            info[
                "joint_velocity_reward"
            ]
        )

        stats["orientation"].append(
            info[
                "orientation_reward"
            ]
        )

        stats["root_vel"].append(
            info[
                "root_velocity_reward"
            ]
        )

        stats["root_ang_vel"].append(
            info[
                "root_angular_velocity_reward"
            ]
        )

        stats["height"].append(
            info[
                "height_reward"
            ]
        )

        # ------------------------------------------
        # Episode finished
        # ------------------------------------------

        if done:
            episode_number += 1

            episode_length = len(
                stats["vx"]
            )

            recent_rewards.append(
                stats["reward"]
            )

            smooth_reward = float(
                np.mean(
                    recent_rewards
                )
            )

            mean_vx = float(
                np.mean(
                    stats["vx"]
                )
            )

            mean_ref_vx = float(
                np.mean(
                    stats["ref_vx"]
                )
            )

            mean_pose = float(
                np.mean(
                    stats["pose"]
                )
            )

            mean_joint_vel = float(
                np.mean(
                    stats["joint_vel"]
                )
            )

            mean_orientation = float(
                np.mean(
                    stats["orientation"]
                )
            )

            mean_root_vel = float(
                np.mean(
                    stats["root_vel"]
                )
            )

            mean_root_ang_vel = float(
                np.mean(
                    stats[
                        "root_ang_vel"
                    ]
                )
            )

            mean_height = float(
                np.mean(
                    stats["height"]
                )
            )

            print(
                f"Ep {episode_number:4d} | "
                f"Step {global_step:7d} | "
                f"Reward {stats['reward']:7.2f} | "
                f"Avg {smooth_reward:7.2f} | "
                f"Len {episode_length:4d} | "
                f"vx {mean_vx:6.3f} | "
                f"ref {mean_ref_vx:6.3f} | "
                f"Pose {mean_pose:5.3f} | "
                f"RootVel {mean_root_vel:5.3f} | "
                f"Orient {mean_orientation:5.3f}"
            )

            with open(
                CSV_PATH,
                "a",
                newline="",
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    episode_number,
                    global_step,

                    stats["reward"],
                    smooth_reward,

                    episode_length,

                    mean_vx,
                    mean_ref_vx,

                    mean_pose,
                    mean_joint_vel,

                    mean_orientation,
                    mean_root_vel,
                    mean_root_ang_vel,

                    mean_height,
                ])

            obs, _ = env.reset()

            stats = (
                new_episode_stats()
            )

        if (
            global_step
            >= TOTAL_TIMESTEPS
        ):
            break

    # --------------------------------------------------
    # Bootstrap
    # --------------------------------------------------

    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    with torch.no_grad():

        next_value = (
            agent.network
            .critic(
                obs_tensor
            )
            .squeeze()
            .item()
        )

    # --------------------------------------------------
    # GAE + PPO update
    # --------------------------------------------------

    advantages, returns = (
        agent.compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
        )
    )

    agent.update(
        observations=np.asarray(
            observations
        ),

        raw_actions=np.asarray(
            raw_actions
        ),

        old_log_probs=np.asarray(
            log_probs
        ),

        advantages=advantages,

        returns=returns,
    )

    print(
        f"--- PPO update at step "
        f"{global_step} ---"
    )

    agent.save(
        str(CHECKPOINT_PATH)
    )


env.close()

print()
print("Training complete.")
print("Checkpoint:", CHECKPOINT_PATH)
print("Log:", CSV_PATH)