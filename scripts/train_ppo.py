import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch

from algorithms.ppo import PPO
from envs.g1_env import G1Env


TOTAL_TIMESTEPS = 1_000_000
ROLLOUT_STEPS = 2048
SMOOTH_WINDOW = 20

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

CHECKPOINTS_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

CSV_PATH = (
    RESULTS_DIR
    / "training_log.csv"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Reward terms logged from G1Env
# --------------------------------------------------

REWARD_TERM_KEYS = [
    "velocity_tracking",
    "yaw_tracking",
    "air_time",
    "clearance",
    "upright",
    "height",
    "orientation",
    "angular_velocity",
    "foot_slide",
    "action_rate",
    "posture",
    "ankle_limits",
    "torque_l2",
    "joint_acc_l2",
    "energy",
    "termination",
]


# --------------------------------------------------
# Environment
# --------------------------------------------------

env = G1Env()

obs_dim = (
    env.observation_space.shape[0]
)

action_dim = (
    env.action_space.shape[0]
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(
    "Device:",
    device,
)

print(
    "Observation dimension:",
    obs_dim,
)

print(
    "Action dimension:",
    action_dim,
)

print(
    "Target velocity:",
    env.target_velocity,
)

print(
    "Control dt:",
    env.control_dt,
)

print()


# --------------------------------------------------
# PPO
# --------------------------------------------------

agent = PPO(
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
)


# --------------------------------------------------
# CSV
# --------------------------------------------------

csv_header = [
    "episode",
    "global_step",
    "episode_reward",
    "smooth_reward",
    "mean_vx",
    "mean_vx_yaw",
    "episode_length",
    "mean_height",
    "mean_upright",
    "left_contact_ratio",
    "right_contact_ratio",
    "single_support_ratio",
] + [
    f"mean_{key}"
    for key in REWARD_TERM_KEYS
]


with open(
    CSV_PATH,
    "w",
    newline="",
) as csv_file:

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        csv_header
    )


# --------------------------------------------------
# Initial state
# --------------------------------------------------

obs, _ = env.reset()

global_step = 0
episode_number = 0

episode_reward = 0.0

episode_vx = []
episode_vx_yaw = []

episode_height = []
episode_upright = []

left_contacts = []
right_contacts = []

single_support = []

reward_term_history = {
    key: []
    for key in REWARD_TERM_KEYS
}

recent_rewards = deque(
    maxlen=SMOOTH_WINDOW
)


# --------------------------------------------------
# Training
# --------------------------------------------------

while global_step < TOTAL_TIMESTEPS:

    observations = []
    raw_actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    # --------------------------------------------------
    # Collect rollout
    # --------------------------------------------------

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

        # PPO rollout
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

        # --------------------------------------------------
        # Episode statistics
        # --------------------------------------------------

        episode_reward += (
            reward
        )

        episode_vx.append(
            info[
                "forward_velocity"
            ]
        )

        episode_vx_yaw.append(
            info[
                "forward_velocity_yaw"
            ]
        )

        episode_height.append(
            info[
                "height"
            ]
        )

        episode_upright.append(
            info[
                "upright"
            ]
        )

        left_contacts.append(
            info[
                "left_contact"
            ]
        )

        right_contacts.append(
            info[
                "right_contact"
            ]
        )

        single_support.append(
            info[
                "single_support"
            ]
        )

        # --------------------------------------------------
        # Reward breakdown
        # --------------------------------------------------

        reward_terms = info.get(
            "reward_terms",
            {},
        )

        for key in REWARD_TERM_KEYS:

            reward_term_history[
                key
            ].append(
                float(
                    reward_terms.get(
                        key,
                        0.0,
                    )
                )
            )

        # --------------------------------------------------
        # Episode completed
        # --------------------------------------------------

        if done:

            episode_number += 1

            episode_length = len(
                episode_vx
            )

            mean_vx = float(
                np.mean(
                    episode_vx
                )
            )

            mean_vx_yaw = float(
                np.mean(
                    episode_vx_yaw
                )
            )

            mean_height = float(
                np.mean(
                    episode_height
                )
            )

            mean_upright = float(
                np.mean(
                    episode_upright
                )
            )

            left_contact_ratio = float(
                np.mean(
                    left_contacts
                )
            )

            right_contact_ratio = float(
                np.mean(
                    right_contacts
                )
            )

            single_support_ratio = float(
                np.mean(
                    single_support
                )
            )

            mean_reward_terms = {

                key:
                    float(
                        np.mean(
                            reward_term_history[
                                key
                            ]
                        )
                    )
                    if reward_term_history[
                        key
                    ]
                    else 0.0

                for key
                in REWARD_TERM_KEYS
            }

            recent_rewards.append(
                episode_reward
            )

            smooth_reward = float(
                np.mean(
                    recent_rewards
                )
            )

            # --------------------------------------------------
            # Console output
            # --------------------------------------------------

            print(
                f"Episode {episode_number:4d} | "
                f"Step {global_step:7d} | "
                f"Reward {episode_reward:8.2f} | "
                f"Avg {smooth_reward:8.2f} | "
                f"vx(body) {mean_vx_yaw:6.3f} | "
                f"Length {episode_length:4d} | "
                f"SingleSupport "
                f"{single_support_ratio:5.2f} | "
                f"r_vel "
                f"{mean_reward_terms['velocity_tracking']:5.2f} | "
                f"r_gait "
                f"{mean_reward_terms['air_time']:5.2f} | "
                f"p_slide "
                f"{mean_reward_terms['foot_slide']:6.3f}"
            )

            # --------------------------------------------------
            # CSV row
            # --------------------------------------------------

            row = [
                episode_number,
                global_step,
                episode_reward,
                smooth_reward,
                mean_vx,
                mean_vx_yaw,
                episode_length,
                mean_height,
                mean_upright,
                left_contact_ratio,
                right_contact_ratio,
                single_support_ratio,
            ] + [
                mean_reward_terms[
                    key
                ]
                for key
                in REWARD_TERM_KEYS
            ]

            with open(
                CSV_PATH,
                "a",
                newline="",
            ) as csv_file:

                writer = csv.writer(
                    csv_file
                )

                writer.writerow(
                    row
                )

            # --------------------------------------------------
            # Reset episode statistics
            # --------------------------------------------------

            obs, _ = env.reset()

            episode_reward = 0.0

            episode_vx = []
            episode_vx_yaw = []

            episode_height = []
            episode_upright = []

            left_contacts = []
            right_contacts = []

            single_support = []

            reward_term_history = {
                key: []
                for key
                in REWARD_TERM_KEYS
            }

        if (
            global_step
            >= TOTAL_TIMESTEPS
        ):
            break

    # --------------------------------------------------
    # Bootstrap final value
    # --------------------------------------------------

    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(
        0
    )

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
    # GAE
    # --------------------------------------------------

    advantages, returns = (
        agent.compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
        )
    )

    # --------------------------------------------------
    # PPO update
    # --------------------------------------------------

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
        f"--- PPO update completed "
        f"at step {global_step} ---"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    agent.save(
        str(
            CHECKPOINTS_DIR
            / "g1_ppo.pt"
        )
    )


env.close()

print()
print(
    "Training complete."
)

print(
    f"Training log saved to: "
    f"{CSV_PATH}"
)