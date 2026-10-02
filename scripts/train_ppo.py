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


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

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
    / "phase1_training_log.csv"
)

CHECKPOINT_PATH = (
    CHECKPOINTS_DIR
    / "g1_phase1_ppo.pt"
)

BEST_CHECKPOINT_PATH = (
    CHECKPOINTS_DIR
    / "g1_phase1_best.pt"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


REWARD_TERM_KEYS = [
    "track_lin_vel_xy",
    "track_ang_vel_z",
    "feet_air_time",
    "feet_slide",
    "dof_pos_limits",
    "joint_deviation_hip",
    "joint_deviation_arms",
    "joint_deviation_fingers",
    "joint_deviation_torso",
    "flat_orientation",
    "ang_vel_xy",
    "action_rate",
    "dof_acc",
    "dof_torques",
    "termination",
]


# ======================================================
# Environment
# ======================================================

env = G1Env(
    observation_noise=False
)

obs_dim = (
    env.observation_space.shape[0]
)

action_dim = (
    env.action_space.shape[0]
)


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


print()
print(
    "G1 curriculum phase 1"
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
    env.command[0],
)
print(
    "Action scale:",
    env.action_scale,
)
print(
    "Velocity std:",
    env.velocity_tracking_std,
)
print(
    "Control dt:",
    env.control_dt,
)
print(
    "Maximum episode length:",
    env.max_steps,
)
print()


# ======================================================
# PPO
# ======================================================

agent = PPO(
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
)

# Current PPO class initializes:
#
# log_std = -0.5  -> std ~= 0.61
#
# That is very aggressive for a 29-DoF humanoid.
# Start this curriculum with smaller exploration.
with torch.no_grad():
    agent.network.log_std.fill_(
        -1.0
    )


# ======================================================
# CSV
# ======================================================

csv_header = [
    "episode",
    "global_step",

    "episode_reward",
    "smooth_reward",

    "episode_length",

    "mean_vx",
    "mean_abs_velocity_error",

    "mean_height",
    "mean_upright",

    "left_contact_ratio",
    "right_contact_ratio",
    "single_support_ratio",
] + [
    f"mean_{key}"
    for key
    in REWARD_TERM_KEYS
]


with open(
    CSV_PATH,
    "w",
    newline="",
) as file:
    writer = csv.writer(
        file
    )

    writer.writerow(
        csv_header
    )


# ======================================================
# Stats
# ======================================================

def new_episode_stats():
    return {
        "reward":
            0.0,

        "vx":
            [],

        "height":
            [],

        "upright":
            [],

        "left_contact":
            [],

        "right_contact":
            [],

        "single_support":
            [],

        "reward_terms":
            {
                key: []
                for key
                in REWARD_TERM_KEYS
            },
    }


obs, _ = env.reset()

global_step = 0
episode_number = 0

stats = new_episode_stats()

recent_rewards = deque(
    maxlen=SMOOTH_WINDOW
)

best_episode_reward = (
    -np.inf
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

        # ----------------------------------------------
        # Episode stats
        # ----------------------------------------------

        stats[
            "reward"
        ] += reward

        stats[
            "vx"
        ].append(
            info[
                "forward_velocity_yaw"
            ]
        )

        stats[
            "height"
        ].append(
            info[
                "height"
            ]
        )

        stats[
            "upright"
        ].append(
            info[
                "upright"
            ]
        )

        stats[
            "left_contact"
        ].append(
            info[
                "left_contact"
            ]
        )

        stats[
            "right_contact"
        ].append(
            info[
                "right_contact"
            ]
        )

        stats[
            "single_support"
        ].append(
            info[
                "single_support"
            ]
        )

        reward_terms = info.get(
            "reward_terms",
            {},
        )

        for key in (
            REWARD_TERM_KEYS
        ):
            stats[
                "reward_terms"
            ][key].append(
                float(
                    reward_terms.get(
                        key,
                        0.0,
                    )
                )
            )

        # ----------------------------------------------
        # End episode
        # ----------------------------------------------

        if done:
            episode_number += 1

            episode_length = len(
                stats["vx"]
            )

            mean_vx = float(
                np.mean(
                    stats["vx"]
                )
            )

            mean_abs_velocity_error = float(
                np.mean(
                    np.abs(
                        np.asarray(
                            stats["vx"]
                        )
                        - env.command[0]
                    )
                )
            )

            mean_height = float(
                np.mean(
                    stats["height"]
                )
            )

            mean_upright = float(
                np.mean(
                    stats["upright"]
                )
            )

            left_contact_ratio = float(
                np.mean(
                    stats[
                        "left_contact"
                    ]
                )
            )

            right_contact_ratio = float(
                np.mean(
                    stats[
                        "right_contact"
                    ]
                )
            )

            single_support_ratio = float(
                np.mean(
                    stats[
                        "single_support"
                    ]
                )
            )

            mean_reward_terms = {
                key:
                    float(
                        np.mean(
                            stats[
                                "reward_terms"
                            ][key]
                        )
                    )
                    if stats[
                        "reward_terms"
                    ][key]
                    else 0.0

                for key
                in REWARD_TERM_KEYS
            }

            recent_rewards.append(
                stats["reward"]
            )

            smooth_reward = float(
                np.mean(
                    recent_rewards
                )
            )

            print(
                f"Ep {episode_number:5d} | "
                f"Step {global_step:7d} | "
                f"R {stats['reward']:8.2f} | "
                f"Avg {smooth_reward:8.2f} | "
                f"Len {episode_length:4d} | "
                f"vx {mean_vx:6.3f}/0.300 | "
                f"err {mean_abs_velocity_error:5.3f} | "
                f"single {single_support_ratio:4.2f} | "
                f"linR "
                f"{mean_reward_terms['track_lin_vel_xy']:.4f} | "
                f"airR "
                f"{mean_reward_terms['feet_air_time']:.4f} | "
                f"term "
                f"{mean_reward_terms['termination']:.4f}"
            )

            # ------------------------------------------
            # CSV
            # ------------------------------------------

            row = [
                episode_number,
                global_step,

                stats["reward"],
                smooth_reward,

                episode_length,

                mean_vx,
                mean_abs_velocity_error,

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
            ) as file:
                writer = csv.writer(
                    file
                )

                writer.writerow(
                    row
                )

            # ------------------------------------------
            # Best model
            # ------------------------------------------

            if (
                stats["reward"]
                > best_episode_reward
            ):
                best_episode_reward = (
                    stats["reward"]
                )

                agent.save(
                    str(
                        BEST_CHECKPOINT_PATH
                    )
                )

            obs, _ = env.reset()

            stats = (
                new_episode_stats()
            )

        if (
            global_step
            >= TOTAL_TIMESTEPS
        ):
            break

    # ==================================================
    # Bootstrap
    # ==================================================

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

    # ==================================================
    # GAE
    # ==================================================

    advantages, returns = (
        agent.compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
        )
    )

    # ==================================================
    # PPO update
    # ==================================================

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

    agent.save(
        str(
            CHECKPOINT_PATH
        )
    )


env.close()


print()
print(
    "Training complete."
)

print(
    "Current checkpoint:",
    CHECKPOINT_PATH,
)

print(
    "Best checkpoint:",
    BEST_CHECKPOINT_PATH,
)

print(
    "Training log:",
    CSV_PATH,
)