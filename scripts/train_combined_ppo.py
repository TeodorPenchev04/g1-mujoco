from pathlib import Path
import csv
from collections import deque

import numpy as np
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


SOURCE_CHECKPOINT = (
    CHECKPOINT_DIR
    / "g1_combined_phase_300k_backup.pt"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "g1_speedmatched_final.pt"
)

CSV_PATH = (
    RESULTS_DIR
    / "speedmatched_training_log.csv"
)


TOTAL_TIMESTEPS = 300_000

ROLLOUT_STEPS = 2048

SMOOTH_WINDOW = 20

CHECKPOINT_INTERVAL = 25_000


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================
# Environment
# ======================================================

env = G1CombinedEnv()

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
    "Reference phase scale:",
    env.reference_phase_scale,
)

print(
    "Effective reference duration:",
    f"{env.effective_motion_duration:.3f} s",
)

print(
    "Source checkpoint:",
    SOURCE_CHECKPOINT,
)

print()


# ======================================================
# PPO
# ======================================================

agent = PPO(
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,

    # Keep same fine-tuning settings as previous run
    # so timing is the primary experimental change.

    lr=1e-4,

    entropy_coef=0.002,
)


if not SOURCE_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Could not find source checkpoint:\n"
        f"{SOURCE_CHECKPOINT}"
    )


agent.load(
    str(
        SOURCE_CHECKPOINT
    )
)


print(
    "Loaded existing phase-aware controller."
)

print()


# ======================================================
# CSV
# ======================================================

with open(
    CSV_PATH,
    "w",
    newline="",
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([
        "episode",
        "global_step",

        "episode_reward",
        "smooth_reward",

        "episode_length",

        "mean_vx",
        "mean_height",
        "mean_upright",

        "mean_locomotion_reward",
        "mean_normalized_locomotion_reward",

        "mean_imitation_reward",

        "mean_pose_reward",
        "mean_reference_joint_velocity_reward",
        "mean_reference_orientation_reward",

        "mean_velocity_tracking",
        "mean_gait_reward",

        "reference_phase_scale",
    ])


# ======================================================
# Statistics helper
# ======================================================

def new_episode_stats():
    return {
        "reward": 0.0,

        "vx": [],
        "height": [],
        "upright": [],

        "locomotion": [],
        "normalized_locomotion": [],

        "imitation": [],

        "pose": [],
        "ref_joint_vel": [],
        "ref_orientation": [],

        "velocity_tracking": [],
        "gait": [],
    }


# ======================================================
# Initialize
# ======================================================

obs, _ = env.reset()

stats = new_episode_stats()

global_step = 0
episode_number = 0

recent_rewards = deque(
    maxlen=SMOOTH_WINDOW
)


next_checkpoint_step = (
    CHECKPOINT_INTERVAL
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

        obs = (
            next_obs
        )

        global_step += 1

        # ==================================================
        # Episode statistics
        # ==================================================

        stats["reward"] += (
            reward
        )

        stats["vx"].append(
            info[
                "forward_velocity"
            ]
        )

        stats["height"].append(
            info[
                "height"
            ]
        )

        stats["upright"].append(
            info[
                "upright"
            ]
        )

        stats[
            "locomotion"
        ].append(
            info[
                "locomotion_reward"
            ]
        )

        stats[
            "normalized_locomotion"
        ].append(
            info[
                "normalized_locomotion_reward"
            ]
        )

        stats[
            "imitation"
        ].append(
            info[
                "imitation_reward"
            ]
        )

        stats[
            "pose"
        ].append(
            info[
                "pose_reward"
            ]
        )

        stats[
            "ref_joint_vel"
        ].append(
            info[
                "reference_joint_velocity_reward"
            ]
        )

        stats[
            "ref_orientation"
        ].append(
            info[
                "reference_orientation_reward"
            ]
        )

        stats[
            "velocity_tracking"
        ].append(
            info[
                "velocity_tracking"
            ]
        )

        stats[
            "gait"
        ].append(
            info[
                "gait_reward"
            ]
        )

        # ==================================================
        # Episode complete
        # ==================================================

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

            mean_locomotion = float(
                np.mean(
                    stats[
                        "locomotion"
                    ]
                )
            )

            mean_normalized_locomotion = float(
                np.mean(
                    stats[
                        "normalized_locomotion"
                    ]
                )
            )

            mean_imitation = float(
                np.mean(
                    stats[
                        "imitation"
                    ]
                )
            )

            mean_pose = float(
                np.mean(
                    stats[
                        "pose"
                    ]
                )
            )

            mean_ref_joint_vel = float(
                np.mean(
                    stats[
                        "ref_joint_vel"
                    ]
                )
            )

            mean_ref_orientation = float(
                np.mean(
                    stats[
                        "ref_orientation"
                    ]
                )
            )

            mean_velocity_tracking = float(
                np.mean(
                    stats[
                        "velocity_tracking"
                    ]
                )
            )

            mean_gait = float(
                np.mean(
                    stats[
                        "gait"
                    ]
                )
            )

            print(
                f"Ep {episode_number:4d} | "
                f"Step {global_step:7d} | "
                f"R {stats['reward']:7.2f} | "
                f"Avg {smooth_reward:7.2f} | "
                f"Len {episode_length:4d} | "
                f"vx {mean_vx:5.3f} | "
                f"LocoN "
                f"{mean_normalized_locomotion:5.3f} | "
                f"Imit {mean_imitation:5.3f} | "
                f"Pose {mean_pose:5.3f} | "
                f"JVel {mean_ref_joint_vel:5.3f} | "
                f"Orient {mean_ref_orientation:5.3f}"
            )

            # ==============================================
            # Write CSV
            # ==============================================

            with open(
                CSV_PATH,
                "a",
                newline="",
            ) as file:

                writer = csv.writer(
                    file
                )

                writer.writerow([
                    episode_number,
                    global_step,

                    stats[
                        "reward"
                    ],
                    smooth_reward,

                    episode_length,

                    mean_vx,
                    mean_height,
                    mean_upright,

                    mean_locomotion,
                    mean_normalized_locomotion,

                    mean_imitation,

                    mean_pose,
                    mean_ref_joint_vel,
                    mean_ref_orientation,

                    mean_velocity_tracking,
                    mean_gait,

                    env.reference_phase_scale,
                ])

            # ==============================================
            # Reset
            # ==============================================

            obs, _ = (
                env.reset()
            )

            stats = (
                new_episode_stats()
            )

        if (
            global_step
            >= TOTAL_TIMESTEPS
        ):
            break

    # ======================================================
    # Bootstrap critic
    # ======================================================

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

    # ======================================================
    # GAE
    # ======================================================

    advantages, returns = (
        agent.compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
        )
    )

    # ======================================================
    # PPO update
    # ======================================================

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
        f"--- Speed-matched PPO update "
        f"at step {global_step} ---"
    )

    # ======================================================
    # Periodic checkpoints
    # ======================================================

    while (
        global_step
        >= next_checkpoint_step
        and next_checkpoint_step
        <= TOTAL_TIMESTEPS
    ):

        checkpoint_name = (
            f"g1_speedmatched_"
            f"{next_checkpoint_step // 1000:03d}k.pt"
        )

        checkpoint_path = (
            CHECKPOINT_DIR
            / checkpoint_name
        )

        agent.save(
            str(
                checkpoint_path
            )
        )

        print(
            "Saved checkpoint:",
            checkpoint_path.name,
        )

        next_checkpoint_step += (
            CHECKPOINT_INTERVAL
        )

    # Keep a continuously updated final file too.

    agent.save(
        str(
            FINAL_CHECKPOINT
        )
    )


env.close()


print()
print(
    "Speed-matched training complete."
)

print(
    "Final checkpoint:",
    FINAL_CHECKPOINT,
)

print(
    "Training log:",
    CSV_PATH,
)