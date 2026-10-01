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
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CSV_PATH = RESULTS_DIR / "ta_reference_training_log.csv"
CHECKPOINT_PATH = CHECKPOINTS_DIR / "g1_ta_reference_ppo.pt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

REWARD_TERM_KEYS = [
    "track_lin_vel_xy",
    "track_ang_vel_z",
    "feet_air_time",
    "feet_slide",
    "dof_pos_limits",
    "joint_deviation_hip",
    "joint_deviation_arms",
    "joint_deviation_torso",
    "flat_orientation",
    "ang_vel_xy",
    "action_rate",
    "dof_acc",
    "dof_torques",
    "termination",
]


env = G1Env(observation_noise=True)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Device:", device)
print("Observation dimension:", obs_dim)
print("Action dimension:", action_dim)
print("Control dt:", env.control_dt)
print("Episode steps:", env.max_steps)
print("Command resample steps:", env.command_resample_steps)
print("Checkpoint:", CHECKPOINT_PATH)
print()

agent = PPO(
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
)

csv_header = [
    "episode",
    "global_step",
    "episode_reward",
    "smooth_reward",
    "episode_length",
    "mean_vx_body",
    "mean_cmd_vx",
    "mean_cmd_yaw",
    "mean_height",
    "left_contact_ratio",
    "right_contact_ratio",
    "single_support_ratio",
] + [f"mean_{key}" for key in REWARD_TERM_KEYS]

with open(CSV_PATH, "w", newline="") as csv_file:
    csv.writer(csv_file).writerow(csv_header)


def new_episode_stats():
    return {
        "reward": 0.0,
        "vx_body": [],
        "cmd_vx": [],
        "cmd_yaw": [],
        "height": [],
        "left_contact": [],
        "right_contact": [],
        "single_support": [],
        "reward_terms": {key: [] for key in REWARD_TERM_KEYS},
    }


obs, _ = env.reset()
stats = new_episode_stats()
global_step = 0
episode_number = 0
recent_rewards = deque(maxlen=SMOOTH_WINDOW)

while global_step < TOTAL_TIMESTEPS:
    observations = []
    raw_actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    for _ in range(ROLLOUT_STEPS):
        action, raw_action, log_prob, value = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        observations.append(obs.copy())
        raw_actions.append(raw_action.copy())
        rewards.append(reward)
        dones.append(float(done))
        log_probs.append(log_prob)
        values.append(value)

        obs = next_obs
        global_step += 1

        stats["reward"] += reward
        stats["vx_body"].append(info["forward_velocity_yaw"])
        stats["cmd_vx"].append(info["target_velocity"])
        stats["cmd_yaw"].append(info["target_yaw_rate"])
        stats["height"].append(info["height"])
        stats["left_contact"].append(info["left_contact"])
        stats["right_contact"].append(info["right_contact"])
        stats["single_support"].append(info["single_support"])

        reward_terms = info.get("reward_terms", {})
        for key in REWARD_TERM_KEYS:
            stats["reward_terms"][key].append(
                float(reward_terms.get(key, 0.0))
            )

        if done:
            episode_number += 1
            episode_length = len(stats["vx_body"])
            recent_rewards.append(stats["reward"])
            smooth_reward = float(np.mean(recent_rewards))

            mean_vx = float(np.mean(stats["vx_body"]))
            mean_cmd_vx = float(np.mean(stats["cmd_vx"]))
            mean_cmd_yaw = float(np.mean(stats["cmd_yaw"]))
            mean_height = float(np.mean(stats["height"]))
            left_contact_ratio = float(np.mean(stats["left_contact"]))
            right_contact_ratio = float(np.mean(stats["right_contact"]))
            single_support_ratio = float(np.mean(stats["single_support"]))

            mean_terms = {
                key: (
                    float(np.mean(stats["reward_terms"][key]))
                    if stats["reward_terms"][key]
                    else 0.0
                )
                for key in REWARD_TERM_KEYS
            }

            print(
                f"Ep {episode_number:4d} | "
                f"Step {global_step:7d} | "
                f"R {stats['reward']:8.2f} | "
                f"Avg {smooth_reward:8.2f} | "
                f"Len {episode_length:4d} | "
                f"vx {mean_vx:5.2f}/{mean_cmd_vx:5.2f} | "
                f"yawcmd {mean_cmd_yaw:5.2f} | "
                f"Single {single_support_ratio:4.2f} | "
                f"Lin {mean_terms['track_lin_vel_xy']:5.3f} | "
                f"Yaw {mean_terms['track_ang_vel_z']:5.3f} | "
                f"Air {mean_terms['feet_air_time']:6.4f} | "
                f"Slide {mean_terms['feet_slide']:7.4f}"
            )

            row = [
                episode_number,
                global_step,
                stats["reward"],
                smooth_reward,
                episode_length,
                mean_vx,
                mean_cmd_vx,
                mean_cmd_yaw,
                mean_height,
                left_contact_ratio,
                right_contact_ratio,
                single_support_ratio,
            ] + [mean_terms[key] for key in REWARD_TERM_KEYS]

            with open(CSV_PATH, "a", newline="") as csv_file:
                csv.writer(csv_file).writerow(row)

            obs, _ = env.reset()
            stats = new_episode_stats()

        if global_step >= TOTAL_TIMESTEPS:
            break

    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    with torch.no_grad():
        next_value = agent.network.critic(obs_tensor).squeeze().item()

    advantages, returns = agent.compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
    )

    agent.update(
        observations=np.asarray(observations),
        raw_actions=np.asarray(raw_actions),
        old_log_probs=np.asarray(log_probs),
        advantages=advantages,
        returns=returns,
    )

    agent.save(str(CHECKPOINT_PATH))
    print(f"--- PPO update completed at step {global_step} ---")


env.close()
print()
print("Training complete.")
print("Checkpoint:", CHECKPOINT_PATH)
print("Training log:", CSV_PATH)
