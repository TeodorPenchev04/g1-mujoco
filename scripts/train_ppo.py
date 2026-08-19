import os
import csv
from collections import deque

import numpy as np
import torch

from algorithms.ppo import PPO
from envs.g1_env import G1Env


TOTAL_TIMESTEPS = 1_000_000
ROLLOUT_STEPS = 2048
SMOOTH_WINDOW = 20

CSV_PATH = "results/training_log.csv"


os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


env = G1Env()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


print("Device:", device)
print("Observation dimension:", obs_dim)
print("Action dimension:", action_dim)
print("Target velocity:", env.target_velocity)
print()


agent = PPO(
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
)


with open(
    CSV_PATH,
    "w",
    newline="",
) as csv_file:

    writer = csv.writer(
        csv_file
    )

    writer.writerow([
        "episode",
        "global_step",
        "episode_reward",
        "smooth_reward",
        "mean_vx",
        "episode_length",
        "mean_height",
        "mean_upright",
        "left_contact_ratio",
        "right_contact_ratio",
        "single_support_ratio",
    ])


obs, _ = env.reset()

global_step = 0
episode_number = 0

episode_reward = 0.0

episode_vx = []
episode_height = []
episode_upright = []

left_contacts = []
right_contacts = []
single_support = []

recent_rewards = deque(
    maxlen=SMOOTH_WINDOW
)


while global_step < TOTAL_TIMESTEPS:

    observations = []
    raw_actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    for _ in range(ROLLOUT_STEPS):

        (
            action,
            raw_action,
            log_prob,
            value,
        ) = agent.select_action(obs)

        (
            next_obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

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

        episode_reward += reward

        episode_vx.append(
            info["forward_velocity"]
        )

        episode_height.append(
            info["height"]
        )

        episode_upright.append(
            info["upright"]
        )

        left_contacts.append(
            info["left_contact"]
        )

        right_contacts.append(
            info["right_contact"]
        )

        single_support.append(
            int(
                info["left_contact"]
                != info["right_contact"]
            )
        )

        if done:

            episode_number += 1

            episode_length = len(
                episode_vx
            )

            mean_vx = float(
                np.mean(episode_vx)
            )

            mean_height = float(
                np.mean(episode_height)
            )

            mean_upright = float(
                np.mean(episode_upright)
            )

            left_contact_ratio = float(
                np.mean(left_contacts)
            )

            right_contact_ratio = float(
                np.mean(right_contacts)
            )

            single_support_ratio = float(
                np.mean(single_support)
            )

            recent_rewards.append(
                episode_reward
            )

            smooth_reward = float(
                np.mean(recent_rewards)
            )

            print(
                f"Episode {episode_number:4d} | "
                f"Step {global_step:7d} | "
                f"Reward {episode_reward:8.2f} | "
                f"Avg {smooth_reward:8.2f} | "
                f"vx {mean_vx:6.3f} | "
                f"Length {episode_length:4d} | "
                f"SingleSupport "
                f"{single_support_ratio:5.2f}"
            )

            with open(
                CSV_PATH,
                "a",
                newline="",
            ) as csv_file:

                writer = csv.writer(
                    csv_file
                )

                writer.writerow([
                    episode_number,
                    global_step,
                    episode_reward,
                    smooth_reward,
                    mean_vx,
                    episode_length,
                    mean_height,
                    mean_upright,
                    left_contact_ratio,
                    right_contact_ratio,
                    single_support_ratio,
                ])

            obs, _ = env.reset()

            episode_reward = 0.0

            episode_vx = []
            episode_height = []
            episode_upright = []

            left_contacts = []
            right_contacts = []
            single_support = []

        if (
            global_step
            >= TOTAL_TIMESTEPS
        ):
            break

    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    with torch.no_grad():

        next_value = (
            agent.network
            .critic(obs_tensor)
            .squeeze()
            .item()
        )

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
        f"--- PPO update completed "
        f"at step {global_step} ---"
    )

    agent.save(
        "checkpoints/g1_ppo.pt"
    )


env.close()

print()
print("Training complete.")
print(
    f"Training log saved to: {CSV_PATH}"
)