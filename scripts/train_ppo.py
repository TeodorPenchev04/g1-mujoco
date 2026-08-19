import os
from collections import deque

import numpy as np
import torch

from algorithms.ppo import PPO
from envs.g1_env import G1Env


TOTAL_TIMESTEPS = 50_000
ROLLOUT_STEPS = 2048
SMOOTH_WINDOW = 20


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


obs, _ = env.reset()

global_step = 0
episode_number = 0

episode_reward = 0.0
episode_vx = []

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

        rewards.append(reward)
        dones.append(float(done))
        log_probs.append(log_prob)
        values.append(value)

        obs = next_obs

        episode_reward += reward

        episode_vx.append(
            info["forward_velocity"]
        )

        global_step += 1

        if done:
            episode_number += 1

            recent_rewards.append(
                episode_reward
            )

            smooth_reward = float(
                np.mean(recent_rewards)
            )

            mean_vx = float(
                np.mean(episode_vx)
            )

            print(
                f"Episode {episode_number:4d} | "
                f"Step {global_step:7d} | "
                f"Reward {episode_reward:8.2f} | "
                f"Avg({len(recent_rewards):2d}) "
                f"{smooth_reward:8.2f} | "
                f"Mean vx {mean_vx:5.3f} | "
                f"Length {len(episode_vx):4d}"
            )

            obs, _ = env.reset()

            episode_reward = 0.0
            episode_vx = []

        if global_step >= TOTAL_TIMESTEPS:
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

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    agent.save(
        "checkpoints/g1_ppo.pt"
    )


env.close()

print("Training complete.")