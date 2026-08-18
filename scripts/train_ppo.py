import os
import numpy as np
import torch

from envs.g1_env import G1Env
from algorithms.ppo import PPO


TOTAL_TIMESTEPS = 50_000 
ROLLOUT_STEPS = 2048


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


agent = PPO(
    obs_dim,
    action_dim,
    device=device,
)


obs, _ = env.reset()

global_step = 0
episode_reward = 0.0
episode_number = 0


while global_step < TOTAL_TIMESTEPS:

    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    for _ in range(ROLLOUT_STEPS):

        action, log_prob, value = (
            agent.select_action(obs)
        )

        # Environment expects [-1, 1]
        clipped_action = np.clip(
            action,
            -1.0,
            1.0,
        )

        next_obs, reward, terminated, truncated, info = (
            env.step(clipped_action)
        )

        done = (
            terminated
            or truncated
        )

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        log_probs.append(log_prob)
        values.append(value)

        obs = next_obs

        episode_reward += reward
        global_step += 1

        if done:
            episode_number += 1

            print(
                f"Episode {episode_number} | "
                f"Step {global_step} | "
                f"Reward {episode_reward:.2f}"
            )

            obs, _ = env.reset()

            episode_reward = 0.0

    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    with torch.no_grad():
        next_value = (
            agent.network.critic(
                obs_tensor
            )
            .squeeze()
            .item()
        )

    advantages, returns = (
        agent.compute_gae(
            rewards,
            values,
            dones,
            next_value,
        )
    )

    agent.update(
        np.array(observations),
        np.array(actions),
        np.array(log_probs),
        advantages,
        returns,
    )

    print(
        f"PPO update completed "
        f"at step {global_step}"
    )

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    agent.save(
        "checkpoints/g1_ppo.pt"
    )


env.close()