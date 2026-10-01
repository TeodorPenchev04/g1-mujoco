import torch

from algorithms.ppo import PPO
from envs.g1_combined_env import G1CombinedEnv


CHECKPOINT_PATH = (
    "checkpoints/"
    "g1_ppo_stable_backup.pt"
)


env = G1CombinedEnv()

obs_dim = (
    env.observation_space.shape[0]
)

action_dim = (
    env.action_space.shape[0]
)


print()
print(
    "Combined observation dimension:",
    obs_dim,
)

print(
    "Combined action dimension:",
    action_dim,
)


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


agent = PPO(
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
)


try:
    agent.load(
        CHECKPOINT_PATH
    )

    print()
    print(
        "SUCCESS: stable locomotion "
        "checkpoint loaded."
    )

except Exception as error:
    print()
    print(
        "FAILED to load checkpoint."
    )

    print(error)

    env.close()

    raise


obs, _ = env.reset()


print()
print(
    "Initial observation shape:",
    obs.shape,
)


for step in range(10):

    (
        action,
        raw_action,
        log_prob,
        value,
    ) = agent.select_action(
        obs
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

    print(
        f"Step {step + 1:2d} | "
        f"Reward {reward:7.3f} | "
        f"Loco {info['locomotion_reward']:7.3f} | "
        f"Imit {info['imitation_reward']:6.3f} | "
        f"Pose {info['pose_reward']:6.3f}"
    )

    if (
        terminated
        or truncated
    ):
        print(
            "Episode terminated "
            "during test."
        )

        break


env.close()