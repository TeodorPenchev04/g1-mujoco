from envs.g1_env import G1Env


env = G1Env()

obs, info = env.reset()

print("G1 environment loaded")
print(
    "Observation shape:",
    obs.shape,
)
print(
    "Action shape:",
    env.action_space.shape,
)
print(
    "Target velocity:",
    env.target_velocity,
)
print(
    "Standing height:",
    env.target_height,
)
print()


for step in range(100):
    action = env.action_space.sample()

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    print(
        f"{step:3d} | "
        f"reward={reward:6.3f} | "
        f"vx={info['forward_velocity']:6.3f} | "
        f"height={info['height']:6.3f}"
    )

    if terminated or truncated:
        print("Episode ended -- resetting")
        obs, info = env.reset()


env.close()