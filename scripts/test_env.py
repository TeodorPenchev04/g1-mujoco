from envs.g1_env import G1Env


env = G1Env()

obs, info = env.reset()

print(
    "Observation:",
    obs.shape,
)

print(
    "Action:",
    env.action_space.shape,
)

for step in range(100):

    action = (
        env.action_space.sample()
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    print(
        f"{step:3d} | "
        f"reward={reward:.3f} | "
        f"vx={info['forward_velocity']:.3f} | "
        f"height={info['height']:.3f}"
    )

    if terminated or truncated:
        print("Reset")

        obs, info = env.reset()


env.close()