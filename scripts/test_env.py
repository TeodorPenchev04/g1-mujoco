import numpy as np

from envs.g1_env import G1Env


env = G1Env(
    observation_noise=False
)

obs, info = env.reset()


print()
print(
    "G1 curriculum phase-1 environment"
)

print(
    "Observation shape:",
    obs.shape,
)

print(
    "Action shape:",
    env.action_space.shape,
)

print(
    "Command:",
    env.command,
)

print(
    "Action scale:",
    env.action_scale,
)

print(
    "Velocity tracking std:",
    env.velocity_tracking_std,
)

print(
    "Yaw tracking weight:",
    env.yaw_tracking_weight,
)

print(
    "Control dt:",
    env.control_dt,
)

print(
    "Maximum steps:",
    env.max_steps,
)

print()


def print_step(
    prefix,
    step,
    reward,
    info,
):
    terms = (
        info["reward_terms"]
    )

    print(
        f"{prefix} {step:3d} | "
        f"R={reward:8.4f} | "
        f"vx="
        f"{info['forward_velocity_yaw']:6.3f}/"
        f"{info['target_velocity']:5.2f} | "
        f"h={info['height']:5.3f} | "
        f"up={info['upright']:5.3f} | "
        f"L/R="
        f"{info['left_contact']}/"
        f"{info['right_contact']} | "
        f"single="
        f"{info['single_support']} | "
        f"lin="
        f"{terms['track_lin_vel_xy']:7.4f} | "
        f"air="
        f"{terms['feet_air_time']:7.4f} | "
        f"slide="
        f"{terms['feet_slide']:8.5f} | "
        f"orient="
        f"{terms['flat_orientation']:8.5f} | "
        f"acc="
        f"{terms['dof_acc']:8.5f} | "
        f"torque="
        f"{terms['dof_torques']:8.5f}"
    )


# ======================================================
# Zero-action test
# ======================================================

print(
    "=== ZERO ACTION ==="
)

for step in range(
    200
):
    action = np.zeros(
        env.action_space.shape,
        dtype=np.float32,
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

    if (
        step % 20 == 0
        or terminated
        or truncated
    ):
        print_step(
            "ZERO",
            step,
            reward,
            info,
        )

    if (
        terminated
        or truncated
    ):
        print(
            "Zero-action episode ended "
            f"at step {step}."
        )
        break


# ======================================================
# Small random actions
# ======================================================

print()
print(
    "=== SMALL RANDOM ACTIONS ==="
)

obs, info = env.reset()


for step in range(
    150
):
    action = (
        0.10
        * env.action_space.sample()
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

    if (
        step % 10 == 0
        or terminated
        or truncated
    ):
        print_step(
            "RAND",
            step,
            reward,
            info,
        )

    if (
        terminated
        or truncated
    ):
        print(
            "Random-action episode ended "
            f"at step {step}."
        )
        break


env.close()