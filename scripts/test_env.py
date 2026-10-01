import numpy as np

from envs.g1_env import G1Env


env = G1Env()

obs, info = env.reset()


print(
    "G1 environment loaded"
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
    "Target velocity:",
    env.target_velocity,
)

print(
    "Standing height:",
    env.target_height,
)

print(
    "Control dt:",
    env.control_dt,
)

print(
    "Posture qpos indices:",
    env.posture_qpos_indices,
)

print(
    "Ankle joint ids:",
    env.ankle_joint_ids,
)

print()


def print_step(
    prefix,
    step,
    reward,
    info,
):

    terms = info.get(
        "reward_terms",
        {},
    )

    print(
        f"{prefix} {step:3d} | "
        f"reward={reward:7.3f} | "
        f"vx_body="
        f"{info['forward_velocity_yaw']:6.3f} | "
        f"height="
        f"{info['height']:6.3f} | "
        f"upright="
        f"{info['upright']:5.3f} | "
        f"L/R="
        f"{info['left_contact']}/"
        f"{info['right_contact']} | "
        f"single="
        f"{info['single_support']} | "
        f"r_vel="
        f"{terms.get('velocity_tracking', 0.0):5.2f} | "
        f"r_air="
        f"{terms.get('air_time', 0.0):5.2f} | "
        f"p_slide="
        f"{terms.get('foot_slide', 0.0):7.4f} | "
        f"p_rate="
        f"{terms.get('action_rate', 0.0):7.4f}"
    )


# --------------------------------------------------
# Test 1:
# Zero-action standing sanity check
# --------------------------------------------------

print(
    "=== Zero-action sanity test ==="
)

for step in range(
    100
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
        step % 10 == 0
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
            "Episode ended during "
            "zero-action test -- resetting"
        )

        obs, info = (
            env.reset()
        )

        break


# --------------------------------------------------
# Test 2:
# Small random actions
# --------------------------------------------------

print()

print(
    "=== Small-random-action test ==="
)

obs, info = env.reset()


for step in range(
    150
):

    # Only 20% of the full action range.
    # This is easier to inspect than completely
    # random [-1, 1] actions.
    action = (
        0.20
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
            "Episode ended -- resetting"
        )

        obs, info = (
            env.reset()
        )

        break


env.close()