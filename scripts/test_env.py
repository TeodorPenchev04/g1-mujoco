import numpy as np

from envs.g1_env import G1Env


env = G1Env(observation_noise=False)

obs, info = env.reset(
    options={
        "command": np.array(
            [0.5, 0.0, 0.0],
            dtype=np.float64,
        )
    }
)

print("G1 TA-reference environment loaded")
print("Observation shape:", obs.shape)
print("Action shape:", env.action_space.shape)
print("Fixed test command:", env.command)
print("Standing height:", env.target_height)
print("Physics dt:", env.model.opt.timestep)
print("Frame skip:", env.frame_skip)
print("Control dt:", env.control_dt)
print("Episode steps:", env.max_steps)
print("Command resample steps:", env.command_resample_steps)
print("Hip deviation joints:", env.hip_deviation_joint_ids)
print("Arm deviation joints:", env.arm_deviation_joint_ids)
print("Torso deviation joints:", env.torso_deviation_joint_ids)
print("Ankle joints:", env.ankle_joint_ids)
print()


def print_step(prefix, step, reward, info):
    terms = info.get("reward_terms", {})

    print(
        f"{prefix} {step:3d} | "
        f"reward={reward:8.3f} | "
        f"vx={info['forward_velocity_yaw']:6.3f}/"
        f"{info['target_velocity']:5.2f} | "
        f"yaw_cmd={info['target_yaw_rate']:6.3f} | "
        f"height={info['height']:6.3f} | "
        f"L/R={info['left_contact']}/{info['right_contact']} | "
        f"single={info['single_support']} | "
        f"lin={terms.get('track_lin_vel_xy', 0.0):6.3f} | "
        f"yaw={terms.get('track_ang_vel_z', 0.0):6.3f} | "
        f"air={terms.get('feet_air_time', 0.0):7.4f} | "
        f"slide={terms.get('feet_slide', 0.0):8.5f} | "
        f"orient={terms.get('flat_orientation', 0.0):8.5f} | "
        f"rate={terms.get('action_rate', 0.0):8.5f}"
    )


print("=== Zero-action test, command = [0.5, 0, 0] ===")

for step in range(100):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 10 == 0 or terminated or truncated:
        print_step("ZERO", step, reward, info)

    if terminated or truncated:
        print("Episode ended during zero-action test.")
        break


print()
print("=== Small-random-action test, command = [0.5, 0, 0] ===")

obs, info = env.reset(
    options={
        "command": np.array(
            [0.5, 0.0, 0.0],
            dtype=np.float64,
        )
    }
)

for step in range(150):
    action = 0.20 * env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 10 == 0 or terminated or truncated:
        print_step("RAND", step, reward, info)

    if terminated or truncated:
        print("Episode ended during random-action test.")
        break


env.close()
