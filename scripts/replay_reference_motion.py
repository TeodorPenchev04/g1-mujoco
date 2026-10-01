import pickle
from pathlib import Path

import imageio
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_ROOT = PROJECT_ROOT.parent / "mujoco_menagerie"
MODEL_PATH = MENAGERIE_ROOT / "unitree_g1" / "scene.xml"
MOTION_PATH = PROJECT_ROOT / "motions" / "g1_walk.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
VIDEO_PATH = RESULTS_DIR / "reference_walk.mp4"


def expmap_to_quat(expmap):
    angle = np.linalg.norm(expmap)

    if angle < 1e-8:
        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    axis = expmap / angle
    half_angle = 0.5 * angle

    quat = np.empty(4, dtype=np.float64)
    quat[0] = np.cos(half_angle)
    quat[1:] = axis * np.sin(half_angle)

    return quat


# --------------------------------------------------
# Check paths
# --------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Could not find MuJoCo model at:\n{MODEL_PATH}"
    )

if not MOTION_PATH.is_file():
    raise FileNotFoundError(
        f"Could not find walking motion at:\n{MOTION_PATH}"
    )


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)


# --------------------------------------------------
# Load reference motion
# --------------------------------------------------

with open(MOTION_PATH, "rb") as f:
    motion = pickle.load(f)

fps = int(motion["fps"])

motion_frames = np.asarray(
    motion["frames"],
    dtype=np.float64,
)

root_pos = motion_frames[:, 0:3]
root_rot = motion_frames[:, 3:6]
joint_pos = motion_frames[:, 6:35]


print("Model path:", MODEL_PATH)
print("Motion path:", MOTION_PATH)
print("FPS:", fps)
print("Frames:", len(motion_frames))
print("Joint shape:", joint_pos.shape)


if joint_pos.shape[1] != model.nu:
    raise RuntimeError(
        f"Motion contains {joint_pos.shape[1]} joints, "
        f"but MuJoCo model has {model.nu} actuators."
    )


# --------------------------------------------------
# Renderer
# --------------------------------------------------

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

renderer = mujoco.Renderer(
    model,
    height=480,
    width=640,
)

camera = mujoco.MjvCamera()

mujoco.mjv_defaultCamera(
    camera
)

camera.azimuth = 135
camera.elevation = -12
camera.distance = 3.0


video_frames = []

NUM_CYCLES = 5

cycle_displacement = (
    root_pos[-1, 0]
    - root_pos[0, 0]
)


# --------------------------------------------------
# Replay motion
# --------------------------------------------------

for cycle in range(NUM_CYCLES):

    for i in range(len(motion_frames)):

        # Root position
        data.qpos[0:3] = root_pos[i]

        # Prevent x-position from resetting each cycle
        data.qpos[0] += (
            cycle * cycle_displacement
        )

        # Root orientation
        data.qpos[3:7] = expmap_to_quat(
            root_rot[i]
        )

        # 29 G1 joints
        data.qpos[7:36] = joint_pos[i]

        # Kinematic replay
        data.qvel[:] = 0.0

        mujoco.mj_forward(
            model,
            data,
        )

        # Camera follows robot
        camera.lookat[:] = [
            data.qpos[0],
            data.qpos[1],
            data.qpos[2] * 0.65,
        ]

        renderer.update_scene(
            data,
            camera=camera,
        )

        frame = renderer.render()

        video_frames.append(
            frame.copy()
        )


# --------------------------------------------------
# Save video
# --------------------------------------------------

imageio.mimsave(
    str(VIDEO_PATH),
    video_frames,
    fps=fps,
)

renderer.close()


print()
print(
    f"Reference video saved to: {VIDEO_PATH}"
)