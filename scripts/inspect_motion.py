import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOTION_PATH = PROJECT_ROOT / "motions" / "g1_walk.pkl"

if not MOTION_PATH.is_file():
    raise FileNotFoundError(
        f"Could not find walking motion at:\n{MOTION_PATH}"
    )


with open(MOTION_PATH, "rb") as f:
    motion = pickle.load(f)


print("Motion type:", type(motion))
print("Keys:", motion.keys())
print()

print("Loop mode:", motion["loop_mode"])
print("FPS:", motion["fps"])

frames = np.asarray(motion["frames"])

print("Frames shape:", frames.shape)
print("Frames dtype:", frames.dtype)
print()

num_frames = frames.shape[0]
values_per_frame = frames.shape[1]

duration = num_frames / motion["fps"]

print("Number of frames:", num_frames)
print("Values per frame:", values_per_frame)
print(f"Duration: {duration:.3f} seconds")
print()


# Assumed MimicKit G1 layout:
#
# [0:3]   root position
# [3:6]   root rotation exponential map
# [6:35]  29 joint positions

root_pos = frames[:, 0:3]
root_rot = frames[:, 3:6]
joint_pos = frames[:, 6:35]


print("Root position shape:", root_pos.shape)
print("Root rotation shape:", root_rot.shape)
print("Joint position shape:", joint_pos.shape)
print()


print("First root position:")
print(root_pos[0])
print()

print("First root rotation:")
print(root_rot[0])
print()

print("First 29 joint positions:")
print(joint_pos[0])
print()


print("Joint ranges:")
for i in range(joint_pos.shape[1]):
    print(
        f"Joint {i:2d}: "
        f"min={joint_pos[:, i].min(): .4f}, "
        f"max={joint_pos[:, i].max(): .4f}"
    )