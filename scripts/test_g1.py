import mujoco
import os

MODEL_PATH = os.path.expanduser(
    "~/Documents/mujoco_menagerie/unitree_g1/scene.xml"
)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("G1 loaded")
print("Number of joints:", model.njnt)
print("Number of actuators:", model.nu)