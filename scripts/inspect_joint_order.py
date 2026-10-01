from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_ROOT = PROJECT_ROOT.parent / "mujoco_menagerie"
MODEL_PATH = MENAGERIE_ROOT / "unitree_g1" / "scene.xml"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Could not find MuJoCo model at:\n{MODEL_PATH}"
    )


model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))


print("=" * 90)
print("G1 ACTUATOR / JOINT ORDER")
print("=" * 90)

print()
print("Number of actuators:", model.nu)
print("Number of joints:", model.njnt)
print()


for actuator_id in range(model.nu):

    actuator_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_id,
    )

    # For the G1 position actuators, the first
    # transmission ID corresponds to the joint ID.
    joint_id = int(
        model.actuator_trnid[actuator_id, 0]
    )

    joint_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id,
    )

    qpos_address = model.jnt_qposadr[joint_id]
    dof_address = model.jnt_dofadr[joint_id]

    print(
        f"Motion index {actuator_id:2d} | "
        f"Joint: {joint_name:35s} | "
        f"Actuator: {actuator_name:35s} | "
        f"qpos[{qpos_address:2d}] | "
        f"qvel[{dof_address:2d}]"
    )