import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class G1Env(gym.Env):
    def __init__(self):
        super().__init__()

        self.model_path = os.path.expanduser(
            "~/Documents/mujoco_menagerie/unitree_g1/scene.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(
            self.model_path
        )
        self.data = mujoco.MjData(self.model)

        # --------------------------------------------------
        # Standing keyframe
        # --------------------------------------------------

        self.stand_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "stand",
        )

        if self.stand_id == -1:
            raise RuntimeError(
                "Could not find G1 'stand' keyframe."
            )

        self.stand_ctrl = (
            self.model.key_ctrl[self.stand_id].copy()
        )

        self.target_height = float(
            self.model.key_qpos[self.stand_id][2]
        )

        # --------------------------------------------------
        # Feet
        # --------------------------------------------------

        self.left_foot_id = self._find_foot_body("left")
        self.right_foot_id = self._find_foot_body("right")

        print(
            "Left foot:",
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.left_foot_id,
            ),
        )

        print(
            "Right foot:",
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.right_foot_id,
            ),
        )

        # --------------------------------------------------
        # Spaces
        # --------------------------------------------------

        self.action_dim = self.model.nu

        self.obs_dim = (
            self.model.nq
            + self.model.nv
            + self.action_dim
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Environment parameters
        # --------------------------------------------------

        self.action_scale = 0.10
        self.frame_skip = 5

        self.max_steps = 1000
        self.current_step = 0

        self.target_velocity = 0.30

        self.previous_action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )

        self.previous_left_contact = True
        self.previous_right_contact = True

        self.left_swing_count = 0
        self.right_swing_count = 0

    # --------------------------------------------------
    # Body lookup
    # --------------------------------------------------

    def _find_foot_body(self, side):
        candidates = []

        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )

            if name is None:
                continue

            name_lower = name.lower()

            if side not in name_lower:
                continue

            if (
                "foot" in name_lower
                or "ankle_roll" in name_lower
                or "ankle" in name_lower
            ):
                candidates.append(
                    (body_id, name_lower)
                )

        if not candidates:
            raise RuntimeError(
                f"Could not find {side} foot body."
            )

        for keyword in [
            "foot",
            "ankle_roll",
            "ankle",
        ]:
            for body_id, name in candidates:
                if keyword in name:
                    return body_id

        return candidates[0][0]

    # --------------------------------------------------
    # Observation
    # --------------------------------------------------

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.previous_action.copy(),
        ]).astype(np.float32)

    # --------------------------------------------------
    # Orientation
    # --------------------------------------------------

    def _rotation_matrix(self):
        quat = self.data.qpos[3:7]

        rotation = np.zeros(9)

        mujoco.mju_quat2Mat(
            rotation,
            quat,
        )

        return rotation.reshape(3, 3)

    def _upright_value(self):
        rotation = self._rotation_matrix()

        return float(
            rotation[2, 2]
        )

    # --------------------------------------------------
    # Feet
    # --------------------------------------------------

    def _foot_positions(self):
        left = self.data.xpos[
            self.left_foot_id
        ].copy()

        right = self.data.xpos[
            self.right_foot_id
        ].copy()

        return left, right

    def _foot_velocities(self):
        left_velocity = np.zeros(6)
        right_velocity = np.zeros(6)

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.left_foot_id,
            left_velocity,
            0,
        )

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.right_foot_id,
            right_velocity,
            0,
        )

        return (
            left_velocity[3:].copy(),
            right_velocity[3:].copy(),
        )

    # --------------------------------------------------
    # Contact detection
    # --------------------------------------------------

    def _foot_contacts(self):
        left_contact = False
        right_contact = False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            if (
                body1 == self.left_foot_id
                or body2 == self.left_foot_id
            ):
                left_contact = True

            if (
                body1 == self.right_foot_id
                or body2 == self.right_foot_id
            ):
                right_contact = True

        return left_contact, right_contact

    # --------------------------------------------------
    # Reward
    # --------------------------------------------------

    def _get_reward(self, action):
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])

        roll_rate = float(self.data.qvel[3])
        pitch_rate = float(self.data.qvel[4])
        yaw_rate = float(self.data.qvel[5])

        height = float(self.data.qpos[2])
        upright = self._upright_value()

        left_pos, right_pos = (
            self._foot_positions()
        )

        left_vel, right_vel = (
            self._foot_velocities()
        )

        left_contact, right_contact = (
            self._foot_contacts()
        )

        # --------------------------------------------------
        # Velocity tracking
        # --------------------------------------------------

        forward_error = (
            vx - self.target_velocity
        )

        track_forward = np.exp(
            -10.0 * forward_error**2
        )

        track_lateral = np.exp(
            -8.0 * vy**2
        )

        velocity_tracking = (
            track_forward * track_lateral
        )

        forward_progress = np.clip(
            vx / self.target_velocity,
            0.0,
            1.0,
        )

        # --------------------------------------------------
        # Balance
        # --------------------------------------------------

        upright_reward = np.clip(
            (upright - 0.70) / 0.30,
            0.0,
            1.0,
        )

        height_reward = np.exp(
            -25.0
            * (
                height
                - self.target_height
            ) ** 2
        )

        orientation_penalty = (
            1.0 - upright
        ) ** 2

        angular_velocity_penalty = (
            roll_rate**2
            + pitch_rate**2
            + 0.5 * yaw_rate**2
        )

        # --------------------------------------------------
        # Alternating gait
        # --------------------------------------------------

        single_support = float(
            left_contact != right_contact
        )

        contact_switch = 0.0

        if (
            left_contact
            != self.previous_left_contact
            or right_contact
            != self.previous_right_contact
        ):
            contact_switch = 1.0

        gait_reward = (
            0.7 * single_support
            + 0.3 * contact_switch
        )

        gait_reward *= forward_progress

        # Count swing events
        if (
            self.previous_left_contact
            and not left_contact
        ):
            self.left_swing_count += 1

        if (
            self.previous_right_contact
            and not right_contact
        ):
            self.right_swing_count += 1

        # Symmetric stepping
        symmetry_error = abs(
            self.left_swing_count
            - self.right_swing_count
        )

        symmetry_reward = np.exp(
            -0.5 * symmetry_error
        )

        # --------------------------------------------------
        # Swing-foot clearance
        # --------------------------------------------------

        target_clearance = 0.08

        clearance_reward = 0.0
        swing_feet = 0

        if not left_contact:
            clearance_reward += np.exp(
                -120.0
                * (
                    left_pos[2]
                    - target_clearance
                ) ** 2
            )
            swing_feet += 1

        if not right_contact:
            clearance_reward += np.exp(
                -120.0
                * (
                    right_pos[2]
                    - target_clearance
                ) ** 2
            )
            swing_feet += 1

        if swing_feet > 0:
            clearance_reward /= swing_feet

        # --------------------------------------------------
        # Foot sliding
        # --------------------------------------------------

        feet_slide = 0.0

        if left_contact:
            feet_slide += (
                left_vel[0] ** 2
                + left_vel[1] ** 2
            )

        if right_contact:
            feet_slide += (
                right_vel[0] ** 2
                + right_vel[1] ** 2
            )

        # --------------------------------------------------
        # Action / joint regularization
        # --------------------------------------------------

        action_rate = np.mean(
            (
                action
                - self.previous_action
            ) ** 2
        )

        action_penalty = np.mean(
            action**2
        )

        joint_velocities = (
            self.data.qvel[6:]
        )

        joint_velocity_penalty = (
            np.mean(
                joint_velocities**2
            )
        )

        # --------------------------------------------------
        # Energy
        # --------------------------------------------------

        actuator_force = (
            self.data.actuator_force
        )

        joint_speed = self.data.qvel[
            6:
            6 + len(actuator_force)
        ]

        if (
            len(joint_speed)
            == len(actuator_force)
        ):
            energy_penalty = np.mean(
                np.abs(
                    actuator_force
                    * joint_speed
                )
            )
        else:
            energy_penalty = 0.0

        # --------------------------------------------------
        # Final reward
        # --------------------------------------------------

        reward = (
            # locomotion
            3.0 * velocity_tracking
            + 1.0 * forward_progress

            # gait
            + 1.0 * gait_reward
            + 0.40 * clearance_reward
            + 0.30 * symmetry_reward

            # balance
            + 1.2 * upright_reward
            + 0.6 * height_reward

            # penalties
            - 2.0 * orientation_penalty
            - 0.20 * angular_velocity_penalty
            - 0.35 * feet_slide
            - 0.05 * action_rate
            - 0.03 * action_penalty
            - 0.015 * joint_velocity_penalty
            - 0.001 * energy_penalty
        )

        self.previous_left_contact = (
            left_contact
        )

        self.previous_right_contact = (
            right_contact
        )

        return float(reward)

    # --------------------------------------------------
    # Termination
    # --------------------------------------------------

    def _is_terminated(self):
        height = float(
            self.data.qpos[2]
        )

        upright = (
            self._upright_value()
        )

        if (
            height
            < self.target_height * 0.78
        ):
            return True

        if upright < 0.65:
            return True

        if not np.isfinite(
            self.data.qpos
        ).all():
            return True

        if not np.isfinite(
            self.data.qvel
        ).all():
            return True

        return False

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self.stand_id,
        )

        self.previous_action[:] = 0.0
        self.current_step = 0

        self.left_swing_count = 0
        self.right_swing_count = 0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        left_contact, right_contact = (
            self._foot_contacts()
        )

        self.previous_left_contact = (
            left_contact
        )

        self.previous_right_contact = (
            right_contact
        )

        return (
            self._get_obs(),
            {},
        )

    # --------------------------------------------------
    # Step
    # --------------------------------------------------

    def step(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        target_ctrl = (
            self.stand_ctrl
            + self.action_scale * action
        )

        ctrl_min = (
            self.model.actuator_ctrlrange[:, 0]
        )

        ctrl_max = (
            self.model.actuator_ctrlrange[:, 1]
        )

        target_ctrl = np.clip(
            target_ctrl,
            ctrl_min,
            ctrl_max,
        )

        self.data.ctrl[:] = target_ctrl

        for _ in range(
            self.frame_skip
        ):
            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.current_step += 1

        reward = self._get_reward(
            action
        )

        terminated = (
            self._is_terminated()
        )

        truncated = (
            self.current_step
            >= self.max_steps
        )

        left_contact, right_contact = (
            self._foot_contacts()
        )

        info = {
            "forward_velocity":
                float(
                    self.data.qvel[0]
                ),

            "lateral_velocity":
                float(
                    self.data.qvel[1]
                ),

            "height":
                float(
                    self.data.qpos[2]
                ),

            "upright":
                self._upright_value(),

            "target_velocity":
                self.target_velocity,

            "left_contact":
                int(left_contact),

            "right_contact":
                int(right_contact),
        }

        self.previous_action = (
            action.copy()
        )

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            info,
        )