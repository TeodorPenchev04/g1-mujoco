from pathlib import Path
import pickle

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MENAGERIE_ROOT = (
    PROJECT_ROOT.parent
    / "mujoco_menagerie"
)

MODEL_PATH = (
    MENAGERIE_ROOT
    / "unitree_g1"
    / "scene.xml"
)

MOTION_PATH = (
    PROJECT_ROOT
    / "motions"
    / "g1_walk.pkl"
)


class G1CombinedEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # ==================================================
        # File checks
        # ==================================================

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Could not find MuJoCo model:\n{MODEL_PATH}"
            )

        if not MOTION_PATH.exists():
            raise FileNotFoundError(
                f"Could not find reference motion:\n{MOTION_PATH}"
            )

        # ==================================================
        # MuJoCo
        # ==================================================

        self.model = mujoco.MjModel.from_xml_path(
            str(MODEL_PATH)
        )

        self.data = mujoco.MjData(
            self.model
        )

        # ==================================================
        # Standing keyframe
        # ==================================================

        self.stand_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "stand",
        )

        if self.stand_id == -1:
            raise RuntimeError(
                "Could not find G1 stand keyframe."
            )

        self.stand_ctrl = (
            self.model
            .key_ctrl[self.stand_id]
            .copy()
        )

        self.target_height = float(
            self.model
            .key_qpos[self.stand_id][2]
        )

        # ==================================================
        # Reference motion
        # ==================================================

        with open(
            MOTION_PATH,
            "rb",
        ) as f:
            motion = pickle.load(f)

        self.motion_fps = float(
            motion["fps"]
        )

        self.motion_frames = np.asarray(
            motion["frames"],
            dtype=np.float64,
        )

        if (
            self.motion_frames.ndim != 2
            or self.motion_frames.shape[1] != 35
        ):
            raise RuntimeError(
                "Expected motion shape (N, 35), "
                f"got {self.motion_frames.shape}"
            )

        self.ref_root_pos = (
            self.motion_frames[:, 0:3]
        )

        self.ref_root_rot = (
            self.motion_frames[:, 3:6]
        )

        self.ref_joint_pos = (
            self.motion_frames[:, 6:35]
        )

        self.num_motion_frames = len(
            self.motion_frames
        )

        self.motion_duration = (
            self.num_motion_frames
            / self.motion_fps
        )

        self.motion_dt = (
            1.0
            / self.motion_fps
        )

        self.ref_joint_vel = (
            self._compute_cyclic_joint_velocities()
        )

        # ==================================================
        # SPEED-MATCHED REFERENCE
        # ==================================================

        # Original reference walk is roughly around 1 m/s,
        # while our locomotion target is 0.30 m/s.
        #
        # We therefore play the reference at 30% speed.

        self.reference_phase_scale = 0.30

        self.effective_motion_duration = (
            self.motion_duration
            / self.reference_phase_scale
        )

        # ==================================================
        # Feet
        # ==================================================

        self.left_foot_id = (
            self._find_foot_body(
                "left"
            )
        )

        self.right_foot_id = (
            self._find_foot_body(
                "right"
            )
        )

        # ==================================================
        # Simulation
        # ==================================================

        self.action_dim = (
            self.model.nu
        )

        self.action_scale = 0.10

        self.frame_skip = 5

        self.control_dt = (
            self.model.opt.timestep
            * self.frame_skip
        )

        self.max_steps = 1000

        self.target_velocity = 0.30

        self.current_step = 0
        self.motion_time = 0.0

        self.previous_action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )

        self.previous_left_contact = True
        self.previous_right_contact = True

        self.left_swing_count = 0
        self.right_swing_count = 0

        # ==================================================
        # Reward weights
        # ==================================================

        self.locomotion_weight = 0.65
        self.imitation_weight = 0.35

        # Approximate maximum positive locomotion reward.
        #
        # Used to put locomotion and imitation onto
        # comparable scales.

        self.locomotion_reward_scale = 7.5

        # ==================================================
        # Observation/action spaces
        # ==================================================

        # qpos
        # qvel
        # previous action
        # sin(phase)
        # cos(phase)

        self.obs_dim = (
            self.model.nq
            + self.model.nv
            + self.action_dim
            + 2
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

        print()
        print(
            "Speed-matched combined environment"
        )

        print(
            "Reference frames:",
            self.num_motion_frames,
        )

        print(
            "Original reference duration:",
            f"{self.motion_duration:.3f} s",
        )

        print(
            "Reference phase scale:",
            self.reference_phase_scale,
        )

        print(
            "Effective reference duration:",
            f"{self.effective_motion_duration:.3f} s",
        )

        print(
            "Observation dimension:",
            self.obs_dim,
        )

        print(
            "Reward:",
            f"{self.locomotion_weight:.2f} locomotion / "
            f"{self.imitation_weight:.2f} imitation",
        )

        print()

    # ======================================================
    # Reference motion
    # ======================================================

    def _compute_cyclic_joint_velocities(
        self,
    ):
        velocities = np.zeros_like(
            self.ref_joint_pos
        )

        for i in range(
            self.num_motion_frames
        ):
            previous_index = (
                i - 1
            ) % self.num_motion_frames

            next_index = (
                i + 1
            ) % self.num_motion_frames

            velocities[i] = (
                self.ref_joint_pos[
                    next_index
                ]
                - self.ref_joint_pos[
                    previous_index
                ]
            ) / (
                2.0
                * self.motion_dt
            )

        return velocities

    def _reference_state(
        self,
    ):
        # motion_time already advances at the
        # speed-matched rate.

        phase = (
            self.motion_time
            / self.motion_duration
        ) % 1.0

        frame_float = (
            phase
            * self.num_motion_frames
        )

        index0 = int(
            np.floor(
                frame_float
            )
        ) % self.num_motion_frames

        index1 = (
            index0 + 1
        ) % self.num_motion_frames

        alpha = (
            frame_float
            - np.floor(
                frame_float
            )
        )

        joint_pos = (
            (1.0 - alpha)
            * self.ref_joint_pos[
                index0
            ]
            + alpha
            * self.ref_joint_pos[
                index1
            ]
        )

        original_joint_vel = (
            (1.0 - alpha)
            * self.ref_joint_vel[
                index0
            ]
            + alpha
            * self.ref_joint_vel[
                index1
            ]
        )

        # CRITICAL:
        #
        # Since the reference is being played at
        # 30% speed, its physically consistent
        # joint velocities must also be 30%.

        joint_vel = (
            original_joint_vel
            * self.reference_phase_scale
        )

        root_rot = (
            (1.0 - alpha)
            * self.ref_root_rot[
                index0
            ]
            + alpha
            * self.ref_root_rot[
                index1
            ]
        )

        return {
            "phase":
                phase,

            "frame":
                index0,

            "joint_pos":
                joint_pos,

            "joint_vel":
                joint_vel,

            "root_rot":
                root_rot,
        }

    # ======================================================
    # Quaternion / orientation utilities
    # ======================================================

    def _expmap_to_quat(
        self,
        expmap,
    ):
        angle = np.linalg.norm(
            expmap
        )

        if angle < 1e-8:
            return np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )

        axis = (
            expmap
            / angle
        )

        half_angle = (
            0.5
            * angle
        )

        quat = np.empty(
            4,
            dtype=np.float64,
        )

        quat[0] = np.cos(
            half_angle
        )

        quat[1:] = (
            axis
            * np.sin(
                half_angle
            )
        )

        return quat

    def _reference_orientation_reward(
        self,
        reference_expmap,
    ):
        reference_quat = (
            self._expmap_to_quat(
                reference_expmap
            )
        )

        current_quat = (
            self.data.qpos[3:7]
        )

        dot = abs(
            np.dot(
                current_quat,
                reference_quat,
            )
        )

        dot = np.clip(
            dot,
            0.0,
            1.0,
        )

        angle = (
            2.0
            * np.arccos(
                dot
            )
        )

        return float(
            np.exp(
                -3.0
                * angle**2
            )
        )

    def _upright_value(self):
        rotation = np.zeros(9)

        mujoco.mju_quat2Mat(
            rotation,
            self.data.qpos[3:7],
        )

        rotation = rotation.reshape(
            3,
            3,
        )

        return float(
            rotation[2, 2]
        )

    # ======================================================
    # Feet
    # ======================================================

    def _find_foot_body(
        self,
        side,
    ):
        candidates = []

        for body_id in range(
            self.model.nbody
        ):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )

            if name is None:
                continue

            name_lower = (
                name.lower()
            )

            if side not in name_lower:
                continue

            if (
                "foot" in name_lower
                or "ankle_roll" in name_lower
                or "ankle" in name_lower
            ):
                candidates.append(
                    (
                        body_id,
                        name_lower,
                    )
                )

        if not candidates:
            raise RuntimeError(
                f"Could not find {side} foot."
            )

        for keyword in [
            "foot",
            "ankle_roll",
            "ankle",
        ]:
            for (
                body_id,
                name,
            ) in candidates:

                if keyword in name:
                    return body_id

        return candidates[0][0]

    def _foot_positions(self):
        return (
            self.data.xpos[
                self.left_foot_id
            ].copy(),

            self.data.xpos[
                self.right_foot_id
            ].copy(),
        )

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

    def _foot_contacts(self):
        left_contact = False
        right_contact = False

        for i in range(
            self.data.ncon
        ):
            contact = (
                self.data.contact[i]
            )

            body1 = (
                self.model.geom_bodyid[
                    contact.geom1
                ]
            )

            body2 = (
                self.model.geom_bodyid[
                    contact.geom2
                ]
            )

            if (
                body1
                == self.left_foot_id
                or body2
                == self.left_foot_id
            ):
                left_contact = True

            if (
                body1
                == self.right_foot_id
                or body2
                == self.right_foot_id
            ):
                right_contact = True

        return (
            left_contact,
            right_contact,
        )

    # ======================================================
    # Observation
    # ======================================================

    def _get_obs(self):
        reference = (
            self._reference_state()
        )

        phase_angle = (
            2.0
            * np.pi
            * reference["phase"]
        )

        phase_features = np.array(
            [
                np.sin(
                    phase_angle
                ),
                np.cos(
                    phase_angle
                ),
            ],
            dtype=np.float32,
        )

        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.previous_action.copy(),
            phase_features,
        ]).astype(
            np.float32
        )

    # ======================================================
    # Locomotion reward
    # ======================================================

    def _locomotion_reward(
        self,
        action,
    ):
        vx = float(
            self.data.qvel[0]
        )

        vy = float(
            self.data.qvel[1]
        )

        roll_rate = float(
            self.data.qvel[3]
        )

        pitch_rate = float(
            self.data.qvel[4]
        )

        yaw_rate = float(
            self.data.qvel[5]
        )

        height = float(
            self.data.qpos[2]
        )

        upright = (
            self._upright_value()
        )

        (
            left_pos,
            right_pos,
        ) = self._foot_positions()

        (
            left_vel,
            right_vel,
        ) = self._foot_velocities()

        (
            left_contact,
            right_contact,
        ) = self._foot_contacts()

        # --------------------------------------------------
        # Velocity tracking
        # --------------------------------------------------

        forward_error = (
            vx
            - self.target_velocity
        )

        track_forward = np.exp(
            -10.0
            * forward_error**2
        )

        track_lateral = np.exp(
            -8.0
            * vy**2
        )

        velocity_tracking = (
            track_forward
            * track_lateral
        )

        forward_progress = np.clip(
            vx
            / self.target_velocity,
            0.0,
            1.0,
        )

        # --------------------------------------------------
        # Balance
        # --------------------------------------------------

        upright_reward = np.clip(
            (
                upright
                - 0.70
            )
            / 0.30,
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
            1.0
            - upright
        ) ** 2

        angular_velocity_penalty = (
            roll_rate**2
            + pitch_rate**2
            + 0.5
            * yaw_rate**2
        )

        # --------------------------------------------------
        # Gait
        # --------------------------------------------------

        single_support = float(
            left_contact
            != right_contact
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
            0.7
            * single_support
            + 0.3
            * contact_switch
        )

        gait_reward *= (
            forward_progress
        )

        # --------------------------------------------------
        # Symmetry
        # --------------------------------------------------

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

        symmetry_error = abs(
            self.left_swing_count
            - self.right_swing_count
        )

        symmetry_reward = np.exp(
            -0.5
            * symmetry_error
        )

        # --------------------------------------------------
        # Foot clearance
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
            clearance_reward /= (
                swing_feet
            )

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
        # Action regularization
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

        joint_velocity_penalty = np.mean(
            self.data.qvel[6:] ** 2
        )

        # --------------------------------------------------
        # Energy
        # --------------------------------------------------

        actuator_force = (
            self.data.actuator_force
        )

        joint_speed = (
            self.data.qvel[
                6:
                6
                + len(
                    actuator_force
                )
            ]
        )

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
        # Locomotion objective
        # --------------------------------------------------

        reward = (
            3.0
            * velocity_tracking

            + 1.0
            * forward_progress

            + 1.0
            * gait_reward

            + 0.40
            * clearance_reward

            + 0.30
            * symmetry_reward

            + 1.2
            * upright_reward

            + 0.6
            * height_reward

            - 2.0
            * orientation_penalty

            - 0.20
            * angular_velocity_penalty

            - 0.35
            * feet_slide

            - 0.05
            * action_rate

            - 0.03
            * action_penalty

            - 0.015
            * joint_velocity_penalty

            - 0.001
            * energy_penalty
        )

        self.previous_left_contact = (
            left_contact
        )

        self.previous_right_contact = (
            right_contact
        )

        return (
            float(reward),
            {
                "velocity_tracking":
                    float(
                        velocity_tracking
                    ),

                "gait_reward":
                    float(
                        gait_reward
                    ),

                "upright_reward":
                    float(
                        upright_reward
                    ),

                "height_reward":
                    float(
                        height_reward
                    ),
            },
        )

    # ======================================================
    # Imitation reward
    # ======================================================

    def _imitation_reward(
        self,
    ):
        reference = (
            self._reference_state()
        )

        current_joint_pos = (
            self.data.qpos[7:36]
        )

        current_joint_vel = (
            self.data.qvel[6:35]
        )

        # --------------------------------------------------
        # Pose
        # --------------------------------------------------

        pose_error = np.mean(
            (
                current_joint_pos
                - reference[
                    "joint_pos"
                ]
            ) ** 2
        )

        pose_reward = np.exp(
            -8.0
            * pose_error
        )

        # --------------------------------------------------
        # Joint velocity
        # --------------------------------------------------

        velocity_error = np.mean(
            (
                current_joint_vel
                - reference[
                    "joint_vel"
                ]
            ) ** 2
        )

        joint_velocity_reward = np.exp(
            -0.05
            * velocity_error
        )

        # --------------------------------------------------
        # Root orientation
        # --------------------------------------------------

        orientation_reward = (
            self._reference_orientation_reward(
                reference[
                    "root_rot"
                ]
            )
        )

        # --------------------------------------------------
        # Imitation objective
        # --------------------------------------------------

        imitation_reward = (
            0.60
            * pose_reward

            + 0.15
            * joint_velocity_reward

            + 0.25
            * orientation_reward
        )

        return (
            float(
                imitation_reward
            ),
            {
                "pose_reward":
                    float(
                        pose_reward
                    ),

                "reference_joint_velocity_reward":
                    float(
                        joint_velocity_reward
                    ),

                "reference_orientation_reward":
                    float(
                        orientation_reward
                    ),

                "reference_frame":
                    int(
                        reference[
                            "frame"
                        ]
                    ),

                "phase":
                    float(
                        reference[
                            "phase"
                        ]
                    ),
            },
        )

    # ======================================================
    # Combined reward
    # ======================================================

    def _get_reward(
        self,
        action,
    ):
        (
            locomotion_reward,
            locomotion_info,
        ) = self._locomotion_reward(
            action
        )

        (
            imitation_reward,
            imitation_info,
        ) = self._imitation_reward()

        normalized_locomotion_reward = (
            locomotion_reward
            / self.locomotion_reward_scale
        )

        total_reward = (
            self.locomotion_weight
            * normalized_locomotion_reward

            + self.imitation_weight
            * imitation_reward
        )

        return (
            float(
                total_reward
            ),
            {
                "locomotion_reward":
                    float(
                        locomotion_reward
                    ),

                "normalized_locomotion_reward":
                    float(
                        normalized_locomotion_reward
                    ),

                "imitation_reward":
                    float(
                        imitation_reward
                    ),

                **locomotion_info,
                **imitation_info,
            },
        )

    # ======================================================
    # Termination
    # ======================================================

    def _is_terminated(self):
        height = float(
            self.data.qpos[2]
        )

        upright = (
            self._upright_value()
        )

        if (
            height
            < self.target_height
            * 0.78
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

    # ======================================================
    # Reset
    # ======================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self.stand_id,
        )

        self.previous_action[:] = 0.0

        self.current_step = 0

        self.motion_time = 0.0

        self.left_swing_count = 0
        self.right_swing_count = 0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        (
            left_contact,
            right_contact,
        ) = self._foot_contacts()

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

    # ======================================================
    # Step
    # ======================================================

    def step(
        self,
        action,
    ):
        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # Same action interpretation as the
        # successful locomotion policy.

        target_ctrl = (
            self.stand_ctrl
            + self.action_scale
            * action
        )

        target_ctrl = np.clip(
            target_ctrl,
            self.model
            .actuator_ctrlrange[:, 0],

            self.model
            .actuator_ctrlrange[:, 1],
        )

        self.data.ctrl[:] = (
            target_ctrl
        )

        for _ in range(
            self.frame_skip
        ):
            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.current_step += 1

        # IMPORTANT:
        #
        # Advance the reference at only 30%
        # of real simulation time.

        self.motion_time += (
            self.control_dt
            * self.reference_phase_scale
        )

        (
            reward,
            reward_info,
        ) = self._get_reward(
            action
        )

        terminated = (
            self._is_terminated()
        )

        truncated = (
            self.current_step
            >= self.max_steps
        )

        info = {
            "forward_velocity":
                float(
                    self.data.qvel[0]
                ),

            "height":
                float(
                    self.data.qpos[2]
                ),

            "upright":
                float(
                    self._upright_value()
                ),

            "target_velocity":
                float(
                    self.target_velocity
                ),

            "reference_phase_scale":
                float(
                    self.reference_phase_scale
                ),

            **reward_info,
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

    def close(self):
        pass