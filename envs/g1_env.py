from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_ROOT = PROJECT_ROOT.parent / "mujoco_menagerie"
MODEL_PATH = MENAGERIE_ROOT / "unitree_g1" / "scene.xml"


class G1Env(gym.Env):
    """
    Curriculum phase 1 for G1 locomotion.

    Based on the TA-provided Isaac Lab reward structure, but simplified
    for learning one stable straight walking behavior in a single
    MuJoCo environment.

    Phase-1 changes from the TA reference:
    - fixed forward command: 0.30 m/s
    - no lateral command
    - no yaw command
    - no observation corruption
    - no random reset yaw
    - action scale reduced from 0.50 to 0.25
    - velocity tracking std reduced from 0.50 to 0.30
    - yaw tracking weight reduced from 2.0 to 0.25

    Reference-motion tracking is not used.
    """

    def __init__(self, observation_noise=False):
        super().__init__()

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Could not find MuJoCo model at:\n{MODEL_PATH}"
            )

        self.model = mujoco.MjModel.from_xml_path(
            str(MODEL_PATH)
        )
        self.data = mujoco.MjData(self.model)

        # ==================================================
        # Default standing configuration
        # ==================================================

        self.stand_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "stand",
        )

        if self.stand_id == -1:
            raise RuntimeError(
                "Could not find G1 'stand' keyframe."
            )

        self.stand_qpos = (
            self.model.key_qpos[self.stand_id].copy()
        )

        self.stand_ctrl = (
            self.model.key_ctrl[self.stand_id].copy()
        )

        self.target_height = float(
            self.stand_qpos[2]
        )

        # ==================================================
        # Actuated joints
        # ==================================================

        self.action_dim = self.model.nu

        self.actuated_joint_ids = np.array(
            [
                int(
                    self.model.actuator_trnid[
                        actuator_id, 0
                    ]
                )
                for actuator_id
                in range(self.model.nu)
            ],
            dtype=np.int32,
        )

        self.actuated_qpos_ids = np.array(
            [
                int(
                    self.model.jnt_qposadr[
                        joint_id
                    ]
                )
                for joint_id
                in self.actuated_joint_ids
            ],
            dtype=np.int32,
        )

        self.actuated_dof_ids = np.array(
            [
                int(
                    self.model.jnt_dofadr[
                        joint_id
                    ]
                )
                for joint_id
                in self.actuated_joint_ids
            ],
            dtype=np.int32,
        )

        self.default_joint_pos = (
            self.stand_qpos[
                self.actuated_qpos_ids
            ].copy()
        )

        # ==================================================
        # Bodies
        # ==================================================

        self.root_body_id = self._find_body(
            [
                "pelvis",
                "pelvis_link",
                "base_link",
            ],
            fallback=1,
        )

        self.torso_id = self._find_body(
            [
                "torso_link",
                "torso",
            ]
        )

        self.left_foot_id = self._find_foot_body(
            "left"
        )

        self.right_foot_id = self._find_foot_body(
            "right"
        )

        print(
            "Left foot:",
            self._body_name(
                self.left_foot_id
            ),
        )

        print(
            "Right foot:",
            self._body_name(
                self.right_foot_id
            ),
        )

        print(
            "Torso:",
            self._body_name(
                self.torso_id
            ),
        )

        # ==================================================
        # TA reward joint groups
        # ==================================================

        self.hip_deviation_joint_ids = (
            self._find_joint_ids(
                [
                    "hip_yaw",
                    "hip_roll",
                ]
            )
        )

        self.arm_deviation_joint_ids = (
            self._find_joint_ids(
                [
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                ]
            )
        )

        self.torso_deviation_joint_ids = (
            self._find_joint_ids(
                [
                    "torso_joint",
                    "waist_yaw",
                    "waist_roll",
                    "waist_pitch",
                ]
            )
        )

        self.finger_deviation_joint_ids = (
            self._find_joint_ids(
                [
                    "five_joint",
                    "three_joint",
                    "six_joint",
                    "four_joint",
                    "zero_joint",
                    "one_joint",
                    "two_joint",
                ]
            )
        )

        self.ankle_joint_ids = (
            self._find_joint_ids(
                [
                    "ankle_pitch",
                    "ankle_roll",
                ]
            )
        )

        self.hip_knee_joint_ids = (
            self._find_joint_ids(
                [
                    "hip_",
                    "knee_joint",
                ]
            )
        )

        self.hip_knee_ankle_joint_ids = (
            self._find_joint_ids(
                [
                    "hip_",
                    "knee_joint",
                    "ankle_",
                ]
            )
        )

        self.hip_knee_dof_ids = (
            self._joint_ids_to_dof_ids(
                self.hip_knee_joint_ids
            )
        )

        self.hip_knee_ankle_dof_ids = (
            self._joint_ids_to_dof_ids(
                self.hip_knee_ankle_joint_ids
            )
        )

        # ==================================================
        # Timing
        # ==================================================

        # Match the reference 50 Hz policy frequency.
        desired_control_dt = 0.020

        self.frame_skip = max(
            1,
            int(
                round(
                    desired_control_dt
                    / float(
                        self.model.opt.timestep
                    )
                )
            ),
        )

        self.control_dt = (
            float(self.model.opt.timestep)
            * self.frame_skip
        )

        # 20-second episode.
        self.max_steps = int(
            round(
                20.0
                / self.control_dt
            )
        )

        self.current_step = 0

        # ==================================================
        # Phase-1 command
        # ==================================================

        self.command = np.array(
            [
                0.30,
                0.00,
                0.00,
            ],
            dtype=np.float64,
        )

        # ==================================================
        # Actions
        # ==================================================

        # TA reference is 0.50.
        # Reduce for curriculum phase 1.
        self.action_scale = 0.25

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # ==================================================
        # Observations
        # ==================================================

        # 3 base linear velocity
        # 3 base angular velocity
        # 3 projected gravity
        # 3 command
        # 29 relative joint positions
        # 29 joint velocities
        # 29 previous actions
        #
        # = 99 for 29-DoF G1
        self.obs_dim = (
            12
            + 3 * self.action_dim
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.observation_noise = bool(
            observation_noise
        )

        # ==================================================
        # Reward parameters
        # ==================================================

        # Narrower than TA's 0.5 so standing still is
        # considerably less rewarding.
        self.velocity_tracking_std = 0.30

        self.yaw_tracking_std = 0.50

        self.air_time_threshold = 0.40

        # Reduced during the straight-walking curriculum.
        self.yaw_tracking_weight = 0.25

        # ==================================================
        # History
        # ==================================================

        self.previous_action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )

        self.previous_target_ctrl = (
            self.stand_ctrl.copy()
        )

        self.left_air_time = 0.0
        self.right_air_time = 0.0

        self.left_contact_time = 0.0
        self.right_contact_time = 0.0

        self.last_reward_terms = {}

    # ==================================================
    # ID helpers
    # ==================================================

    def _body_name(self, body_id):
        return mujoco.mj_id2name(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

    def _find_body(
        self,
        candidates,
        fallback=None,
    ):
        candidates = [
            candidate.lower()
            for candidate in candidates
        ]

        for candidate in candidates:
            for body_id in range(
                self.model.nbody
            ):
                name = self._body_name(
                    body_id
                )

                if (
                    name is not None
                    and name.lower()
                    == candidate
                ):
                    return body_id

        for body_id in range(
            self.model.nbody
        ):
            name = self._body_name(
                body_id
            )

            if name is None:
                continue

            name_lower = name.lower()

            if any(
                candidate in name_lower
                for candidate
                in candidates
            ):
                return body_id

        if fallback is not None:
            return fallback

        raise RuntimeError(
            f"Could not find body: {candidates}"
        )

    def _find_foot_body(
        self,
        side,
    ):
        candidates = []

        for body_id in range(
            self.model.nbody
        ):
            name = self._body_name(
                body_id
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
                    (
                        body_id,
                        name_lower,
                    )
                )

        if not candidates:
            raise RuntimeError(
                f"Could not find {side} foot body."
            )

        for keyword in [
            "ankle_roll",
            "foot",
            "ankle",
        ]:
            for body_id, name in candidates:
                if keyword in name:
                    return body_id

        return candidates[0][0]

    def _find_joint_ids(
        self,
        keywords,
    ):
        keywords = [
            keyword.lower()
            for keyword in keywords
        ]

        ids = []

        for joint_id in range(
            self.model.njnt
        ):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )

            if name is None:
                continue

            name_lower = name.lower()

            if any(
                keyword in name_lower
                for keyword
                in keywords
            ):
                ids.append(
                    joint_id
                )

        return ids

    def _joint_ids_to_dof_ids(
        self,
        joint_ids,
    ):
        return np.array(
            [
                int(
                    self.model.jnt_dofadr[
                        joint_id
                    ]
                )
                for joint_id
                in joint_ids
            ],
            dtype=np.int32,
        )

    # ==================================================
    # Root kinematics
    # ==================================================

    def _rotation_matrix(self):
        matrix = np.zeros(
            9,
            dtype=np.float64,
        )

        mujoco.mju_quat2Mat(
            matrix,
            self.data.qpos[3:7],
        )

        return matrix.reshape(
            3,
            3,
        )

    def _root_world_velocity(self):
        velocity = np.zeros(
            6,
            dtype=np.float64,
        )

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.root_body_id,
            velocity,
            0,
        )

        angular = (
            velocity[:3].copy()
        )

        linear = (
            velocity[3:].copy()
        )

        return (
            linear,
            angular,
        )

    def _root_body_velocity(self):
        linear_world, angular_world = (
            self._root_world_velocity()
        )

        rotation = (
            self._rotation_matrix()
        )

        return (
            rotation.T @ linear_world,
            rotation.T @ angular_world,
        )

    def _yaw_frame_velocity_xy(self):
        linear_world, _ = (
            self._root_world_velocity()
        )

        rotation = (
            self._rotation_matrix()
        )

        yaw = np.arctan2(
            rotation[1, 0],
            rotation[0, 0],
        )

        c = np.cos(yaw)
        s = np.sin(yaw)

        return np.array(
            [
                (
                    c * linear_world[0]
                    + s * linear_world[1]
                ),
                (
                    -s * linear_world[0]
                    + c * linear_world[1]
                ),
            ],
            dtype=np.float64,
        )

    def _projected_gravity(self):
        gravity_world = np.array(
            [
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float64,
        )

        return (
            self._rotation_matrix().T
            @ gravity_world
        )

    def _upright_value(self):
        return float(
            self._rotation_matrix()[
                2,
                2,
            ]
        )

    # ==================================================
    # Observation
    # ==================================================

    def _get_obs(self):
        (
            base_lin_vel,
            base_ang_vel,
        ) = self._root_body_velocity()

        projected_gravity = (
            self._projected_gravity()
        )

        joint_pos_rel = (
            self.data.qpos[
                self.actuated_qpos_ids
            ]
            - self.default_joint_pos
        )

        joint_vel = (
            self.data.qvel[
                self.actuated_dof_ids
            ].copy()
        )

        if self.observation_noise:
            base_lin_vel += (
                self.np_random.uniform(
                    -0.1,
                    0.1,
                    size=3,
                )
            )

            base_ang_vel += (
                self.np_random.uniform(
                    -0.2,
                    0.2,
                    size=3,
                )
            )

            projected_gravity += (
                self.np_random.uniform(
                    -0.05,
                    0.05,
                    size=3,
                )
            )

            joint_pos_rel += (
                self.np_random.uniform(
                    -0.01,
                    0.01,
                    size=self.action_dim,
                )
            )

            joint_vel += (
                self.np_random.uniform(
                    -1.5,
                    1.5,
                    size=self.action_dim,
                )
            )

        observation = np.concatenate(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                self.command.copy(),
                joint_pos_rel,
                joint_vel,
                self.previous_action.copy(),
            ]
        )

        return observation.astype(
            np.float32
        )

    # ==================================================
    # Contacts
    # ==================================================

    def _body_has_world_contact(
        self,
        body_id,
    ):
        for i in range(
            self.data.ncon
        ):
            contact = (
                self.data.contact[i]
            )

            body1 = int(
                self.model.geom_bodyid[
                    contact.geom1
                ]
            )

            body2 = int(
                self.model.geom_bodyid[
                    contact.geom2
                ]
            )

            if (
                body1 == body_id
                and body2 == 0
            ):
                return True

            if (
                body2 == body_id
                and body1 == 0
            ):
                return True

        return False

    def _foot_contacts(self):
        return (
            self._body_has_world_contact(
                self.left_foot_id
            ),
            self._body_has_world_contact(
                self.right_foot_id
            ),
        )

    def _body_linear_velocity(
        self,
        body_id,
    ):
        velocity = np.zeros(
            6,
            dtype=np.float64,
        )

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )

        return velocity[
            3:
        ].copy()

    def _update_contact_timers(
        self,
        left_contact,
        right_contact,
    ):
        if left_contact:
            self.left_contact_time += (
                self.control_dt
            )
            self.left_air_time = 0.0
        else:
            self.left_air_time += (
                self.control_dt
            )
            self.left_contact_time = 0.0

        if right_contact:
            self.right_contact_time += (
                self.control_dt
            )
            self.right_air_time = 0.0
        else:
            self.right_air_time += (
                self.control_dt
            )
            self.right_contact_time = 0.0

    # ==================================================
    # Reward helpers
    # ==================================================

    def _joint_deviation_l1(
        self,
        joint_ids,
    ):
        total = 0.0

        for joint_id in joint_ids:
            qpos_id = int(
                self.model.jnt_qposadr[
                    joint_id
                ]
            )

            total += abs(
                float(
                    self.data.qpos[
                        qpos_id
                    ]
                )
                - float(
                    self.stand_qpos[
                        qpos_id
                    ]
                )
            )

        return float(total)

    def _ankle_soft_limit_penalty(
        self,
    ):
        """
        Approximation of Isaac soft joint limits using
        the inner 90% of the MuJoCo hard range.
        """

        penalty = 0.0

        for joint_id in (
            self.ankle_joint_ids
        ):
            if not bool(
                self.model.jnt_limited[
                    joint_id
                ]
            ):
                continue

            qpos_id = int(
                self.model.jnt_qposadr[
                    joint_id
                ]
            )

            q = float(
                self.data.qpos[
                    qpos_id
                ]
            )

            low, high = (
                self.model.jnt_range[
                    joint_id
                ]
            )

            low = float(low)
            high = float(high)

            center = (
                0.5
                * (low + high)
            )

            half_range = (
                0.5
                * (high - low)
            )

            soft_half_range = (
                0.90
                * half_range
            )

            soft_low = (
                center
                - soft_half_range
            )

            soft_high = (
                center
                + soft_half_range
            )

            if q < soft_low:
                penalty += (
                    soft_low - q
                )

            elif q > soft_high:
                penalty += (
                    q - soft_high
                )

        return float(penalty)

    # ==================================================
    # Reward
    # ==================================================

    def _get_reward(
        self,
        target_ctrl,
    ):
        # --------------------------------------------------
        # Linear velocity tracking
        # --------------------------------------------------

        velocity_xy = (
            self._yaw_frame_velocity_xy()
        )

        linear_error = float(
            np.sum(
                (
                    self.command[:2]
                    - velocity_xy
                ) ** 2
            )
        )

        track_lin_vel_xy = float(
            np.exp(
                -linear_error
                / self.velocity_tracking_std**2
            )
        )

        # --------------------------------------------------
        # Yaw rate tracking
        # --------------------------------------------------

        _, angular_world = (
            self._root_world_velocity()
        )

        yaw_error = float(
            self.command[2]
            - angular_world[2]
        )

        track_ang_vel_z = float(
            np.exp(
                -(yaw_error**2)
                / self.yaw_tracking_std**2
            )
        )

        # --------------------------------------------------
        # Feet
        # --------------------------------------------------

        (
            left_contact,
            right_contact,
        ) = self._foot_contacts()

        self._update_contact_timers(
            left_contact,
            right_contact,
        )

        single_support = (
            left_contact
            != right_contact
        )

        left_mode_time = (
            self.left_contact_time
            if left_contact
            else self.left_air_time
        )

        right_mode_time = (
            self.right_contact_time
            if right_contact
            else self.right_air_time
        )

        if (
            single_support
            and np.linalg.norm(
                self.command[:2]
            ) > 0.1
        ):
            feet_air_time = float(
                min(
                    left_mode_time,
                    right_mode_time,
                    self.air_time_threshold,
                )
            )
        else:
            feet_air_time = 0.0

        left_velocity = (
            self._body_linear_velocity(
                self.left_foot_id
            )
        )

        right_velocity = (
            self._body_linear_velocity(
                self.right_foot_id
            )
        )

        feet_slide = 0.0

        if left_contact:
            feet_slide += float(
                np.linalg.norm(
                    left_velocity[:2]
                )
            )

        if right_contact:
            feet_slide += float(
                np.linalg.norm(
                    right_velocity[:2]
                )
            )

        # --------------------------------------------------
        # Orientation
        # --------------------------------------------------

        projected_gravity = (
            self._projected_gravity()
        )

        flat_orientation_l2 = float(
            np.sum(
                projected_gravity[
                    :2
                ] ** 2
            )
        )

        _, angular_body = (
            self._root_body_velocity()
        )

        ang_vel_xy_l2 = float(
            np.sum(
                angular_body[
                    :2
                ] ** 2
            )
        )

        # --------------------------------------------------
        # Smoothness
        # --------------------------------------------------

        action_rate_l2 = float(
            np.sum(
                (
                    target_ctrl
                    - self.previous_target_ctrl
                ) ** 2
            )
        )

        # --------------------------------------------------
        # Joint acceleration
        # --------------------------------------------------

        if len(
            self.hip_knee_dof_ids
        ) > 0:
            dof_acc_l2 = float(
                np.sum(
                    self.data.qacc[
                        self.hip_knee_dof_ids
                    ] ** 2
                )
            )
        else:
            dof_acc_l2 = 0.0

        # --------------------------------------------------
        # Actuator effort
        # --------------------------------------------------

        if len(
            self.hip_knee_ankle_dof_ids
        ) > 0:
            dof_torques_l2 = float(
                np.sum(
                    self.data.qfrc_actuator[
                        self.hip_knee_ankle_dof_ids
                    ] ** 2
                )
            )
        else:
            dof_torques_l2 = 0.0

        # --------------------------------------------------
        # Joint regularization
        # --------------------------------------------------

        ankle_limits = (
            self._ankle_soft_limit_penalty()
        )

        hip_deviation = (
            self._joint_deviation_l1(
                self.hip_deviation_joint_ids
            )
        )

        arm_deviation = (
            self._joint_deviation_l1(
                self.arm_deviation_joint_ids
            )
        )

        torso_deviation = (
            self._joint_deviation_l1(
                self.torso_deviation_joint_ids
            )
        )

        finger_deviation = (
            self._joint_deviation_l1(
                self.finger_deviation_joint_ids
            )
        )

        # --------------------------------------------------
        # Weighted reward
        #
        # Most weights are the TA G1 configuration.
        # Yaw tracking weight is intentionally reduced only
        # for this straight-walking curriculum.
        # --------------------------------------------------

        weighted_terms = {
            "track_lin_vel_xy":
                1.0
                * track_lin_vel_xy,

            "track_ang_vel_z":
                self.yaw_tracking_weight
                * track_ang_vel_z,

            "feet_air_time":
                0.25
                * feet_air_time,

            "feet_slide":
                -0.10
                * feet_slide,

            "dof_pos_limits":
                -1.0
                * ankle_limits,

            "joint_deviation_hip":
                -0.10
                * hip_deviation,

            "joint_deviation_arms":
                -0.10
                * arm_deviation,

            "joint_deviation_fingers":
                -0.05
                * finger_deviation,

            "joint_deviation_torso":
                -0.10
                * torso_deviation,

            "flat_orientation":
                -1.0
                * flat_orientation_l2,

            "ang_vel_xy":
                -0.05
                * ang_vel_xy_l2,

            "action_rate":
                -0.005
                * action_rate_l2,

            "dof_acc":
                -1.25e-7
                * dof_acc_l2,

            "dof_torques":
                -1.5e-7
                * dof_torques_l2,
        }

        # Match Isaac-style per-step reward scaling.
        reward_terms = {
            name:
                float(
                    value
                    * self.control_dt
                )
            for name, value
            in weighted_terms.items()
        }

        reward = float(
            sum(
                reward_terms.values()
            )
        )

        metrics = {
            "forward_velocity_yaw":
                float(
                    velocity_xy[0]
                ),

            "lateral_velocity_yaw":
                float(
                    velocity_xy[1]
                ),

            "left_contact":
                int(
                    left_contact
                ),

            "right_contact":
                int(
                    right_contact
                ),

            "single_support":
                int(
                    single_support
                ),

            "left_air_time":
                float(
                    self.left_air_time
                ),

            "right_air_time":
                float(
                    self.right_air_time
                ),

            "feet_slide_raw":
                float(
                    feet_slide
                ),
        }

        return (
            reward,
            reward_terms,
            metrics,
        )

    # ==================================================
    # Termination
    # ==================================================

    def _is_terminated(self):
        # Same core G1 termination concept:
        # torso contacting ground.
        if self._body_has_world_contact(
            self.torso_id
        ):
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

    # ==================================================
    # Reset
    # ==================================================

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

        # Curriculum phase 1:
        # always begin from exactly the standing pose.
        self.data.qpos[:] = (
            self.stand_qpos
        )

        self.data.qvel[:] = 0.0

        self.data.ctrl[:] = (
            self.stand_ctrl
        )

        self.current_step = 0

        self.previous_action[:] = 0.0

        self.previous_target_ctrl = (
            self.stand_ctrl.copy()
        )

        self.left_air_time = 0.0
        self.right_air_time = 0.0

        self.left_contact_time = 0.0
        self.right_contact_time = 0.0

        self.last_reward_terms = {}

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return (
            self._get_obs(),
            {
                "command":
                    self.command.copy()
            },
        )

    # ==================================================
    # Step
    # ==================================================

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

        target_ctrl = (
            self.stand_ctrl
            + self.action_scale
            * action
        )

        for actuator_id in range(
            self.model.nu
        ):
            if bool(
                self.model.actuator_ctrllimited[
                    actuator_id
                ]
            ):
                low, high = (
                    self.model.actuator_ctrlrange[
                        actuator_id
                    ]
                )

                target_ctrl[
                    actuator_id
                ] = np.clip(
                    target_ctrl[
                        actuator_id
                    ],
                    low,
                    high,
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

        (
            reward,
            reward_terms,
            metrics,
        ) = self._get_reward(
            target_ctrl
        )

        terminated = (
            self._is_terminated()
        )

        truncated = (
            self.current_step
            >= self.max_steps
        )

        if terminated:
            # TA termination weight -200, with step-dt scaling.
            termination_reward = (
                -200.0
                * self.control_dt
            )

            reward += (
                termination_reward
            )

            reward_terms[
                "termination"
            ] = (
                termination_reward
            )
        else:
            reward_terms[
                "termination"
            ] = 0.0

        linear_world, _ = (
            self._root_world_velocity()
        )

        info = {
            "forward_velocity":
                float(
                    linear_world[0]
                ),

            "lateral_velocity":
                float(
                    linear_world[1]
                ),

            "forward_velocity_yaw":
                metrics[
                    "forward_velocity_yaw"
                ],

            "lateral_velocity_yaw":
                metrics[
                    "lateral_velocity_yaw"
                ],

            "target_velocity":
                float(
                    self.command[0]
                ),

            "target_lateral_velocity":
                float(
                    self.command[1]
                ),

            "target_yaw_rate":
                float(
                    self.command[2]
                ),

            "height":
                float(
                    self.data.qpos[2]
                ),

            "upright":
                self._upright_value(),

            "left_contact":
                metrics[
                    "left_contact"
                ],

            "right_contact":
                metrics[
                    "right_contact"
                ],

            "single_support":
                metrics[
                    "single_support"
                ],

            "left_air_time":
                metrics[
                    "left_air_time"
                ],

            "right_air_time":
                metrics[
                    "right_air_time"
                ],

            "feet_slide_raw":
                metrics[
                    "feet_slide_raw"
                ],

            "reward_terms":
                reward_terms,
        }

        self.previous_action = (
            action.copy()
        )

        self.previous_target_ctrl = (
            target_ctrl.copy()
        )

        self.last_reward_terms = (
            reward_terms
        )

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self):
        pass