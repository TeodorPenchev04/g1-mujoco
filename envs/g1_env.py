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
    MuJoCo Unitree G1 velocity-tracking locomotion environment.

    Reward design is inspired by the Isaac Lab G1 locomotion task:
    - commanded velocity tracking
    - biped single-support timing
    - swing-foot clearance
    - foot-slip suppression
    - upright posture
    - smooth action targets
    - selective posture regularization
    - ankle-limit regularization
    - small effort / acceleration penalties
    """

    def __init__(self):
        super().__init__()

        self.model_path = MODEL_PATH

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Could not find MuJoCo model at:\n{self.model_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(
            str(self.model_path)
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

        self.stand_qpos = (
            self.model.key_qpos[self.stand_id].copy()
        )

        self.target_height = float(
            self.stand_qpos[2]
        )

        # --------------------------------------------------
        # Feet
        # --------------------------------------------------

        self.left_foot_id = self._find_foot_body(
            "left"
        )

        self.right_foot_id = self._find_foot_body(
            "right"
        )

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
        # Joints used for posture regularization
        # --------------------------------------------------

        # Keep these joints relatively close to the standing
        # posture while leaving the primary walking joints
        # freer to generate locomotion.
        self.posture_qpos_indices = (
            self._find_hinge_qpos_indices(
                [
                    "hip_yaw",
                    "hip_roll",
                    "waist",
                    "torso",
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
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

        self.control_dt = (
            self.model.opt.timestep
            * self.frame_skip
        )

        self.max_steps = 1000
        self.current_step = 0

        # --------------------------------------------------
        # Velocity command
        # --------------------------------------------------

        # Keep the command fixed for now.
        #
        # If we later randomize this during training,
        # target velocity should also be added to the
        # observation vector.
        self.target_velocity = 0.30
        self.target_lateral_velocity = 0.0
        self.target_yaw_rate = 0.0

        self.velocity_tracking_std = 0.50
        self.yaw_tracking_std = 0.50

        # --------------------------------------------------
        # Biped gait parameters
        # --------------------------------------------------

        # Same basic threshold used in the Isaac G1
        # reference.
        self.air_time_threshold = 0.40

        # Desired swing-foot height above nominal standing
        # foot height.
        self.target_clearance = 0.08

        self.nominal_left_foot_z = 0.0
        self.nominal_right_foot_z = 0.0

        # --------------------------------------------------
        # Previous actions / contact state
        # --------------------------------------------------

        self.previous_action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )

        self.previous_target_ctrl = (
            self.stand_ctrl.copy()
        )

        self.previous_left_contact = True
        self.previous_right_contact = True

        self.left_air_time = 0.0
        self.right_air_time = 0.0

        self.left_contact_time = 0.0
        self.right_contact_time = 0.0

        self.last_reward_terms = {}

    # --------------------------------------------------
    # Body lookup
    # --------------------------------------------------

    def _find_foot_body(self, side):
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
            "foot",
            "ankle_roll",
            "ankle",
        ]:
            for body_id, name in candidates:

                if keyword in name:
                    return body_id

        return candidates[0][0]

    # --------------------------------------------------
    # Joint lookup
    # --------------------------------------------------

    def _find_joint_ids(
        self,
        keywords,
    ):
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
                for keyword in keywords
            ):
                ids.append(
                    joint_id
                )

        return ids

    def _find_hinge_qpos_indices(
        self,
        keywords,
    ):
        indices = []

        for joint_id in (
            self._find_joint_ids(
                keywords
            )
        ):
            joint_type = int(
                self.model.jnt_type[
                    joint_id
                ]
            )

            if joint_type in (
                int(
                    mujoco.mjtJoint.mjJNT_HINGE
                ),
                int(
                    mujoco.mjtJoint.mjJNT_SLIDE
                ),
            ):
                indices.append(
                    int(
                        self.model.jnt_qposadr[
                            joint_id
                        ]
                    )
                )

        return indices

    # --------------------------------------------------
    # Observation
    # --------------------------------------------------

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                self.previous_action.copy(),
            ]
        ).astype(
            np.float32
        )

    # --------------------------------------------------
    # Orientation
    # --------------------------------------------------

    def _rotation_matrix(self):
        quat = self.data.qpos[
            3:7
        ]

        rotation = np.zeros(
            9
        )

        mujoco.mju_quat2Mat(
            rotation,
            quat,
        )

        return rotation.reshape(
            3,
            3,
        )

    def _upright_value(self):
        return float(
            self._rotation_matrix()[
                2,
                2,
            ]
        )

    # --------------------------------------------------
    # Base velocity
    # --------------------------------------------------

    def _yaw_aligned_xy_velocity(self):
        """
        Express the root XY velocity in a yaw-aligned
        robot frame.

        This makes forward velocity correspond to the
        robot's current heading instead of world +X.
        """

        rotation = (
            self._rotation_matrix()
        )

        yaw = np.arctan2(
            rotation[1, 0],
            rotation[0, 0],
        )

        vx_world = float(
            self.data.qvel[0]
        )

        vy_world = float(
            self.data.qvel[1]
        )

        c = np.cos(yaw)
        s = np.sin(yaw)

        vx_yaw = (
            c * vx_world
            + s * vy_world
        )

        vy_yaw = (
            -s * vx_world
            + c * vy_world
        )

        return (
            float(vx_yaw),
            float(vy_yaw),
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

        return (
            left,
            right,
        )

    def _foot_velocities(self):

        left_velocity = np.zeros(
            6
        )

        right_velocity = np.zeros(
            6
        )

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

        # mj_objectVelocity:
        #
        # [angular velocity, linear velocity]
        return (
            left_velocity[3:].copy(),
            right_velocity[3:].copy(),
        )

    # --------------------------------------------------
    # Contact detection
    # --------------------------------------------------

    def _foot_contacts(self):
        """
        Detect foot contact with static world geometry.

        body_id == 0 corresponds to MuJoCo's world body.

        Restricting support contact this way prevents a
        foot touching another G1 link from being counted
        as valid ground support.
        """

        left_contact = False
        right_contact = False

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
                (
                    body1
                    == self.left_foot_id
                    and body2 == 0
                )
                or (
                    body2
                    == self.left_foot_id
                    and body1 == 0
                )
            ):
                left_contact = True

            if (
                (
                    body1
                    == self.right_foot_id
                    and body2 == 0
                )
                or (
                    body2
                    == self.right_foot_id
                    and body1 == 0
                )
            ):
                right_contact = True

        return (
            left_contact,
            right_contact,
        )

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

    # --------------------------------------------------
    # Reward helpers
    # --------------------------------------------------

    def _selective_posture_penalty(
        self
    ):

        if not self.posture_qpos_indices:
            return 0.0

        idx = np.asarray(
            self.posture_qpos_indices,
            dtype=np.int32,
        )

        deviation = (
            self.data.qpos[idx]
            - self.stand_qpos[idx]
        )

        return float(
            np.mean(
                np.abs(
                    deviation
                )
            )
        )

    def _ankle_soft_limit_penalty(
        self
    ):
        penalties = []

        for joint_id in (
            self.ankle_joint_ids
        ):

            if not bool(
                self.model.jnt_limited[
                    joint_id
                ]
            ):
                continue

            qpos_adr = int(
                self.model.jnt_qposadr[
                    joint_id
                ]
            )

            q = float(
                self.data.qpos[
                    qpos_adr
                ]
            )

            low, high = (
                self.model.jnt_range[
                    joint_id
                ]
            )

            span = float(
                high - low
            )

            if span <= 1e-8:
                continue

            # Outer 10% of the allowed joint range
            # is treated as the soft-limit area.
            soft_low = float(
                low
                + 0.10 * span
            )

            soft_high = float(
                high
                - 0.10 * span
            )

            violation = 0.0

            if q < soft_low:
                violation = (
                    soft_low - q
                ) / span

            elif q > soft_high:
                violation = (
                    q - soft_high
                ) / span

            penalties.append(
                violation**2
            )

        if not penalties:
            return 0.0

        return float(
            np.mean(
                penalties
            )
        )

    # --------------------------------------------------
    # Reward
    # --------------------------------------------------

    def _get_reward(
        self,
        action,
        target_ctrl,
    ):

        vx_yaw, vy_yaw = (
            self._yaw_aligned_xy_velocity()
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

        left_pos, right_pos = (
            self._foot_positions()
        )

        left_vel, right_vel = (
            self._foot_velocities()
        )

        left_contact, right_contact = (
            self._foot_contacts()
        )

        self._update_contact_timers(
            left_contact,
            right_contact,
        )

        # --------------------------------------------------
        # Velocity tracking
        # --------------------------------------------------

        linear_velocity_error = (
            (
                vx_yaw
                - self.target_velocity
            ) ** 2
            +
            (
                vy_yaw
                - self.target_lateral_velocity
            ) ** 2
        )

        velocity_tracking = np.exp(
            -linear_velocity_error
            / self.velocity_tracking_std**2
        )

        yaw_error = (
            yaw_rate
            - self.target_yaw_rate
        )

        yaw_tracking = np.exp(
            -(yaw_error**2)
            / self.yaw_tracking_std**2
        )

        command_active = (
            np.hypot(
                self.target_velocity,
                self.target_lateral_velocity,
            )
            > 0.1
        )

        # --------------------------------------------------
        # Balance
        # --------------------------------------------------

        upright_reward = np.clip(
            (
                upright - 0.70
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
            1.0 - upright
        ) ** 2

        angular_velocity_penalty = (
            roll_rate**2
            + pitch_rate**2
            + 0.25 * yaw_rate**2
        )

        # --------------------------------------------------
        # Biped gait timing
        # --------------------------------------------------

        single_support = (
            left_contact
            != right_contact
        )

        air_time_reward = 0.0

        if (
            single_support
            and command_active
        ):

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

            single_stance_time = min(
                left_mode_time,
                right_mode_time,
                self.air_time_threshold,
            )

            air_time_reward = (
                single_stance_time
                / self.air_time_threshold
            )

        # --------------------------------------------------
        # Swing-foot clearance
        # --------------------------------------------------

        clearance_reward = 0.0
        swing_feet = 0

        if not left_contact:

            left_clearance = (
                left_pos[2]
                - self.nominal_left_foot_z
            )

            clearance_reward += np.exp(
                -120.0
                * (
                    left_clearance
                    - self.target_clearance
                ) ** 2
            )

            swing_feet += 1

        if not right_contact:

            right_clearance = (
                right_pos[2]
                - self.nominal_right_foot_z
            )

            clearance_reward += np.exp(
                -120.0
                * (
                    right_clearance
                    - self.target_clearance
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

            feet_slide += float(
                np.linalg.norm(
                    left_vel[:2]
                )
            )

        if right_contact:

            feet_slide += float(
                np.linalg.norm(
                    right_vel[:2]
                )
            )

        # --------------------------------------------------
        # Smoothness
        # --------------------------------------------------

        # Use the actual actuator targets rather than
        # normalized [-1, 1] policy actions.
        action_rate = float(
            np.mean(
                (
                    target_ctrl
                    - self.previous_target_ctrl
                ) ** 2
            )
        )

        # --------------------------------------------------
        # Posture and joint limits
        # --------------------------------------------------

        posture_penalty = (
            self._selective_posture_penalty()
        )

        ankle_limit_penalty = (
            self._ankle_soft_limit_penalty()
        )

        # --------------------------------------------------
        # Torque / acceleration
        # --------------------------------------------------

        actuator_force = (
            self.data.actuator_force.copy()
        )

        if len(
            actuator_force
        ) > 0:

            torque_l2 = float(
                np.mean(
                    actuator_force**2
                )
            )

        else:
            torque_l2 = 0.0

        joint_acc = (
            self.data.qacc[6:]
        )

        if len(
            joint_acc
        ) > 0:

            joint_acc_l2 = float(
                np.mean(
                    joint_acc**2
                )
            )

        else:
            joint_acc_l2 = 0.0

        # --------------------------------------------------
        # Energy
        # --------------------------------------------------

        joint_speed = self.data.qvel[
            6:
            6 + len(actuator_force)
        ]

        if (
            len(joint_speed)
            == len(actuator_force)
            and len(actuator_force) > 0
        ):

            energy_penalty = float(
                np.mean(
                    np.abs(
                        actuator_force
                        * joint_speed
                    )
                )
            )

        else:
            energy_penalty = 0.0

        # --------------------------------------------------
        # Weighted reward terms
        # --------------------------------------------------

        reward_terms = {

            # Task
            "velocity_tracking":
                3.0
                * float(
                    velocity_tracking
                ),

            "yaw_tracking":
                0.40
                * float(
                    yaw_tracking
                ),

            # Gait
            "air_time":
                0.60
                * float(
                    air_time_reward
                ),

            "clearance":
                0.20
                * float(
                    clearance_reward
                ),

            # Balance
            "upright":
                1.00
                * float(
                    upright_reward
                ),

            "height":
                0.40
                * float(
                    height_reward
                ),

            # Stability penalties
            "orientation":
                -2.00
                * float(
                    orientation_penalty
                ),

            "angular_velocity":
                -0.05
                * float(
                    angular_velocity_penalty
                ),

            # Contact quality
            "foot_slide":
                -0.10
                * float(
                    feet_slide
                ),

            # Smoothness
            "action_rate":
                -0.05
                * float(
                    action_rate
                ),

            # Pose regularization
            "posture":
                -0.10
                * float(
                    posture_penalty
                ),

            # Limits
            "ankle_limits":
                -1.00
                * float(
                    ankle_limit_penalty
                ),

            # Effort
            "torque_l2":
                -1.0e-5
                * float(
                    torque_l2
                ),

            "joint_acc_l2":
                -1.0e-7
                * float(
                    joint_acc_l2
                ),

            "energy":
                -5.0e-4
                * float(
                    energy_penalty
                ),
        }

        reward = float(
            sum(
                reward_terms.values()
            )
        )

        self.previous_left_contact = (
            left_contact
        )

        self.previous_right_contact = (
            right_contact
        )

        self.last_reward_terms = (
            reward_terms
        )

        metrics = {
            "vx_yaw":
                float(
                    vx_yaw
                ),

            "vy_yaw":
                float(
                    vy_yaw
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

            "posture_penalty_raw":
                float(
                    posture_penalty
                ),

            "action_rate_raw":
                float(
                    action_rate
                ),
        }

        return (
            reward,
            reward_terms,
            metrics,
        )

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

        super().reset(
            seed=seed
        )

        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self.stand_id,
        )

        self.previous_action[:] = 0.0

        self.previous_target_ctrl = (
            self.stand_ctrl.copy()
        )

        self.current_step = 0

        self.left_air_time = 0.0
        self.right_air_time = 0.0

        self.left_contact_time = 0.0
        self.right_contact_time = 0.0

        self.last_reward_terms = {}

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # Nominal standing foot heights.
        left_pos, right_pos = (
            self._foot_positions()
        )

        self.nominal_left_foot_z = (
            float(
                left_pos[2]
            )
        )

        self.nominal_right_foot_z = (
            float(
                right_pos[2]
            )
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

        # Position target around standing controller.
        target_ctrl = (
            self.stand_ctrl
            + self.action_scale
            * action
        )

        ctrl_min = (
            self.model.actuator_ctrlrange[
                :,
                0,
            ]
        )

        ctrl_max = (
            self.model.actuator_ctrlrange[
                :,
                1,
            ]
        )

        target_ctrl = np.clip(
            target_ctrl,
            ctrl_min,
            ctrl_max,
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
            action,
            target_ctrl,
        )

        terminated = (
            self._is_terminated()
        )

        truncated = (
            self.current_step
            >= self.max_steps
        )

        # Isaac's configuration uses a large termination
        # penalty, but reward scaling is different there.
        # Use a moderate direct penalty here.
        if terminated:

            termination_penalty = -5.0

            reward += (
                termination_penalty
            )

            reward_terms = dict(
                reward_terms
            )

            reward_terms[
                "termination"
            ] = termination_penalty

        else:

            reward_terms = dict(
                reward_terms
            )

            reward_terms[
                "termination"
            ] = 0.0

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

            "forward_velocity_yaw":
                metrics["vx_yaw"],

            "lateral_velocity_yaw":
                metrics["vy_yaw"],

            "height":
                float(
                    self.data.qpos[2]
                ),

            "upright":
                self._upright_value(),

            "target_velocity":
                self.target_velocity,

            "left_contact":
                int(
                    left_contact
                ),

            "right_contact":
                int(
                    right_contact
                ),

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

            "posture_penalty_raw":
                metrics[
                    "posture_penalty_raw"
                ],

            "action_rate_raw":
                metrics[
                    "action_rate_raw"
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

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )