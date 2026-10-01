import pickle
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_ROOT = PROJECT_ROOT.parent / "mujoco_menagerie"
MODEL_PATH = MENAGERIE_ROOT / "unitree_g1" / "scene.xml"
MOTION_PATH = PROJECT_ROOT / "motions" / "g1_walk.pkl"


class G1ImitationEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # --------------------------------------------------
        # Paths
        # --------------------------------------------------

        self.model_path = MODEL_PATH
        self.motion_path = MOTION_PATH

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Could not find MuJoCo model at:\n{self.model_path}"
            )

        if not self.motion_path.is_file():
            raise FileNotFoundError(
                f"Could not find walking motion at:\n{self.motion_path}"
            )

        # --------------------------------------------------
        # MuJoCo
        # --------------------------------------------------

        self.model = mujoco.MjModel.from_xml_path(
            str(self.model_path)
        )

        self.data = mujoco.MjData(self.model)

        # --------------------------------------------------
        # Motion
        # --------------------------------------------------

        with open(self.motion_path, "rb") as f:
            motion = pickle.load(f)

        self.motion_fps = float(
            motion["fps"]
        )

        self.frames = np.asarray(
            motion["frames"],
            dtype=np.float64,
        )

        if self.frames.shape[1] != 35:
            raise RuntimeError(
                f"Expected 35 motion values, got "
                f"{self.frames.shape[1]}"
            )

        self.ref_root_pos = (
            self.frames[:, 0:3].copy()
        )

        self.ref_root_rot = (
            self.frames[:, 3:6].copy()
        )

        self.ref_joint_pos = (
            self.frames[:, 6:35].copy()
        )

        self.num_frames = len(
            self.frames
        )

        self.motion_duration = (
            self.num_frames
            / self.motion_fps
        )

        self.motion_dt = (
            1.0 / self.motion_fps
        )

        self.cycle_displacement = (
            self.ref_root_pos[-1]
            - self.ref_root_pos[0]
        )

        # --------------------------------------------------
        # Precompute reference velocities
        # --------------------------------------------------

        self.ref_joint_vel = (
            self._compute_joint_velocities()
        )

        self.ref_root_lin_vel = (
            self._compute_root_linear_velocities()
        )

        self.ref_root_ang_vel = (
            self._compute_root_angular_velocities()
        )

        print(
            "Reference frames:",
            self.num_frames
        )

        print(
            "Reference FPS:",
            self.motion_fps
        )

        print(
            "Reference duration:",
            f"{self.motion_duration:.3f}s"
        )

        print(
            "Mean reference vx:",
            f"{np.mean(self.ref_root_lin_vel[:, 0]):.3f} m/s"
        )

        # --------------------------------------------------
        # Simulation
        # --------------------------------------------------

        self.frame_skip = 5

        self.control_dt = (
            self.model.opt.timestep
            * self.frame_skip
        )

        self.max_steps = 1000

        # Keep this small at first.
        self.residual_scale = 0.05

        self.current_step = 0
        self.motion_time = 0.0

        self.previous_action = np.zeros(
            self.model.nu,
            dtype=np.float32,
        )

        self.current_action = np.zeros(
            self.model.nu,
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Reset noise
        # --------------------------------------------------

        # Start close to reference initially.
        self.joint_position_noise = 0.005
        self.joint_velocity_noise = 0.02

        # --------------------------------------------------
        # Spaces
        # --------------------------------------------------

        self.action_dim = self.model.nu

        # Observation:
        #
        # joint position error       29
        # joint velocity error       29
        # root linear vel error       3
        # root angular vel error      3
        # orientation error           3
        # previous action            29
        # sin/cos phase               2
        #
        # total = 98

        self.obs_dim = (
            29
            + 29
            + 3
            + 3
            + 3
            + 29
            + 2
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(29,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    # ======================================================
    # Quaternion helpers
    # ======================================================

    def _expmap_to_quat(self, expmap):
        angle = np.linalg.norm(
            expmap
        )

        if angle < 1e-8:
            return np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )

        axis = expmap / angle

        half_angle = (
            0.5 * angle
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
            * np.sin(half_angle)
        )

        return quat

    def _quat_conjugate(self, q):
        return np.array(
            [
                q[0],
                -q[1],
                -q[2],
                -q[3],
            ],
            dtype=np.float64,
        )

    def _quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array(
            [
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ],
            dtype=np.float64,
        )

    def _quat_error_vector(
        self,
        current,
        reference,
    ):
        # q_error = reference * inverse(current)

        q_error = self._quat_multiply(
            reference,
            self._quat_conjugate(
                current
            ),
        )

        if q_error[0] < 0.0:
            q_error = -q_error

        norm = np.linalg.norm(
            q_error
        )

        if norm > 1e-8:
            q_error /= norm

        w = np.clip(
            q_error[0],
            -1.0,
            1.0,
        )

        angle = (
            2.0
            * np.arccos(w)
        )

        sin_half = np.sqrt(
            max(
                1.0 - w*w,
                0.0,
            )
        )

        if sin_half < 1e-8:
            return np.zeros(
                3,
                dtype=np.float64,
            )

        axis = (
            q_error[1:4]
            / sin_half
        )

        return (
            axis * angle
        )

    # ======================================================
    # Reference derivatives
    # ======================================================

    def _compute_joint_velocities(self):
        velocities = np.zeros_like(
            self.ref_joint_pos
        )

        for i in range(
            self.num_frames
        ):
            previous_index = (
                i - 1
            ) % self.num_frames

            next_index = (
                i + 1
            ) % self.num_frames

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

    def _compute_root_linear_velocities(self):
        velocities = np.zeros_like(
            self.ref_root_pos
        )

        for i in range(
            self.num_frames
        ):
            previous_index = (
                i - 1
            ) % self.num_frames

            next_index = (
                i + 1
            ) % self.num_frames

            previous_position = (
                self.ref_root_pos[
                    previous_index
                ].copy()
            )

            next_position = (
                self.ref_root_pos[
                    next_index
                ].copy()
            )

            # Correct wrap-around for looping motion.

            if i == 0:
                previous_position -= (
                    self.cycle_displacement
                )

            if (
                i
                == self.num_frames - 1
            ):
                next_position += (
                    self.cycle_displacement
                )

            velocities[i] = (
                next_position
                - previous_position
            ) / (
                2.0
                * self.motion_dt
            )

        return velocities

    def _compute_root_angular_velocities(self):
        angular_velocities = np.zeros(
            (
                self.num_frames,
                3,
            ),
            dtype=np.float64,
        )

        quaternions = np.array([
            self._expmap_to_quat(
                rot
            )
            for rot in self.ref_root_rot
        ])

        for i in range(
            self.num_frames
        ):
            previous_index = (
                i - 1
            ) % self.num_frames

            next_index = (
                i + 1
            ) % self.num_frames

            q_previous = (
                quaternions[
                    previous_index
                ]
            )

            q_next = (
                quaternions[
                    next_index
                ]
            )

            rotation_vector = (
                self._quat_error_vector(
                    q_previous,
                    q_next,
                )
            )

            angular_velocities[i] = (
                rotation_vector
                / (
                    2.0
                    * self.motion_dt
                )
            )

        return angular_velocities

    # ======================================================
    # Reference state
    # ======================================================

    def _reference_state(
        self,
        motion_time=None,
    ):
        if motion_time is None:
            motion_time = (
                self.motion_time
            )

        phase = (
            motion_time
            / self.motion_duration
        ) % 1.0

        frame_float = (
            phase
            * self.num_frames
        )

        index0 = int(
            np.floor(
                frame_float
            )
        ) % self.num_frames

        index1 = (
            index0 + 1
        ) % self.num_frames

        alpha = (
            frame_float
            - np.floor(
                frame_float
            )
        )

        def interpolate(array):
            return (
                (1.0 - alpha)
                * array[index0]
                + alpha
                * array[index1]
            )

        return {
            "phase":
                phase,

            "frame":
                index0,

            "joint_pos":
                interpolate(
                    self.ref_joint_pos
                ),

            "joint_vel":
                interpolate(
                    self.ref_joint_vel
                ),

            "root_height":
                interpolate(
                    self.ref_root_pos
                )[2],

            "root_rot":
                interpolate(
                    self.ref_root_rot
                ),

            "root_lin_vel":
                interpolate(
                    self.ref_root_lin_vel
                ),

            "root_ang_vel":
                interpolate(
                    self.ref_root_ang_vel
                ),
        }

    # ======================================================
    # Observation
    # ======================================================

    def _get_obs(self):
        reference = (
            self._reference_state()
        )

        # Joint errors

        joint_position_error = (
            self.data.qpos[7:36]
            - reference[
                "joint_pos"
            ]
        )

        joint_velocity_error = (
            self.data.qvel[6:35]
            - reference[
                "joint_vel"
            ]
        )

        # Root linear velocity error

        root_linear_velocity_error = (
            self.data.qvel[0:3]
            - reference[
                "root_lin_vel"
            ]
        )

        # Root angular velocity error

        root_angular_velocity_error = (
            self.data.qvel[3:6]
            - reference[
                "root_ang_vel"
            ]
        )

        # Orientation error

        current_quat = (
            self.data.qpos[3:7]
        )

        reference_quat = (
            self._expmap_to_quat(
                reference[
                    "root_rot"
                ]
            )
        )

        orientation_error = (
            self._quat_error_vector(
                current_quat,
                reference_quat,
            )
        )

        phase_angle = (
            2.0
            * np.pi
            * reference["phase"]
        )

        observation = np.concatenate([
            joint_position_error,
            joint_velocity_error,

            root_linear_velocity_error,
            root_angular_velocity_error,

            orientation_error,

            self.previous_action,

            np.array([
                np.sin(
                    phase_angle
                ),
                np.cos(
                    phase_angle
                ),
            ]),
        ])

        return observation.astype(
            np.float32
        )

    # ======================================================
    # Reward
    # ======================================================

    def _get_reward(self):
        reference = (
            self._reference_state()
        )

        # --------------------------------------------------
        # Joint pose
        # --------------------------------------------------

        pose_error = np.mean(
            (
                self.data.qpos[7:36]
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

        joint_velocity_error = np.mean(
            (
                self.data.qvel[6:35]
                - reference[
                    "joint_vel"
                ]
            ) ** 2
        )

        joint_velocity_reward = np.exp(
            -0.05
            * joint_velocity_error
        )

        # --------------------------------------------------
        # Root orientation
        # --------------------------------------------------

        reference_quat = (
            self._expmap_to_quat(
                reference[
                    "root_rot"
                ]
            )
        )

        orientation_error_vector = (
            self._quat_error_vector(
                self.data.qpos[3:7],
                reference_quat,
            )
        )

        orientation_error = (
            np.dot(
                orientation_error_vector,
                orientation_error_vector,
            )
        )

        orientation_reward = np.exp(
            -3.0
            * orientation_error
        )

        # --------------------------------------------------
        # Root linear velocity
        # --------------------------------------------------

        root_velocity_error = np.mean(
            (
                self.data.qvel[0:3]
                - reference[
                    "root_lin_vel"
                ]
            ) ** 2
        )

        root_velocity_reward = np.exp(
            -2.0
            * root_velocity_error
        )

        # --------------------------------------------------
        # Root angular velocity
        # --------------------------------------------------

        root_angular_velocity_error = np.mean(
            (
                self.data.qvel[3:6]
                - reference[
                    "root_ang_vel"
                ]
            ) ** 2
        )

        root_angular_velocity_reward = np.exp(
            -0.5
            * root_angular_velocity_error
        )

        # --------------------------------------------------
        # Root height
        # --------------------------------------------------

        height_error = (
            self.data.qpos[2]
            - reference[
                "root_height"
            ]
        )

        height_reward = np.exp(
            -40.0
            * height_error**2
        )

        # --------------------------------------------------
        # Action regularization
        # --------------------------------------------------

        action_rate = np.mean(
            (
                self.current_action
                - self.previous_action
            ) ** 2
        )

        action_penalty = np.mean(
            self.current_action**2
        )

        # --------------------------------------------------
        # Final reward
        # --------------------------------------------------

        reward = (
            0.35 * pose_reward
            + 0.15 * joint_velocity_reward

            + 0.20 * orientation_reward
            + 0.15 * root_velocity_reward
            + 0.10 * root_angular_velocity_reward

            + 0.05 * height_reward

            - 0.003 * action_rate
            - 0.001 * action_penalty
        )

        info = {
            "pose_reward":
                float(
                    pose_reward
                ),

            "joint_velocity_reward":
                float(
                    joint_velocity_reward
                ),

            "orientation_reward":
                float(
                    orientation_reward
                ),

            "root_velocity_reward":
                float(
                    root_velocity_reward
                ),

            "root_angular_velocity_reward":
                float(
                    root_angular_velocity_reward
                ),

            "height_reward":
                float(
                    height_reward
                ),

            "forward_velocity":
                float(
                    self.data.qvel[0]
                ),

            "reference_velocity":
                float(
                    reference[
                        "root_lin_vel"
                    ][0]
                ),

            "height":
                float(
                    self.data.qpos[2]
                ),

            "phase":
                float(
                    reference["phase"]
                ),

            "reference_frame":
                int(
                    reference["frame"]
                ),
        }

        return (
            float(reward),
            info,
        )

    # ======================================================
    # Termination
    # ======================================================

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

    def _is_terminated(self):
        if (
            self.data.qpos[2]
            < 0.50
        ):
            return True

        if (
            self._upright_value()
            < 0.45
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

    # ======================================================
    # Reference-state initialization
    # ======================================================

    def _set_reference_state(
        self,
        frame_index,
    ):
        self.motion_time = (
            frame_index
            / self.motion_fps
        )

        reference = (
            self._reference_state()
        )

        # World position starts at origin.
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0

        self.data.qpos[2] = (
            reference[
                "root_height"
            ]
        )

        self.data.qpos[3:7] = (
            self._expmap_to_quat(
                reference[
                    "root_rot"
                ]
            )
        )

        self.data.qpos[7:36] = (
            reference[
                "joint_pos"
            ]
        )

        # IMPORTANT:
        # Initialize root velocities from reference.

        self.data.qvel[0:3] = (
            reference[
                "root_lin_vel"
            ]
        )

        self.data.qvel[3:6] = (
            reference[
                "root_ang_vel"
            ]
        )

        self.data.qvel[6:35] = (
            reference[
                "joint_vel"
            ]
        )

        self.data.ctrl[:] = np.clip(
            reference[
                "joint_pos"
            ],
            self.model.actuator_ctrlrange[
                :,
                0,
            ],
            self.model.actuator_ctrlrange[
                :,
                1,
            ],
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

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

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        self.current_step = 0

        self.previous_action[:] = 0.0
        self.current_action[:] = 0.0

        # --------------------------------------------------
        # RANDOM REFERENCE FRAME
        # --------------------------------------------------

        if (
            options is not None
            and "reference_frame"
            in options
        ):
            frame_index = int(
                options[
                    "reference_frame"
                ]
            ) % self.num_frames

        else:
            frame_index = int(
                self.np_random.integers(
                    0,
                    self.num_frames,
                )
            )

        self._set_reference_state(
            frame_index
        )

        # --------------------------------------------------
        # SMALL PERTURBATION
        # --------------------------------------------------

        self.data.qpos[7:36] += (
            self.np_random.normal(
                0.0,
                self.joint_position_noise,
                size=29,
            )
        )

        self.data.qvel[6:35] += (
            self.np_random.normal(
                0.0,
                self.joint_velocity_noise,
                size=29,
            )
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return (
            self._get_obs(),
            {
                "start_reference_frame":
                    frame_index
            },
        )

    # ======================================================
    # Step
    # ======================================================

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

        self.current_action = (
            action.copy()
        )

        # Reference pose at next control step.

        next_reference = (
            self._reference_state(
                self.motion_time
                + self.control_dt
            )
        )

        residual = (
            self.residual_scale
            * action
        )

        target_ctrl = (
            next_reference[
                "joint_pos"
            ]
            + residual
        )

        target_ctrl = np.clip(
            target_ctrl,
            self.model.actuator_ctrlrange[
                :,
                0,
            ],
            self.model.actuator_ctrlrange[
                :,
                1,
            ],
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

        self.motion_time += (
            self.control_dt
        )

        reward, info = (
            self._get_reward()
        )

        terminated = (
            self._is_terminated()
        )

        truncated = (
            self.current_step
            >= self.max_steps
        )

        observation = (
            self._get_obs()
        )

        self.previous_action = (
            action.copy()
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        pass