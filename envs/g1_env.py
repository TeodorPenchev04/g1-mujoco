from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_ROOT = PROJECT_ROOT.parent / "mujoco_menagerie"
MODEL_PATH = MENAGERIE_ROOT / "unitree_g1" / "scene.xml"


class G1Env(gym.Env):
    """MuJoCo port of the TA-provided Isaac Lab G1 velocity task.

    The goal of this environment is to make the comparison as clean as possible:
    use the TA reference reward structure, command structure, observations and
    position-target action interpretation without using reference-motion tracking.

    Isaac-specific pieces that do not exist in this flat MuJoCo setup (terrain
    height scanner, thousands of parallel environments, PhysX randomization) are
    intentionally omitted.
    """

    def __init__(self, observation_noise=True):
        super().__init__()

        self.model_path = MODEL_PATH
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Could not find MuJoCo model at:\n{self.model_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        # --------------------------------------------------
        # Standing/default pose
        # --------------------------------------------------

        self.stand_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "stand",
        )
        if self.stand_id == -1:
            raise RuntimeError("Could not find G1 'stand' keyframe.")

        self.stand_qpos = self.model.key_qpos[self.stand_id].copy()
        self.stand_ctrl = self.model.key_ctrl[self.stand_id].copy()
        self.target_height = float(self.stand_qpos[2])

        # --------------------------------------------------
        # Important bodies
        # --------------------------------------------------

        self.left_foot_id = self._find_foot_body("left")
        self.right_foot_id = self._find_foot_body("right")
        self.torso_id = self._find_body(["torso_link", "torso"])

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
        print(
            "Torso:",
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.torso_id,
            ),
        )

        # --------------------------------------------------
        # Joint groups corresponding to the TA reference
        # --------------------------------------------------

        self.hip_deviation_joint_ids = self._find_joint_ids(
            ["hip_yaw_joint", "hip_roll_joint"]
        )
        self.arm_deviation_joint_ids = self._find_joint_ids(
            [
                "shoulder_pitch_joint",
                "shoulder_roll_joint",
                "shoulder_yaw_joint",
                "elbow_pitch_joint",
                "elbow_roll_joint",
            ]
        )

        # Isaac's G1 config calls this the torso joint.  The Menagerie G1 uses
        # waist joints, so these are the nearest model-equivalent joints.
        self.torso_deviation_joint_ids = self._find_joint_ids(
            ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "torso_joint"]
        )

        self.ankle_joint_ids = self._find_joint_ids(
            ["ankle_pitch_joint", "ankle_roll_joint"]
        )
        self.hip_knee_joint_ids = self._find_joint_ids(
            ["hip_", "knee_joint"]
        )
        self.hip_knee_ankle_joint_ids = self._find_joint_ids(
            ["hip_", "knee_joint", "ankle_"]
        )

        self.hip_knee_dof_ids = self._joint_ids_to_dof_ids(
            self.hip_knee_joint_ids
        )
        self.effort_actuator_ids = self._joint_ids_to_actuator_ids(
            self.hip_knee_ankle_joint_ids
        )

        # --------------------------------------------------
        # Simulation/control timing
        # --------------------------------------------------

        # TA reference: sim dt = 0.005, decimation = 4 -> 0.02 s policy dt.
        # Preserve the 50 Hz policy rate even though the Menagerie XML uses a
        # different MuJoCo physics timestep.
        target_control_dt = 0.02
        self.frame_skip = max(
            1,
            int(round(target_control_dt / float(self.model.opt.timestep))),
        )
        self.control_dt = float(self.model.opt.timestep) * self.frame_skip

        # TA reference episode length is 20 s.
        self.max_steps = int(round(20.0 / self.control_dt))
        self.current_step = 0

        # --------------------------------------------------
        # Command generator
        # --------------------------------------------------

        # G1 rough config:
        #   lin_vel_x in [0, 1]
        #   lin_vel_y = 0
        #   ang_vel_z in [-1, 1]
        # Parent command config resamples every 10 s, uses heading mode and
        # heading-control stiffness 0.5.
        self.command_resample_seconds = 10.0
        self.command_resample_steps = max(
            1,
            int(round(self.command_resample_seconds / self.control_dt)),
        )
        self.heading_control_stiffness = 0.5

        self.command = np.zeros(3, dtype=np.float64)
        self.heading_target = 0.0
        self.command_override = False

        # --------------------------------------------------
        # Action space
        # --------------------------------------------------

        self.action_dim = self.model.nu

        # TA reference JointPositionActionCfg:
        # scale=0.5 and use_default_offset=True.
        self.action_scale = 0.5

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Observation space
        # --------------------------------------------------

        # Flat-ground equivalent of the TA policy observations:
        #   base linear velocity      3
        #   base angular velocity     3
        #   projected gravity         3
        #   velocity command          3
        #   relative joint position  29
        #   joint velocity           29
        #   previous action          29
        # = 99 values
        #
        # The Isaac rough-terrain height scan is intentionally omitted because
        # this MuJoCo scene is currently flat and has no height scanner.
        self.obs_dim = 12 + 3 * self.action_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.observation_noise = bool(observation_noise)

        # --------------------------------------------------
        # Reward parameters/state
        # --------------------------------------------------

        self.velocity_tracking_std = 0.5
        self.yaw_tracking_std = 0.5
        self.air_time_threshold = 0.4

        self.previous_action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )
        self.previous_target_ctrl = self.stand_ctrl.copy()

        self.left_air_time = 0.0
        self.right_air_time = 0.0
        self.left_contact_time = 0.0
        self.right_contact_time = 0.0

        self.last_reward_terms = {}

    # ==================================================
    # Lookup utilities
    # ==================================================

    def _find_body(self, candidates):
        lowered = [candidate.lower() for candidate in candidates]

        for exact in lowered:
            for body_id in range(self.model.nbody):
                name = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                )
                if name is not None and name.lower() == exact:
                    return body_id

        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if name is None:
                continue
            name_lower = name.lower()
            if any(candidate in name_lower for candidate in lowered):
                return body_id

        raise RuntimeError(
            f"Could not find body matching any of: {candidates}"
        )

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
                candidates.append((body_id, name_lower))

        if not candidates:
            raise RuntimeError(f"Could not find {side} foot body.")

        for keyword in ["foot", "ankle_roll", "ankle"]:
            for body_id, name in candidates:
                if keyword in name:
                    return body_id

        return candidates[0][0]

    def _find_joint_ids(self, keywords):
        ids = []
        keywords = [keyword.lower() for keyword in keywords]

        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            if name is None:
                continue

            name_lower = name.lower()
            if any(keyword in name_lower for keyword in keywords):
                ids.append(joint_id)

        return ids

    def _joint_ids_to_dof_ids(self, joint_ids):
        dof_ids = []
        for joint_id in joint_ids:
            dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
        return dof_ids

    def _joint_ids_to_actuator_ids(self, joint_ids):
        joint_set = set(int(joint_id) for joint_id in joint_ids)
        actuator_ids = []

        for actuator_id in range(self.model.nu):
            transmitted_joint_id = int(
                self.model.actuator_trnid[actuator_id, 0]
            )
            if transmitted_joint_id in joint_set:
                actuator_ids.append(actuator_id)

        return actuator_ids

    # ==================================================
    # Orientation/velocity utilities
    # ==================================================

    def _rotation_matrix(self):
        rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        return rotation.reshape(3, 3)

    def _yaw(self):
        rotation = self._rotation_matrix()
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    def _base_velocities_body_frame(self):
        rotation = self._rotation_matrix()
        linear_world = np.asarray(self.data.qvel[0:3], dtype=np.float64)
        angular_world = np.asarray(self.data.qvel[3:6], dtype=np.float64)
        return rotation.T @ linear_world, rotation.T @ angular_world

    def _yaw_aligned_xy_velocity(self):
        yaw = self._yaw()
        c = np.cos(yaw)
        s = np.sin(yaw)

        vx_world = float(self.data.qvel[0])
        vy_world = float(self.data.qvel[1])

        return np.array(
            [
                c * vx_world + s * vy_world,
                -s * vx_world + c * vy_world,
            ],
            dtype=np.float64,
        )

    def _projected_gravity(self):
        rotation = self._rotation_matrix()
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return rotation.T @ gravity_world

    # ==================================================
    # Commands
    # ==================================================

    def _sample_command(self):
        # Parent reference uses rel_standing_envs=0.02.
        if self.np_random.random() < 0.02:
            self.command[:] = 0.0
            self.heading_target = self._yaw()
            return

        self.command[0] = self.np_random.uniform(0.0, 1.0)
        self.command[1] = 0.0
        self.heading_target = self.np_random.uniform(-np.pi, np.pi)
        self._update_heading_command()

    def _update_heading_command(self):
        if self.command_override:
            return

        heading_error = self._wrap_angle(
            self.heading_target - self._yaw()
        )
        self.command[2] = np.clip(
            self.heading_control_stiffness * heading_error,
            -1.0,
            1.0,
        )

    # ==================================================
    # Observations
    # ==================================================

    def _get_obs(self):
        base_lin_vel, base_ang_vel = self._base_velocities_body_frame()
        projected_gravity = self._projected_gravity()

        joint_pos_rel = self.data.qpos[7:7 + self.action_dim] - self.stand_qpos[
            7:7 + self.action_dim
        ]
        joint_vel = self.data.qvel[6:6 + self.action_dim].copy()

        if self.observation_noise:
            base_lin_vel = base_lin_vel + self.np_random.uniform(
                -0.1, 0.1, size=3
            )
            base_ang_vel = base_ang_vel + self.np_random.uniform(
                -0.2, 0.2, size=3
            )
            projected_gravity = projected_gravity + self.np_random.uniform(
                -0.05, 0.05, size=3
            )
            joint_pos_rel = joint_pos_rel + self.np_random.uniform(
                -0.01, 0.01, size=self.action_dim
            )
            joint_vel = joint_vel + self.np_random.uniform(
                -1.5, 1.5, size=self.action_dim
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

        return observation.astype(np.float32)

    # ==================================================
    # Contacts and foot kinematics
    # ==================================================

    def _body_has_world_contact(self, body_id):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])

            if (
                (body1 == body_id and body2 == 0)
                or (body2 == body_id and body1 == 0)
            ):
                return True

        return False

    def _foot_contacts(self):
        return (
            self._body_has_world_contact(self.left_foot_id),
            self._body_has_world_contact(self.right_foot_id),
        )

    def _foot_velocities(self):
        left_velocity = np.zeros(6, dtype=np.float64)
        right_velocity = np.zeros(6, dtype=np.float64)

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

        return left_velocity[3:].copy(), right_velocity[3:].copy()

    def _update_contact_timers(self, left_contact, right_contact):
        if left_contact:
            self.left_contact_time += self.control_dt
            self.left_air_time = 0.0
        else:
            self.left_air_time += self.control_dt
            self.left_contact_time = 0.0

        if right_contact:
            self.right_contact_time += self.control_dt
            self.right_air_time = 0.0
        else:
            self.right_air_time += self.control_dt
            self.right_contact_time = 0.0

    # ==================================================
    # Reward helpers
    # ==================================================

    def _joint_deviation_l1(self, joint_ids):
        total = 0.0
        for joint_id in joint_ids:
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            total += abs(
                float(self.data.qpos[qpos_adr])
                - float(self.stand_qpos[qpos_adr])
            )
        return total

    def _ankle_pos_limit_penalty(self):
        # Isaac's joint_pos_limits uses the articulation's soft joint limits.
        # MuJoCo exposes the hard joint range, so use the inner 90% as the
        # closest simple analogue to a soft limit.
        penalty = 0.0

        for joint_id in self.ankle_joint_ids:
            if not bool(self.model.jnt_limited[joint_id]):
                continue

            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            q = float(self.data.qpos[qpos_adr])
            low, high = self.model.jnt_range[joint_id]
            low = float(low)
            high = float(high)

            center = 0.5 * (low + high)
            half_range = 0.5 * (high - low)
            soft_half_range = 0.9 * half_range
            soft_low = center - soft_half_range
            soft_high = center + soft_half_range

            if q < soft_low:
                penalty += soft_low - q
            elif q > soft_high:
                penalty += q - soft_high

        return penalty

    # ==================================================
    # Reward
    # ==================================================

    def _get_reward(self, target_ctrl):
        left_contact, right_contact = self._foot_contacts()
        left_vel, right_vel = self._foot_velocities()

        self._update_contact_timers(left_contact, right_contact)

        # --------------------------------------------------
        # Velocity command tracking
        # --------------------------------------------------

        vel_yaw_xy = self._yaw_aligned_xy_velocity()
        lin_vel_error = float(
            np.sum((self.command[:2] - vel_yaw_xy) ** 2)
        )
        track_lin_vel_xy = float(
            np.exp(-lin_vel_error / self.velocity_tracking_std**2)
        )

        yaw_rate_error = float(self.command[2] - self.data.qvel[5])
        track_ang_vel_z = float(
            np.exp(-(yaw_rate_error**2) / self.yaw_tracking_std**2)
        )

        # --------------------------------------------------
        # Positive biped air-time reward
        # --------------------------------------------------

        in_contact = np.array(
            [left_contact, right_contact],
            dtype=bool,
        )
        in_mode_time = np.array(
            [
                self.left_contact_time if left_contact else self.left_air_time,
                self.right_contact_time if right_contact else self.right_air_time,
            ],
            dtype=np.float64,
        )

        if (
            np.sum(in_contact) == 1
            and np.linalg.norm(self.command[:2]) > 0.1
        ):
            feet_air_time = float(
                min(np.min(in_mode_time), self.air_time_threshold)
            )
        else:
            feet_air_time = 0.0

        # --------------------------------------------------
        # Foot sliding
        # --------------------------------------------------

        feet_slide = 0.0
        if left_contact:
            feet_slide += float(np.linalg.norm(left_vel[:2]))
        if right_contact:
            feet_slide += float(np.linalg.norm(right_vel[:2]))

        # --------------------------------------------------
        # Generic inherited penalties kept by G1RoughEnvCfg
        # --------------------------------------------------

        projected_gravity = self._projected_gravity()
        flat_orientation_l2 = float(
            projected_gravity[0] ** 2 + projected_gravity[1] ** 2
        )

        ang_vel_xy_l2 = float(
            self.data.qvel[3] ** 2 + self.data.qvel[4] ** 2
        )

        action_rate_l2 = float(
            np.sum((target_ctrl - self.previous_target_ctrl) ** 2)
        )

        if self.hip_knee_dof_ids:
            dof_acc_l2 = float(
                np.sum(self.data.qacc[self.hip_knee_dof_ids] ** 2)
            )
        else:
            dof_acc_l2 = 0.0

        if self.effort_actuator_ids:
            dof_torques_l2 = float(
                np.sum(self.data.actuator_force[self.effort_actuator_ids] ** 2)
            )
        else:
            dof_torques_l2 = 0.0

        dof_pos_limits = self._ankle_pos_limit_penalty()
        joint_deviation_hip = self._joint_deviation_l1(
            self.hip_deviation_joint_ids
        )
        joint_deviation_arms = self._joint_deviation_l1(
            self.arm_deviation_joint_ids
        )
        joint_deviation_torso = self._joint_deviation_l1(
            self.torso_deviation_joint_ids
        )

        # --------------------------------------------------
        # TA-reference weights
        # --------------------------------------------------

        reward_terms = {
            "track_lin_vel_xy": 1.0 * track_lin_vel_xy,
            "track_ang_vel_z": 2.0 * track_ang_vel_z,
            "feet_air_time": 0.25 * feet_air_time,
            "feet_slide": -0.1 * feet_slide,
            "dof_pos_limits": -1.0 * dof_pos_limits,
            "joint_deviation_hip": -0.1 * joint_deviation_hip,
            "joint_deviation_arms": -0.1 * joint_deviation_arms,
            "joint_deviation_torso": -0.1 * joint_deviation_torso,
            "flat_orientation": -1.0 * flat_orientation_l2,
            "ang_vel_xy": -0.05 * ang_vel_xy_l2,
            "action_rate": -0.005 * action_rate_l2,
            "dof_acc": -1.25e-7 * dof_acc_l2,
            "dof_torques": -1.5e-7 * dof_torques_l2,
        }

        reward = float(sum(reward_terms.values()))
        self.last_reward_terms = reward_terms

        metrics = {
            "forward_velocity_yaw": float(vel_yaw_xy[0]),
            "lateral_velocity_yaw": float(vel_yaw_xy[1]),
            "left_contact": int(left_contact),
            "right_contact": int(right_contact),
            "single_support": int(left_contact != right_contact),
            "left_air_time": float(self.left_air_time),
            "right_air_time": float(self.right_air_time),
            "feet_slide_raw": float(feet_slide),
        }

        return reward, reward_terms, metrics

    # ==================================================
    # Termination
    # ==================================================

    def _is_terminated(self):
        # TA G1 rough config terminates when torso_link contacts the ground.
        if self._body_has_world_contact(self.torso_id):
            return True

        if not np.isfinite(self.data.qpos).all():
            return True
        if not np.isfinite(self.data.qvel).all():
            return True

        return False

    # ==================================================
    # Reset
    # ==================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self.stand_id,
        )

        # TA reset: random x/y/yaw, zero root velocity, joints exactly at
        # default pose.
        self.data.qpos[0] = self.stand_qpos[0] + self.np_random.uniform(
            -0.5, 0.5
        )
        self.data.qpos[1] = self.stand_qpos[1] + self.np_random.uniform(
            -0.5, 0.5
        )

        yaw_delta = self.np_random.uniform(-np.pi, np.pi)
        yaw_quat = np.array(
            [np.cos(0.5 * yaw_delta), 0.0, 0.0, np.sin(0.5 * yaw_delta)],
            dtype=np.float64,
        )
        self.data.qpos[3:7] = self._quat_multiply(
            yaw_quat,
            self.stand_qpos[3:7],
        )
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.stand_ctrl

        self.current_step = 0
        self.previous_action[:] = 0.0
        self.previous_target_ctrl = self.stand_ctrl.copy()

        self.left_air_time = 0.0
        self.right_air_time = 0.0
        self.left_contact_time = 0.0
        self.right_contact_time = 0.0
        self.last_reward_terms = {}

        mujoco.mj_forward(self.model, self.data)

        if options is not None and "command" in options:
            command = np.asarray(options["command"], dtype=np.float64)
            if command.shape != (3,):
                raise ValueError("options['command'] must have shape (3,).")
            self.command[:] = command
            self.command_override = True
        else:
            self.command_override = False
            self._sample_command()

        return self._get_obs(), {
            "command": self.command.copy(),
        }

    # ==================================================
    # Step
    # ==================================================

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # TA reference: desired joint positions are default pose + 0.5 * action.
        target_ctrl = self.stand_ctrl + self.action_scale * action
        target_ctrl = np.clip(
            target_ctrl,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

        self.data.ctrl[:] = target_ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # Heading-mode yaw command is continuously updated from heading error.
        self._update_heading_command()

        reward, reward_terms, metrics = self._get_reward(target_ctrl)

        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        # TA reference termination penalty.
        if terminated:
            reward_terms = dict(reward_terms)
            reward_terms["termination"] = -200.0
            reward -= 200.0
        else:
            reward_terms = dict(reward_terms)
            reward_terms["termination"] = 0.0

        self.last_reward_terms = reward_terms

        self.previous_action = action.copy()
        self.previous_target_ctrl = target_ctrl.copy()

        # Resample after rewarding the action that was selected under the old
        # command, so the returned observation contains the new command.
        if (
            not self.command_override
            and self.current_step % self.command_resample_steps == 0
            and not terminated
            and not truncated
        ):
            self._sample_command()

        info = {
            "forward_velocity": float(self.data.qvel[0]),
            "lateral_velocity": float(self.data.qvel[1]),
            "forward_velocity_yaw": metrics["forward_velocity_yaw"],
            "lateral_velocity_yaw": metrics["lateral_velocity_yaw"],
            "height": float(self.data.qpos[2]),
            "target_velocity": float(self.command[0]),
            "target_lateral_velocity": float(self.command[1]),
            "target_yaw_rate": float(self.command[2]),
            "left_contact": metrics["left_contact"],
            "right_contact": metrics["right_contact"],
            "single_support": metrics["single_support"],
            "left_air_time": metrics["left_air_time"],
            "right_air_time": metrics["right_air_time"],
            "feet_slide_raw": metrics["feet_slide_raw"],
            "reward_terms": reward_terms,
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self):
        pass
