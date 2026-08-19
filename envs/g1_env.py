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

        # Find standing keyframe
        self.stand_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "stand",
        )

        if self.stand_id == -1:
            raise RuntimeError(
                "Could not find G1 'stand' keyframe."
            )

        # Actuator commands for standing pose
        self.stand_ctrl = (
            self.model.key_ctrl[self.stand_id].copy()
        )

        # Use actual standing height from model
        self.target_height = float(
            self.model.key_qpos[self.stand_id][2]
        )

        self.action_dim = self.model.nu

        # qpos + qvel + previous action
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

        # Policy changes joint targets around standing pose
        self.action_scale = 0.20

        # MuJoCo physics steps per RL step
        self.frame_skip = 5

        self.max_steps = 1000
        self.current_step = 0

        # Walking target
        self.target_velocity = 0.6

        self.previous_action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.previous_action.copy(),
        ]).astype(np.float32)

    def _upright_value(self):
        quat = self.data.qpos[3:7]

        rotation = np.zeros(9)

        mujoco.mju_quat2Mat(
            rotation,
            quat,
        )

        rotation = rotation.reshape(3, 3)

        return float(rotation[2, 2])

    def _get_reward(self, action):
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        yaw_rate = float(self.data.qvel[5])

        height = float(self.data.qpos[2])

        # Target forward velocity
        velocity_error = (
            vx - self.target_velocity
        )

        velocity_reward = np.exp(
            -8.0 * velocity_error**2
        )

        upright_reward = self._upright_value()

        height_reward = np.exp(
            -10.0
            * (height - self.target_height) ** 2
        )

        lateral_penalty = vy**2

        yaw_penalty = yaw_rate**2

        action_penalty = np.mean(
            action**2
        )

        action_rate_penalty = np.mean(
            (
                action
                - self.previous_action
            ) ** 2
        )

        reward = (
            4.0 * velocity_reward
            + 0.5 * upright_reward
            + 0.2 * height_reward
            - 0.20 * lateral_penalty
            - 0.10 * yaw_penalty
            - 0.01 * action_penalty
            - 0.02 * action_rate_penalty
        )

        return float(reward)

    def _is_terminated(self):
        height = float(
            self.data.qpos[2]
        )

        upright = self._upright_value()

        # Fall
        if height < self.target_height * 0.70:
            return True

        # Large tilt
        if upright < 0.5:
            return True

        # Numerical instability
        if not np.isfinite(
            self.data.qpos
        ).all():
            return True

        if not np.isfinite(
            self.data.qvel
        ).all():
            return True

        return False

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

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self._get_obs(), {}

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

        for _ in range(self.frame_skip):
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

        info = {
            "forward_velocity": float(
                self.data.qvel[0]
            ),
            "lateral_velocity": float(
                self.data.qvel[1]
            ),
            "height": float(
                self.data.qpos[2]
            ),
            "upright": (
                self._upright_value()
            ),
            "target_velocity": (
                self.target_velocity
            ),
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