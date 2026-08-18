import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class G1Env(gym.Env):
    def __init__(self):
        super().__init__()

        self.model_path = os.path.expanduser(
            "~/Documents/mujoco_menagerie/unitree_g1/scene.xml"
        )

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Find standing keyframe
        self.stand_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "stand",
        )

        if self.stand_id == -1:
            raise RuntimeError("Could not find G1 'stand' keyframe.")

        # Save standing actuator targets
        self.stand_ctrl = self.model.key_ctrl[self.stand_id].copy()

        self.action_dim = self.model.nu

        # Observation:
        # qpos + qvel
        self.obs_dim = self.model.nq + self.model.nv

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

        # How far PPO can move joints away from standing pose
        self.action_scale = 0.25

        # MuJoCo physics steps per PPO action
        self.frame_skip = 5

        self.current_step = 0
        self.max_steps = 1000

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self.stand_id,
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.current_step = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # PPO controls deviations from standing pose
        target_ctrl = (
            self.stand_ctrl
            + self.action_scale * action
        )

        # Respect actuator limits
        ctrl_min = self.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.model.actuator_ctrlrange[:, 1]

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

        obs = self._get_obs()

        forward_velocity = float(
            self.data.qvel[0]
        )

        height = float(
            self.data.qpos[2]
        )

        # Simple first reward
        forward_reward = forward_velocity
        alive_reward = 1.0

        action_penalty = (
            0.01 * np.sum(action ** 2)
        )

        reward = (
            forward_reward
            + alive_reward
            - action_penalty
        )

        terminated = (
            height < 0.5
            or not np.isfinite(obs).all()
        )

        truncated = (
            self.current_step >= self.max_steps
        )

        info = {
            "forward_velocity": forward_velocity,
            "height": height,
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )