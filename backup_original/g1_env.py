"""Minimal G1 velocity-tracking Gymnasium environment, standard MuJoCo (CPU).

Model: github.com/google-deepmind/mujoco_menagerie/unitree_g1/scene.xml
Baseline adapted conceptually from provided Isaac Lab velocity/reward configs.
"""
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class G1Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, model_path: str, render_mode=None, command_speed=0.5):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.005
        self.decimation = 4
        self.dt = self.model.opt.timestep * self.decimation
        self.max_steps = 1000  # 20 seconds
        self.command_speed = float(command_speed)
        self.render_mode = render_mode
        self.viewer = None
        self.n_act = self.model.nu
        if self.n_act < 12:
            raise ValueError(f"Expected G1 actuator model; found {self.n_act} actuators")
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id < 0:
            raise ValueError("G1 MJCF has no 'stand' keyframe")
        self.stand_qpos = self.model.key_qpos[key_id].copy()
        self.stand_ctrl = self.model.key_ctrl[key_id].copy()
        self.default_height = float(self.stand_qpos[2])
        self.prev_action = np.zeros(self.n_act, dtype=np.float64)
        self.steps = 0
        # This G1 Menagerie model uses joint-position actuators: ctrl is target joint angle.
        self.action_scale = 0.25
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_act,), dtype=np.float32)
        # body-frame lin vel (3), ang vel (3), gravity direction (3), command (1),
        # joint positions vs standing (nu), joint velocities (nu), previous actions (nu).
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(10 + 3 * self.n_act,), dtype=np.float32
        )

    def _state(self):
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        R = rot.reshape(3, 3)
        # Free-joint translational and angular velocities expressed in world frame.
        v_body = R.T @ self.data.qvel[:3]
        w_body = R.T @ self.data.qvel[3:6]
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
        q = self.data.qpos[7:7 + self.n_act]
        dq = self.data.qvel[6:6 + self.n_act]
        return v_body, w_body, gravity_body, q, dq

    def _obs(self):
        v, w, gravity, q, dq = self._state()
        return np.concatenate([
            v, w, gravity, [self.command_speed],
            q - self.stand_ctrl, dq * 0.1, self.prev_action
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.stand_qpos
        self.data.ctrl[:] = self.stand_ctrl
        self.data.qpos[0:2] += self.np_random.uniform(-0.02, 0.02, size=2)
        self.data.qpos[7:7 + self.n_act] += self.np_random.uniform(-0.015, 0.015, self.n_act)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.prev_action.fill(0.0)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self.stand_ctrl + self.action_scale * action
        # Avoid commanding beyond actuator target limits (when specified).
        limited = self.model.actuator_ctrllimited.astype(bool)
        target[limited] = np.clip(target[limited],
                                  self.model.actuator_ctrlrange[limited, 0],
                                  self.model.actuator_ctrlrange[limited, 1])
        self.data.ctrl[:] = target
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        v, w, grav, q, dq = self._state()
        upright = grav[2] < -0.65
        height = float(self.data.qpos[2])
        fallen = (not upright) or height < 0.55 or height > 1.10
        invalid = not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        terminated = bool(fallen or invalid)
        truncated = bool(self.steps >= self.max_steps)
        # TA-inspired baseline: velocity, orientation, posture, smoothness, effort.
        vel_reward = np.exp(-((v[0] - self.command_speed) ** 2 + v[1] ** 2) / 0.25)
        yaw_reward = np.exp(-(w[2] ** 2) / 0.25)
        tilt_penalty = grav[0] ** 2 + grav[1] ** 2
        posture_penalty = np.mean((q - self.stand_ctrl) ** 2)
        action_rate = np.mean((action - self.prev_action) ** 2)
        effort = np.mean(np.square(self.data.actuator_force))
        reward = (
            2.0 * vel_reward + 0.3 * yaw_reward + 0.3
            - 1.0 * tilt_penalty - 0.15 * posture_penalty
            - 0.05 * action_rate - 1e-5 * effort
        )
        if terminated:
            reward -= 10.0
        self.prev_action = action.copy()
        info = {"forward_velocity": float(v[0]), "height": height,
                "velocity_reward": float(vel_reward)}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
