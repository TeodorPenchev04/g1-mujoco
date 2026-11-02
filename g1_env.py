"""G1 velocity-tracking Gymnasium environment, standard MuJoCo (CPU).

Model: github.com/google-deepmind/mujoco_menagerie/unitree_g1/scene.xml
Rewards adapted conceptually from Isaac Lab / legged_gym velocity-tracking configs.

Benchmark v1 is frozen: any change to observations, actions, rewards, dynamics or episode
setup must bump ENV_VERSION, and results from different versions must not be compared.
"""
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

ENV_VERSION = "v1"

LEG_JOINTS = [
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM_JOINTS = [f"{side}_{name}_joint" for side in ("left", "right")
              for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
# Joints the policy controls; the rest (wrists, and arms unless included) hold the default pose.
JOINT_SETS = {"legs": LEG_JOINTS, "legs_waist": LEG_JOINTS + WAIST_JOINTS,
              "legs_waist_arms": LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS, "full": None}
# When the arms are controlled: arms hang (shoulder pitch 0) and use soft actuators so they
# swing like pendulums. Elbow convention: 0 rad = bent ~82°, larger = straighter (1.28 rad =
# 15° bend in the 'stand' keyframe). The default 0.5 rad (55° bend) is the action center the
# 23-DOF checkpoints were trained with; the arm-posture reward pulls the elbow to
# RELAXED_ELBOW (~25° bend). For a fresh run, the default could be set to RELAXED_ELBOW directly.
RELAXED_ARM_POSE = {"shoulder_pitch": 0.0, "elbow": 0.5}
RELAXED_ELBOW = 1.1
ARM_VERTICAL = 0.087  # shoulder pitch (rad) that holds the upper arm vertical
ARM_KP, ARM_KV = 50.0, 6.0  # vs kp 500 on the legs; kv 3 let the forearms bounce on impacts
FOOT_BODIES =("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_SITES = ("left_foot", "right_foot")
# Menagerie's 'stand' keyframe has perfectly straight legs, which lets the policy lock its
# knees against the joint limit. Use a bent-knee default instead (foot stays flat:
# hip + knee + ankle pitch = 0).
BENT_LEG_POSE = {"hip_pitch": -0.3, "knee": 0.6, "ankle_pitch": -0.3}
# Action range per joint as (below default, above default) in rad. The knee can bend to
# ~86° but only straighten to ~6°, so it cannot be driven into its hyperextension limit.
# Ranges were widened where policies sat at the limit (hip pitch, ankle roll, waist, elbow):
# a saturated action holds the target still and then releases it in a jump.
ACTION_RANGE = {"knee": (0.5, 0.9),
                # Hip pitch -1.0..+0.5 rad: room for the leg to extend behind the body.
                "hip_pitch": (0.7, 0.8),
                # The ankle roll joint itself only goes to ±0.26 rad.
                "ankle_roll": (0.26, 0.26),
                # The upper body is 45% of the mass: limited waist ranges so it can
                # counter-rotate and lean for balance, but not throw the torso around.
                "waist_yaw": (0.4, 0.4), "waist_roll": (0.25, 0.25), "waist_pitch": (0.25, 0.25),
                # Arms: swing mainly in pitch; elbow 0.0-1.8 rad (bent ~82° to nearly straight).
                "shoulder_pitch": (0.6, 0.6), "shoulder_roll": (0.2, 0.2),
                "shoulder_yaw": (0.2, 0.2), "elbow": (0.5, 1.3)}
DEFAULT_ACTION_RANGE = (0.5, 0.5)


def joint_set_for_action_dim(n):
    """Which joint set a trained policy used, from its number of actions."""
    return next(k for k, v in JOINT_SETS.items() if (len(v) if v else 29) == n)

class G1Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, model_path: str, render_mode=None, command_range=(0.0, 1.2),
                 stand_prob=0.3, joints="legs_waist_arms", fixed_command=None, termination_penalty=200.0,
                 gait_period=(1.1, 0.8), swing_height=0.09, step_width=0.14):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.005
        self.decimation = 4
        self.dt = self.model.opt.timestep * self.decimation
        self.max_steps = 1000  # 20 seconds
        self.command_range = command_range
        self.stand_prob = stand_prob
        self.fixed_command = fixed_command
        self.termination_penalty = termination_penalty
        self.command = 0.0
        self.command_target = 0.0
        self.command_ramp = 0.5  # s to ramp the commanded speed in from standstill
        self.render_mode = render_mode
        self.viewer = None

        # Menagerie G1: actuator i drives hinge joint i + 1 (joint 0 is the free joint),
        # and ctrl is the target joint angle of a position actuator.
        self.n_joints = self.model.nu
        if self.n_joints < 12:
            raise ValueError(f"Expected G1 actuator model; found {self.n_joints} actuators")
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id < 0:
            raise ValueError("G1 MJCF has no 'stand' keyframe")
        self.stand_qpos = self.model.key_qpos[key_id].copy()
        self.default_q = self.stand_qpos[7:7 + self.n_joints].copy()

        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                   self.model.actuator_trnid[i, 0]) for i in range(self.n_joints)]
        # Joints outside the chosen set (arms by default) are held at the default pose.
        controlled = JOINT_SETS[joints]
        self.act_idx = (np.arange(self.n_joints) if controlled is None
                        else np.array([names.index(j) for j in controlled]))
        self.n_act = len(self.act_idx)
        self.arms = controlled is None or ARM_JOINTS[0] in controlled
        # Deviation penalty for the remaining joints (wrists; arms when they aren't controlled).
        # Legs, waist and controlled arms have their own terms.
        own_terms = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll", "waist")
        if self.arms:
            own_terms += ("shoulder", "elbow")
        self.posture_weights = np.array([0.0 if any(f in n for f in own_terms) else 1.0 for n in names])
        side_idx = lambda joint: np.array([names.index(f"{s}_{joint}_joint") for s in ("left", "right")])
        self.shoulder_pitch_idx = side_idx("shoulder_pitch")
        self.shoulder_roll_yaw_idx = np.concatenate([side_idx("shoulder_roll"), side_idx("shoulder_yaw")])
        self.elbow_idx = side_idx("elbow")
        self.arm_idx = np.concatenate([self.shoulder_pitch_idx, self.shoulder_roll_yaw_idx, self.elbow_idx])
        if self.arms:
            for i, n in enumerate(names):
                for k, angle in RELAXED_ARM_POSE.items():
                    if k in n:
                        self.stand_qpos[7 + i] = angle
            for i in self.arm_idx:
                self.model.actuator_gainprm[i, 0] = ARM_KP
                self.model.actuator_biasprm[i, 1] = -ARM_KP
                self.model.actuator_biasprm[i, 2] = -ARM_KV
        self.waist_idx = np.array([names.index(j) for j in WAIST_JOINTS])
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.hip_roll_idx = np.array([i for i, n in enumerate(names) if "hip_roll" in n])
        self.hip_yaw_idx = np.array([i for i, n in enumerate(names) if "hip_yaw" in n])
        self.hip_pitch_idx = np.array([i for i, n in enumerate(names) if "hip_pitch" in n])
        # Mirror map for the symmetry reward: swap left/right joints, and flip the sign of
        # roll/yaw joints (a mirrored leg rolls and turns the other way). Waist joints map
        # to themselves (yaw/roll flipped).
        swap = lambda n: n.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_")
        self.mirror_perm = np.array([names.index(swap(n)) for n in names])
        self.mirror_sign = np.array([-1.0 if ("roll" in n or "yaw" in n) else 1.0 for n in names])
        self.knee_idx = np.array([i for i, n in enumerate(names) if "knee" in n])
        ranges = [next((r for k, r in ACTION_RANGE.items() if k in names[i]), DEFAULT_ACTION_RANGE)
                  for i in self.act_idx]
        self.action_lo = np.array([r[0] for r in ranges])
        self.action_hi = np.array([r[1] for r in ranges])

        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0:
            raise ValueError("scene has no 'floor' geom")
        self.floor_id = floor_id
        self.foot_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                              for b in FOOT_BODIES]
        self.foot_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s)
                              for s in FOOT_SITES]
        # Foot site height when standing on the floor; swing height is measured above it.
        self.data.qpos[:] = self.stand_qpos
        mujoco.mj_forward(self.model, self.data)
        self.foot_rest_z = float(np.mean(self.data.site_xpos[self.foot_site_ids, 2]))
        # Bend the legs, then lower the pelvis so the feet stay on the floor.
        for i, n in enumerate(names):
            for k, angle in BENT_LEG_POSE.items():
                if k in n:
                    self.stand_qpos[7 + i] = angle
        self.data.qpos[:] = self.stand_qpos
        mujoco.mj_forward(self.model, self.data)
        self.stand_qpos[2] -= float(np.mean(self.data.site_xpos[self.foot_site_ids, 2])) - self.foot_rest_z
        self.default_q = self.stand_qpos[7:7 + self.n_joints].copy()
        self.default_height = float(self.stand_qpos[2])

        # Gait clock: phase_cnt counts control steps; the legs are half a cycle apart.
        # The period depends on the command (slow walking takes slower, longer steps):
        # gait_period[0] at <= 0.3 m/s down to gait_period[1] at >= 1.0 m/s.
        self.gait_period_range = gait_period
        self.swing_height = swing_height  # target foot lift above rest height (m)
        self.step_width = step_width  # target sideways distance between the feet (m)
        self._set_gait_period()
        self.phase_cnt = 0
        self.target_yaw = 0.0
        self.gait_quality = 0.5

        self.prev_action = np.zeros(self.n_act)
        self.prev_v = np.zeros(3)  # body-frame velocity last step, for acceleration
        self.prev_prev_action = np.zeros(self.n_act)
        self.q_history = []  # joint angles over the last half gait cycle, for symmetry
        self.shoulder_avg = np.zeros(2)  # running average of shoulder pitch around vertical
        self.feet_air_time = np.zeros(2)
        self.raw_air_time = np.zeros(2)  # unfiltered time off the ground, for bounce detection
        self.last_contact = np.zeros(2, dtype=bool)
        self.prev_foot_pos = np.zeros((2, 3))
        self.steps = 0
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_act,), dtype=np.float32)
        # body-frame lin vel (3), ang vel (3), gravity direction (3), command (1),
        # heading error (1), gait phase sin/cos for left and right leg (4),
        # joint positions vs standing (nu), joint velocities (nu), previous actions (n_act).
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(15 + 2 * self.n_joints + self.n_act,), dtype=np.float32
        )

    def _state(self):
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        R = rot.reshape(3, 3)
        # Free joint: qvel[:3] is world-frame linear velocity, qvel[3:6] is already body-frame.
        v_body = R.T @ self.data.qvel[:3]
        w_body = self.data.qvel[3:6].copy()
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
        q = self.data.qpos[7:7 + self.n_joints]
        dq = self.data.qvel[6:6 + self.n_joints]
        return v_body, w_body, gravity_body, q, dq

    def _foot_contacts(self):
        contact = np.zeros(2, dtype=bool)
        n = self.data.ncon
        if n == 0:
            return contact
        g1 = self.data.contact.geom1[:n]
        g2 = self.data.contact.geom2[:n]
        other = np.where(g1 == self.floor_id, g2, np.where(g2 == self.floor_id, g1, -1))
        bodies = self.model.geom_bodyid[other[other >= 0]]
        for i, b in enumerate(self.foot_body_ids):
            contact[i] = np.any(bodies == b)
        return contact

    def mirror_maps(self):
        """Left/right mirror for observations and actions, as (perm, sign) pairs:
        mirrored[i] = sign[i] * x[perm[i]]. Used by the symmetry loss in SymmetricPPO.
        Mirroring across the sagittal plane flips lateral velocity, roll/yaw rates, lateral
        gravity and heading error, swaps the two legs' phase signals, and maps each joint to
        its counterpart (roll/yaw joints change sign)."""
        nj = self.n_joints
        obs_perm = np.arange(self.observation_space.shape[0])
        obs_sign = np.ones(self.observation_space.shape[0])
        obs_sign[[1, 3, 5, 7, 10]] = -1.0           # v_y, w_x, w_z, gravity_y, heading error
        obs_perm[11:15] = [13, 14, 11, 12]           # (sin, cos) left <-> right
        for off in (15, 15 + nj):                    # joint positions, joint velocities
            obs_perm[off:off + nj] = off + self.mirror_perm
            obs_sign[off:off + nj] = self.mirror_sign
        act_pos = {j: i for i, j in enumerate(self.act_idx)}
        act_perm = np.array([act_pos[self.mirror_perm[j]] for j in self.act_idx])
        act_sign = self.mirror_sign[self.act_idx]
        off = 15 + 2 * nj                            # previous action
        obs_perm[off:] = off + act_perm
        obs_sign[off:] = act_sign
        return obs_perm, obs_sign, act_perm, act_sign

    def _yaw(self):
        w, x, y, z = self.data.qpos[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2)))

    def _moving(self):
        # Uses the episode's target, so the clock runs from the start while the
        # commanded speed ramps in.
        return self.command_target > 0.1

    def _set_gait_period(self):
        slow, fast = self.gait_period_range
        self.gait_period = float(np.interp(self.command_target, [0.3, 1.0], [slow, fast]))
        self.period_steps = int(round(self.gait_period / self.dt))

    def _gait_phase(self):
        """Per-leg phase signals: left = sin(2π·phase), right = -sin(2π·phase).
        Cos is included so the policy can tell a rising signal from a falling one.
        A leg is in stance while its sin signal is above -0.15 (short double support
        around the crossings), in swing otherwise. Zero (both stance) when standing."""
        if not self._moving():
            return np.zeros(2), np.zeros(2), np.ones(2, dtype=bool)
        phase = (self.phase_cnt % self.period_steps) / self.period_steps
        s = np.sin(2 * np.pi * phase) * np.array([1.0, -1.0])
        c = np.cos(2 * np.pi * phase) * np.array([1.0, -1.0])
        return s, c, s > -0.15

    def _swing_progress(self, sin_phase, cos_phase):
        """Per leg, how far through its swing (0 at lift-off, 1 at touchdown); 0 in stance.
        Swing is where the leg's sin signal is below -0.15, i.e. angle in (π + a, 2π - a)."""
        a = np.arcsin(0.15)
        theta = np.mod(np.arctan2(sin_phase, cos_phase), 2 * np.pi)
        return np.clip((theta - (np.pi + a)) / (np.pi - 2 * a), 0.0, 1.0) * (sin_phase < -0.15)

    def _obs(self):
        v, w, gravity, q, dq = self._state()
        sin_phase, cos_phase, _ = self._gait_phase()
        heading_err = np.angle(np.exp(1j * (self._yaw() - self.target_yaw)))
        return np.concatenate([
            v, w, gravity, [self.command, heading_err],
            [sin_phase[0], cos_phase[0], sin_phase[1], cos_phase[1]],
            q - self.default_q, dq * 0.1, self.prev_action
        ]).astype(np.float32)

    def _sample_command(self):
        if self.fixed_command is not None:
            return float(self.fixed_command)
        if self.np_random.uniform() < self.stand_prob:
            return 0.0
        return float(self.np_random.uniform(*self.command_range))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.stand_qpos
        self.data.ctrl[:] = self.default_q
        self.data.qpos[0:2] += self.np_random.uniform(-0.02, 0.02, size=2)
        # Noise only on the joints the policy controls: the fixed arm joints barely move
        # in training, so noise there looks like a 10-25 std outlier to the normalizer.
        self.data.qpos[7 + self.act_idx] += self.np_random.uniform(-0.015, 0.015, self.n_act)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.command_target = self._sample_command()
        self.command = 0.0  # ramps up to command_target over command_ramp seconds
        self._set_gait_period()
        # Start in double support (both feet meant to be down), with either leg leading.
        self.phase_cnt = int(self.np_random.integers(2)) * (self.period_steps // 2)
        self.prev_action.fill(0.0)
        self.prev_v = self._state()[0]
        self.prev_prev_action.fill(0.0)
        self.q_history = []
        self.shoulder_avg = np.zeros(2)
        self.feet_air_time.fill(0.0)
        self.raw_air_time.fill(0.0)
        self.last_contact[:] = True
        self.prev_foot_pos = self.data.site_xpos[self.foot_site_ids].copy()
        self.target_yaw = self._yaw()
        self.gait_quality = 0.5
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self.default_q.copy()
        target[self.act_idx] += np.where(action < 0, self.action_lo, self.action_hi) * action
        # Avoid commanding beyond actuator target limits (when specified).
        limited = self.model.actuator_ctrllimited.astype(bool)
        target[limited] = np.clip(target[limited],
                                  self.model.actuator_ctrlrange[limited, 0],
                                  self.model.actuator_ctrlrange[limited, 1])
        self.data.ctrl[:] = target
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self.command = self.command_target * min(1.0, self.steps * self.dt / self.command_ramp)
        v, w, grav, q, dq = self._state()
        # Pelvis surging forward and dropping at each touchdown (falling onto the new foot
        # instead of rolling over the stance leg) shows up as forward/vertical acceleration.
        body_acc = (v - self.prev_v) / self.dt
        self.prev_v = v.copy()
        upright = grav[2] < -0.65
        height = float(self.data.qpos[2])
        fallen = (not upright) or height < 0.55 or height > 1.10
        invalid = not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        terminated = bool(fallen or invalid)
        truncated = bool(self.steps >= self.max_steps)

        self.phase_cnt += 1
        moving = self._moving()
        sin_phase, cos_phase, stance = self._gait_phase()
        swing_p = self._swing_progress(sin_phase, cos_phase)
        late_swing = ~stance & (swing_p > 0.7)

        # Foot contact state. Air time is rewarded on touchdown, as in legged_gym:
        # swings longer than 0.3 s earn reward (each foot swings ~45% of the 0.8-1.1 s period).
        contact = self._foot_contacts()
        touchdown = contact & ~self.last_contact
        # A touchdown after less than 0.1 s in the air is a bounce, not a step.
        bounce = touchdown & (self.raw_air_time < 0.1)
        self.raw_air_time = np.where(contact, 0.0, self.raw_air_time + self.dt)
        contact_filt = contact | self.last_contact
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self.dt
        air_time = float(np.sum((self.feet_air_time - 0.3) * first_contact)) if moving else 0.0
        self.feet_air_time[contact_filt] = 0.0
        self.last_contact = contact
        foot_pos = self.data.site_xpos[self.foot_site_ids].copy()
        foot_vel = (foot_pos - self.prev_foot_pos) / self.dt
        foot_vel_xy = foot_vel[:, :2]
        self.prev_foot_pos = foot_pos
        foot_speed = np.linalg.norm(foot_vel_xy, axis=1)
        # Linear in speed so slow stance-foot sliding (~0.15 m/s) is penalized too.
        slip = float(np.sum(contact * foot_speed))
        # The foot should arrive nearly still, not skid in at walking speed. Penalized every
        # step while an airborne foot is within 3 cm of the floor (a steady signal instead of
        # one hit at contact): horizontal speed plus downward speed.
        near_ground = ~contact & (foot_pos[:, 2] - self.foot_rest_z < 0.03)
        landing_speed = float(np.sum(near_ground * (foot_speed + np.clip(-foot_vel[:, 2], 0.0, None))))
        # Feet flat on the ground: tilt = horizontal part of gravity in the foot frame.
        foot_tilt = np.zeros(2)
        for i, b in enumerate(self.foot_body_ids):
            g_foot = self.data.xmat[b].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
            foot_tilt[i] = g_foot[0] ** 2 + g_foot[1] ** 2
        # Also in late swing, so the foot arrives level instead of correcting after contact.
        flat_foot = float(np.sum((contact | late_swing) * foot_tilt))
        flight = float(not contact.any())
        # Standing command: feet down and still, body not drifting.
        if moving:
            stand_still = 0.0
        else:
            stand_still = (float(np.sum(~contact))
                           + float(np.sum(np.linalg.norm(foot_vel_xy, axis=1)))
                           + 2.0 * float(np.linalg.norm(v[:2])))

        # Gait phase: each foot should be on the ground in stance and in the air in swing.
        # Standing (phase zeroed, both stance) means both feet stay down.
        gait_contact = float(np.mean(contact == stance))
        # Gait quality: phase match averaged over ~half a gait cycle (EMA). Keeping both feet
        # down already matches ~50% (the stance foot), so 50% counts as zero quality. The
        # velocity reward is scaled by it: a shuffle that ignores the clock earns ~30%.
        self.gait_quality += 0.1 * (gait_contact - self.gait_quality)
        quality = np.clip((self.gait_quality - 0.5) / 0.5, 0.0, 1.0)
        vel_scale = 0.3 + 0.7 * quality if moving else 1.0
        # Foot clearance: in swing, track a target height that rises from 0 at lift-off
        # to swing_height at mid-swing and back to 0 at touchdown.
        foot_h = foot_pos[:, 2] - self.foot_rest_z
        swing = ~stance
        if swing.any():
            # sin² profile: zero vertical speed at lift-off and touchdown (the old linear ramp
            # asked the foot to come down at ~0.6 m/s right up to contact).
            target_h = self.swing_height * np.sin(np.pi * swing_p) ** 2
            # Wide enough that partial lifts (3-5 cm) already earn reward.
            swing_height = float(np.mean(np.exp(-((foot_h - target_h)[swing] ** 2) / 0.003)))
        else:
            swing_height = 0.0
        # Feet too close sideways (in the heading frame) leads to leg collisions and tripping.
        yaw = self._yaw()
        d = foot_pos[0, :2] - foot_pos[1, :2]
        feet_lateral = -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]  # left minus right, ~0.24 m standing
        feet_too_close = max(0.0, 0.15 - feet_lateral)
        # Step width: walking with the feet wide apart forces a big sideways weight shift
        # every step, which the policy makes by swaying the upper body.
        # Measured per foot from the pelvis (each foot step_width/2 out to its side), in
        # stance and in swing: this rules out both the tightrope stance (stance foot under
        # the centerline) and the outward arc the swing leg made to get around it.
        left_dir = np.array([-np.sin(yaw), np.cos(yaw)])
        foot_lat = (foot_pos[:, :2] - self.data.qpos[:2]) @ left_dir * np.array([1.0, -1.0])
        width_err = float(np.sum((foot_lat - self.step_width / 2) ** 2)) if moving else 0.0
        # Catwalk: a foot landing on or past the body's centerline (stepping in front of the
        # other leg). Each foot is rewarded separately at touchdown for landing step_width/2
        # (~7 cm, human-like for the G1's hip width) out on its own side, so one foot can't
        # make up for the other; a foot on the ground within 3 cm of the centerline costs steeply.
        landing_width = 0.0
        if moving:
            for i in np.flatnonzero(touchdown & ~bounce):
                landing_width += np.exp(-((foot_lat[i] - self.step_width / 2) ** 2) / 0.0004)
        crossover = float(np.sum(contact * np.clip(0.03 - foot_lat, 0.0, None))) if moving else 0.0
        # Symmetry: each joint now vs its mirror half a gait cycle ago (left leg now should
        # do what the right leg did half a cycle earlier, and vice versa).
        self.q_history.append(q.copy())
        half = int(round(self.period_steps / 2))
        symmetry_err = arm_symmetry_err = 0.0
        if len(self.q_history) > half:
            q_half = self.q_history.pop(0)
            if moving:
                mirrored = self.mirror_sign * q_half[self.mirror_perm]
                symmetry_err = float(np.mean((q - mirrored)[self.act_idx] ** 2))
                # Separately for the arms, where an offset is otherwise diluted by the legs.
                if self.arms:
                    arm_symmetry_err = float(np.mean((q - mirrored)[self.arm_idx] ** 2))
        # Body-frame velocity reward is satisfied by walking in circles, so also hold
        # the heading the episode started with (wrapped to [-pi, pi]).
        heading_err = np.angle(np.exp(1j * (yaw - self.target_yaw)))
        # Stride: at each (non-bounce) touchdown, how far the landing foot is ahead of the
        # other foot. The clock fixes cadence, so speed needs step = command * period / 2.
        step_length = 0.0
        swing_reach = 0.0
        if moving:
            fwd = np.array([np.cos(yaw), np.sin(yaw)])
            target_step = self.command * self.gait_period / 2
            for i in np.flatnonzero(touchdown & ~bounce):
                step = float(fwd @ (foot_pos[i, :2] - foot_pos[1 - i, :2]))
                step_length += np.exp(-((step - target_step) ** 2) / 0.005)
            # Dense version: the swing foot travels from target_step behind the stance foot to
            # target_step ahead of it along a smoothstep, so it starts and arrives with zero
            # forward speed relative to the stance foot.
            for i in np.flatnonzero(swing):
                reach = float(fwd @ (foot_pos[i, :2] - foot_pos[1 - i, :2]))
                p = swing_p[i]
                target_reach = target_step * (2 * (3 * p ** 2 - 2 * p ** 3) - 1)
                swing_reach += np.exp(-((reach - target_reach) ** 2) / 0.01)
        # Upper body: the pelvis tilt term doesn't see the torso once the waist can move.
        g_torso = self.data.xmat[self.torso_id].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        torso_tilt = g_torso[0] ** 2 + g_torso[1] ** 2
        # Waist: yaw/roll held near neutral (with arms fixed, the torso is the only twisting
        # mass and the policy over-used it, lopsidedly, also to steer the pelvis).
        waist_dev = ((q - self.default_q)[self.waist_idx] ** 2) * np.array([5.0, 5.0, 1.0])
        waist_yaw_rate = dq[self.waist_idx[0]]
        # Heading of the torso too, so twisting the upper body can't fake a straight pelvis.
        R_torso = self.data.xmat[self.torso_id].reshape(3, 3)
        torso_heading_err = np.angle(np.exp(1j * (np.arctan2(R_torso[1, 0], R_torso[0, 0])
                                                  - self.target_yaw)))
        # Side-to-side rocking of the torso (roll rate about its own forward axis).
        torso_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.torso_id, torso_vel, 1)
        torso_roll_rate = torso_vel[0]
        # Arms (when controlled): each arm swings like a pendulum, opposite to the same-side
        # leg (left leg forward -> left arm back), as a smooth sinusoid from the gait clock.
        # (Copying the hip angle passed on the leg's slow-stance / fast-swing rhythm, so the
        # arms drifted forward slowly and snapped back.) The leg's cos signal is +1 when that
        # foot is furthest forward, so arm forward swing = -amplitude x cos. Shoulder pitch is
        # negative = forward, and joint 0 holds the upper arm ~5° forward of vertical, so the
        # swing is centered on ARM_VERTICAL.
        arm_swing = arm_posture = arm_vel = arm_center = 0.0
        if self.arms:
            d_q = q - self.default_q
            amplitude = np.interp(self.command, [0.0, 0.5, 1.2], [0.0, 0.26, 0.44])  # 0, 15°, 25°
            arm_forward = -amplitude * cos_phase
            target_sp = ARM_VERTICAL - arm_forward
            sp_dev = d_q[self.shoulder_pitch_idx]
            arm_swing = float(np.mean(np.exp(-((sp_dev - target_sp) ** 2) / 0.02)))
            # Each arm should swing evenly through vertical: its running average stays there.
            self.shoulder_avg += (sp_dev - ARM_VERTICAL - self.shoulder_avg) / self.period_steps
            arm_center = float(np.sum(self.shoulder_avg ** 2)) if moving else 0.0
            # Elbow bends more as the arm swings forward and straightens as it swings back
            # (~±6° around the relaxed ~25° bend for a ±0.3 rad arm swing; 0.8 overshot to ~35°).
            elbow_target = RELAXED_ELBOW + 0.4 * (sp_dev - ARM_VERTICAL)
            arm_posture = (2.0 * float(np.sum(d_q[self.shoulder_roll_yaw_idx] ** 2))
                           + 4.0 * float(np.sum((q[self.elbow_idx] - elbow_target) ** 2)))
            arm_vel = float(np.sum(dq[self.arm_idx] ** 2))
        # Stance knee around 17° (humans ~15-20°): neither crouched nor locked straight.
        stance_knee_err = (float(np.sum((contact & stance) * (q[self.knee_idx] - 0.3) ** 2))
                           if moving else 0.0)

        terms = {
            # σ² = 0.1: stepping in place at a 0.5 m/s command keeps only ~8% of this reward.
            # σ² = 0.05: a 0.1 m/s surge within a step costs ~20% of this reward.
            # Weight 2.0 (was 1.5): the arm terms cost reward and the policy traded speed for them.
            "vel_tracking": 2.0 * vel_scale * np.exp(-((v[0] - self.command) ** 2 + v[1] ** 2) / 0.05),
            "body_acc": -0.01 * (body_acc[0] ** 2 + body_acc[2] ** 2),
            "yaw_tracking": 1.0 * np.exp(-(w[2] ** 2) / 0.1),
            "heading": 0.5 * np.exp(-(heading_err ** 2) / 0.25),
            "torso_heading": 0.5 * np.exp(-(torso_heading_err ** 2) / 0.25),
            "alive": 0.3,
            "gait_contact": 1.5 * gait_contact,
            "swing_height": 1.5 * swing_height,
            "feet_distance": -5.0 * feet_too_close,
            # Knees straighter than ~9° (locked, resting on the joint limit) cost reward.
            "knee_bend": -5.0 * float(np.sum(np.clip(0.15 - q[self.knee_idx], 0.0, None))),
            # Lift the foot with the knee, not by twisting (yaw) the hip. Hip roll is penalized
            # lightly: it is needed for side-to-side balance, otherwise the ankle roll takes
            # over and the feet end up standing on their edges.
            "hip_yaw": -5.0 * float(np.sum((q - self.default_q)[self.hip_yaw_idx] ** 2)),
            "hip_roll": -1.0 * float(np.sum((q - self.default_q)[self.hip_roll_idx] ** 2)),
            # Pelvis a little above the bent-knee default (~20° stance knee instead of 34°).
            "base_height": -30.0 * (height - (self.default_height + 0.01)) ** 2,
            "stance_knee": -10.0 * stance_knee_err,
            "step_length": 3.0 * step_length,
            "swing_reach": 1.0 * swing_reach,
            "step_width": -30.0 * width_err,
            "landing_width": 2.0 * landing_width,
            "crossover": -50.0 * crossover,
            "symmetry": -10.0 * symmetry_err,
            "arm_symmetry": -20.0 * arm_symmetry_err,
            "flat_foot": -10.0 * flat_foot,
            "landing_speed": -1.0 * landing_speed,
            "bounce": -3.0 * float(np.sum(bounce)),
            "air_time": 0.25 * air_time,
            "lin_vel_z": -4.0 * v[2] ** 2,
            "ang_vel_xy": -0.05 * (w[0] ** 2 + w[1] ** 2),
            "tilt": -1.0 * (grav[0] ** 2 + grav[1] ** 2),
            "posture": -0.5 * float(np.sum(self.posture_weights * (q - self.default_q) ** 2)),
            # Torso stays upright; the waist is used when it helps and otherwise stays neutral.
            "torso_upright": -8.0 * torso_tilt,
            "torso_roll_rate": -0.2 * torso_roll_rate ** 2,
            "pelvis_roll": -5.0 * grav[1] ** 2,
            "waist": -1.0 * float(np.sum(waist_dev)),
            "waist_twist_rate": -0.1 * waist_yaw_rate ** 2,
            "arm_swing": 1.0 * arm_swing,
            "arm_center": -5.0 * arm_center,
            "arm_posture": -1.0 * arm_posture,  # shoulders in close (roll/yaw), elbow relaxed
            "arm_vel": -0.001 * arm_vel,
            "foot_slip": -0.5 * slip,
            "flight": -0.5 * flight,
            # Weight 2.0 (was 0.5): at 0.5 a policy that learned to shuffle in place at a 0 m/s
            # command never unlearned it, although standing still scored higher.
            "stand_still": -2.0 * stand_still,
            "action_rate": -0.01 * float(np.sum((action - self.prev_action) ** 2)),
            # Jerk: change in the action rate, so targets ramp instead of jumping.
            "action_jerk": -0.2 * float(np.sum((action - 2 * self.prev_action + self.prev_prev_action) ** 2)),
            "effort": -1e-5 * float(np.sum(np.square(self.data.actuator_force))),
        }
        reward = float(sum(terms.values()))
        # Falling must cost far more than the velocity reward it can earn by lunging forward
        # (Isaac Lab's G1 config makes it worth ~200 steps of tracking reward).
        if terminated:
            reward -= self.termination_penalty
        self.prev_prev_action = self.prev_action
        self.prev_action = action.copy()
        info = {"forward_velocity": float(v[0]), "command": self.command, "height": height,
                "left_contact": bool(contact[0]), "right_contact": bool(contact[1]),
                "left_stance": bool(stance[0]), "right_stance": bool(stance[1]),
                "foot_height": foot_h.tolist(), "heading_error": float(heading_err),
                "reward_terms": {k: float(x) for k, x in terms.items()}}
        return self._obs(), reward, terminated, truncated, info

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
