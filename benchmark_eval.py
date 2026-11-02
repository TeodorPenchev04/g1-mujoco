"""Standard evaluation for benchmark v1: deterministic rollouts at fixed commands, gait metrics.

    python benchmark_eval.py --model ../mujoco_menagerie/unitree_g1/scene.xml \
        --policies checkpoints/g1_ppo_seed1_16M.zip checkpoints/g1_ppo_seed2_16M.zip --json results.json

Each policy runs --episodes episodes of 20 s per command (reset seeds 0..episodes-1); the first
2 s of each episode are skipped for steady-state metrics. With several policies (e.g. seeds of
one algorithm), the table also shows the mean and spread (max - min) across them.
"""
import argparse
import json
import warnings
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from g1_env import ENV_VERSION, G1Env, joint_set_for_action_dim
from train import vecnormalize_path

warnings.filterwarnings('ignore')
SKIP = 100  # steps (2 s) skipped before steady-state metrics


def rollout(agent, model_path, joints, cmd, seed):
    """One 20 s episode; returns per-step records and the env (for its model data)."""
    env = VecNormalize.load(str(vecnormalize_path(agent.path)), DummyVecEnv([lambda: G1Env(
        model_path, fixed_command=cmd, joints=joints)]))
    env.training = False
    env.norm_reward = False
    env.seed(seed)
    g = env.envs[0]; m = g.model; d = g.data
    feet_geoms = [[i for i in range(m.ngeom) if m.geom_bodyid[i] == b and m.geom_contype[i]] for b in g.foot_body_ids]
    body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    arm_bodies = [(body(s + '_shoulder_pitch_link'), body(s + '_elbow_link')) for s in ('left', 'right')]
    obs = env.reset()
    rec, fell = [], False
    prev_foot = d.site_xpos[g.foot_site_ids].copy()
    for t in range(1000):
        prev_contact = g.last_contact.copy()
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        info = infos[0]
        if done[0]:
            fell = not info.get('TimeLimit.truncated', False)
            break
        foot = d.site_xpos[g.foot_site_ids].copy()
        foot_vel = (foot - prev_foot) / g.dt
        prev_foot = foot
        yaw = g._yaw()
        fwd, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        touching = {d.contact.geom2[k] if d.contact.geom1[k] == g.floor_id else d.contact.geom1[k]
                    for k in range(d.ncon) if g.floor_id in (d.contact.geom1[k], d.contact.geom2[k])}
        R_torso = d.xmat[g.torso_id].reshape(3, 3)
        w, x, y, z = d.qpos[3:7]
        r = dict(
            t=t, contact=np.array([info['left_contact'], info['right_contact']]), prev_contact=prev_contact,
            stance=np.array([info['left_stance'], info['right_stance']]), vx=info['forward_velocity'],
            heading=info['heading_error'], foot_height=np.array(info['foot_height']),
            foot_speed=np.linalg.norm(foot_vel[:, :2], axis=1),
            foot_fwd=np.array([fwd @ (foot[k, :2] - foot[1 - k, :2]) for k in (0, 1)]),
            foot_out=np.array([left @ (foot[0, :2] - d.qpos[:2]), -(left @ (foot[1, :2] - d.qpos[:2]))]),
            foot_roll=np.degrees(np.arctan2(*(lambda Rf: (Rf[2, 1], Rf[2, 2]))(d.xmat[g.foot_body_ids[0]].reshape(3, 3)))),
            flat=sum(s in touching for s in feet_geoms[0]) >= 3,
            pelvis_roll=np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
            xy=d.qpos[:2].copy())
        if g.arms:
            r['arm'] = np.array([np.degrees(np.arctan2(*(lambda u: (u[0], -u[2]))(R_torso.T @ (d.xpos[e] - d.xpos[s]))))
                                 for s, e in arm_bodies])
            r['elbow'] = d.qpos[7 + g.elbow_idx].copy()
        rec.append(r)
    return rec, fell, g


def walking_metrics(recs, cmd, g):
    out = {}
    steady = [r for rec in recs for r in rec if r['t'] >= SKIP]
    C = np.array([r['contact'] for r in steady]); ST = np.array([r['stance'] for r in steady])
    out['speed'] = float(np.mean([r['vx'] for r in steady]))
    out['speed_error'] = abs(out['speed'] - cmd)
    out['phase_match'] = float(np.mean(C == ST))
    out['max_heading_error_deg'] = float(np.degrees(max(abs(r['heading']) for r in steady)))
    td, land_speed, land_out, steps = 0, [], [], []
    for rec in recs:
        for r in rec:
            if r['t'] < SKIP:
                continue
            for k in np.flatnonzero(r['contact'] & ~r['prev_contact']):
                td += k == 0
                land_speed.append(r['foot_speed'][k]); land_out.append((k, r['foot_out'][k])); steps.append(r['foot_fwd'][k])
    cycles = len(steady) / g.period_steps
    out['touchdowns_per_cycle'] = td / cycles
    out['stance_slip'] = float(np.median([r['foot_speed'][0] for r in steady if r['contact'][0]]))
    out['landing_speed'] = float(np.median(land_speed))
    out['stance_foot_roll_deg'] = float(np.mean([abs(r['foot_roll']) for r in steady if r['contact'][0]]))
    out['flat_contact'] = float(np.mean([r['flat'] for r in steady if r['contact'][0]]))
    lo = np.array([o for k, o in land_out if k == 0]); ro = np.array([o for k, o in land_out if k == 1])
    out['landing_out_left_cm'], out['landing_out_right_cm'] = float(np.median(lo) * 100), float(np.median(ro) * 100)
    out['crossovers'] = float(np.mean(np.r_[lo, ro] < 0))
    out['step_length'] = float(np.median(steps)); out['step_length_target'] = cmd * g.gait_period / 2
    swing_out = [r['foot_out'][k] for r in steady for k in (0, 1) if not r['stance'][k]]
    out['swing_closest_to_centerline_cm'] = float(min(swing_out) * 100)
    out['pelvis_roll_range_deg'] = float(np.ptp([r['pelvis_roll'] for r in steady]))
    # Forward-speed surge at touchdown: peak-to-peak of the speed profile averaged around touchdowns.
    v = np.array([r['vx'] for r in recs[0]]); c = np.array([r['contact'] for r in recs[0]])
    tds = np.sort(np.concatenate([np.flatnonzero(~c[:-1, k] & c[1:, k]) + 1 for k in (0, 1)]))
    win = [v[t - 12:t + 13] for t in tds if t >= SKIP and t + 13 <= len(v)]
    out['touchdown_speed_surge'] = float(np.ptp(np.mean(win, axis=0)) / max(out['speed'], 1e-6)) if win else float('nan')
    if g.arms:
        A = np.array([r['arm'] for r in steady]); E = np.array([r['elbow'] for r in steady])
        out['arm_back_deg'] = float(np.mean(-A.min(axis=0))); out['arm_forward_deg'] = float(np.mean(A.max(axis=0)))
        speed = np.abs(np.gradient(A, axis=0)) / g.dt
        out['arm_peak_speed_deg_s'] = float(speed.max(axis=0).mean())
        out['elbow_left_right_diff_deg'] = float(abs(np.degrees(E[:, 0].mean() - E[:, 1].mean())))
    return out


def standing_metrics(recs):
    steady = [r for rec in recs for r in rec if r['t'] >= SKIP]
    return {'both_feet_down': float(np.mean([r['contact'].all() for r in steady])),
            'foot_lift_p95_cm': float(np.percentile([r['foot_height'].max() for r in steady], 95) * 100),
            'drift_m': float(np.mean([np.linalg.norm(rec[-1]['xy'] - rec[SKIP]['xy']) for rec in recs if len(rec) > SKIP]))}


def evaluate(policy, model_path, commands, episodes):
    agent = PPO.load(policy, device='cpu')
    agent.path = policy
    joints = joint_set_for_action_dim(agent.action_space.shape[0])
    result = {'policy': policy, 'joints': joints, 'falls': 0, 'episodes': 0}
    for cmd in commands:
        recs = []
        for seed in range(episodes):
            rec, fell, g = rollout(agent, model_path, joints, cmd, seed)
            recs.append(rec); result['falls'] += fell; result['episodes'] += 1
        result[f'cmd_{cmd}'] = standing_metrics(recs) if cmd == 0 else walking_metrics(recs, cmd, g)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--policies', nargs='+', required=True)
    p.add_argument('--commands', nargs='+', type=float, default=[0.0, 0.5, 1.0])
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--json', help='Write all results to this file')
    args = p.parse_args()
    results = [evaluate(pol, args.model, args.commands, args.episodes) for pol in args.policies]
    names = [r['policy'].split('/')[-1].removesuffix('.zip') for r in results]
    print(f"benchmark {ENV_VERSION}, {args.episodes} episodes x 20 s per command, falls: "
          + ", ".join(f"{n} {r['falls']}/{r['episodes']}" for n, r in zip(names, results)))
    for cmd in args.commands:
        key = f'cmd_{cmd}'
        print(f"\n{'standing' if cmd == 0 else f'{cmd} m/s'}" + "".join(f" | {n:>18s}" for n in names)
              + (" |  mean | spread" if len(results) > 1 else ""))
        for metric in results[0][key]:
            vals = [r[key][metric] for r in results]
            line = f"  {metric:32s}" + "".join(f" | {v:18.3f}" for v in vals)
            if len(vals) > 1:
                line += f" | {np.mean(vals):5.3f} | {np.ptp(vals):.3f}"
            print(line)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'env_version': ENV_VERSION, 'results': results}, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == '__main__':
    main()
