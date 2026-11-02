"""Render a trained policy to an MP4 offscreen (no viewer window needed)."""
import argparse
import imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from g1_env import G1Env, joint_set_for_action_dim
from train import vecnormalize_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--policy', default='checkpoints/g1_ppo_final.zip')
    p.add_argument('--command', type=float, default=0.5, help='Forward speed command (m/s)')
    p.add_argument('--seconds', type=float, default=10.0)
    p.add_argument('--azimuth', type=float, default=90.0, help='Camera angle; 90 = side view')
    p.add_argument('--out', default='policy.mp4')
    args = p.parse_args()
    agent = PPO.load(args.policy, device='cpu')
    joints = joint_set_for_action_dim(agent.action_space.shape[0])
    env = DummyVecEnv([lambda: G1Env(args.model, joints=joints, fixed_command=args.command)])
    env = VecNormalize.load(str(vecnormalize_path(args.policy)), env)
    env.training = False
    env.norm_reward = False
    g1 = env.envs[0]

    renderer = mujoco.Renderer(g1.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(g1.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    cam.distance, cam.azimuth, cam.elevation = 2.8, args.azimuth, -12.0

    obs = env.reset()
    frames, vx, falls = [], [], 0
    for _ in range(int(args.seconds / g1.dt)):
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        if done[0] and not infos[0].get('TimeLimit.truncated'):
            falls += 1
        vx.append(infos[0]['forward_velocity'])
        renderer.update_scene(g1.data, camera=cam)
        frames.append(renderer.render())
    imageio.mimsave(args.out, frames, fps=round(1 / g1.dt), macro_block_size=8)
    print(f"Wrote {args.out}: command {args.command} m/s, mean vx {np.mean(vx):.2f} m/s, falls {falls}")
    env.close()


if __name__ == '__main__':
    main()
