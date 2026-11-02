import argparse
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from g1_env import G1Env, joint_set_for_action_dim
from train import vecnormalize_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--policy', default='checkpoints/g1_ppo_final.zip')
    p.add_argument('--command', type=float, default=0.5, help='Forward speed command (m/s)')
    p.add_argument('--steps', type=int, default=5000)
    args = p.parse_args()
    agent = PPO.load(args.policy, device='cpu')
    joints = joint_set_for_action_dim(agent.action_space.shape[0])
    env = DummyVecEnv([lambda: G1Env(args.model, render_mode='human', joints=joints,
                                     fixed_command=args.command)])
    # Use the observation statistics from training, frozen.
    env = VecNormalize.load(str(vecnormalize_path(args.policy)), env)
    env.training = False
    env.norm_reward = False
    g1 = env.envs[0]
    try:
        obs = env.reset()
        for _ in range(args.steps):
            start = time.perf_counter()
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            g1.render()
            if done[0]:
                info = infos[0]
                print(f"Episode ended: vx={info['forward_velocity']:.2f} m/s, "
                      f"height={info['height']:.2f} m, truncated={info.get('TimeLimit.truncated')}")
            # Play back in real time.
            time.sleep(max(0.0, g1.dt - (time.perf_counter() - start)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
