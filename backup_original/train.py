import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from g1_env import G1Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to Menagerie unitree_g1/scene.xml')
    parser.add_argument('--steps', type=int, default=200_000)
    parser.add_argument('--envs', type=int, default=4)
    args = parser.parse_args()
    # SubprocVecEnv on macOS uses spawn: start process creation only under __main__.
    env = make_vec_env(G1Env, n_envs=args.envs, env_kwargs={'model_path': args.model},
                       vec_env_cls=SubprocVecEnv if args.envs > 1 else None)
    model = PPO('MlpPolicy', env, verbose=1, device='cpu', n_steps=1024,
                batch_size=256, learning_rate=3e-4, gamma=0.99,
                policy_kwargs={'net_arch': [256, 256]}, tensorboard_log='./runs')
    try:
        model.learn(total_timesteps=args.steps, progress_bar=True)
    finally:
        model.save('g1_ppo_baseline')
        env.close()


if __name__ == '__main__':
    main()
