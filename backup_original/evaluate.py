import argparse
from stable_baselines3 import PPO
from g1_env import G1Env


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--policy', default='g1_ppo_baseline.zip')
    args = p.parse_args()
    env = G1Env(args.model, render_mode='human')
    agent = PPO.load(args.policy, device='cpu')
    try:
        obs, _ = env.reset()
        for _ in range(5000):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                print('Episode ended:', info)
                obs, _ = env.reset()
    finally:
        env.close()


if __name__ == '__main__':
    main()
