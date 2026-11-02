import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from disc_ppo import DISCPPO
from g1_env import ENV_VERSION, G1Env
from symmetric_ppo import SymmetricPPO


def vecnormalize_path(policy_path):
    """VecNormalize stats saved next to a policy: g1_ppo_final.zip -> g1_ppo_final_vecnormalize.pkl,
    checkpoint g1_ppo_500000_steps.zip -> g1_ppo_vecnormalize_500000_steps.pkl."""
    p = Path(policy_path)
    m = re.fullmatch(r'(.*)_(\d+_steps)', p.stem)
    name = f'{m[1]}_vecnormalize_{m[2]}.pkl' if m else f'{p.stem}_vecnormalize.pkl'
    return p.with_name(name)


def linear_schedule(start, end):
    """Learning rate decaying linearly from start to end over the run. With a constant 3e-4,
    a from-scratch run peaked at 12-16M steps and then degraded (falls, speed overshoot)."""
    return lambda progress_remaining: end + (start - end) * progress_remaining


class RewardTermLogger(BaseCallback):
    """Log the mean of each reward term over a rollout to TensorBoard (rewards/<term>)."""

    def _on_rollout_start(self):
        self.sums = defaultdict(float)
        self.count = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            for k, v in info.get("reward_terms", {}).items():
                self.sums[k] += v
            self.count += 1
        return True

    def _on_rollout_end(self):
        for k, v in self.sums.items():
            self.logger.record(f"rewards/{k}", v / max(self.count, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to Menagerie unitree_g1/scene.xml')
    parser.add_argument('--steps', type=int, default=20_000_000)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--joints', choices=['legs', 'legs_waist', 'legs_waist_arms', 'full'],
                        default='legs_waist_arms',
                        help='Joints the policy controls (the rest hold the default pose)')
    parser.add_argument('--out', default='checkpoints', help='Directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=1_000_000, help='Env steps between checkpoints')
    parser.add_argument('--resume', help='Path to a saved model .zip to continue training')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Starting learning rate; it decays linearly to --lr-end over this run')
    parser.add_argument('--lr-end', type=float, default=3e-5, help='Learning rate at the end of the run')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Minibatch size; with DISC, the on-policy part (8192 / batch-size gradient steps per epoch)')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per update')
    parser.add_argument('--sym-coef', type=float, default=1.0,
                        help='Weight of the left/right mirror symmetry loss (0 = plain PPO)')
    parser.add_argument('--algo', choices=['ppo', 'disc'], default='ppo',
                        help='ppo: SymmetricPPO (the benchmark baseline); disc: DISCPPO (disc_ppo.py)')
    parser.add_argument('--disc-eps', type=float, default=0.4, help='DISC per-dimension clipping factor')
    parser.add_argument('--j-targ', type=float, default=1e-4, help='DISC IS-loss target (paper, Table A.1)')
    parser.add_argument('--eps-b', type=float, default=0.1, help='DISC old-batch inclusion threshold')
    parser.add_argument('--replay', type=int, default=16, help='DISC replay buffer length (rollouts)')
    parser.add_argument('--device', default='cpu',
                        help="Torch device; 'mps' (M2 GPU) speeds up DISC's large replay minibatches ~4x")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    (out / 'run_info.json').write_text(json.dumps({**vars(args), 'env_version': ENV_VERSION,
                                                   'git_commit': commit}, indent=2))
    algo_cls = DISCPPO if args.algo == 'disc' else SymmetricPPO
    # SubprocVecEnv on macOS uses spawn: start process creation only under __main__.
    # Monitor CSVs (out/monitor/*.monitor.csv) record each episode's un-normalized return and length.
    env = make_vec_env(G1Env, n_envs=args.envs, seed=args.seed, monitor_dir=str(out / 'monitor'),
                       env_kwargs={'model_path': args.model, 'joints': args.joints},
                       vec_env_cls=SubprocVecEnv if args.envs > 1 else None)
    if args.resume:
        env = VecNormalize.load(str(vecnormalize_path(args.resume)), env)
        model = algo_cls.load(args.resume, env=env, device=args.device)
        # A loaded model keeps its saved learning rate, so set the schedule here. SB3 measures
        # progress against the total step count (including the steps already trained), so
        # rescale it to decay over this run's steps only.
        total = model.num_timesteps + args.steps
        run_schedule = linear_schedule(args.lr, args.lr_end)
        model.learning_rate = lambda p: run_schedule(min(1.0, p * total / args.steps))
        model.lr_schedule = model.learning_rate
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        # ~8k samples per update with 8 envs; small initial action noise so the robot
        # does not fall from random joint targets before it learns anything.
        # DISC bounds its update with the adaptive IS loss instead of PPO's KL early stopping.
        disc_kwargs = (dict(disc_eps=args.disc_eps, j_targ=args.j_targ, eps_b=args.eps_b,
                            replay_length=args.replay) if args.algo == 'disc' else {})
        model = algo_cls('MlpPolicy', env, verbose=1, device=args.device, seed=args.seed,
                    n_steps=1024, batch_size=args.batch_size, n_epochs=args.epochs,
                    learning_rate=linear_schedule(args.lr, args.lr_end), gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.005, vf_coef=0.5, max_grad_norm=1.0,
                    target_kl=None if args.algo == 'disc' else 0.03,
                    policy_kwargs={'net_arch': dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                                   'activation_fn': torch.nn.ELU, 'log_std_init': -1.0},
                    tensorboard_log='./runs', **disc_kwargs)
    model.mirror_maps = G1Env(args.model, joints=args.joints).mirror_maps()
    model.sym_coef = args.sym_coef

    checkpoints = CheckpointCallback(save_freq=max(args.save_every // args.envs, 1),
                                     save_path=str(out), name_prefix='g1_ppo',
                                     save_vecnormalize=True)
    try:
        model.learn(total_timesteps=args.steps, progress_bar=True,
                    callback=CallbackList([checkpoints, RewardTermLogger()]),
                    reset_num_timesteps=not args.resume)
    finally:
        model.save(out / 'g1_ppo_final')
        env.save(str(out / 'g1_ppo_final_vecnormalize.pkl'))
        print(f"Saved {out / 'g1_ppo_final.zip'} and its VecNormalize stats")
        env.close()


if __name__ == '__main__':
    main()
