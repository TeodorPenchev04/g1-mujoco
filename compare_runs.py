"""Compare the sample efficiency of training runs (e.g. PPO vs DISC) on benchmark v1.

    python compare_runs.py --model ../mujoco_menagerie/unitree_g1/scene.xml \
        --runs ../g1_experiments/ppo_vs_disc/ppo_s1 ../g1_experiments/ppo_vs_disc/disc_s1 --out compare.png

Plots each run's un-normalized episode return and length against env steps (from the Monitor
CSVs train.py writes to <run>/monitor), and evaluates every saved checkpoint with
benchmark_eval.py to find the first env step count at which the policy walks: no falls at
0 / 0.5 / 1.0 m/s, and at 0.5 and 1.0 m/s speed within 0.1 m/s of the command, phase match
>= 95% and heading error <= 15 deg, holding at the next checkpoint too. Checkpoint results are
cached in <run>/eval_cache.json.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMMANDS = [0.0, 0.5, 1.0]


def episodes(run):
    """All episodes of a run as (env steps at episode end, return, length), sorted by steps.
    The envs step in lockstep, so global steps = n_envs x the env's own step count."""
    files = sorted((run / 'monitor').glob('*.monitor.csv'))
    frames = []
    for f in files:
        df = pd.read_csv(f, skiprows=1)
        df['steps'] = df['l'].cumsum() * len(files)
        frames.append(df)
    return pd.concat(frames).sort_values('steps')


def binned(df, col, width=200_000):
    bins = (df['steps'] // width) * width + width / 2
    return df.groupby(bins)[col].mean()


def metric(result, cmd, name):
    return result.get(f'cmd_{cmd}', {}).get(name, np.nan)


def walks(result):
    if result['falls'] > 0:
        return False
    return all(result[f'cmd_{c}']['speed_error'] <= 0.1 and result[f'cmd_{c}']['phase_match'] >= 0.95
               and result[f'cmd_{c}']['max_heading_error_deg'] <= 15 for c in (0.5, 1.0))


def _evaluate(job):
    from benchmark_eval import evaluate
    path, model, n = job
    try:
        return path, evaluate(path, model, COMMANDS, n)
    except (IndexError, ValueError):
        # Every episode fell within benchmark_eval's 2 s warm-up, so there is nothing to measure.
        return path, {'falls': len(COMMANDS) * n, 'no_steady_state': True}


def evaluate_checkpoints(runs, model, n_episodes, workers):
    caches = {run: json.loads((run / 'eval_cache.json').read_text()) if (run / 'eval_cache.json').exists() else {}
              for run in runs}
    jobs = [(str(p), model, n_episodes) for run in runs for p in sorted(run.glob('g1_ppo_*_steps.zip'))
            if p.name not in caches[run]]
    if jobs:
        print(f'Evaluating {len(jobs)} checkpoints...')
        with Pool(workers) as pool:
            for path, result in pool.imap_unordered(_evaluate, jobs):
                p = Path(path)
                caches[p.parent][p.name] = result
                (p.parent / 'eval_cache.json').write_text(json.dumps(caches[p.parent], indent=1))
    out = {}
    for run in runs:
        steps = {int(re.search(r'_(\d+)_steps', k)[1]): v for k, v in caches[run].items()}
        out[run] = dict(sorted(steps.items()))
    return out


def steps_to_walk(evals):
    s = list(evals)
    ok = [walks(evals[k]) for k in s]
    for i, k in enumerate(s):
        if ok[i] and (i + 1 == len(s) or ok[i + 1]):
            return k
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--runs', nargs='+', required=True, help='train.py --out directories')
    p.add_argument('--episodes', type=int, default=1, help='Eval episodes per command per checkpoint')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default='compare.png')
    args = p.parse_args()
    runs = [Path(r) for r in args.runs]
    labels = {}
    for run in runs:
        info = json.loads((run / 'run_info.json').read_text())
        labels[run] = f"{info['algo'].upper()} seed {info['seed']} ({run.name})"
    evals = evaluate_checkpoints(runs, args.model, args.episodes, args.workers)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for run in runs:
        df = episodes(run)
        label = labels[run]
        line, = ax[0, 0].plot(*zip(*binned(df, 'r').items()), label=label)
        ax[0, 1].plot(*zip(*binned(df, 'l').items()), color=line.get_color(), label=label)
        e = evals[run]
        if e:
            steps = np.array(list(e))
            ax[1, 0].plot(steps, [metric(e[k], 1.0, 'speed_error') for k in steps], 'o-', color=line.get_color(), label=label)
            ax[1, 1].plot(steps, [metric(e[k], 1.0, 'phase_match') * 100 for k in steps], 'o-', color=line.get_color(), label=label)
            first = steps_to_walk(e)
            if first is not None:
                for a in ax.flat:
                    a.axvline(first, color=line.get_color(), ls='--', lw=1)
        print(f"{label}: {len(df)} episodes; walks from {steps_to_walk(e) or 'never'} steps "
              f"({len(e)} checkpoints evaluated)")
    ax[0, 0].set_title('Episode return (un-normalized, 200k-step bins)')
    ax[0, 1].set_title('Episode length (steps, max 1000)')
    ax[1, 0].set_title('Checkpoint eval: speed error at 1.0 m/s (m/s)')
    ax[1, 1].set_title('Checkpoint eval: phase match at 1.0 m/s (%)')
    ax[1, 0].axhline(0.1, color='gray', lw=0.8)
    ax[1, 1].axhline(95, color='gray', lw=0.8)
    for a in ax[1]:
        a.set_xlabel('env steps')
    for a in ax.flat:
        a.grid(alpha=0.3)
    ax[0, 0].legend()
    fig.suptitle('Dashed lines: first checkpoint that meets the walking standard')
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
