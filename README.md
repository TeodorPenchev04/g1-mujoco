# G1 MuJoCo locomotion benchmark (v1)

Velocity-tracking walking for the Unitree G1 humanoid (MuJoCo Menagerie model) in standard CPU MuJoCo, trained with PPO (Stable-Baselines3) on an M2 Mac. The env aims for a human-like gait: an alternating gait on a clock, bent knees, flat-footed landings, straight symmetric leg paths, arm swing opposite the legs, and standing still at 0 m/s. Rewards follow Isaac Lab / legged_gym conceptually; the weights are not an exact transfer.

**Benchmark v1 is frozen** (`ENV_VERSION = "v1"` in `g1_env.py`): a from-scratch PPO baseline reproduces the gait across seeds (see [Results](#benchmark-v1-results)). Any change to observations, actions, rewards, dynamics or episode setup must bump the version.

## Install (macOS)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# From the parent directory:
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
```

## Train

```bash
# Benchmark baseline: 16M steps from scratch (~1.5 h on an M2 with 8 envs)
python train.py --model ../mujoco_menagerie/unitree_g1/scene.xml --steps 16000000 --envs 8 --seed 1 --out runs_seed1
# Continue from a checkpoint (learning rate decays over the resumed steps; --lr sets its start)
python train.py --model ../mujoco_menagerie/unitree_g1/scene.xml --steps 4000000 --lr 9e-5 --resume checkpoints/g1_ppo_seed1_16M.zip
# DISC (disc_ppo.py) with the tuned settings from "DISC vs PPO" below (~4.5–5 h for 16M steps)
python train.py --model ../mujoco_menagerie/unitree_g1/scene.xml --steps 16000000 --envs 8 --seed 1 --algo disc --j-targ 1e-2 --replay 4 --out runs_disc_seed1
# Training curves, including each reward term under rewards/ and train/symmetry_loss
tensorboard --logdir runs
```

Checkpoints and their VecNormalize stats are saved every `--save-every` env steps; the final model is `<out>/g1_ppo_final.zip`, also on Ctrl-C. Every policy `.zip` needs its `*_vecnormalize*.pkl` next to it. Each run also writes `<out>/run_info.json` (arguments, env version, git commit) and per-episode Monitor CSVs in `<out>/monitor/`.

Other options: `--lr` / `--lr-end` (linear decay, default 3e-4 to 3e-5), `--batch-size`, `--epochs`, `--sym-coef` (mirror symmetry loss weight), and for DISC `--disc-eps`, `--j-targ`, `--eps-b`, `--replay`.

## Evaluate

```bash
# Standard benchmark metrics (table, plus mean/spread over several policies; --json to save)
python benchmark_eval.py --model ../mujoco_menagerie/unitree_g1/scene.xml --policies checkpoints/g1_ppo_seed1_16M.zip checkpoints/g1_ppo_seed2_16M.zip
# Video without a viewer window (azimuth 90 = side, 180 = front)
python render_video.py --model ../mujoco_menagerie/unitree_g1/scene.xml --policy checkpoints/g1_ppo_seed1_16M.zip --command 0.5 --azimuth 180
# Interactive viewer (macOS needs mjpython)
mjpython evaluate.py --model ../mujoco_menagerie/unitree_g1/scene.xml --policy checkpoints/g1_ppo_seed1_16M.zip --command 0.5
```

`benchmark_eval.py` runs each policy for 3 episodes × 20 s per command (0, 0.5 and 1.0 m/s; reset seeds 0–2) with deterministic actions, skips the first 2 s for steady-state metrics, and reports falls; speed and its error; phase match; touchdowns per gait cycle (1.0 = one clean touchdown per step); stance slip; landing speed; foot roll and flat (4-point) contact; landing distance from the body's centerline per foot; crossovers; step length vs target; closest swing-foot approach to the centerline; pelvis roll; the forward-speed surge around touchdowns; arm swing back/forward, peak arm speed and elbow left/right difference; and, at 0 m/s, both-feet-down share, foot lift and drift.

## Environment (v1)

* **Robot and control:** Menagerie G1, 50 Hz policy (0.005 s physics × 4). Actions are PD position targets = default pose + action × range.
* **Joints:** 23 controlled by default (`--joints legs_waist_arms`): 12 leg joints, waist yaw/roll/pitch, shoulder pitch/roll/yaw and elbow per arm; the wrists hold their pose. `legs` (12), `legs_waist` (15) and `full` (29) also exist.
* **Default pose:** bent knees (hip pitch −0.3, knee +0.6, ankle pitch −0.3 rad; pelvis 0.76 m), because Menagerie's straight-legged keyframe let policies lock their knees. Arms hang (shoulder pitch 0, elbow 0.5 rad) with soft actuators (kp 50, kv 6 instead of kp 500) so they swing like pendulums. Elbow convention: 0 rad is bent ~82°, larger is straighter.
* **Action ranges (rad, below/above default):** hip pitch 0.7/0.8, knee 0.5/0.9, ankle roll ±0.26 (its joint limit), other leg joints ±0.5; waist yaw ±0.4, roll/pitch ±0.25; shoulder pitch ±0.6, roll/yaw ±0.2; elbow 0.5/1.3.
* **Commands:** forward speed, sampled per episode from 0–1.2 m/s, 30% of episodes standing (0 m/s). The speed ramps in over 0.5 s; episodes start in double support with either leg leading.
* **Gait clock:** period 1.1 s at ≤ 0.3 m/s down to 0.8 s at ≥ 1.0 m/s (slower, longer steps at low speed). Leg signals `left = sin(2π·phase)`, `right = −sin(2π·phase)`; a leg is in stance while its signal is above −0.15. Zero (both in stance) when standing.
* **Observations (96):** body-frame linear and angular velocity, gravity direction, command, heading error, sin/cos of each leg's phase, all 29 joint positions and velocities, previous action. Normalized by `VecNormalize`.
* **Rewards,** grouped by purpose (exact weights in `G1Env.step`):
  * Task: velocity tracking (scaled by how well the feet follow the clock, so a shuffle earns ~30%), yaw rate, heading of pelvis and torso, alive bonus; falling ends the episode at −200.
  * Gait: feet on the clock (stance/swing), swing foot height (sin² profile peaking at 9 cm), swing reach and step length (command × period / 2), air time, flight, left/right symmetry of all joints and of the arms.
  * Feet: flat foot in stance and late swing, slip, landing speed near the ground, bounces, per-foot landing width (7 cm out from the centerline), no crossovers, step width.
  * Posture: knee bend and ~17° stance knee, hip yaw and roll, base height, torso upright and roll rate, pelvis roll, waist, body acceleration, vertical velocity, tilt.
  * Arms: swing as a clock-driven sinusoid centered on vertical (±15° at 0.5 m/s to ±25° at 1.2 m/s), opposite the same-side leg; centered and symmetric; elbow bend coupled to the forward swing.
  * Standing: feet down and still, no drift (weight 2.0).
  * Smoothness: action rate and jerk, joint effort, arm joint speed.
* **Training:** `SymmetricPPO` (`symmetric_ppo.py`) adds a mirror symmetry loss (weight 1.0) to PPO. 8 envs × 1024 steps per update, minibatch 2048, 5 epochs, ELU [512, 256, 128] actor and critic, initial action std ≈ 0.37, learning rate 3e-4 decaying linearly to 3e-5.

## Benchmark v1 results

PPO baseline (`SymmetricPPO`, default settings), trained from scratch for 16M steps, two seeds. Both walk with full 20 s episodes from about 7–8M steps and stay stable to the end. `benchmark_eval.py`, 3 episodes × 20 s per command, **0 falls in 18 episodes**:

| Metric | 0.5 m/s, seed 1 / seed 2 | 1.0 m/s, seed 1 / seed 2 |
|---|---|---|
| Speed | 0.489 / 0.496 m/s | 0.998 / 1.005 m/s |
| Phase match | 98.8% / 99.0% | 97.8% / 97.0% |
| Touchdowns per gait cycle | 1.02 / 1.02 | 1.14 / 1.07 |
| Stance slip (median) | 0.044 / 0.027 m/s | 0.068 / 0.058 m/s |
| Landing speed (median) | 0.20 / 0.14 m/s | 0.37 / 0.16 m/s |
| Stance foot roll | 1.3° / 0.7° | 1.5° / 1.2° |
| Flat 4-point contact | 68% / 71% | 62% / 39% |
| Landing out from centerline, left / right | 6.3 / 6.5 cm, 6.3 / 6.6 cm | 6.7 / 6.6 cm, 6.8 / 6.9 cm |
| Crossovers | 0% / 0% | 0% / 0% |
| Step length (target 0.254 / 0.400 m) | 0.257 / 0.249 m | 0.395 / 0.392 m |
| Swing foot closest to centerline | 6.4 / 6.5 cm | 6.1 / 6.8 cm |
| Max heading error | 3.2° / 4.0° | 4.1° / 8.5° |
| Pelvis roll range | 4.5° / 4.4° | 7.2° / 3.9° |
| Speed surge at touchdown | 12% / 15% | 11% / 8% |
| Arm swing back / forward | 16° / 17°, 17° / 16° | 26° / 25°, 27° / 26° |
| Elbow left/right difference | 0.3° / 0.3° | 0.8° / 0.8° |

Standing (0 m/s), both seeds: both feet down 100% of the time, foot lift 0.16 cm, drift ≤ 2 mm.

Checkpoints: `checkpoints/g1_ppo_seed1_16M.zip` and `checkpoints/g1_ppo_seed2_16M.zip` (with VecNormalize stats). Earlier checkpoints were trained on older env versions and are not included.

## DISC vs PPO (sample efficiency)

`disc_ppo.py` implements DISC (Han & Sung, ICML 2019, "Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning"), following the authors' code: each action dimension's IS ratio is clipped separately, an adaptive IS loss (α_IS, target J_targ) bounds the update, and recent rollouts are reused with GAE-V advantages when their per-dimension IS deviation is below ε_b. It is built on `SymmetricPPO`, so the mirror symmetry loss applies as in the baseline, and it checks out against SB3: on a single on-policy batch its log-probs, values, advantages and targets match SB3's GAE to float precision.

To compare runs, `compare_runs.py` plots episode return and length against env steps (from the Monitor CSVs) and evaluates every checkpoint with `benchmark_eval.py`. It reports the first checkpoint that walks: no falls at 0 / 0.5 / 1.0 m/s, and at 0.5 and 1.0 m/s a speed error ≤ 0.1 m/s, phase match ≥ 95% and heading error ≤ 15°, also at the next checkpoint.

```bash
python compare_runs.py --model ../mujoco_menagerie/unitree_g1/scene.xml --runs runs_seed1 runs_disc_seed1 --out compare.png
```

**Tuning.** With the paper's settings (ε 0.4, J_targ 1e-4, ε_b 0.1, 16 replayed rollouts ≈ the paper's 131k samples), the IS loss held each update to ~1% of PPO's (approx KL 0.0002–0.0005 vs ~0.02–0.045), α_IS sat at its cap of 64, no dimension was ever clipped, and DISC had not learned to walk after 16M steps. A screen of 8M-step runs on seed 1 (only DISC was tuned; PPO keeps the benchmark settings) found:

| DISC setting | Effect |
|---|---|
| J_targ 1e-3 → 1e-2 | Largest single gain; at 1e-2, without replay, it tracks PPO's curve until PPO starts walking |
| 32 gradient steps per epoch (paper) instead of 4 | No gain, slower early |
| Replay of 4 rollouts with J_targ 1e-2 | Best: full-length episodes ~2M steps before PPO |
| J_targ 1e-1, no replay | Fast at first, then unstable over 16M steps (KL ~0.6); its deterministic policy falls |
| J_targ 3e-2 with replay of 4 | Unstable (KL ~0.3) |

**Result** (tuned DISC: `--algo disc --j-targ 1e-2 --replay 4`, other settings as the PPO baseline; 16M steps each):

| | PPO seed 1 | DISC seed 1 | PPO seed 2 | DISC seed 2 |
|---|---|---|---|---|
| First walking checkpoint | 10M | **8M** | 11M | **10M** |
| Episode return at 6M / 8M / 10M | −204 / 1306 / 3270 | 1095 / 4055 / 5254 | −259 / 1689 / 3895 | 1577 / 3682 / 5110 |
| Episode return at 16M | 5811 | 6134 | 5977 | 6339 |
| Samples with zero policy gradient (5th–95th percentile of updates) | 13–24% | ~0% | 13–24% | ~0% |
| Wall-clock, 16M steps (two runs in parallel) | 3.6 h | 4.9 h | 3.4 h | 4.7 h |

On both seeds DISC reaches each return level ~2M steps before PPO and walks 1–2M steps sooner. Seed 2 was not used for tuning. Two seeds is a small sample, and training is not bit-for-bit reproducible: two seed-1 runs of the same setting differed by ~2× in return at 8M.

At 16M, `benchmark_eval.py` shows 0 falls for all four, and equal standing, phase match and foot placement, but a less polished DISC gait:

| Metric | PPO, seeds 1 / 2 | DISC, seeds 1 / 2 |
|---|---|---|
| Speed error at 1.0 m/s | 0.003 / 0.005 m/s | 0.032 / 0.053 m/s |
| Flat contact at 0.5 m/s | 71% / 72% | 55% / 55% |
| Touchdowns per gait cycle at 1.0 m/s | 1.10 / 1.07 | 1.25 / 1.32 |
| Pelvis roll range at 0.5 m/s | 4.1° / 4.4° | 9.7° / 8.4° |

DISC's action std falls to ~0.05 (PPO ~0.11), i.e. it stops exploring earlier; a higher entropy coefficient for DISC is the next thing to try. The DISC checkpoints were not added to the repo.

## Known limitations

* A forward-speed surge of ~10–15% remains around each touchdown.
* Flat contact at 1.0 m/s varies between seeds (39–62%).
* The env is flat ground only, with no pushes or randomized dynamics, so robustness to disturbances is untested.

`backup_original/` holds the starter code this project began from.
