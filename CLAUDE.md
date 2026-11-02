# CLAUDE.md

Velocity-tracking walking for the Unitree G1 humanoid in standard CPU MuJoCo, trained with PPO (Stable-Baselines3) on an M2 Mac. The env (benchmark v1) targets a human-like gait and is meant for benchmarking RL algorithms. README.md has the full env/reward description and the v1 results.

## Benchmark v1 is frozen

- `ENV_VERSION = "v1"` in `g1_env.py`. Any change to observations, actions (default pose, action ranges, joint set, gains), rewards, dynamics or episode setup must bump `ENV_VERSION` and be re-validated from scratch. Never compare results across versions.
- v1 baseline: `SymmetricPPO`, default `train.py` settings, 16M steps from scratch. Seeds 1 and 2 both walk from ~7–8M steps and reach the same gait (`checkpoints/g1_ppo_seed{1,2}_16M.zip`).
- Report results with `benchmark_eval.py` (several seeds per algorithm, same step budget), not with `ep_rew_mean`.

## Layout

- `g1_env.py`: the Gymnasium env (`G1Env`): control, observations, gait clock, all reward terms, and `mirror_maps()`.
- `train.py`: `SymmetricPPO` + `VecNormalize`, linear learning-rate decay (`--lr`, to 3e-5), checkpoints, `--resume`, per-term reward logging (`rewards/<term>` in TensorBoard). Also defines `vecnormalize_path()`, used by the other scripts.
- `symmetric_ppo.py`: `SymmetricPPO`, PPO plus a left/right mirror symmetry loss (`--sym-coef`, default 1.0; 0 = plain PPO). Its `train()` is copied from SB3 2.9's `PPO.train`, so re-check it if SB3 is upgraded. Mirror maps come from `G1Env.mirror_maps()`, verified exactly against a physically mirrored simulator state; update them if the observation layout changes.
- `benchmark_eval.py`: the standard evaluation (3 × 20 s episodes at 0, 0.5, 1.0 m/s; gait, foot, placement, arm and standing metrics; mean/spread over several policies; `--json`).
- `render_video.py`: offscreen MP4 of a checkpoint; `evaluate.py`: interactive viewer (macOS needs `mjpython`).
- `checkpoints/`: saved policies. `videos/`: renders. `backup_original/`: the starter code; don't edit.
- MuJoCo Menagerie lives outside the repo at `../mujoco_menagerie`; the model is `../mujoco_menagerie/unitree_g1/scene.xml`.

## Commands

```bash
python train.py --model ../mujoco_menagerie/unitree_g1/scene.xml --steps 16000000 --envs 8 --seed <n> --out <dir>
python train.py --model ../mujoco_menagerie/unitree_g1/scene.xml --steps 2000000 --lr 9e-5 --resume <zip>
python benchmark_eval.py --model ../mujoco_menagerie/unitree_g1/scene.xml --policies <zip> [<zip> ...] --json <file>
python render_video.py --model ../mujoco_menagerie/unitree_g1/scene.xml --policy <zip> --command 0.5 --azimuth 180 --out videos/<name>.mp4
```

Throughput is ~3k env steps/s with 8 envs (16M steps ≈ 1.5 h). Run anything over ~2 minutes in the background, wrapped in `caffeinate -i` for long runs. Put experiment checkpoints outside the project or in a throwaway `--out` dir; copy only keepers into `checkpoints/`.

## Gotchas

- **Each policy `.zip` needs its VecNormalize stats file next to it.** `vecnormalize_path()` maps `x.zip` → `x_vecnormalize.pkl` and SB3 checkpoints `g1_ppo_N_steps.zip` → `g1_ppo_vecnormalize_N_steps.pkl`. Copy both when saving a checkpoint.
- Changing the observation size, the action meaning (default pose, action ranges, arm gains) or the joint set makes old checkpoints unusable; reward-only changes can `--resume`. `ACTION_RANGE` applies to every joint set.
- On `--resume`, SB3 keeps the saved learning rate and measures progress against the total step count; `train.py` overrides the schedule and rescales it to the resumed steps.
- **Evaluation windows must stay inside one episode (1000 steps).** The VecEnv auto-resets at step 1000; a rollout longer than that records the reset as a fake jump. That once produced a phantom "arm snap" in the analysis.
- Free-joint velocities: `qvel[0:3]` is world frame (rotate by `R.T`), `qvel[3:6]` is already body frame (don't rotate).
- Menagerie G1: actuator *i* drives hinge joint *i + 1*; the `stand` keyframe has straight legs, so the env builds its own bent-knee default (`BENT_LEG_POSE`). Elbow joint: 0 rad = bent ~82°, larger = straighter. Shoulder pitch 0 holds the upper arm ~5° forward (`ARM_VERTICAL` corrects for it). Left and right roll/yaw joints share an axis, so mirroring flips their sign.
- Feet = four contact spheres on each `*_ankle_roll_link`; foot positions come from the `left_foot` / `right_foot` sites. The floor geom is `floor`.
- The velocity reward is body-frame, so a policy can satisfy it while walking in circles; heading must be checked separately.
- PPO tends to settle into lopsided gaits; symmetry needs the mirror loss, not just reward terms. A long chain of resumes under changing rewards acts as an implicit curriculum and left old habits in the policy; final quality should come from a from-scratch run.

## Evaluating a change

Run `benchmark_eval.py` and render front and side videos, and look at them: circling, marching in place, locked knees, catwalk feet and asymmetric swing paths only showed up in video or detailed metrics, not in reward curves.

## Working style

- Change one or two reward/env things per experiment, test with a 2M-step resume, and confirm with a from-scratch run before freezing anything.
- Keep README.md's env description and results in sync with `g1_env.py`.
- Don't offer a commit after every experiment; commit when asked.
- Git remote: github.com/TeodorPenchev04/g1-mujoco (main); the earlier approach is tagged `old-approach`. Only commit checkpoints that match the current env version; `checkpoints/*` is gitignored, so add a keeper and its stats file with `git add -f`. Videos, runs/ and .venv are gitignored.
