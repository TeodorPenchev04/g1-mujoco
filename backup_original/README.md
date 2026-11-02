# G1 MuJoCo PPO baseline — Mac M2

This is a **starter training baseline**, not a pre-trained walking policy. Uses standard CPU MuJoCo, Stable-Baselines3 PPO, and official MuJoCo Menagerie G1 (29 DOF) MJCF. The 29-DOF choice follows Menagerie's current `unitree_g1/g1.xml`, which provides built-in position actuators and a `stand` keyframe. The environment uses the TA's Isaac Lab rewards conceptually, not as an identical reproduction.

## Install (macOS)

```bash
# Install Python 3.11 first, if needed.
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# From repository parent dir (separate from this zip):
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
```

Run from the unzipped `g1_mujoco_ppo` directory. Adjust path as needed:

```bash
python train.py --model ../mujoco_menagerie/unitree_g1/scene.xml --steps 200000 --envs 4
python evaluate.py --model ../mujoco_menagerie/unitree_g1/scene.xml
```

### Checks and caveats

* `train.py` saves `g1_ppo_baseline.zip` at the end, including if training is interrupted with Ctrl-C. `evaluate.py` opens the MuJoCo GUI; run it on your Mac desktop, not headless Colab.
* This is intentionally an early-stage baseline. It **has not been verified to learn a stable gait**, and learning may need several million steps and reward/controller tuning. Start with 200k to check the pipeline, not convergence.
* The official Menagerie model has **29 position actuators**, rather than the 23-DOF minimal G1 from Isaac Lab, and the provided TA reward weights cannot be transferred exactly because simulation units and normalization differ.
* This implementation uses a *single* fixed command speed 0.5 m/s, basic orientation/torque/position/action penalties, and no foot contact term yet. The next stage should introduce robust foot contacts, air-time, slip, and a command curriculum.
* Train headlessly (no rendering) for speed. Try `--envs 1` versus `--envs 4` to benchmark on the M2. If `mujoco.viewer` requires macOS's `mjpython` launcher, replace `python evaluate.py` with `mjpython evaluate.py`.
