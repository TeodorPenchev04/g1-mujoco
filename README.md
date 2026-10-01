# G1 MuJoCo

A simple reinforcement learning project for training the **Unitree G1 humanoid robot** in **MuJoCo** using **PPO**.

## Current Features

* Unitree G1 model from MuJoCo Menagerie
* Custom Gymnasium environment
* PPO implementation in PyTorch
* GAE and PPO clipping
* Training and evaluation scripts
* Video rendering of trained behavior

## Project Structure

```text
g1-mujoco/
├── algorithms/
│   └── ppo.py
├── envs/
│   └── g1_env.py
├── scripts/
│   ├── test_env.py
│   ├── train_ppo.py
│   ├── evaluate_ppo.py
│   └── render_best_episode.py
├── checkpoints/
└── results/
```

## Install

```bash
pip install mujoco gymnasium torch numpy matplotlib imageio imageio-ffmpeg
```

Clone the G1 model separately:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

Place `mujoco_menagerie` as a sibling of this repository. The environment then
resolves the model at:

```text
../mujoco_menagerie/unitree_g1/scene.xml
```

## Run

Test the environment:

```bash
python -m scripts.test_env
```

Train PPO:

```bash
python -m scripts.train_ppo
```

Evaluate:

```bash
python -m scripts.evaluate_ppo
```

Render the best episode:

```bash
python -m scripts.render_best_episode
```

The video is saved to:

```text
results/best_episode.mp4
```

## Goal

The current goal is to train the G1 to achieve stable forward walking using PPO.
