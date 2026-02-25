# SuperMario Gym AI

This repository contains a Reinforcement Learning agent trained to play Super Mario Bros using Proximal Policy Optimization (PPO) from `stable-baselines3`.

## Installation

This project uses Poetry for dependency management. To install the dependencies:

```bash
poetry install
```

Alternatively, you can install the dependencies using pip:

```bash
pip install "gymnasium>=0.29.1" "stable-baselines3>=2.3.0" "gym-super-mario-bros==7.4.0" "nes-py==8.2.1" "shimmy>=1.3.0" "opencv-python>=4.8.0" "tensorboard>=2.15.0" "numpy<2.0.0"
```

## Usage

The project provides a command-line interface to train and play the agent.

### Training

To train the agent, run:

```bash
python -m supermario_gym_ai.main train --total_timesteps 1000000 --n_envs 4
```

Arguments:
- `--total_timesteps`: Total number of timesteps to train (default: 1000000).
- `--n_envs`: Number of parallel environments to use (default: 4).
- `--checkpoint_dir`: Directory to save checkpoints (default: `./checkpoints/`).
- `--log_dir`: Directory to save TensorBoard logs (default: `./logs/`).
- `--seed`: Random seed (default: 42).

### Playing

To watch the trained agent play:

```bash
python -m supermario_gym_ai.main play --model_path ./checkpoints/ppo_mario_final
```

Arguments:
- `--model_path`: Path to the trained model (zip file).

## Environment

The environment is based on `gym-super-mario-bros`. It is wrapped to be compatible with `gymnasium` and `stable-baselines3`.
Preprocessing includes:
- Skipping every 4th frame.
- Resizing to 84x84 grayscale.
- Stacking 4 consecutive frames.
- Normalizing observations (0-255).

## Monitoring

You can monitor the training progress using TensorBoard:

```bash
tensorboard --logdir ./logs/
```
