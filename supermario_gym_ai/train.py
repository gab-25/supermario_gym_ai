import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from supermario_gym_ai.utils import create_vec_env

def train(args):
    # Hyperparameters
    learning_rate = 2.5e-4
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.1
    ent_coef = 0.01

    # Environment
    print(f"Creating {args.n_envs} environments...")
    env = create_vec_env(n_envs=args.n_envs, seed=args.seed)

    # Model
    print("Initializing PPO model...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // args.n_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ppo_mario"
    )

    # Train
    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, "ppo_mario_final")
    model.save(final_model_path)
    print(f"Training finished! Model saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on Super Mario Bros")
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="Directory to save tensorboard logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train(args)
