import argparse
import sys
from supermario_gym_ai.train import train
from supermario_gym_ai.play import play

def run():
    parser = argparse.ArgumentParser(description="Super Mario Bros PPO Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timesteps to train")
    train_parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    train_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/", help="Directory to save checkpoints")
    train_parser.add_argument("--log_dir", type=str, default="./logs/", help="Directory to save tensorboard logs")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Play command
    play_parser = subparsers.add_parser("play", help="Play using a trained agent")
    play_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model zip file")

    # Parse arguments
    # If no arguments are provided, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "play":
        play(args)

if __name__ == "__main__":
    run()
