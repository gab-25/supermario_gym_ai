import argparse
import time
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from supermario_gym_ai.utils import make_mario_env
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def play(args):
    # Check if model exists
    if not os.path.exists(args.model_path):
        if not os.path.exists(args.model_path + ".zip"):
            print(f"Model file not found: {args.model_path}")
            return

    print(f"Loading model from {args.model_path}...")

    # Need to recreate the environment setup exactly as during training
    # PPO CnnPolicy expects stacked frames if trained with them

    # Create environment
    env = make_mario_env(env_id="SuperMarioBros-v0", movement=SIMPLE_MOVEMENT)

    # We need to wrap it in DummyVecEnv and VecFrameStack to match training
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    model = PPO.load(args.model_path, env=env)

    obs = env.reset()

    print("Starting playback...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.01) # Slow down slightly to watch

            if done:
                 # VecEnv automatically resets the environment
                 pass

    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Super Mario Bros using a trained PPO agent")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model zip file")
    args = parser.parse_args()

    play(args)
