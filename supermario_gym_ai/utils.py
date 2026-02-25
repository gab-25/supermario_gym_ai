import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
import numpy as np
import cv2
import gym_super_mario_bros
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

class MarioGymnasiumAdapter(gym.Env):
    """
    Adapter to convert the gym-super-mario-bros environment (gym<0.26)
    to a gymnasium environment.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = Discrete(env.action_space.n)
        self.observation_space = Box(low=0, high=255, shape=env.observation_space.shape, dtype=np.uint8)
        self.metadata = env.metadata
        self.render_mode = "human" # Default for nes-py

    def reset(self, seed=None, options=None):
        if seed is not None:
            try:
                self.env.seed(seed)
            except AttributeError:
                pass

        # gym-super-mario-bros reset returns only observation
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Convert done to terminated and truncated
        terminated = done
        truncated = False

        # Check if TimeLimit caused the done
        if info.get('TimeLimit.truncated', False):
            truncated = True
            terminated = False

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

class CustomGrayScaleResize(gym.ObservationWrapper):
    """
    Resize the image to 84x84 and convert to grayscale.
    """
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape
        self.observation_space = Box(low=0, high=255, shape=(self.shape[0], self.shape[1], 1), dtype=np.uint8)

    def observation(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, -1)

class SkipFrame(gym.Wrapper):
    """
    Return only every `skip`-th frame.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False

        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

def make_mario_env(env_id="SuperMarioBros-v0", movement=SIMPLE_MOVEMENT):
    """
    Create and wrap the Super Mario Bros environment.
    """
    # Instantiate directly to avoid TimeLimit wrapper from gym.make which causes crash with old-style envs
    if env_id == "SuperMarioBros-v0":
        env = SuperMarioBrosEnv()
    else:
        # Fallback to make but expect issues if not handled
        env = gym_super_mario_bros.make(env_id)

    # Apply JoypadSpace to simplify actions
    env = JoypadSpace(env, movement)

    # Adapt to Gymnasium
    env = MarioGymnasiumAdapter(env)

    # Preprocessing
    env = SkipFrame(env, skip=4)
    env = CustomGrayScaleResize(env, shape=84)

    return env

def create_vec_env(env_id="SuperMarioBros-v0", n_envs=1, seed=None):
    """
    Create a vectorized environment for training.
    """
    # Create a list of lambda functions, each returning a Monitor-wrapped env
    env_fns = []
    for i in range(n_envs):
        def make_env_fn(rank=i):
            env = make_mario_env(env_id=env_id, movement=SIMPLE_MOVEMENT)
            if seed is not None:
                env.reset(seed=seed + rank)
            return Monitor(env)
        env_fns.append(make_env_fn)

    env = DummyVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    return env
