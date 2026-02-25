import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


# --- ADATTATORE ---
class MarioGymnasiumAdapter(gym.Env):
    def __init__(self, env):
        self.env = env
        # Conversione Action Space
        self.action_space = Discrete(env.action_space.n)
        # Conversione Observation Space
        self.observation_space = Box(low=0, high=255, shape=env.observation_space.shape, dtype=np.uint8)
        self.metadata = env.metadata

    def reset(self, seed=None, options=None):
        if seed is not None:
            try:
                self.env.seed(seed)
            except AttributeError:
                pass  # Alcuni env vecchi non hanno seed
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# --- PREPROCESSING ---
class CustomGrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape
        self.observation_space = Box(low=0, high=255, shape=(self.shape[0], self.shape[1], 1), dtype=np.uint8)

    def observation(self, observation):
        import cv2

        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, -1)


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)
        return True


def make_env():
    # 1. Crea ambiente originale
    env = gym_super_mario_bros.make("SuperMarioBros-v0")

    # FIX CRITICO: Rimuovi wrapper TimeLimit se usa la nuova API su vecchio env
    # Scendiamo fino a trovare l'ambiente base o rimuoviamo TimeLimit
    while hasattr(env, "env"):
        if "TimeLimit" in str(type(env)):
            # Trovato il colpevole! Lo rimuoviamo prendendo l'env interno
            env = env.env
        else:
            # Continuiamo a scendere (es. se c'è altro wrapper)
            # Ma attenzione a non scendere troppo se JoypadSpace deve essere applicato DOPO.
            # JoypadSpace va applicato all'env NES nudo.
            # Ma gym.make restituisce TimeLimit(NesEnv).
            break

    # Se abbiamo rimosso troppi wrapper, non importa, Mario ha i suoi limiti interni di tempo.

    # 2. Applica JoypadSpace
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # 3. Adatta a Gymnasium
    env = MarioGymnasiumAdapter(env)

    # 4. Preprocessing
    env = CustomGrayScaleResize(env, shape=84)

    return env


def run():
    CHECKPOINT_DIR = "./train/"
    LOG_DIR = "./logs/"

    # 4. Vettorializzazione e Stack
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=1e-6, n_steps=512)

    print("Inizio l'addestramento...")
    model.learn(total_timesteps=100000, callback=callback)

    model.save("mario_final_model")
    print("Modello salvato!")


if __name__ == "__main__":
    run()
