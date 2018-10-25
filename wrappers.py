import gym
import numpy as np
from gym import spaces

class ConcatVisualUnity(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._env = self.env._env
        nb_channel = 0
        brain_name = self.env._env.external_brain_names[0]
        brain = self.env._env.brains[brain_name]
        for i in range(0, len(brain.camera_resolutions)):
            if(brain.camera_resolutions[i]["blackAndWhite"]):
                nb_channel += 1
            else:
                nb_channel += 3

        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], nb_channel), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        ob, reward, done, info = self.step(self.action_space.sample())
        return np.concatenate(info["brain_info"].visual_observations, axis=3).squeeze(0)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return np.concatenate(info["brain_info"].visual_observations, axis=3).squeeze(0), reward, done, info

class ConcatContinuousState(gym.Wrapper):
    def __init__(self, env):
        self._env = self.env._env
        gym.Wrapper.__init__(self, env)
        brain_name = self.env._env.external_brain_names[0]
        brain = self.env._env.brains[brain_name]
        self.nb_vector_observation = brain.vector_observation_space_size
        shp = env.observation_space.shape
        self.nb_dim = shp[0] // self.nb_vector_observation
        print(self.nb_dim)
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] + 1), dtype=np.uint8)
