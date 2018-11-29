import gym
from gym import spaces
from collections import deque
import numpy as np
from gym.wrappers import TimeLimit


class DifferentialActions(gym.Wrapper):
    old_action = None
    alpha = 0.1
    
    def step(self, action):
        da = self.alpha * action
        self.old_action = da if self.old_action is None else (1-self.alpha) * self.old_action + da
        return self.env.step(self.old_action)


class ConcatVisualUnity(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
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
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], nb_channel))

    def reset(self):
        ob = self.env.reset()
        ob, reward, done, info = self.step(self.action_space.sample())
        return np.concatenate(info["brain_info"].visual_observations, axis=3).squeeze(0)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return np.concatenate(info["brain_info"].visual_observations, axis=3).squeeze(0), reward, done, info


class VideoSaver(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._env = self.env._env
        self.video_buffer = deque(maxlen=10000)

    def reset(self):
        self.video_buffer.clear()
        ob = self.env.reset()
        return ob

    def save_video(self, path="/tmp/gif_avenue.gif"):
        import imageio
        imageio.mimsave(path, self.video_buffer)
        os.chmod(path, 0o777)  # TODO: is that really necessary?

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.video_buffer.append((ob * 255).astype(np.uint8))
        return ob, reward, done, info