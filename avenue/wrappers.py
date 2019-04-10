import gym
from gym import spaces
from collections import deque
import numpy as np
import os


class DifferentialActions(gym.ObservationWrapper):
    action = None

    def __init__(self, env, alpha=0.2):
        super().__init__(env)
        self.alpha = alpha
        low = np.array((*env.observation_space.low, *env.action_space.low))
        high = np.array((*env.observation_space.high, *env.action_space.high))
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def observation(self, m):
        return np.concatenate((m, self.action))

    def reset(self, **kwargs):
        self.action = np.zeros(self.action_space.shape, np.float32)
        return super().reset(**kwargs)

    def step(self, action):
        da = self.alpha * np.asarray(action, dtype=np.float32)
        # self.action = (1-self.alpha) * self.action + da
        action = self.action + da
        self.action = np.clip(action, -1, 1)
        return super().step(self.action)


class DifferentialActionsVisual(DifferentialActions):

    def __init__(self, env, alpha=0.2):
        self.alpha = alpha
        super(gym.ObservationWrapper, self).__init__(env)
        vsp = self.env.observation_space.spaces['vector']
        low = np.array((*vsp.low, *env.action_space.low))
        high = np.array((*vsp.high, *env.action_space.high))
        self.observation_space = spaces.Dict(dict(
            self.env.observation_space.spaces,
            vector=spaces.Box(low, high, dtype=np.float32)
        ))

    def observation(self, m):
        return dict(m, vector=np.concatenate((m['vector'], self.action)))


class BrakeDiscreteControl(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(2)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        return ob["visual"]

    def step(self, action):

        new_action = np.zeros(3, np.float32)
        if action  == 1:
            new_action[2] = 1
        elif action == 0:
            new_action[2] = 0
        ob, reward, done, info = super().step(new_action)
        return ob["visual"], reward, done, info


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

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        ob, reward, done, info = self.step(self.action_space.sample())
        return np.concatenate(info["brain_info"].visual_observations, axis=3).squeeze(0)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return np.concatenate(info["brain_info"].visual_observations, axis=3).squeeze(0), reward, done, info


class VideoSaver(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._env = self.env
        self.video_buffer = deque(maxlen=10000)

    def reset(self, **kwargs):
        self.video_buffer.clear()
        ob = self.env.reset(**kwargs)
        return ob

    def save_video(self, path="/tmp/gif_avenue.gif"):
        import imageio
        imageio.mimsave(path, self.video_buffer)
        os.chmod(path, 0o777)  # TODO: is that really necessary?

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.video_buffer.append((ob["visual"]))
        return ob, reward, done, info


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.spaces["visual"].shape
        new_shape = (obs_shape[2], obs_shape[1], obs_shape[0])
        self.observation_space = spaces.Box(low=0, high=1, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)/255


class MaxStep(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(MaxStep, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            return True

        return False

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._past_limit():
            if self.metadata.get('semantics.autoreset'):
                _ = self.reset() # automatically reset the env
            done = True

        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
