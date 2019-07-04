import gym
from gym import spaces
from collections import deque
import numpy as np
import os


class ConcatComplex(gym.ObservationWrapper):

    def __init__(self, env, observation_dict):
        """

        :param env:
        :param observation_dict: This is a dict of str to a list of strings
        """
        super().__init__(env)
        self.observation_dict = observation_dict
        self.observation_space = spaces.Dict({})
        for k, v in observation_dict.items():
            assert all(name in env.observation_space.spaces.keys() for name in v), \
                f"All values in {v} must be in {env.observation_space.spaces.keys()}"
            shapes, lows, highs, dtypes = zip(*[(box.shape, box.low, box.high, box.dtype) for name, box in env.observation_space.spaces.items() if name in v])
            low = lows[0].flatten()[0]
            high = highs[0].flatten()[0]
            shapes_dim = [s[:-1] for s in shapes]
            assert all(s == shapes_dim[0] for s in shapes_dim), "All dimensions must match!"
            assert all(all(s.flatten() == low) for s in lows), "All low must match!"
            assert all(all(s.flatten() == high) for s in highs), "All high must match!"
            assert all(s == dtypes[0] for s in dtypes), f"All dtypes must match!{dtypes}"

            new_shape = shapes_dim[0] + (sum(s[-1] for s in shapes),)
            self.observation_space.spaces[k] = spaces.Box(low=low, high=high, dtype=dtypes[0],shape= new_shape)

    def observation(self, state):
        return {k: np.concatenate([x for name, x in state.items() if name in v], axis=-1) for k, v in self.observation_dict.items()}


class DifferentialActions(gym.ObservationWrapper):
    action = None

    def __init__(self, env, alpha=0.1, key_to_concat="vector"):
        super().__init__(env)
        self.alpha = alpha
        self.key_to_concat = key_to_concat
        self.observation_space.spaces[self.key_to_concat].shape = \
            (self.observation_space.spaces[self.key_to_concat].shape[0] + self.action_space.shape[0],)

    def observation(self, m):
        m[self.key_to_concat] = np.concatenate((m[self.key_to_concat], self.action))
        return m

    def reset(self, **kwargs):
        self.action = np.zeros(self.action_space.shape, np.float32)
        return super().reset(**kwargs)

    def step(self, action):
        da = self.alpha * np.asarray(action, dtype=np.float32)
        # self.action = (1-self.alpha) * self.action + da
        action = self.action + da
        self.action = np.clip(action, -1, 1)
        return super().step(self.action)

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
