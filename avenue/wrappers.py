import gym
from collections import deque
import numpy as np
import os


class RandomizedEnv(gym.Wrapper):
    def __init__(self, env_fn, n=10000):
        self.n = n
        self.env_fn = env_fn
        self.epsiodes = 0
        self.steps_since_config = 0
        super().__init__(env_fn())

    def reset(self, **kwargs):
        self.epsiodes += 1
        if self.steps_since_config > self.n:
            self.env.close()
            self.env = self.env_fn()
            self.steps_since_config = 0

        return self.env.reset()

    def step(self, action):
        self.steps_since_config += 1
        return super().step(action)


class ConcatComplex(gym.ObservationWrapper):
    def __init__(self, env, observation_dict):
        """observation_dict is a dict of lists of keys to be concatenated
        """
        super().__init__(env)
        self.observation_dict = observation_dict
        spaces = {k: concat_spaces_from_dict(env.observation_space.spaces, v) for k, v in observation_dict.items()}
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, state):
        return {k: np.concatenate([x for name, x in state.items() if name in v], axis=-1) for k, v in self.observation_dict.items()}


class DictToTupleWrapper(gym.ObservationWrapper):
    def __init__(self, env, *args):
        """Each argument contains a key or tuple of keys which will be concatenated"""
        super().__init__(env)
        self.keys = [k if isinstance(k, (list, tuple)) else (k,) for k in args]
        assert isinstance(env.observation_space, gym.spaces.Dict)
        spaces = [concat_spaces_from_dict(env.observation_space.spaces, k) for k in self.keys]
        self.observation_space = gym.spaces.Tuple(spaces)

    def observation(self, state):
        return tuple(np.concatenate([x for name, x in state.items() if name in k], axis=-1) for k in self.keys)


def concat_spaces_from_dict(spaces: dict, keys):
    """Used in ConcatComplex and DictToTupleWrapper"""
    assert all(name in spaces.keys() for name in keys), f"All values in {keys} must be in {spaces.keys()}"
    shapes, lows, highs, dtypes = zip(*[(box.shape, box.low, box.high, box.dtype) for name, box in spaces.items() if name in keys])
    low = lows[0].flatten()[0]
    high = highs[0].flatten()[0]
    shapes_dim = [s[:-1] for s in shapes]
    assert all(s == shapes_dim[0] for s in shapes_dim), "All dimensions must match!"
    assert all(all(s.flatten() == low) for s in lows), "All low must match!"
    assert all(all(s.flatten() == high) for s in highs), "All high must match!"
    assert all(s == dtypes[0] for s in dtypes), f"All dtypes must match!{dtypes}"

    new_shape = shapes_dim[0] + (sum(s[-1] for s in shapes),)
    return gym.spaces.Box(low=low, high=high, dtype=dtypes[0], shape=new_shape)


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
        self.action = (1-self.alpha) * self.action + da
        self.action = np.clip(self.action, -1, 1)
        return super().step(self.action)


class ReduceActionSpace(gym.Wrapper):
    def __init__(self, env, action_dim):
        super(ReduceActionSpace, self).__init__(env)
        self.old_action_space = env.action_space
        self.action_space = gym.spaces.Box(-1, 1, (action_dim,))

    def step(self, action):
        action_new_shape = np.zeros(self.old_action_space.shape[0])
        action_new_shape[0:self.action_space.shape[0]] = action
        return self.env.step(action_new_shape)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


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
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)/255