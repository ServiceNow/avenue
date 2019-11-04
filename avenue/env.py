from mlagents.gym_unity.envs.unity_env import UnityEnv as GymUnityEnv
import os
import platform
import zipfile
import gdown
import random
import gym
from gym import spaces
import numpy as np
from .util import ensure_executable, compute_assed_id, compute_asset_path
from avenue.avenue_states import *
import math
from enum import Enum


class ControllerType(Enum):
    CAR = 1
    DRONE = 2

class UnityEnv(gym.Wrapper):
    """
        Base class for avenue gym wrapper and automatic download.
    """
    host_ids: dict
    visual: bool = False
    asset_name: str
    ctrl_type: ControllerType

    def __init__(self, config=None):
        self.config = config
        path = self.get_assets()
        env = GymUnityEnv(environment_filename=path, use_visual=self.visual, worker_id=random.randint(1000, 20000))
        env.reset(config)
        super().__init__(env)

    @classmethod
    def get_assets(cls):
        assert cls.asset_name
        asset_id = compute_assed_id(cls.asset_name, platform.system())
        path = compute_asset_path(asset_id)
        run_path = os.path.join(path, asset_id)
        if os.path.isdir(path):
            return run_path

        print('downloading', path + '.zip')
        system = platform.system().lower()
        if system not in cls.host_ids:
            raise KeyError("There are no assets available for {} on {}".format(cls.asset_name, system))
        id = cls.host_ids[system]
        gdown.download("https://drive.google.com/uc?id="+id, path + ".zip", False)
        zip_ref = zipfile.ZipFile(path + '.zip', 'r')
        print('unpacking ...')
        zip_ref.extractall(path)
        zip_ref.close()
        assert os.path.isdir(path)
        os.remove(path + '.zip')
        print("Unpacked !")
        ensure_executable(run_path)
        return run_path

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, a):
        return self.env.step(a)


class BaseAvenue(UnityEnv):
    """
        Avenue env with vector state return, rgb and segmentation.
    """
    visual = True
    drone = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        state_dims = globals()[self.vector_state_class]()
        self.state_idx = [sum(state_dims[:i + 1]) for i in range(len(state_dims) - 1)]
        self.env.reset()

        #import pdb; pdb.set_trace()

        # Get the info to find the resolutions and number of camera
        _, _, _, info = self.env.step(self.env.action_space.sample())

        # Since we change the resolution in the config we need to find the new visual observations spaces (rgb,
        # segmentation).
        self.observation_space = spaces.Dict(dict(
            {k : spaces.Box(low= -100, high=100,shape=(v,), dtype=np.float32) for k,v in state_dims._asdict().items()},
            rgb=spaces.Box(0, 255, info["brain_info"].visual_observations[0].shape[1:], np.uint8),
        ))

    def step(self, a):
        _, r, d, info = self.env.step(a)

        vec_obs, = info['brain_info'].vector_observations
        vec_obs = np.asarray(vec_obs, dtype=np.float32)
        self.state = globals()[self.vector_state_class](*np.split(vec_obs, self.state_idx))
        reward = self.compute_reward(self.state, r, d)
        done = self.compute_terminal(self.state, r, d)
        info = dict(info, reset=False, avenue_state=self.state)  # reset=False, i.e. all dones are true terminals
        rgb = (255 * info["brain_info"].visual_observations[0].squeeze(0)).astype(np.uint8)
        m = dict(self.state._asdict(), rgb=rgb)
        return m, reward, done, info

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        m, _, _, info = self.step(self.env.action_space.sample())  # we need to step to get info
        return m

    def compute_terminal(self, s, r, d):
        return d

    def compute_reward(self, s, r, d):
        return r
