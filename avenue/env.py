from avenue.gym_unity.envs import UnityEnv as GymUnityEnv
import os
import platform
import zipfile
import gdown
import random
import gym
from gym import spaces
import numpy as np
from .util import ensure_executable, asset_id, asset_path
from avenue.avenue_states import *
import math

class UnityEnv(gym.Wrapper):
    """
        Base class for avenue gym wrapper and automatic download.
    """
    host_ids: dict
    visual: bool = False
    asset_name: str

    def __init__(self, config=None, seed=0):
        system = platform.system().lower()

        # Check if the binary is missing, in this can download it.
        if self.asset_name is not None:

            id_asset = asset_id(self.asset_name, platform.system())
            path_asset = asset_path(id_asset)
            if not os.path.isdir(path_asset):
                self.download_assets(path_asset)
        else:
            raise KeyError("There are no assets available for {} on {}".format(self.asset_name, system))

        path = os.path.join(path_asset, id_asset)
        ensure_executable(path)
        env = GymUnityEnv(environment_filename=path, use_visual=self.visual, worker_id=random.randint(0, 10000))
        env.reset(config)
        super().__init__(env)

    def download_assets(self, path):
        print('downloading', path + '.zip')
        system = platform.system().lower()
        if system not in self.host_ids:
            raise KeyError("There are no assets available for {} on {}".format(self.asset_name, system))
        id = self.host_ids[system]
        gdown.download("https://drive.google.com/uc?id="+id, path + ".zip", False)
        zip_ref = zipfile.ZipFile(path + '.zip', 'r')
        print('unpacking ...')
        zip_ref.extractall(path)
        zip_ref.close()
        assert os.path.isdir(path)
        os.remove(path + '.zip')
        print("Unpacked !")

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, a):
        return self.env.step(a)

# WARNING: This is obsolete we now use AvenueBase
class AvenueEnv(UnityEnv):
    """
        Avenue env with vector state return.
    """
    StateType = AvenueState
    state: AvenueState = None

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        state_dims = globals()[self.vector_state_class]()
        self.state_idx = [sum(state_dims[:i+1]) for i in range(len(state_dims)-1)]
        self.env.reset()

        # Get the info to find the resolutions and number of camera
        _, _, _, info = self.env.step(self.env.action_space.sample())


        # Since we change the resolution in the config we need to find the new visual observations spaces (rgb,
        # segmentation).
        self.observation_space = spaces.Dict(dict(
            vector=spaces.Box(-1, 1, (sum(state_dims),)),
            visual=spaces.Box(0, 255, info["brain_info"].visual_observations[0].shape[1:], np.uint8),
            segmentation=spaces.Box(0, 255, (info["brain_info"].visual_observations[0].shape[1],
                                             info["brain_info"].visual_observations[0].shape[2],
                                             1))
        ))

        self.last_horizontal_error = 0

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        m, _, _, info = self.step(self.action_space.sample())  # we need to step to get info

        return m

    def step(self, a):
        """
        :returns (observation, reward, done, info)
        info contains 'reset' which is 1 if 'done' should be ignored by the agent / learning algorithm
        """
        _, r, d, info = self.env.step(a)

        vec_obs, = info['brain_info'].vector_observations
        vec_obs = np.asarray(vec_obs, dtype=np.float32)

        self.state = globals()[self.vector_state_class](*np.split(vec_obs, self.state_idx))
        reward = self.compute_reward(self.state, r, d)
        done = self.compute_terminal(self.state, r, d)

        info = dict(info, reset=False, avenue_state=self.state)  # reset=False, i.e. all dones are true terminals
        return vec_obs, reward, done, info

    def compute_terminal(self, s, r, d):
        return d

    def compute_reward(self, s, r, d):
        return r

    def pid_action(self, s):
        theta = (s["vector"].angle_to_next_waypoint_in_degrees / 360 * 2 * np.pi)
        horizontal_error =  s["vector"].horizontal_force - theta
        horizontal_force = 0.76 * horizontal_error  +  0.5 * (horizontal_error - self.last_horizontal_error)
        vertical_force = random.random()
        brake_force = 0
        self.last_horizontal_error = horizontal_error
        return np.array([vertical_force, horizontal_force, brake_force])


class AllStatesAvenueEnv(AvenueEnv):
    """
        Avenue env with vector state return, rgb and segmentation.
    """
    visual = True

    def step(self, a):
        s, r, d, info = super().step(a)
        s = globals()[self.vector_state_class](*np.split(s, self.state_idx))
        # Put each channel as a key of a dictionnary for visual observations
        visual_obs = dict(rgb=info["brain_info"].visual_observations[0].squeeze(0),
                          segmentation=info["brain_info"].visual_observations[1].squeeze(0))
        visual_obs["rgb"] = (255 * visual_obs["rgb"]).astype(np.uint8)
        visual_obs["segmentation"] = (255 * visual_obs["segmentation"]).astype(np.uint8)
        m = dict(vector=s, visual=visual_obs)
        return m, r, d, info


class BaseAvenue(UnityEnv):
    """
        Avenue env with vector state return, rgb and segmentation.
    """
    visual = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        state_dims = globals()[self.vector_state_class]()
        self.state_idx = [sum(state_dims[:i + 1]) for i in range(len(state_dims) - 1)]
        self.env.reset()

        # Get the info to find the resolutions and number of camera
        _, _, _, info = self.env.step(self.env.action_space.sample())

        # Since we change the resolution in the config we need to find the new visual observations spaces (rgb,
        # segmentation).
        self.observation_space = spaces.Dict(dict(
            {k : spaces.Box(low= -100, high=100,shape=(v,), dtype=np.float32) for k,v in state_dims._asdict().items()},
            rgb=spaces.Box(0, 255, info["brain_info"].visual_observations[0].shape[1:], np.uint8),
            segmentation=spaces.Box(0, 255, (info["brain_info"].visual_observations[0].shape[1],
                                             info["brain_info"].visual_observations[0].shape[2],
                                             1))
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
        segmentation = (255 * info["brain_info"].visual_observations[1].squeeze(0)).astype(np.uint8)
        m = dict(self.state._asdict(), rgb=rgb, segmentation=segmentation)
        return m, reward, done, info

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        m, _, _, info = self.step(self.action_space.sample())  # we need to step to get info
        return m

    def compute_terminal(self, s, r, d):
        return d

    def compute_reward(self, s, r, d):
        theta = math.radians(s.angle_to_next_waypoint_in_degrees[0])
        velocity_magnitude = s.velocity_magnitude[0]
        top_speed = s.top_speed[0]
        r = -math.fabs(1 - (math.cos(theta) * velocity_magnitude/top_speed)) + 1
        return r
