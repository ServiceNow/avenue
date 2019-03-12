from avenue.gym_unity.envs import UnityEnv as GymUnityEnv
import os
import platform
import zipfile
import gdown
import random
import gym
import numpy as np
from gym import spaces
from numpy.linalg import norm
from .util import ensure_executable, namedtuple


class AvenueState(namedtuple):
    waypoint_0 = 2  # TODO: waypoints are relative but should be absolute
    waypoint_1 = 2
    waypoint_2 = 2
    waypoint_3 = 2
    waypoint_4 = 2
    velocity_magnitude = 1
    angle_to_next_waypoint_in_degrees = 1
    horizontal_force = 1
    vertical_force = 1
    velocity = 3
    top_speed = 1
    ground_col = 1
    collide_car = 1
    collide_pedestrian = 1
    position = 3
    forward = 3
    closest_waypoint = 3



class AvenueStateZoom(namedtuple):
    waypoint_0 = 2  # TODO: waypoints are relative but should be absolute
    waypoint_1 = 2
    waypoint_2 = 2
    waypoint_3 = 2
    waypoint_4 = 2
    velocity_magnitude = 1
    angle_to_next_waypoint_in_degrees = 1
    horizontal_force = 1
    vertical_force = 1
    velocity = 3
    top_speed = 1
    ground_col = 1
    collide_car = 1
    collide_pedestrian = 1
    position = 3
    forward = 3
    closest_waypoint = 3
    object_distance = 1
    object_class = 1


class Humanware(namedtuple):
    house_number = 1
    height = 1
    width = 1
    x_top_left = 1
    y_top_left = 1
    screen_height = 1
    screen_width = 1

class UnityEnv(gym.Wrapper):
    host_ids: dict
    visual: bool = False
    asset_name: str

    def __init__(self, config=None, seed=0):
        system = platform.system().lower()

        if self.asset_name is not None:

            id_asset = asset_id(self.asset_name, platform.system())
            path_asset = asset_path(id_asset)
            if not os.path.isdir(path_asset):
                self.download_assets(path_asset)
        else:
            raise KeyError("There are no assets available for {} on {}".format(self.asset_name, system))

        path = os.path.join(path_asset, id_asset)
        ensure_executable(path)
        env = GymUnityEnv(environment_filename=path, worker_id=seed, use_visual=self.visual)
        env.reset(config)

        for i in range(seed):
            env.step(env.action_space.sample())

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


def asset_id(name, system):
    system = system.lower()
    assert system in ['windows', 'darwin', 'linux'], 'only windows, linux, mac are supported'
    path = '{}-{}'.format(name, system)
    return path

def asset_path(asset_id):
    project_root = os.path.dirname(os.path.dirname(__file__))
    default_path = os.path.join(project_root, 'unity_assets')
    dir = os.environ.get('AVENUE_ASSETS', default_path)
    path = os.path.join(dir, asset_id)
    return path

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class AvenueEnv(UnityEnv):
    StateType = AvenueState
    state: AvenueState = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        state_dims = globals()[self.vector_state_class]()
        self.state_idx = [sum(state_dims[:i+1]) for i in range(len(state_dims)-1)]
        self.observation_space = spaces.Box(-1, 1, (sum(state_dims),), np.float32)
        # TODO: make observation space tuple (and create a wrapper to convert it into a vector)

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        m, _, _, _ = self.step(self.action_space.sample())  # we need to step to get info
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


class AllStatesAvenueEnv(AvenueEnv):
    visual = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Dict(dict(
            vector=spaces.Box(-1, 1, (1,), np.float32),
            visual=spaces.Box(0, 255, self.env.observation_space.shape, np.uint8)
        ))

    def step(self, a):
        _, r, d, info = super().step(a)
        s = globals()[self.vector_state_class](info['avenue_state'])
        (vis_obs,), = info['brain_info'].visual_observations
        vis_obs = (255 * vis_obs).astype(np.uint8)
        m = dict(vector=s, visual=vis_obs)
        return m, r, d, info


class RoundcourseEnv(AllStatesAvenueEnv):
    def compute_terminal(self, s, r, d):
        # return s.collide_car or s.collide_pedestrian  # TODO: what else?
        return d

    def compute_reward(self, s: AvenueState, r, d):
        """ Partially inspired by https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
        """
        theta = s.angle_to_next_waypoint_in_degrees / 360 * 2 * np.pi

        normalized_velocity = min(s.velocity_magnitude / s.top_speed, 1)

        r = 0.
        # r += 0.2 * normalized_velocity

        r += 1. * np.cos(theta) * normalized_velocity

        r -= 1. * np.abs(np.sin(theta) * normalized_velocity)
        r, = r

        return r


class VisualAvenueEnv(AllStatesAvenueEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = self.observation_space.spaces["visual"]

    def step(self, a):
        _, r, d, info = super().step(a)
        (vis_obs,), = info['brain_info'].visual_observations
        vis_obs = (255 * vis_obs).astype(np.uint8)
        return vis_obs, r, d, info
