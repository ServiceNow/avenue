import gym_unity.envs
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


class UnityEnv(gym.Wrapper):
    host_ids: dict
    visual: bool
    asset_name: str

    def __init__(self):
        seed = random.randint(10000, 20000)
        system = platform.system().lower()
        if(self.asset_name is not None):
            id_asset = asset_id(self.asset_name, platform.system())
            path_asset = asset_path(id_asset)
            if not os.path.isdir(path_asset):
                self.download_assets(path_asset)
        else:
            raise KeyError("There are no assets available for {} on {}".format(self.asset_name, system))

        path = os.path.join(path_asset, id_asset)
        ensure_executable(path)
        env = gym_unity.envs.UnityEnv(environment_filename=path, worker_id=seed, use_visual=self.visual)
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

    def reset(self):
        return self.env.reset()
    
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


class AvenueState(namedtuple):
    waypoint_0 = 2  # TODO: waypoints are relative but should be absolute
    waypoint_1 = 2
    waypoint_2 = 2
    waypoint_3 = 2
    waypoint_4 = 2
    velocity_magnitude = 1
    angle_to_next_waypoint_in_degrees = 1
    velocity = 3
    top_speed = 1
    ground_col = 1
    collide_car = 1 
    collide_pedestrian = 1
    position = 3
    forward = 3
    closest_waypoint = 3


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class AvenueEnv(UnityEnv):
    def __init__(self):
        super().__init__()
        state_dims = AvenueState()
        self.state_idx = [sum(state_dims[:i+1]) for i in range(len(state_dims)-1)]
        self.observation_space = spaces.Box(-1, 1, (sum(state_dims),), np.float32)

    def reset(self, **kwargs):
        _ = self.env.reset()
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

        state = AvenueState(*np.split(vec_obs, self.state_idx))
        reward = self.compute_reward(state)
        # reward = r
        # done = self.compute_terminal(state)
        done = d
        info = dict(info, reset=False, avenue_state=state)  # reset=False, i.e. all dones are true terminals
        return vec_obs, reward, done, info


    def compute_terminal(self, s: AvenueState):
        return s.collide_car or s.collide_pedestrian  # TODO: what else?

    def compute_reward(self, s: AvenueState):
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

    def compute_reward_old(self, s: AvenueState):
        """ Partially inspired by https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html

        The target reward function is

        reward = 0.2f * Mathf.Min(rb.velocity.magnitude/top_speed,1) + 0.8f * (1 / (1 + Mathf.Exp( -1 * 10 * (Mathf.Cos(theta) * Mathf.Min(rb.velocity.magnitude/top_speed,1) - 0.5f)))) -  1 * (Mathf.Sin(theta) * Mathf.Min(rb.velocity.magnitude/ top_speed,1));

        There is actually a bug: theta (obtained through Vector3.SignedAngle) is in degrees while Mathf.sin takes radiants!
        Still even with that bug I am not able to reproduce the original reward.
        """
        v_norm = norm(s.velocity)
        assert v_norm == s.velocity_magnitude, 'sanity check failed'

        # theta = s.angle_to_next_waypoint_in_degrees / 360 * 2 * np.pi
        theta = s.angle_to_next_waypoint_in_degrees  # this line is only for testing. the previous line is correct!

        v_clip = min(s.velocity_magnitude / s.top_speed, 1)

        r = 0.2 * v_clip

        r += 0.8 * sigmoid(10 * (np.cos(theta) * v_clip - 0.5))

        r -= 1. * np.sin(theta) * v_clip

        return r


class AllStatesAvenueEnv(AvenueEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(dict(
            vector=spaces.Box(-1, 1, (1,), np.float32),
            visual=spaces.Box(0, 255, self.env.observation_space.shape, np.uint8)
        ))

    def step(self, a):
        _ , r, d, info = super().step(a)
        s: AvenueState = info['avenue_state']

        (vis_obs,), = info['brain_info'].visual_observations
        vis_obs = (255 * vis_obs).astype(np.uint8)
        m = dict(vector=s.velocity_magnitude / s.top_speed, visual=vis_obs)
        return m, r, d, info


class VisualAvenueEnv(AvenueEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(0, 255, self.env.observation_space.shape, np.uint8)

    def step(self, a):
        _ , r, d, info = super().step(a)
        (vis_obs,), = info['brain_info'].visual_observations
        vis_obs = (255 * vis_obs).astype(np.uint8)
        return vis_obs, r, d, info
