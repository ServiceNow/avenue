from gym_unity.envs import UnityEnv
import os
import platform
import zipfile
import gdown
import random
import gym
import numpy as np
from collections import deque
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
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], nb_channel))

    def reset(self):
        ob = self.env.reset()
        ob, reward, done, info = self.step(self.action_space.sample())
        return np.concatenate(info["brain_info"].visual_observations, axis=3).squeeze(0)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return np.concatenate(info["brain_info"].visual_observations, axis=3).squeeze(0), reward, done, info

import imageio

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
        imageio.mimsave(path, self.video_buffer)
        os.chmod(path, 0o777)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.video_buffer.append((ob * 255).astype(np.uint8))
        return ob, reward, done, info


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

def download_assets(path, env_info):
    print('downloading', path + '.zip')
    system = platform.system().lower()
    if system not in env_info["host_ids"]:
        raise KeyError("There are no assets available for {} on {}".format(env_info["asset_name"], system))
    id = env_info["host_ids"][system]
    gdown.download("https://drive.google.com/uc?id="+id, path + ".zip", False)
    zip_ref = zipfile.ZipFile(path + '.zip', 'r')
    print('unpacking ...')
    zip_ref.extractall(path)
    zip_ref.close()
    assert os.path.isdir(path)
    os.remove(path + '.zip')
    print("Unpacked !")

def ensure_executable(bin):
    if platform.system().lower() not in ('linux', 'darwin'):
        return
    import stat
    for ext in ['x86', 'x86_64']:
            filename = bin + '.' + ext
            st = os.stat(filename)
            os.chmod(filename, st.st_mode | stat.S_IEXEC)

dict_envs = {
    "Circuit" : {
        "host_ids" : {'linux': '1t0Uy29qUteBUzot2pfEyXKWWnrBAJRN-'},
        "visual": False,
        "asset_name": 'circuit'
    },
    "CircuitSegmentation": {
        "host_ids": {'linux': '1hjB1OimKrkEzOHBAQTOI6syzi-UT_GH9'},
        "visual": True,
        "asset_name": 'circuit_segmentation'
    },
    "DatasetCollection": {
        "host_ids": {'linux': '1WE--vDGYKYMBYPsuCqJJehTKHCIx8zgl'},
        "visual": True,
        "asset_name": 'dataset_collection'
    },
    "CircuitVisual": {
        "host_ids": {'linux': '175-NVmuqQawlubyMd_1eT6qigIa0RNBi'},
        "visual": True,
        "asset_name": 'circuit_visual'
    },
    "RaceAgainstTime": {
        "host_ids": {'linux': '1dmPPK4mFTnYPnatpSWIme0QmYSvFR9l-', 'darwin':'15Z21R9RlaQGN1jv-ipZSoN5PJYjNS3DB'},
        "visual": True,
        "asset_name": 'race_against_time'
    },
    "RaceAgainstTimeSolo": {
        "host_ids": {'linux': '1imEoe9CWyij9fIQwQwHEVdspDWDEHjNH'},
        "visual": True,
        "asset_name": 'race_against_time_solo'
    },
    "CircuitRgb": {
        "host_ids": {'linux': '1UZ-Wv-yFBhjlr-mNEECDUXwrnwOEEjt0'},
        "visual": True,
        "asset_name": 'circuit_rgb'
    }
}

def make(env_name):
    seed = random.randint(10000, 20000)
    asset_name = dict_envs[env_name]["asset_name"]
    system = platform.system().lower()
    if(asset_name is not None):
        id_asset = asset_id(asset_name, platform.system())
        path_asset = asset_path(id_asset)
        if not os.path.isdir(path_asset):
            download_assets(path_asset, dict_envs[env_name])
    else:
        raise KeyError("There are no assets available for {} on {}".format(asset_name, system))

    bin = os.path.join(path_asset, id_asset)
    ensure_executable(bin)
    env = UnityEnv(environment_filename=bin,worker_id=seed, use_visual=dict_envs[env_name]["visual"])
    # if dict_envs[env_name]["visual"]:
    #     env = ConcatVisualUnity(env)
        # env = VideoSaver(env)
    return env