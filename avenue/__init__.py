from gym_unity.envs import UnityEnv
import os
import platform
import zipfile
import gdown
import random
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

def asset_id(name, system: str):
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

def ensure_executable(bin):
    if platform.system().lower() not in ('linux', 'darwin'):
        return
    import stat
    for ext in ['x86_64']:
        filename = bin + '.' + ext
        st = os.stat(filename)
        #os.chmod(filename, st.st_mode | stat.S_IEXEC)

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
        "host_ids": {'linux': '1WE--vDGYKYMBYPsuCqJJehTKHCIx8zgl'},
        "visual": True,
        "asset_name": 'circuit_visual'
    },
    "RaceAgainstTime": {
        "host_ids": {'linux': '1WE--vDGYKYMBYPsuCqJJehTKHCIx8zgl', 'darwin':'15Z21R9RlaQGN1jv-ipZSoN5PJYjNS3DB'},
        "visual": True,
        "asset_name": 'race_against_time'
    }
}

def make(env_name):
    seed = random.randint(1, 20000)
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
    if(env_name != "CircuitVisual"):
        return UnityEnv(environment_filename=bin,worker_id=seed, use_visual=dict_envs[env_name]["visual"])
    else:
        return ConcatVisualUnity(UnityEnv(environment_filename=bin,worker_id=seed, use_visual=dict_envs[env_name]["visual"]))
