from gym_unity.envs import UnityEnv
import os
import platform
import zipfile
import gdown
import random
import gym
import numpy as np
from gym import spaces

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
    system = platform.system().lower()
    if system == 'linux':
        for ext in ['x86_64']:
            filename = bin + '.' + ext
            st = os.stat(filename)
            #os.chmod(filename, st.st_mode | stat.S_IEXEC)
    elif system == 'darwin':
        for ext in ['app']:
            filename = bin + '.' + ext
            st = os.stat(filename)

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
