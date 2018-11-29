from gym_unity.envs import UnityEnv
import os
import platform
import zipfile
import gdown
import random
import gym
import numpy as np
from gym import spaces


class AvenueEnv(gym.Wrapper):
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
        env = UnityEnv(environment_filename=path, worker_id=seed, use_visual=self.visual)
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

def ensure_executable(bin):
    if platform.system().lower() not in ('linux', 'darwin'):
        return
    import stat
    for ext in ['x86', 'x86_64']:
            filename = bin + '.' + ext
            st = os.stat(filename)
            os.chmod(filename, st.st_mode | stat.S_IEXEC)