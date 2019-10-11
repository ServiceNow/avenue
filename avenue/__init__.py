from . import envs
from . import wrappers


def make(env_name, **kwargs):
    env = getattr(envs, env_name)(**kwargs)
    return env


def download(env_name):
    env_cls = getattr(envs, env_name)  # this will only work for a subset of the envs specified via classes
    return env_cls.get_assets()