from .env import AvenueEnv
from . import envs
from . import wrappers


def make(env_name, **kwargs):
    env = getattr(envs, env_name)(**kwargs)
    return env
