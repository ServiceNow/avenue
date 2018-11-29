from .env import AvenueEnv
from . import envs
from . import wrappers


def make(env_name):
    env = getattr(envs, env_name)()
    return env
