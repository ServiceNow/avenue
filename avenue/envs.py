from .env import AvenueEnv, VisualAvenueEnv, UnityEnv, AllStatesAvenueEnv
from .wrappers import DifferentialActions, DifferentialActionsVisual
from gym.wrappers import TimeLimit


class Circuit(AvenueEnv):
    host_ids = {'linux': '1t0Uy29qUteBUzot2pfEyXKWWnrBAJRN-'}
    asset_name = 'circuit'
    visual = False


class CircuitGreyscale(AllStatesAvenueEnv):
    host_ids = {'linux': '16ppvjY8xT7p5R-bVGe6OtNle6rLW8PuN'}
    visual = True
    asset_name = 'circuit_rgb'


class ScenarioZoom(AllStatesAvenueEnv):
    host_ids = {'linux': '16ppvjY8xT7p5R-bVGe6OtNle6rLW8PuN'}
    visual = True
    asset_name = 'scenario_zoom'
    vector_state_class = "AvenueStateZoom"


def Circuit_v1():
    env = Circuit()
    env = TimeLimit(env, max_episode_steps=10000)
    env = DifferentialActions(env)
    return env


def CircuitGreyscale_v1():
    env = CircuitGreyscale()
    env = DifferentialActionsVisual(env)
    return env


def ScenarioZoom_v1():
    env = ScenarioZoom()
    return env
