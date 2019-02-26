from .env import AvenueEnv, VisualAvenueEnv, UnityEnv, AllStatesAvenueEnv, RoundcourseEnv, AvenueStateZoom
from .wrappers import DifferentialActions, DifferentialActionsVisual
from gym.wrappers import TimeLimit


class Circuit(AllStatesAvenueEnv):
    host_ids = {'linux': '1zwBR0dFZx4oH6YgRz5V4C-kxHiU_HTPc'}
    asset_name = 'circuit'


class ScenarioZoom(AllStatesAvenueEnv):
    StateType = AvenueStateZoom
    host_ids = {'linux': '1A15E-aQjrf_VnPUQmLXSkwfng-BD5H8W'}
    asset_name = 'scenario_zoom'


class Roundcourse(RoundcourseEnv):
    asset_name = 'roundcourse'

class Humanware(AllStatesAvenueEnv):
    host_ids = {'linux': '1A15E-aQjrf_VnPUQmLXSkwfng-BD5H8W'}
    visual = True
    asset_name = 'humanware'
    vector_state_class = "Humanware"

def Circuit_v1():
    env = Circuit()
    env = DifferentialActionsVisual(env)
    return env


def ScenarioZoom_v1():
    env = ScenarioZoom()
    return env

def Humanware_v1():
    env = Humanware()
    return env

