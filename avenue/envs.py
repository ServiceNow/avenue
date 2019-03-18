from .env import AvenueEnv, VisualAvenueEnv, UnityEnv, AllStatesAvenueEnv, RoundcourseEnv, AvenueStateZoom
from .wrappers import DifferentialActions, DifferentialActionsVisual, BrakeDiscreteControl, WrapPyTorch
from gym.wrappers import TimeLimit


class ScenarioZoom(AllStatesAvenueEnv):
    StateType = AvenueStateZoom
    host_ids = {'linux': '1A15E-aQjrf_VnPUQmLXSkwfng-BD5H8W'}
    asset_name = 'scenario_zoom'
    vector_state_class = "AvenueStateZoom"


class Roundcourse(RoundcourseEnv):
    asset_name = 'roundcourse'


class Humanware(AllStatesAvenueEnv):
    host_ids = {'linux': '107U0_pePmwSHddWkb479Rz4wRSLzOXK-'}
    visual = True
    asset_name = 'humanware'
    vector_state_class = "Humanware"


class AvenueContinuous(AllStatesAvenueEnv):
    host_ids = {'linux': '1SqPdQQti3Sb1qj1R_yEACO2fb0r5eeqP'}
    visual = True
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"


class AvenueContinuousNoVisual(AvenueEnv):
    host_ids = {'linux': '1C9m9moICFwCIda3vtFqTcSNYf4F3kKoz'}
    visual = True
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"


def ScenarioZoom_v1():
    env = ScenarioZoom()
    return env


def Humanware_v1():
    env = Humanware()
    return env


def AvenueContinuous_v1(**kwargs):
    env = AvenueContinuous(**kwargs)
    env = DifferentialActionsVisual(env)
    return env


def ZoomBrakingSunny_v1(config=None, **kwargs):

    old_config = {"curvature": 0, "lane_number": 1, "road_length": 750, "weather_condition": 0, "vehicle_types": 0,
              "time": 12, "city_seed": 121, "night_mode": 0, "task": 1, "starting_speed": 20}

    if config:
        old_config.update(config)
        config = old_config
    else:
        config = old_config

    env = AvenueContinuous(config = config, **kwargs)
    env = BrakeDiscreteControl(env)
    env = WrapPyTorch(env)
    return env
