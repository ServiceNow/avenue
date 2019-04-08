from .env import AvenueEnv, VisualAvenueEnv, UnityEnv, AllStatesAvenueEnv, RoundcourseEnv, AvenueStateZoom
from .wrappers import DifferentialActions, DifferentialActionsVisual, BrakeDiscreteControl, WrapPyTorch
from gym.wrappers import TimeLimit
import random

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
    host_ids = {'linux': '17_cxtMwRv814jzwDTIxVwmubBAuP83zi'}
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
              "time": 12, "city_seed": 223123, "night_mode": 0, "task": 1, "starting_speed": 20}
    if config:
        old_config.update(config)
        config = old_config
    else:
        config = old_config
    env = AvenueContinuous(config = config, **kwargs)
    env = BrakeDiscreteControl(env)
    env = WrapPyTorch(env)
    return env


def PedestrianClassification_v1(config=None, **kwargs):

    random_hour = random.randint(6, 20)
    if random_hour < 9 or random_hour > 17:
        night_mode = True
    else:
        night_mode = False

    # Randomize config here
    old_config = {
        "road_length": 500,
        "curvature": random.randint(0, 100),
        "lane_number": random.randint(1, 4),
        "task": 2,
        "time": random_hour,
        "city_seed": random.randint(0, 100000),
        "skip_frame": 30,
        "height": 600,
        "width": 800,
        "night_mode" : night_mode,
        "pedestrian_distracted_percent": random.random(),
        "pedestrian_density" : random.randint(3, 30),
        "weather_condition" : 0
    }

    if config:
        old_config.update(config)
        config = old_config
    else:
        config = old_config
    env = AvenueContinuous(config=config, **kwargs)
    return env
