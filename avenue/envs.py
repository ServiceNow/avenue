from .env import *
from .wrappers import *
import random

"""
This file give the ability to register new environments. Given the inherited class, you will have different classes of
input.

AvenueEnv: 
    return the state as a vector of class vector_state_class

RgbAvenueEnv:
    return the state as an rgb image.

SegmentationAvenueEnv:
    return the state as a segmentation of the scene.

AllStatesAvenue: 
    return the state as a dict composed of a vector entry with the vector state, and a visual that have
    an rgb image and segmentation.


host_ids: give the google drive links id given the os.

asset_name: refer to the right binary folder in unity_assets.
 
vector_state_class: refer to the type of vector that we want. (see in avenue_states.py)

TODO: doc overwrite reward etc.
"""


class AvenueContinuous(AllStatesAvenueEnv):
    host_ids = {'linux': '1c5s_HhWSEmwm1JbP7tyy6V252zYVPl25'}
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"


class AvenueCar(BaseAvenue):
    host_ids = {'linux': '1c5s_HhWSEmwm1JbP7tyy6V252zYVPl25'}
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"


def AvenueContinuous_v1(**kwargs):
    env = AvenueContinuous(**kwargs)
    env = DifferentialActionsVisual(env)
    return env


def DriveAndAvoidPedestrian(config=None, **kwargs):

    # Randomize config here
    old_config = {
        "road_length": 500,
        "curvature": 0,
        "lane_number": 2,
        "task": 0,
        "time": 13,
        "city_seed": 211,
        "skip_frame": 8,
        "height": 64,
        "width": 256,
        "night_mode":False,
        "pedestrian_distracted_percent": 0.5,
        "pedestrian_density": 50,
        "weather_condition": 0
    }

    env = AvenueCar(config=dict(old_config, **config) if config else old_config, **kwargs)
    env = ConcatComplex(env, {"rgb": ["rgb"], "vector": ["velocity_magnitude"]})
    return env


def ZoomRL(config=None, **kwargs):

    # Randomize config here
    old_config = {
        "road_length": 500,
        "curvature": 0,
        "lane_number": 2,
        "task": 0,
        "time": 13,
        "city_seed": 211,
        "skip_frame": 8,
        "height": 512,
        "width": 512,
        "night_mode":False,
        "pedestrian_distracted_percent": 0.5,
        "pedestrian_density": 50,
        "weather_condition": 0
    }

    if config:
        old_config.update(config)
        config = old_config
    else:
        config = old_config
    env = AvenueContinuous(config=config, **kwargs)
    return env

def LaneAvoidance(config=None, **kwargs):

    # Randomize config here
    old_config = {
        "road_length": 1000,
        "curvature": 0,
        "lane_number": 4,
        "task": 1,
        "time": 13,
        "city_seed": 211,
        "skip_frame": 8,
        "height": 512,
        "width": 512,
        "night_mode":False,
        "pedestrian_distracted_percent": 0,
        "pedestrian_density": 0,
        "weather_condition": 0,
        "no_decor": 1
    }

    if config:
        old_config.update(config)
        config = old_config
    else:
        config = old_config
    env = AvenueContinuous(config=config, **kwargs)
    return env


def Climate(config=None, climat_change=False, **kwargs):
    curr_time = random.randint(8, 17)
    if not climat_change:
        night = False
        weather = 0
    else:
        night = True
        weather = 4

    if random.random() < 0.5:
        road_type = 0
    else:
        road_type = 1

    if random.random() < 0.5:
        curvature = random.randint(-100, 100)
    else:
        curvature = 0

    # Randomize config here
    old_config = {
        "road_length": 500,
        "curvature": curvature,
        "lane_number": random.randint(1, 3),
        "task": 3,
        "time": curr_time,
        "city_seed": random.randint(100, 10000),
        "skip_frame": 0,
        "height": 1028,
        "width": 2024,
        "night_mode":night,
        "pedestrian_distracted_percent": 0,
        "pedestrian_density": 0,
        "weather_condition": weather,
        "road_type": road_type,
        "black_and_white": 0
    }

    if config:
        old_config.update(config)
        config = old_config
    else:
        config = old_config
    env = AvenueContinuous(config=config, **kwargs)
    return env
