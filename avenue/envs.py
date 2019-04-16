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
    host_ids = {'linux': '1nErYyVaXJPVo6emPiMoau1340ksPiGvx'}
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"


class AvenueContinuousVector(AvenueEnv):
    host_ids = {'linux': '1SqPdQQti3Sb1qj1R_yEACO2fb0r5eeqP'}
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"


class Humanware(AllStatesAvenueEnv):
    host_ids = {'linux': '107U0_pePmwSHddWkb479Rz4wRSLzOXK-'}
    asset_name = 'humanware'
    vector_state_class = "Humanware"


class RoundcourseEnv(AllStatesAvenueEnv):

    asset_name = 'roundcourse'

    def compute_terminal(self, s, r, d):
        # return s.collide_car or s.collide_pedestrian  # TODO: what else?
        return d

    def compute_reward(self, s: AvenueState, r, d):
        """ Partially inspired by https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
        """
        theta = s.angle_to_next_waypoint_in_degrees / 360 * 2 * np.pi
        normalized_velocity = min(s.velocity_magnitude / s.top_speed, 1)
        r = 0.
        # r += 0.2 * normalized_velocity
        r += 1. * np.cos(theta) * normalized_velocity
        r -= 1. * np.abs(np.sin(theta) * normalized_velocity)
        r, = r
        return r


"""
Here we can register really specific environments for randomize parameters, curriculum learning ...
TODO: complete doc 
"""


def Humanware_v1():
    env = Humanware()
    return env


def en(**kwargs):
    env = AvenueContinuous(**kwargs)
    env = DifferentialActionsVisual(env)
    return env


def StraightDriveCity_v1(**kwargs):
    config = {
        "road_length": 500,
        "curvature": 30,
        "lane_number": 2,
        "task": 0,
        "time": 15,
        "city_seed": 1221,
        "skip_frame": 8,
        "height": 64,
        "width": 128,
        "night_mode": False,
        "pedestrian_distracted_percent": 0,
        "pedestrian_density": 0,
        "weather_condition": 0
    }

    env = AvenueContinuous(**kwargs, config=config)
    env = MaxStep(env, max_episode_steps=10000)
    env = DifferentialActionsFullState(env)
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
        "pedestrian_density": random.randint(3, 30),
        "weather_condition": 0
    }

    if config:
        old_config.update(config)
        config = old_config
    else:
        config = old_config
    env = AvenueContinuous(config=config, **kwargs)
    return env
