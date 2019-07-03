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

class AvenueStart(BaseAvenue):
    host_ids = {'linux': '1c5s_HhWSEmwm1JbP7tyy6V252zYVPl25'}
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"

class LearnToBrakeSegmentation(AllStatesAvenueEnv):
    host_ids = {'linux': '1c5s_HhWSEmwm1JbP7tyy6V252zYVPl25'}
    asset_name = 'learn_to_brake_segmentation'
    vector_state_class = "AvenueState"



class AvenueContinuousVector(AvenueEnv):
    host_ids = {'linux': '1SqPdQQti3Sb1qj1R_yEACO2fb0r5eeqP'}
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"


class Humanware(AllStatesAvenueEnv):
    host_ids = {'linux': '107U0_pePmwSHddWkb479Rz4wRSLzOXK-'}
    asset_name = 'humanware'
    vector_state_class = "Humanware"


class Adapt(AllStatesAvenueEnv):
    host_ids = {'linux': '107U0_pePmwSHddWkb479Rz4wRSLzOXK-'}
    asset_name = 'adapt_test'
    vector_state_class = "AvenueState"

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
def Adapt_v1():
    config = {
        "height": 512,
         "width": 512,
         "skip_frame": 8
    }
    env = Adapt(config=config)
    return env


def AvenueContinuous_v1(**kwargs):
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
        "width": 256,
        "night_mode": False,
        "pedestrian_distracted_percent": 0,
        "pedestrian_density": 0,
        "weather_condition": 0
    }

    env = AvenueContinuous(**kwargs, config=config)
    env = MaxStep(env, max_episode_steps=10000)
    env = DifferentialActionsFullState(env)
    return env


def PhysicsGeneralization(car_mass, **kwargs):

    config = {
        "road_length": 500,
        "curvature": 100,
        "lane_number": 2,
        "task": 0,
        "time": 15,
        "city_seed": 100,
        "skip_frame": 8,
        "height": 1,
        "width": 1,
        "night_mode": False,
        "pedestrian_distracted_percent": 0,
        "pedestrian_density": 0,
        "weather_condition": 0,
        "car_mass": car_mass
    }

    env = AvenueContinuous(**kwargs, config=config)
    env = MaxStep(env, max_episode_steps=10000)
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

    if config:
        old_config.update(config)
        config = old_config
    else:
        config = old_config
    env = AvenueContinuous(config=config, **kwargs)
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

def ClimateProjet(config=None, **kwargs):

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
