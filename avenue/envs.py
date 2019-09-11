from .env import *
from .wrappers import *

"""
This file give the ability to create new environments. Given the inherited class, you will have different classes of
input.

host_ids: give the google drive links id given the os.

asset_name: refer to the right binary folder in unity_assets.
 
vector_state_class: refer to the type of vector that we want. (see in avenue_states.py)

TODO: doc overwrite reward etc.
"""


class AvenueCar(BaseAvenue):
    host_ids = {'linux': '1c5s_HhWSEmwm1JbP7tyy6V252zYVPl25'}
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"


class AvenueCarDev(BaseAvenue):
    host_ids = {'linux': '1eAAA-N0lO8SuXRGeCDNHgCTJX0ASfWof'}
    asset_name = 'avenue_continuous_dev'
    vector_state_class = "AvenueState"


class UnboundedLaneFollowing(UnboundedLaneFollowing):
    host_ids = {'linux': '1eAAA-N0lO8SuXRGeCDNHgCTJX0ASfWof'}
    asset_name = 'avenue_continuous_dev'
    vector_state_class = "AvenueState"


class LaneFollowingDev(LaneFollowing):
    host_ids = {'linux': '1eAAA-N0lO8SuXRGeCDNHgCTJX0ASfWof'}
    asset_name = 'avenue_continuous_new_vehicle'
    vector_state_class = "AvenueState"

"""
    Example of created environment where you have to drive while avoiding pedestrians.
"""


def PedestrianAvoidance(config=None, **kwargs):
    # Randomize config here
    old_config = {
        "road_length": 800,
        "curvature": 0,
        "lane_number": 3,
        "task": 0,
        "time": 13,
        "city_seed": 211,
        "skip_frame": 6,
        "height": 368,
        "width": 368,
        "night_mode":False,
        "pedestrian_distracted_percent": 0.5,
        "pedestrian_density": 50,
        "weather_condition": 5,
        "no_decor": 0,
        "top_speed": 21  # m/s approximately 50 km / h
    }
    env = LaneFollowingDev(config=dict(old_config, **config) if config else old_config, **kwargs)
    env = ConcatComplex(env, {"rgb": ["rgb"], "vector": ["velocity_magnitude", "velocity", "angular_velocity"]})

    env = MaxStep(env, max_episode_steps=500)
    return env
