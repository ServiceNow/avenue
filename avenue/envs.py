from .env import *
from .wrappers import *

"""
This file give the ability to create new environments. Given the inherited class, you will have different classes of
input.

host_ids: give the google drive links id given the os.

asset_name: refer to the right binary folder in unity_assets.
 
vector_state_class: refer to the type of vector that we want. (see in avenue_states.py)

"""

class AvenueCar_v0(BaseAvenue):
    host_ids = {'linux': '1aXpNIVvAsK3-8ByAPvRD1gtcypIp39Zz'}
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"


class AvenueCarDev(BaseAvenue):
    host_ids = {'linux': '1aXpNIVvAsK3-8ByAPvRD1gtcypIp39Zz'}
    asset_name = 'avenue_with_traffic'
    vector_state_class = "AvenueState"

"""
    Example of created environment where you have to drive while avoiding pedestrians.
"""




class LaneFollowing(AvenueCar_v0):
    def lane_following_reward(self, s, r, d):
        theta = math.radians(s.angle_to_next_waypoint_in_degrees[0])
        velocity_magnitude = s.velocity_magnitude[0]
        top_speed = s.top_speed[0]
        r = -math.fabs(1 - (math.cos(theta) * velocity_magnitude / top_speed)) + 1
        return r


class UnboundedLaneFollowing(AvenueCar_v0):

    def compute_reward(self, s, r, d):
        theta = math.radians(s.angle_to_next_waypoint_in_degrees[0])
        velocity_magnitude = s.velocity_magnitude[0]
        top_speed = s.top_speed[0]
        r = (math.cos(theta) * velocity_magnitude/top_speed)
        return r


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
    env = LaneFollowing(config=dict(old_config, **config) if config else old_config, **kwargs)
    env = ConcatComplex(env, {"rgb": ["rgb"], "vector": ["velocity_magnitude", "velocity", "angular_velocity"]})
    env = MaxStep(env, max_episode_steps=500)
    return env
