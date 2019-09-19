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
    host_ids = {'linux': '1K122iLjvwL62ApWVaa92HfSWFcS-Lns_'}
    asset_name = 'avenue_continuous'
    vector_state_class = "AvenueState"


class AvenueCarDev(BaseAvenue):
    host_ids = {'linux': '1K122iLjvwL62ApWVaa92HfSWFcS-Lns_'}
    asset_name = 'avenue_follow_car'
    vector_state_class = "AvenueState"

"""
    Example of created environment where you have to drive while avoiding pedestrians.
"""


class LaneFollowing(AvenueCarDev):

    def compute_reward(self, s, r, d):
        theta = math.radians(s.angle_to_next_waypoint_in_degrees[0])
        velocity_magnitude = s.velocity_magnitude[0]
        top_speed = s.top_speed[0]
        r = -math.fabs(1 - (math.cos(theta) * velocity_magnitude / top_speed)) + 1
        return r


class FollowCar(AvenueCarDev):

    vector_state_class = "FollowCar"

    def __init__(self, max_distance_reward, **kwargs):
        super().__init__(**kwargs)
        self.max_distance_reward = max_distance_reward

    def compute_reward(self, s, r, d):
        reward = 1 - (self.get_distance_car_to_follow(s) / self.max_distance_reward)
        return reward

    def compute_terminal(self, s, r, d):
        # Car arrive at destination
        return self.get_3d_distance(s.follow_car_pos, s.end_point) < 30

    def get_distance_car_to_follow(self, s):
        return self.get_3d_distance(s.follow_car_pos, s.position)

    def get_3d_distance(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2)**2))


def Drive(config=None, **kwargs):
    # Randomize config here
    old_config = {
        "road_length": 500,
        "curvature": 140,
        "lane_number": 2,
        "task": 0,
        "time": 13,
        "city_seed": 211,
        "skip_frame": 4,
        "height": 368,
        "width": 368,
        "night_mode":False,
        "pedestrian_distracted_percent": 0.5,
        "pedestrian_density": 50,
        "weather_condition": 5,
        "no_decor": 0,
        "top_speed": 28,  # m/s approximately 50 km / h
        "car_number": 30
    }
    env = LaneFollowing(config=dict(old_config, **config) if config else old_config, **kwargs)
    env = ConcatComplex(env, {"rgb": ["rgb"], "vector": ["velocity_magnitude", "velocity", "angular_velocity"]})
    env = MaxStep(env, max_episode_steps=500)
    return env
