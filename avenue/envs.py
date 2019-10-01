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
    ctrl_type = ControllerType.CAR


class AvenueCarDev(BaseAvenue):
    host_ids = {'linux': '1K122iLjvwL62ApWVaa92HfSWFcS-Lns_'}
    asset_name = 'avenue_follow_car'
    vector_state_class = "AvenueState"
    ctrl_type = ControllerType.CAR


class AvenueDroneDev(BaseAvenue):
    host_ids = {'linux': '1K122iLjvwL62ApWVaa92HfSWFcS-Lns_'}
    asset_name = 'avenue_follow_car'
    vector_state_class = "Drone"
    ctrl_type = ControllerType.DRONE

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

    def compute_terminal(self, s, r, d):
        return d


class DronePath(AvenueDroneDev):

    def compute_reward(self, s, r, d):
        theta = math.radians(s.angle_to_next_waypoint_in_degrees[0])
        velocity_magnitude = s.velocity_magnitude[0]
        top_speed = 20
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
        return self.get_3d_distance(s.follow_car_pos, s.end_point) < 30 or s.is_car_visible == False

    def get_distance_car_to_follow(self, s):
        return self.get_3d_distance(s.follow_car_pos, s.position)

    def get_3d_distance(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2)**2))


def Drive(config=None, **kwargs):
    curr_time = random.randint(8, 21)
    if curr_time > 18:
        night = True
    else:
        night = False

    if random.random() > 0.5:
        weather = 0
    else:
        weather = 5

    if random.random() > 0.6:
        layout = 0
    else:
        layout = 1

    # Randomize config here
    old_config = {
        "road_length": random.randint(100, 400),
        "curvature": random.randint(0, 100),
        "lane_number": random.randint(1, 4),
        "task": 0,
        "time": curr_time,
        "city_seed": random.randint(0, 10000),
        "skip_frame": 4,
        "height": 368,
        "width": 368,
        "night_mode":night,
        "road_type": random.randint(0, 4),
        "pedestrian_distracted_percent": random.random(),
        "pedestrian_density": random.randint(0, 3),
        "weather_condition": weather,
        "no_decor": 0,
        "top_speed": 28,  # m/s approximately 50 km / h
        "car_number": random.randint(0, 40),
        "layout": layout,
        "done_unity": 1,
        "starting_speed": random.randint(10, 50)
    }

    env = LaneFollowing(config=dict(old_config, **config) if config else old_config, **kwargs)
    env = DifferentialActions(ConcatComplex(env, {"rgb": ["rgb"], "vector": ["velocity_magnitude", "velocity", "angular_velocity"]}))
    env = MaxStep(env, max_episode_steps=100)
    return env
