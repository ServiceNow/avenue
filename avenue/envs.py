from gym.wrappers import TimeLimit

from .env import *
from .wrappers import *
from .util import min_max_norm

"""
This file give the ability to create new environments. Given the inherited class, you will have different classes of
input.

host_ids: give the google drive links id given the os.

asset_name: refer to the right binary folder in unity_assets.
 
vector_state_class: refer to the type of vector that we want. (see in avenue_states.py)

"""


class AvenueCar(BaseAvenueCtrl):
    # host_ids = {'linux': '1K122iLjvwL62ApWVaa92HfSWFcS-Lns_'}
    host_ids = {'linux': '1eRKQaRxp2dJL9krKviqyecNv5ikFnMrC'}
    asset_name = 'avenue_follow_car'
    vector_state_class = "AvenueState"
    ctrl_type = ControllerType.CAR


class AvenueCarDev(BaseAvenueCtrl):
    host_ids = {'linux': '1K122iLjvwL62ApWVaa92HfSWFcS-Lns_'}
    asset_name = 'avenue_follow_car'
    vector_state_class = "AvenueState"
    ctrl_type = ControllerType.CAR


class AvenueDroneDev(BaseAvenueCtrl):
    host_ids = {'linux': '1K122iLjvwL62ApWVaa92HfSWFcS-Lns_'}
    asset_name = 'avenue_follow_car'
    vector_state_class = "Drone"
    ctrl_type = ControllerType.DRONE

"""
    Example of created environment where you have to drive while avoiding pedestrians.
"""


class LaneFollowing(AvenueCar):
    def __init__(self, config):
        super().__init__(config=dict(config, task=0))  # TODO: This is weird. What is task 0?

    def compute_reward(self, s, r, d):
        theta = math.radians(s.angle_to_next_waypoint_in_degrees[0])
        velocity_magnitude = s.velocity_magnitude[0]
        top_speed = s.top_speed[0]
        r = (math.cos(theta) * velocity_magnitude) / top_speed
        if s.close_car[0] == 1 or s.close_pedestrian[0] == 1:
            r = 0
        if d:
            r = -40

        return min_max_norm(r, -40, 1)

    def compute_terminal(self, s, r, d):
        return d


class AutoSpeed(gym.Wrapper):
    def __init__(self, env, speed_target=28):
        super(AutoSpeed, self).__init__(env)
        self.speed_target = speed_target
        self.action_space = gym.spaces.Box(-1, 1, (1,))
        self.last_speed = 0

    def step(self, action):
        calculated_speed = math.fabs((self.speed_target - self.last_speed) * 0.05)
        brake = 0
        action = np.array([calculated_speed, action[0]])
        observation, reward, done, info = self.env.step(action)
        self.last_speed = observation["vector"][3]
        return observation, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def RaceSolo(concat_complex=False):

    def generate_env():
        return LaneFollowing(dict(
            lane_number=2,
            task=0,
            time=random.randint(8, 17),
            city_seed=random.randint(0, 10000),
            skip_frame=4,
            width=256,
            height=64,
            night_mode=False,
            road_type=6,
            pedestrian_distracted_percent=random.random(),
            pedestrian_density=0,
            weather_condition=1,
            no_decor=0,
            top_speed=26,  # m/s approximately 50 km / h
            car_number=0,
            layout=1,
            done_unity=1,
            starting_speed=random.randint(0, 10),
            hd_rendering=0
        ))

    env = RandomizedEnv(generate_env, n=10000)
    env = TimeLimit(env, max_episode_steps=1000)
    if concat_complex:
        env = ConcatComplex(env, {"rgb": ["rgb"], "vector": ["velocity_magnitude"]})
    # env = DictToTupleWrapper(env, "rgb", ["velocity_magnitude", "velocity", "angular_velocity"])
    else:
        env = DictToTupleWrapper(env, "rgb", ["velocity_magnitude"])
    return env
