from gym.wrappers import TimeLimit

from .env import *
from .wrappers import *
from .util import min_max_norm, np_distance

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

class AvenueCar_v0(BaseAvenueCtrl):
    host_ids = {'linux': '19dTKAJ8BEGbza85I1q33IGCLKktz2uos'}
    asset_name = 'avenue_follow_car'
    vector_state_class = "AvenueCar"
    ctrl_type = ControllerType.CAR

class AvenueDroneDev(BaseAvenueCtrl):
    # TODO: Should we keep this in for the release?
    host_ids = {'linux': '1K122iLjvwL62ApWVaa92HfSWFcS-Lns_'}
    asset_name = 'avenue_follow_car'
    vector_state_class = "Drone"
    ctrl_type = ControllerType.DRONE


class LaneFollowing(AvenueCar):
    def __init__(self, config):
        super().__init__(config=dict(config, task=0))  # TODO: What is task 0?

    def compute_reward(self, s, r, d):
        theta = math.radians(s.angle_to_next_waypoint_in_degrees[0])
        velocity_magnitude = s.velocity_magnitude[0]
        top_speed = s.top_speed[0]
        r = (math.cos(theta) * velocity_magnitude) / top_speed
        if s.close_car[0] == 1 or s.close_pedestrian[0] == 1:
            r = 0
        if d:
            r = -5.

        return min_max_norm(r, -5, 1)

    def compute_terminal(self, s, r, d):
        return d


class Car_v0(AvenueCar_v0):

    reward_close = -1.
    reward_ground_col = -4
    reward_low_speed = -3
    reward_pedestrian_hit = -5
    reward_car_hit = -5
    reward_obstacle_hit = -5

    _min_speed = 1
    _max_count_low_speed = 100
    _max_dist_next_wp = 20
    _max_count_sidewalk = 20

    def __init__(self, config):
        super().__init__(config=dict(config, task=0))
        self._counter_low_speed = 0
        self._counter_sidewalk = 0
        self._touched_sidewalk = False

    def reset(self, **kwargs):
        self._counter_low_speed = 0
        self._counter_sidewalk = 0
        self._touched_sidewalk = False
        return super().reset(**kwargs)

    def step(self, action):
        ob, reward, done, info = super().step(action)
        if not(ob["close_car"][0] == 1 or ob["close_pedestrian"][0] == 1) and ob["velocity_magnitude"][0] < \
                self._min_speed:
            self._counter_low_speed += 1
        else:
            self._counter_low_speed = 0

        if ob["ground_col"] == 1:
            self._counter_sidewalk += 1

        self._touched_sidewalk = (ob["ground_col"] == 1) or self._touched_sidewalk

        reset = self.compute_reset(ob, reward, done)

        if reset:
            info["reset"] = True

        done = done or reset

        return ob, reward, done, info

    def compute_reward(self, s, r, d):
        theta = math.radians(s.angle_to_next_waypoint_in_degrees[0])
        velocity_magnitude = s.velocity_magnitude[0]
        top_speed = s.top_speed[0]

        r = (math.cos(theta) * velocity_magnitude) / top_speed

        if s.close_car[0] == 1 or s.close_pedestrian[0] == 1:
            r = max(self.reward_close * velocity_magnitude / top_speed, self.get_min_reward())

        if s.ground_col[0] == 1:
            r = self.reward_ground_col

        if s.collide_other[0] == 1:
            r = self.reward_obstacle_hit

        if s.collide_car[0] == 1:
            r = self.reward_car_hit

        if s.collide_pedestrian[0] == 1:
            r = self.reward_pedestrian_hit

        return min_max_norm(r, self.get_min_reward(), 1)

    def get_min_reward(self):
        return min(self.reward_close, self.reward_ground_col, self.reward_low_speed, self.reward_obstacle_hit,
                   self.reward_pedestrian_hit, self.reward_car_hit)

    def compute_terminal(self, s, r, d):
        if s.collide_other[0] == 1:
            return True

        if s.collide_car[0] == 1:
            return True

        if s.collide_pedestrian[0] == 1:
            return True

        return False

    def compute_reset(self, s, r, d):
        if np_distance(s["closest_waypoint"], s["position"]) > self._max_dist_next_wp:
            return True

        if self._counter_low_speed > self._max_count_low_speed:
            return True

        if self._touched_sidewalk and (self._counter_sidewalk > self._max_count_sidewalk or s["ground_col"][0] == 0):
            return True

        return False


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
            layout=1,  # race track
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


def RaceSolo_v0(concat_complex=False):
    def generate_env():
        return Car_v0(dict(
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
            layout=1,  # race track
            done_unity=1,
            starting_speed=random.randint(0, 10),
            hd_rendering=0
        ))

    env = RandomizedEnv(generate_env, n=10000)
    env = TimeLimit(env, max_episode_steps=1000)
    if concat_complex:
        env = ConcatComplex(env, {"rgb": ["rgb"], "vector": ["velocity_magnitude", "steering_angle"]})
    # env = DictToTupleWrapper(env, "rgb", ["velocity_magnitude", "velocity", "angular_velocity"])
    else:
        env = DictToTupleWrapper(env, "rgb", ["velocity_magnitude"])
    return env

def RaceObstacles_v0(concat_complex=False):
    def generate_env():
        return Car_v0(dict(
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
            layout=1,  # race track
            done_unity=1,
            starting_speed=random.randint(0, 10),
            hd_rendering=0,
            nb_obstacles=200
        ))

    env = RandomizedEnv(generate_env, n=10000)
    env = TimeLimit(env, max_episode_steps=1000)
    if concat_complex:
        env = ConcatComplex(env, {"rgb": ["rgb"], "vector": ["velocity_magnitude", "steering_angle"]})
    # env = DictToTupleWrapper(env, "rgb", ["velocity_magnitude", "velocity", "angular_velocity"])
    else:
        env = DictToTupleWrapper(env, "rgb", ["velocity_magnitude"])
    return env

def CityPedestrians():
    def generate_env():
        return LaneFollowing(dict(
            lane_number=2,
            task=0,
            time=12,
            # city_seed=random.randint(0, 10000),
            city_seed=55,
            skip_frame=4,
            width=256,
            height=64,
            night_mode=False,
            road_type=0,  # city
            pedestrian_distracted_percent=0.2,
            pedestrian_density=5,
            weather_condition=0,
            no_decor=0,
            top_speed=26,  # m/s approximately 50 km / h
            car_number=20,
            layout=0,  # 0 = straight road, 1 = curvy road, 2 = crossroads
            done_unity=1,
            starting_speed=random.randint(0, 10),  # TODO: what does this do?
            hd_rendering=0
        ))

    env = RandomizedEnv(generate_env, n=10000)
    env = TimeLimit(env, max_episode_steps=1000)
    # env = ConcatComplex(env, {"rgb": ["rgb"], "vector": ["velocity_magnitude", "velocity", "angular_velocity"]})
    # env = DictToTupleWrapper(env, "rgb", ["velocity_magnitude", "velocity", "angular_velocity"])
    env = DictToTupleWrapper(env, "rgb", ["velocity_magnitude", "angular_velocity"])
    return env