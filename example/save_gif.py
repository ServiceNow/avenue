import avenue
from avenue.wrappers import VideoSaver
from avenue.envs import *

config = {"curvature" : 300, "lane_number": 4, "road_length": 1000, "weather_condition": 1, "vehicle_types": 0, "time" : 20, "city_seed" : 121, "night_mode" : 1, "task" : 1, "starting_speed" : 20}
env = avenue.make("AvenueContinuous_v1", config=config)
env = VideoSaver(env)

env.reset()
env.reset()

done = False

for i in range(0, 600):
    _, _, done, _ = env.step(1)

env.save_video("AvenueContinuous_v1" + ".gif")
