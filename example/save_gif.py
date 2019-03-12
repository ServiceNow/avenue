import avenue
from avenue.wrappers import VideoSaver

config = {"curvature" : 21, "lane_number": 2, "road_length": 1000, "weather_condition": 7, "vehicle_types": 0, "time" : 12, "city_seed" : 20, "night_mode" : 0}

env = avenue.make("AvenueContinuous_v1", config = config)
env = VideoSaver(env)

env.reset()

done = False


for i in range(0, 600):
    _, _, done, _ = env.step(env.action_space.sample())

env.save_video("AvenueContinuous_v1" + ".gif")