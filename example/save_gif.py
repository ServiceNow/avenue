import avenue
from avenue.wrappers import VideoSaver

config = {"curvature" : 300, "lane_number": 4, "road_length": 1000, "weather_condition": 1, "vehicle_types": 0, "time" : 18, "city_seed" : 20, "night_mode" : 1}

env = avenue.make("AvenueContinuous_v1", config = config)
env = VideoSaver(env)

env.reset()

done = False


for i in range(0, 600):
    _, _, done, _ = env.step(env.action_space.sample())

env.save_video("AvenueContinuous_v1" + ".gif")