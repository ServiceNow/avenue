import avenue
from avenue.wrappers import VideoSaver
env = avenue.make("ScenarioZoom_v1")

env = VideoSaver(env)

env.reset()

done = False


for i in range(0, 150):
    _, _, done, _ = env.step(env.action_space.sample())


env.save_video("ScenarioZoom_v1" + ".gif")