import avenue
from avenue.wrappers import VideoSaver
env = avenue.make("ScenarioZoom")

env = VideoSaver(env)

env.reset()

done = False


for i in range(0, 100):
    _, _, done, _ = env.step(env.action_space.sample())


env.save_video("ScenarioZoom" + ".gif")