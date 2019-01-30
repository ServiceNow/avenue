import avenue

env = avenue.make("ScenarioZoom")
env.reset()

for i in range(0, 1000):
    ob, _, _, _ = env.step(env.action_space.sample())
