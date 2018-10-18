import avenue

env = avenue.make("DatasetCollection")
env.reset()

for i in range(0, 10):
    env.step(env.action_space.sample())