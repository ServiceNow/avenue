import avenue

env = avenue.make("Circuit")
env.reset()

for i in range(0, 1000):
    env.step(env.action_space.sample())