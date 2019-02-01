import avenue

env = avenue.make("CircuitGreyscale_v1")
env.reset()

for i in range(0, 1000):
    ob, _, _, info = env.step(env.action_space.sample())
