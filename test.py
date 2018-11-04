import avenue

env = avenue.make("RaceAgainstTimeSolo")

for i in range(0, 1000):
    obs, _, _, _ = env.step(env.action_space.sample())
    print(obs)