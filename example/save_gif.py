import avenue

env = avenue.make("RaceAgainstTimeSolo")
env.reset()
done = False

while not done:
    _, _, done, _ = env.step(env.action_space.sample())


env.save_video("RaceAgainstTimeSolo" + ".gif")