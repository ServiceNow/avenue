import avenue
import time
import random


env = avenue.make("RaceSolo")
env.reset(train_mode=True)

start_time = time.time()

for i in range(0, 1000):
    step_time = time.time()
    ob, _, done, info = env.step(env.action_space.sample())
    print("FPS: ", i / (time.time() - start_time))
    if done:
        env.reset()
