import avenue
import numpy as np
import time
import random

random.seed(2313)

env = avenue.make("LaneAvoidance")
env.reset(train_mode=True)

start_time = time.time()

for i in range(0, 1000):
    step_time = time.time()
    ob, _, done, info = env.step(env.action_space.sample())
    print("Step time took")
    print(time.time() - step_time)
    if done:
        env.reset()
        print("FPS: ", i / (time.time() - start_time))

