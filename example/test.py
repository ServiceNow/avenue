import avenue
import numpy as np
import time
import random

random.seed(2313)

env = avenue.make("PedestrianClassification_v1")
env.reset(train_mode=False)

start_time = time.time()

for i in range(0, 1000):
    ob, _, _, info = env.step(env.action_space.sample())
    print("FPS: ", i / (time.time() - start_time))

