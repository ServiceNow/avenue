import avenue
import numpy as np
import time

env = avenue.make("ZoomBrakingSunny_v1")
env.reset(train_mode=True)

start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0

for i in range(0, 1000):
    action = np.random.binomial(1, 0.01, 1)
    ob, _, _, info = env.step(action)
    print(ob.shape)
    counter+=1
    print("FPS: ", counter / (time.time() - start_time))

