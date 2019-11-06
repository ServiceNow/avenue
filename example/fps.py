import avenue
import time
import scipy.misc

env = avenue.make("RaceSolo_v0")
env.reset()

start_time = time.time()

for i in range(0, 1000):
    step_time = time.time()
    ob, r, done, info = env.step([1, 0])
    scipy.misc.imsave('race_solo.jpg', ob[0][:, :, 0])

    print("FPS: ", i / (time.time() - start_time))
    if done:
        env.reset()
