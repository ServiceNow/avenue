import avenue
import time


env = avenue.make("CityPedestrians_v0")
env.reset(train_mode=True)

start_time = time.time()

for i in range(0, 1000):
    step_time = time.time()
    ob, r, done, info = env.step([1, 0])
    print(r)
    print("FPS: ", i / (time.time() - start_time))
    if done:
        env.reset()
