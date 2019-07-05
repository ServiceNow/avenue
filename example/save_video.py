import avenue
import cv2
from time import time
import imageio

import matplotlib.pyplot as plt
from avenue.wrappers import VideoSaver
# env = avenue.make("AvenueContinuous_v1")
env = avenue.make("DriveAndAvoidPedestrian", config=dict(skip_frame=4, city_seed=3))

# video_writer = imageio.get_writer('test.mp4', fps = 30, codec='mjpeg', quality=10, pixelformat='yuvj444p')


video_writer = imageio.get_writer('test.mp4', fps=40)

env.reset()

t0 = time()
for i in range(500):
    print('step', i, end='\r')
    a = env.action_space.sample()
    # a = (0., 0., 0.)
    state, _, done, _ = env.step(a)
    if True:
      video_writer.append_data(state["visual"]['rgb'])
    if done:
      env.reset()
      # print("terminated early after", i, 'steps')
      # break
video_writer.close()

print('fps', i/(time() - t0))