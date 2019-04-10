import avenue
import cv2
import time
import imageio

import matplotlib.pyplot as plt
from avenue.wrappers import VideoSaver
env = avenue.make("AvenueContinuous_v1")

video_writer = imageio.get_writer('test.mp4', fps = 30, codec='mjpeg', quality=10, pixelformat='yuvj444p')

env.reset()

done = False

program_starts = time.time()
i = 0
for i in range(1700):
    print(i)
    state, _, done, _ = env.step(env.action_space.sample())
    video_writer.append_data(state["visual"])

video_writer.close()
