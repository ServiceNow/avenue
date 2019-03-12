import avenue
import cv2
import time
import imageio

import matplotlib.pyplot as plt
from avenue.wrappers import VideoSaver
env = avenue.make("AvenueContinuous_v1")
#fourcc = cv2.VideoWriter_fourcc(*'MPEG')
#cap = cv2.VideoCapture(0)
#videowriter = imageio.get_writer('video2.mp4', fps=24)
videowriter = imageio.get_writer('test.mp4', fps = 30,
    codec='mjpeg', quality=10, pixelformat='yuvj444p')
#1video = cv2.VideoWriter('video.avi',fourcc=fourcc,fps=float(140),frameSize=(1024,768))

env.reset()
env.reset()

done = False

program_starts = time.time()
i = 0
#while not i < 3944:
for i in range(1700):
    print(i)
    state, _, done, _ = env.step(env.action_space.sample())
    videowriter.append_data(state["visual"])
    #im = cv2.cvtColor(state["visual"], cv2.COLOR_BGR2RGB)
    #video.write(im)

videowriter.close()
#cv2.destroyAllWindows()
#video.release()
import imageio
# h264 ffmpeg by default
