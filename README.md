*ServiceNow completed its acquisition of Element AI on January 8, 2021. All references to Element AI in the materials that are part of this project should refer to ServiceNow.*

# Avenue simulator

![Title Image](resources/head_2.png)

Avenue is a simulator designed to test and prototype reinforcement learning algorithms. It is based on the Unity3D game engine. Check out the [NeurIPS '19 paper Real-Time Reinforcement Learning](https://github.com/rmst/rtrl) in which Avenue is used to compare time-constrained reinforcement learning agents.

### Quick start

```bash
pip install git+https://github.com/ElementAI/Avenue.git 
```

```python
import avenue
import time

env = avenue.make("RaceSolo-v0")
env.reset()
start_time = time.time()

for i in range(0, 1000):
    ob, r, done, info = env.step(env.action_space.sample())
    print("FPS: ", i / (time.time() - start_time))
    if done:
        env.reset()
```

### Environments
<p align="center">
  <img src="/resources/race_solo.jpg" width=49.5% />
  <img src="/resources/city_pedestrians.jpg" width=49.5% /> 
</p>

In all environments the agent controls a car with a two-dimensional, continuous action space (target steering angle and gas/brake). The observation is a tuple containing (1) a 256x64 gray-scale image and (2) a vector containing the car's normalized velocity magnitude and its true steering angle. All environments are procedually re-generated every 10000 steps.

**RaceSolo-v0** is a simple race track environment in which the agent is incentivized to drive at a target speed of 45km/h (in the direction of the road) and not to leave the road. An episode terminates if the agent leaves the road or arrives at the end of the track.

**RaceObstacles-v0** is similar to RaceSolo but with traffic cones on the race track that have to be avoided. 

**CityPedestrians-v0** is a straight avenue in an inner city environment with pedestrians crossing the street. The car is incentivized to stay on the road, avoid pedestrians and drive at the target speed of 45km/h. 

**Custom environments**  can be created by changing the extensive configuration options that can be found in `avenue/envs.py`. Even though the source code for the simulator binary is currently not public this will allow you to change road layout and type, weather presets, time of day, traffic and more.

### Performance
Depending on the environment used, on our laptops `env.step` requires approximately 0.02 seconds, i.e. the simulator runs at 50 frames per second including the interprocess communation between Python and Unity3D.


## Citing
If you use Avenue in your research, you can cite it as follows:
```bibtex
@misc{ibrahim2019avenue,
    author={Ibrahim, Cyril and Ramstedt, Simon and Pal, Christopher},
    title={Avenue},
    year={2019},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/elementai/avenue}},
}
```
