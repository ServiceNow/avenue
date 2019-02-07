# AVENUE Simulator

AVENUE Simulator is a simulator based on Unity 3D game engine designed to test reinforcement learning algorithm, imitation learning and to collect data.
![Alt text](images/AVENUE.jpg?raw=true "Title")

## Installation
```bash
# install avenue
pip install 'git+https://github.com/ElementAI/avenue.git'
```
## Quick start

```python
import avenue
env = avenue.make("Circuit_v1")
state = env.reset()
for i in range(0, 1000):
    state, reward, done, info = env.step(env.action_space.sample())
```

## Documentation

[Dataset creation and loading](docs/DATASET.md)

[Documentation of available environments](docs/ENVIRONMENTS.md)

## Run on Docker

``` nvidia-docker run --rm -ti -v /tmp/cowbow-X11-unix:/tmp/.X11-unix -v /mnt/home/$USER:/home/$USER --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools --device=/dev/nvidia4 --volume-driver=nvidia-docker --volume=nvidia_driver_390.30:/usr/local/nvidia:ro -e LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib -e DISPLAY=:0 -e HOME=/home/$USER avenue bash```

## Run on Borgy (Element AI)
