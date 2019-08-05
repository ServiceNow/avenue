# AVENUE Simulator

AVENUE is a simulator based on the Unity 3D game engine designed to test reinforcement learning algorithms, for imitation learning and to collect data.
![Alt text](images/AVENUE.jpg?raw=true "Title")

## Installation
```bash
pip install git+https://github.com/ElementAI/Avenue.git 
```

## Quick start

```python
import avenue
env = avenue.make("DriveAndAvoidPedestrians")
state = env.reset()
for i in range(0, 1000):
    state, reward, done, info = env.step(env.action_space.sample())
```

## Documentation
[Dataset creation and loading](docs/DATASET.md) \
[Documentation of available environments](docs/ENVIRONMENTS.md) \
[How to create an environment](avenue/envs.py)
