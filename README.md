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