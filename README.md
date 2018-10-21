# Avenue - Simulator
![Alt text](images/avenue.png?raw=true "Title")

## Installation
```bash
# install avenue
pip install 'git+https://github.com/cyrilibrahim/Avenue.git'
```
## Run example

```python
import avenue
env = avenue.make("Circuit")
state = env.reset()
for i in range(0, 1000):
    state, reward, done, info = env.step(env.action_space.sample())
```

## Available environments

### *Circuit segmentation*
#### Description of the environment
#### State
#### Action
#### Done

### *Circuit*

#### Description of the environment
Same as *Circuit* but with a continuous state. Since we don't need rendering at each action this environment is really
fast. 
#### State

State of size 13 that contains: [..., ..., ...]. 

#### Action



##### Type: Continuous
##### Size: 3
##### Description: **Delta steering angle**, **Delta speed**, **Throttle**
##### Done

### *Dataset collector*
#### Description of the environment
#### State
#### Action
#### Done
