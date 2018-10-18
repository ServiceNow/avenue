# Avenue - Simulator

## Installation
```bash
# install gdown for downloading assets automatically from google drive
pip install gdown

# install unity ml agents
pip install 'git+https://github.com/Unity-Technologies/ml-agents.git#egg=python&subdirectory=python'

# install avenue
pip install avenue
```
## Run

```python
import avenue
env = avenue.make("Circuit")
state = env.reset()
state, reward, done, info = env.step([0, 0, 0])
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
