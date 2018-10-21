# Avenue Simulator

Avenue Simulator is a simulator based on Unity 3D game engine designed to test reinforcment learning algorithm, imitation learning and to collect data.
![Alt text](images/AVENUE.jpg?raw=true "Title")

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

## Environments

### Continuous states environments

#### *Circuit*

##### Description of the environment
Same as *Circuit* but with a continuous state. Since we don't need rendering at each action this environment is really
fast. 
##### State

State of size 13 that contains:
 <ul>
    <li>5 next (x, y) relative waypoints coordinates</li>
    <li>Signed angle with the road in degree</li>
    <li>current steering angle normalized between -1 and 1 [- max steering angle, max steering angle].</li>
    <li>current speed normalized between -1 and 1 [0, max speed].</li>
 </ul>
 
##### Action

##### Reward

### *Circuit segmentation*
#### Description of the environment
#### State
#### Action
#### Done





##### Type: Continuous
##### Size: 3
##### Description: **Delta steering angle**, **Delta speed**, **Throttle**
##### Done

### *Dataset collector*
#### Description of the environment
#### State
#### Action
#### Done
