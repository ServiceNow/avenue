# AVENUE Simulator

AVENUE Simulator is a simulator based on Unity 3D game engine designed to test reinforcment learning algorithm, imitation learning and to collect data.
![Alt text](images/AVENUE.jpg?raw=true "Title")

## Installation
```bash
# install avenue
pip install 'git+https://github.com/cyrilibrahim/Avenue.git'
```
## Quick start

```python
import avenue
env = avenue.make("Circuit")
state = env.reset()
for i in range(0, 1000):
    state, reward, done, info = env.step(env.action_space.sample())
```

## Environments

#### Default reward

TODO

### Continuous states environments

##### Action space

The action space is the same for all continuous environments : 

[delta_steering_angle, delta_speed, brake]

The two first actions modify the current speed and angle. We made this decision because it simplify the exploration 
problem for self-driving car. Although, the brake just apply after a certains treshold (0.8 by default) to be able to
explore the action space fastly. 
The 3 actions have a range between -1 and 1.  
 

#### *Circuit segmentation*
``` env = env.make("CircuitSegmentation") ```

#### Description of the environment
The car must drive on a circuit as fast as possible with going out the track.
#### State
Semantic segmentation of the track of size 84 by 84. 

#### *Circuit*
``` env = env.make("Circuit") ```
##### Description of the environment
Same as *CircuitSegmentation* but with a continuous state. Since we don't need rendering at each action this environment is really
fast. 
##### State

State of size 13 that contains:
 <ul>
    <li>5 next (x, y) relative waypoints coordinates.</li>
    <li>Signed angle with the road in degree.</li>
    <li>current steering angle normalized between -1 and 1 [- max steering angle, max steering angle].</li>
    <li>current speed normalized between -1 and 1 [0, max speed].</li>
 </ul>


### Special environments
``` env = env.make("DatasetCollector") ```
#### *Dataset collector*
##### Description of the environment
This environment come with an autopilot and doesn't need reinforcment learning. There's a 
step to do every 500 simulations step (to get diverse images) where you can collect the data. The action space is of size 0. 

There's a special repository to collect and read data efficiently here: 
TODO

## Run on Docker

## Run on Borgy (Element AI)

## Usage

### Reinforcment learning baselines

TODO

### Imitation learning baselines

TODO

### Modular pipeline baselines

TODO