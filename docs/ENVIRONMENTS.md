
## Environments

#### Default reward

Negative y projection of the velocity of the car on the road + positive x projection of the velocity of the car on the 
road (Both normalized between -1 and 1).

-1 if the car touch a pedestrian or a car.

-0.01 per step if the car touch the sidewalk.

-0.05 per step if the car move at a red light.


### Continuous states environments

##### Action space

The action space is the same for all continuous environments : 

[delta_steering_angle, delta_speed, brake]

The two first actions modify the current speed and angle. We made this decision because it simplify the exploration 
problem for self-driving car. Although, the brake just apply after a certains treshold (0.8 by default) to be able to
explore the action space fastly. 
The 3 actions have a range between -1 and 1.  
 

#### *Circuit Visual (segmentation + depth)*
``` env = env.make("CircuitVisual") ```

##### Description of the environment
The car must drive on a circuit as fast as possible with going out the track.
##### State
Semantic segmentation and depth map of size 84 by 84. 
##### Example
![Alt text](../example/CircuitSegmentation.gif?raw=true "Title")

#### *Circuit Rgb*
``` env = env.make("CircuitRgb") ```

##### Description of the environment
The car must drive on a circuit as fast as possible with going out the track.
##### State
RGB(grayscale) of size 128 by 256. 
##### Example
![Alt text](../example/CircuitRgb.gif?raw=true "Title")

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

##### Example

![Alt text](../example/CircuitSegmentation.gif?raw=true "Title")

#### *BirdView*
``` env = env.make("BirdView") ```
##### Description of the environment
On road with bird view segmentation and traffic on the road.
##### State
Camera segmentation of size 210 x 160.
##### Example

![Alt text](../example/RaceAgainstTimeSolo.gif?raw=true "Title")

#### *BirdView*
``` env = env.make("BirdViewSolo") ```
##### Description of the environment
Same as BirdView without traffic.
##### State
Camera segmentation of size 210 x 160.
##### Example

![Alt text](../example/RaceAgainstTime.gif?raw=true "Title")

### Special environments
``` env = env.make("DatasetCollector") ```
#### *Dataset collector*
##### Description of the environment
This environment come with an autopilot and doesn't need reinforcement learning. There's a 
step to do every 500 simulations step (to get diverse images) where you can collect the data. The action space is of size 0. 

There's a special repository to collect and read data efficiently here: 
TODO

##### Example

![Alt text](../example/DatasetCollection.gif?raw=true "Title")
