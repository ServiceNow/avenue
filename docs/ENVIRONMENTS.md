
## Environments

### Default reward
You can overwrite the reward function if necessary, but the default reward is an addition of theses terms:

| Name | Weight | Description | Sign |
|:---: | :----: | :---------: | :---:|
| Crash  |   1    |The agent touch a pedestrian or a car.|  - |
| Sidewalk  |   0.01    |The bottom of the agent touch the sidewalk.|  - |
| RedLight  |   0.05    |The car move at a red light.|  - |
| YProjection  |   1  |Y projection of the agent velocity on the direction of the road.|  + |
| XProjection  |   1  |X projection of the agent velocity on the direction of the road.|  - |

The reward is normalized between -1 and 1.

### Action spaces
The action space is dependent of each environment.
##### Continuous action space
| Name                            | Description  | Range|
| :-----------:                   |:------------ |:----:|
|delta_steering_angle |Value added to the normalized steering angle|[-1, 1]|
|delta_speed |Value added to the normalized acceleration.|[-1, 1]|
|brake |Braking of the car.|[-1, 1]|



NB: The brake just apply after a certains treshold (0.8 by default) to be able to
explore the action space faster. 

##### Discrete action space

| Name                            | Value|
| :-----------:                   |:----:|
|Forward|0|
|Backward|1|
|Left|2|
|Right|3|
|Brake|4|
|Noop|5|
##### None action space

No action, useful for data collection.

### State space

#### Vector state
All the environment have access to the following vector state, but some of them can have additional informations.


| Name                            | Description  | Size |
| :-----------:                   |:------------ |:----:|
|waypoint_0                       |Give the 1st waypoint absolute (X,Z) coordinates to follow.|  2   |
|waypoint_1                       |Give the 2nd waypoint absolute (X,Z) coordinates to follow.|  2   |
|waypoint_2                       |Give the 3rd waypoint absolute (X,Z) coordinates to follow.|  2   |
|waypoint_3                       |Give the 4th waypoint absolute (X,Z) coordinates to follow.|  2   |
|waypoint_4                       |Give the 5th waypoint absolute (X,Z) coordinates to follow.|  2   |
|velocity_magnitude               |Magnitude of the current velocity|  1   |
|angle_to_next_waypoint_in_degrees|Give the angle between car direction and the closest waypoint direction.|  1   |
|velocity                         |Give the car velocity (X, Y, Z).|  3   |
|top_speed                        |Give the car maximum speed allowed (useful to normalize speed).|  1   |
|ground_col                       |Detect if a collision occurs between the bottom of the car and something else than the road (sidewalks, terrain, etc...).|  1   |
|collide_car                      |Detect front collision with a car.|  1   |
|collide_pedestrian               |Detect front collision with a pedestrian.|  1   |
|position                         |Center of the car absolute position (X, Y, Z).|  3   |
|forward                          |Forward direction of the car (X, Y, Z).|  3   |
|closest_waypoint                 |Closest waypoint of the car position (X, Y, Z).|  3   |
|horizontal_force                 |Current horizontal force applied (normalized steering angle).|  1   |
|vertical_force                   |Current vertical force applied (normalized acceleration).|  1   |


N.B: The closest waypoint and waypoint_0, waypoint_0 is the next targeted waypoint, and closest waypoint is the one
with the minimum distance in space.

##### How to acces this state ?

In all the case you can access these informations for each state as a dictionary in the *info* variable, for example:

```
    state, reward, done, info = env.step(env.action_space.sample())
    info["avenue_state"]
    
    # info["avenue_state"] contains a named tuple with the previously introduced variables
    AvenueState(waypoint_0=array([14.266401  , -0.05009404], dtype=float32), waypoint_1=array([16.185207  , -0.05794133], dtype=float32), waypoint_2=array([18.119398  , -0.06565315], dtype=float32), waypoint_3=array([19.723623  , -0.07186319], dtype=float32), waypoint_4=array([21.304419  , -0.07777983], dtype=float32), velocity_magnitude=array([4.423185], dtype=float32), angle_to_next_waypoint_in_degrees=array([-22.980762], dtype=float32), velocity=array([ 4.0277534 , -0.02018626, -1.8279392 ], dtype=float32), top_speed=array([15.], dtype=float32), ground_col=array([1.], dtype=float32), collide_car=array([0.], dtype=float32), collide_pedestrian=array([0.], dtype=float32), position=array([-9.8921570e+01,  3.5929665e-02, -1.7101895e+02], dtype=float32), forward=array([ 8.7174380e-01, -3.5436172e-04, -4.8996192e-01], dtype=float32), closest_waypoint=array([-1.0704553e+02,  5.9997559e-02, -1.7537646e+02], dtype=float32), horizontal_force=array([1], dtype=float32), vertical_force=array([1], dtype=float32))
    
```

Or if you want to access it in the state as a vector (not as a dictionary), you can access it like that:
```
    state, reward, done, info = env.step(env.action_space.sample())
    state["vector"]
```

#### Visual state
It depend on the environment and it's specified for each environment which input(s) are available.
The visual states are all concatenated in the 3rd dimension, in this order:

| Name                            | Dimension number|
| :-----------:                   |:----:|
|Grayscale|1|
|RGB|3|
|Depth|1|
|Segmentation|1|

### Environments
| Name |Action space|RGB   |Greyscale|Segmentation|Depth|Visual dimensions|Description|Additional vector state(s)| Reward |
| :---:|:----------:|:----:|:-------:|:----------:|:---:|:--------:|:---------:|:---------------------------:|:---:|
| CircuitGreyscale|Continuous action space|:x:|:heavy_check_mark:|:x:|:x:|64 x 256|Simple track.|None|Default|
| Circuit|Continuous action space|:x:|:x:|:x:|:x:|0|Simple track.|None|Default|
| ZoomScenario|None action space|:x:|:heavy_check_mark:|:x:|:x:|600 x 400|Zoom project.|See Details below.|See Details below.|

### Details
#### *ZoomScenario*
##### Reward
| Name | Weight | Description | Sign |
|:---: | :----: | :---------: | :---:|


##### Additional state
| Name                            | Description  | Size |
| :-----------:                   |:------------ |:----:|
|object_distance | Distance of the object to detect in Unity metrics (close to meter). | 1|
|object_class | Class of the object to detect. [0 : Box, 1: Ball, 2: Trash]| 1|

### Demo
#### *Circuit Greyscale / Circuit*

![Alt text](../example/CircuitRgb.gif?raw=true "Title")
#### *ScenarioZoom*
![Alt text](../example/ScenarioZoom.gif?raw=true "Title")

