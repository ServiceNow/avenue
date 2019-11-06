from .util import namedtuple

"""
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
|steering_angle                   |Normalized steering angle between -1 and 1.|  1   |
|close_pedestrian                 ||  1   |
|close_car                        ||  1   |
|target_waypoint_dist             ||  1   |
|current_waypoint                 ||  1   |
|num_waypoints                    ||  1   |
|diff_next_angle                  ||  1   |

"""


class AvenueCar(namedtuple):
    waypoint_0 = 2
    waypoint_1 = 2
    waypoint_2 = 2
    waypoint_3 = 2
    waypoint_4 = 2
    velocity_magnitude = 1
    angle_to_next_waypoint_in_degrees = 1
    steering_angle = 1
    vertical_force = 1
    velocity = 3
    angular_velocity = 3
    top_speed = 1
    ground_col = 1
    collide_car = 1
    collide_other = 1
    collide_pedestrian = 1
    position = 3
    forward = 3
    closest_waypoint = 3
    diff_next_angle = 1
    close_pedestrian = 1
    close_car = 1
    target_waypoint_dist = 1
    current_waypoint = 1
    num_waypoints = 1


class FollowCar(namedtuple):
    waypoint_0 = 2
    waypoint_1 = 2
    waypoint_2 = 2
    waypoint_3 = 2
    waypoint_4 = 2
    velocity_magnitude = 1
    angle_to_next_waypoint_in_degrees = 1
    steering_angle = 1
    vertical_force = 1
    velocity = 3
    angular_velocity = 3
    top_speed = 1
    ground_col = 1
    collide_car = 1
    collide_other = 1
    collide_pedestrian = 1
    position = 3
    forward = 3
    closest_waypoint = 3
    diff_next_angle = 1
    close_pedestrian = 1
    close_car = 1
    target_waypoint_dist = 1
    current_waypoint = 1
    path_length = 1
    follow_car_pos = 3
    end_point = 3
    car_to_follow_forward = 1
    is_car_visible = 1
    dir_projection_car = 1
