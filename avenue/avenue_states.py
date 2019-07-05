from .util import namedtuple


class AvenueState(namedtuple):
    waypoint_0 = 2
    waypoint_1 = 2
    waypoint_2 = 2
    waypoint_3 = 2
    waypoint_4 = 2
    velocity_magnitude = 1
    angle_to_next_waypoint_in_degrees = 1
    horizontal_force = 1
    vertical_force = 1
    velocity = 3
    top_speed = 1
    ground_col = 1
    collide_car = 1
    collide_pedestrian = 1
    position = 3
    forward = 3
    closest_waypoint = 3


class Humanware(namedtuple):
    house_number = 1
    height = 1
    width = 1
    x_top_left = 1
    y_top_left = 1
    screen_height = 1
    screen_width = 1
