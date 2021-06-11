import airsim
from airsim.types import DrivetrainType, YawMode
import numpy as np
import random
from time import sleep
import math




drone = airsim.MultirotorClient()
drone.reset()
drone.enableApiControl(True)
drone.armDisarm(True)
pos = drone.getMultirotorState().kinematics_estimated.position
pose = drone.simGetVehiclePose()
pose.position.x_val = 0
pose.position.y_val = 0
pose.position.z_val = -12
drone.simSetVehiclePose(pose, True)
#print(drone.simGetObjectPose("KargoArabasi").position)
#print(random.randint(0, 4))
"""def transform_angle(yaw):
    phi = np.linspace(-1, 1, 360)
    for i, value in enumerate(phi):
        if value >= yaw:
            degree = i
            break
    return degree
print(transform_angle(0))
geoFenceCoords = [0, 0, 0, 0]
geofence_corners = ((geoFenceCoords[0], geoFenceCoords[1]),
                    (geoFenceCoords[0], geoFenceCoords[3]),
                    (geoFenceCoords[2], geoFenceCoords[3]),
                    (geoFenceCoords[2], geoFenceCoords[1]))
dx = geofence_corners[0][0] - 1
dy = geofence_corners[0][1] - 1"""

"""drone.reset()
drone.enableApiControl(True)
drone.armDisarm(True)
drone.takeoffAsync()

vel = drone.getMultirotorState().kinematics_estimated.linear_velocity
drone.moveByVelocityZAsync(
    vel.x_val,
    vel.y_val,
    -15,
    3
).join()
"""
def interpret_action(action):
        _, _, yaw  = airsim.to_eularian_angles(drone.simGetVehiclePose().orientation)
        if action == 0: # forward
            vx = math.cos(yaw)
            vy = math.sin(yaw)
            z = 0
            yaw_rate = 0
        elif action == 1: # right
            vx = math.cos(yaw + math.pi/2)
            vy = math.sin(yaw + math.pi/2)
            z = 0
            yaw_rate=270
        elif action == 2: # backward
            vx = math.cos(yaw - math.pi)
            vy = math.sin(yaw - math.pi)
            z = 0
            yaw_rate=180
        elif action == 3: # left
            vx = math.cos(yaw - math.pi/2)
            vy = math.sin(yaw - math.pi/2)
            z = 0
            yaw_rate=90
        elif action == 4: # down
            vx = 0
            vy = 0
            z = 1
            yaw_rate = 0
        elif action == 5: # up
            vx = 0
            vy = 0
            z = -1
            yaw_rate = 0
        else:
            vx = 0
            vy = 0
            z = 0
            yaw_rate = 0

        return (vx*2.5, vy*2.5, z), YawMode(is_rate=False, yaw_or_rate=yaw_rate)


def calculate_target_location():
    pos = drone.simGetObjectPose("KargoArabasi").position
    target = [pos.x_val, pos.y_val, pos.z_val]
    return target

def calculate_angle(target):
        pos = drone.getMultirotorState().kinematics_estimated.position
        target_degree = math.degrees(math.atan2((target[1] - pos.y_val) , (target[0] - pos.x_val)))
        return target_degree

import time

def correctOrientation(target):
    vel = drone.getMultirotorState().kinematics_estimated.linear_velocity
    zpos = drone.getMultirotorState().kinematics_estimated.position.z_val
    degree = calculate_angle(target)

    yawMode = YawMode(is_rate=False, yaw_or_rate=degree)
    drone.moveByVelocityZAsync(
        vel.x_val,
        vel.y_val,
        zpos,
        3,
        yaw_mode=yawMode)
    sleep(.5)
    return degree

def get_distance(quad_state, target):
        """Get distance between current state and goal state"""
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - target)
        return [dist]

while True:
    target = calculate_target_location()
    degree = correctOrientation(target) + 180

    vel = drone.getMultirotorState().kinematics_estimated.linear_velocity
    zpos = drone.getMultirotorState().kinematics_estimated.position.z_val

    quad_offset, yawMode = interpret_action(0)
    drone.moveByVelocityZAsync(
        quad_offset[0],
        quad_offset[1],
        zpos + quad_offset[2],
        10,
        drivetrain=DrivetrainType.ForwardOnly,
        yaw_mode=yawMode
    )
    print(get_distance(drone.getMultirotorState().kinematics_estimated.position, target))
    sleep(.2)
