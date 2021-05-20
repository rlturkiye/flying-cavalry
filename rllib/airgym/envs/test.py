import airsim
from airsim.types import YawMode
import numpy as np
import random
from time import sleep
import math

drone = airsim.CarClient()

#print(drone.getCarState().kinematics_estimated.position)
#print(drone.simGetObjectPose("KargoArabasi").position)
#print(random.randint(0, 4))
def transform_angle(yaw):
    phi = np.linspace(-1, 1, 360)
    for i, value in enumerate(phi):
        if value >= yaw:
            degree = i
            break
    return degree
print(transform_angle(0))



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

def interpret_action(action):
        step_length = 10
        if action == 0:
            quad_offset = (step_length, 0, 1)
        elif action == 1:
            quad_offset = (0, step_length, 1)
        elif action == 2:
            quad_offset = (-step_length, 0, 1)
        elif action == 3:
            quad_offset = (0, -step_length, 1)

        elif action == 4:
            quad_offset = (step_length, 0, -1)
        elif action == 5:
            quad_offset = (0, step_length, -1)
        elif action == 6:
            quad_offset = (-step_length, 0, -1)
        elif action == 7:
            quad_offset = (0, -step_length, -1)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset

while True:
    yawMode = YawMode(is_rate=True, yaw_or_rate=random.randint(0, 1))
    quad_offset = interpret_action(random.randint(0, 9))
    vel = drone.getMultirotorState().kinematics_estimated.linear_velocity
    zpos = drone.getMultirotorState().kinematics_estimated.position.z_val
    drone.moveByVelocityZAsync(
        vel.x_val + quad_offset[0],
        vel.y_val + quad_offset[1],
        zpos + quad_offset[2],
        10,
        yaw_mode=yawMode
    )
    sleep(1)
"""
