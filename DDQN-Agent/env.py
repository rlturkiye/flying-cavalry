import csv
import math
import pprint
import time

import torch
from PIL import Image

import numpy as np

import airsim
#import setup_path
import random


class DroneEnv(object):
    """Drone environment class using AirSim python API"""

    def __init__(self, useDepth=False):
        self.client = airsim.MultirotorClient()
        self.last_dist = self.get_distance(self.client.getMultirotorState().kinematics_estimated.position)
        self.quad_offset = (0, 0, 0)
        self.useDepth = useDepth

    def step(self, action):
        """Step"""
        #print("new step ------------------------------")

        self.quad_offset = self.interpret_action(action)
        #print("quad_offset: ", self.quad_offset)

        q = self.client.getMultirotorState().kinematics_estimated.position
        self.client.moveToPositionAsync(
            q.x_val + self.quad_offset[0],
            q.y_val + self.quad_offset[1],
            q.z_val,
            3
        )

        time.sleep(.4)

        result, done = self.compute_reward()
        state, image, quad_vel = self.get_obs()

        return state, result, done, image, quad_vel

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        quad_state = self.client.getMultirotorState().kinematics_estimated.position

        self.client.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 10).join()
        time.sleep(.5)
        self.last_dist = self.get_distance(self.client.getMultirotorState().kinematics_estimated.position)


        obs, image, quad_vel = self.get_obs()

        return obs, image, quad_vel

    def get_obs(self):
        if self.useDepth:
            # get depth image
            responses = self.client.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.DepthPlanner, pixels_as_float=True)])
            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float)
            img1d = img1d * 3.5 + 30
            img1d[img1d > 255] = 255
            image = np.reshape(img1d, (responses[0].height, responses[0].width))
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")
        else:
            # Get rgb image
            responses = self.client.simGetImages(
                [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
            )
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            image = img1d.reshape(response.height, response.width, 3)
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")

        obs = np.array(image_array)
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])

        return obs, image, quad_vel

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        pts = np.array([3, -76, -7])
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - pts)
        return dist

    def compute_reward(self):
        """Compute reward"""

        reward = -1.5


        collision = self.client.simGetCollisionInfo().has_collided
        quad_state = self.client.getMultirotorState().kinematics_estimated.position

        if collision:
            reward = -120
        elif quad_state.z_val < -22 or quad_state.z_val > -1:
                reward = -120
        else:
            dist = self.get_distance(quad_state)
            diff = self.last_dist - dist

            if dist < 10:
                reward = 500
            elif diff > 0:
                reward += diff * 3
            else:
                reward += diff * 3

            self.last_dist = dist

        done = 0
        if reward <= -10:
            done = 1
        elif reward > 499:
            done = 1

        print(reward)

        return reward, done


    def interpret_action(self, action):
        """Interprete action"""
        scaling_factor = 5

        if action == 0:
            self.quad_offset = (scaling_factor, 0, 0)
        elif action == 1:
            self.quad_offset = (-scaling_factor, 0, 0)
        elif action == 2:
            self.quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            self.quad_offset = (0, -scaling_factor, 0)
        elif action == 4:
            self.quad_offset = (0, 0, scaling_factor)
        elif action == 5:
            self.quad_offset = (0, 0, -scaling_factor)
        elif action == 6:
            self.quad_offset = (0, 0, 0)

        return self.quad_offset
