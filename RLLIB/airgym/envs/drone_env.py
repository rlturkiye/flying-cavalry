import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from PIL import Image


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, useDepth):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.action_space = spaces.Discrete(7)

        self.drone = airsim.MultirotorClient()
        self.last_dist = self.get_distance(self.drone.getMultirotorState().kinematics_estimated.position)
        self.quad_offset = (0, 0, 0)
        self.useDepth = useDepth

        self._setup_flight()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        quad_state = self.drone.getMultirotorState().kinematics_estimated.position
        self.drone.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 10).join()
        time.sleep(.5)

        self.last_dist = self.get_distance(self.drone.getMultirotorState().kinematics_estimated.position)

    def _get_obs(self):

        if self.useDepth:
            # get depth image
            responses = self.drone.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.DepthPlanner, pixels_as_float=True)])
            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float)
            img1d = img1d * 3.5 + 30
            img1d[img1d > 255] = 255
            image = np.reshape(img1d, (responses[0].height, responses[0].width))
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")
        else:
            # Get rgb image
            responses = self.drone.simGetImages(
                [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
            )
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            image = img1d.reshape(response.height, response.width, 3)
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")

        obs = np.array(image_array).reshape(84, 84, 1)

        self.quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.quad_vel = np.array([self.quad_vel.x_val, self.quad_vel.y_val, self.quad_vel.z_val])

        return obs


    def _compute_reward(self):
        """Compute reward"""

        reward = -1.5

        collision = self.drone.simGetCollisionInfo().has_collided
        quad_state = self.drone.getMultirotorState().kinematics_estimated.position

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

    def step(self, action):
        """Step"""
        self.quad_offset = self.interpret_action(action)
        # print("quad_offset: ", self.quad_offset)

        """
        self.drone.moveToPositionAsync(
            q.x_val + self.quad_offset[0],
            q.y_val + self.quad_offset[1],
            q.z_val + self.quad_offset[1],
            3
        )"""

        vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            vel.x_val + self.quad_offset[0],
            vel.y_val + self.quad_offset[1],
            vel.z_val + self.quad_offset[1],
            3
        )

        time.sleep(1)

        reward, done = self._compute_reward()
        obs = self._get_obs()

        return obs, reward, done,  {"quad_vel": self.quad_vel}

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        pts = np.array([3, -76, -7])
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - pts)
        return dist

    def close(self):
        print("close func called")
        pass