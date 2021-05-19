from numpy.core.fromnumeric import size
import setup_path
import airsim
from airsim.types import YawMode
import numpy as np
import math
import time
from argparse import ArgumentParser
import gym
from gym import spaces
from PIL import Image
from CameraRL.RGB import RGB
from CameraRL.DepthVision import DepthVision


# ip_address, step_length, image_shape, useDepth 
# spaces.Discrete(7)
class AirSimDroneEnv(gym.Env):
    def __init__(self, config):

        self.observation_space = config["observation_space"]
        self.action_space = config["action_space"]
        self.onlySensor = config["onlySensor"]
        if not self.onlySensor:
            self.useDepth = config["use_depth"]
            self.image_size = config["image_size"]
        self.step_length = config["step_length"]

        self.drone = airsim.MultirotorClient()

        if not self.onlySensor:
            if self.useDepth:
                self.camera = DepthVision(client=self.drone, camera_name="0", size=self.image_size)
            else:
                self.camera = RGB(client=self.drone, camera_name="0", size=self.image_size)        
        
        self.last_dist = self.get_distance(self.drone.getMultirotorState().kinematics_estimated.position)
        self.quad_offset = (0, 0, 0)
        self._setup_flight()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.setGeoFenceCoords()

        quad_state = self.drone.getMultirotorState().kinematics_estimated.position
        self.drone.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 10).join()
        time.sleep(.5)

        self.last_dist = self.get_distance(self.drone.getMultirotorState().kinematics_estimated.position)

    def _get_obs(self):
        if not self.onlySensor:
            img = self.camera.fetch_single_img()

        linear_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        linear_acc = self.drone.getMultirotorState().kinematics_estimated.linear_acceleration
        angular_vel = self.drone.getMultirotorState().kinematics_estimated.angular_velocity
        angular_acc = self.drone.getMultirotorState().kinematics_estimated.angular_acceleration
        
        """gps_latitude = self.drone.gps_location.latitude
        gps_longitude = self.drone.gps_location.longitude
        gps_altitude = self.drone.gps_location.altitude"""
        
        pos = self.drone.getMultirotorState().kinematics_estimated.position
        orient = self.drone.getMultirotorState().kinematics_estimated.orientation


        if not self.onlySensor:
            obs = {
                "img": img,
                "linear_vel": np.array([linear_vel.x_val, linear_vel.y_val, linear_vel.z_val]),
                "linear_acc": np.array([linear_acc.x_val, linear_acc.y_val, linear_acc.z_val]),
                "angular_vel": np.array([angular_vel.x_val, angular_vel.y_val, angular_vel.z_val]),
                "angular_acc": np.array([angular_acc.x_val, angular_acc.y_val, angular_acc.z_val]),
            }
        else:
            obs = {
                "linear_vel": np.array([linear_vel.x_val, linear_vel.y_val, linear_vel.z_val]),
                "linear_acc": np.array([linear_acc.x_val, linear_acc.y_val, linear_acc.z_val]),
                "angular_vel": np.array([angular_vel.x_val, angular_vel.y_val, angular_vel.z_val]),
                "angular_acc": np.array([angular_acc.x_val, angular_acc.y_val, angular_acc.z_val]),
            }


      
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
                reward += diff
            else:
                reward += diff

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
        quad_offset = self.interpret_action(action)
        # print("quad_offset: ", self.quad_offset)
        yawMode = YawMode(is_rate=True, yaw_or_rate=10)
        vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        zpos = self.drone.getMultirotorState().kinematics_estimated.position.z_val
        self.drone.moveByVelocityZAsync(
            vel.x_val + quad_offset[0],
            vel.y_val + quad_offset[1],
            zpos + quad_offset[2],
            10,
            yaw_mode=yawMode
        )

        time.sleep(1)

        reward, done = self._compute_reward()
        obs = self._get_obs()

        return obs, reward, done, {}

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 1)
        elif action == 1:
            quad_offset = (0, self.step_length, 1)
        elif action == 2:
            quad_offset = (-self.step_length, 0, 1)
        elif action == 3:
            quad_offset = (0, -self.step_length, 1)

        elif action == 4:
            quad_offset = (self.step_length, 0, -1)
        elif action == 5:
            quad_offset = (0, self.step_length, -1)
        elif action == 6:
            quad_offset = (-self.step_length, 0, -1)
        elif action == 7:
            quad_offset = (0, -self.step_length, -1)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        target = np.array([3, -76, -7])
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - target)
        return dist

    def setGeoFenceCoords(self):
        # x1, y1, x2, y2
        self.geoFenceCoords = []
        pos1 = self.drone.simGetObjectPose("SM_House_27").position
        self.geoFenceCoords.append(pos1.x_val)
        self.geoFenceCoords.append(pos1.y_val)
        pos2 = self.drone.simGetObjectPose("SM_House_86").position
        self.geoFenceCoords.append(pos2.x_val)
        self.geoFenceCoords.append(pos2.y_val)
        #print("GeoFenceCoords:", self.geoFenceCoords)

    def close(self):
        print("close func called")
        pass