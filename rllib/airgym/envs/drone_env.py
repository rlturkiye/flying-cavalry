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
        
        self.target = np.array([0, 0, -19])
        self.last_distances = self.get_distance(self.drone.getMultirotorState().kinematics_estimated.position)
        self.quad_offset = (0, 0, 0)
        self._setup_flight()

        self.total_step = 0

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.setGeoFenceCoords()
        self.target = np.array([0, 0, -19])

        quad_state = self.drone.getMultirotorState().kinematics_estimated.position
        self.drone.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 10).join()
        time.sleep(.5)

        self.last_distances = self.get_distance(self.drone.getMultirotorState().kinematics_estimated.position)

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
                "pos": [pos.x_val, pos.y_val, pos.z_val],
                "target_pos": [self.target[0], self.target[1], self.target[2]],
                "target_dist": self.last_distances,
                "linear_vel": [linear_vel.x_val, linear_vel.y_val, linear_vel.z_val],
                "linear_acc": [linear_acc.x_val, linear_acc.y_val, linear_acc.z_val],
                "angular_vel": [angular_vel.x_val, angular_vel.y_val, angular_vel.z_val],
                "angular_acc": [angular_acc.x_val, angular_acc.y_val, angular_acc.z_val]
            }
        else:
            obs = {
                "pos": [pos.x_val, pos.y_val, pos.z_val],
                "target_pos": [self.target[0], self.target[1], self.target[2]],
                "target_dist": self.last_distances,
                "linear_vel": [linear_vel.x_val, linear_vel.y_val, linear_vel.z_val],
                "linear_acc": [linear_acc.x_val, linear_acc.y_val, linear_acc.z_val],
                "angular_vel": [angular_vel.x_val, angular_vel.y_val, angular_vel.z_val],
                "angular_acc": [angular_acc.x_val, angular_acc.y_val, angular_acc.z_val]
            }


      
        return obs


    def _compute_reward(self):
        """Compute reward"""

        reward = -1.5

        collision = self.drone.simGetCollisionInfo().has_collided
        quad_state = self.drone.getMultirotorState().kinematics_estimated.position

        if collision:
            reward = -120
        elif not self.inGeoFence(quad_state):
            reward = -120
        else:
            dist, dx, dy, dz = self.get_distance(quad_state)
            diff = self.last_distances[0] - dist

            if dist < 10:
                reward = 500
            elif diff > 0:
                reward += diff
            else:
                reward += diff

            self.last_distances = dist, dx, dy, dz

        done = 0
        if reward <= -50:
            done = 1
        elif reward > 499:
            done = 1

        return reward, done

    def step(self, action):
        """Step"""
        quad_offset = self.interpret_action(action)
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

        time.sleep(.2)

        reward, done = self._compute_reward()
        obs = self._get_obs()

        self.total_step += 1
        if reward > 499:
            print("bitti")

        return obs, reward, done, {}

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 3:
            quad_offset = (0, -self.step_length, 0)

        elif action == 4:
            quad_offset = (self.step_length, 0, 0)
        elif action == 5:
            quad_offset = (0, self.step_length, 0)
        elif action == 6:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 7:
            quad_offset = (0, -self.step_length, 0)
        elif action == 8:
            quad_offset = (0, 0, 1)
        elif action == 9:
            quad_offset = (0, 0, -1)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - self.target)
        dx = self.target[0] - quad_pt[0]
        dy = self.target[1] - quad_pt[1]
        dz = self.target[2] - quad_pt[2]
        return dist, dx, dy, dz

    def setGeoFenceCoords(self):
        # x1, y1, x2, y2
        self.geoFenceCoords = {}
        pos1 = self.drone.simGetObjectPose("SM_House_27").position
        self.geoFenceCoords["x1"] = pos1.x_val
        self.geoFenceCoords["y1"] = pos1.y_val
        pos2 = self.drone.simGetObjectPose("SM_House_86").position
        self.geoFenceCoords["x2"] = pos2.x_val
        self.geoFenceCoords["y2"] = pos2.y_val

    def inGeoFence(self, quad_state):
        if self.geoFenceCoords["x1"] < quad_state.x_val < self.geoFenceCoords["x2"]:
            if self.geoFenceCoords["y2"] < quad_state.y_val < self.geoFenceCoords["y1"]:
                if quad_state.z_val > -20:
                    return True
        return False


    def close(self):
        print("close func called")
        pass