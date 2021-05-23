import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser
import gym
from gym import spaces
from PIL import Image
from CameraRL.RGB import RGB
from CameraRL.DepthVision import DepthVision


class AirSimDroneEnv(gym.Env):
    def __init__(self):
        self.drone = airsim.MultirotorClient()
        self.last_dist = self.get_distance(
            self.drone.getMultirotorState().kinematics_estimated.position)
        self.quad_offset = (0, 0, 0)
        self.target = np.array([3, -76, -7])
        self._setup_flight()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

    def test_distance_code(self):
        quad_state = self.drone.getMultirotorState().kinematics_estimated.position
        print(self.get_distance(quad_state))

    def setGeoFenceCoords(self):
        # x1, y1, x2, y2
        self.geoFenceCoords = []
        pos1 = self.drone.simGetObjectPose("SM_House_27").position
        self.geoFenceCoords.append(pos1.x_val)
        self.geoFenceCoords.append(pos1.y_val)
        pos2 = self.drone.simGetObjectPose("SM_House_86").position
        self.geoFenceCoords.append(pos2.x_val)
        self.geoFenceCoords.append(pos2.y_val)
        print("GeoFenceCoords:", self.geoFenceCoords)

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        quad_pt = np.array(
            list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - self.target)
        inGeoFence = False
        if(self.geoFenceCoords[0] < quad_state.x_val < self.geoFenceCoords[2] or
           self.geoFenceCoords[1] < quad_state.y_val < self.geoFenceCoords[3]):
            inGeoFence = True
        return dist, inGeoFence
