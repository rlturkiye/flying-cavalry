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
import random
import math

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
                self.camera = DepthVision(
                    client=self.drone, camera_name="0", size=self.image_size)
            else:
                self.camera = RGB(client=self.drone,
                                  camera_name="0", size=self.image_size)

        self.target = np.array([0, 0, -19])
        self.target_house = "SM_House_27"
        self.last_distances = self.get_distance(
            self.drone.getMultirotorState().kinematics_estimated.position)
        self.quad_offset = (0, 0, 0)
        self.total_step = 0
        self.total_check = 5
        self.current_final = False
        self.starting_positions = (
            [293, -349, -2], [-212, 7, -2], [23, -14, -2], [-216, -362, -2], [160, -66, -2])
        self.houses = ["SM_House_27", "SM_House_85",
                       "SM_House_22", "SM_House_287", "SM_House_4333"]
        self._setup_flight()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.setGeoFenceCoords()
        self.current_final = False
        self.starting_pos = self.starting_positions[random.randint(0, 4)]
        # teleport the drone
        pose = self.drone.simGetVehiclePose()
        pose.position.x_val = self.starting_pos[0]
        pose.position.y_val = self.starting_pos[1]
        pose.position.z_val = self.starting_pos[2]
        self.drone.simSetVehiclePose(pose, True)
        time.sleep(1)
        quad_state = self.drone.getMultirotorState().kinematics_estimated.position
        # move up otherwise the drone will stuck
        self.drone.moveToPositionAsync(
            quad_state.x_val, quad_state.y_val, -7, 5).join()
        # correct orientation
        self.correctOrientation()
        # set target house pos
        pos = self.drone.simGetObjectPose(self.houses[0]).position
        pos = [pos.x_val, pos.y_val, pos.z_val]
        self.target_house_pos = np.array(pos)
        self.calculate_target_location()
        self.last_distances = self.get_distance(
            self.drone.getMultirotorState().kinematics_estimated.position)

    def _get_obs(self):
        if not self.onlySensor:
            img = self.camera.fetch_single_img()

        linear_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        linear_acc = self.drone.getMultirotorState(
        ).kinematics_estimated.linear_acceleration
        angular_vel = self.drone.getMultirotorState().kinematics_estimated.angular_velocity
        angular_acc = self.drone.getMultirotorState(
        ).kinematics_estimated.angular_acceleration

        # orient = self.drone.getMultirotorState().kinematics_estimated.orientation TODO discrete yapılcak

        distanceToGeoFence = self.distanceToGeoFence(
            self.drone.getMultirotorState().kinematics_estimated.position)

        if not self.onlySensor:
            obs = {
                "img": img,
                "target_dist": self.last_distances,
                "linear_vel": [linear_vel.x_val, linear_vel.y_val, linear_vel.z_val],
                "linear_acc": [linear_acc.x_val, linear_acc.y_val, linear_acc.z_val],
                "angular_vel": [angular_vel.x_val, angular_vel.y_val, angular_vel.z_val],
                "angular_acc": [angular_acc.x_val, angular_acc.y_val, angular_acc.z_val],
                "distToGeoFence": distanceToGeoFence
            }
        else:
            obs = {
                "target_dist": self.last_distances,
                "linear_vel": [linear_vel.x_val, linear_vel.y_val, linear_vel.z_val],
                "linear_acc": [linear_acc.x_val, linear_acc.y_val, linear_acc.z_val],
                "angular_vel": [angular_vel.x_val, angular_vel.y_val, angular_vel.z_val],
                "angular_acc": [angular_acc.x_val, angular_acc.y_val, angular_acc.z_val],
                "distToGeoFence": distanceToGeoFence
            }

        return obs

    def _compute_reward(self):
        """Compute reward"""

        reward = -1.5
        # col=self.drone.simGetCollisionInfo()
        collision = self.drone.simGetCollisionInfo().has_collided
        quad_state = self.drone.getMultirotorState().kinematics_estimated.position

        if collision:
            reward = -100
            print("Collision")
        elif not self.inGeoFence(quad_state):
            reward = -100
            print("Not in geofence")
        else:
            dist, dx, dy, dz = self.get_distance(quad_state)
            diff = self.last_distances[0] - dist
            self.last_distances = [dist, dx, dy, dz]

            if dist < 20:
                if self.current_final:
                    reward = 500
                else:
                    reward = 499
                    self.current_final = True
                    self.target = self.target_house_pos
                    self.last_distances = self.get_distance(quad_state)
            elif diff > 0:
                reward += diff
            else:
                reward += diff

        done = 0
        if reward <= -50:
            print("terminate: 1")
            done = 1
        elif reward > 499:
            done = 1
            print("terminate: 2")
        print("Reward:", reward, "Done:", done,
              "current_final:", self.current_final)
        return reward, done

    def step(self, action):
        """Step"""
        self.calculate_target_location()

        if self.total_step % self.total_check == 0:
            self.correctOrientation()

        quad_offset = self.interpret_action(action)
        vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        zpos = self.drone.getMultirotorState().kinematics_estimated.position.z_val
        self.drone.moveByVelocityZAsync(
            vel.x_val + quad_offset[0],
            vel.y_val + quad_offset[1],
            zpos + quad_offset[2],
            1
        )

        #self.drone.moveToPositionAsync(self.target[0], self.target[1], -13, 10)

        time.sleep(1)

        reward, done = self._compute_reward()
        obs = self._get_obs()

        self.total_step += 1
        if reward > 499:
            print("bitti")

        return obs, reward, done, {}

    def reset(self):
        print(self.total_step)
        self.total_step = 0
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        # joinle kullanılmaz
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 3:
            quad_offset = (0, -self.step_length, 0)
        elif action == 4:
            quad_offset = (0, 0, 1)
        elif action == 5:
            quad_offset = (0, 0, -1)
        else:
            quad_offset = (0, 0, 0)

        """if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 3:
            quad_offset = (self.step_length, -self.step_length, 0)
        if action == 4:
            quad_offset = (self.step_length, self.step_length, 0)
        elif action == 5:
            quad_offset = (-self.step_length, self.step_length, 0)
        elif action == 6:
            quad_offset = (-self.step_length, -self.step_length, 0)
        elif action == 7:
            quad_offset = (0, 0, 1)
        elif action == 8:
            quad_offset = (0, 0, -1)
        else:
            quad_offset = (0, 0, 0)"""

        return quad_offset

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        quad_pt = np.array(
            list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - self.target)
        dx = self.target[0] - quad_pt[0]
        dy = self.target[1] - quad_pt[1]
        dz = self.target[2] - quad_pt[2]
        return [dist, dx, dy, dz]

    def setGeoFenceCoords(self):
        # x1, y1, x2, y2
        self.geoFenceCoords = {}
        pos1 = self.drone.simGetObjectPose("SM_House_27").position
        self.geoFenceCoords["x1"] = pos1.x_val - 20
        self.geoFenceCoords["y1"] = pos1.y_val + 20
        pos2 = self.drone.simGetObjectPose("SM_House_86").position
        self.geoFenceCoords["x2"] = pos2.x_val + 20
        self.geoFenceCoords["y2"] = pos2.y_val - 20

    def inGeoFence(self, quad_state):
        if self.geoFenceCoords["x1"] < quad_state.x_val < self.geoFenceCoords["x2"]:
            if self.geoFenceCoords["y2"] < quad_state.y_val < self.geoFenceCoords["y1"]:
                if quad_state.z_val > -25:
                    return True
        return False

    def distanceToGeoFence(self, quad_state):
        # self.geoFenceCoords = [x1 y1 x2 y2]
        geofence_corners = ([self.geoFenceCoords["x1"], self.geoFenceCoords["y1"]],
                            [self.geoFenceCoords["x1"], self.geoFenceCoords["y2"]],
                            [self.geoFenceCoords["x2"], self.geoFenceCoords["y2"]],
                            [self.geoFenceCoords["x2"], self.geoFenceCoords["y1"]])

        closest_dist = 99999
        closest_index = 0
        for i, corner in enumerate(geofence_corners):
            dist = np.linalg.norm(
                np.array([quad_state.x_val, quad_state.y_val]) - np.array(corner))
            if dist < closest_dist:
                closest_dist = dist
                closest_index = i

        dx = geofence_corners[closest_index][0] - quad_state.x_val
        dy = geofence_corners[closest_index][1] - quad_state.y_val

        return [closest_dist, dx, dy]

    def calculate_angle(self):
        pos = self.drone.getMultirotorState().kinematics_estimated.position
        target_degree = math.degrees(math.atan2(
            (self.target[1] - pos.y_val), (self.target[0] - pos.x_val)))
        return target_degree

    def calculate_target_location(self):
        if not self.current_final:
            pos = self.drone.simGetObjectPose("KargoArabasi").position
            self.target = [pos.x_val, pos.y_val, pos.z_val]

    """def transform_angle(self, yaw):
        phi = np.linspace(-1, 1, 360)
        degree = 360
        for i, value in enumerate(phi):
            if value >= yaw:
                degree = i
                break
        return degree"""

    def correctOrientation(self):
        yawMode = YawMode(is_rate=False, yaw_or_rate=0)
        vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        zpos = self.drone.getMultirotorState().kinematics_estimated.position.z_val
        degree = self.calculate_angle()
        if degree < 10:
            pass
        yawMode = YawMode(is_rate=False, yaw_or_rate=degree)
        self.drone.moveByVelocityZAsync(
            vel.x_val,
            vel.y_val,
            zpos,
            1,
            yaw_mode=yawMode)
        time.sleep(1)

    def close(self):
        print("close func called")
        pass
