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

import gc
import psutil
def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return

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
        self.sim_speed = config["sim_speed"]
        self.map = config["map"]
        self.reward_multiplier = 2
        self.speed = 2.5

        self.drone = airsim.MultirotorClient()

        if not self.onlySensor:
            if self.useDepth:
                self.camera = DepthVision(client=self.drone, camera_name="0", size=self.image_size)
            else:
                self.camera = RGB(client=self.drone, camera_name="0", size=self.image_size)        
        
        self.target = np.array([0, 0, -19])
        self.target_house = "SM_House_27"
        self.target_house_pos = np.array([0, 0, 0])
        self.last_distances = self.get_distance(self.drone.getMultirotorState().kinematics_estimated.position)
        self.quad_offset = (0, 0, 0)
        self.total_step = 0
        self.total_check = 5
        self.current_final = False
        self.starting_positions = [[0, 0, -10], [-212, 7, -10], [-212, 7, -10], [23, -14, -10], [-216, -362, -10], [160, -66, -10]]
        self.houses = ["SM_House_27", "SM_House_85", "SM_House_22", "SM_House_287", "SM_House_4333"]
        self._setup_flight()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.current_final = False
        self.calculate_target_location()
        # set target house pos
        pos = self.drone.simGetObjectPose(self.houses[random.randint(0, len(self.houses)-1)]).position 
        #pos = self.drone.simGetObjectPose("SM_House_80").position 
        pos = [pos.x_val, pos.y_val, pos.z_val]
        self.target_house_pos = np.array(pos)
        self.starting_pos = self.starting_positions[random.randint(0, len(self.starting_positions)-1)]

        can_start = False
        while not can_start:
            can_start = self.isResetPositionAvaible()
            if not can_start:
                print("Start position changed")
                self.starting_pos = self.starting_positions[random.randint(0, len(self.starting_positions)-1)]
                time.sleep(1 / self.sim_speed)

        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.setGeoFenceCoords(map=self.map)
        # teleport the drone
        pose = self.drone.simGetVehiclePose()
        pose.position.x_val = self.starting_pos[0]
        pose.position.y_val = self.starting_pos[1]
        pose.position.z_val = self.starting_pos[2]
        self.last_collision_id = self.drone.simGetCollisionInfo().object_id
        self.drone.simSetVehiclePose(pose, True)
        quad_state = self.drone.getMultirotorState().kinematics_estimated.position
        # move up otherwise the drone will stuck
        self.drone.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 5).join()
        # correct orientation
        self.correctOrientation()
        self.last_position = quad_state
        self.last_distances = self.get_distance(self.drone.getMultirotorState().kinematics_estimated.position)

    def _get_obs(self):
        if not self.onlySensor:
            img = self.camera.fetch_single_img()

        linear_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        linear_acc = self.drone.getMultirotorState().kinematics_estimated.linear_acceleration
        angular_vel = self.drone.getMultirotorState().kinematics_estimated.angular_velocity
        angular_acc = self.drone.getMultirotorState().kinematics_estimated.angular_acceleration
        
        #orient = self.drone.getMultirotorState().kinematics_estimated.orientation TODO discrete yapılcak

        distanceToGeoFence = self.distanceToGeoFence(self.drone.getMultirotorState().kinematics_estimated.position)

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
        done = 0
        #col=self.drone.simGetCollisionInfo()
        collision = self.last_collision_id != self.drone.simGetCollisionInfo().object_id
        quad_state = self.drone.getMultirotorState().kinematics_estimated.position


        terminate_reason = "None"

        if self.last_position == quad_state:
            done = 1
            terminate_reason = "Stuck"
        elif collision:
            reward = -100
            done = 1
            terminate_reason = "Collision"
        elif not self.inGeoFence(quad_state):
            reward = -100
            done = 1
            terminate_reason = "Not in geofence"
        else:
            dist, dx, dy, dz = self.get_distance(quad_state)
            diff = self.last_distances[0] - dist
            self.last_distances = [dist, dx, dy, dz]

            if dist < 15 and self.current_final:
                reward = 500
                done = 1
                terminate_reason = "Done"
            elif dist < 12 and not self.current_final:
                reward = 499 
                done = 0
                if self.map == "Small":
                    reward = 500
                    done = 1
                terminate_reason = "Done"
                self.current_final = True
                self.target = self.target_house_pos
                self.last_distances = self.get_distance(quad_state)
            elif diff > 0:
                reward += diff * self.reward_multiplier
            else:
                reward += diff * self.reward_multiplier

        self.last_position = quad_state
        
        if done == 1:
            print("Terminate :", terminate_reason, "| current_final:", self.current_final, "| Steps in episode:", self.total_step)
        
        return reward, done

    def step(self, action):
        """Step"""
        auto_garbage_collect()

        self.correctOrientation()
        zpos = self.drone.getMultirotorState().kinematics_estimated.position.z_val
        quad_offset, yawMode = self.interpret_action(action)
        self.drone.moveByVelocityZAsync(
            quad_offset[0],
            quad_offset[1],
            zpos + quad_offset[2],
            10,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=yawMode
        )

        time.sleep(1/self.sim_speed)

        reward, done = self._compute_reward()
        obs = self._get_obs()

        self.total_step += 1
        
        self.calculate_target_location()

        return obs, reward, done, {}

    def reset(self):
        self.total_step = 0
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        _, _, yaw  = airsim.to_eularian_angles(self.drone.simGetVehiclePose().orientation)
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

        return (vx*self.speed, vy*self.speed, z), YawMode(is_rate=False, yaw_or_rate=yaw_rate)

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - self.target)
        dx = self.target[0] - quad_pt[0]
        dy = self.target[1] - quad_pt[1]
        dz = self.target[2] - quad_pt[2]
        return [dist, dx, dy, dz]

    def setGeoFenceCoords(self, map="Default"):
        # x1, y1, x2, y2
        self.geoFenceCoords = {}
        if map == "Default":
            pos1 = self.drone.simGetObjectPose("SM_House_27").position
            self.geoFenceCoords["x1"] = pos1.x_val - 20
            self.geoFenceCoords["y1"] = pos1.y_val + 20 
            pos2 = self.drone.simGetObjectPose("SM_House_86").position
            self.geoFenceCoords["x2"] = pos2.x_val + 20
            self.geoFenceCoords["y2"] = pos2.y_val - 20
        elif map == "Small":
            self.geoFenceCoords["x1"] = -70
            self.geoFenceCoords["y1"] = 80
            self.geoFenceCoords["x2"] = 80
            self.geoFenceCoords["y2"] = -70
        else:
            assert NotImplementedError

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
            dist = np.linalg.norm(np.array([quad_state.x_val, quad_state.y_val]) - np.array(corner))
            if dist < closest_dist:
                closest_dist = dist
                closest_index = i

        dx = geofence_corners[closest_index][0] - quad_state.x_val
        dy = geofence_corners[closest_index][1] - quad_state.y_val
                
        return [closest_dist, dx, dy]

    def calculate_angle(self):
        pos = self.drone.getMultirotorState().kinematics_estimated.position
        target_degree = math.degrees(math.atan2((self.target[1] - pos.y_val) , (self.target[0] - pos.x_val)))
        return target_degree

    def calculate_target_location(self):
        if not self.current_final:
            pos = self.drone.simGetObjectPose("KargoArabasi").position
            self.target = [pos.x_val, pos.y_val, pos.z_val]
        else:
            self.target = self.target_house_pos

    def correctOrientation(self):
        vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        zpos = self.drone.getMultirotorState().kinematics_estimated.position.z_val
        degree = self.calculate_angle()
        yawMode = YawMode(is_rate=False, yaw_or_rate=degree)
        self.drone.moveByVelocityZAsync(
            vel.x_val,
            vel.y_val,
            zpos,
            3,
            yaw_mode=yawMode)
        time.sleep(1 / self.sim_speed)

    def isResetPositionAvaible(self):
        pos = self.drone.simGetObjectPose("KargoArabasi").position
        dist = np.linalg.norm(np.array(self.starting_pos) - np.array([pos.x_val, pos.y_val, pos.z_val]))
        if dist > 20:
            return True
        else:
            return False

    def close(self):
        print("close func called")
        pass
