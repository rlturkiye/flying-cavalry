import math
import random
import airsim
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from setuptools import glob
import time
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from ray.rllib.utils.typing import ModelConfigDict
import gym
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from airgym.envs.drone_env import AirSimDroneEnv
import numpy
from ray.rllib.utils.typing import ModelConfigDict


# num_outputs/num_actions = 7
class SensorNetwork(TorchModelV2,nn.Module): 

    def __init__(self, obs_space, action_space, num_outputs, model_config, name): 
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
   
        self.counter = 0
        self.nn_layers = nn.ModuleList()
        self.fc_hidden_dict = model_config["custom_model_config"]["fcnet_hiddens"]
        for i in range(len(self.fc_hidden_dict)) : 
            self.nn_layers.append(nn.Linear(self.fc_hidden_dict[i][0],self.fc_hidden_dict[i][1]))

        if(model_config["custom_model_config"]["fcnet_activation"] == "relu"):
            self.fully_connect_activation = F.relu

    def forward(self, input_dict, state, seq_lens):
        target_dist = input_dict["obs"]["target_dist"]
        linear_vel = input_dict["obs"]["linear_vel"]
        linear_acc = input_dict["obs"]["linear_acc"]
        angular_vel = input_dict["obs"]["angular_vel"]
        angular_acc = input_dict["obs"]["angular_acc"]
        distToGeoFence = input_dict["obs"]["distToGeoFence"]

        target_dist = target_dist.to(torch.float32).to(device)
        linear_vel = linear_vel.to(torch.float32).to(device)
        linear_acc = linear_acc.to(torch.float32).to(device)
        angular_vel = angular_vel.to(torch.float32).to(device)
        angular_acc = angular_acc.to(torch.float32).to(device)
        distToGeoFence = distToGeoFence.to(torch.float32).to(device)
            
        x = target_dist
        x = torch.cat((x, linear_vel), 1)
        x = torch.cat((x, linear_acc), 1)
        x = torch.cat((x, angular_vel), 1)
        x = torch.cat((x, angular_acc), 1)
        x = torch.cat((x, distToGeoFence), 1)

        for j in range(len(self.fc_hidden_dict)-1) : 
            x = self.fully_connect_activation(self.nn_layers[j](x))
        x = self.nn_layers[-1](x)

        return x, []

class OneHotEnv(gym.core.ObservationWrapper):
    # Override `observation` to custom process the original observation
    # coming from the env.
    def observation(self, observation):
        # E.g. one-hotting a float obs [0.0, 5.0[.
        return observation