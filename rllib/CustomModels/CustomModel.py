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
class CustomNetwork(TorchModelV2,nn.Module): 

    def __init__(self, obs_space, action_space, num_outputs, model_config, name): 
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        #super(CustomNetwork, self).__init__(obs_space, action_space, num_outputs, model_config,name) 
   

        self.nn_layers = nn.ModuleList()
        self.conv_filters_dict = model_config["custom_model_config"]["conv_filters"]
        input_c = 1
        for conv in range(len(self.conv_filters_dict)): 
            self.nn_layers.append(nn.Conv2d(input_c, self.conv_filters_dict[conv][0], kernel_size= self.conv_filters_dict[conv][1], stride= self.conv_filters_dict[conv][2]))
            input_c = self.conv_filters_dict[conv][0]

        if(model_config["custom_model_config"]["conv_activation"] == "relu"):
            self.conv_activ = F.relu
        
        self.fc_hidden_dict = model_config["custom_model_config"]["fcnet_hiddens"]
        input_c = 1
        for i in range(len(self.fc_hidden_dict)) : 
            self.nn_layers.append(nn.Linear(self.fc_hidden_dict[i][0],self.fc_hidden_dict[i][1]))

        if(model_config["custom_model_config"]["fcnet_activation"] == "relu"):
            self.fully_connect_activation = F.relu

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["img"] # 32, 1, 84, 84
        quad_vel = input_dict["obs"]["quad_vel"]
        linear_vel = input_dict["obs"]["linear_vel"]
        linear_acc = input_dict["obs"]["linear_acc"]
        angular_vel = input_dict["obs"]["angular_vel"]
        angular_acc = input_dict["obs"]["angular_acc"]

        x = x.to(torch.float32).to(device)
        quad_vel = quad_vel.to(torch.float32).to(device)
        linear_vel = linear_vel.to(torch.float32).to(device)
        linear_acc = linear_acc.to(torch.float32).to(device)
        angular_vel = angular_vel.to(torch.float32).to(device)
        angular_acc = angular_acc.to(torch.float32).to(device)

        for i in range(len(self.conv_filters_dict)):
            x = self.conv_activ(self.nn_layers[i](x))
            
        x = x.view(x.size(0), -1)
        x = torch.cat((x, quad_vel), 1)
        x = torch.cat((x, linear_vel), 1)
        x = torch.cat((x, linear_acc), 1)
        x = torch.cat((x, angular_vel), 1)
        x = torch.cat((x, angular_acc), 1)

        for j in range(i+1, len(self.conv_filters_dict) + len(self.fc_hidden_dict)-1, 1) : 
            x = self.fully_connect_activation(self.nn_layers[j](x))
        x = self.nn_layers[-1](x)

        return x, []

class OneHotEnv(gym.core.ObservationWrapper):
    # Override `observation` to custom process the original observation
    # coming from the env.
    def observation(self, observation):
        # E.g. one-hotting a float obs [0.0, 5.0[.
        return observation


if __name__ == '__main__':
    batch_size = 1

    num_action_examples = 1
    actions = torch.randn(batch_size * num_action_examples, 4)

    steps = torch.randn(batch_size)

    obs_space = gym.spaces.Box(0, 255,[84,84,1])

    obs = numpy.ones((1, 1, 84, 84))
    obs = numpy.vstack([obs]*32)
    obs = torch.FloatTensor(obs).to(device)
    obs = obs.float()
    print("-----", obs.shape)

    action_space = gym.spaces.Discrete(7)

    deneme = ModelConfigDict()
    deneme["custom_model_config"] = {}
    deneme["custom_model_config"]["conv_filters"] = [[84, 4, 4],[42,4,2],[21,2,2]]
    deneme["custom_model_config"]["conv_activation"] = "relu"
    deneme["custom_model_config"]["fcnet_activation"] = "relu"
    deneme["custom_model_config"]["fcnet_hiddens"] = [[21 * 4 * 4 , 171],[171,7]]

    dqn = CustomNetwork(obs_space, action_space, 7, deneme, "name")
    dqn.to(device)
  
    print(obs.shape, steps.shape, actions.shape)
    print(dqn)
    
    inputs = {"obs": obs}
    out, _ = dqn.forward(inputs, None, None)
    
    print("Output shape", out.shape)