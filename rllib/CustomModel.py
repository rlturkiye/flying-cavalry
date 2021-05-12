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


# num_outputs/num_actions = 7
class CustomNetwork(TorchModelV2,nn.Module): 

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        
        action_space = gym.spaces.Discrete(num_outputs)
        
        obs_space = gym.spaces.Box(0, 255,[1, 84, 84])

        nn.Module.__init__(self)
        super(CustomNetwork, self).__init__(obs_space, action_space, num_outputs, model_config,name)  
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
   

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
        x = input_dict["obs"] # 32, 84, 84, 1
        x = x.to(torch.float32).to(device)

        for i in range(len(self.conv_filters_dict)):
            x = self.conv_activ(self.nn_layers[i](x))
        x = x.view(x.size(0), -1)

        for j in range(i+1, len(self.conv_filters_dict) + len(self.fc_hidden_dict)-1, 1) : 
            x = self.fully_connect_activation(self.nn_layers[j](x))
        x = self.nn_layers[-1](x)

        return x, []




if __name__ == '__main__':
    batch_size = 1

    num_action_examples = 1
    actions = torch.randn(batch_size * num_action_examples, 4)

    steps = torch.randn(batch_size)

    obs_space = gym.spaces.Box(0, 255,[84,84,1])

     
    from airgym.envs.drone_env import AirSimDroneEnv
    import numpy
    obs = numpy.ones((1, 1, 84, 84))
    obs = numpy.vstack([obs]*32)
    obs = torch.FloatTensor(obs).to(device)
    obs = obs.float()
    print("-----", obs.shape)

    action_space = gym.spaces.Discrete(7)

    from ray.rllib.utils.typing import ModelConfigDict
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