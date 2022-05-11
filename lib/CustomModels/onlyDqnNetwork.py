import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from ray.rllib.utils.typing import ModelConfigDict
import gym
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import numpy
from ray.rllib.utils.typing import ModelConfigDict

# num_outputs/num_actions = 7
class OnlyDqnNetwork(TorchModelV2,nn.Module): 

    def __init__(self, obs_space, action_space, num_outputs, model_config, name): 
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
   
        self.counter = 0
        self.nn_layers = nn.ModuleList()
        self.conv_filters_dict = model_config["custom_model_config"]["conv_filters"]
        input_c = model_config["custom_model_config"]["inChannel"]
        for conv in range(len(self.conv_filters_dict)): 
            self.nn_layers.append(nn.Conv2d(input_c, self.conv_filters_dict[conv][0], kernel_size= self.conv_filters_dict[conv][1], stride=self.conv_filters_dict[conv][2], padding=self.conv_filters_dict[conv][3] ))
            if self.conv_filters_dict[conv][4][0] == 1:
                self.nn_layers.append(nn.MaxPool2d(kernel_size=self.conv_filters_dict[conv][4][1], stride=self.conv_filters_dict[conv][4][2]))
                self.counter += 1
            input_c = self.conv_filters_dict[conv][0]
            self.counter += 1

        if(model_config["custom_model_config"]["conv_activation"] == "relu"):
            self.conv_activ = F.relu
        
        self.fc_hidden_dict = model_config["custom_model_config"]["fcnet_hiddens"]
        input_c = 1
        for i in range(len(self.fc_hidden_dict)) : 
            self.nn_layers.append(nn.Linear(self.fc_hidden_dict[i][0],self.fc_hidden_dict[i][1]))

        if(model_config["custom_model_config"]["fcnet_activation"] == "relu"):
            self.fully_connect_activation = F.relu
        
        # Holds the current "base" output (before logits layer).
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["img"] # 32, inChannel, img_shape, img_shape

        x = x.to(torch.float32).to(device)

        for i in range(self.counter):
            x = self.conv_activ(self.nn_layers[i](x))
            
        x = x.view(x.size(0), -1)

        for j in range(i+1, self.counter + len(self.fc_hidden_dict)-1, 1): 
            x = self.fully_connect_activation(self.nn_layers[j](x))
        x = self.nn_layers[-1](x)
        self._features = x

        return x, []

    def value_function(self):
        return torch.reshape(torch.mean(self._features, -1), [-1])

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

    obs_space = gym.spaces.Box(0, 255,[1,84,84])

    obs = numpy.ones((1, 3, 84, 84))
    obs = numpy.vstack([obs]*32)
    obs = torch.FloatTensor(obs).to(device)
    obs = obs.float()
    print("-----", obs.shape)

    action_space = gym.spaces.Discrete(7)

    deneme = ModelConfigDict()
    deneme["custom_model_config"] = {}
    deneme["custom_model_config"]["conv_filters"] = [[32, 4, 4, 0, [0]],
                                                    [64, 4, 2, 0, [0]],
                                                    [128, 2, 2, 0, [0]]]
    deneme["custom_model_config"]["conv_activation"] = "relu"
    deneme["custom_model_config"]["fcnet_activation"] = "relu"
    deneme["custom_model_config"]["fcnet_hiddens"] =[[2048, 64], [64, 7]]
    deneme["custom_model_config"]["inChannel"] = 3

    dqn = OnlyDqnNetwork(obs_space, action_space, 7, deneme, "name")
    dqn.to(device)
  
    print(obs.shape, steps.shape, actions.shape)
    print(dqn)
    
    inputs = {"obs": {"img": obs}}
    out, _ = dqn.forward(inputs, None, None)
    
    print("Output shape", out.shape)
