import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import numpy as np
from airgym.envs.drone_env import AirSimDroneEnv
import airgym
from config.config_handler import getConfig
import argparse
import sys
import torch
from gym import spaces
#sys.path.append('D:\RL\RLLIB') # TODO Investigate whether is this necessary or not

from CustomModels.CustomModel import CustomNetwork
from ray.rllib.utils.typing import ModelConfigDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="DQN", help="RL algorithm")
    parser.add_argument("--network", default="VGG", help="Vision network")
    args = parser.parse_args()

    global ALG, NETWORK
    ALG = args.alg
    NETWORK = args.network

if __name__ == "__main__":
    main()

    ray.init(local_mode=True)

    num_actions = 9

    env_config = {
        "observation_space": spaces.Dict({
            "img": spaces.Box(0, 255, [1, 84, 84]),
            "quad_vel": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
            "linear_vel": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
            "linear_acc": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
            "angular_vel": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
            "angular_acc": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
            }),
        "action_space": spaces.Discrete(num_actions),
        "use_depth": False,
        "step_length": 10,
        "image_size": [84, 84]
    }    

    register_env("drone_env", 
                 lambda config: AirSimDroneEnv(env_config))

    ModelCatalog.register_custom_model('CustomNetwork', CustomNetwork)

    modelConfig = ModelConfigDict()
    modelConfig["conv_filters"] = [[84, 4, 4],[42, 4, 2],[21, 2, 2]]
    modelConfig["conv_activation"] = "relu"
    modelConfig["fcnet_activation"] = "relu"
    modelConfig["fcnet_hiddens"] = [[21 * 4 * 4 + 5 * 3, 171], [171, num_actions]]

    config = getConfig(ALG)
    config["lr"] = 1e-4
    config["env"] = "drone_env"
    config["num_gpus"] = 1 if torch.cuda.is_available() else 0
    config["model"] = {"custom_model": "CustomNetwork",
                       "custom_model_config": modelConfig
                        }

    results = tune.run(ALG, config=config)