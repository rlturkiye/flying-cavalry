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

from CustomModels.jointModel import JointModel
from ray.rllib.utils.typing import ModelConfigDict
from config.model_config import getModelConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="DQN", help="RL algorithm")
    parser.add_argument("--network", default="joint", help="Vision network")
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

    register_env("drone_env", lambda config: AirSimDroneEnv(env_config))

    ModelCatalog.register_custom_model('JointModel', JointModel)

    modelConfig = getModelConfig(NETWORK)

    config = getConfig(ALG)
    config["lr"] = 1e-4
    config["env"] = "drone_env"
    config["num_gpus"] = 1 if torch.cuda.is_available() else 0
    config["model"] = {"custom_model": "JointModel",
                       "custom_model_config": modelConfig
                        }

    results = tune.run(ALG, config=config)