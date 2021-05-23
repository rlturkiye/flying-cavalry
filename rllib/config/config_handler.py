from .conf_ppo import PPOconf
from .conf_dqn import DQNconf
from config.model_config import registerEnvGetModelConfig
import torch

def getConfig(algo, NETWORK, num_actions, step_length, image_width, image_height, sim_speed):
    

    if algo == "PPO":
        config = PPOconf()

    elif algo == "DQN":
        config = DQNconf()
    
    else:
        print("Config cannot be generated")
        return None

    modelConfig = registerEnvGetModelConfig(NETWORK, num_actions, step_length, image_width, image_height, sim_speed)    
    config["model"] = {"custom_model": "CustomNetwork" if NETWORK != "SENSOR" else "SENSOR",
                       "custom_model_config": modelConfig
                        }
    return config
