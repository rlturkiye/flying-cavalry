import numpy as np
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from CustomModels.customNetwork import CustomNetwork
from CustomModels.sensorNetwork import SensorNetwork
from CustomModels.onlyDqnNetwork import OnlyDqnNetwork
from gym import spaces
from ray.tune.registry import register_env
from airgym.envs.drone_env import AirSimDroneEnv

def vgg16(num_actions, inChannel):
    modelConfig = ModelConfigDict()
    modelConfig["custom_model"] = "CustomNetwork"
    modelConfig["inChannel"] = inChannel
    """modelConfig["conv_filters"] = [[64, 3, 1, 1, [0]], #[out_channels, kernel_size, stride, padding, [pooling (max), kernel, stride]]
                                   [64, 3, 1, 1, [1, 2, 2]],
                                   [128, 3, 1, 1, [0]],
                                   [128, 3, 1, 1, [1, 2, 2]],
                                   [256, 3, 1, 1, [0]],
                                   [256, 3, 1, 1, [0]],
                                   [256, 3, 1, 1, [1, 2, 2]],
                                   [512, 3, 1, 1, [0]],
                                   [512, 3, 1, 1, [0]],
                                   [512, 3, 1, 1, [1, 2, 2]],
                                   [512, 3, 1, 1, [0]],
                                   [512, 3, 1, 1, [0]],
                                   [512, 3, 1, 1, [1, 2, 2]]]   """
    modelConfig["conv_filters"] = [[64, 3, 1, 1, [0]], # [out_channels, kernel_size, stride, padding, [pooling (max), kernel, stride]]
                                   [64, 3, 1, 1, [1, 2, 2]],
                                   [128, 3, 1, 1, [0]],
                                   [128, 3, 1, 1, [1, 2, 2]],
                                   [256, 3, 1, 1, [0]],
                                   [256, 3, 1, 1, [1, 2, 2]]]   
    modelConfig["conv_activation"] = "relu"
    modelConfig["fcnet_hiddens"] = [[256 * 32 * 32 + 5 * 3 + 4, 64], [64, num_actions]]
    modelConfig["fcnet_activation"] = "relu"

    return modelConfig
    

def jointModel(num_actions, inChannel):
    modelConfig = ModelConfigDict()
    modelConfig["custom_model"] = "CustomNetwork"
    modelConfig["inChannel"] = inChannel
    modelConfig["conv_filters"] = [[32, 4, 4, 0, [0]],
                                   [64, 4, 2, 0, [0]],
                                   [128, 2, 2, 0, [0]]]
    modelConfig["conv_activation"] = "relu"
    modelConfig["fcnet_hiddens"] = [[128 * 15 * 15 + 5 * 3 + 4, 64], [64, num_actions]]
    modelConfig["fcnet_activation"] = "relu"

    return modelConfig

def rgbModel(num_actions, inChannel):
    modelConfig = ModelConfigDict()
    modelConfig["custom_model"] = "OnlyDqnNetwork"
    modelConfig["inChannel"] = inChannel
    modelConfig["conv_filters"] = [[32, 4, 4, 0, [0]],
                                   [64, 4, 2, 0, [0]],
                                   [128, 2, 2, 0, [0]]]
    modelConfig["conv_activation"] = "relu"
    modelConfig["fcnet_hiddens"] = [[2048, 64], [64, num_actions]]
    modelConfig["fcnet_activation"] = "relu"

    return modelConfig


def sensorModel(num_actions):
    modelConfig = ModelConfigDict()
    modelConfig["custom_model"] = "SENSOR"
    modelConfig["fcnet_hiddens"] = [[5 * 3 + 4, 32], [32, 64], [64, 128], [128, 64], [64, 32], [32, num_actions]]
    modelConfig["fcnet_activation"] = "relu" 

    return modelConfig


def registerEnv(num_actions, use_depth, step_length, image_width, image_height, sim_speed, map):
    env_config = {
        "observation_space": spaces.Dict({
            "img": spaces.Box(0, 255, [1, image_width, image_height]) if use_depth else spaces.Box(0, 255, [3, image_width, image_height]),
            "target_dist": spaces.Box(low=-2048, high=2048, shape=(4,), dtype=np.float64),
            "linear_vel": spaces.Box(low=-32, high=32, shape=(3,), dtype=np.float64),
            "linear_acc": spaces.Box(low=-32, high=32, shape=(3,), dtype=np.float64),
            "angular_vel": spaces.Box(low=-32, high=32, shape=(3,), dtype=np.float64),
            "angular_acc": spaces.Box(low=-256, high=256, shape=(3,), dtype=np.float64),
            "distToGeoFence": spaces.Box(low=-2048, high=2048, shape=(3,), dtype=np.float64),
            }),
        "action_space": spaces.Discrete(num_actions),
        "use_depth": use_depth,
        "step_length": step_length,
        "image_size": [image_width, image_height],
        "onlySensor": False,
        "sim_speed": sim_speed,
        "map": map
    }
    register_env("drone_env", lambda config: AirSimDroneEnv(env_config))


def registerSensorEnv(num_actions, step_length, sim_speed):
    env_config = {
        "observation_space": spaces.Dict({
            "target_dist": spaces.Box(low=-2048, high=2048, shape=(4,), dtype=np.float64),
            "linear_vel": spaces.Box(low=-32, high=32, shape=(3,), dtype=np.float64),
            "linear_acc": spaces.Box(low=-32, high=32, shape=(3,), dtype=np.float64),
            "angular_vel": spaces.Box(low=-32, high=32, shape=(3,), dtype=np.float64),
            "angular_acc": spaces.Box(low=-256, high=256, shape=(3,), dtype=np.float64),
            "distToGeoFence": spaces.Box(low=-2048, high=2048, shape=(3,), dtype=np.float64),
            }),
        "action_space": spaces.Discrete(num_actions),
        "step_length": step_length,
        "onlySensor": True,
        "sim_speed": sim_speed,
        "map": map
    }
    register_env("drone_env", lambda config: AirSimDroneEnv(env_config))


def registerEnvGetModelConfig(network, num_actions = 9, step_length=10, image_width=84, image_height=84, sim_speed=1, map="Default"):
    if network == "VGG-16":
        registerEnv(num_actions, use_depth=False, step_length=step_length, image_width=image_width, image_height=image_height, sim_speed=sim_speed, map=map)
        ModelCatalog.register_custom_model('CustomNetwork', CustomNetwork)
        modelConfig = vgg16(num_actions, inChannel=3)
    elif network ==  "JOINT":
        registerEnv(num_actions, use_depth=False, step_length=step_length, image_width=image_width, image_height=image_height, sim_speed=sim_speed, map=map)
        ModelCatalog.register_custom_model('CustomNetwork', CustomNetwork)
        modelConfig = jointModel(num_actions, inChannel=3)
    elif network ==  "RGB":
        registerEnv(num_actions, use_depth=False, step_length=step_length, image_width=image_width, image_height=image_height, sim_speed=sim_speed, map=map)
        ModelCatalog.register_custom_model('OnlyDqnNetwork', OnlyDqnNetwork)
        modelConfig = rgbModel(num_actions, inChannel=3)
    elif network == "DEPTH":
        registerEnv(num_actions, use_depth=True, step_length=step_length, image_width=image_width, image_height=image_height, sim_speed=sim_speed, map=map)
        ModelCatalog.register_custom_model('CustomNetwork', CustomNetwork)
        modelConfig = jointModel(num_actions, 1)
    elif network == "SENSOR":
        registerSensorEnv(num_actions, step_length=step_length, sim_speed=sim_speed, map=map)
        ModelCatalog.register_custom_model('SENSOR', SensorNetwork)
        modelConfig = sensorModel(num_actions)
    
    return modelConfig