import numpy
from ray.rllib.utils.typing import ModelConfigDict

def vgg16(num_output):
    modelConfig = ModelConfigDict()
    modelConfig["conv_filters"] = [[64, 3, 1, 1, [0]], #[out_channels, kernel_size, stride, padding, [pooling (max), kernel, stride]]
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
                                   [512, 3, 1, 1, [1, 2, 2]]]   
    modelConfig["conv_activation"] = "relu"
    modelConfig["fcnet_hiddens"] = [[8 * 8 * 512, 171], [171, num_output]]
    modelConfig["fcnet_activation"] = "relu"

    return modelConfig

def jointModel(num_output):
    modelConfig = ModelConfigDict()
    modelConfig["conv_filters"] = [[84, 4, 4, 0, [0]],
                                   [42, 4, 2, 0, [0]],
                                   [21, 2, 2, 0, [0]]]
    modelConfig["conv_activation"] = "relu"
    modelConfig["fcnet_hiddens"] = [[21 * 4 * 4 + 4 * 3, 171], [171, num_output]]
    modelConfig["fcnet_activation"] = "relu"

    return modelConfig

def getModelConfig(network, num_output = 9):
    if network == "VGG-16":
        modelConfig = vgg16(num_output)
        return modelConfig
    elif network ==  "joint":
        modelConfig = jointModel(num_output)
        return modelConfig