import os
from sys import platform
from typing import Dict
from typing import List

import gym
from gym.vector.utils import spaces
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

from src.utils.gripper_mode import GripperMode

if platform == "darwin":
  # For MacOs
  os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# torch.set_default_tensor_type(torch.DoubleTensor)

# source https://gist.github.com/wassname/ecd2dac6fc8f9918149853d17e3abf02
class LayerNormConv2d(nn.Module):
  """
  Layer norm the just works on the channel axis for a Conv2d
  Ref:
  - code modified from https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/LayerNorm.py
  - paper: https://arxiv.org/abs/1607.06450
  Usage:
      ln = LayerNormConv2d(3)
      x = Variable(torch.rand((1,3,4,2)))
      ln(x).size()
  """
  
  def __init__(self, features, eps=1e-6):
    super().__init__()
    self.register_parameter("gamma", nn.Parameter(torch.ones(features)))
    self.register_parameter("beta", nn.Parameter(torch.zeros(features)))
    self.eps = eps
    self.features = features
  
  def _check_input_dim(self, input):
    if input.size(1) != self.gamma.nelement():
      raise ValueError('got {}-feature tensor, expected {}'
                       .format(input.size(1), self.features))
  
  def forward(self, x):
    self._check_input_dim(x)
    x_flat = x.transpose(1, -1).contiguous().view((-1, x.size(1)))
    mean = x_flat.mean(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
    std = x_flat.std(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
    return self.gamma.unsqueeze(-1).unsqueeze(-1).expand_as(x) * (x - mean) / (std + self.eps) + self.beta.unsqueeze(-1).unsqueeze(-1).expand_as(x)

def weights_init(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    # torch.nn.init.ones_(m.weight.data)
    # torch.nn.init.ones_(m.bias.data)
    torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
    torch.nn.init.normal_(m.bias.data, mean=0.0, std=0.01)

class QNetwork(TorchModelV2, nn.Module):
  
  def __init__(self, obs_space: gym.spaces.Space,
               action_space: gym.spaces.Space, num_outputs: int,
               model_config: ModelConfigDict, name: str, **customized_model_kwargs):
    TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                          model_config, name)
    nn.Module.__init__(self)
    
    conv_stack_nb_channel = 32
    
    self.use_timestep = customized_model_kwargs["use_timestep"]
    self.use_gripper_pos = customized_model_kwargs["use_gripper_pos"]
    
    nb_channel = conv_stack_nb_channel + self.use_timestep
    if self.use_gripper_pos:
      self.gripper_mode = customized_model_kwargs["gripper_mode"]
      if self.gripper_mode == GripperMode.X_Y_Z_alpha.value:
        nb_channel += 4 # for the x,y,z and alpha
    
    # Define the layers here !
    
    self.img_conv_stack = nn.Sequential(
      self.conv(1, conv_stack_nb_channel, 3, 2),
      self.conv(conv_stack_nb_channel, conv_stack_nb_channel, 3, 2),
      self.conv(conv_stack_nb_channel, conv_stack_nb_channel, 3, 2)
    )
    
    self.fc_action = self.fc(4, nb_channel)
    
    self.fc_end = self.fc(7 * 7 * nb_channel, 32)
    self.linear = nn.Linear(32, 1)
    
    # Initialize the network
    self.apply(weights_init)
    
  def fc(self, in_features: int, out_features: int, bias: bool = True):
    return nn.Sequential(nn.Linear(in_features, out_features, bias), torch.nn.LayerNorm(out_features), nn.ReLU())
  
  # TODO : check if LayerNormConv2d is working as expected
  def conv(self, in_channels, out_channels, kernel_size=3, stride=2):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride), LayerNormConv2d(out_channels), nn.ReLU())
  
  def add_actions(self, net, actions):
    """Merges visual perception with context using elementwise addition.

    Actions are reshaped to match net dimension depth-wise, and are added to
    the conv layers by broadcasting element-wise across H, W extent.

    Args:
      net: Tensor of shape [batch_size, H, W, C].
      actions: Tensor of shape [batch_size * num_examples, C].
    Returns:
      Tensor with shape [batch_size * num_examples, H, W, C]
    """
    
    batch_size, d1, h, w = net.shape
    actions_ln, d2, _, _ = actions.shape
    num_examples = actions_ln//batch_size
    assert d1 == d2
    
    actions = actions.view(batch_size, num_examples, d1, 1, 1)
    net = net.unsqueeze(1)
    net = net.repeat(1, num_examples, 1, 1, 1)

    net = torch.add(net, actions)
    
    return net.view(batch_size * num_examples, d1, h, w)

  @override(TorchModelV2)
  def forward(self, input_dict: Dict[str, TensorType],
              state: List[TensorType],
              seq_lens: TensorType) -> (TensorType, List[TensorType]):
    
    # Process the observation
    obs = input_dict["obs"]
    self.test = input_dict
    
    if isinstance(obs, torch.Tensor) and hasattr(self.obs_space, "original_space") and \
        (isinstance(self.obs_space.original_space, spaces.Dict) or isinstance(self.original_space, spaces.Tuple)):
      obs = restore_original_dimensions(obs, self.obs_space, "torch")
      
    img = next(iter(obs["imgs"].values())) # get the first img
    _, height, width, channel = img.shape
    img = img.permute(0, 3, 1, 2) # Permute to torch tensor convention (N, H, W, C) --> (N, C, H, W)
    img = img.float()
    
    steps = obs["t"].float()
    actions = input_dict["actions"].float()

    net = self.img_conv_stack(img)
    
    if self.use_gripper_pos:
      gripper_pose = obs["gripper_pose_from_table"].float()
      if self.gripper_mode == GripperMode.X_Y_Z_alpha.value:
        # make the steps a batch tensor ([batch_size, 4] --> [batch_size,4,1,1]
        gripper_pose = gripper_pose.unsqueeze(-1).unsqueeze(-1)
        gripper_pose = gripper_pose.repeat(1, 1, 7, 7)
        net = torch.cat([net, gripper_pose], 1)
    
    if self.use_timestep:
      # make the steps a batch tensor ([batch_size, 1] --> [batch_size,1,1,1]
      steps = torch.argmax(steps, dim=1) # Revert the one-hot encoding caused by the spaces.Discrete
      steps = steps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
      steps = steps.repeat(1, 1, 7, 7)
      net = torch.cat([net, steps], 1)
    
    actions = self.fc_action(actions)
    # make the actions a batch tensor ([batch_size, nb_actions] --> [batch_size,1,1,nb_actions]
    actions = actions.unsqueeze(-1).unsqueeze(-1)
    
    net = self.add_actions(net, actions)
    net = net.view(net.shape[0], -1) # flatten
    net = self.fc_end(net)
    outputs = self.linear(net)
    
    return outputs, state


if __name__ == '__main__':
  from src.config.camera_conf import get_camera
  from src.custom_env.observation import Observation


  # a = torch.ones((5, 5, 2))
  # a[:, :, 0] = 2
  # a[:, :, 1] = 3
  # print(a[:, :, 0])
  # a = a.permute(2, 0, 1)
  # print(a.shape, a[0, :, :])
  # exit(0)

  # In Pytorch, tensor size -> (N, C_out, H_out, W_out)
  # In PIL (resize parameter (width, height)) -> to numpy (height, width, channel)
  
  batch_size = 1

  num_action_examples = 1
  actions = torch.randn(batch_size * num_action_examples, 4)
  
  steps = torch.randn(batch_size)

  # obs_space = spaces.Dict({"img": spaces.Box(0.0, 255.0, (64, 64, 1), np.float32)})
  # obs = torch.randn(batch_size, 64, 64, 1)
  # obs = obs.view(obs.shape[0], -1)
  
  obs_space = Observation.make_obs_space([get_camera("pmd_0", depth_modes=["Normal"])], downsample_sizes=[(64, 64)])
  original_obs = obs_space.sample()
  obs = torch.tensor(DictFlatteningPreprocessor(obs_space).transform(original_obs))
  obs = obs.repeat(batch_size, 1) # Make it a batch

  action_space = spaces.Box(low=-1, high=1, shape=(4,))
  custom_model_config = {
    "use_timestep": True,
    "use_gripper_pos": False,
    "gripper_mode": GripperMode.X_Y_Z_alpha.value
  }
  net = QNetwork(obs_space=DictFlatteningPreprocessor(obs_space).observation_space,
                 action_space=action_space,
                 num_outputs=1,
                 model_config={},
                 name='CustomNet',
                 **custom_model_config)
  
  print(obs.shape, steps.shape, actions.shape)
  print(net)
  
  inputs = {"obs": obs, "actions": actions}
  out, _ = net.forward(inputs, None, None)
  
  print("Output shape", out.shape)

  # from src.utils.rl_lib.logger import CustomTBXLogger
  # logger = CustomTBXLogger({}, "results/TBXlogger-tests")
  #
  # logger._file_writer.add_graph(net, inputs)

