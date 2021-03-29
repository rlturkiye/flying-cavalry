from airgym.envs.drone_env import AirSimDroneEnv
import airgym

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

if __name__ == "__main__":
    ray.init(local_mode=True)

    """ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel if args.torch else RNNModel)"""

    register_env("drone_env", 
                 lambda config: AirSimDroneEnv("127.0.0.1", 
                 3,
                 (84, 84, 1),
                 useDepth=False))

    config = {
        "env": "drone_env", 
        "env_config": {
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "lr": 1e-4,  
        #"num_workers": 1,  
        "framework": "torch",
        "log_level" : "DEBUG"
    }

    results = tune.run("DDQN", config=config)