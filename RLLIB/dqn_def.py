import ray
from ray import tune
#from env import DroneEnv
from ray.tune.registry import register_env
from airgym.envs.drone_env import AirSimDroneEnv
import airgym

if __name__ == "__main__":
    ray.init(local_mode=True)

    register_env("drone_env", 
                 lambda config: AirSimDroneEnv("127.0.0.1", 
                 0.25,
                (84, 84, 1)))

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

    results = tune.run("DQN", config=config)