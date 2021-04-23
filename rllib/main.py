import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from airgym.envs.drone_env import AirSimDroneEnv
import airgym
from config.config_handler import getConfig
import argparse
import sys
# sys.path.append('D:\RL\RLLIB') # TODO Investigate whether is this necessary or not


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="DQN", help="RL algorithm")
    args = parser.parse_args()

    global ALG
    ALG = args.alg

if __name__ == "__main__":
    main()

    ray.init(local_mode=True)

    register_env("drone_env", 
                 lambda config: AirSimDroneEnv("127.0.0.1", 
                 3,
                 (84, 84, 1),
                 useDepth=False))

    config = getConfig(ALG)
    config["lr"] = 1e-4
    config["env"] = "drone_env"

    results = tune.run(ALG, config=config)