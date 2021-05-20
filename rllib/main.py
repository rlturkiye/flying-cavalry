import ray
from ray import tune
from config.config_handler import getConfig
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="DQN", help="RL algorithm")
    parser.add_argument("--network", default="SENSOR", help="Vision network")
    args = parser.parse_args()

    global ALG, NETWORK
    ALG = args.alg
    NETWORK = args.network

if __name__ == "__main__":
    main()

    ray.init(local_mode=True)

    num_actions = 6
    step_length=10
    image_width=256
    image_height=256

    config = getConfig(ALG, NETWORK, num_actions, step_length, image_width, image_height)
    config["lr"] = 1e-4
    config["timesteps_per_iteration"] = 32
    config["learning_starts"] = 32
    config["train_batch_size"] = 32
    config["target_network_update_freq"] = 32
    config["exploration_config"]["epsilon_timesteps"] = 5000
    config["env"] = "drone_env"
    config["num_gpus"] = 1 if torch.cuda.is_available() else 0

    results = tune.run(ALG, config=config)
