#!/usr/bin/env python3

import ray
from ray import tune
from ray.tune import CLIReporter

from config.config_handler import getConfig
from commit_message import intro_repository

import argparse
import torch
import os
import numpy as np

SEED_VALUE = 5
TOTAL_STEP = 1000  # total steps
TOTAL_TRAIN_ITER = 1000000  # how many times network updated


def make_deterministic(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    # see https://github.com/pytorch/pytorch/issues/47672
    cuda_version = torch.version.cuda
    if cuda_version is not None and float(torch.version.cuda) >= 10.2:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
    else:
        torch.set_deterministic(True)  # Not all Operations support this.
    # This is only for Convolution no problem
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="DQN", help="RL algorithm",
                        choices=["DQN", "PPO", "DDPG"])
    parser.add_argument("--network", default="SENSOR", help="Vision network",
                        choices=["RGB", "DEPTH", "SENSOR", "VGG-16"])
    args = parser.parse_args()

    global ALG, NETWORK
    ALG = args.alg
    NETWORK = args.network


if __name__ == "__main__":
    main()
    if torch.cuda.is_available():
        print("## CUDA available")
        print(f"Current device: {torch.cuda.current_device()}")
        print(
            f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("## CUDA not available")

    ray.init(local_mode=True)

    num_actions = 6
    step_length = 10
    image_width = 256
    image_height = 256

    config = getConfig(ALG, NETWORK, num_actions,
                       step_length, image_width, image_height)

    config["lr"] = 1e-4
    config["timesteps_per_iteration"] = 32
    config["learning_starts"] = 32
    config["train_batch_size"] = 32
    config["target_network_update_freq"] = 512
    config["exploration_config"]["epsilon_timesteps"] = 5000
    config["env"] = "drone_env"
    config["num_gpus"] = 1 if torch.cuda.is_available() else 0

    # Determinism
    config['seed'] = SEED_VALUE
    make_deterministic(SEED_VALUE)

    # Stopping criteria

    stopx = {"timesteps_total": TOTAL_STEP}
    # "training_iteration":TOTAL_TRAIN_ITER }

    # Load Checkpoints

    # Load state from checkpoint.
    # if(train_config['checkpoint']):
    #     agent.restore(train_config['checkpoint_path'])
    # if required latedr
    # tune run -> trial_dirname_creator=trial_name_id

    # Setup the stopping condition

    checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
    config_str = intro_repository(config)
    print(config_str)
    results = tune.run(ALG, config=config,
                       checkpoint_freq=5,
                       max_failures=5,
                       log_to_file=["logs_out.txt", "logs_err.txt"],
                       checkpoint_at_end=True,
                       progress_reporter=CLIReporter(
                           metric_columns=["loss", "date", "training_iteration", "timesteps_total"]),
                       local_dir=checkpoint_path,
                       keep_checkpoints_num=10,
                       stop=stopx)
    ray.shutdown()
