import ray
from ray import tune
from ray.tune import CLIReporter

from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.dqn import DQNTorchPolicy, DQNTrainer

from config.config_handler import getConfig
from commit_message import intro_repository

import argparse
import torch
import os
import numpy as np
import subprocess
import gym

SEED_VALUE = 5
TOTAL_STEP = 7000000 # total steps 
TOTAL_TRAIN_ITER = 2 # how many times network updated

def make_deterministic(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    # see https://github.com/pytorch/pytorch/issues/47672
    cuda_version = torch.version.cuda
    if cuda_version is not None and float(torch.version.cuda) >= 10.2:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
    else:
        torch.set_deterministic(True)  #Not all Operations support this.
    torch.backends.cudnn.deterministic = True #This is only for Convolution no problem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="DQN", help="RL algorithm")
    parser.add_argument("--network", default="RGB", help="Vision network")
    args = parser.parse_args()

    global ALG, NETWORK
    ALG = args.alg
    NETWORK = args.network

GOOGLE_COLLAB = False

if __name__ == "__main__":
    # Start The Simulation - Mainly for google collab
    if GOOGLE_COLLAB:
        proc_settings = subprocess.call('mkdir -p /home/RLTurkey/Documents/AirSim/ && cp settings.json /home/RLTurkey/Documents/AirSim', shell=True)
        proc = subprocess.Popen('./content/LinuxNoEditor/RLTurkiyeVersion1.sh', shell=True)
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
    step_length=10
    image_width=256
    image_height=256

    config = getConfig(ALG, NETWORK, num_actions, step_length, image_width, image_height)
    config["lr"] = 1e-4
    config["timesteps_per_iteration"] = 1000
    config["learning_starts"] = 512
    config["train_batch_size"] = 512
    config["target_network_update_freq"] = 512
    config["env"] = "drone_env"
    config["num_gpus"] = 1 if torch.cuda.is_available() else 0
    config["explore"] = False
    config["evaluation_num_episodes"] = 1
    config["in_evaluation"] = True
    config["evaluation_num_episodes"]= 1


    
    # Determinism 
    config['seed'] = SEED_VALUE
    make_deterministic(SEED_VALUE)

    # Stopping criteria

    stopx = {"training_iteration":TOTAL_TRAIN_ITER }

    # Load Checkpoints

    # Load state from checkpoint.
    # if(train_config['checkpoint']):
    #     agent.restore(train_config['checkpoint_path'])
    # if required latedr                        
    # tune run -> trial_dirname_creator=trial_name_id, trial_name_creator=trial_name_id,

     # Setup the stopping condition

    checkpoint_path = os.path.join(os.getcwd(), "checkpoints/DQN/")
    if GOOGLE_COLLAB:
        checkpoint_path = os.path.join("/content/drive/MyDrive/","checkpoints")
    #config_str = intro_repository(config)
    #print(config_str)
  

    agent = DQNTrainer(config=config, env=config["env"])
    agent.restore(checkpoint_path)
    env = gym.make(config["env"])

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    ray.shutdown()

