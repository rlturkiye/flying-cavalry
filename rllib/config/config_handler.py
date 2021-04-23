from .conf_ppo import PPOconf
from .conf_dqn import DQNconf

def getConfig(algo):
    
    if algo == "PPO":
        config = PPOconf()
        return config

    elif algo == "DQN":
        config = DQNconf()
        return config
    
    else:
        print("Config cannot be generated")
