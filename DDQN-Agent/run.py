from agent import Agent as DQN_Agent
from ddqn_agent import DDQN_Agent

if __name__ == "__main__":
    ddqn_agent = DDQN_Agent(useDepth=False)
    ddqn_agent.train()
    #ddqn_agent.test_agent()
