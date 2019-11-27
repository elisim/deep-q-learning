import gym
from double_deep_q_learning import *


if __name__ == '__main__':
    cartpole_env = gym.make("CartPole-v1").env

    agent = DDQNAgent(env=cartpole_env)
    print("Agent Created")

    agent.train_agent(episodes=50000, steps_per_episode=500, batch_size=32)
    print("Training Done")
