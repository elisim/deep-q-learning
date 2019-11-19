import numpy as np
import random
from agent import Agent


class TrainResults(object):
    def __init__(self, rewards):
        self._rewards = rewards

    @property
    def rewards(self):
        return self._rewards


class QLearningAgent(Agent):
    def __init__(self,
                 env,
                 epsilon,
                 min_epsilon,
                 epsilon_decay,
                 gamma,
                 alpha):
        self._env = env
        self._q_table = np.zeros([self._env.observation_space.n, self._env.action_space.n])
        self._epsilon = epsilon
        self._min_epsilon = min_epsilon
        self._epsilon_decay = epsilon_decay
        self._gamma = gamma
        self._alpha = alpha

    def training_choose_action(self, state):
        if random.uniform(0, 1) < self._epsilon:
            return self._env.action_space.sample()  # Explore action space
        else:
            return np.argmax(self._q_table[state])  # Exploit learned values

    def update_on_step_result(self, reward, state, action, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self._gamma * np.max(self._q_table[next_state])

        # update rule for q-value
        self._q_table[state, action] = (1 - self._alpha) * self._q_table[state, action] + self._alpha * target

    def episode_start(self, episode_number):
        self._epsilon = max(self._min_epsilon, self._epsilon * (self._epsilon_decay ** episode_number))  # decaying epsilon-greedy probability

    def testing_choose_action(self, state):
        return np.argmax(self._q_table[state])
