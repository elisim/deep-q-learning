import numpy as np
import random
from tqdm import tqdm
from agent import Agent


class TrainResults(object):
    def __init__(self, rewards):
        self._rewards = rewards

    @property
    def rewards(self):
        return self._rewards


class QLearningAgent(Agent):
    def __init__(self, env):
        self._env = env
        self._q_table = np.zeros([self._env.observation_space.n, self._env.action_space.n])

    def train(self, episodes,
              steps_per_episode,
              alpha=0.1,
              gamma=0.6,
              epsilon=1,
              min_epsilon=0.1,
              epsilon_decay=0.999):

        rewards = 0

        for i in tqdm(range(1, episodes + 1)):
            done = False  # indicating if the returned state is a terminal state
            curr_steps = 0  # init current steps for this episode
            state = self._env.reset()  # get initial state s
            epsilon = max(min_epsilon, epsilon * (epsilon_decay ** i))  # decaying epsilon-greedy probability

            while not done and curr_steps < steps_per_episode:
                if random.uniform(0, 1) < epsilon:
                    action = self._env.action_space.sample()  # Explore action space
                else:
                    action = np.argmax(self._q_table[state])  # Exploit learned values

                next_state, reward, done, _ = self._env.step(action)

                rewards += reward
                if done:
                    target = reward
                else:
                    target = reward + gamma * np.max(self._q_table[next_state])

                # update rule for q-value
                self._q_table[state, action] = (1 - alpha) * self._q_table[state, action] + alpha * target

                # go to next state and increment steps
                state = next_state
                curr_steps += 1

        return TrainResults(rewards)

    def choose_action(self, state):
        return np.argmax(self._q_table[state])
