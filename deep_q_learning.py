import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tqdm import tqdm

# turn off warnings and tensorflow logging  
import tensorflow as tf
import warnings
tf.get_logger().setLevel(tf.logging.ERROR)
warnings.filterwarnings("ignore")


__all__ = ['DQNAgent']

# Todo:
# 1. add second neural network


class DQNAgent:
    """
    Basic DQN algorithm
    """
    def __init__(self,
                 env,
                 gamma=0.95,
                 epsilon=1.0,
                 min_epsilon=0.01,
                 epsilon_decay=0.995,
                 learning_rate=0.001,):
        """
        :param env: Open AI env
        :param gamma: discount factor 𝛾,
        :param epsilon: initial epsilon
        :param min_epsilon: min epsilon rate (end of decaying)
        :param epsilon_decay: decay rate for decaying epsilon-greedy probability
        :param learning_rate: learning rate for neural network optimizer
        """
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.experience_replay = deque(maxlen=2000)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """
        Neural Network for Q-value approximation.
        The network takes a state as an input (or a minibatch of states)
        and output the predicted q-value of each action for that state.
        """
        model = Sequential()
        model.add(Dense(units=32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _sample_action(self, state):
        """
        choose an action with decaying 𝜀-greedy method, given state 'state'
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()

        q_values = self.model.predict(state)[0]  # predict q-value given state
        return np.argmax(q_values)  # return action with max q-value

    def _sample_batch(self, batch_size):
        """
        sample a minibatch randomly from the experience_replay in 'batch_size' size
        """
        return random.sample(self.experience_replay, batch_size)

    def _replay(self, batch_size):
        """
        sample random minibatch, update y, and perform gradient descent step
        """
        # wait for 'experience_replay' to contain at least 'batch_size' transitions
        if len(self.experience_replay) <= batch_size:
            return

        minibatch = self._sample_batch(batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:  # for terminal transition
                target = reward
            else:  # for non-terminal transition
                target = (reward + self.gamma*np.max(self.model.predict(next_state)[0]))

            # update y
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # perform a gradient descent step
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # decaying epsilon-greedy probability
        self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)

    def _correct_state_size(self, state):
        """
        correct state size from (state_size,) to (1, state_size) for the network
        """
        return np.reshape(state, [1, self.state_size])

    def train_agent(self,
                    episodes,
                    steps_per_episode,
                    batch_size,
                    ):
        """
        train the agent with the DQN algorithm

        :param episodes: number of episodes
        :param steps_per_episode: max steps per episode
        :param batch_size: batch size
        """
        for i in tqdm(range(1, episodes+1)):
            # get initial state s
            state = self._correct_state_size(self.env.reset())

            for step in range(steps_per_episode):
                # select action using 𝜀-greedy method
                action = self._sample_action(state)

                # execute action in emulator and observe reward, next state, and episode termination signal
                next_state, reward, done, _ = self.env.step(action)
                next_state = self._correct_state_size(next_state)

                # store transition in replay memory
                self.experience_replay.append((state, action, reward, next_state, done))

                # update current state to next state
                state = next_state

                # break episode on terminal state
                if done:
                    break

                # sample random minibatch, update y, and perform gradient descent step
                self._replay(batch_size)

    def test_agent(self, episodes):
        """
        test the agent on a new episode with the trained model
        :param episodes: number of episodes
        """
        for _ in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                state = self._correct_state_size(state)
                action = np.argmax(self.model.predict(state)[0])
                state, reward, done, _ = self.env.step(action)
                self.env.render()
