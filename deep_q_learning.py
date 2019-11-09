import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tqdm import tqdm

# Todo:
# 1. run the model on cartpole
# 2. check things with ipdb
# 3. add second neural network


class DQNAgent:
    """
    Basic DQN algorithm
    """
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.experience_replay = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
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
        choose an action with decaying 𝜀-greedy method
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model.predict(state)  # predict q-value given state
        import ipdb
        ipdb.set_trace() # what is act_values?
        return np.argmax(act_values[0])  # returns action

    def _sample_batch(self, batch_size):
        """
        sample a minibatch randomly from the experience_replay
        """
        return random.sample(self.experience_replay, batch_size)

    def _replay(self, batch_size):
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

    def train_agent(self,
                    episodes,
                    steps_per_episode,
                    batch_size,
                    ):
        """
        train the agent with the DQN algorithm
        """
        for i in tqdm(range(1, episodes+1)):
            state = self.env.reset() # get initial state s
            import ipdb
            ipdb.set_trace()  # what is state shape now?
            state = np.reshape(state, [1, self.state_size])
            for step in range(steps_per_episode):
                # select action using 𝜀-greedy method
                action = self._sample_action(state)

                # execute action in emulator and observe reward, next state, and episode termination signal
                next_state, reward, done, _ = self.env.step(action)

                # store transition in replay memory
                self.experience_replay.append((state, action, reward, next_state, done))
                state = np.reshape(next_state, [1, self.state_size])

                if done:
                    break

                if len(self.experience_replay) > batch_size:
                    self._replay(batch_size)

    def test_agent(self):
        """
        test the agent on a new episode with the trained model
        """
        pass

