import numpy as np
import random
import csv
import gym
import optuna


class QLearningAgent:
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
        self._q_table_snapshots = {}

    def training_choose_action(self, state):
        if random.uniform(0, 1) < self._epsilon:
            return self._env.action_space.sample()  # Explore action space
        else:
            return np.argmax(self._q_table[state])  # Exploit learned values

    def update_on_step_result(self, reward, state, action, next_state, done, global_step_number):
        if done:
            target = reward
        else:
            target = reward + self._gamma * np.max(self._q_table[next_state])

        # update rule for q-value
        self._q_table[state, action] = (1 - self._alpha) * self._q_table[state, action] + self._alpha * target

    def episode_start(self, episode_number):
        self._epsilon = max(self._min_epsilon, self._epsilon * (
                    self._epsilon_decay ** episode_number))  # decaying epsilon-greedy probability

    def testing_choose_action(self, state):
        return np.argmax(self._q_table[state])

    def get_q_table(self):
        return self._q_table.copy()

    def train(self, episodes_to_snapshot_q=None, episodes=5000, steps_per_episode=100, csv_path=None):
        stats_log = []
        episodes_to_snapshot_q = [] if episodes_to_snapshot_q is None else episodes_to_snapshot_q
        global_step_number = 0

        for episode_number in range(1, episodes + 1):
            reward = 0

            if episode_number in episodes_to_snapshot_q:
                self._q_table_snapshots[episode_number] = self.get_q_table()
            done = False  # indicating if the returned state is a terminal state
            curr_steps = 0  # init current steps for this episode
            state = self._env.reset()  # get initial state s

            self.episode_start(episode_number)

            while not done and curr_steps < steps_per_episode:
                action = self.training_choose_action(state)

                next_state, reward, done, _ = self._env.step(action)

                self.update_on_step_result(reward, state, action, next_state, done, global_step_number)

                # go to next state and increment steps
                state = next_state
                curr_steps += 1
                global_step_number += 1

            steps_taken = 100 if reward == 0 else curr_steps
            stats_log.append({"episode": episode_number, "reward": reward, "steps": steps_taken})

        if csv_path is not None:
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['episode', 'reward', "steps"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for episode_data in stats_log:
                    writer.writerow(episode_data)

    def test(self, episodes):
        rewards, good, bad = 0, 0, 0
        for _ in range(episodes):
            state = self._env.reset()
            done = False

            while not done:
                action = self.testing_choose_action(state)
                state, reward, done, _ = self._env.step(action)
                rewards += reward
                if done:
                    if reward > 0:
                        good += 1
                    else:
                        bad += 1

        return good, bad
    
    
def optimize(n_trials=500):
    def objective(trial):
        """
        Define an objective function to be minimized.
        """
        frozen_lake_env = gym.make("FrozenLake-v0").env
        model_params = {
              "alpha": trial.suggest_uniform("alpha", 0.2, 0.3),
              "gamma": trial.suggest_uniform("gamma", 0.8, 0.999),
              "min_epsilon": trial.suggest_uniform("min_epsilon", 0.01, 0.1),
              "epsilon_decay": trial.suggest_uniform("epsilon_decay", 0.8, 0.99999)
        }
        agent = QLearningAgent(frozen_lake_env, epsilon=1, **model_params)
        agent.train()
        good, bad = agent.test(1000)
        return bad # A objective value linked with the Trial object.
    
    study = optuna.create_study()  # Create a new study.
    optuna.logging.disable_default_handler()
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # Invoke optimization of the objective function.
    return study.best_params


if __name__ == '__main__':
    frozen_lake_env = gym.make("FrozenLake-v0").env

    agent = QLearningAgent(frozen_lake_env,
                           alpha=0.279458040604177,
                           gamma=0.9932380015539644,
                           epsilon=1,
                           min_epsilon=0.08832403884569218,
                           epsilon_decay=0.9999392953141722)
    agent.train(episodes_to_snapshot_q=[500, 2000])

    print("Done training, testing on 1000 episodes:")
    good, bad = agent.test(1000)

    print(f'{good} good games, {bad} bad games')
