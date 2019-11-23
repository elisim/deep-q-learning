from tqdm import tqdm
import csv
import logging


class TestResults():
    def __init__(self, rewards, good, bad):
        self._rewards = rewards
        self._good = good
        self._bad = bad

    @property
    def rewards(self):
        return self._rewards

    @property
    def good(self):
        return self._good

    @property
    def bad(self):
        return self._bad


class TestRunner(object):
    def __init__(self, env, agent):
        self._env = env
        self._agent = agent

    def test(self, episodes):
        rewards, good, bad = 0, 0, 0
        for _ in range(episodes):
            state = self._env.reset()
            done = False

            while not done:
                action = self._agent.testing_choose_action(state)
                state, reward, done, _ = self._env.step(action)
                rewards += reward
                if done:
                    if reward > 0:
                        good += 1
                    else:
                        bad += 1

        return TestResults(rewards, good, bad)


class TrainRunner(object):
    def __init__(self, env, agent, csv_path=None):
        self._env = env
        self._agent = agent
        self._q_table_snapshots = {}
        self._csv_path = csv_path

    def train(self,
              episodes,
              steps_per_episode,
              episodes_to_snapshot_q=None,
              logger=logging.getLogger('dummy')):

        stats_log = []
        episodes_to_snapshot_q = [] if episodes_to_snapshot_q is None else episodes_to_snapshot_q
        global_step_number = 0

        for episode_number in range(1, episodes + 1):
            reward = 0

            if episode_number in episodes_to_snapshot_q:
                self._q_table_snapshots[episode_number] = self._agent.get_q_table()
            done = False  # indicating if the returned state is a terminal state
            curr_steps = 0  # init current steps for this episode
            state = self._env.reset()  # get initial state s

            self._agent.episode_start(episode_number)

            while not done and curr_steps < steps_per_episode:
                action = self._agent.training_choose_action(state)

                next_state, reward, done, _ = self._env.step(action)

                self._agent.update_on_step_result(reward, state, action, next_state, done, global_step_number)

                # go to next state and increment steps
                state = next_state
                curr_steps += 1
                global_step_number += 1

            steps_taken = 100 if reward == 0 else curr_steps
            stats_log.append({"episode": episode_number, "reward": reward, "steps": steps_taken})
            logger.info(f'Episode {episode_number} Reward: {reward}  Steps: {curr_steps-1}')

        if self._csv_path is not None:
            with open(self._csv_path, 'w', newline='') as csvfile:
                fieldnames = ['episode', 'reward', "steps"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for episode_data in stats_log:
                    writer.writerow(episode_data)
