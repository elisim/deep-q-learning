from tqdm import tqdm


class TestResults(object):
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
    def __init__(self, env, agent):
        self._env = env
        self._agent = agent

    def train(self,
              episodes,
              steps_per_episode):

        rewards = 0

        for episode_number in tqdm(range(1, episodes + 1)):
            done = False  # indicating if the returned state is a terminal state
            curr_steps = 0  # init current steps for this episode
            state = self._env.reset()  # get initial state s

            self._agent.episode_start(episode_number)

            while not done and curr_steps < steps_per_episode:
                action = self._agent.training_choose_action(state)

                next_state, reward, done, _ = self._env.step(action)

                self._agent.update_on_step_result(reward, state, action, next_state, done)

                rewards += reward

                # go to next state and increment steps
                state = next_state
                curr_steps += 1
