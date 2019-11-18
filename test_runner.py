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
                action = self._agent.choose_action(state)
                state, reward, done, _ = self._env.step(action)
                rewards += reward
                if done:
                    if reward > 0:
                        good += 1
                    else:
                        bad += 1

        return TestResults(rewards, good, bad)
