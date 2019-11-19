from abc import ABC


class Agent(ABC):
    def train(self):
        pass

    def testing_choose_action(self):
        pass
