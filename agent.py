from abc import ABC


class Agent(ABC):
    def train(self):
        pass

    def choose_action(self):
        pass
