from .agent import Agent


class DQNAgent(Agent):

    def __init__(self, model, memory): # TODO memory/experience replay
        Agent.__init__(self, model)

    def train(self):
        raise NotImplemented('Write me pls')

    def play(self):
        raise NotImplemented('Write me pls')
