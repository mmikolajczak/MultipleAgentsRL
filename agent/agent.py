from abc import abstractmethod, ABCMeta


class Agent(metaclass=ABCMeta):

    def __init__(self, model):
        self._model = model

    @abstractmethod
    def train(self, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1, 0.1], epsilon_rate=0.5,
              visualizer=None, reset_memory=False, save_model=True):
        pass

    @abstractmethod
    def play(self, game, episodes, epsilon, visualizer=None):
        pass
