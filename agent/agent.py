from abc import abstractmethod, ABCMeta


class Agent(metaclass=ABCMeta):

    def __init__(self, model, visualizator):
        self._model = model

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def play(self):
        pass
