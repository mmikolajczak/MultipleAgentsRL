from abc import ABCMeta, abstractmethod


class Visualizator(metaclass=ABCMeta):

    def __init__(self, game_name):
        self._game_name = game_name

    @abstractmethod
    def visualize_state(self, state):
        pass

    @property
    def game_name(self):
        return self._game_name
