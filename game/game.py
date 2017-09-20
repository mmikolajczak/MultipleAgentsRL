from abc import abstractmethod, ABCMeta


class Game(metaclass=ABCMeta):

    def __init__(self):
        pass

    @property
    def name(self):
        return 'Game'

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def is_over(self):
        pass

    @abstractmethod
    def get_score(self):
        pass