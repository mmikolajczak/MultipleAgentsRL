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
    def play(self, actions):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def game_is_over(self):
        pass

    # might not be need but could be interesting for player
    @abstractmethod
    def game_is_won(self):
        pass

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def nb_actions(self): # for single player
        pass

    @abstractmethod
    def nb_players(self):
        pass

    @abstractmethod
    def get_reward(self):
        pass
