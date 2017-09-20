import numpy as np
from .game import Game


class MultiPlayerCatch(Game):

    def __init__(self, nb_players, board_size=100):
        Game.__init__(self)
        assert 1 <= self.nb_players <= 4, 'Up to 4 players supported currently'
        self._nb_players = nb_players
        self._board_size = board_size
        self.reset()

    @property
    def name(self):
        return 'MultiPlayerCatch'

    @property
    def nb_players(self):
        return self._nb_players

    @property
    def nb_actions(self):
        return 4

    def reset(self):
        self._state = np.zeros(shape=(self._board_size, self._board_size, 3))
        self._initialize_players()


    def _initialize_players(self):
        available_colors = [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255]]
        self._players = []
        for i in range(self.nb_players):
            self.

    def play(self, actions):
        assert len(actions) == self._nb_players, 'Actions for all players must be provided'

    def get_state(self):
        return self._state