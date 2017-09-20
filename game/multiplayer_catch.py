from .game import Game


class MultiPlayerCatch(Game):

    def __init__(self, nb_players):
        Game.__init__(self)
        self.nb_players = nb_players
        self.reset()

    @property
    def name(self):
        return 'MultiPlayerCatch'


    def reset(self):
        

    def initialize_players(self):
        assert 1 <= self.nb_players <= 5, 'Up to 5 players supported currently'

