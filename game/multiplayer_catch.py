import numpy as np
from .game import Game
from enum import IntEnum
import cv2


class MultiPlayerCatch(Game):

    def __init__(self, nb_players, board_size=40, food_spawn_rate=0.05, score_to_win=10):
        Game.__init__(self)
        assert 1 <= nb_players <= 4, 'Up to 4 players supported currently'
        self._nb_players = nb_players
        self._board_size = board_size
        self._score_to_win = score_to_win
        self._food_spawn_rate = food_spawn_rate
        self.reset()

    @property
    def name(self):
        return 'MultiPlayerCatch'

    @property
    def nb_players(self):
        return self._nb_players

    @property
    def nb_actions(self):
        return 5

    @property
    def game_is_over(self):
        return self._game_is_over

    @property
    def game_is_won(self):
        return self._current_score >= self._score_to_win

    def reset(self):
        self._state = np.zeros(shape=(self._board_size, self._board_size, 3))
        self._previous_round_score = 0
        self._current_score = 0
        self._initialize_players()
        self._foods = [] # food is that thing that fly from the sky and agent is supposed to catch it
        self._game_is_over = False
        self._cnt = 0 # used for speed management, players are 3 times faster than that dropping stuff
        self._update_information_about_state()

    def _initialize_players(self):
        available_colors = [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255]] # TODO: magic const
        self._players = []
        for i in range(self.nb_players):
            x = np.random.randint(0, self._board_size)
            y = np.random.randint(0, self._board_size)
            color = available_colors[i]
            new_player = MPCPlayer(x, y, color)
            self._players.append(new_player)

    def play(self, actions):
        assert len(actions) == self._nb_players, 'Actions for all players must be provided'
        self._update_players_positions(actions)
        if self._cnt % 3 == 0:
            self._update_food_positions()
            self._spawn_new_food()
        self._update_information_about_state()
        self._cnt += 1

    def _update_players_positions(self, actions):
        for player, action in zip(self._players, actions):
            # print(action)
            assert action in range(len(MPCActions)), 'Invalid action value'
            # update position based on action
            if action == MPCActions.IDLE:
                new_x, new_y = player.x, player.y
            elif action == MPCActions.UP:
                new_x, new_y = player.x, player.y + 1
            elif action == MPCActions.DOWN:
                new_x, new_y = player.x, player.y - 1
            elif action == MPCActions.LEFT:
                new_x, new_y = player.x - 1, player.y
            elif action == MPCActions.RIGHT:
                new_x, new_y = player.x + 1, player.y

            # check if new position isn't out of board, or  - if so then stay in the same place
            if not self._is_position_on_board(new_x, new_y):
                new_x, new_y = player.x, player.y

            # It has flaws but for now to avoid players in one point, similar check as the one above is performed
            if self._is_position_collide_with_other_players(player, new_x, new_y):
                new_x, new_y = player.x, player.y

            # food catching and score update
            self._previous_round_score = self._current_score
            if [new_x, new_y] in self._foods:
                # self._foods.remove([new_y, new_y]) looks like remove uses 'is' instead of '==' so it won't work
                del self._foods[self._foods.index([new_x, new_y])]
                self._current_score += 1

            # all check done, update player state
            player.x, player.y = new_x, new_y

    def _update_food_positions(self):
        for food in self._foods:
            food[1] -= 1
            if food[1] < 0:
                self._game_is_over = True

    def _spawn_new_food(self):
        if np.random.rand() < self._food_spawn_rate or not len(self._foods):
            food_x = np.random.randint(0, self._board_size)
            food_y = self._board_size - 1
            self._foods.append([food_x, food_y])

    def _update_information_about_state(self):
        new_state = np.empty((self._board_size, self._board_size, 3), np.uint8)
        new_state.fill(255)  # board is white

        for food in self._foods:  # this way of setting values in numpy array can be probably optimised if needed
            new_state[food[0], food[1]] = [255, 0, 0]

        for player in self._players:
            new_state[player.x, player.y] = player.color

        self._state = new_state

    def _is_position_on_board(self, x, y):
        return 0 <= x <= self._board_size - 1 and 0 <= y <= self._board_size - 1

    def _is_position_collide_with_other_players(self, player, new_x, new_y):
        other_players_positions = [[pl.x, pl.y] for pl in self._players if pl is not player]
        return [new_x, new_y] in other_players_positions

    def get_state(self):
        # resized for nn - temporary
        # state_gray = cv2.cvtColor(self._state, cv2.COLOR_BGR2GRAY)
        state_gray = self._state
        resized = cv2.resize(state_gray, None, None, fx=4, fy=4)
        return resized

    def get_score(self):
        return self._current_score

    # This one shouldn't be here
    def get_reward(self):
        if self.game_is_over:
            return -2
        elif self._previous_round_score < self._current_score:
            return self._current_score - self._previous_round_score
        else:
            return 0


class MPCActions(IntEnum):
    IDLE = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    RIGHT = 4


class MPCPlayer:

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
