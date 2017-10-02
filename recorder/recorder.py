from abc import abstractmethod, ABCMeta
import os


class Recorder(metaclass=ABCMeta):

    def __init__(self, recording_path):
        self._recording_path = recording_path
        dir_name = self._recording_path
        os.makedirs(dir_name, exist_ok=True)
        self._additional_game_info = None

    @property
    def additional_game_info(self):
        return self._additional_game_info

    @additional_game_info.setter
    def additional_game_info(self, info_dict):
        self._additional_game_info = info_dict

    @abstractmethod
    def record_state(self, state):
        pass
