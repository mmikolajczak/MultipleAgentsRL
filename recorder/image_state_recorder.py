from .recorder import Recorder
import cv2
import os


class ImageStateRecorder(Recorder):

    def __init__(self, recording_path):
        Recorder.__init__(self, recording_path)
        self._current_frame = 0
        self._last_game_no = -1

    def record_state(self, state):
        if self._last_game_no != self.additional_game_info['game_no']:
            self._current_frame = 0
            self._last_game_no = self.additional_game_info['game_no']
        else:
            self._current_frame += 1
        filename = '{}_game_{}_frame_{}_score_{}.png'.format(self.additional_game_info['game_name'],
                                                    self.additional_game_info['game_no'], self._current_frame,
                                                             self.additional_game_info['game_score'])
        state_save_full_path = os.path.join(self._recording_path, filename)
        cv2.imwrite(state_save_full_path, state)
