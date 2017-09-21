from .visualizator import Visualizator
import cv2
import numpy as np


class ImageStateVisualizator(Visualizator):

    def __init__(self, game_name, zoom):
        Visualizator.__init__(self, game_name)
        self._zoom = zoom

    @property
    def zoom(self):
        return self._zoom

    def visualize_state(self, state):
        screen = cv2.resize(state, None, fx=self.zoom, fy=self.zoom)
        screen = np.rot90(screen, 1)
        cv2.imshow(self.game_name, screen)
        cv2.waitKey(150)
        # TODO: fps/time