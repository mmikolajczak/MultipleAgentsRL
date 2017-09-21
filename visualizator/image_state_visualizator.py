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
        interpolation = cv2.INTER_CUBIC if self.zoom >= 1 else cv2.INTER_AREA
        screen = cv2.resize(state, None, fx=self.zoom, fy=self.zoom, interpolation=interpolation)
        screen = np.rot90(screen, 1)
        cv2.imshow(self.game_name, screen)
        cv2.waitKey(150) # for now give user some reasonable visual experience during watching the game
        # TODO: fps/time