from .visualizator import Visualizator
import cv2


class ImageStateVisualizator(Visualizator):

    def __init__(self, game_name, zoom):
        Visualizator.__init__(self, game_name)
        self._zoom = zoom

    @property
    def zoom(self):
        return self._zoom

    def visualize_state(self, state):
        screen = cv2.resize(state, None, self.zoom, self.zoom)
        cv2.imshow(self.game_name, screen)
        # TODO: fps/time