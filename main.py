from game.multiplayer_catch import MultiPlayerCatch
from visualizator.image_state_visualizator import ImageStateVisualizator
import numpy as np

catch_game_object = MultiPlayerCatch(2)
visualizer = ImageStateVisualizator('lol', 24)

while True:
    action1 = np.random.randint(0, 5)
    action2 = np.random.randint(0, 5)
    catch_game_object.play([action1, action2])
    state = catch_game_object.get_state()
    if catch_game_object.game_is_over:
        break
    visualizer.visualize_state(state)


# Note: potential problem: spawning food when flayer is in top of the board