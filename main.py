from game.multiplayer_catch import MultiPlayerCatch
from visualizator.image_state_visualizator import ImageStateVisualizator
import numpy as np


def game_initial_test_demo():
    catch_game_object = MultiPlayerCatch(2, food_spawn_rate=0.5)
    visualizer = ImageStateVisualizator('lol', 24)

    while True:
        action1 = np.random.randint(0, 5)
        action2 = np.random.randint(0, 5)
        catch_game_object.play([action1, action2])
        state = catch_game_object.get_state()
        if catch_game_object.game_is_over:
            break
        visualizer.visualize_state(state)


def single_dqn_test_demo():
    pass


def agent_manager_test_demo():
    pass


if __name__ == '__main__':
    game_initial_test_demo()


# Note: potential problem catch game problem: spawning food when flayer is in top of the board
