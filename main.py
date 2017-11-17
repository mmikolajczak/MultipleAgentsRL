from game.multiplayer_catch import MultiPlayerCatch
from visualizator.image_state_visualizator import ImageStateVisualizator
from recorder.image_state_recorder import ImageStateRecorder
from agent.dqnagent import DQNAgent
import numpy as np


def game_initial_test_demo():
    catch_game = MultiPlayerCatch(2, board_size=20, food_spawn_rate=1)
    visualizer = ImageStateVisualizator('MPCatch visualization', 2)

    while True:
        action1 = np.random.randint(0, 5)
        action2 = np.random.randint(0, 5)
        state = catch_game.get_state()
        catch_game.play([action1, action2])

        if catch_game.game_is_over:
            break
        visualizer.visualize_state(state)


def get_test_model1():
    # architecture from original 'playing atari...' paper
    # only input size is 4px lower
    from keras.models import Model, load_model
    from keras.layers import Dense, Flatten
    from keras.layers import Convolution2D
    from keras.layers import Input
    from keras.optimizers import Adam

    inputs = Input(shape=(80, 80, 6))
    conv1 = Convolution2D(16, (8, 8), strides=(4, 4), activation='relu')(inputs)
    conv2 = Convolution2D(32, (4, 4), strides=(2, 2), activation='relu')(conv1)
    flatten = Flatten()(conv2)
    fc1 = Dense(256, activation='relu')(flatten)
    outputs = Dense(5, activation='linear')(fc1)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mse'])
    print(model.summary())
    return model


def load_trained_model(path):
    from keras.models import load_model
    model = load_model(path)
    return model


def single_dqn_test_demo():
    catch_game_object = MultiPlayerCatch(1, board_size=20, food_spawn_rate=0.05)
    visualizer = ImageStateVisualizator('MPCatch visualization', 10)
    recorder = ImageStateRecorder('MPCatch_rgb_trained_network_results')

    #model = get_test_model1()
    model = load_trained_model('final.h5')

    agent = DQNAgent(model, 10000)
    #agent.train(catch_game_object, epochs=100000, batch_size=50, gamma=0.9, epsilon=0.1, visualizer=None)
    #agent.play(catch_game_object, 100,  visualizer=visualizer)
    agent.train(catch_game_object, epochs=100000, batch_size=50, gamma=0.9, epsilon=0.1, visualizer=visualizer)  # current version


def agent_manager_test_demo():
    pass


if __name__ == '__main__':
    #game_initial_test_demo()
    single_dqn_test_demo()
    #catch_contrib_test()


# Note: potential problem catch game problem: spawning food when flayer is in top of the board
