from game.multiplayer_catch import MultiPlayerCatch
from visualizator.image_state_visualizator import ImageStateVisualizator
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

    inputs = Input(shape=(80, 80, 3))
    conv1 = Convolution2D(16, (8, 8), strides=(4, 4), activation='relu')(inputs)
    conv2 = Convolution2D(32, (4, 4), strides=(2, 2), activation='relu')(conv1)
    flatten = Flatten()(conv2)
    fc1 = Dense(256, activation='relu')(flatten)
    outputs = Dense(5, activation='linear')(fc1)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mse'])
    print(model.summary())
    return model


def single_dqn_test_demo():
    catch_game_object = MultiPlayerCatch(1, board_size=20, food_spawn_rate=0.05)
    visualizer = ImageStateVisualizator('MPCatch visualization', 10)

    model = get_test_model1()

    agent = DQNAgent(model, 10000)
    agent.train(catch_game_object, epochs=1200, batch_size=50, gamma=0.9, epsilon=0.1, visualizer=visualizer)


def agent_manager_test_demo():
    pass


def catch_contrib_test():
    # something wrong here
    from keras.models import Sequential
    from keras.layers import Flatten, Dense
    from games import Catch
    from keras.optimizers import sgd

    grid_size = 10
    hidden_size = 100
    nb_frames = 1

    model = Sequential()
    model.add(Flatten(input_shape=(nb_frames, grid_size, grid_size)))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(3))
    model.compile(sgd(lr=.2), "mse")

    catch = Catch(grid_size)
    agent = DQNAgent(model=model, memory_size=1000)
    #visualizer = ImageStateVisualizator('test', 20)
    agent.train(catch, epochs=1200, batch_size=10, gamma=0.9)#, visualizer=visualizer)
    #agent.play(catch)


if __name__ == '__main__':
    #game_initial_test_demo()
    single_dqn_test_demo()
    #catch_contrib_test()


# Note: potential problem catch game problem: spawning food when flayer is in top of the board
