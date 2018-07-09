from game.multiplayer_catch import MultiPlayerCatch
from visualizator.image_state_visualizator import ImageStateVisualizator
from recorder.image_state_recorder import ImageStateRecorder
from agent.dqn_agent import DQNAgent
from agent.pre_dqn_agent import DQNPREAgent
from agent.pre_dqn_one_net_multiple_players import DQNPREMultiplayerAgent
from agent.pre_dqn_two_nets_multiple_players import DQNPREMMultiplayerMultinetAgent
from utils.json import load_json_file
from utils.others import get_path_to_file_last_in_numerical_order
import numpy as np
import enum
import os.path as osp


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


def get_test_model(nb_players):
    # architecture from original 'playing atari...' paper
    # only input size is 4px lower
    from keras.models import Model
    from keras.layers import Dense, Flatten
    from keras.layers import Convolution2D
    from keras.layers import Input
    from keras.optimizers import Adam

    inputs = Input(shape=(80, 80, 6))
    conv1 = Convolution2D(16, (8, 8), strides=(4, 4), activation='relu')(inputs)
    conv2 = Convolution2D(32, (4, 4), strides=(2, 2), activation='relu')(conv1)
    flatten = Flatten()(conv2)
    fc1 = Dense(256, activation='relu')(flatten)
    outputs = Dense(nb_players * 5, activation='linear')(fc1)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['mse'])
    print(model.summary())
    return model


def load_trained_model(path):
    from keras.models import load_model
    model = load_model(path)
    return model


def _single_dqn_test_demo():
    catch_game_object = MultiPlayerCatch(1, board_size=20, food_spawn_rate=0.05)
    visualizer = ImageStateVisualizator('MPCatch visualization', 10)
    recorder = ImageStateRecorder('MPCatch_rgb_trained_network_results')

    model = load_trained_model('final.h5')

    agent = DQNAgent(model, 10000)
    agent.train(catch_game_object, epochs=100000, batch_size=50, gamma=0.9, epsilon=0.1, visualizer=visualizer)  # current version


def _PRE_dqn_training():
    catch_game_object = MultiPlayerCatch(1, board_size=20, food_spawn_rate=0.05)

    config = load_json_file('config.json')
    if config['RESTORE_BACKUP']:
        try:
            model = load_trained_model(config['BACKUP_MODEL_PATH'])
        except (FileNotFoundError, OSError):
            model = get_test_model(1)
        try:
            last_epoch_stats_file_path = get_path_to_file_last_in_numerical_order(config['BACKUP_STATS_DIR_PATH'])
            restored_train_stats = load_json_file(last_epoch_stats_file_path)
        except (FileNotFoundError, OSError, StopIteration):
            restored_train_stats = None
    else:
        model = get_test_model(1)
        restored_train_stats = None

    agent = DQNPREAgent(model, 100000)
    agent.train(catch_game_object, epochs=100000, batch_size=50, gamma=0.9, epsilon=0.1, visualizer=None,
                restored_training_stats=restored_train_stats)


def _PRE_dqn_training_one_net_two_players():
    catch_game_object = MultiPlayerCatch(2, board_size=20, food_spawn_rate=0.05)

    config = load_json_file('config_multiplayer_one_net.json')
    if config['RESTORE_BACKUP']:
        try:
            model = load_trained_model(config['BACKUP_MODEL_PATH'])
        except (FileNotFoundError, OSError):
            model = get_test_model(2)
        try:
            last_epoch_stats_file_path = get_path_to_file_last_in_numerical_order(config['BACKUP_STATS_DIR_PATH'])
            restored_train_stats = load_json_file(last_epoch_stats_file_path)
        except (FileNotFoundError, OSError, StopIteration):
            restored_train_stats = None
    else:
        model = get_test_model(2)
        restored_train_stats = None

    agent = DQNPREMultiplayerAgent(model, 100000)
    agent.train(catch_game_object, epochs=100000, batch_size=50, gamma=0.9, epsilon=0.1, visualizer=None,
                restored_training_stats=restored_train_stats)


def _PRE_dqn_training_two_nets_two_players():
    catch_game_object = MultiPlayerCatch(2, board_size=20, food_spawn_rate=0.05)

    config = load_json_file('config_multiplayer_many_nets.json')
    if config['RESTORE_BACKUP']:
        try:
            m1_backup_path = osp.join(config['BACKUP_MODELS_PATH'], 'net0.h5')
            m2_backup_path = osp.join(config['BACKUP_MODELS_PATH'], 'net1.h5')
            models = [load_trained_model(path) for path in (m1_backup_path, m2_backup_path)]
        except (FileNotFoundError, OSError):
            models = [get_test_model(1) for _ in range(2)]
        try:
            last_epoch_stats_file_path = get_path_to_file_last_in_numerical_order(config['BACKUP_STATS_DIR_PATH'])
            restored_train_stats = load_json_file(last_epoch_stats_file_path)
        except (FileNotFoundError, OSError, StopIteration):
            restored_train_stats = None
    else:
        models = [get_test_model(1) for _ in range(2)]
        restored_train_stats = None

    agent = DQNPREMMultiplayerMultinetAgent(100000, models=models)
    agent.train(catch_game_object, epochs=100000, batch_size=50, gamma=0.9, epsilon=0.1, visualizer=None,
                restored_training_stats=restored_train_stats)


class Experiments(enum.IntEnum):
    PRE_DQN_CATCH_TWO_PLAYERS_ONE_NET = 1,
    PRE_DQN_CATCH_TWO_PLAYERS_MANY_NETS = 2,
    DQN_SINGLE_PLAYER = 3,
    DQN_PRE_SINGLE_PLAYER = 4


def run_experiment(experiment_id):
    run_experiment.available_experiments = {
        Experiments.PRE_DQN_CATCH_TWO_PLAYERS_ONE_NET:  _PRE_dqn_training_one_net_two_players,
        Experiments.PRE_DQN_CATCH_TWO_PLAYERS_MANY_NETS: _PRE_dqn_training_two_nets_two_players,
        Experiments.DQN_SINGLE_PLAYER: _single_dqn_test_demo,
        Experiments.DQN_PRE_SINGLE_PLAYER: _PRE_dqn_training
    }
    run_experiment.available_experiments[experiment_id]()
