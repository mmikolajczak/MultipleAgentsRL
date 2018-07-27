from experiments.experiments import run_experiment, Experiments
from utils.json import load_json_file


def main():
    config_path = './config/config_multiplayer_multi_net_play.json'
    config = load_json_file(config_path)
    run_experiment(Experiments.PRE_DQN_CATCH_TWO_PLAYERS_MANY_NETS_PLAY, config)


if __name__ == '__main__':
    main()

