from .agent import Agent
import numpy as np
from memory.prioritized_experiance_replay import ProportionalPER
import os
import os.path as osp
from utils.json import save_json_file


class DQNPREMMultiplayerMultinetAgent(Agent):
    def __init__(self, memory_size, models=None, nb_models=None, model_generator=None):
        Agent.__init__(self, None)
        self._model_gen = model_generator
        if models:
            self._models = models
        else:
            self._models = [model_generator() for _ in range(nb_models)]
        self._memories = [ProportionalPER(memory_size) for _ in range(len(self._models))]

    # TODO: Deal somehow with model restoration
    def train(self, game, epochs=1000, batch_size=50, gamma=0.9, epsilon=[1, 0.1], epsilon_rate=0.5, observe=0,
              visualizer=None, recorder=None, reset_memory=False, save_model=False, restored_training_stats=None,
              backup_models_save_dir_path=None, backup_stats_save_dir_path=None):

        backup_models_save_dir_path = backup_models_save_dir_path if backup_models_save_dir_path else './backup_models'
        backup_stats_save_dir_path = backup_stats_save_dir_path if backup_stats_save_dir_path else './backup_stats'
        os.makedirs(backup_models_save_dir_path, exist_ok=True)
        os.makedirs(backup_stats_save_dir_path, exist_ok=True)

        win_count = restored_training_stats['win_count'] if restored_training_stats else 0
        start_epoch = restored_training_stats['epoch'] + 1 if restored_training_stats else 0

        if type(epsilon) in {tuple, list}:
            delta = (epsilon[0] - epsilon[1]) / (epochs * epsilon_rate)
            final_epsilon = epsilon[0]
            epsilon = epsilon[1]
        else:
            final_epsilon = epsilon

        for epoch in range(start_epoch, epochs):
            game.reset()
            loss = 0

            game_over = False
            state = game.get_state()
            last_frames = np.zeros((80, 80, 6), dtype=np.uint8)  # for using multiple frames as memory state

            while not game_over:
                if np.random.random() < epsilon:
                    actions = [np.random.randint(0, game.nb_actions) for _ in range(game.nb_players)]
                else:
                    net_input_ = np.expand_dims(last_frames, axis=0)
                    q_values = [model.predict(net_input_)[0] for model in
                                self._models]  # [0] -  getting rid of additional nb_samples dimension in predict
                    actions = np.array([np.argmax(single_net_q_values) for single_net_q_values in q_values])

                if visualizer:
                    visualizer.visualize_state(state)

                if recorder:
                    additional_info = {'game_no': epoch + 1, 'game_name': game.name, 'game_score': game.get_score()}
                    recorder.additional_game_info = additional_info
                    recorder.record_state(state)

                game.play(actions)
                reward = game.get_reward()
                next_state = game.get_state()
                game_over = game.game_is_over

                last_frames = np.roll(last_frames, axis=2, shift=-1)
                last_frames[:, :, -1] = state
                next_frames = np.roll(last_frames, axis=2, shift=-1)
                next_frames[:, :, -1] = next_state

                for model_idx in range(len(self._models)):
                    transistion = [last_frames, [actions[model_idx]], reward, next_frames]
                    error = self._get_batch_preds_and_errors([(0, transistion)], model_idx)[2][0]
                    self._memories[model_idx].remember(error, transistion)

                state = next_state

                for model_idx in range(len(self._models)):
                    if epoch >= observe and len(self._memories[model_idx]) >= batch_size:

                        batch = self._memories[model_idx].get_batch(batch_size)
                        if batch:
                            X, y, errors = self._get_batch_preds_and_errors(batch, model_idx)

                            # update errors
                            for i in range(len(batch)):
                                idx = batch[i][0]
                                self._memories[model_idx].update(idx, errors[i])

                            batch_train_loss = self._models[model_idx].train_on_batch(X, y)[0]
                            loss += float(batch_train_loss)

                # update exploration/exploitation ratio
                if epsilon > final_epsilon and epsilon >= observe:
                    epsilon -= delta
                print('Training, epoch: {}, loss (total, all models): {}, game score: {}, total wins: {}'.format(epoch,
                                                                                             loss,
                                                                                             game.get_score(),
                                                                                             win_count))

            if game.game_is_won:
                win_count += 1

            # saving results in case of crashes/power off/zombie apocalypse/etc.
            epoch_stats = {
                'epoch': epoch,
                'loss': loss,
                'game_score': game.get_score(),
                'win_count': win_count
            }
            current_epoch_file_path = osp.join(backup_stats_save_dir_path, f'epoch{epoch}.json')
            save_json_file(current_epoch_file_path, epoch_stats)

            if epoch % 500 == 0:
                for i, model in enumerate(self._models):
                    model_save_path = osp.join(backup_models_save_dir_path, f'net{i}.h5')
                    model.save(model_save_path)

        for i, model in enumerate(self._models):
            model.save(f'final_model_net_{i}.h5')

    def play(self, game, epochs, epsilon=0, visualizer=None, recorder=None):
        win_count = 0

        for epoch in range(epochs):
            # restoration/initialization of initial environment state
            game.reset()
            game_over = False
            state = game.get_state()
            last_frames = np.zeros((80, 80, 6), dtype=np.uint8)

            while not game_over:
                # do stuff
                if np.random.random() < epsilon:
                    actions = [np.random.randint(0, game.nb_actions) for _ in range(game.nb_players)]
                else:
                    net_input_ = np.expand_dims(last_frames, axis=0)
                    q_values = [model.predict(net_input_)[0] for model in
                                self._models]  # [0] -  getting rid of additional nb_samples dimension in predict
                    actions = np.array([np.argmax(single_net_q_values) for single_net_q_values in q_values])

                game.play(actions)

                next_state = game.get_state()
                last_frames = np.roll(last_frames, axis=2, shift=-1)
                last_frames[:, :, -1] = state
                next_frames = np.roll(last_frames, axis=2, shift=-1)
                next_frames[:, :, -1] = next_state

                state = next_state
                game_over = game.game_is_over
                # visualization of current state
                if visualizer:
                    visualizer.visualize_state(game.get_state(cvt_to_gray=False))

                if recorder:
                    additional_info = {'game_no': epoch + 1, 'game_name': game.name, 'game_score': game.get_score()}
                    recorder.additional_game_info = additional_info
                    recorder.record_state(state)

            if game.game_is_won:
                win_count += 1

    # TODO: Much TODO
    def _get_batch_preds_and_errors(self, batch, model_idx):  # batch consist of tuples (idx, transistion)
        nn_input_shape = (80, 80, 6)
        no_state = np.zeros(nn_input_shape)
        gamma = 0.9

        states = np.array([row[1][0] for row in batch])
        next_states = np.array([(no_state if row[1][3] is None else row[1][3]) for row in batch])

        preds = self._models[model_idx].predict(states)
        next_preds = self._models[model_idx].predict(next_states)

        x = np.zeros(((len(batch),) + nn_input_shape))
        y = np.zeros((len(batch), 5))  # 5 = env possible actions, TODO: fix it
        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            transistion = batch[i][1]
            state = transistion[0]
            action = transistion[1]
            reward = transistion[2]
            next_state = transistion[3]

            current_state_preds = preds[i]
            action_q = current_state_preds[action]
            if next_state is None:
                current_state_preds[action] = reward
            else:
                current_state_preds[action] = reward + gamma * np.max(next_preds[i])

            x[i] = state
            y[i] = current_state_preds
            errors[i] = abs(action_q - current_state_preds[action])

        return x, y, errors
        # TODO move gamma, etc where they should actually be, not as consts/magics in script

