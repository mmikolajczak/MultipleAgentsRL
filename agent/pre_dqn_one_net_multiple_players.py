from .agent import Agent
import numpy as np
from memory.prioritized_experiance_replay import ProportionalPER
import os
import os.path as osp
from timeit import default_timer
from utils.json import save_json_file


class DQNPREMultiplayerAgent(Agent):

    def __init__(self, model, memory_size):
        Agent.__init__(self, model)
        self._memory = ProportionalPER(memory_size)

    def train(self, game, epochs=1000, batch_size=50, gamma=0.9, epsilon=[1, 0.1], epsilon_rate=0.5, observe=0,
              visualizer=None, recorder=None, reset_memory=False, save_model=False, restored_training_stats=None,
              backup_model_save_path=None, backup_stats_save_path=None):

        backup_model_save_path = backup_model_save_path if backup_model_save_path else './backup_model.h5'
        backup_stats_save_path = backup_stats_save_path if backup_stats_save_path else './backup_stats'
        os.makedirs(backup_stats_save_path, exist_ok=True)

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
                # t1 = default_timer()
                if np.random.random() < epsilon:
                    actions = [np.random.randint(0, game.nb_actions) for _ in range(game.nb_players)]
                else:
                    q_values = self._model.predict(np.expand_dims(last_frames, axis=0))[0]  # getting rid of additional nb_samples dimension in predict
                    actions = [np.argmax(q_values[player_idx * game.nb_actions:
                    (player_idx + 1) * game.nb_actions]) for player_idx in range(game.nb_players)]
                # print(actions)

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

                transistion = [last_frames, actions, reward, next_frames]
                error = self._get_batch_preds_and_errors([(0, transistion)], game.nb_actions, game.nb_players)[2][0]
                self._memory.remember(error, transistion)

                state = next_state

                if epoch >= observe and len(self._memory) >= batch_size:

                    batch = self._memory.get_batch(batch_size)
                    if batch:
                        X, y, errors = self._get_batch_preds_and_errors(batch, game.nb_actions, game.nb_players)

                        # update errors
                        for i in range(len(batch)):
                            idx = batch[i][0]
                            self._memory.update(idx, errors[i])

                        batch_train_loss = self._model.train_on_batch(X, y)[0]
                        loss += float(batch_train_loss)


                # update exploration/exploitation ratio
                if epsilon > final_epsilon and epsilon >= observe:
                    epsilon -= delta
                #t2 = default_timer()
                print('Training, epoch: {}, loss: {}, game score: {}, total wins: {}'.format(epoch,
                                                                                             loss,
                                                                                             game.get_score(),
                                                                                             win_count))

                #print('Time elapsed:', default_timer() - t1)
            if game.game_is_won:
                win_count += 1

            # saving results in case of crashes/power off/zombie apocalypse/etc.
            epoch_stats = {
                    'epoch': epoch,
                    'loss': loss,
                    'game_score': game.get_score(),
                    'win_count': win_count
            }
            current_epoch_file_path = osp.join(backup_stats_save_path, f'epoch{epoch}.json')
            save_json_file(current_epoch_file_path, epoch_stats)

            if epoch % 500 == 0:
                self._model.save(backup_model_save_path)

        self._model.save('final_model.h5')

    def play(self, game, epochs, epsilon=0, visualizer=None, recorder=None):
        win_count = 0

        for epoch in range(epochs):
            # restoration/initialization of initial environment state
            game.reset()
            game_over = False
            state = game.get_state()

            while not game_over:
                # do stuff
                if np.random.random() < epsilon:
                    action = np.random.randint(0, game.nb_actions)
                else:
                    q_values = self._model.predict(np.expand_dims(state, axis=0))[0]
                    action = np.argmax(q_values)

                game.play([action])
                state = game.get_state()

                game_over = game.game_is_over
                # visualization of current state
                if visualizer:
                    visualizer.visualize_state(state)

                if recorder:
                    additional_info = {'game_no': epoch + 1, 'game_name': game.name, 'game_score': game.get_score()}
                    recorder.additional_game_info = additional_info
                    recorder.record_state(state)

            if game.game_is_won:
                win_count += 1

    def _get_batch_preds_and_errors(self, batch, env_possible_actions, env_players):  # batch consist of tuples (idx, transistion)
        nn_input_shape = (80, 80, 6)#self._model.layers[0].input_shape  # wild guess, we shall see if its work (doesn't, magic const)
        no_state = np.zeros(nn_input_shape)
        gamma = 0.9

        states = np.array([row[1][0] for row in batch])
        next_states = np.array([(no_state if row[1][3] is None else row[1][3]) for row in batch])

        preds = self._model.predict(states)
        next_preds = self._model.predict(next_states)

        x = np.zeros(((len(batch), ) + nn_input_shape))
        y = np.zeros((len(batch), env_possible_actions * env_players))
        errors = np.zeros(len(batch))

        # Note:
        # transistion = [last_frames, actions, reward, next_frames]
        # actions = [argmax for each env_nb_actions in net output], earlier it was one int - requires changes

        for i in range(len(batch)):
            transistion = batch[i][1]
            state = transistion[0]
            actions = transistion[1]
            reward = transistion[2]
            next_state = transistion[3]

            current_state_preds = preds[i]

            # transform actions from game representation to indexes in flattened preds vector
            actions = [i * env_possible_actions + action for i, action in enumerate(actions)]
            actions_q = current_state_preds[actions]

            if next_state is None:
                current_state_preds[actions] = reward
            else:
                current_state_preds[actions] = reward + gamma * np.max(next_preds[i])  # ???

            x[i] = state
            y[i] = current_state_preds
            errors[i] = np.sum(np.abs(actions_q - current_state_preds[actions]))

        return x, y, errors
        # TODO move gamma, etc where they should actually be, not as consts/magics in script

