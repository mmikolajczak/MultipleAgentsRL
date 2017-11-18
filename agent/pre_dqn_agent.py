from .agent import Agent
import numpy as np
from memory.prioritized_experiance_replay import ProportionalPER
from timeit import default_timer


class DQNPREAgent(Agent):

    def __init__(self, model, memory_size):
        Agent.__init__(self, model)
        self._memory = ProportionalPER(memory_size)

    def train(self, game, epochs=1000, batch_size=50, gamma=0.9, epsilon=[1, 0.1], epsilon_rate=0.5, observe=0,
              visualizer=None, recorder=None, reset_memory=False, save_model=False):
        win_count = 0

        if type(epsilon) in {tuple, list}:
            delta = (epsilon[0] - epsilon[1]) / (epochs * epsilon_rate)
            final_epsilon = epsilon[0]
            epsilon = epsilon[1]
        else:
            final_epsilon = epsilon

        for epoch in range(epochs):
            game.reset()
            loss = 0

            game_over = False
            state = game.get_state()

            while not game_over:
                #t1 = default_timer()
                if np.random.random() < epsilon:
                    action = np.random.randint(0, game.nb_actions)
                else:
                    import cv2
                    cv2.imshow('kek', state)
                    cv2.waitKey()
                    q_values = self._model.predict(np.expand_dims(state, axis=0))
                    action = np.argmax(q_values)

                if visualizer:
                    visualizer.visualize_state(state)

                if recorder:
                    additional_info = {'game_no': epoch + 1, 'game_name': game.name, 'game_score': game.get_score()}
                    recorder.additional_game_info = additional_info
                    recorder.record_state(state)

                game.play([action])
                reward = game.get_reward()
                next_state = game.get_state()
                game_over = game.game_is_over

                transistion = [state, action, reward, next_state]
                error = self._get_batch_preds_and_errors([(0, transistion)])
                self._memory.remember(error, transistion)

                state = next_state

                if epoch >= observe and len(self._memory) >= batch_size:

                    batch = self._memory.get_batch(batch_size)
                    if batch:
                        X, y, errors = self._get_batch_preds_and_errors(batch)

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

            if game.game_is_won:
                win_count += 1

            if epoch % 500 == 0:
                self._model.save('backup_model.h5')

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

    def _get_batch_preds_and_errors(self, batch):  # batch consist of tuples (idx, transistion)
        nn_input_shape = self._model.layers[0].input_shape  # wild guess, we shall see if its work
        no_state = np.zeros(nn_input_shape)
        gamma = 0.9

        states = np.array([row[1][0] for row in batch])
        next_states = np.array([(no_state if row[1][3] is None else row[1][3]) for row in batch])

        preds = self._model.predict(states)
        next_preds = self._model.predict(next_states)

        x = np.zeros(((len(batch), ) + nn_input_shape))
        y = np.zeros((len(batch), 5))  # 5 = env possible actions
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

