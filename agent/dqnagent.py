from .agent import Agent
import numpy as np
from memory.experiance_replay import ExperienceReplay
from timeit import default_timer


class DQNAgent(Agent):

    def __init__(self, model, memory_size): # TODO memory/experience replay
        Agent.__init__(self, model)
        self._memory = ExperienceReplay(memory_size)

    def train(self, game, epochs=1000, batch_size=50, gamma=0.9, epsilon=[1, 0.1], epsilon_rate=0.5, observe=0,
              visualizer=None, reset_memory=False, save_model=False):
        win_count = 0

        if type(epsilon) in {tuple, list}:
            delta = (epsilon[0] - epsilon[1]) / (epochs * epsilon_rate) #???
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
                    q_values = self._model.predict(np.expand_dims(state, axis=0))
                    action = np.argmax(q_values)

                if visualizer:
                    visualizer.visualize_state(state)

                game.play([action])
                reward = game.get_reward()
                next_state = game.get_state()
                game_over = game.game_is_over
                transistion = [state, action, reward, next_state, game_over]
                self._memory.remember(*transistion)

                state = next_state

                if epoch >= observe and len(self._memory) >= batch_size:
                    batch = self._memory.get_batch(batch_size, self._model, gamma=0.9)
                    if batch:
                        # do other stuff
                        inputs, targets = batch
                        tmp = self._model.train_on_batch(inputs, targets)[0]
                        loss += float(tmp)

                if game.game_is_won:
                    win_count += 1

                # update exploration/exploitation ratio
                if epsilon > final_epsilon and epsilon >= observe:
                    epsilon -= delta
                #t2 = default_timer()
                #print('Time for ingame iteration:', t2-t1) ~0.35, half of this time is on get batch
                print('Training, epoch: {}, loss: {}, game score: {}, total wins: {}'.format(epoch,
                                                                                             loss,
                                                                                             game.get_score(),
                                                                                             win_count))

        self._model.save('final.h5')

    def play(self, game, epochs, epsilon=0, visualizer=None):
        win_count = 0
        # memory?

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
                    q_values = self._model.predict(state)[0]
                    action = np.argmax(q_values)

                game.play(action)
                state = game.get_state()

                game_over = game.game_is_over
                # visualization of current state
                if visualizer:
                    visualizer.visualize_state(state)

            if game.game_is_won():
                win_count += 1
