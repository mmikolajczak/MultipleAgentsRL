from .memory import Memory
import numpy as np
import random
from timeit import default_timer


class ExperienceReplay(Memory):

    def __init__(self, capacity):
        Memory.__init__(self, capacity)
        self._memory = []

    def remember(self, state, action, reward, next_state, game_over):
        self._input_shape = state.shape
        #self._memory.append([state, action, reward, next_state, game_over]) old one
        self._memory.append(np.concatenate([state.flatten(), np.array(action).flatten(), np.array(reward).flatten(),
                                            next_state.flatten(), 1 * np.array(game_over).flatten()]))
        if len(self._memory) > self._capacity:
            del self._memory[0]

    def get_batch(self, batch_size, model, gamma):
        # batch size requirements handled outside
        t1 = default_timer()
        '''
        samples = np.array(random.sample(self._memory, batch_size))

        states = []
        targets = []

        for state, action, reward, next_state, done in samples:
            target = reward
            if not done:
                target = (reward + gamma *
                          np.amax(model.predict(np.expand_dims(next_state, axis=0))[0]))
            target_f = model.predict(np.expand_dims(state, axis=0))
            target_f[0][action] = target

            states.append(state)
            targets.append(target_f)
        '''



        # code below is quite strange in second half
        nb_actions = model.output_shape[-1]
        samples = np.array(random.sample(self._memory, batch_size))
        total_input_dims = np.prod(self._input_shape)

        states = samples[:, 0: total_input_dims]
        actions = samples[:, total_input_dims]
        rewards = samples[:, total_input_dims + 1]
        next_states = samples[:, total_input_dims + 2: 2 * total_input_dims + 2]
        game_over = samples[:, 2 * total_input_dims + 2]

        rewards = rewards.repeat(nb_actions).reshape((batch_size, nb_actions))
        game_over = game_over.repeat(nb_actions).reshape((batch_size, nb_actions))
        states = states.reshape((batch_size,) + self._input_shape)
        next_states = next_states.reshape((batch_size,) + self._input_shape)

        X = np.concatenate([states, next_states], axis=0)
        Y = model.predict(X)

        Qsa = np.max(Y[batch_size:], axis=1).repeat(nb_actions).reshape((batch_size, nb_actions)) # ok, ma sens
        delta = np.zeros((batch_size, nb_actions))
        actions = np.cast['int'](actions)
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * Y[:batch_size] + delta * (rewards + gamma * (1 - game_over) * Qsa)

        t2 = default_timer()
        print('Elapsed time on get batch:', t2 - t1) # ~0.2s, now (np version) 0.03

        return states, targets

    def __len__(self):
        return len(self._memory)
