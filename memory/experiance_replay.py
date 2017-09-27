from .memory import Memory
import numpy as np
import random
from timeit import default_timer


class ExperienceReplay(Memory):

    def __init__(self, capacity):
        Memory.__init__(self, capacity)
        self._memory = []

    def remember(self, state, action, reward, next_state, game_over):
        self.input_shape = state.shape[1:]
        self._memory.append([state, action, reward, next_state, game_over])
        if len(self._memory) > self._capacity:
            del self._memory[0]

    def get_batch(self, batch_size, model, gamma):
        # batch size requirements handled outside
        t1 = default_timer()
        samples = np.array(random.sample(self._memory, batch_size))
        S = []
        targets = []
        for state, action, reward, next_state, done in samples:
            target = reward
            if not done:
                target = (reward + gamma *
                          np.amax(model.predict(np.expand_dims(next_state, axis=0))[0]))
            target_f = model.predict(np.expand_dims(state, axis=0))
            target_f[0][action] = target
            S.append(state)
            targets.append(target_f)

        S = np.array(S)
        targets = np.array(targets).reshape(batch_size, -1)
        t2 = default_timer()
        print('Elapsed time on get batch:', t2 - t1) # ~0.2s
        return S, targets

    def __len__(self):
        return len(self._memory)
