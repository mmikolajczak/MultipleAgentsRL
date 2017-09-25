from .memory import Memory
import numpy as np


class SimpleExperienceReplay(Memory):

    def __init__(self, capacity):
        Memory.__init__(self, capacity)
        self._memory = []

    def remember(self, state, action, reward, next_state):
        self._memory.append([state, action, reward, next_state])
        if len(self._memory) > self._capacity:
            del self._memory[0]

    def get_batch(self, batch_size):
        raise NotImplemented()
        if len(self._memory) < batch_size:
            batch_size = len(self._memory)
        indexes = np.random.permutation(self.capacity)[:batch_size]
        return self._memory[indexes]
