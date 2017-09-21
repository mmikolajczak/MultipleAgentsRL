from .memory import Memory
import numpy as np


class SimpleExperienceReplay(Memory):

    def __init__(self, capacity):
        Memory.__init__(self, capacity)
        self.memory = []

    def remember(self, state, action, reward, next_state):
        self.memory.append(np.array(state, action, reward, next_state))

    def get_sample(self, sample_size):
        indexes = np.random.permutation(self.capacity)[:sample_size]
        return self.memory[indexes]
