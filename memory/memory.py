from abc import abstractmethod, ABCMeta


class Memory(metaclass=ABCMeta):

    def __init__(self, capacity):
        self._capacity = capacity

    @property
    def capacity(self):
        return self._capacity

    @abstractmethod
    def remember(self, state, action, reward, next_state):
        pass

    @abstractmethod
    def get_batch(self, sample_size):
        pass
