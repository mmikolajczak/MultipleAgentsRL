from abc import abstractmethod, ABCMeta


class Memory(metaclass=ABCMeta):

    def __init__(self, capacity):
        self._capacity = capacity

    @property
    def capacity(self):
        return self._capacity

    @property
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def remember(self, *args):
        pass

    @abstractmethod
    def get_batch(self, sample_size):
        pass
