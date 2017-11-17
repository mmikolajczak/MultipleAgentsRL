from .memory import Memory
from .sum_tree import SumTree
import random


class ProportionalPER(Memory):   # stored as ( s, a, r, s_ ) in SumTree, proportional version of PER

    def __init__(self, capacity, alpha, epsilon):
        Memory.__init__(self, capacity)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = epsilon

    def _getPriority(self, error):
        return (error + self.epsilon) ** self.alpha

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def _sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def get_batch(self, sample_size):
        return [data for idx, data in self._sample(sample_size)]

    def __len__(self):
        return self.tree.total()

    def remember(self, state):
        self.add(None, state) #TODO

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)
