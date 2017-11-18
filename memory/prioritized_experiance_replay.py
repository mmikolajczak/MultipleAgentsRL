from .memory import Memory
from .sum_tree import SumTree
import random


class ProportionalPER(Memory):  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        Memory.__init__(self, capacity)
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def _add(self, error, sample):
        p = self._get_priority(error)
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

    def get_batch(self, batch_size):
        return self._sample(batch_size)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def remember(self, error, sample):
        self._add(error, sample)

    def __len__(self):
        return int(self.tree.total())
