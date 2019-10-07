# utils for beam search
# Scripts adapted from show_and_tell_coco.
# @author: Ziyu Yao

import heapq


class BeamHyp:
    def __init__(self, logprob, str_idxes, state):
        self.logprob = logprob
        self.str_idxes = str_idxes
        self.state = state

    def update(self, step_logprob, step_idx):
        self.logprob += step_logprob
        self.str_idxes.append(step_idx)

    def __cmp__(self, other):
        assert isinstance(other, BeamHyp)
        if self.logprob == other.logprob:
            return 0
        elif self.logprob < other.logprob:
            return -1
        else:
            return 1


class TopN:
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().
        Args:
          sort: Whether to return the elements in descending sorted order.
        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []