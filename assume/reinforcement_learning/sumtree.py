"""SumTree data structure for O(log n) proportional sampling in PER.

Reference: Schaul et al. (2016) "Prioritized Experience Replay"
"""

import numpy as np


class SumTree:
    """Binary tree where each parent is the sum of its children.

    Supports O(log n) priority-proportional sampling, update, and total/min queries.
    Leaf nodes store priorities; internal nodes store sums.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree has capacity-1 internal nodes + capacity leaves = 2*capacity - 1
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._min_cache_valid = False
        self._min_cache = float("inf")
        self.entries = 0  # number of leaves written at least once

    def _leaf_to_tree(self, leaf_idx: int) -> int:
        return leaf_idx + self.capacity - 1

    def _tree_to_leaf(self, tree_idx: int) -> int:
        return tree_idx - self.capacity + 1

    def total(self) -> float:
        return float(self.tree[0])

    def min_priority(self) -> float:
        """Minimum priority among populated leaves."""
        if self._min_cache_valid:
            return self._min_cache
        if self.entries == 0:
            return 0.0
        leaves = self.tree[self.capacity - 1 : self.capacity - 1 + self.entries]
        self._min_cache = float(leaves.min())
        self._min_cache_valid = True
        return self._min_cache

    def add(self, priority: float, data_idx: int):
        """Set priority for leaf at position data_idx."""
        tree_idx = self._leaf_to_tree(data_idx)
        self.update(tree_idx, priority)
        if data_idx >= self.entries:
            self.entries = min(data_idx + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        """Update priority at tree_idx and propagate change up."""
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._min_cache_valid = False
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get(self, s: float):
        """Sample a leaf proportional to priority.

        Args:
            s: A value in [0, total()), used to select a leaf.

        Returns:
            (tree_idx, priority, data_idx)
        """
        idx = 0  # start at root
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                # Reached a leaf
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx, self.tree[idx], self._tree_to_leaf(idx)
