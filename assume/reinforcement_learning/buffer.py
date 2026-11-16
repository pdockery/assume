# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings
from typing import NamedTuple

import numpy as np
import torch as th

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    rewards: th.Tensor


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    log_probs: th.Tensor


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        act_dim: int,
        n_rl_units: int,
        device: str,
        float_type,
    ):
        """
        A class that represents a replay buffer for storing observations, actions, and rewards.
        The replay buffer is implemented as a circular buffer, where the oldest experiences are discarded when the buffer is full.

        Args:
            buffer_size (int): The maximum size of the buffer.
            obs_dim (int): The dimension of the observation space.
            act_dim (int): The dimension of the action space.
            n_rl_units (int): The number of reinforcement learning units.
            device (str): The device to use for storing the data (e.g., 'cpu' or 'cuda').
            float_type (torch.dtype): The data type to use for the stored data.
            observations (numpy.ndarray): The stored observations.
            actions (numpy.ndarray): The stored actions.
            rewards (numpy.ndarray): The stored rewards.
        """

        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.pos = 0
        self.full = False

        self.device = device

        # future: use float16 for GPU
        self.np_float_type = np.float16 if float_type == th.float16 else np.float32
        self.th_float_type = float_type

        self.n_rl_units = n_rl_units

        self.observations = np.zeros(
            (self.buffer_size, self.n_rl_units, self.obs_dim), dtype=self.np_float_type
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_rl_units, self.act_dim), dtype=self.np_float_type
        )
        self.rewards = np.zeros(
            (self.buffer_size, self.n_rl_units), dtype=self.np_float_type
        )

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

            total_memory_usage = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes
            )

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system apparently does not have enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def size(self):
        # write docstring for this function
        """
        Return the current size of the buffer (i.e. number of transitions
        stored in the buffer).

        Returns:
            buffer_size(int): The current size of the buffer

        """
        return self.buffer_size if self.full else self.pos

    def to_torch(self, array: np.array, copy=True):
        """
        Converts a numpy array to a PyTorch tensor. Note: It copies the data by default.

        Args:
            array (numpy.ndarray): The numpy array to convert.
            copy (bool, optional): Whether to copy the data or not
                (may be useful to avoid changing things by reference). Defaults to True.

        Returns:
            torch.Tensor: The converted PyTorch tensor.
        """

        if copy:
            return th.tensor(array, dtype=self.th_float_type, device=self.device)

        return th.as_tensor(array, dtype=self.th_float_type, device=self.device)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        reward: np.ndarray,
    ):
        """
        Adds an observation, action, and reward of all agents to the replay buffer.

        Args:
            obs (numpy.ndarray): The observation to add.
            actions (numpy.ndarray): The actions to add.
            reward (numpy.ndarray): The reward to add.
        """
        # copying all to avoid modification
        len_obs = obs.shape[0]
        self.observations[self.pos : self.pos + len_obs] = obs.copy()
        self.actions[self.pos : self.pos + len_obs] = actions.copy()
        self.rewards[self.pos : self.pos + len_obs] = np.squeeze(
            reward.copy(), axis=-1
        )  # alsways one reward value per agent and time-step hence squezze

        self.pos += len_obs
        if self.pos + len_obs >= self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Samples a random batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            ReplayBufferSamples: A named tuple containing the sampled observations, actions, and rewards.

        Raises:
            Exception: If there are less than two entries in the buffer.
        """

        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound < 2:
            raise Exception("at least two entries needed to sample")
        batch_inds = np.random.randint(0, upper_bound - 1, size=batch_size)

        data = (
            self.observations[batch_inds, :, :],
            self.actions[batch_inds, :, :],
            self.observations[batch_inds + 1, :, :],
            self.rewards[batch_inds],
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class PrioritizedReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    rewards: th.Tensor
    weights: th.Tensor  # importance-sampling weights
    tree_indices: np.ndarray  # for priority updates


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with proportional prioritization via SumTree.

    Reference: Schaul et al. (2016) "Prioritized Experience Replay"
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        act_dim: int,
        n_rl_units: int,
        device: str,
        float_type,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_anneal_steps: int = 100_000,
        epsilon: float = 1e-6,
    ):
        super().__init__(buffer_size, obs_dim, act_dim, n_rl_units, device, float_type)
        from assume.reinforcement_learning.sumtree import SumTree

        self.tree = SumTree(buffer_size)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        self._max_priority = 1.0
        self._sample_step = 0

    @property
    def beta(self) -> float:
        """Current IS exponent, linearly annealed from beta_start to 1.0."""
        frac = min(1.0, self._sample_step / max(self.beta_anneal_steps, 1))
        return self.beta_start + (1.0 - self.beta_start) * frac

    def add(self, obs: np.ndarray, actions: np.ndarray, reward: np.ndarray):
        """Add transitions with max priority (ensures new samples get replayed)."""
        len_obs = obs.shape[0]
        for i in range(len_obs):
            data_idx = (self.pos + i) % self.buffer_size
            self.tree.add(self._max_priority**self.alpha, data_idx)
        super().add(obs, actions, reward)

    def sample(self, batch_size: int) -> PrioritizedReplayBufferSamples:
        """Sample proportional to priorities, returning IS weights."""
        self._sample_step += 1
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound < 2:
            raise Exception("at least two entries needed to sample")

        total = self.tree.total()
        segment = total / batch_size

        batch_inds = np.empty(batch_size, dtype=np.int64)
        tree_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            tree_idx, priority, data_idx = self.tree.get(s)
            # Avoid sampling the last position (need s' = obs[idx+1])
            if data_idx >= upper_bound - 1:
                data_idx = max(0, upper_bound - 2)
                tree_idx = self.tree._leaf_to_tree(data_idx)
                priority = self.tree.tree[tree_idx]
            tree_indices[i] = tree_idx
            batch_inds[i] = data_idx
            priorities[i] = priority

        # Importance-sampling weights
        probs = priorities / (total + 1e-10)
        n = self.size()
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()  # normalize so max weight = 1

        data = (
            self.observations[batch_inds, :, :],
            self.actions[batch_inds, :, :],
            self.observations[batch_inds + 1, :, :],
            self.rewards[batch_inds],
        )
        tensors = tuple(map(self.to_torch, data))
        weights_t = th.tensor(weights, dtype=self.th_float_type, device=self.device)

        return PrioritizedReplayBufferSamples(
            *tensors, weights=weights_t, tree_indices=tree_indices
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities from absolute TD errors."""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(int(tree_idx), float(priority))
            self._max_priority = max(self._max_priority, float(priority) ** (1.0 / self.alpha))


class RolloutBuffer:
    """On-policy buffer for PPO. Stores full rollouts and returns all data at once.

    Unlike ReplayBuffer, this buffer:
    - Appends transitions (no circular overwrite)
    - Returns ALL stored data via get() (no random sampling)
    - Is cleared after each policy update
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_rl_units: int,
        device: str,
        float_type,
        **kwargs,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_rl_units = n_rl_units
        self.device = device
        self.np_float_type = np.float16 if float_type == th.float16 else np.float32
        self.th_float_type = float_type

        self._observations = []
        self._actions = []
        self._rewards = []

    def size(self):
        return len(self._observations)

    def add(self, obs: np.ndarray, actions: np.ndarray, reward: np.ndarray):
        """Add a batch of transitions. Same signature as ReplayBuffer.add()."""
        self._observations.append(obs.copy())
        self._actions.append(actions.copy())
        self._rewards.append(np.squeeze(reward.copy(), axis=-1))

    def get(self) -> RolloutBufferSamples:
        """Return all stored transitions as tensors. log_probs is zeros (PPO computes them)."""
        obs = np.concatenate(self._observations, axis=0)
        actions = np.concatenate(self._actions, axis=0)
        rewards = np.concatenate(self._rewards, axis=0)

        obs_t = th.tensor(obs, dtype=self.th_float_type, device=self.device)
        actions_t = th.tensor(actions, dtype=self.th_float_type, device=self.device)
        rewards_t = th.tensor(rewards, dtype=self.th_float_type, device=self.device)
        # Placeholder — PPO will compute old log_probs from current actor before updating
        log_probs_t = th.zeros(rewards_t.shape, dtype=self.th_float_type, device=self.device)

        return RolloutBufferSamples(obs_t, actions_t, rewards_t, log_probs_t)

    def clear(self):
        """Discard all stored transitions (called after each PPO update)."""
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()

    def sample(self, batch_size: int):
        """Compatibility shim — PPO uses get(), not sample()."""
        return self.get()
