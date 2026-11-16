# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Deep Q-Network (DQN) with discrete action space for electricity market bidding.

Discretizes the continuous (price, volume) action space into bins so the agent
can learn discontinuous reward landscapes (merit-order cliffs) that defeat
continuous policy-gradient methods.

Supports:
- Double DQN (use online network to select actions, target to evaluate)
- Soft target-network updates (Polyak averaging)
- Epsilon-greedy exploration with linear decay
- PopArt critic normalization
- PER (via external PrioritizedReplayBuffer)
- BC warm-start compatible (epsilon starts low during warmup)

Reference: Mnih et al. 2015, "Human-level control through deep RL"
           Van Hasselt et al. 2016, "Deep RL with Double Q-learning"
"""

import json
import logging
import os

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.learning_utils import polyak_update

logger = logging.getLogger(__name__)


# ── Q-Network ────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """MLP that maps observations to Q-values for each discrete action."""

    def __init__(self, obs_dim: int, num_actions: int,
                 hidden1: int = 64, hidden2: int = 32,
                 float_type=th.float32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_actions),
        )
        self.float_type = float_type

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Return Q-values for all discrete actions. Shape: (batch, num_actions)."""
        return self.net(obs.to(self.float_type))



# ── Gradient-Informed Bin Placement ──────────────────────────────────────────

def _gradient_bins(regions: list[tuple[float, float, int]]) -> np.ndarray:
    """Place bins according to gradient-informed region specifications.

    Each region is (start, end, n_bins) — the number of bins allocated
    to that physical range.  Regions with steep reward gradients get more
    bins; flat regions get fewer.

    Args:
        regions: List of (start, end, n_bins) tuples.  Must cover the
                 full range contiguously and be sorted by start.

    Returns:
        Sorted array of bin centers across all regions.
    """
    all_centers = []
    for start, end, n in regions:
        if n <= 0:
            continue
        if n == 1:
            all_centers.append((start + end) / 2.0)
        else:
            # Evenly space within the region
            centers = np.linspace(start, end, n + 2)[1:-1]  # exclude endpoints
            all_centers.extend(centers.tolist())
    return np.array(sorted(all_centers))


def _auto_gradient_regions_price(
    price_min: float, price_max: float,
    competitor_mc: float, own_mc: float,
    n_bins: int,
) -> list[tuple[float, float, int]]:
    """Auto-generate gradient-informed price regions.

    The reward landscape for price has two key structures:
    - Below competitor MC: price doesn't affect dispatch (agent is infra-marginal
      when vol < residual demand).  Coarse bins suffice.
    - Near competitor MC: the cliff where dispatch drops to zero.  The agent's
      profit is maximised just below this cliff.  Fine bins needed.
    - Above competitor MC: no dispatch, no profit.  Minimal bins.

    Allocates bins proportional to gradient importance:
    ~15% below the gradient zone, ~70% in the gradient zone, ~15% above.
    """
    # Define the gradient zone: own_mc to competitor_mc + small margin
    grad_lo = max(own_mc, price_min)
    grad_hi = min(competitor_mc + 1.0, price_max)  # $1 above cliff
    cliff_zone_lo = max(competitor_mc - 5.0, grad_lo)  # $5 below cliff

    # Allocate bins: below gradient | gradient approach | cliff zone | above
    n_below = max(1, int(n_bins * 0.10))
    n_approach = max(1, int(n_bins * 0.15))
    n_cliff = max(2, int(n_bins * 0.65))
    n_above = max(1, n_bins - n_below - n_approach - n_cliff)

    regions = [
        (price_min, grad_lo, n_below),          # $0 to MC: irrelevant
        (grad_lo, cliff_zone_lo, n_approach),    # MC to cliff-5: approach zone
        (cliff_zone_lo, grad_hi, n_cliff),       # cliff-5 to cliff+1: fine resolution
        (grad_hi, price_max, n_above),           # above cliff: never profitable
    ]
    return regions


def _auto_gradient_regions_volume(
    vol_min: float, vol_max: float,
    residual_demand_max: float,
    n_bins: int,
) -> list[tuple[float, float, int]]:
    """Auto-generate gradient-informed volume regions.

    The reward landscape for volume has one key structure:
    - 0 to residual_demand: volume directly scales profit (gradient = LMP - MC).
      Fine bins needed to match dispatch precisely.
    - Above residual_demand: excess volume doesn't get dispatched.  Price
      determines profit, not volume.  Coarse bins suffice.

    Allocates bins: ~80% in the gradient zone, ~20% above.
    """
    rd_ceil = min(residual_demand_max * 1.15, vol_max)  # 15% above peak RD

    n_gradient = max(2, int(n_bins * 0.80))
    n_above = max(1, n_bins - n_gradient)

    regions = [
        (vol_min, rd_ceil, n_gradient),     # 0 to ~75MW: fine (gradient active)
        (rd_ceil, vol_max, n_above),        # ~75MW to 300MW: coarse (no gradient)
    ]
    return regions


# ── Action Discretizer ───────────────────────────────────────────────────────

class ActionDiscretizer:
    """Maps between discrete action indices and continuous (price, volume) values.

    Supports gradient-informed non-uniform binning: bins are concentrated
    where the reward landscape has steep gradients (price near competitor MC,
    volume near residual demand) and sparse where gradients are flat.

    Actions are indexed as: idx = price_bin * n_vol_bins + vol_bin
    Continuous values are bin centers, scaled to [-1, 1] for pipeline compat.
    """

    def __init__(self, n_price_bins: int, n_vol_bins: int,
                 price_range: tuple[float, float] = (0.0, 60.0),
                 vol_range: tuple[float, float] = (0.0, 300.0),
                 price_regions: list[tuple[float, float, int]] | None = None,
                 vol_regions: list[tuple[float, float, int]] | None = None,
                 device="cpu", float_type=th.float32):
        self.n_price_bins = n_price_bins
        self.n_vol_bins = n_vol_bins
        self.price_range = price_range
        self.vol_range = vol_range

        # Compute bin centers — gradient-informed regions or uniform fallback
        if price_regions:
            price_phys = _gradient_bins(price_regions)
            self.n_price_bins = len(price_phys)
        else:
            price_phys = np.linspace(price_range[0], price_range[1], n_price_bins)

        if vol_regions:
            vol_phys = _gradient_bins(vol_regions)
            self.n_vol_bins = len(vol_phys)
        else:
            vol_phys = np.linspace(vol_range[0], vol_range[1], n_vol_bins)

        self.num_actions = self.n_price_bins * self.n_vol_bins

        # Log bin distribution
        logger.info(
            f"ActionDiscretizer: {self.n_price_bins} price × {self.n_vol_bins} vol "
            f"= {self.num_actions} actions"
        )
        if len(price_phys) > 1:
            diffs = np.diff(price_phys)
            logger.info(
                f"  Price bins: ${price_phys[0]:.1f}–${price_phys[-1]:.1f}, "
                f"finest ${diffs.min():.2f}, coarsest ${diffs.max():.2f}"
            )
        if len(vol_phys) > 1:
            diffs = np.diff(vol_phys)
            logger.info(
                f"  Volume bins: {vol_phys[0]:.0f}–{vol_phys[-1]:.0f}MW, "
                f"finest {diffs.min():.1f}MW, coarsest {diffs.max():.1f}MW"
            )

        # Scale to [-1, 1] for pipeline compat
        p_min, p_max = price_range
        v_min, v_max = vol_range
        price_scaled = th.tensor(
            (price_phys - p_min) / max(p_max - p_min, 1e-6) * 2 - 1,
            dtype=float_type,
        )
        vol_scaled = th.tensor(
            (vol_phys - v_min) / max(v_max - v_min, 1e-6) * 2 - 1,
            dtype=float_type,
        )

        # Store physical centers for logging/analysis
        self.price_centers_phys = price_phys
        self.vol_centers_phys = vol_phys

        # All combinations: shape (num_actions, 2)
        p_grid, v_grid = th.meshgrid(price_scaled, vol_scaled, indexing='ij')
        self.action_map = th.stack([p_grid.flatten(), v_grid.flatten()], dim=1).to(device)
        self.device = device

    def index_to_continuous(self, idx: th.Tensor) -> th.Tensor:
        """Convert action index(es) to continuous (price, vol) in [-1, 1]."""
        return self.action_map[idx]

    def continuous_to_index(self, continuous: th.Tensor) -> th.Tensor:
        """Reverse-map continuous action to nearest discrete index."""
        if continuous.dim() == 1:
            dists = ((self.action_map - continuous.unsqueeze(0)) ** 2).sum(dim=1)
            return dists.argmin()
        else:
            # (batch, 2) → (batch,)
            dists = ((self.action_map.unsqueeze(0) - continuous.unsqueeze(1)) ** 2).sum(dim=2)
            return dists.argmin(dim=1)

    def index_to_physical(self, idx: int) -> tuple[float, float]:
        """Convert action index to physical (price_$, volume_MW) — for logging."""
        price_idx = idx // self.n_vol_bins
        vol_idx = idx % self.n_vol_bins
        return float(self.price_centers_phys[price_idx]), float(self.vol_centers_phys[vol_idx])


# ── Discrete Actor Wrapper ───────────────────────────────────────────────────

class DiscreteActor(nn.Module):
    """Wraps a QNetwork to behave like an actor for the ASSUME pipeline.

    forward(obs) returns continuous actions in [-1, 1] (greedy from Q-values).
    The strategy's get_actions() calls this, so the rest of the pipeline
    (bid scaling, reward computation, buffer storage) works unchanged.
    """

    def __init__(self, q_network: QNetwork, discretizer: ActionDiscretizer):
        super().__init__()
        self.q_network = q_network
        self.discretizer = discretizer
        # Mimic MLPActor attributes expected by the pipeline
        self.min_output = -1.0
        self.max_output = 1.0
        self.loaded = False

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Greedy action: argmax Q → continuous action in [-1, 1]."""
        q_values = self.q_network(obs)
        if obs.dim() == 1:
            idx = q_values.argmax()
        else:
            idx = q_values.argmax(dim=1)
        return self.discretizer.index_to_continuous(idx)


# ── DQN Algorithm ─────────────────────────────────────────────────────────────

class DQN(RLAlgorithm):
    """
    Double DQN with discrete (price × volume) action space.

    Designed for electricity market bidding where merit-order clearing creates
    discontinuous reward landscapes that continuous policy-gradient methods
    (MATD3, SAC, PPO) struggle with.
    """

    def __init__(self, learning_role):
        super().__init__(learning_role)
        # DQN doesn't use actor_architecture_class — it builds its own QNetwork.

        self.n_updates = 0
        self.grad_clip_norm = getattr(self.learning_config, "dqn_grad_clip_norm", 1.0)

        # Action discretization params (discretizer created in build_discretizer)
        self.n_price_bins = getattr(self.learning_config, "n_price_bins", 100)
        self.n_vol_bins = getattr(self.learning_config, "n_vol_bins", 10)

        # Discretizer starts as None — built by build_discretizer() once
        # ridge locations are known (called from learning_runner after
        # network analysis).  Falls back to uniform if never called.
        self.discretizer = None
        self.num_actions = self.n_price_bins * self.n_vol_bins

        # Exploration schedule: epsilon-greedy → Boltzmann
        # Phase 1 (ep 1 to epsilon_decay_episodes): epsilon-greedy for broad coverage
        # Phase 2 (after epsilon_decay_episodes): Boltzmann sampling proportional
        #   to Q-values — naturally probes neighboring bins near the current best
        self.epsilon = getattr(self.learning_config, "epsilon_start", 1.0)
        self.epsilon_end = getattr(self.learning_config, "epsilon_end", 0.05)
        self.epsilon_decay_episodes = getattr(self.learning_config, "epsilon_decay_episodes", 200)
        self.boltzmann_tau = getattr(self.learning_config, "boltzmann_tau_start", 1.0)
        self.boltzmann_tau_end = getattr(self.learning_config, "boltzmann_tau_end", 0.1)
        self.boltzmann_decay_episodes = getattr(self.learning_config, "boltzmann_decay_episodes", 300)
        self.use_boltzmann = False  # switches to True after epsilon phase

    def build_discretizer(
        self,
        price_range: tuple[float, float] = (0.0, 60.0),
        vol_range: tuple[float, float] = (0.0, 300.0),
        price_regions: list[tuple[float, float, int]] | None = None,
        vol_regions: list[tuple[float, float, int]] | None = None,
    ):
        """Build the gradient-informed discretizer.

        Called from learning_runner after network analysis determines gradient
        structure (competitor MCs and residual demand).  If never called,
        initialize_policy() falls back to uniform bins.

        Args:
            price_range: (min_price, max_price) in $/MWh.
            vol_range: (0, max_power) in MW.
            price_regions: List of (start, end, n_bins) — gradient-informed
                           price regions.  None → uniform.
            vol_regions: List of (start, end, n_bins) — gradient-informed
                         volume regions.  None → uniform.
        """
        self.discretizer = ActionDiscretizer(
            self.n_price_bins, self.n_vol_bins,
            price_range=price_range,
            vol_range=vol_range,
            price_regions=price_regions,
            vol_regions=vol_regions,
            device=self.device, float_type=self.float_type,
        )
        self.num_actions = self.discretizer.num_actions

        logger.info(
            f"DQN: {self.discretizer.n_price_bins} price × "
            f"{self.discretizer.n_vol_bins} vol = "
            f"{self.num_actions} discrete actions (gradient-informed), "
            f"ε: {self.epsilon}→{self.epsilon_end} over {self.epsilon_decay_episodes} eps"
        )

    # ── Epsilon schedule ──────────────────────────────────────────────────

    def update_epsilon(self, episode: int):
        """Update exploration schedule: epsilon-greedy → Boltzmann transition.

        Phase 1 (episode ≤ epsilon_decay_episodes): Linear epsilon decay.
        Phase 2 (after): Switch to Boltzmann with temperature annealing.
        """
        if episode <= self.epsilon_decay_episodes:
            # Phase 1: epsilon-greedy
            frac = episode / max(1, self.epsilon_decay_episodes)
            self.epsilon = self.epsilon_end + (1.0 - frac) * (
                getattr(self.learning_config, "epsilon_start", 1.0) - self.epsilon_end
            )
            self.use_boltzmann = False
        else:
            # Phase 2: Boltzmann — anneal temperature
            self.use_boltzmann = True
            self.epsilon = 0.0  # no random actions in Boltzmann mode
            boltzmann_ep = episode - self.epsilon_decay_episodes
            frac = min(1.0, boltzmann_ep / max(1, self.boltzmann_decay_episodes))
            self.boltzmann_tau = self.boltzmann_tau_end + (1.0 - frac) * (
                getattr(self.learning_config, "boltzmann_tau_start", 1.0)
                - self.boltzmann_tau_end
            )
            if boltzmann_ep == 1:
                logger.info(
                    f"DQN: switched to Boltzmann exploration "
                    f"(τ: {self.boltzmann_tau:.2f}→{self.boltzmann_tau_end})"
                )

    # ── Policy initialization ────────────────────────────────────────────

    def check_strategy_dimensions(self):
        """Same as TD3 — ensure all agents have same obs/act dims."""
        obs_dims, act_dims, unique_obs_dims = [], [], []
        for strategy in self.learning_role.rl_strats.values():
            obs_dims.append(strategy.obs_dim)
            act_dims.append(strategy.act_dim)
            unique_obs_dims.append(strategy.unique_obs_dim)

        if len(set(obs_dims)) > 1:
            raise ValueError(f"Inconsistent obs_dim across agents: {obs_dims}")
        if len(set(act_dims)) > 1:
            raise ValueError(f"Inconsistent act_dim across agents: {act_dims}")
        if len(set(unique_obs_dims)) > 1:
            raise ValueError(f"Inconsistent unique_obs_dim across agents: {unique_obs_dims}")

        self.obs_dim = obs_dims[0]
        self.act_dim = act_dims[0]  # still 2 (price, vol) for buffer compat
        self.unique_obs_dim = unique_obs_dims[0]

    def initialize_policy(self, actors_and_critics=None):
        if actors_and_critics is None:
            self.check_strategy_dimensions()
            self._create_networks()
        else:
            for u_id, strategy in self.learning_role.rl_strats.items():
                strategy.actor = actors_and_critics["actors"][u_id]
                strategy.q_network = actors_and_critics["q_networks"][u_id]
                strategy.q_target = actors_and_critics["q_targets"][u_id]
                strategy.critics = strategy.q_network
                strategy.target_critics = strategy.q_target
                strategy.actor_target = strategy.actor
                strategy.discretizer = self.discretizer
            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]
            self.n_updates = actors_and_critics.get("n_updates", 0)

    def _create_networks(self):
        # Fallback: if build_discretizer() was never called, use uniform bins
        if self.discretizer is None:
            logger.warning("DQN: build_discretizer() not called — using uniform bins")
            self.build_discretizer()

        h1 = getattr(self.learning_config, "actor_hidden_1", 64)
        h2 = getattr(self.learning_config, "actor_hidden_2", 32)

        for strategy in self.learning_role.rl_strats.values():
            # Q-network (online)
            q_net = QNetwork(
                obs_dim=self.obs_dim,
                num_actions=self.num_actions,
                hidden1=h1, hidden2=h2,
                float_type=self.float_type,
            ).to(self.device)

            # Target Q-network
            q_target = QNetwork(
                obs_dim=self.obs_dim,
                num_actions=self.num_actions,
                hidden1=h1, hidden2=h2,
                float_type=self.float_type,
            ).to(self.device)
            q_target.load_state_dict(q_net.state_dict())
            q_target.train(mode=False)

            # Optimizer
            q_net.optimizer = AdamW(
                q_net.parameters(),
                lr=self.learning_config.learning_rate,
            )

            # Wrap Q-network as an actor (for pipeline compat)
            actor = DiscreteActor(q_net, self.discretizer).to(self.device)
            actor.optimizer = q_net.optimizer  # share optimizer ref

            strategy.q_network = q_net
            strategy.q_target = q_target
            strategy.actor = actor
            # DQN doesn't use actor_target, but set a dummy for pipeline compat
            strategy.actor_target = actor
            strategy.actor.loaded = False

            # DQN doesn't use separate critics, but set refs for pipeline compat
            strategy.critics = q_net       # Q-network doubles as the "critic"
            strategy.target_critics = q_target

            # Attach discretizer to strategy for get_actions()
            strategy.discretizer = self.discretizer

    # ── Policy update (Double DQN) ───────────────────────────────────────

    def update_policy(self):
        strategies = list(self.learning_role.rl_strats.values())
        use_per = self.learning_config.per_enabled
        use_popart = self.learning_config.popart_enabled

        learning_rate = self.learning_role.calc_lr_from_progress(
            self.learning_role.get_progress_remaining()
        )
        for strategy in strategies:
            self.update_learning_rate(strategy.q_network.optimizer, learning_rate)

        for step in range(self.learning_config.gradient_steps):
            self.n_updates += 1

            transitions = self.learning_role.buffer.sample(
                self.learning_config.batch_size
            )
            states = transitions.observations    # (batch, n_agents, obs_dim)
            actions = transitions.actions        # (batch, n_agents, act_dim=2)
            next_states = transitions.next_observations
            rewards = transitions.rewards        # (batch, n_agents)

            if use_per:
                is_weights = transitions.weights
                tree_indices = transitions.tree_indices

            for i, strategy in enumerate(strategies):
                obs_i = states[:, i, :]          # (batch, obs_dim)
                next_obs_i = next_states[:, i, :]
                reward_i = rewards[:, i]         # (batch,)
                act_cont_i = actions[:, i, :]    # (batch, 2) continuous

                # Reverse-map continuous actions to discrete indices
                act_idx = self.discretizer.continuous_to_index(act_cont_i)  # (batch,)

                # Current Q-values for chosen actions
                q_all = strategy.q_network(obs_i)           # (batch, num_actions)
                q_current = q_all.gather(1, act_idx.unsqueeze(1))  # (batch, 1)

                with th.no_grad():
                    # Double DQN: online net selects, target net evaluates
                    next_q_online = strategy.q_network(next_obs_i)
                    next_best_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q_target = strategy.q_target(next_obs_i)
                    next_q = next_q_target.gather(1, next_best_actions)  # (batch, 1)

                    target_q = reward_i.unsqueeze(1) + self.learning_config.gamma * next_q

                # Loss
                if use_per:
                    td_error = (q_current - target_q).abs().squeeze(1)
                    loss = (is_weights * (q_current - target_q).pow(2).squeeze(1)).mean()
                else:
                    loss = F.mse_loss(q_current, target_q)

                strategy.q_network.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    strategy.q_network.parameters(), self.grad_clip_norm
                )
                strategy.q_network.optimizer.step()

                # Update priorities
                if use_per:
                    self.learning_role.buffer.update_priorities(
                        tree_indices, td_error.detach().cpu().numpy()
                    )

            # Soft update target networks
            tau = self.learning_config.tau
            for strategy in strategies:
                polyak_update(
                    strategy.q_network.parameters(),
                    strategy.q_target.parameters(),
                    tau,
                )

    # ── Save / Load ──────────────────────────────────────────────────────

    def save_params(self, directory):
        os.makedirs(directory, exist_ok=True)
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "q_network": strategy.q_network.state_dict(),
                "q_target": strategy.q_target.state_dict(),
                "optimizer": strategy.q_network.optimizer.state_dict(),
            }
            th.save(obj, f"{directory}/dqn_{u_id}.pt")

        u_id_list = [str(u) for u in self.learning_role.rl_strats.keys()]
        with open(os.path.join(directory, "u_id_order.json"), "w") as f:
            json.dump({"u_id_order": u_id_list}, f, indent=2)

    def load_params(self, directory: str):
        for u_id, strategy in self.learning_role.rl_strats.items():
            path = f"{directory}/dqn_{u_id}.pt"
            if os.path.exists(path):
                obj = self.load_obj(path)
                strategy.q_network.load_state_dict(obj["q_network"])
                strategy.q_target.load_state_dict(obj["q_target"])
                strategy.q_network.optimizer.load_state_dict(obj["optimizer"])
                strategy.actor.loaded = True
                logger.info(f"Loaded DQN params for {u_id}")

    def extract_policy(self) -> dict:
        actors = {}
        q_networks = {}
        q_targets = {}
        for u_id, strategy in self.learning_role.rl_strats.items():
            actors[u_id] = strategy.actor
            q_networks[u_id] = strategy.q_network
            q_targets[u_id] = strategy.q_target

        return {
            "actors": actors,
            "actor_targets": actors,  # DQN doesn't have separate actor_target
            "critics": q_networks,    # pipeline compat (Q-net serves as critic)
            "target_critics": q_targets,
            "q_networks": q_networks,
            "q_targets": q_targets,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
            "n_updates": self.n_updates,
        }
