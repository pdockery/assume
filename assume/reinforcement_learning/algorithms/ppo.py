# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Proximal Policy Optimization (PPO) algorithm for ASSUME.

Adapted from upstream ``drl-ppo`` branch to match the current main-branch
constructor/storage conventions (config-driven, critics on strategy objects).

PPO is an on-policy algorithm: it collects a full rollout, performs multiple
epochs of minibatch updates on that data, then discards it.  This makes it
less sample-efficient than off-policy methods (TD3, SAC) but it explores
more naturally via its stochastic policy and entropy bonus.

Key differences from TD3/SAC:
  - On-policy: uses RolloutBuffer (cleared after each update), not ReplayBuffer
  - DistActor outputs a Normal distribution; log_probs stored in buffer
  - Clipped surrogate objective + entropy bonus
  - CriticPPO is a single value network (not twin Q)
  - No target networks or Polyak updates
  - GAE advantage estimation

References:
  - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import json
import logging
import os

import torch as th
from torch.nn import functional as F
from torch.optim import AdamW

from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.neural_network_architecture import (
    CriticPPO,
    DistActor,
)

logger = logging.getLogger(__name__)


class PPO(RLAlgorithm):
    """Proximal Policy Optimization with clipped surrogate objective.

    Requires ``actor_architecture: "dist"`` in learning config.
    """

    def __init__(self, learning_role):
        super().__init__(learning_role)

        self.n_updates = 0
        self.grad_clip_norm = getattr(self.learning_config, "ppo_max_grad_norm", 0.5)

        # PPO-specific hyperparameters (with defaults)
        self.clip_ratio = getattr(self.learning_config, "clip_ratio", 0.2)
        self.vf_coef = getattr(self.learning_config, "vf_coef", 0.5)
        self.entropy_coef = getattr(self.learning_config, "entropy_coef", 0.01)
        self.gae_lambda = getattr(self.learning_config, "gae_lambda", 0.95)

    # ------------------------------------------------------------------
    # Policy initialization
    # ------------------------------------------------------------------

    def initialize_policy(self, actors_and_critics=None):
        if actors_and_critics is None:
            self.check_strategy_dimensions()
            self.create_actors()
            self.create_critics()
        else:
            for u_id, strategy in self.learning_role.rl_strats.items():
                strategy.actor = actors_and_critics["actors"][u_id]
                strategy.actor_target = actors_and_critics["actor_targets"][u_id]
                strategy.critics = actors_and_critics["critics"][u_id]
                strategy.target_critics = actors_and_critics["target_critics"][u_id]
            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]

    def check_strategy_dimensions(self):
        foresight_list, obs_dim_list, act_dim_list = [], [], []
        unique_obs_dim_list, num_ts_list = [], []

        for strategy in self.learning_role.rl_strats.values():
            foresight_list.append(strategy.foresight)
            obs_dim_list.append(strategy.obs_dim)
            act_dim_list.append(strategy.act_dim)
            unique_obs_dim_list.append(strategy.unique_obs_dim)
            num_ts_list.append(strategy.num_timeseries_obs_dim)

        for name, lst in [
            ("foresight", foresight_list),
            ("act_dim", act_dim_list),
            ("unique_obs_dim", unique_obs_dim_list),
            ("num_timeseries_obs_dim", num_ts_list),
            ("obs_dim", obs_dim_list),
        ]:
            if len(set(lst)) > 1:
                raise ValueError(
                    f"All {name} values must be the same for all RL agents. Got: {lst}"
                )

        self.foresight = foresight_list[0]
        self.act_dim = act_dim_list[0]
        self.unique_obs_dim = unique_obs_dim_list[0]
        self.num_timeseries_obs_dim = num_ts_list[0]
        self.obs_dim = obs_dim_list[0]

    def create_actors(self):
        lr = self.learning_role.calc_lr_from_progress(1)
        h1 = getattr(self.learning_config, "actor_hidden_1", 256)
        h2 = getattr(self.learning_config, "actor_hidden_2", 128)
        for strategy in self.learning_role.rl_strats.values():
            strategy.actor = DistActor(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
                unique_obs_dim=self.unique_obs_dim,
                num_timeseries_obs_dim=self.num_timeseries_obs_dim,
                actor_hidden_1=h1,
                actor_hidden_2=h2,
            ).to(self.device)

            # PPO has no target actor, but store a reference for save/load compat
            strategy.actor_target = DistActor(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
                unique_obs_dim=self.unique_obs_dim,
                num_timeseries_obs_dim=self.num_timeseries_obs_dim,
                actor_hidden_1=h1,
                actor_hidden_2=h2,
            ).to(self.device)
            strategy.actor_target.load_state_dict(strategy.actor.state_dict())
            strategy.actor_target.train(mode=False)

            strategy.actor.optimizer = AdamW(strategy.actor.parameters(), lr=lr)
            strategy.actor.loaded = False

    def create_critics(self):
        n_agents = len(self.learning_role.rl_strats)
        lr = self.learning_role.calc_lr_from_progress(1)
        for strategy in self.learning_role.rl_strats.values():
            strategy.critics = CriticPPO(
                n_agents=n_agents,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                unique_obs_dim=self.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            # No target critic for PPO, but store for save/load compat
            strategy.target_critics = CriticPPO(
                n_agents=n_agents,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                unique_obs_dim=self.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)
            strategy.target_critics.load_state_dict(strategy.critics.state_dict())
            strategy.target_critics.train(mode=False)

            strategy.critics.optimizer = AdamW(strategy.critics.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_params(self, directory):
        self.save_critic_params(directory=f"{directory}/critics")
        self.save_actor_params(directory=f"{directory}/actors")

    def save_critic_params(self, directory):
        os.makedirs(directory, exist_ok=True)
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "critic": strategy.critics.state_dict(),
                "critic_optimizer": strategy.critics.optimizer.state_dict(),
            }
            th.save(obj, f"{directory}/critic_{u_id}.pt")
        u_id_list = [str(u) for u in self.learning_role.rl_strats.keys()]
        with open(os.path.join(directory, "u_id_order.json"), "w") as f:
            json.dump({"u_id_order": u_id_list}, f, indent=2)

    def save_actor_params(self, directory):
        os.makedirs(directory, exist_ok=True)
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "actor": strategy.actor.state_dict(),
                "actor_target": strategy.actor_target.state_dict(),
                "actor_optimizer": strategy.actor.optimizer.state_dict(),
            }
            th.save(obj, f"{directory}/actor_{u_id}.pt")

    def load_params(self, directory: str) -> None:
        self.load_critic_params(directory)
        self.load_actor_params(directory)

    def load_critic_params(self, directory: str) -> None:
        logger.info("Loading PPO critic parameters...")
        critic_dir = os.path.join(directory, "critics")
        if not os.path.exists(critic_dir):
            logger.warning("No critic directory found; using random init.")
            return
        for u_id, strategy in self.learning_role.rl_strats.items():
            path = os.path.join(critic_dir, f"critic_{u_id}.pt")
            if not os.path.exists(path):
                continue
            try:
                params = th.load(path, weights_only=True)
                strategy.critics.load_state_dict(params["critic"])
                strategy.critics.optimizer.load_state_dict(params["critic_optimizer"])
            except Exception as e:
                logger.warning(f"Failed to load critic for {u_id}: {e}")

    def load_actor_params(self, directory: str) -> None:
        logger.info("Loading PPO actor parameters...")
        actor_dir = os.path.join(directory, "actors")
        if not os.path.exists(actor_dir):
            logger.warning("No actor directory found; using random init.")
            return
        for u_id, strategy in self.learning_role.rl_strats.items():
            try:
                params = self.load_obj(f"{actor_dir}/actor_{u_id}.pt")
                strategy.actor.load_state_dict(params["actor"])
                strategy.actor_target.load_state_dict(params["actor_target"])
                strategy.actor.optimizer.load_state_dict(params["actor_optimizer"])
                strategy.actor.loaded = True
            except Exception:
                logger.warning(f"No actor values loaded for agent {u_id}")

    def extract_policy(self) -> dict:
        actors, actor_targets = {}, {}
        critics, target_critics = {}, {}
        for u_id, strategy in self.learning_role.rl_strats.items():
            actors[u_id] = strategy.actor
            actor_targets[u_id] = strategy.actor_target
            critics[u_id] = strategy.critics
            target_critics[u_id] = strategy.target_critics
        return {
            "actors": actors,
            "actor_targets": actor_targets,
            "critics": critics,
            "target_critics": target_critics,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
        }

    # ------------------------------------------------------------------
    # GAE advantage estimation
    # ------------------------------------------------------------------

    def _compute_gae(self, rewards, values):
        """Compute GAE advantages and returns.

        Args:
            rewards: tensor of shape (T, n_agents)
            values: tensor of shape (T, n_agents)

        Returns:
            advantages: tensor (T, n_agents), normalized
            returns: tensor (T, n_agents)
        """
        T = rewards.shape[0]
        advantages = th.zeros_like(rewards)
        last_advantage = 0.0

        for t in reversed(range(T)):
            next_value = values[t + 1] if t < T - 1 else 0.0
            delta = rewards[t] - values[t] + self.learning_config.gamma * next_value
            advantages[t] = last_advantage = (
                delta + self.learning_config.gamma * self.gae_lambda * last_advantage
            )

        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update_policy(self):
        logger.debug("PPO: Updating Policy")

        strategies = list(self.learning_role.rl_strats.values())
        n_rl_agents = len(strategies)

        # Update learning rate
        learning_rate = self.learning_role.calc_lr_from_progress(
            self.learning_role.get_progress_remaining()
        )
        for strategy in strategies:
            self.update_learning_rate(
                [strategy.critics.optimizer, strategy.actor.optimizer],
                learning_rate=learning_rate,
            )

        # Get all data from rollout buffer
        full_transitions = self.learning_role.buffer.get()
        all_states = full_transitions.observations  # (T, n_agents, obs_dim)
        all_actions = full_transitions.actions  # (T, n_agents, act_dim)
        all_rewards = full_transitions.rewards  # (T, n_agents)

        # Compute old log_probs from current actor (before any updates).
        # This is correct because PPO is on-policy: the actor hasn't changed
        # since these transitions were collected.
        T = all_states.shape[0]
        old_log_probs = th.zeros(T, n_rl_agents, device=self.device)
        with th.no_grad():
            for i, strategy in enumerate(strategies):
                lp, _ = strategy.actor.evaluate(all_states[:, i, :], all_actions[:, i, :])
                old_log_probs[:, i] = lp.squeeze(-1)

        all_actions_flat = all_actions.view(T, -1)

        # Precompute unique obs for centralized critic
        unique_obs = all_states[
            :, :, self.obs_dim - self.unique_obs_dim :
        ].reshape(T, n_rl_agents, -1)

        # Compute values for each agent using centralized critic
        all_values = th.zeros(T, n_rl_agents, device=self.device)

        for i, strategy in enumerate(strategies):
            other_unique = th.cat(
                (unique_obs[:, :i], unique_obs[:, i + 1 :]), dim=1
            )
            states_i = th.cat(
                (
                    all_states[:, i, :].reshape(T, -1),
                    other_unique.reshape(T, -1),
                ),
                dim=1,
            )
            with th.no_grad():
                all_values[:, i] = strategy.critics(states_i, all_actions_flat).squeeze(-1)

        # Compute GAE advantages per agent
        advantages, returns = self._compute_gae(all_rewards, all_values.detach())

        # Multiple epochs of minibatch updates
        batch_size = self.learning_config.batch_size

        for _ in range(self.learning_config.gradient_steps):
            self.n_updates += 1

            # Sample minibatch
            if T <= batch_size:
                batch_inds = th.arange(T)
            else:
                batch_inds = th.randint(0, T, (batch_size,))

            b_states = all_states[batch_inds]
            b_actions = all_actions[batch_inds]
            b_old_log_probs = old_log_probs[batch_inds]
            b_advantages = advantages[batch_inds]
            b_returns = returns[batch_inds]
            b_actions_flat = b_actions.view(len(batch_inds), -1)
            B = len(batch_inds)

            b_unique_obs = b_states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(B, n_rl_agents, -1)

            for i, strategy in enumerate(strategies):
                actor = strategy.actor
                critic = strategy.critics

                # Evaluate current policy on batch
                state_i = b_states[:, i, :]
                new_log_probs, entropy = actor.evaluate(state_i, b_actions[:, i, :])
                new_log_probs = new_log_probs.squeeze(-1)
                entropy = entropy.squeeze(-1)

                # Policy ratio
                # old_log_probs may be summed per-agent (scalar per timestep per agent)
                ratio = (new_log_probs - b_old_log_probs[:, i]).exp()

                # Clipped surrogate
                adv_i = b_advantages[:, i]
                surr1 = ratio * adv_i
                surr2 = th.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_i
                policy_loss = -th.min(surr1, surr2).mean()

                # Value loss
                other_unique = th.cat(
                    (b_unique_obs[:, :i], b_unique_obs[:, i + 1 :]), dim=1
                )
                states_i_full = th.cat(
                    (state_i.reshape(B, -1), other_unique.reshape(B, -1)),
                    dim=1,
                )
                values_i = critic(states_i_full, b_actions_flat).squeeze(-1)
                value_loss = F.mse_loss(values_i, b_returns[:, i])

                # Total loss
                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                # Update
                actor.optimizer.zero_grad()
                critic.optimizer.zero_grad()
                total_loss.backward()

                th.nn.utils.clip_grad_norm_(
                    actor.parameters(), self.grad_clip_norm
                )
                th.nn.utils.clip_grad_norm_(
                    critic.parameters(), self.grad_clip_norm
                )

                actor.optimizer.step()
                critic.optimizer.step()

        # Sync actor_target for extract_policy compat
        for strategy in strategies:
            strategy.actor_target.load_state_dict(strategy.actor.state_dict())
            strategy.target_critics.load_state_dict(strategy.critics.state_dict())

        # Clear the rollout buffer (on-policy — data used once)
        if hasattr(self.learning_role.buffer, "clear"):
            self.learning_role.buffer.clear()
