# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Soft Actor-Critic (SAC) algorithm for ASSUME.

SAC uses maximum-entropy reinforcement learning: the agent maximizes expected
reward *plus* an entropy bonus, which prevents premature convergence to
deterministic policies.  This is particularly useful for electricity market
bidding where MATD3's decaying Gaussian noise tends to collapse to marginal-
cost bidding before discovering profitable markup strategies.

Key differences from TD3 (matd3.py):
  - Stochastic actor (StochasticActor) instead of deterministic (MLPActor)
  - No target actor network (entropy handles exploration)
  - Auto-tuned entropy temperature alpha
  - No policy delay (actor updates every gradient step)
  - No target policy noise smoothing (entropy serves same purpose)

References:
  - Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor" (2018)
  - Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2019)
"""

import json
import logging
import os

import torch as th
from torch.nn import functional as F
from torch.optim import AdamW

from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.learning_utils import polyak_update
from assume.reinforcement_learning.neural_network_architecture import (
    CriticTD3,
    StochasticActor,
)

logger = logging.getLogger(__name__)


class SAC(RLAlgorithm):
    """Soft Actor-Critic with auto-tuned entropy temperature.

    Reuses CriticTD3 (twin Q-networks) from the existing codebase.
    Requires ``actor_architecture: "stochastic"`` in learning config so that
    ``self.actor_architecture_class`` resolves to StochasticActor.
    """

    def __init__(self, learning_role):
        super().__init__(learning_role)

        self.n_updates = 0
        self.grad_clip_norm = 1.0

        # SAC-specific config (with defaults)
        self.alpha_auto = getattr(self.learning_config, "sac_alpha_auto", True)
        self.initial_alpha = getattr(self.learning_config, "sac_initial_alpha", 0.2)

    # ------------------------------------------------------------------
    # Policy initialization
    # ------------------------------------------------------------------

    def initialize_policy(self, actors_and_critics=None):
        if actors_and_critics is None:
            self.check_strategy_dimensions()
            self.create_actors()
            self.create_critics()
            self._init_alpha()
        else:
            for u_id, strategy in self.learning_role.rl_strats.items():
                strategy.actor = actors_and_critics["actors"][u_id]
                strategy.actor_target = actors_and_critics["actor_targets"][u_id]
                strategy.critics = actors_and_critics["critics"][u_id]
                strategy.target_critics = actors_and_critics["target_critics"][u_id]
            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]
            self.n_updates = actors_and_critics.get("n_updates", 0)
            # Restore frozen BC actors for regularization
            bc_actors = actors_and_critics.get("bc_actors", {})
            for u_id, strategy in self.learning_role.rl_strats.items():
                if u_id in bc_actors and bc_actors[u_id] is not None:
                    strategy.bc_actor = bc_actors[u_id]
            self._init_alpha()

    def _init_alpha(self):
        """Set up the learnable entropy temperature."""
        configured = getattr(self.learning_config, "sac_target_entropy", None)
        self.target_entropy = configured if configured is not None else -self.act_dim
        self.log_alpha = th.tensor(
            self.initial_alpha, dtype=th.float32, device=self.device
        ).log()
        if self.alpha_auto:
            self.log_alpha.requires_grad_(True)
            self.alpha_optimizer = AdamW(
                [self.log_alpha],
                lr=self.learning_role.calc_lr_from_progress(1),
            )

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def check_strategy_dimensions(self):
        """Reuse TD3's dimension checking (same centralized-critic requirement)."""
        # Identical logic — factored out of TD3 via copy for now
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
            strategy.actor = StochasticActor(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
                unique_obs_dim=self.unique_obs_dim,
                num_timeseries_obs_dim=self.num_timeseries_obs_dim,
                actor_hidden_1=h1,
                actor_hidden_2=h2,
            ).to(self.device)

            # SAC has no target actor, but we store a dummy reference for compat
            # with save/load and extract_policy (same weights as actor).
            strategy.actor_target = StochasticActor(
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

        if self.learning_config.distributional_critic:
            from assume.reinforcement_learning.neural_network_architecture import (
                DistributionalCriticTD3,
            )
            _critic_cls = DistributionalCriticTD3
            _critic_kwargs = dict(n_quantiles=self.learning_config.n_quantiles)
        else:
            _critic_cls = CriticTD3
            _critic_kwargs = dict(
                use_popart=self.learning_config.popart_enabled,
                popart_beta=self.learning_config.popart_beta,
            )

        _common = dict(
            n_agents=n_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            unique_obs_dim=self.unique_obs_dim,
            float_type=self.float_type,
        )

        for strategy in self.learning_role.rl_strats.values():
            strategy.critics = _critic_cls(**_common, **_critic_kwargs).to(self.device)
            strategy.target_critics = _critic_cls(**_common, **_critic_kwargs).to(self.device)

            strategy.target_critics.load_state_dict(strategy.critics.state_dict())
            strategy.target_critics.train(mode=False)

            strategy.critics.optimizer = AdamW(strategy.critics.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_params(self, directory):
        self.save_critic_params(directory=f"{directory}/critics")
        self.save_actor_params(directory=f"{directory}/actors")
        self._save_alpha(directory)

    def _save_alpha(self, directory):
        os.makedirs(directory, exist_ok=True)
        obj = {"log_alpha": self.log_alpha.detach().cpu()}
        if self.alpha_auto:
            obj["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        th.save(obj, os.path.join(directory, "alpha.pt"))

    def save_critic_params(self, directory):
        os.makedirs(directory, exist_ok=True)
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "critic": strategy.critics.state_dict(),
                "critic_target": strategy.target_critics.state_dict(),
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
        self._load_alpha(directory)

    def _load_alpha(self, directory):
        path = os.path.join(directory, "alpha.pt")
        if not os.path.exists(path):
            logger.warning("No alpha checkpoint found; using initial alpha.")
            return
        obj = th.load(path, map_location=self.device, weights_only=True)
        self.log_alpha.data.copy_(obj["log_alpha"])
        if self.alpha_auto and "alpha_optimizer" in obj:
            self.alpha_optimizer.load_state_dict(obj["alpha_optimizer"])

    def load_critic_params(self, directory: str) -> None:
        logger.info("Loading SAC critic parameters...")
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
                strategy.target_critics.load_state_dict(params["critic_target"])
                strategy.critics.optimizer.load_state_dict(params["critic_optimizer"])
            except Exception as e:
                logger.warning(f"Failed to load critic for {u_id}: {e}")

    def load_actor_params(self, directory: str) -> None:
        logger.info("Loading SAC actor parameters...")
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
        # Preserve frozen BC actors for regularization
        bc_actors = {}
        for u_id, strategy in self.learning_role.rl_strats.items():
            bc_actors[u_id] = getattr(strategy, "bc_actor", None)

        return {
            "actors": actors,
            "actor_targets": actor_targets,
            "critics": critics,
            "target_critics": target_critics,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
            "n_updates": self.n_updates,
            "bc_actors": bc_actors,
        }

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update_policy(self):
        logger.debug("SAC: Updating Policy")

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
        if self.alpha_auto:
            self.update_learning_rate(self.alpha_optimizer, learning_rate=learning_rate)

        unit_params = [
            {
                u_id: {
                    "actor_loss": None,
                    "actor_total_grad_norm": None,
                    "actor_max_grad_norm": None,
                    "critic_loss": None,
                    "critic_total_grad_norm": None,
                    "critic_max_grad_norm": None,
                }
                for u_id in self.learning_role.rl_strats.keys()
            }
            for _ in range(self.learning_config.gradient_steps)
        ]

        use_per = self.learning_config.per_enabled
        use_popart = self.learning_config.popart_enabled
        use_distributional = self.learning_config.distributional_critic

        for step in range(self.learning_config.gradient_steps):
            self.n_updates += 1

            transitions = self.learning_role.buffer.sample(
                self.learning_config.batch_size
            )
            states = transitions.observations
            actions = transitions.actions
            next_states = transitions.next_observations
            rewards = transitions.rewards

            # PER: extract IS weights and tree indices
            if use_per:
                is_weights = transitions.weights  # [batch_size]
                tree_indices = transitions.tree_indices

            # Precompute unique obs parts
            unique_obs = states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.learning_config.batch_size, n_rl_agents, -1)
            next_unique_obs = next_states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.learning_config.batch_size, n_rl_agents, -1)

            # ----------------------------------------------------------
            # CRITIC UPDATE
            # ----------------------------------------------------------
            with th.no_grad():
                # Sample next actions from current stochastic policy
                next_actions_list = []
                next_log_probs_list = []
                for j, strat in enumerate(strategies):
                    na, nlp = strat.actor.sample(next_states[:, j, :])
                    next_actions_list.append(na)
                    next_log_probs_list.append(nlp)

                next_actions = th.stack(next_actions_list, dim=1)
                next_actions_flat = next_actions.view(-1, n_rl_agents * self.act_dim)

            all_actions_flat = actions.view(self.learning_config.batch_size, -1)

            for strategy in strategies:
                strategy.critics.optimizer.zero_grad(set_to_none=True)

            total_critic_loss = 0.0
            # Collect per-sample TD errors across agents for PER
            if use_per:
                all_td_errors = []

            for i, strategy in enumerate(strategies):
                critic_target = strategy.target_critics

                # Build state representations
                other_unique = th.cat(
                    (unique_obs[:, :i], unique_obs[:, i + 1 :]), dim=1
                )
                other_next_unique = th.cat(
                    (next_unique_obs[:, :i], next_unique_obs[:, i + 1 :]), dim=1
                )

                all_states_i = th.cat(
                    (
                        states[:, i, :].reshape(self.learning_config.batch_size, -1),
                        other_unique.reshape(self.learning_config.batch_size, -1),
                    ),
                    dim=1,
                )
                all_next_states_i = th.cat(
                    (
                        next_states[:, i, :].reshape(self.learning_config.batch_size, -1),
                        other_next_unique.reshape(self.learning_config.batch_size, -1),
                    ),
                    dim=1,
                )

                with th.no_grad():
                    # Target Q: min(Q1, Q2) - alpha * log_pi
                    next_q1, next_q2 = critic_target(
                        all_next_states_i, next_actions_flat
                    )
                    if use_distributional:
                        # Element-wise min preserves quantile structure
                        next_q = th.min(next_q1, next_q2)  # (batch, N)
                        next_q = next_q - self.alpha * next_log_probs_list[i]
                        reward_scale = self.learning_config.reward_scale
                        target_q = (
                            rewards[:, i].unsqueeze(1) / reward_scale
                            + self.learning_config.gamma * next_q
                        )
                    else:
                        next_q = th.min(next_q1, next_q2)

                        # PopArt: denormalize target critic output to raw scale
                        if use_popart and critic_target.q1_popart is not None:
                            next_q = critic_target.q1_popart.denormalize(next_q)

                        next_q = next_q - self.alpha * next_log_probs_list[i]
                        raw_target = (
                            rewards[:, i].unsqueeze(1)
                            + self.learning_config.gamma * next_q
                        )

                        # PopArt: update stats and normalize target
                        if use_popart and strategy.critics.q1_popart is not None:
                            strategy.critics.q1_popart.update_stats(raw_target)
                            strategy.critics.q2_popart.update_stats(raw_target)
                            target_q = strategy.critics.q1_popart.normalize(raw_target)
                        else:
                            target_q = raw_target

                q1, q2 = strategy.critics(all_states_i, all_actions_flat)

                if use_distributional:
                    from assume.reinforcement_learning.learning_utils import quantile_huber_loss
                    taus = strategy.critics.taus
                    kappa = self.learning_config.quantile_huber_kappa
                    w = is_weights if use_per else None
                    critic_loss = (
                        quantile_huber_loss(q1, target_q, taus, kappa, w)
                        + quantile_huber_loss(q2, target_q, taus, kappa, w)
                    )
                    if use_per:
                        with th.no_grad():
                            q1m = q1.mean(dim=1)
                            target_m = target_q.mean(dim=1)
                            td_err = (target_m - q1m).abs()
                            all_td_errors.append(td_err)
                elif use_per:
                    # Per-sample MSE weighted by IS weights
                    per_sample_loss_q1 = (q1 - target_q).pow(2)
                    per_sample_loss_q2 = (q2 - target_q).pow(2)
                    w = is_weights.unsqueeze(1)
                    critic_loss = (w * per_sample_loss_q1).mean() + (w * per_sample_loss_q2).mean()
                    # TD error for priority update (use min Q)
                    with th.no_grad():
                        td_err = (target_q - th.min(q1, q2)).abs().squeeze(1)
                        all_td_errors.append(td_err)
                else:
                    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                unit_params[step][strategy.unit_id]["critic_loss"] = critic_loss.item()
                total_critic_loss += critic_loss

            total_critic_loss.backward()

            for strategy in strategies:
                params = list(strategy.critics.parameters())
                max_grad = max(p.grad.norm() for p in params)
                total_norm = th.nn.utils.clip_grad_norm_(
                    params, max_norm=self.grad_clip_norm
                )
                strategy.critics.optimizer.step()
                unit_params[step][strategy.unit_id]["critic_total_grad_norm"] = total_norm
                unit_params[step][strategy.unit_id]["critic_max_grad_norm"] = max_grad

            # PER: update priorities using max TD error across agents
            if use_per:
                stacked_td = th.stack(all_td_errors, dim=0)  # [n_agents, batch]
                max_td, _ = stacked_td.max(dim=0)
                self.learning_role.buffer.update_priorities(
                    tree_indices, max_td.cpu().numpy()
                )

            # ----------------------------------------------------------
            # ACTOR UPDATE (every step — no policy delay in SAC)
            # Skip during critic warmup to preserve BC-trained actor weights.
            # ----------------------------------------------------------
            critic_warmup = self.learning_config.critic_warmup_steps
            if self.n_updates <= critic_warmup:
                # Critic-only warmup: skip actor and alpha updates entirely
                if self.n_updates == critic_warmup:
                    logger.info(
                        f"SAC: Critic warmup complete ({critic_warmup} steps). "
                        f"Enabling actor updates."
                    )
                # Still need polyak update for target critics
                all_critic_params = []
                all_target_critic_params = []
                for strategy in strategies:
                    all_critic_params.extend(strategy.critics.parameters())
                    all_target_critic_params.extend(strategy.target_critics.parameters())
                polyak_update(
                    all_critic_params,
                    all_target_critic_params,
                    self.learning_config.tau,
                )
                continue

            for strategy in strategies:
                strategy.actor.optimizer.zero_grad(set_to_none=True)

            total_actor_loss = 0.0
            all_log_probs_for_alpha = []

            for i, strategy in enumerate(strategies):
                state_i = states[:, i, :]
                action_i, log_prob_i = strategy.actor.sample(state_i)

                other_unique = th.cat(
                    (unique_obs[:, :i], unique_obs[:, i + 1 :]), dim=1
                )
                all_states_i = th.cat(
                    (
                        state_i.reshape(self.learning_config.batch_size, -1),
                        other_unique.reshape(self.learning_config.batch_size, -1),
                    ),
                    dim=1,
                )

                # Replace agent i's action with freshly sampled action
                all_actions_clone = actions.clone().detach()
                all_actions_clone[:, i, :] = action_i
                all_actions_clone = all_actions_clone.view(
                    self.learning_config.batch_size, -1
                )

                # Actor loss: alpha * log_pi - min(Q1, Q2) + λ * BC_reg
                # Use normalized Q-values (no denormalize) so the actor gradient
                # is scale-invariant across regimes (off-peak ~1 vs peak ~$900).
                if use_distributional:
                    # quantile_means returns (batch, 1) — mean of N quantiles
                    q1m, q2m = strategy.critics.quantile_means(
                        all_states_i, all_actions_clone
                    )
                    min_q = th.min(q1m, q2m)
                else:
                    q1, q2 = strategy.critics(all_states_i, all_actions_clone)
                    min_q = th.min(q1, q2)
                actor_loss = (self.alpha * log_prob_i - min_q).mean()

                # BC regularization: penalize deviation from frozen BC policy
                bc_lambda = self.learning_config.bc_regularization_lambda
                if bc_lambda > 0 and hasattr(strategy, "bc_actor"):
                    # Decay lambda over training
                    decay = self.learning_config.bc_regularization_decay
                    if decay > 0:
                        bc_lambda = bc_lambda * max(0.0, 1.0 - decay * self.n_updates)
                    if bc_lambda > 0:
                        with th.no_grad():
                            bc_target = strategy.bc_actor(state_i)
                        bc_reg = F.mse_loss(action_i, bc_target)
                        actor_loss = actor_loss + bc_lambda * bc_reg

                unit_params[step][strategy.unit_id]["actor_loss"] = actor_loss.item()
                total_actor_loss += actor_loss
                all_log_probs_for_alpha.append(log_prob_i.detach())

            total_actor_loss.backward()

            for strategy in strategies:
                params = list(strategy.actor.parameters())
                max_grad = max(p.grad.norm() for p in params)
                total_norm = th.nn.utils.clip_grad_norm_(
                    params, max_norm=self.grad_clip_norm
                )
                strategy.actor.optimizer.step()
                unit_params[step][strategy.unit_id]["actor_total_grad_norm"] = total_norm
                unit_params[step][strategy.unit_id]["actor_max_grad_norm"] = max_grad

            # ----------------------------------------------------------
            # ALPHA UPDATE (auto-tune entropy temperature)
            # ----------------------------------------------------------
            if self.alpha_auto:
                self.alpha_optimizer.zero_grad()
                # Average log_probs across all agents
                mean_log_prob = th.cat(all_log_probs_for_alpha, dim=0).mean()
                alpha_loss = -(
                    self.log_alpha * (mean_log_prob + self.target_entropy)
                )
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # Entropy floor: prevent alpha from collapsing to zero.
                # Without this, the policy goes deterministic before the
                # critic learns the reward landscape near merit-order cliffs.
                alpha_min = getattr(self.learning_config, "sac_alpha_min", 0.0)
                if alpha_min > 0:
                    with th.no_grad():
                        log_alpha_min = th.tensor(alpha_min, device=self.device).log()
                        self.log_alpha.clamp_(min=log_alpha_min.item())

            # ----------------------------------------------------------
            # TARGET CRITIC UPDATE (Polyak)
            # ----------------------------------------------------------
            all_critic_params = []
            all_target_critic_params = []
            for strategy in strategies:
                all_critic_params.extend(strategy.critics.parameters())
                all_target_critic_params.extend(strategy.target_critics.parameters())

            polyak_update(
                all_critic_params,
                all_target_critic_params,
                self.learning_config.tau,
            )

            # Also sync actor_target (for extract_policy / save compat)
            all_actor_params = []
            all_target_actor_params = []
            for strategy in strategies:
                all_actor_params.extend(strategy.actor.parameters())
                all_target_actor_params.extend(strategy.actor_target.parameters())
            polyak_update(
                all_actor_params,
                all_target_actor_params,
                self.learning_config.tau,
            )

        self.learning_role.write_rl_grad_params_to_output(learning_rate, unit_params)
