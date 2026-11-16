# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pure-profit RL strategy with unified reward, cliff shaping, and soft guardrails.

Replaces the two-regime (off-peak / peak) reward design of
WithholdingStrategy with a single reward signal:

    reward = SCED_profit + wavelet_shaping + guardrails

where SCED_profit = (LMP - MC) * accepted_volume * duration, and
PopArt normalizes adaptively so no manual normalizer is needed.

Key differences from WithholdingStrategy:
    1. Action space: price maps to [0, price_cap] (not [MC, price_cap]).
       The agent CAN bid below MC, which lets it learn the loss zone
       naturally — but also opens a degenerate 0-volume equilibrium.
    2. Guardrails: small exponential penalties near 0-volume and 0-price
       prevent the degenerate solution (always hide at 0 volume to
       avoid losses).
    3. No regime switch: off-peak naturally gives ~$0 profit (the agent
       isn't dispatched).  No hand-crafted off-peak reward that bleeds
       volume bias into peak hours.
    4. Ricker wavelet shaping at merit-order edges (peak hours):
       - Price ceiling wavelet at competitor MC: peaks just below the
         cliff, goes negative just above — bidirectional gradient.
       - MC floor wavelet at own MC: peaks just above MC, goes negative
         just below — prevents bidding below cost.
       - Volume wavelet at residual demand: peaks just below RD, goes
         negative just above — prevents crossing the marginal threshold.
       - Off-peak: Gaussian MC mountain (safe harbor at MC + full vol).

Observation space (4 dimensions):
    [0] price forecast this hour  (scaled 0-1 within episode min/max)
    [1] load forecast this hour   (scaled 0-1 within episode min/max)
    [2] regime indicator           (1.0 = peak: forecast+$1 >= MC, 0.0 = off-peak)
    [3] residual demand fraction   (residual_demand / max_power, 0-1)
        Standard market information — any generator can estimate from public
        ISO demand forecasts and published competitor capacity/costs (Wolak 2003).

Actions (2 dimensions, actor output in [-1, 1]):
    action[0] -> bid price   (scaled to [0, price_cap])
    action[1] -> volume frac (scaled to [0, max_power])
"""

import logging
import numpy as np
import torch as th
from datetime import timedelta

logger = logging.getLogger(__name__)

from assume.common.fast_pandas import FastSeries
from assume.common.market_objects import MarketConfig, Orderbook
from assume.strategies.learning_strategies import EnergyLearningStrategy


class _PerDimActionNoise:
    """Action noise with independent sigma per dimension (price vs volume)."""

    def __init__(self, sigma_price: float, sigma_volume: float,
                 scale: float = 1.0, dt: float = 1.0):
        self.sigma_price = sigma_price
        self.sigma_volume = sigma_volume
        self.scale = scale
        self.dt = dt

    def noise(self, device=None, dtype=th.float):
        price_n = th.normal(mean=0.0, std=self.sigma_price, size=(1,),
                            dtype=dtype, device=device)
        vol_n = th.normal(mean=0.0, std=self.sigma_volume, size=(1,),
                          dtype=dtype, device=device)
        return self.dt * self.scale * th.cat([price_n, vol_n])

    def update_noise_decay(self, updated_decay: float):
        self.dt = updated_decay


class ProfitStrategy(EnergyLearningStrategy):
    """Pure-profit RL strategy with Ricker wavelet shaping and soft guardrails.

    Reward = SCED profit + wavelet shaping + guardrails.
    PopArt handles normalization.  No regime switch.
    """

    # ── Guardrail defaults (overridable via kwargs / YAML) ──────────────
    # Volume guardrail: exponential penalty near vol_frac=0
    #   penalty = -vol_penalty * exp(-vol_frac / vol_steepness)
    #   At vol_frac=0: penalty = -vol_penalty (e.g. -$50)
    #   At vol_frac=0.1: penalty ≈ -$1.8  (negligible)
    #   At vol_frac=0.3: penalty ≈ -$0.002 (effectively zero)
    DEFAULT_VOL_PENALTY = 50.0      # $/h — cost of offering zero volume
    DEFAULT_VOL_STEEPNESS = 0.03    # controls how fast penalty decays

    # Price guardrail: exponential penalty near price_frac=0
    #   penalty = -price_penalty * exp(-price_frac / price_steepness)
    #   At price_frac=0 (bid=$0): penalty = -price_penalty (e.g. -$30)
    #   At price_frac=0.35 (bid=MC for gen_0): penalty ≈ -$0.003
    DEFAULT_PRICE_PENALTY = 30.0    # $/h — cost of bidding at $0
    DEFAULT_PRICE_STEEPNESS = 0.05  # controls how fast penalty decays

    # ── Terrain shaping defaults ────────────────────────────────────────
    # Ricker wavelet shaping at merit-order edges.  Each wavelet peaks
    # just before a strategic boundary and goes NEGATIVE just after,
    # providing bidirectional gradient (pull toward edge from both sides).
    #
    # Formula:  delta = (value - threshold) / width
    #           shaping = scale * delta * exp(-0.5 * delta^2)
    #
    # PEAK hours (residual > 0):
    #   1. Price ceiling wavelet at competitor MC — approach from below
    #   2. MC floor wavelet at own MC — stay above cost
    #   3. Volume wavelet at residual demand — approach from below
    #
    # OFF-PEAK hours (residual == 0):
    #   MC mountain (Gaussian safe harbor at MC + full volume) — unchanged
    #
    # All wavelets scaled by (residual / max_power) so magnitude is
    # proportional to achievable profit.
    DEFAULT_WAVELET_SCALE = 100.0      # $ — base magnitude for price wavelets
    DEFAULT_VOL_WAVELET_SCALE = 300.0  # $ — base magnitude for volume wavelet
                                       #   (separate so volume can be tuned independently)
    DEFAULT_PRICE_WAVELET_WIDTH = 2.0  # $/MWh — width of price wavelets
                                       #   peak at threshold ± width
    DEFAULT_VOL_WAVELET_WIDTH = 5.0    # MW — width of volume wavelet
                                       #   peak at RD ± 5MW
    DEFAULT_MC_MOUNTAIN_SCALE = 300.0  # $ — MC mountain (off-peak only)
    DEFAULT_MC_MOUNTAIN_SIGMA = 3.0    # $/MWh — Gaussian width for MC mountain
    DEFAULT_MC_MOUNTAIN_TAU = 0.5      # $/MWh — sigmoid width for MC mountain
    DEFAULT_SHAPING_ANNEAL = 0.0       # exp decay rate per episode
                                       #   0.0 = permanent (wavelets don't anneal)
                                       #   >0  = legacy annealing if needed

    def __init__(self, *args, **kwargs):
        kwargs['num_timeseries_obs_dim'] = 0
        kwargs['foresight'] = 1
        kwargs['unique_obs_dim'] = 4  # price_fcast, load_fcast, regime, residual_demand

        # Strategy-specific kwargs — pop before super().__init__
        self._exploration_center = kwargs.pop('exploration_center', None)
        self._noise_sigma_volume = kwargs.pop('noise_sigma_volume', None)

        # Guardrail parameters (configurable via YAML learning_config)
        self.vol_penalty = kwargs.pop('vol_penalty', self.DEFAULT_VOL_PENALTY)
        self.vol_steepness = kwargs.pop('vol_steepness', self.DEFAULT_VOL_STEEPNESS)
        self.price_penalty = kwargs.pop('price_penalty', self.DEFAULT_PRICE_PENALTY)
        self.price_steepness = kwargs.pop('price_steepness', self.DEFAULT_PRICE_STEEPNESS)

        # Terrain shaping parameters (Ricker wavelets)
        self.wavelet_scale = kwargs.pop('wavelet_scale', self.DEFAULT_WAVELET_SCALE)
        self.vol_wavelet_scale = kwargs.pop('vol_wavelet_scale', self.DEFAULT_VOL_WAVELET_SCALE)
        self.price_wavelet_width = kwargs.pop('price_wavelet_width', self.DEFAULT_PRICE_WAVELET_WIDTH)
        self.vol_wavelet_width = kwargs.pop('vol_wavelet_width', self.DEFAULT_VOL_WAVELET_WIDTH)
        self.mc_mountain_scale = kwargs.pop('mc_mountain_scale', self.DEFAULT_MC_MOUNTAIN_SCALE)
        self.mc_mountain_sigma = kwargs.pop('mc_mountain_sigma', self.DEFAULT_MC_MOUNTAIN_SIGMA)
        self.mc_mountain_tau = kwargs.pop('mc_mountain_tau', self.DEFAULT_MC_MOUNTAIN_TAU)
        self.shaping_anneal = kwargs.pop('shaping_anneal', self.DEFAULT_SHAPING_ANNEAL)
        # Pop old params that may still be in legacy YAML configs
        for old_key in ('price_cliff_bonus', 'price_cliff_tau',
                        'vol_cliff_bonus', 'vol_cliff_tau',
                        'mountain_scale', 'mountain_sigma', 'mountain_tau',
                        'vol_ridge_scale', 'vol_ridge_tau'):
            kwargs.pop(old_key, None)

        # Pop old WithholdingStrategy kwargs that may appear in shared YAML
        for stale_key in ('tau', 'sigmoid_offset_max', 'sigmoid_offset_min',
                          'sigmoid_offset_anneal_rate', 'popart_enabled',
                          'obs_horizon_decay'):
            kwargs.pop(stale_key, None)

        super().__init__(*args, **kwargs)

        # Per-dimension noise
        if self._noise_sigma_volume is not None and self.learning_mode:
            self.action_noise = _PerDimActionNoise(
                sigma_price=self.learning_config.noise_sigma,
                sigma_volume=self._noise_sigma_volume,
                scale=self.learning_config.noise_scale,
                dt=self.learning_config.noise_dt,
            )

        self._last_bid_volume: dict = {}
        self._last_bid_price: dict = {}

    # ── Observations (identical to WithholdingStrategy) ─────────────────

    def prepare_observations(self, unit, market_id):
        super().prepare_observations(unit, market_id)

        price_series = unit.forecaster.price[market_id]
        self._raw_price_fcast = np.array(
            [price_series[t] for t in unit.index], dtype=float
        )

        res_load_series = unit.forecaster.residual_load.get(
            market_id, FastSeries(index=unit.index, value=0)
        )
        self._raw_res_load = np.array(
            [res_load_series[t] for t in unit.index], dtype=float
        )

        self._price_min = float(self._raw_price_fcast.min())
        self._price_max = float(self._raw_price_fcast.max())
        self._load_min = float(self._raw_res_load.min())
        self._load_max = float(self._raw_res_load.max())

        # Per-hour residual demand for volume cliff shaping:
        # residual = total_demand - cheaper_supply (capacity priced below our MC)
        cheaper_supply = getattr(unit, 'cheaper_supply', 0.0)
        self._residual_demand = np.maximum(
            self._raw_res_load - cheaper_supply, 0.0
        )

        # Per-hour regime indicator: 1.0 = peak (strategic bidding matters),
        # 0.0 = off-peak (agent won't be dispatched, no profit gradient).
        # Threshold: forecast_price + $1 < MC → off-peak.
        mc = float(unit.calculate_marginal_cost(unit.index[0], 0))
        self._regime_indicator = (
            self._raw_price_fcast + 1.0 >= mc
        ).astype(float)

    def create_observation(self, unit, market_id, start, end):
        if not hasattr(self, '_raw_price_fcast'):
            self.prepare_observations(unit, market_id)

        h = int((start - unit.index[0]).total_seconds() / 3600)

        price_range = max(self._price_max - self._price_min, 1e-6)
        load_range = max(self._load_max - self._load_min, 1e-6)

        scaled_price = (self._raw_price_fcast[h] - self._price_min) / price_range
        scaled_load = (self._raw_res_load[h] - self._load_min) / load_range
        regime = self._regime_indicator[h]  # 1.0 = peak, 0.0 = off-peak
        # Residual demand as fraction of max_power — directly tells the actor
        # what volume to target.  This is standard market information: any
        # generator can estimate total_demand - cheaper_supply from public data
        # (ISO demand forecasts, published competitor capacity and fuel costs).
        max_power = float(unit.max_power) if hasattr(unit, 'max_power') else 300.0
        scaled_residual = self._residual_demand[h] / max(max_power, 1e-6)

        observation = th.tensor(
            [scaled_price, scaled_load, regime, scaled_residual],
            dtype=self.float_type, device=self.device,
        )

        if self.learning_mode:
            self.learning_role.add_observation_to_cache(
                self.unit_id, start, observation
            )

        return observation

    # ── Bid formulation ─────────────────────────────────────────────────

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        start = product_tuples[0][0]
        end = product_tuples[0][1]

        min_power, max_power = unit.calculate_min_max_power(start, end)
        min_power = min_power[0]
        max_power = max_power[0]

        next_observation = self.create_observation(
            unit=unit,
            market_id=market_config.market_id,
            start=start,
            end=end,
        )

        actions, noise = self.get_actions(next_observation)

        # Price: map [-1, 1] → [0, price_cap]
        # Floor is 0, NOT MC — agent must learn to avoid below-MC bids
        price_ceiling = getattr(unit, "price_cap", self.max_bid_price)
        a0 = float(actions[0] + 1) / 2.0
        a0 = max(0.0, min(1.0, a0))
        bid_price = a0 * price_ceiling

        # Volume: map [-1, 1] → [0, max_power]
        vol_fraction = float(actions[1] + 1) / 2.0
        vol_fraction = max(0.0, min(1.0, vol_fraction))
        bid_volume = vol_fraction * max_power

        self._last_bid_volume[start] = bid_volume
        self._last_bid_price[start] = bid_price

        bids = [
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": bid_price,
                "volume": bid_volume,
                "node": unit.node,
            },
        ]

        if self.learning_mode:
            self.learning_role.add_actions_to_cache(
                self.unit_id, start, actions, noise
            )

        return bids

    # ── Reward ──────────────────────────────────────────────────────────

    def calculate_reward(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """Pure profit reward with Ricker wavelet shaping and soft guardrails.

        reward = profit + wavelet_shaping + guardrails

        Profit:   (LMP - MC) * accepted_volume * duration
        Wavelets: Ricker wavelets at competitor MC, own MC, and residual demand
        Guards:   small penalties near 0-volume and 0-price
        """
        product_type = marketconfig.product_type
        start = orderbook[0]["start_time"]
        end = orderbook[0]["end_time"]
        end_excl = end - unit.index.freq

        marginal_cost = unit.calculate_marginal_cost(
            start, unit.outputs[product_type].at[start]
        )
        market_clearing_price = orderbook[0]["accepted_price"]
        duration = (end - start) / timedelta(hours=1)

        # Actual SCED dispatch
        income = 0.0
        operational_cost = 0.0
        accepted_volume_total = 0

        for order in orderbook:
            accepted_volume = order.get("accepted_volume", 0)
            accepted_volume_total += accepted_volume
            income += market_clearing_price * accepted_volume * duration
            operational_cost += marginal_cost * accepted_volume * duration

        # Start-up cost handling
        if (
            unit.outputs[product_type].at[start] != 0
            and unit.outputs[product_type].at[start - unit.index.freq] == 0
        ):
            operational_cost += unit.hot_start_cost / 2
        elif (
            unit.outputs[product_type].at[start] == 0
            and unit.outputs[product_type].at[start - unit.index.freq] != 0
        ):
            operational_cost += unit.hot_start_cost / 2

        profit = income - operational_cost

        # ── Bid info ────────────────────────────────────────────────
        bid_volume = self._last_bid_volume.get(start, unit.max_power)
        bid_price = self._last_bid_price.get(start, marginal_cost)
        price_ceiling = getattr(unit, "price_cap", self.max_bid_price)

        vol_frac = bid_volume / max(unit.max_power, 1.0)
        price_frac = bid_price / max(price_ceiling, 1.0)

        # ── Terrain shaping (Ricker wavelets at merit-order edges) ────
        # Each wavelet peaks just before a strategic boundary and goes
        # NEGATIVE just after, providing bidirectional gradient.
        # Formula: delta = (value - threshold) / width
        #          shaping = scale * delta * exp(-0.5 * delta^2)

        h = int((start - unit.index[0]).total_seconds() / 3600)
        residual = self._residual_demand[h] if hasattr(self, '_residual_demand') else 0.0

        price_ceil_wavelet = 0.0
        mc_floor_wavelet = 0.0
        vol_wavelet = 0.0
        mc_mountain = 0.0

        # Optional anneal (default 0.0 = permanent wavelets)
        if self.shaping_anneal > 0 and self.learning_mode:
            episode = getattr(self.learning_role, 'episode_count', 0)
            anneal = np.exp(-self.shaping_anneal * episode)
        else:
            anneal = 1.0

        if residual > 0:
            # ── PEAK: Ricker wavelets at all three edges ──
            resid_frac = min(residual / max(unit.max_power, 1.0), 1.0)
            w_scale = self.wavelet_scale * resid_frac * anneal
            pw = max(self.price_wavelet_width, 0.01)

            counterfactual = self._get_counterfactual_price(unit, start)

            # 1. Price ceiling wavelet — approach competitor MC from below
            #    Positive when bid < counterfactual, negative when bid > counterfactual
            #    No vol_frac scaling — price edges matter at all volume levels
            delta_ceil = (counterfactual - bid_price) / pw
            price_ceil_wavelet = w_scale * delta_ceil * np.exp(-0.5 * delta_ceil ** 2)

            # 2. MC floor wavelet — stay above own MC
            #    Positive when bid > MC, negative when bid < MC
            delta_mc = (bid_price - marginal_cost) / pw
            mc_floor_wavelet = w_scale * delta_mc * np.exp(-0.5 * delta_mc ** 2)

            # 3. Volume wavelet — approach residual demand from below
            #    Positive when vol < RD, negative when vol > RD
            vw = max(self.vol_wavelet_width, 0.01)
            delta_vol = (residual - bid_volume) / vw
            vw_scale = self.vol_wavelet_scale * resid_frac * anneal
            vol_wavelet = vw_scale * delta_vol * np.exp(-0.5 * delta_vol ** 2)

        else:
            # ── OFF-PEAK: MC mountain only (Gaussian safe harbor) ──
            sigma = max(self.mc_mountain_sigma, 0.01)
            tau = max(self.mc_mountain_tau, 0.01)
            mc_scale = self.mc_mountain_scale * anneal
            mc_floor = 1.0 / (1.0 + np.exp(-(bid_price - marginal_cost) / tau))
            mc_decay = np.exp(-max(bid_price - marginal_cost, 0.0) ** 2
                              / (2.0 * sigma ** 2))
            mc_mountain = 2.0 * mc_scale * mc_floor * mc_decay * vol_frac

        shaping = price_ceil_wavelet + mc_floor_wavelet + vol_wavelet + mc_mountain

        # ── Guardrails ──────────────────────────────────────────────
        # Volume guardrail: exponential penalty near vol_frac=0
        vol_guard = -self.vol_penalty * np.exp(
            -vol_frac / max(self.vol_steepness, 1e-6)
        )

        # Price guardrail: exponential penalty near price_frac=0
        price_guard = -self.price_penalty * np.exp(
            -price_frac / max(self.price_steepness, 1e-6)
        )

        reward = profit + shaping + vol_guard + price_guard

        # ── Diagnostics ─────────────────────────────────────────────
        _episode = getattr(self.learning_role, 'episode_count', 0) if self.learning_mode else -1
        if _episode <= 2 and self.learning_mode:
            _hour = start.hour if hasattr(start, 'hour') else -1
            logger.info(
                f"PROFIT_REWARD ep={_episode} h={_hour:02d} | "
                f"bid=${bid_price:.1f} vol={bid_volume:.0f}MW "
                f"mcp=${market_clearing_price:.1f} acc={accepted_volume_total:.0f}MW "
                f"profit=${profit:.0f} ceil_w=${price_ceil_wavelet:.1f} "
                f"mc_w=${mc_floor_wavelet:.1f} vol_w=${vol_wavelet:.1f} "
                f"mc_mtn=${mc_mountain:.1f} (resid={residual:.0f}MW) "
                f"guards=${vol_guard + price_guard:.1f} reward=${reward:.0f}"
            )

        # Store results
        unit.outputs["profit"].loc[start:end_excl] += profit
        unit.outputs["total_costs"].loc[start:end_excl] += operational_cost

        if self.learning_mode:
            self.learning_role.add_reward_to_cache(
                unit.id, start, reward, 0.0, profit
            )

    def _get_counterfactual_price(self, unit, start) -> float:
        """Read per-agent counterfactual price (SCED with this agent removed)."""
        cf = getattr(unit, 'counterfactual_price', None)
        if cf is not None:
            try:
                return float(cf[start])
            except (KeyError, IndexError, TypeError):
                pass
        # Fallback to observation forecast
        try:
            price_series = unit.forecaster.price["EOM"]
            if hasattr(price_series, '__getitem__'):
                return float(price_series[start])
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        return self.max_bid_price

    # ── Exploration override (same as WithholdingStrategy) ──────────────

    def get_actions(self, next_observation):
        # ── DQN: epsilon-greedy ALWAYS (bypasses BC initial experience) ──
        # DQN must use random discrete actions for exploration, not BC centers.
        # BC-centered Gaussian noise concentrates data in a small region of the
        # discrete action space, causing Q-value overestimation for those bins.
        # Epsilon-greedy over the full discrete space ensures diverse coverage.
        if (
            self.learning_mode
            and hasattr(self, 'discretizer')
            and self.discretizer is not None
        ):
            from assume.reinforcement_learning.algorithms.dqn import DQN
            rl_algo = self.learning_role.rl_algorithm if hasattr(self, 'learning_role') else None
            if isinstance(rl_algo, DQN):
                if self.evaluation_mode:
                    # Eval: pure greedy
                    curr_action = self.actor(next_observation).detach()
                elif rl_algo.use_boltzmann:
                    # Phase 2: Boltzmann sampling — sample proportional to
                    # exp(Q/τ), naturally probes bins near the current best
                    q_values = self.actor.q_network(next_observation)
                    tau = max(rl_algo.boltzmann_tau, 1e-4)
                    probs = th.softmax(q_values / tau, dim=-1)
                    idx = th.multinomial(probs, 1).item()
                    curr_action = self.discretizer.index_to_continuous(
                        th.tensor(idx, device=self.device)
                    )
                elif th.rand(1).item() < rl_algo.epsilon:
                    # Phase 1: epsilon-greedy random
                    rand_idx = th.randint(0, self.discretizer.num_actions, (1,)).item()
                    curr_action = self.discretizer.index_to_continuous(
                        th.tensor(rand_idx, device=self.device)
                    )
                else:
                    # Phase 1: epsilon-greedy exploit
                    curr_action = self.actor(next_observation).detach()
                noise = th.zeros(2, dtype=self.float_type, device=self.device)
                return curr_action, noise

        # ── BC initial experience (continuous algorithms only) ────────
        if (
            self.learning_mode
            and not self.evaluation_mode
            and self.collect_initial_experience_mode
            and self._exploration_center is not None
        ):
            center = self._exploration_center
            if isinstance(center, list):
                if not hasattr(self, '_explore_step'):
                    self._explore_step = 0
                hour = self._explore_step % len(center)
                self._explore_step += 1
                c0, c1 = center[hour]
            else:
                c0, c1 = center[0], center[1]

            c0_scaled = c0 * 2 - 1
            c1_scaled = c1 * 2 - 1

            if isinstance(self.action_noise, _PerDimActionNoise) and (
                self.action_noise.sigma_price > 0 or self.action_noise.sigma_volume > 0
            ):
                sigma_p = self.action_noise.sigma_price
                sigma_v = self.action_noise.sigma_volume
            else:
                # Fallback for PPO / zero-noise configs: use exploration_noise_std
                sigma_p = sigma_v = getattr(self, 'exploration_noise_std', 0.3)

            noise = th.tensor([
                float(th.normal(mean=0.0, std=sigma_p, size=(1,))),
                float(th.normal(mean=0.0, std=sigma_v, size=(1,))),
            ], dtype=self.float_type, device=self.device)

            curr_action = th.tensor(
                [c0_scaled, c1_scaled], dtype=self.float_type, device=self.device
            ) + noise
            curr_action = th.clamp(curr_action, -1.0, 1.0)

            return curr_action, noise

        curr_action, noise = super(EnergyLearningStrategy, self).get_actions(
            next_observation
        )
        return curr_action, noise
