# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
RL strategy that lets the agent control both bid price AND offered volume.

The standard EnergyLearningStrategy uses 2 actions for 2 prices (Pmin, flex).
Volumes are hardcoded to min_power and max_power - min_power.  This means the
agent can only exercise market power through *price* manipulation.

In nodal SCED markets, the more powerful mechanism is *economic withholding*:
offer a reduced volume at a competitive price so that a more expensive unit
becomes the marginal generator, raising the LMP above your cost.

This strategy reinterprets the 2 action dimensions as:
    action[0] -> bid price  (scaled to [MC, price_cap])
    action[1] -> volume fraction  (0 = offer nothing, 1 = offer max_power)

The agent submits a single bid with (price, volume) where volume = action[1] * max_power.

Observation space (2 values):
    [0] price forecast this hour  (scaled 0-1 within episode min/max)
    [1] load forecast this hour   (scaled 0-1 within episode min/max)

Reward — two regimes based on price forecast vs MC:
    OFF-PEAK (forecast < MC):
        Fixed reward for competitive bidding: bonus for full volume, penalty
        for price above MC.  No profit to chase, just train competitive eq.

    PEAK (forecast >= MC):
        Pure profit signal from actual SCED dispatch.  The agent keeps what
        it earns: (MCP - MC) * accepted_volume.  No artificial penalties.
"""

import logging
import numpy as np
import torch as th
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

from assume.common.fast_pandas import FastSeries
from assume.common.market_objects import MarketConfig, Orderbook
from assume.strategies.learning_strategies import EnergyLearningStrategy


class _PerDimActionNoise:
    """Action noise with independent sigma per dimension (price vs volume).

    Drop-in replacement for NormalActionNoise.  Dimension 0 uses
    ``sigma_price``, dimension 1 uses ``sigma_volume``.  This lets you
    crank up volume exploration without disturbing price noise.
    """

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


class WithholdingStrategy(EnergyLearningStrategy):
    """RL strategy with price + volume control for economic withholding."""

    def __init__(self, *args, **kwargs):
        # Minimal observation: just 2 scalars (price forecast, load forecast)
        # No time series windows — with gamma=0 each hour is independent.
        kwargs['num_timeseries_obs_dim'] = 0
        kwargs['foresight'] = 1
        kwargs['unique_obs_dim'] = 2  # price forecast, load forecast
        # Exploration center: action-space fractions [price_frac, volume_frac]
        # used during initial-experience mode instead of hardcoded 0.5.
        self._exploration_center = kwargs.pop('exploration_center', None)
        # Per-dimension noise: optional separate sigma for the volume action.
        self._noise_sigma_volume = kwargs.pop('noise_sigma_volume', None)
        # Sigmoid parameters for peak reward shaping
        self.tau = kwargs.pop('tau', 0.3)                         # sigmoid steepness ($/MWh)
        self.sigmoid_offset_max = kwargs.pop('sigmoid_offset_max', 1.0)   # initial offset above cf
        self.sigmoid_offset_min = kwargs.pop('sigmoid_offset_min', 0.05)  # floor offset
        self.sigmoid_offset_anneal_rate = kwargs.pop('sigmoid_offset_anneal_rate', 0.01)
        self.popart_enabled = kwargs.pop('popart_enabled', False)
        # Pop any leftover strategy-specific kwargs
        kwargs.pop('obs_horizon_decay', None)
        super().__init__(*args, **kwargs)
        # act_dim=2 inherited: action[0]=price, action[1]=volume fraction

        # Override the parent's uniform action_noise with per-dimension sigmas
        if self._noise_sigma_volume is not None and self.learning_mode:
            self.action_noise = _PerDimActionNoise(
                sigma_price=self.learning_config.noise_sigma,
                sigma_volume=self._noise_sigma_volume,
                scale=self.learning_config.noise_scale,
                dt=self.learning_config.noise_dt,
            )

        # Track bid volume and price per timestep for reward
        self._last_bid_volume: dict = {}  # {start_time: bid_volume}
        self._last_bid_price: dict = {}   # {start_time: bid_price}

    def prepare_observations(self, unit, market_id):
        """Cache raw price and load arrays for the episode."""
        super().prepare_observations(unit, market_id)

        price_series = unit.forecaster.price[market_id]
        self._raw_price_fcast = np.array([price_series[t] for t in unit.index], dtype=float)

        res_load_series = unit.forecaster.residual_load.get(
            market_id, FastSeries(index=unit.index, value=0)
        )
        self._raw_res_load = np.array([res_load_series[t] for t in unit.index], dtype=float)

        self._price_min = float(self._raw_price_fcast.min())
        self._price_max = float(self._raw_price_fcast.max())
        self._load_min  = float(self._raw_res_load.min())
        self._load_max  = float(self._raw_res_load.max())

    def create_observation(self, unit, market_id, start, end):
        """Build observation: just this hour's price forecast and load forecast.

        Two scalars, scaled to [0, 1] within the episode's range.
        With gamma=0 each hour is independent — the network only needs
        to know the current hour's conditions to decide price and volume.
        """
        if not hasattr(self, '_raw_price_fcast'):
            self.prepare_observations(unit, market_id)

        h = int((start - unit.index[0]).total_seconds() / 3600)

        price_range = max(self._price_max - self._price_min, 1e-6)
        load_range  = max(self._load_max  - self._load_min,  1e-6)

        scaled_price = (self._raw_price_fcast[h] - self._price_min) / price_range
        scaled_load  = (self._raw_res_load[h]    - self._load_min)  / load_range

        observation = th.tensor(
            [scaled_price, scaled_load],
            dtype=self.float_type, device=self.device,
        )

        if self.learning_mode:
            self.learning_role.add_observation_to_cache(self.unit_id, start, observation)

        return observation

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        start = product_tuples[0][0]
        end = product_tuples[0][1]

        min_power, max_power = unit.calculate_min_max_power(start, end)
        min_power = min_power[0]
        max_power = max_power[0]

        # 1. Observe
        next_observation = self.create_observation(
            unit=unit,
            market_id=market_config.market_id,
            start=start,
            end=end,
        )

        # 2. Act
        actions, noise = self.get_actions(next_observation)

        # 3. Transform: rescale tanh/softsign [-1,1] -> [0,1] for both dims.
        #    This uses the full actor output range (no dead gradient zone).
        mc = float(unit.calculate_marginal_cost(start, max_power))
        price_floor = max(mc, self.min_bid_price)
        price_ceiling = getattr(unit, "price_cap", self.max_bid_price)
        a0 = float(actions[0] + 1) / 2.0  # [-1,1] -> [0,1]
        a0 = max(0.0, min(1.0, a0))       # safety clamp for numerical edge cases
        bid_price = price_floor + a0 * (price_ceiling - price_floor)

        # Volume: rescale [-1,1] -> [0,1] -> fraction of max_power
        vol_fraction = float(actions[1] + 1) / 2.0
        vol_fraction = max(0.0, min(1.0, vol_fraction))
        bid_volume = min_power + vol_fraction * (max_power - min_power)

        # Store for reward calculation
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
            self.learning_role.add_actions_to_cache(self.unit_id, start, actions, noise)

        return bids

    def calculate_reward(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """Two-regime reward based on price forecast vs MC.

        OFF-PEAK (forecast < MC):
            The agent can't profit -- cheaper generators cover demand.
            Reward = vol_frac - price_frac
            Full volume at MC = +1.0, zero volume at price_cap = -1.0.

        PEAK (forecast >= MC):
            Sigmoid-smoothed profit using the counterfactual price
            (SCED with this agent removed) as the dispatch cliff.
            shaped_profit = sigma((cf - bid_price) / tau) * (MCP - MC) * accepted_vol
            Normalized to roughly [-1, 1] range.
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

        # Actual dispatch and profit from SCED
        income = 0.0
        operational_cost = 0.0
        accepted_volume_total = 0

        for order in orderbook:
            accepted_volume = order.get("accepted_volume", 0)
            accepted_volume_total += accepted_volume
            income += market_clearing_price * accepted_volume * duration
            operational_cost += marginal_cost * accepted_volume * duration

        # Start-up cost handling (same as base)
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

        # Retrieve bid decisions
        bid_volume = self._last_bid_volume.get(start, unit.max_power)
        bid_price  = self._last_bid_price.get(start, marginal_cost)

        vol_frac = bid_volume / max(unit.max_power, 1.0)
        price_cap = getattr(unit, "price_cap", self.max_bid_price)
        price_frac = float(np.clip(
            (bid_price - marginal_cost) / max(price_cap - marginal_cost, 1.0),
            0.0, 1.0,
        ))

        # Get forecast price for this hour
        forecast_price = self._get_forecast_price(unit, marketconfig.market_id, start)

        # Diagnostic: log reward components for first episode
        _episode = getattr(self.learning_role, 'episode_count', 0) if self.learning_mode else -1
        if _episode <= 2 and self.learning_mode:
            _hour = start.hour if hasattr(start, 'hour') else -1
            logger.info(
                f"REWARD_DBG ep={_episode} h={_hour:02d} | "
                f"forecast=${forecast_price:.1f} mc=${marginal_cost:.1f} "
                f"bid=${bid_price:.1f} vol={bid_volume:.0f}MW "
                f"mcp=${market_clearing_price:.1f} acc_vol={accepted_volume_total:.0f}MW "
                f"regime={'OFF' if forecast_price < marginal_cost else 'PEAK'}"
            )

        if forecast_price < marginal_cost:
            # ── OFF-PEAK: reward competitive equilibrium ───────────────
            # vol_frac=1, price_frac=0 (MC) -> reward = +1.0
            # vol_frac=0, price_frac=1 (cap) -> reward = -1.0
            reward = vol_frac - price_frac
        else:
            # ── PEAK: offset-sigmoid profit ────────────────────────────
            # Sigmoid centered ABOVE the counterfactual by an offset,
            # so bids at or just below the cliff get nearly full reward.
            # The offset anneals to zero over training:
            #   Early: center at cf+$1 → forgiving near the cliff
            #   Late:  center at cf+$0.05 → nearly hard step function
            #
            #   bid=$49.00 → 0.999    bid=$50.00 → 0.966
            #   bid=$50.50 → 0.841    bid=$52.00 → 0.034
            counterfactual = self._get_counterfactual_price(unit, start)
            tau = self._get_tau()
            offset = self._get_sigmoid_offset()
            price_diff = (counterfactual + offset) - bid_price
            soft_dispatch = 1.0 / (1.0 + np.exp(-price_diff / max(tau, 0.01)))

            # Shaped profit uses accepted_volume (actual SCED dispatch)
            shaped_profit = (
                soft_dispatch
                * max(market_clearing_price - marginal_cost, 0.0)
                * accepted_volume_total
                * duration
            )

            if self.popart_enabled:
                # PopArt: pass raw shaped profit — the critic normalizes
                # adaptively.  No manual normalizer needed.
                reward = shaped_profit
            else:
                # Normalize by a FIXED reference profit that doesn't depend
                # on the agent's bid.  This ensures that earning more total
                # profit gives a higher reward — otherwise normalizing by
                # bid_volume makes MC bids at tiny volume always score 1.0.
                #
                # Reference: (cf - MC) × max_power/3
                #   $35 at 28MW, profit=$420 → reward = 420/1500 = 0.28
                #   $48 at 65MW, profit=$845 → reward = 845/1500 = 0.56
                typical_dispatch = unit.max_power / 3
                normalizer = (counterfactual - marginal_cost) * typical_dispatch * duration
                reward = shaped_profit / max(normalizer, 1.0)

        if _episode <= 2 and self.learning_mode:
            if forecast_price >= marginal_cost:
                logger.info(
                    f"  -> PEAK reward={reward:.2f} shaped_profit={shaped_profit:.1f} "
                    f"cf=${counterfactual:.1f} soft_dispatch={soft_dispatch:.3f} "
                    f"offset={offset:.2f} popart={self.popart_enabled}"
                )
            else:
                logger.info(f"  -> OFF reward={reward:.3f} vol_frac={vol_frac:.2f} price_frac={price_frac:.2f}")

        # Store results
        unit.outputs["profit"].loc[start:end_excl] += profit
        unit.outputs["total_costs"].loc[start:end_excl] += operational_cost

        if self.learning_mode:
            self.learning_role.add_reward_to_cache(
                unit.id, start, reward, 0.0, profit
            )

    def _get_counterfactual_price(self, unit, start) -> float:
        """Read the per-agent counterfactual price for sigmoid reward shaping.

        This is the LMP from a SCED with only THIS agent's capacity removed.
        Falls back to the observation forecast if no counterfactual is stored.
        """
        cf = getattr(unit, 'counterfactual_price', None)
        if cf is not None:
            try:
                return float(cf[start])
            except (KeyError, IndexError, TypeError):
                pass
        return self._get_forecast_price(unit, "EOM", start)

    def _get_tau(self) -> float:
        """Sigmoid steepness — fixed, configured via YAML ``tau``."""
        return self.tau

    def _get_sigmoid_offset(self) -> float:
        """Offset above counterfactual, annealed over training.

        Configured via YAML:
          sigmoid_offset_max: starting offset (default $1.00)
          sigmoid_offset_min: floor offset (default $0.05)
          sigmoid_offset_anneal_rate: exp decay per episode (default 0.01)

        In eval mode, uses offset_min (tight cliff).
        """
        if not self.learning_mode or not hasattr(self.learning_role, 'episode_count'):
            return self.sigmoid_offset_min
        episode = getattr(self.learning_role, 'episode_count', 0)
        offset = self.sigmoid_offset_max * np.exp(-self.sigmoid_offset_anneal_rate * episode)
        return max(offset, self.sigmoid_offset_min)

    def _get_forecast_price(self, unit, market_id: str, start) -> float:
        """Read the price forecast for this hour from the unit's forecaster."""
        try:
            price_series = unit.forecaster.price[market_id]
            if hasattr(price_series, '__getitem__'):
                val = price_series[start]
                return float(val)
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        return self.max_bid_price

    def get_actions(self, next_observation):
        """Override to center exploration on the BC-trained policy.

        During initial experience, samples actions from a normal distribution
        CENTERED ON the Cournot targets (per-hour price and RSI volume),
        not on zero.  This ensures exploration covers the theoretically
        optimal region rather than being uniformly random.

        Uses a step counter to track which hour we're in (since hour-of-day
        is no longer in the observation).
        """
        if (
            self.learning_mode
            and not self.evaluation_mode
            and self.collect_initial_experience_mode
            and self._exploration_center is not None
        ):
            # Sample centered on per-hour BC target in [-1,1] space
            center = self._exploration_center
            if isinstance(center, list):
                if not hasattr(self, '_explore_step'):
                    self._explore_step = 0
                hour = self._explore_step % len(center)
                self._explore_step += 1
                c0, c1 = center[hour]
            else:
                c0, c1 = center[0], center[1]

            # Convert [0,1] centers to [-1,1] actor output space
            c0_scaled = c0 * 2 - 1
            c1_scaled = c1 * 2 - 1

            # Use per-dim noise if available, else fall back to exploration_noise_std
            if isinstance(self.action_noise, _PerDimActionNoise):
                sigma_p = self.action_noise.sigma_price
                sigma_v = self.action_noise.sigma_volume
            else:
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

        # Normal mode: use actor network + noise
        curr_action, noise = super(EnergyLearningStrategy, self).get_actions(
            next_observation
        )
        return curr_action, noise
