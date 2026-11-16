# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
RL training loop for IEEE 9-bus testbed with Egret SCED clearing.

Trains RL agents (gen_0, gen_1) to learn strategic bidding in a nodal
SCED market while gen_slack_0 bids naively at marginal cost.

Usage:
    cd assume/  # repo root
    python -m testbed.learning_runner
    python -m testbed.learning_runner --episodes 50 --mechanism egret_sced
    python -m testbed.learning_runner --scenario congested
    python -m testbed.learning_runner --scenario heavy_congestion --episodes 200
"""

import logging
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import rrule as rr
from tqdm import tqdm

# Ensure repo root on sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from assume import World
from assume.common.forecaster import PowerplantForecaster, DemandForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.markets import clearing_mechanisms
from assume.strategies import bidding_strategies
from testbed.clearing.sced_clearing import ScedClearingRole
from testbed.ieee9_network import build_ieee9_network, assign_zones
from testbed.strategies.withholding_strategy import WithholdingStrategy
from testbed.strategies.profit_strategy import ProfitStrategy

logger = logging.getLogger(__name__)

# Register Egret SCED clearing
clearing_mechanisms["egret_sced"] = ScedClearingRole

# Register withholding strategy (price + volume control)
bidding_strategies["powerplant_energy_withholding"] = WithholdingStrategy

# Register pure-profit strategy (unified reward, no regime switch)
bidding_strategies["powerplant_energy_profit"] = ProfitStrategy

# Default RL generators — can be overridden per scenario via YAML `training.rl_agents`
RL_GENERATORS = {"gen_0", "gen_1"}

# Default RL strategy — can be overridden per scenario via YAML `training.rl_strategy`
RL_STRATEGY = "powerplant_energy_learning"


def _apply_generator_overrides(network, generator_params: dict) -> None:
    """Apply YAML generator overrides (price, capacity, bus) to the PyPSA network.

    This lets scenarios.yaml be the single source of truth for generator
    parameters.  Also supports adding new generators not in the base case9
    (e.g. gen_peaker_0).

    Recognized keys per generator: bus, p_max, price.
    Other keys (e.g. fixed_cost_rate) are ignored here — they're consumed
    downstream by setup_learning_world.
    """
    for gen_name, params in generator_params.items():
        if gen_name in network.generators.index:
            # Override existing generator
            if "price" in params:
                network.generators.at[gen_name, "marginal_cost"] = float(params["price"])
            if "p_max" in params:
                network.generators.at[gen_name, "p_nom"] = float(params["p_max"])
            if "bus" in params:
                network.generators.at[gen_name, "bus"] = str(params["bus"])
        else:
            # Add a new generator (e.g. gen_peaker_0)
            if "bus" not in params or "p_max" not in params or "price" not in params:
                logger.warning(
                    f"Skipping new generator '{gen_name}': needs bus, p_max, price"
                )
                continue
            network.add(
                "Generator",
                gen_name,
                bus=str(params["bus"]),
                p_nom=float(params["p_max"]),
                p_nom_min=0.0,
                marginal_cost=float(params["price"]),
                carrier=params.get("carrier", "gas"),
                ramp_limit_start_up=0.5,
                ramp_limit_shut_down=0.5,
                min_up_time=2,
                min_down_time=1,
            )
            logger.info(
                f"Added generator '{gen_name}': bus={params['bus']}, "
                f"p_max={params['p_max']}, price={params['price']}"
            )

    gen_summary = ", ".join(
        f"{g}(${network.generators.at[g, 'marginal_cost']:.0f}, "
        f"{network.generators.at[g, 'p_nom']:.0f}MW)"
        for g in network.generators.index
    )
    logger.info(f"Generator overrides applied: {gen_summary}")


def _apply_line_overrides(network, line_overrides: dict) -> None:
    """Apply YAML line overrides (s_nom, etc.) to the PyPSA network.

    This lets scenarios.yaml override line thermal ratings for congestion
    studies.  Egret reads line limits from the PyPSA network via grid_data,
    so overrides here propagate to the SCED LP.

    Recognized keys per line: s_nom (MVA thermal rating).
    """
    for line_name, params in line_overrides.items():
        if line_name not in network.lines.index:
            logger.warning(f"Line override: '{line_name}' not in network, skipping")
            continue
        if "s_nom" in params:
            network.lines.at[line_name, "s_nom"] = float(params["s_nom"])

    line_summary = ", ".join(
        f"{l}({network.lines.at[l, 's_nom']:.0f}MW)"
        for l in network.lines.index
    )
    logger.info(f"Line overrides applied: {line_summary}")


def _compute_residual_load(network, datetime_index):
    """Compute total system demand as residual load (no renewables in IEEE 9-bus)."""
    total_demand = network.loads_t.p_set.sum(axis=1)
    total_demand = total_demand.reindex(datetime_index)
    return total_demand


def _compute_naive_lmp_forecast(
    network, datetime_index, exclude_agents: set | None = None,
) -> dict[str, pd.Series]:
    """Run one SCED *without* the RL agent(s) to get counterfactual LMPs.

    Returns a dict {bus_id: pd.Series} so each generator can receive the LMP
    forecast for its specific bus.  This matters under congestion where buses
    have different LMPs.  In uncongested cases all buses return the same value.

    By excluding the RL agent's capacity, the SCED shows what prices WOULD be
    if the agent didn't participate.  This gives a much stronger signal than
    including the agent at MC (which just shows the agent's own cost as the LMP).

    For the IEEE 9-bus withholding case: without gen_0, gen_1 ($50) becomes
    marginal in peak hours → LMP ≈ $50.  The agent sees "$50 is the price
    without me" and learns it can bid anywhere in ($35, $50) profitably.
    With the old approach (gen_0 included at MC), the forecast was flat $35 —
    identical to the agent's own cost — giving zero profit signal.
    """
    from testbed.clearing.sced_clearing import clear_sced
    from datetime import timedelta

    if exclude_agents is None:
        exclude_agents = set()

    gen_df = network.generators.copy()
    gen_df.rename(columns={"bus": "node", "p_nom": "max_power"}, inplace=True)

    loads_df = network.loads.copy()
    loads_df.rename(columns={"bus": "node", "p_set": "min_power"}, inplace=True)

    grid_data = {
        "buses": network.buses,
        "lines": network.lines,
        "generators": gen_df,
        "loads": loads_df,
    }

    # Build naive orderbook: each non-RL gen bids MC for full capacity
    orderbook = []
    market_products = []
    for i, ts in enumerate(datetime_index):
        start = ts
        end = ts + timedelta(hours=1)
        market_products.append((start, end, None))

        for gen_name, gen in gen_df.iterrows():
            if gen_name in exclude_agents:
                continue  # Skip RL agents — counterfactual pricing
            orderbook.append({
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": float(gen["marginal_cost"]),
                "volume": float(gen["max_power"]),
                "node": gen["node"],
                "unit_id": gen_name,
                "bid_id": f"{gen_name}_{i}",
            })

        # Demand bids (high willingness to pay)
        for load_name in network.loads.index:
            if load_name in network.loads_t.p_set.columns:
                demand = float(network.loads_t.p_set.at[ts, load_name])
            else:
                demand = float(loads_df.at[load_name, "min_power"])
            orderbook.append({
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": 1000.0,
                "volume": -demand,
                "node": network.loads.at[load_name, "bus"],
                "unit_id": load_name,
                "bid_id": f"{load_name}_{i}",
            })

    # Run SCED
    accepted, rejected, meta, flows = clear_sced(
        orderbook, market_products, grid_data, solver="highs",
    )

    # Extract per-bus LMPs — one series per bus so each generator sees its own
    # nodal price forecast.  Matters under congestion where bus LMPs diverge.
    bus_lmps_by_hour: dict = {}
    system_price_by_hour: dict = {}
    for m in meta:
        ts = m["product_start"]
        system_price_by_hour[ts] = m["price"]
        bus_lmps_by_hour[ts] = m.get("bus_lmps", {})

    all_buses = list(network.buses.index)
    bus_series: dict[str, pd.Series] = {}
    for bus in all_buses:
        hourly = {}
        for ts in datetime_index:
            bus_lmps = bus_lmps_by_hour.get(ts, {})
            hourly[ts] = bus_lmps.get(bus, system_price_by_hour.get(ts, 0.0))
        s = pd.Series(hourly).reindex(datetime_index)
        bus_series[bus] = s.fillna(s.mean())

    excluded_str = ", ".join(exclude_agents) if exclude_agents else "none"
    for bus, s in bus_series.items():
        logger.info(
            f"Counterfactual LMP forecast bus {bus} (excluded: {excluded_str}): "
            f"min=${s.min():.1f}, max=${s.max():.1f}, mean=${s.mean():.1f}"
        )
    return bus_series


def setup_learning_world(
    network,
    episode: int = 1,
    eval_episode: int = 1,
    learning_config: dict | None = None,
    market_mechanism: str = "egret_sced",
    db_uri: str = "",
    export_csv: str = "",
    price_forecast: dict[str, pd.Series] | None = None,
    generator_params: dict | None = None,
    per_agent_counterfactual: dict[str, dict[str, pd.Series]] | None = None,
) -> World:
    """Build a World with learning support for the IEEE 9-bus SCED testbed.

    Args:
        network: PyPSA network (IEEE 9-bus).
        episode: Training episode number.
        eval_episode: Evaluation episode number.
        learning_config: Learning configuration dict.
        market_mechanism: Clearing algorithm name.
        db_uri: Database URI for results/rewards.
        export_csv: CSV export path.
        price_forecast: Bus-keyed dict {bus_id: pd.Series} of LMP forecasts.
                        Each generator receives the forecast for its own bus.
        generator_params: Per-generator config from scenarios.yaml (e.g. fixed_cost_rate).
        per_agent_counterfactual: Per-agent counterfactual forecasts for sigmoid
                        reward shaping. {agent_id: {bus_id: pd.Series}}.
                        Each agent gets a SCED with only its own capacity removed.

    Returns:
        Configured World ready for world.run().
    """
    if generator_params is None:
        generator_params = {}
    datetime_index = pd.DatetimeIndex(network.snapshots)
    start = datetime_index[0]
    end = datetime_index[-1]

    simulation_id = f"ieee9_rl_{market_mechanism}"

    world = World(
        database_uri=db_uri,
        export_csv_path=export_csv,
        log_level="WARNING",
    )

    # Pass strategy-specific params (e.g. obs_horizon_decay) via bidding_params
    # so they reach the WithholdingStrategy constructor as kwargs.
    # These must be stripped from learning_dict before world.setup() because
    # LearningConfig only accepts known RL hyperparameters.
    _lc = dict(learning_config or {})
    _bidding_params = {}
    _strategy_keys = ["obs_horizon_decay", "noise_sigma_volume",
                      "tau", "sigmoid_offset_max", "sigmoid_offset_min",
                      "sigmoid_offset_anneal_rate",
                      "vol_penalty", "vol_steepness",
                      "price_penalty", "price_steepness",
                      # Ricker wavelet params (current)
                      "wavelet_scale", "vol_wavelet_scale",
                      "price_wavelet_width", "vol_wavelet_width",
                      "mc_mountain_scale", "mc_mountain_sigma",
                      "mc_mountain_tau", "shaping_anneal",
                      # Legacy params (popped by ProfitStrategy.__init__)
                      "mountain_scale", "vol_ridge_scale",
                      "mountain_sigma", "mountain_tau", "vol_ridge_tau"]
    for _key in _strategy_keys:
        if _key in _lc:
            _bidding_params[_key] = _lc.pop(_key)
    # popart_enabled is needed by BOTH LearningConfig and the strategy reward,
    # so copy (don't pop) it into bidding_params.
    if _lc.get("popart_enabled"):
        _bidding_params["popart_enabled"] = _lc["popart_enabled"]
    # Pop keys that are used in the training loop but not by LearningConfig
    _lc.pop("param_noise_sigma", None)
    _lc.pop("bc_reexplore_interval", None)
    _lc.pop("warmup_noise_scale", None)

    world.setup(
        start=start,
        end=end,
        simulation_id=simulation_id,
        save_frequency_hours=48,
        bidding_params=_bidding_params,
        learning_dict=_lc,
        episode=episode,
        eval_episode=eval_episode,
        index=datetime_index,
    )

    # ── Rename columns for ASSUME compatibility ──────────────────────
    # Work on copies to avoid modifying the original network across episodes
    gen_df = network.generators.copy()
    loads_df = network.loads.copy()

    gen_df.rename(columns={"bus": "node", "p_nom": "max_power"}, inplace=True)
    loads_df.rename(columns={"bus": "node", "p_set": "min_power"}, inplace=True)
    if "max_power" not in loads_df.columns:
        loads_df["max_power"] = 0

    grid_data = {
        "buses": network.buses,
        "lines": network.lines,
        "generators": gen_df,
        "loads": loads_df,
    }

    # ── Market design ────────────────────────────────────────────────
    additional_fields = ["node", "max_power", "min_power"]
    param_dict = {}
    if market_mechanism == "egret_sced":
        param_dict["solver"] = "highs"

    marketdesign = [
        MarketConfig(
            market_id="EOM",
            opening_hours=rr.rrule(
                rr.HOURLY, interval=1, dtstart=start, until=end
            ),
            opening_duration=timedelta(hours=1),
            market_mechanism=market_mechanism,
            market_products=[
                MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))
            ],
            additional_fields=additional_fields,
            maximum_bid_volume=1e9,
            maximum_bid_price=1e9,
            param_dict=param_dict,
        )
    ]

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    for mc in marketdesign:
        mc.param_dict["grid_data"] = grid_data
        world.add_market(mo_id, mc)

    # ── Compute forecasts for RL observations ────────────────────────
    residual_load = _compute_residual_load(network, datetime_index)
    # Counterfactual LMP forecast: SCED without RL agents shows what prices
    # would be if they didn't participate. Gives a stronger learning signal.
    if price_forecast is None:
        price_forecast = _compute_naive_lmp_forecast(
            network, datetime_index, exclude_agents=RL_GENERATORS,
        )
    # Build a system-average fallback for buses not in the dict
    _avg_forecast = pd.concat(list(price_forecast.values()), axis=1).mean(axis=1)

    # ── Generators ───────────────────────────────────────────────────
    world.add_unit_operator("powerplant_operator")

    for gen_name, generator in gen_df.iterrows():
        # Determine strategy: RL for mid-merit/peaker, naive for baseload
        if gen_name in RL_GENERATORS:
            strategy_name = RL_STRATEGY
        else:
            strategy_name = "powerplant_energy_naive"

        bidding_strategies = {mc.market_id: strategy_name for mc in marketdesign}

        max_power = generator.max_power or 1000
        ramp_up = generator.ramp_limit_start_up * max_power
        ramp_down = generator.ramp_limit_shut_down * max_power
        min_operating_time = max(generator.min_up_time, 1)
        min_down_time = max(generator.min_down_time, 1)
        min_power = min(generator.p_nom_min, max_power)

        # Build fuel prices mapping
        mc_val = generator.marginal_cost
        gen_series = pd.Series(float(mc_val), index=datetime_index)
        fuel_prices_mapping = {gen_name: gen_series}
        if getattr(generator, "carrier", None):
            fuel_prices_mapping[generator.carrier] = gen_series.copy()

        # Market prices: use the bus-specific LMP forecast for this generator.
        # Each agent sees the nodal price at its own bus, not a system average.
        gen_bus = str(generator["node"])
        bus_forecast = price_forecast.get(gen_bus, _avg_forecast)
        market_prices = {"EOM": bus_forecast.copy()}
        res_load_dict = {"EOM": residual_load.copy()}

        forecaster = PowerplantForecaster(
            datetime_index,
            fuel_prices=fuel_prices_mapping,
            market_prices=market_prices,
            residual_load=res_load_dict,
            availability=1,
        )

        world.add_unit(
            gen_name,
            "power_plant",
            "powerplant_operator",
            {
                "min_power": min_power,
                "max_power": max_power,
                "bidding_strategies": bidding_strategies,
                "technology": "conventional",
                "node": generator.node,
                "efficiency": 1,
                "fuel_type": generator.carrier,
                "ramp_up": ramp_up,
                "ramp_down": ramp_down,
                "min_operating_time": min_operating_time,
                "min_down_time": min_down_time,
            },
            forecaster,
        )

        # Attach per-generator params from scenarios.yaml
        gen_yaml = generator_params.get(gen_name, {})
        unit_obj = world.unit_operators["powerplant_operator"].units[gen_name]
        unit_obj.fixed_cost_rate  = gen_yaml.get("fixed_cost_rate", 0.0)
        unit_obj.markup_cost_rate = gen_yaml.get("markup_cost_rate", 0.0)
        # price_cap normalises markup_fraction in the reward; defaults to scenario max_bid_price
        unit_obj.price_cap = gen_yaml.get("price_cap", learning_config.get("max_bid_price", 100.0))
        # Volume S-curve shaping (see WithholdingStrategy.calculate_reward)
        unit_obj.volume_shaping_rate = gen_yaml.get("volume_shaping_rate", 0.0)
        unit_obj.volume_shaping_midpoint = gen_yaml.get("volume_shaping_midpoint", 0.5)
        unit_obj.volume_shaping_steepness = gen_yaml.get("volume_shaping_steepness", 0.1)

        # Cheaper supply: total competitor capacity priced below this gen's MC.
        # Used by ProfitStrategy for residual demand cliff shaping.
        own_mc = float(generator.marginal_cost)
        cheaper_mask = (gen_df["marginal_cost"] < own_mc - 0.01) & (gen_df.index != gen_name)
        unit_obj.cheaper_supply = float(gen_df.loc[cheaper_mask, "max_power"].sum())

        # Per-agent counterfactual price for sigmoid reward shaping.
        # Stored as a plain dict {timestamp: float} for fast lookup
        # without depending on FastIndex plumbing.
        if per_agent_counterfactual and gen_name in per_agent_counterfactual:
            cf_bus_dict = per_agent_counterfactual[gen_name]
            cf_series = cf_bus_dict.get(gen_bus, list(cf_bus_dict.values())[0])
            unit_obj.counterfactual_price = {
                ts: float(cf_series.at[ts]) for ts in datetime_index
            }
        else:
            unit_obj.counterfactual_price = None

    # ── Loads ─────────────────────────────────────────────────────────
    world.add_unit_operator("demand_operator")

    for load_name, load in loads_df.iterrows():
        if load_name not in network.loads_t["p_set"].columns:
            continue

        load_t = network.loads_t["p_set"][load_name].reindex(datetime_index)
        load_t_negated = -load_t

        bidding_strategies = {mc.market_id: "demand_energy_naive" for mc in marketdesign}

        world.add_unit(
            load_name,
            "demand",
            "demand_operator",
            {
                "min_power": 0,
                "max_power": -load_t.max(),
                "bidding_strategies": bidding_strategies,
                "technology": "demand",
                "node": load.node,
                "price": 1e3,
            },
            DemandForecaster(datetime_index, demand=load_t_negated),
        )

    return world


def _track_rewards(world: World) -> list:
    """Patch learning_role.add_reward_to_cache to accumulate rewards in a list.

    Returns the list that will be populated as rewards arrive during world.run().
    This is reliable regardless of whether store_to_buffer_and_update fires
    (which swaps and clears the internal cache).
    """
    reward_log = []
    orig_add = world.learning_role.add_reward_to_cache

    def _tracking_add(unit_id, start, reward, regret, profit):
        reward_log.append(float(reward))
        orig_add(unit_id, start, reward, regret, profit)

    world.learning_role.add_reward_to_cache = _tracking_add
    return reward_log


def _clip_rewards(world: World, clip_range: float = 1.0) -> None:
    """Patch add_reward_to_cache to clip rewards to [-clip_range, +clip_range].

    Without this, extreme negative rewards (e.g. -8 per timestep from bidding
    far below MC) saturate the critic — it can't distinguish "bad" from "slightly
    less bad" and the policy gradient becomes noise.  Clipping keeps gradients
    informative so the actor gets a clear "go up" signal.
    """
    orig_add = world.learning_role.add_reward_to_cache

    def _clipped_add(unit_id, start, reward, regret, profit):
        clipped = max(-clip_range, min(clip_range, reward))
        orig_add(unit_id, start, clipped, regret, profit)

    world.learning_role.add_reward_to_cache = _clipped_add


def _force_mc_exploration(world: World, flush_buffer: bool = True) -> None:
    """Force all RL strategies into initial-experience mode for one episode.

    This resets exploration to MC-centered noise (not actor-based), giving the
    replay buffer data from the profitable region.  Without this, once the actor
    drifts to negative bids, local noise (~N(0,0.1)) can never reach the +0.3
    to +0.7 action region where bids are profitable — the policy is trapped.

    If flush_buffer is True, also clears the replay buffer so poisoned data
    (90%+ negative-bid transitions) doesn't dominate the critic after reset.
    """
    for strategy in world.learning_role.rl_strats.values():
        strategy.collect_initial_experience_mode = True

    if flush_buffer:
        buffer = world.learning_role.buffer
        if hasattr(buffer, "clear"):
            buffer.clear()
            logger.info("  Buffer flushed (poisoned data cleared)")
        elif hasattr(buffer, "pos") and hasattr(buffer, "full"):
            buffer.pos = 0
            buffer.full = False
            logger.info("  Replay buffer flushed (poisoned data cleared)")


def _perturb_actor_weights(world: World, sigma: float) -> None:
    """Apply parameter-space noise to actor networks (Plappert et al. 2017).

    Instead of adding noise to actions (which jitters randomly each step),
    this adds Gaussian noise to the actor's weights at episode start.  The
    perturbed network produces a coherent alternative strategy for the
    entire episode — e.g., one episode might consistently bid low volume,
    another might bid higher prices.

    Weights are automatically restored next episode when
    load_inter_episodic_data() reloads the trained (unperturbed) weights.

    Args:
        world: World with RL strategies.
        sigma: Std dev of weight perturbations, relative to each layer's
               weight magnitude.  0.1 = 10% perturbation.
    """
    import torch as th

    for uid, strategy in world.learning_role.rl_strats.items():
        actor = strategy.actor
        with th.no_grad():
            for param in actor.parameters():
                noise = th.randn_like(param) * sigma * (param.abs().mean() + 1e-8)
                param.add_(noise)


def _clamp_actions_nonnegative(world: World) -> None:
    """Patch RL strategies to clamp actions to [0, 1] (non-negative bids only).

    In SCED, generators have no economic reason to bid below $0 — negative bids
    create negative cost curves in the LP, drag LMPs negative, and produce
    catastrophic losses.  Yet the default action space is [-1, 1] with softsign
    activation, so random initialization centers near 0 and exploration noise
    can easily push into the negative half.

    This clamp eliminates the death spiral root cause: the actor can only
    produce bids in [0, max_bid_price], which is the economically meaningful
    range.  The RL agent still has full freedom to bid anywhere from $0
    (price-taker) to max_bid_price (full market power).
    """
    import torch as th

    for uid, strategy in world.learning_role.rl_strats.items():
        original_get_actions = strategy.get_actions

        def _clamped_get_actions(obs, _orig=original_get_actions):
            actions, noise = _orig(obs)
            actions = th.clamp(actions, min=0.0, max=1.0)
            return actions, noise

        strategy.get_actions = _clamped_get_actions
        logger.info(f"  Actions clamped to [0,1] for {uid} (non-negative bids only)")


def _build_dqn_discretizer(world, network) -> None:
    """Build gradient-informed discretizer from network structure.

    Analyzes the merit-order and demand profile to identify:
    - Price gradient zone: near competitor MCs (dispatch cliffs)
    - Volume gradient zone: 0 to peak residual demand (dispatch scaling)

    Bins are concentrated where the reward landscape has steep gradients
    and sparse where it's flat.  This uses the same market structure
    information available to any participant (competitor costs, demand
    forecasts), making it generalizable to any configuration.
    """
    from assume.reinforcement_learning.algorithms.dqn import (
        _auto_gradient_regions_price, _auto_gradient_regions_volume,
    )

    gen_df = network.generators.copy()
    gen_df.rename(columns={"bus": "node", "p_nom": "max_power"}, inplace=True)

    all_mcs = sorted(gen_df["marginal_cost"].unique())

    for uid, strategy in world.learning_role.rl_strats.items():
        unit = None
        for uo in world.unit_operators.values():
            if uid in uo.units:
                unit = uo.units[uid]
                break
        if unit is None:
            continue

        mc = float(unit.calculate_marginal_cost(unit.index[0], 0))
        max_power = float(unit.max_power)
        price_cap = getattr(unit, "price_cap", 60.0)

        # Next competitor MC above ours (the dispatch cliff)
        competitor_mcs = [float(c) for c in all_mcs if c > mc + 0.01]
        competitor_mc = competitor_mcs[0] if competitor_mcs else price_cap

        # Peak residual demand (max volume where dispatch gradient is active)
        cheaper_mask = (gen_df["marginal_cost"] < mc - 0.01) & (gen_df.index != uid)
        cheaper_supply = float(gen_df.loc[cheaper_mask, "max_power"].sum())

        if hasattr(network, "loads_t") and len(network.loads_t.p_set) > 0:
            hourly_demand = network.loads_t.p_set.sum(axis=1).values
            residual = np.maximum(hourly_demand - cheaper_supply, 0.0)
            residual_demand_max = float(residual.max())
        else:
            residual_demand_max = max_power * 0.25

        dqn = world.learning_role.rl_algorithm
        n_price = dqn.n_price_bins
        n_vol = dqn.n_vol_bins

        # Build gradient-informed regions
        price_regions = _auto_gradient_regions_price(
            0.0, price_cap, competitor_mc, mc, n_price,
        )
        vol_regions = _auto_gradient_regions_volume(
            0.0, max_power, residual_demand_max, n_vol,
        )

        logger.info(
            f"DQN gradient-informed bins for {uid}: "
            f"competitor_mc=${competitor_mc:.1f}, residual_demand_max={residual_demand_max:.0f}MW"
        )
        for label, regions in [("Price", price_regions), ("Volume", vol_regions)]:
            for start, end, n in regions:
                if label == "Price":
                    logger.info(f"  {label}: ${start:.1f}–${end:.1f} → {n} bins")
                else:
                    logger.info(f"  {label}: {start:.0f}–{end:.0f}MW → {n} bins")

        dqn.build_discretizer(
            price_range=(0.0, price_cap),
            vol_range=(0.0, max_power),
            price_regions=price_regions,
            vol_regions=vol_regions,
        )
        break  # single-agent for now


def _pretrain_actor_heuristic(
    world: World,
    network,
    datetime_index: pd.DatetimeIndex,
    bc_config: dict | None = None,
) -> None:
    """Pre-train RL actors via behavioral cloning on a configurable heuristic.

    Generates (observation, target_action) pairs by:
      1. Calling strategy.create_observation() for each hour (same obs the RL agent sees)
      2. Computing target actions from the selected heuristic using raw market data
      3. Training the actor network with MSE loss for bc_steps iterations

    Args:
        world: Initialized World with RL strategies registered.
        network: PyPSA network (for raw forecast/MC/demand data).
        datetime_index: Simulation timestamps.
        bc_config: Dict from YAML with keys: strategy, params, bc_steps, bc_lr.
                   If None, falls back to Cournot with defaults.
    """
    import torch as th
    import torch.nn.functional as F
    from testbed.strategies.heuristics import BidContext, get_heuristic

    if bc_config is None:
        bc_config = {}

    strategy_name = bc_config.get("strategy", "forecast_markup")
    heuristic_params = bc_config.get("params", {})
    bc_steps = bc_config.get("bc_steps", 300)
    bc_lr = bc_config.get("bc_lr", 0.01)

    heuristic_fn = get_heuristic(strategy_name)
    logger.info(
        f"BC warm-start: strategy={strategy_name}, params={heuristic_params}, "
        f"steps={bc_steps}, lr={bc_lr}"
    )

    # ── Precompute network-level data (shared across agents) ─────────
    gen_df = network.generators.copy()
    gen_df.rename(columns={"bus": "node", "p_nom": "max_power"}, inplace=True)

    total_supply = gen_df["max_power"].sum()

    # Sorted competitor MCs for looking up "next more expensive"
    mc_list = sorted(gen_df["marginal_cost"].unique())

    def _next_competitor_mc(own_mc: float) -> float:
        """Find the MC of the cheapest competitor more expensive than own_mc."""
        for mc_val in mc_list:
            if mc_val > own_mc + 0.01:
                return mc_val
        # No more expensive competitor — use price cap
        return 200.0  # fallback

    def _cheaper_supply(own_mc: float, own_name: str) -> float:
        """Total competitor capacity priced below own MC."""
        mask = (gen_df["marginal_cost"] < own_mc - 0.01) & (gen_df.index != own_name)
        return float(gen_df.loc[mask, "max_power"].sum())

    # ── Per-agent behavioral cloning ─────────────────────────────────
    for uid, rl_strategy in world.learning_role.rl_strats.items():
        # Find the unit object
        unit = None
        for uo in world.unit_operators.values():
            if uid in uo.units:
                unit = uo.units[uid]
                break
        if unit is None:
            logger.warning(f"  BC: unit {uid} not found, skipping")
            continue

        mc = float(unit.calculate_marginal_cost(datetime_index[0], 0))
        max_power = float(unit.max_power)
        min_power = float(unit.min_power)
        competitor_mc = _next_competitor_mc(mc)
        cheaper_sup = _cheaper_supply(mc, uid)
        # Per-unit price_cap defines the action space ceiling (matching calculate_bids)
        price_cap = getattr(unit, "price_cap", rl_strategy.max_bid_price)
        # ProfitStrategy maps price to [0, price_cap]; WithholdingStrategy to [MC, price_cap]
        min_bid = 0.0 if isinstance(rl_strategy, ProfitStrategy) else mc
        bid_range = price_cap - min_bid
        if bid_range <= 0:
            continue

        raw_forecast = unit.forecaster.price["EOM"]

        # ── Generate (obs, target_action) pairs ──────────────────────
        obs_list = []
        action_list = []

        for ts in datetime_index:
            end_ts = ts + timedelta(hours=1)

            # Build observation using the strategy's own method
            try:
                obs = rl_strategy.create_observation(
                    unit=unit, market_id="EOM", start=ts, end=end_ts,
                )
                obs_list.append(obs.detach())
            except Exception:
                continue

            # Raw forecast price for this hour
            try:
                forecast_price = float(raw_forecast[ts])
            except (KeyError, IndexError, TypeError):
                forecast_price = mc

            # Total demand this hour
            total_demand = float(
                network.loads_t.p_set.loc[ts].sum()
            ) if ts in network.loads_t.p_set.index else 315.0

            # Build context and call heuristic
            ctx = BidContext(
                mc=mc,
                max_power=max_power,
                min_power=min_power,
                forecast_price=forecast_price,
                total_demand=total_demand,
                total_supply=total_supply,
                own_capacity=max_power,
                competitor_mc=competitor_mc,
                price_cap=price_cap,
                hour=ts.hour if hasattr(ts, "hour") else 0,
                cheaper_supply=cheaper_sup,
            )

            target_price, target_vol_frac = heuristic_fn(ctx, **heuristic_params)

            # Map to actor output space [-1, 1]
            # (strategy rescales via (action+1)/2 to get [0,1] fraction)
            action_price = (target_price - min_bid) / bid_range  # [0,1]
            action_price = max(0.0, min(1.0, action_price))
            action_price = action_price * 2 - 1  # [0,1] -> [-1,1]
            action_vol = max(0.0, min(1.0, target_vol_frac))
            action_vol = action_vol * 2 - 1  # [0,1] -> [-1,1]

            action_list.append([action_price, action_vol])

        if not obs_list:
            logger.warning(f"  BC: no observations collected for {uid}")
            continue

        obs_tensor = th.stack(obs_list)
        target_tensor = th.tensor(
            action_list, dtype=rl_strategy.float_type, device=rl_strategy.device,
        )

        # ── Supervised pre-training ──────────────────────────────────
        # Train on post-softsign output (no clamp) against targets in [0, 1].
        # The actor's softsign activation outputs in [-1, 1]; the MSE loss
        # provides clean gradients to push outputs toward the [0, 1] target
        # range.  No clamping here — it would kill gradients for negative
        # initial outputs.
        actor = rl_strategy.actor
        optimizer = th.optim.Adam(actor.parameters(), lr=bc_lr)

        actor.train()
        for step in range(bc_steps):
            pred = actor(obs_tensor)  # post-softsign, in [-1, 1]
            loss = F.mse_loss(pred, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent softsign saturation overshoot.
            # Without this, the optimizer pushes weights past the target
            # and softsign's vanishing gradient traps the output at ±1.
            th.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            optimizer.step()

        # Copy trained weights to target network
        if hasattr(rl_strategy, "actor_target"):
            rl_strategy.actor_target.load_state_dict(actor.state_dict())

        # ── Log results ──────────────────────────────────────────────
        # Actor outputs in [-1,1]; rescale to [0,1] for interpretable logging
        with th.no_grad():
            raw_pred = actor(obs_tensor)  # [-1, 1]
            rescaled = (raw_pred + 1) / 2  # [0, 1]
            avg_price_frac = rescaled[:, 0].mean().item()
            avg_vol_frac = rescaled[:, 1].mean().item()
            avg_bid = min_bid + avg_price_frac * bid_range

        # Summarize heuristic targets (action_list is in [-1,1])
        target_fracs = [(a[0] + 1) / 2 for a in action_list]
        target_prices = [min_bid + f * bid_range for f in target_fracs]
        avg_target = sum(target_prices) / len(target_prices)

        loss_str = f"loss={loss.item():.6f}, " if bc_steps > 0 else ""
        logger.info(
            f"  BC {uid} ({strategy_name}): {bc_steps} steps, "
            f"{loss_str}"
            f"heuristic avg=${avg_target:.1f}, actor avg=${avg_bid:.1f}, "
            f"vol_frac={avg_vol_frac:.2f}"
        )

        # Per-hour breakdown
        logger.info(f"  BC {uid} per-hour policy (target → learned):")
        for h_idx, (tgt, obs) in enumerate(zip(action_list, obs_list)):
            tgt_p = min_bid + (tgt[0] + 1) / 2 * bid_range
            tgt_v = (tgt[1] + 1) / 2
            act_p = min_bid + rescaled[h_idx, 0].item() * bid_range
            act_v = rescaled[h_idx, 1].item()
            regime = "PEAK" if tgt_p > mc + 0.5 else "OFF "
            logger.info(
                f"    H{h_idx+1:02d} {regime}: "
                f"target=${tgt_p:5.1f} vol={tgt_v:.2f}  →  "
                f"actor=${act_p:5.1f} vol={act_v:.2f}"
            )


def _compute_exploration_centers(
    network,
    bc_config: dict | None = None,
    generator_params: dict | None = None,
    price_forecast: dict | None = None,
    rl_strategy_name: str = "powerplant_energy_withholding",
) -> dict[str, list[tuple[float, float]]]:
    """Compute per-hour BC-based exploration centers for RL generators.

    Uses the same heuristic as BC warm-start (e.g. forecast_markup Cournot
    odds-ratio) to determine where initial-experience exploration should be
    centered in action space.  Returns a dict {gen_id: list of 24 (price_frac,
    vol_frac) tuples}, one per hour.

    Hour-dependent logic (matches BC warm-start exactly):
      - forecast < MC → center at (0.0, 1.0) — bid at MC, full volume
      - forecast >= MC → center at (cournot_frac, 1.0) — markup zone
    """
    from testbed.strategies.heuristics import BidContext, get_heuristic

    if bc_config is None:
        bc_config = {}
    if generator_params is None:
        generator_params = {}
    if price_forecast is None:
        price_forecast = {}
    strategy_name = bc_config.get("strategy", "forecast_markup")
    heuristic_params = bc_config.get("params", {})
    heuristic_fn = get_heuristic(strategy_name)

    gen_df = network.generators
    total_supply = gen_df["p_nom"].sum()
    mc_list = sorted(gen_df["marginal_cost"].unique())

    def _next_competitor_mc(own_mc: float) -> float:
        for mc_val in mc_list:
            if mc_val > own_mc + 0.01:
                return mc_val
        return own_mc * 1.5  # fallback

    def _cheaper_supply(own_mc: float, own_name: str) -> float:
        mask = (gen_df["marginal_cost"] < own_mc - 0.01) & (gen_df.index != own_name)
        return float(gen_df.loc[mask, "p_nom"].sum())

    datetime_index = network.snapshots
    centers = {}

    for gen_name, row in gen_df.iterrows():
        mc = float(row["marginal_cost"])
        max_power = float(row["p_nom"])
        min_power = 0.0
        competitor_mc = _next_competitor_mc(mc)
        cheaper_sup = _cheaper_supply(mc, gen_name)
        gen_yaml = generator_params.get(gen_name, {})
        price_cap = float(gen_yaml.get("price_cap", 100.0))
        bus = str(row["bus"])

        # Get per-bus forecast series (from naive LMP forecast)
        bus_forecast = price_forecast.get(bus, None)

        hourly_centers = []
        for ts in datetime_index:
            total_demand = float(
                network.loads_t.p_set.loc[ts].sum()
            ) if ts in network.loads_t.p_set.index else 315.0

            # Use actual SCED forecast if available, else fallback
            if bus_forecast is not None and ts in bus_forecast.index:
                forecast_price = float(bus_forecast.loc[ts])
            else:
                forecast_price = mc + 5.0

            ctx = BidContext(
                mc=mc, max_power=max_power, min_power=min_power,
                forecast_price=forecast_price, total_demand=total_demand,
                total_supply=total_supply, own_capacity=max_power,
                competitor_mc=competitor_mc, price_cap=price_cap,
                hour=ts.hour if hasattr(ts, "hour") else 0,
                cheaper_supply=cheaper_sup,
            )
            target_price, target_vol = heuristic_fn(ctx, **heuristic_params)

            # Convert to action-space fraction
            # ProfitStrategy maps to [0, price_cap]; WithholdingStrategy to [MC, price_cap]
            price_floor = 0.0 if rl_strategy_name == "powerplant_energy_profit" else mc
            bid_range = price_cap - price_floor
            if bid_range > 0:
                price_frac = (target_price - price_floor) / bid_range
            else:
                price_frac = 0.0
            price_frac = max(0.0, min(1.0, price_frac))
            # Use heuristic's volume target for exploration centering:
            #   Off-peak (forecast < MC): center at full volume (competitive)
            #   Peak (forecast >= MC):    center at residual demand fraction
            #     so exploration samples around the RSI-informed volume
            if price_frac < 0.01:
                vol_frac = 1.0
            else:
                vol_frac = max(0.0, min(1.0, target_vol))

            hourly_centers.append((price_frac, vol_frac))

        centers[gen_name] = hourly_centers

        # Log summary: how many hours at MC vs markup, with volume centers
        mc_hours = sum(1 for pf, _ in hourly_centers if pf < 0.01)
        markup_hours = len(hourly_centers) - mc_hours
        if markup_hours > 0:
            avg_markup_frac = sum(pf for pf, _ in hourly_centers if pf >= 0.01) / markup_hours
            _pf = 0.0 if rl_strategy_name == "powerplant_energy_profit" else mc
            avg_markup_price = _pf + avg_markup_frac * (price_cap - _pf)
            avg_vol_frac = sum(vf for pf, vf in hourly_centers if pf >= 0.01) / markup_hours
        else:
            avg_markup_frac = 0.0
            avg_markup_price = mc
            avg_vol_frac = 1.0
        logger.info(
            f"  Exploration center {gen_name}: {mc_hours}h at MC (vol=1.0), "
            f"{markup_hours}h at avg markup frac={avg_markup_frac:.3f} "
            f"(${avg_markup_price:.1f}), avg vol={avg_vol_frac:.3f}"
        )

    return centers


def _apply_exploration_centers(
    world: World,
    centers: dict[str, list[tuple[float, float]]],
) -> None:
    """Apply pre-computed per-hour exploration centers to RL strategies."""
    for uid, strategy in world.learning_role.rl_strats.items():
        if uid in centers:
            strategy._exploration_center = centers[uid]


def _patch_strategies_for_bid_recording(
    world: World, bid_log: list, episode_ref: list, reward_log: list | None = None,
) -> None:
    """Monkey-patch RL units to record bid prices during simulation.

    Wraps calculate_bids() on each RL UNIT (not strategy) to capture the
    actual bid prices submitted to the market (which are otherwise ephemeral).
    Patching at the unit level is more robust than patching the strategy,
    because ASSUME's internal flow always calls unit.calculate_bids().

    Bid type labeling:
      - 1 bid  → "single" (e.g. WithholdingStrategy: price + volume control)
      - 2 bids → "pmin" (i=0) + "flex" (i=1) (e.g. EnergyLearningStrategy)
      - 3+ bids → "part_0", "part_1", ... (future multi-part strategies)
    """
    rl_uids = set(world.learning_role.rl_strats.keys())
    patched_count = 0

    for op in world.unit_operators.values():
        for uid, unit in op.units.items():
            if uid not in rl_uids:
                continue

            original = unit.calculate_bids

            def _make_patched(_orig, _uid, _bid_log, _episode_ref):
                _call_count = [0]
                def patched(market_config, product_tuples, **kwargs):
                    _call_count[0] += 1
                    bids = _orig(market_config=market_config, product_tuples=product_tuples, **kwargs)
                    n_parts = len(bids)
                    for i, bid in enumerate(bids):
                        if n_parts == 1:
                            bid_type = "single"
                        elif n_parts == 2:
                            bid_type = "pmin" if i == 0 else "flex"
                        else:
                            bid_type = f"part_{i}"
                        _bid_log.append({
                            "episode": _episode_ref[0],
                            "timestamp": bid["start_time"],
                            "hour": bid["start_time"].hour if hasattr(bid["start_time"], "hour") else 0,
                            "unit_id": _uid,
                            "node": bid.get("node", ""),
                            "bid_type": bid_type,
                            "bid_price": float(bid["price"]),
                            "bid_volume": float(bid["volume"]),
                        })
                    return bids
                return patched

            unit.calculate_bids = _make_patched(original, uid, bid_log, episode_ref)
            patched_count += 1

    logger.debug(f"Patched {patched_count} RL units for bid recording (rl_uids={rl_uids})")


def _scalar_mc(unit, ts) -> float:
    """Extract a scalar marginal cost from a unit, handling FastSeries/Series/float."""
    mc = getattr(unit, "marginal_cost", 0.0)
    if isinstance(mc, (int, float)):
        return float(mc)
    # FastSeries or pd.Series — index by timestamp
    try:
        return float(mc[ts])
    except (KeyError, IndexError, TypeError):
        pass
    # Fallback: first value
    try:
        return float(mc.iloc[0])
    except (AttributeError, IndexError):
        return 0.0


def _collect_episode_results(world: World, episode: int, collector: list) -> None:
    """Extract accepted prices (LMPs) and dispatch from unit outputs after world.run()."""
    for uo in world.unit_operators.values():
        for uid, unit in uo.units.items():
            outputs = getattr(unit, "outputs", {})
            acc_price = outputs.get("energy_accepted_price")
            energy = outputs.get("energy")
            if acc_price is None:
                continue
            for ts in unit.index:
                try:
                    price_val = float(acc_price[ts])
                    energy_val = float(energy[ts]) if energy is not None else 0.0
                except (KeyError, IndexError):
                    continue
                collector.append({
                    "episode": episode,
                    "timestamp": ts,
                    "hour": ts.hour if hasattr(ts, "hour") else 0,
                    "unit_id": uid,
                    "node": getattr(unit, "node", ""),
                    "marginal_cost": _scalar_mc(unit, ts),
                    "accepted_price": price_val,
                    "dispatch": energy_val,
                    "is_rl": uid in RL_GENERATORS,
                })


def run_learning_testbed(
    market_mechanism: str = "egret_sced",
    training_episodes: int = 20,
    episodes_collecting_initial_experience: int = 3,
    validation_episodes_interval: int = 5,
    exploration_reset_interval: int = 10,
    rl_agents: list | None = None,
    rl_strategy: str | None = None,
    algorithm: str = "matd3",
    db_uri: str = "sqlite:///./testbed_results/learning.db",
    export_csv: str = "",
    periods: int = 24,
    start_hour: int = 0,
    min_bid_price: float = 0.0,
    max_bid_price: float = 100.0,
    save_episode_interval: int = 1,
    results_dir: str = "testbed_results/rl_episodes",
    bc_config: dict | None = None,
    extra_learning_config: dict | None = None,
    generator_params: dict | None = None,
    line_overrides: dict | None = None,
    forecast_mode: str = "exclude_rl",
    learning_config_overrides: dict | None = None,
    profile_cfg: dict | None = None,
) -> dict:
    """Run RL training loop for SCED testbed.

    Args:
        market_mechanism: Clearing algorithm.
        training_episodes: Total training episodes.
        episodes_collecting_initial_experience: Random exploration episodes.
        validation_episodes_interval: Evaluation interval.
        exploration_reset_interval: Force MC exploration every N episodes (0 = off).
        rl_agents: List of generator names to use RL. None = use module default.
        db_uri: Database URI for reward tracking.
        export_csv: CSV export path (empty = none during training).
        periods: Number of hourly snapshots.
        start_hour: Hour of day to start (0-23). Use 7 for peak-only training.
        max_bid_price: Max bid price for RL action scaling.
        save_episode_interval: Save CSV results every N episodes (0 = never).
        results_dir: Directory for per-episode CSV results.
        bc_config: Behavioral cloning warm-start config from YAML.
                   Keys: strategy, params, bc_steps, bc_lr.
                   None = Cournot with defaults.
        extra_learning_config: Additional learning config overrides (e.g. from
                   benchmark runner hyperparameter tuning).
        forecast_mode: How to compute the price forecast for RL observations.
                   "exclude_rl" = counterfactual SCED without RL agents (default).
                   "include_all" = SCED with all gens at MC (no exclusion).

    Returns:
        Dict with training results.
    """
    global RL_GENERATORS, RL_STRATEGY
    if rl_agents is not None:
        RL_GENERATORS = set(rl_agents)
        logger.info(f"RL agents overridden: {RL_GENERATORS}")
    if rl_strategy is not None:
        RL_STRATEGY = rl_strategy
        logger.info(f"RL strategy overridden: {RL_STRATEGY}")
    from assume.reinforcement_learning.buffer import ReplayBuffer, RolloutBuffer

    # Map algorithm → actor architecture
    _ALGO_ACTOR_MAP = {
        "matd3": "mlp",
        "sac": "stochastic",
        "ppo": "dist",
        "dqn": "mlp",  # DQN builds its own QNetwork; "mlp" is a placeholder
    }
    actor_arch = _ALGO_ACTOR_MAP.get(algorithm, "mlp")

    # Build network once (reused across episodes)
    start_dt = f"2025-01-01 {start_hour:02d}:00"
    network = build_ieee9_network(start=start_dt, periods=periods, profile_cfg=profile_cfg)
    assign_zones(network)

    # Apply YAML overrides to the PyPSA network
    if generator_params:
        _apply_generator_overrides(network, generator_params)
    if line_overrides:
        _apply_line_overrides(network, line_overrides)

    # Defaults for learning config — all overridable via scenarios.yaml
    # `training.learning_config` or via extra_learning_config (benchmark runner).
    learning_config = {
        "learning_mode": True,
        "algorithm": algorithm,
        "actor_architecture": actor_arch,
        "learning_rate": 0.0003,
        "training_episodes": training_episodes,
        "episodes_collecting_initial_experience": episodes_collecting_initial_experience,
        "validation_episodes_interval": validation_episodes_interval,
        # train_freq must be < simulation horizon so store_to_buffer_and_update
        # fires at least once.  With 24 periods (H00-H23), end=H23.  The task
        # starts at start + train_freq, so train_freq=24h → start at H24 (too late!).
        # Use periods//2 so it fires at least once mid-simulation.
        "train_freq": f"{max(periods // 2, 1)}h",
        "gradient_steps": 10,
        "batch_size": 64,
        "gamma": 0.0,  # each hour is independent — no future reward coupling
        "device": "cpu",
        "noise_sigma": 0.1,
        "noise_scale": 1,
        "noise_dt": 1,  # no per-step decay; schedule handles decay over episodes
        "action_noise_schedule": "linear",
        "max_bid_price": max_bid_price,
        "min_bid_price": min_bid_price,
        "trained_policies_save_path": f"testbed_results/learned_strategies/{algorithm}",
        "trained_policies_load_path": "",
        "replay_buffer_size": 50000,
        "early_stopping_threshold": 0.05,
        "exploration_noise_std": 0.1,
    }

    # Apply YAML learning_config overrides (scenarios.yaml training.learning_config)
    if learning_config_overrides:
        learning_config.update(learning_config_overrides)
        logger.info(f"YAML learning_config overrides applied: {list(learning_config_overrides.keys())}")

    # Apply extra overrides (e.g. from benchmark runner hyperparameter tuning)
    if extra_learning_config:
        learning_config.update(extra_learning_config)

    logger.info(
        f"Starting RL training: {training_episodes} episodes, "
        f"{len(RL_GENERATORS)} learning agents, mechanism={market_mechanism}"
    )

    # ── Ensure results directory exists ─────────────────────────────
    if save_episode_interval > 0:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    def _episode_csv_path(ep: int) -> str:
        if save_episode_interval > 0 and ep % save_episode_interval == 0:
            return f"{results_dir}/episode_{ep:03d}"
        return ""

    # ── Pre-compute LMP forecasts (once, reused across all episodes) ─────
    datetime_index = pd.DatetimeIndex(network.snapshots)

    # 1. Observation forecast: what the agent sees (controlled by forecast_mode)
    if forecast_mode == "include_all":
        naive_lmp_forecast = _compute_naive_lmp_forecast(
            network, datetime_index, exclude_agents=set(),
        )
        logger.info("Observation forecast: include_all (all gens at MC)")
    else:
        naive_lmp_forecast = _compute_naive_lmp_forecast(
            network, datetime_index, exclude_agents=RL_GENERATORS,
        )
        logger.info(f"Observation forecast: exclude_rl (excluded: {RL_GENERATORS})")

    # 2. Per-agent counterfactual: each RL agent gets a SCED with only
    #    its OWN capacity removed.  Used exclusively for the sigmoid reward
    #    reference — tells the reward function where the dispatch cliff is.
    #    The agent never observes this; it only shapes the gradient.
    per_agent_counterfactual: dict[str, dict[str, pd.Series]] = {}
    for agent_id in RL_GENERATORS:
        cf = _compute_naive_lmp_forecast(
            network, datetime_index, exclude_agents={agent_id},
        )
        per_agent_counterfactual[agent_id] = cf
        agent_bus = str(network.generators.at[agent_id, "bus"])
        cf_series = cf.get(agent_bus, list(cf.values())[0])
        logger.info(
            f"Counterfactual for {agent_id} (bus {agent_bus}): "
            f"min=${cf_series.min():.1f}, max=${cf_series.max():.1f}"
        )

    # Save per-bus forecasts so the notebook can use the exact same series
    if save_episode_interval > 0:
        _fcast_rows = []
        for _bus, _s in naive_lmp_forecast.items():
            _df = _s.rename("lmp").reset_index()
            _df.columns = ["timestamp", "lmp"]
            _df["bus"] = _bus
            _df["forecast_mode"] = forecast_mode
            _fcast_rows.append(_df)
        pd.concat(_fcast_rows).to_csv(f"{results_dir}/price_forecast.csv", index=False)
        logger.info(f"Saved per-bus price forecast ({forecast_mode}) to {results_dir}/price_forecast.csv")

    # ── Episode 1: initialize ────────────────────────────────────────
    world = setup_learning_world(
        network,
        episode=1,
        learning_config=learning_config,
        market_mechanism=market_mechanism,
        db_uri=db_uri,
        export_csv=_episode_csv_path(1),
        price_forecast=naive_lmp_forecast,
        generator_params=generator_params,
        per_agent_counterfactual=per_agent_counterfactual,
    )

    # ── DQN: build adaptive discretizer with ridge-aware bins ──────────
    if algorithm == "dqn":
        _build_dqn_discretizer(world, network)

    # Initialize policies and pre-train via behavioral cloning
    world.learning_role.rl_algorithm.initialize_policy()
    _pretrain_actor_heuristic(world, network, datetime_index, bc_config=bc_config)

    # Save frozen BC actor for regularization during RL training.
    # This preserves the per-observation BC mapping (off-peak vs peak)
    # so the actor loss can penalize deviation from the BC clusters.
    import copy
    for strategy in world.learning_role.rl_strats.values():
        strategy.bc_actor = copy.deepcopy(strategy.actor)
        strategy.bc_actor.eval()
        for p in strategy.bc_actor.parameters():
            p.requires_grad_(False)

    # ── Compute BC-based exploration centers for initial experience mode ──
    # Uses the same Cournot formula as BC warm-start to center exploration
    # around the heuristic policy rather than the arbitrary action midpoint.
    # Stored in a dict so they can be re-applied when the world is re-created.
    exploration_centers = _compute_exploration_centers(
        network, bc_config, generator_params, price_forecast=naive_lmp_forecast,
        rl_strategy_name=RL_STRATEGY,
    )
    _apply_exploration_centers(world, exploration_centers)

    # Create buffer: RolloutBuffer for PPO (on-policy), ReplayBuffer for others
    _buf_kwargs = dict(
        obs_dim=world.learning_role.rl_algorithm.obs_dim,
        act_dim=world.learning_role.rl_algorithm.act_dim,
        n_rl_units=len(world.learning_role.rl_strats),
        device=world.learning_role.device,
        float_type=world.learning_role.float_type,
    )
    if algorithm == "ppo":
        buffer = RolloutBuffer(**_buf_kwargs)
    elif learning_config.get("per_enabled", False):
        from assume.reinforcement_learning.buffer import PrioritizedReplayBuffer

        buffer = PrioritizedReplayBuffer(
            buffer_size=learning_config["replay_buffer_size"],
            alpha=learning_config.get("per_alpha", 0.6),
            beta_start=learning_config.get("per_beta_start", 0.4),
            beta_anneal_steps=learning_config.get("per_beta_anneal_steps", 100_000),
            **_buf_kwargs,
        )
    else:
        buffer = ReplayBuffer(
            buffer_size=learning_config["replay_buffer_size"],
            **_buf_kwargs,
        )

    # Extract BC-trained actors/critics so load_inter_episodic_data preserves
    # the warm-started weights instead of re-initializing from scratch.
    bc_actors_and_critics = world.learning_role.rl_algorithm.extract_policy()

    inter_episodic_data = {
        "buffer": buffer,
        "actors_and_critics": bc_actors_and_critics,
        "max_eval": defaultdict(lambda: -1e9),
        "all_eval": defaultdict(list),
        "avg_all_eval": [],
        "episodes_done": 0,
        "eval_episodes_done": 0,
    }

    world.learning_role.load_inter_episodic_data(inter_episodic_data)

    n_rl_units = len(world.learning_role.rl_strats)
    logger.info(f"Registered {n_rl_units} RL strategies: {list(world.learning_role.rl_strats.keys())}")

    # ── Training loop ────────────────────────────────────────────────
    eval_episode = 1
    best_train_reward = -1e9
    mc_baseline_reward = None  # reward from the first MC episode (baseline)
    best_reward = -1e9
    rewards_history = []
    all_bid_log = []      # Raw bid prices from RL strategies
    all_results = []      # Accepted prices (LMPs) and dispatch
    episode_ref = [1]     # Mutable ref for monkey-patch closure
    consecutive_worse_than_mc = 0  # count of episodes worse than MC baseline

    # Patch bid recording (action clamping removed — BC warm-start provides
    # a reasonable starting policy, so the agent doesn't need artificial bounds)
    _patch_strategies_for_bid_recording(world, all_bid_log, episode_ref)

    for episode in tqdm(
        range(1, training_episodes + 1),
        desc="Training Episodes",
    ):
        episode_ref[0] = episode

        if episode != 1:
            world = setup_learning_world(
                network,
                episode=episode,
                learning_config=learning_config,
                market_mechanism=market_mechanism,
                db_uri=db_uri,
                export_csv=_episode_csv_path(episode),
                price_forecast=naive_lmp_forecast,
                generator_params=generator_params,
            )
            world.learning_role.load_inter_episodic_data(inter_episodic_data)
            _apply_exploration_centers(world, exploration_centers)
            # Re-patch strategies on the new world
            _patch_strategies_for_bid_recording(world, all_bid_log, episode_ref)

        # NOTE: reward clipping removed.  PopArt handles adaptive normalization
        # of the critic targets, making hard clipping redundant.  With clip=1.0,
        # both $242/h and $974/h profits mapped to +1.0 — the critic could not
        # distinguish the optimal ($49.99) region from the BC warmstart ($38-43).
        # _clip_rewards(world, clip_range=1.0)  # DISABLED — PopArt replaces this

        # ── DQN epsilon decay ─────────────────────────────────────────
        if algorithm == "dqn" and episode > episodes_collecting_initial_experience:
            world.learning_role.rl_algorithm.update_epsilon(
                episode - episodes_collecting_initial_experience
            )

        # ── Decide whether to force exploration this episode ─────────
        # Four triggers:
        # 1. Periodic MC exploration (safety net)
        # 2. Adaptive: if last N episodes were worse than the MC baseline
        # 3. First post-exploration episode establishes the MC baseline
        # 4. BC re-exploration: periodically revisit BC-centered actions
        #    so the replay buffer retains data near the heuristic policy
        force_mc = False
        force_bc = False
        bc_reexplore = learning_config.get("bc_reexplore_interval", 0)
        if episode <= episodes_collecting_initial_experience:
            pass  # initial experience is already BC-centered
        elif (
            bc_reexplore > 0
            and episode % bc_reexplore == 0
        ):
            force_bc = True
            logger.info(f"Episode {episode}: BC re-exploration (centers from warm-start)")
        elif (
            exploration_reset_interval > 0
            and episode % exploration_reset_interval == 0
        ):
            force_mc = True
            logger.info(f"Episode {episode}: scheduled MC exploration")
        elif (
            mc_baseline_reward is not None
            and consecutive_worse_than_mc >= 3
        ):
            force_mc = True
            consecutive_worse_than_mc = 0
            logger.info(
                f"Episode {episode}: adaptive MC reset "
                f"(3 consecutive episodes worse than MC baseline {mc_baseline_reward:.4f})"
            )

        if force_mc:
            _force_mc_exploration(world, flush_buffer=False)
        elif force_bc:
            # Re-enter initial-experience mode for one episode so
            # get_actions uses the BC-derived exploration centers
            # (per-hour price markup + RSI volume) instead of actor output.
            for strategy in world.learning_role.rl_strats.values():
                strategy.collect_initial_experience_mode = True

        # ── Parameter-space noise: perturb actor weights for coherent exploration ──
        # Each episode gets a different weight perturbation, producing a coherent
        # alternative strategy (e.g., consistently higher prices or lower volume)
        # rather than random per-step action jitter.
        # Skip during initial experience (pure noise) and evaluation.
        param_noise_sigma = learning_config.get("param_noise_sigma", 0.0)
        if (
            param_noise_sigma > 0
            and episode > episodes_collecting_initial_experience
            and not force_mc
        ):
            _perturb_actor_weights(world, sigma=param_noise_sigma)

        # ── Critic warmup noise reduction ──────────────────────────
        # During critic warmup, reduce action noise so the BC policy
        # generates clean, consistent data for the critic to learn from.
        # After warmup, restore full noise for exploration.
        critic_warmup_steps = learning_config.get("critic_warmup_steps", 0)
        warmup_noise_scale = learning_config.get("warmup_noise_scale", 0.1)
        if critic_warmup_steps > 0 and not force_mc:
            grad_steps = learning_config.get("gradient_steps", 15)
            warmup_episodes = episodes_collecting_initial_experience + \
                critic_warmup_steps // max(grad_steps, 1) + 1
            if episode <= warmup_episodes:
                # Low noise: BC policy with small perturbation
                for strategy in world.learning_role.rl_strats.values():
                    strategy.action_noise.update_noise_decay(warmup_noise_scale)
                if episode == episodes_collecting_initial_experience + 1:
                    logger.info(
                        f"Critic warmup: reducing noise to {warmup_noise_scale}x "
                        f"for episodes {episodes_collecting_initial_experience+1}-{warmup_episodes}"
                    )
            elif episode == warmup_episodes + 1:
                # Restore full noise
                for strategy in world.learning_role.rl_strats.values():
                    strategy.action_noise.update_noise_decay(1.0)
                logger.info(f"Critic warmup complete: restoring full noise")

        train_reward_log = _track_rewards(world)
        world.run()

        # Log training reward
        train_reward = float(np.mean(train_reward_log)) if train_reward_log else 0.0

        # Track MC baseline from the first exploration episode's reward
        if episode == episodes_collecting_initial_experience and mc_baseline_reward is None:
            mc_baseline_reward = train_reward
            logger.info(f"MC baseline reward established: {mc_baseline_reward:.4f}")

        # Update adaptive trigger: count consecutive episodes worse than MC
        if mc_baseline_reward is not None and episode > episodes_collecting_initial_experience:
            if train_reward < mc_baseline_reward:
                consecutive_worse_than_mc += 1
            else:
                consecutive_worse_than_mc = 0
                if train_reward > best_train_reward:
                    best_train_reward = train_reward

        if episode % 5 == 0 or episode == 1:
            mc_tag = " [MC]" if force_mc else ""
            logger.info(f"Episode {episode}{mc_tag} | train_reward={train_reward:.4f}")

        # ── Capture accepted prices and dispatch ─────────────────────
        _collect_episode_results(world, episode, all_results)

        # Store inter-episodic data
        inter_episodic_data = world.learning_role.get_inter_episodic_data()
        inter_episodic_data["episodes_done"] = episode

        # ── Save actor params periodically ─────────────────────────
        save_path = learning_config["trained_policies_save_path"]
        policy_save_interval = max(1, training_episodes // 5)  # ~5 checkpoints
        if episode % policy_save_interval == 0 or episode == training_episodes:
            world.learning_role.rl_algorithm.save_params(
                directory=f"{save_path}/episode_{episode:03d}"
            )

        # ── Evaluation ───────────────────────────────────────────────
        if (
            episode % validation_episodes_interval == 0
            and episode >= episodes_collecting_initial_experience + validation_episodes_interval
        ):
            eval_config = learning_config.copy()
            eval_config["learning_mode"] = True
            eval_config["evaluation_mode"] = True

            eval_world = setup_learning_world(
                network,
                episode=episode,
                eval_episode=eval_episode,
                learning_config=eval_config,
                market_mechanism=market_mechanism,
                db_uri=db_uri,
                export_csv="",
                price_forecast=naive_lmp_forecast,
                generator_params=generator_params,
            )
            eval_world.learning_role.load_inter_episodic_data(inter_episodic_data)
            eval_reward_log = _track_rewards(eval_world)
            eval_world.run()

            # Compute average reward from tracked cache entries.
            # The DB path (get_sum_reward) fails because store_to_buffer_and_update
            # often doesn't fire before the eval world exits.
            avg_reward = float(np.mean(eval_reward_log)) if eval_reward_log else 0.0

            rewards_history.append(avg_reward)
            if avg_reward > best_reward:
                best_reward = avg_reward

            logger.info(
                f"Episode {episode} | Eval {eval_episode} | "
                f"avg_reward={avg_reward:.4f} | best={best_reward:.4f}"
            )

            inter_episodic_data["eval_episodes_done"] = eval_episode
            eval_episode += 1

        # Save latest policies
        world.learning_role.rl_algorithm.save_params(
            directory=f"{save_path}/last"
        )

    # ── Save bid and result history CSVs ─────────────────────────────
    if all_bid_log:
        bids_df = pd.DataFrame(all_bid_log)
        bids_path = f"{results_dir}/bid_history.csv"
        bids_df.to_csv(bids_path, index=False)
        logger.info(f"Saved bid history ({len(bids_df)} rows) to {bids_path}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = f"{results_dir}/results_history.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results history ({len(results_df)} rows) to {results_path}")

    # ── Noise-free policy evaluation ─────────────────────────────────
    # Run one episode with the final policy in evaluation mode (no noise,
    # no buffer updates). Save bids and results to separate CSVs so the
    # notebook can show exactly what the trained network would bid.
    logger.info("Running noise-free policy evaluation...")
    policy_bid_log = []
    policy_results = []
    policy_episode_ref = [training_episodes + 1]

    try:
        policy_eval_config = learning_config.copy()
        policy_eval_config["learning_mode"] = True
        policy_eval_config["evaluation_mode"] = True

        policy_world = setup_learning_world(
            network,
            episode=training_episodes + 1,
            eval_episode=training_episodes + 1,
            learning_config=policy_eval_config,
            market_mechanism=market_mechanism,
            db_uri=db_uri,
            export_csv="",
            price_forecast=naive_lmp_forecast,
            generator_params=generator_params,
        )
        policy_world.learning_role.load_inter_episodic_data(inter_episodic_data)
        _patch_strategies_for_bid_recording(
            policy_world, policy_bid_log, policy_episode_ref
        )
        policy_world.run()
        _collect_episode_results(
            policy_world, training_episodes + 1, policy_results
        )

        if policy_bid_log:
            policy_bids_df = pd.DataFrame(policy_bid_log)
            policy_bids_path = f"{results_dir}/final_policy_bids.csv"
            policy_bids_df.to_csv(policy_bids_path, index=False)
            logger.info(
                f"Saved final policy bids ({len(policy_bids_df)} rows) "
                f"to {policy_bids_path}"
            )

        if policy_results:
            policy_results_df = pd.DataFrame(policy_results)
            policy_results_path = f"{results_dir}/final_policy_results.csv"
            policy_results_df.to_csv(policy_results_path, index=False)
            logger.info(
                f"Saved final policy results ({len(policy_results_df)} rows) "
                f"to {policy_results_path}"
            )
    except Exception as e:
        logger.warning(f"Noise-free policy evaluation failed: {e}")

    # ── Final evaluation with CSV export ─────────────────────────────
    logger.info("Running final evaluation with CSV export...")
    final_config = learning_config.copy()
    final_config["learning_mode"] = False
    final_config["evaluation_mode"] = False
    final_config["trained_policies_load_path"] = f"{learning_config['trained_policies_save_path']}/last"

    try:
        final_world = setup_learning_world(
            network,
            episode=training_episodes,
            learning_config=final_config,
            market_mechanism=market_mechanism,
            db_uri=db_uri,
            export_csv=export_csv or "testbed_results/rl_final",
            price_forecast=naive_lmp_forecast,
            generator_params=generator_params,
        )
        final_world.run()
        logger.info("Final evaluation complete.")
    except Exception as e:
        logger.warning(f"Final evaluation failed (expected if no policies saved yet): {e}")

    return {
        "episodes": training_episodes,
        "best_reward": best_reward,
        "best_train_reward": best_train_reward,
        "rewards_history": rewards_history,
        "n_rl_units": n_rl_units,
        "rl_units": list(RL_GENERATORS),
        "results_dir": results_dir,
        "bid_history_path": f"{results_dir}/bid_history.csv",
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import yaml

    parser = argparse.ArgumentParser(description="RL training for SCED testbed")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scenario name from scenarios.yaml (loads training config)")
    parser.add_argument("--episodes", type=int, default=None, help="Training episodes (overrides YAML)")
    parser.add_argument("--mechanism", type=str, default="egret_sced", help="Market mechanism")
    parser.add_argument("--periods", type=int, default=None, help="Hourly periods (overrides YAML)")
    parser.add_argument("--start-hour", type=int, default=None, help="Start hour of day (0-23, overrides YAML)")
    parser.add_argument("--max-bid", type=float, default=None, help="Max bid price (overrides YAML)")
    parser.add_argument("--algorithm", type=str, default=None,
                        choices=["matd3", "sac", "ppo"],
                        help="RL algorithm (overrides YAML, default: matd3)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory (default: testbed_results/rl_{algorithm})")
    parser.add_argument("--forecast-mode", type=str, default=None,
                        choices=["exclude_rl", "include_all"],
                        help="Price forecast mode (overrides YAML)")
    parser.add_argument("--db", type=str, default="sqlite:///./testbed_results/learning.db", help="Database URI")
    args = parser.parse_args()

    # Load defaults from scenarios.yaml if --scenario is given
    scenario_cfg = {}
    if args.scenario:
        yaml_path = Path(__file__).parent / "scenarios.yaml"
        with open(yaml_path) as f:
            all_scenarios = yaml.safe_load(f)
        if args.scenario not in all_scenarios:
            print(f"Error: scenario '{args.scenario}' not found in {yaml_path}")
            print(f"Available: {list(all_scenarios.keys())}")
            sys.exit(1)
        scenario_cfg = all_scenarios[args.scenario]
        print(f"Loading scenario: {scenario_cfg.get('name', args.scenario)}")

    training = scenario_cfg.get("training", {})

    # CLI args override YAML; YAML overrides hardcoded defaults
    episodes = args.episodes or training.get("episodes", 20)
    periods = args.periods or scenario_cfg.get("periods", 24)
    start_hour = args.start_hour if args.start_hour is not None else scenario_cfg.get("start_hour", 0)
    min_bid = training.get("min_bid_price", 0.0)
    max_bid = args.max_bid or training.get("max_bid_price", 100.0)
    initial_exp = training.get("initial_experience_episodes", 3)
    val_interval = training.get("validation_interval", 5)
    explore_reset = training.get("exploration_reset_interval", 10)
    rl_agents_list = training.get("rl_agents", None)  # None = use default
    rl_strategy_name = training.get("rl_strategy", None)  # None = use default
    bc_warmstart = training.get("bc_warmstart", None)  # None = Cournot defaults
    gen_params = scenario_cfg.get("generators", {})
    line_ovr = scenario_cfg.get("line_overrides", {})
    profile_ovr = scenario_cfg.get("profile", None)
    algo = args.algorithm if hasattr(args, "algorithm") and args.algorithm else training.get("algorithm", "matd3")
    forecast_mode = args.forecast_mode or training.get("forecast_mode", "exclude_rl")
    lc_overrides = training.get("learning_config", None)
    results_dir = (
        args.results_dir
        or training.get("results_dir")
        or f"testbed_results/rl_{algo}"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_learning_testbed(
        market_mechanism=args.mechanism,
        training_episodes=episodes,
        episodes_collecting_initial_experience=initial_exp,
        validation_episodes_interval=val_interval,
        exploration_reset_interval=explore_reset,
        rl_agents=rl_agents_list,
        rl_strategy=rl_strategy_name,
        algorithm=algo,
        periods=periods,
        start_hour=start_hour,
        min_bid_price=min_bid,
        max_bid_price=max_bid,
        db_uri=args.db,
        bc_config=bc_warmstart,
        results_dir=results_dir,
        generator_params=gen_params,
        line_overrides=line_ovr,
        forecast_mode=forecast_mode,
        learning_config_overrides=lc_overrides,
        profile_cfg=profile_ovr,
    )

    print(f"\n{'='*60}")
    print(f"Training complete: {results['episodes']} episodes")
    print(f"RL agents: {results['rl_units']}")
    print(f"Best avg reward: {results['best_reward']:.4f}")
    if results['rewards_history']:
        print(f"Reward history: {[f'{r:.3f}' for r in results['rewards_history']]}")
    print(f"{'='*60}")
