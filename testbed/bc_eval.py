"""Evaluate BC-trained actor per hour to verify hour differentiation."""
import sys
import logging
import torch as th
import pandas as pd
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

import yaml
from testbed.learning_runner import (
    setup_learning_world,
    build_ieee9_network, assign_zones,
    _apply_generator_overrides, _pretrain_actor_heuristic,
    _compute_naive_lmp_forecast,
)

with open("testbed/scenarios.yaml") as f:
    cfg = yaml.safe_load(f)["base"]

training = cfg["training"]
gen_params = cfg.get("generators", {})

# Build network
network = build_ieee9_network(start="2025-01-01 00:00", periods=24, profile_cfg=cfg.get("profile"))
assign_zones(network)
if gen_params:
    _apply_generator_overrides(network, gen_params)

datetime_index = pd.DatetimeIndex(network.snapshots)

# Learning config
lc = {
    "learning_mode": True, "algorithm": "matd3",
    "actor_architecture": "mlp", "learning_rate": 0.0003,
    "training_episodes": 2, "episodes_collecting_initial_experience": 1,
    "validation_episodes_interval": 5, "train_freq": "12h",
    "min_bid_price": 0.0, "max_bid_price": 100.0,
    "trained_policies_save_path": "", "trained_policies_load_path": "",
    "replay_buffer_size": 50000, "early_stopping_threshold": 0.05,
    "exploration_noise_std": 0.1,
}
if training.get("learning_config"):
    lc.update(training["learning_config"])

# Counterfactual LMP forecast
rl_agents_set = set(training.get("rl_agents", []))
forecast_mode = training.get("forecast_mode", "exclude_rl")
naive_lmp_forecast = _compute_naive_lmp_forecast(
    network, datetime_index,
    exclude_agents=rl_agents_set if forecast_mode == "exclude_rl" else set(),
)

# Build world
world = setup_learning_world(
    network, learning_config=lc, market_mechanism="egret_sced",
    price_forecast=naive_lmp_forecast, generator_params=gen_params,
)
world.learning_role.rl_algorithm.initialize_policy()

# BC pre-train
bc_cfg = training.get("bc_warmstart", {}).copy()
bc_cfg["bc_steps"] = 200
_pretrain_actor_heuristic(world, network, datetime_index, bc_config=bc_cfg)

# Evaluate per hour
out = open("bc_per_hour.log", "w")
for uid, strat in world.learning_role.rl_strats.items():
    actor = strat.actor
    actor.eval()
    unit = None
    for uo in world.unit_operators.values():
        if uid in uo.units:
            unit = uo.units[uid]
            break
    mc = float(unit.calculate_marginal_cost(datetime_index[0], 0))
    price_cap = getattr(unit, "price_cap", strat.max_bid_price)
    bid_range = price_cap - mc
    max_p = float(unit.max_power)

    out.write(f"{uid} MC=${mc} price_cap=${price_cap}\n")
    out.write(f"{'Hour':>4}  {'PriceAct':>8}  {'VolAct':>6}  {'Bid$':>6}  {'VolMW':>6}\n")
    for i, ts in enumerate(datetime_index):
        end_ts = ts + timedelta(hours=1)
        try:
            obs = strat.create_observation(unit=unit, market_id="EOM", start=ts, end=end_ts)
            with th.no_grad():
                action = th.clamp(actor(obs.unsqueeze(0)), 0, 1).squeeze()
            bid_price = mc + action[0].item() * bid_range
            vol_mw = action[1].item() * max_p
            out.write(f"{i:4d}  {action[0].item():8.3f}  {action[1].item():6.3f}  {bid_price:6.1f}  {vol_mw:6.0f}\n")
        except Exception as e:
            out.write(f"{i:4d}  ERROR: {e}\n")
out.close()
print("Done — see bc_per_hour.log")
