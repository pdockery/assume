"""Integration test: SCED clearing via the adapter on IEEE 9-bus.

Builds synthetic ASSUME-style orders from the IEEE 9-bus network,
runs them through clear_sced(), and validates the results.

Usage:
    python testbed/test_sced_integration.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure testbed is importable
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
testbed_dir = str(Path(__file__).resolve().parent)
if testbed_dir not in sys.path:
    sys.path.insert(0, testbed_dir)

print("=== SCED Integration Test: IEEE 9-Bus ===\n", flush=True)

from clearing.sced_clearing import clear_sced
from ieee9_network import build_ieee9_network

# ── Build network and extract grid_data ────────────────────────────
network = build_ieee9_network(periods=24)

grid_data = {
    "buses": network.buses[["v_nom"]].copy(),
    "lines": network.lines[["bus0", "bus1", "s_nom", "x_pu_eff", "r_pu_eff"]].copy(),
}

print(f"Grid: {len(grid_data['buses'])} buses, {len(grid_data['lines'])} lines", flush=True)
print(f"Lines columns: {list(grid_data['lines'].columns)}", flush=True)

# ── Build synthetic ASSUME-style orders ────────────────────────────
# Mirror what ASSUME's naive bidding strategy produces
start_time = network.snapshots[0].to_pydatetime()
market_products = []
for i in range(24):
    t_start = start_time + timedelta(hours=i)
    t_end = t_start + timedelta(hours=1)
    market_products.append((t_start, t_end, None))

orderbook = []

# Generator orders (supply, volume > 0)
gen_data = {
    "gen_slack_0": {"bus": "1", "p_max": 250.0, "price": 20.0},
    "gen_0": {"bus": "2", "p_max": 300.0, "price": 35.0},
    "gen_1": {"bus": "3", "p_max": 270.0, "price": 50.0},
}

for gen_name, gd in gen_data.items():
    for i, (t_start, t_end, _) in enumerate(market_products):
        orderbook.append({
            "bid_id": f"{gen_name}_{i}",
            "start_time": t_start,
            "end_time": t_end,
            "volume": gd["p_max"],
            "accepted_volume": 0.0,
            "price": gd["price"],
            "accepted_price": 0.0,
            "agent_addr": f"agent_{gen_name}",
            "unit_id": gen_name,
            "node": gd["bus"],
            "only_hours": None,
        })

# Load orders (demand, volume < 0)
# Use the same diurnal profile as ieee9_network.py
hours = np.arange(24) % 24
profile = 0.6 + 0.4 * np.sin(np.pi * (hours - 4) / 16)
profile = np.clip(profile, 0.5, 1.0)

load_data = {
    "load_0": {"bus": "5", "p_base": 90.0},
    "load_1": {"bus": "7", "p_base": 100.0},
    "load_2": {"bus": "9", "p_base": 125.0},
}

for load_name, ld in load_data.items():
    for i, (t_start, t_end, _) in enumerate(market_products):
        orderbook.append({
            "bid_id": f"{load_name}_{i}",
            "start_time": t_start,
            "end_time": t_end,
            "volume": -ld["p_base"] * profile[i],
            "accepted_volume": 0.0,
            "price": 1000.0,  # demand bids at high price
            "accepted_price": 0.0,
            "agent_addr": f"agent_{load_name}",
            "unit_id": load_name,
            "node": ld["bus"],
            "only_hours": None,
        })

print(f"Orders: {len(orderbook)} ({sum(1 for o in orderbook if o['volume'] > 0)} supply, "
      f"{sum(1 for o in orderbook if o['volume'] < 0)} demand)", flush=True)

# ── Run SCED clearing ─────────────────────────────────────────────
print("\nRunning clear_sced()...", flush=True)
accepted, rejected, meta, flows = clear_sced(
    orderbook=orderbook,
    market_products=market_products,
    grid_data=grid_data,
    solver="highs",
)

print(f"\nResults: {len(accepted)} accepted, {len(rejected)} rejected, "
      f"{len(meta)} meta entries, {len(flows)} flow entries", flush=True)

# ── Display results ───────────────────────────────────────────────
print("\n── Accepted Supply Orders (sample) ──")
supply_accepted = [o for o in accepted if o["volume"] > 0]
for o in supply_accepted[:6]:
    print(f"  {o['unit_id']} t={o['start_time'].hour:02d}: "
          f"bid={o['volume']:.1f} MW, dispatch={o['accepted_volume']:.1f} MW, "
          f"LMP=${o['accepted_price']:.2f}/MWh")

print(f"\n── Meta (sample) ──")
for m in meta[:4]:
    print(f"  {m['product_start'].hour:02d}h: price=${m['price']:.2f}, "
          f"supply={m['supply_volume']:.1f} MW, demand={m['demand_volume']:.1f} MW")

# ── Validation ────────────────────────────────────────────────────
print("\n── Validation ──")
errors = []

# Check supply = demand
for i, (t_start, t_end, _) in enumerate(market_products):
    supply = sum(o["accepted_volume"] for o in accepted
                 if o["start_time"] == t_start and o["accepted_volume"] > 0)
    demand = sum(abs(o["accepted_volume"]) for o in accepted
                 if o["start_time"] == t_start and o["accepted_volume"] < 0)
    if abs(supply - demand) > 0.5:
        errors.append(f"Hour {i}: supply={supply:.1f} != demand={demand:.1f}")

# Check LMPs are positive
for o in accepted:
    if o["accepted_price"] < 0:
        errors.append(f"Negative LMP: {o['unit_id']} t={o['start_time'].hour}: ${o['accepted_price']:.2f}")

# Check gen_slack_0 dispatches most (cheapest)
hour12_supply = {o["unit_id"]: o["accepted_volume"]
                 for o in accepted
                 if o["start_time"].hour == 12 and o["volume"] > 0}
if hour12_supply:
    print(f"  Peak hour dispatch: {hour12_supply}")
    if hour12_supply.get("gen_slack_0", 0) < hour12_supply.get("gen_1", 0):
        errors.append("gen_slack_0 ($20) dispatches less than gen_1 ($50) at peak")

if errors:
    print(f"\n  FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("\n  ALL CHECKS PASSED")
    print("\nSCED INTEGRATION TEST COMPLETE", flush=True)
