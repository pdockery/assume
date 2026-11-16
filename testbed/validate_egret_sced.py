"""Standalone Egret SCED validation on the IEEE 9-bus network.

Builds an Egret ModelData from the canonical Anderson & Fouad 9-bus system
(same topology as ieee9_network.py), solves SCED (LP with all generators
committed), and validates dispatch, LMPs, and line flows.

Usage:
    conda activate assume
    python testbed/validate_egret_sced.py
"""

import sys
import numpy as np

print("=== Egret SCED Validation: IEEE 9-Bus ===\n", flush=True)

from egret.data.model_data import ModelData
from egret.models.unit_commitment import solve_unit_commitment

# ── Monkey-patch for HiGHS + Egret 0.6.0 compatibility ─────────────────
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clearing.highs_compat import patch_egret_for_highs

patch_egret_for_highs()

# ── IEEE 9-bus parameters ──────────────────────────────────────────────
# Per-unit impedances on 100 MVA base, 345 kV
# Source: pandapower.networks.case9() (PYPOWER / Anderson & Fouad 1980)

BUSES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Line data: (from_bus, to_bus, reactance_pu, resistance_pu, rating_MW)
LINES = [
    ("1", "4", 0.0576, 0.0000, 250.0),
    ("4", "5", 0.0920, 0.0170, 250.0),
    ("5", "6", 0.1700, 0.0390, 150.0),
    ("6", "3", 0.0586, 0.0000, 300.0),
    ("6", "7", 0.1008, 0.0119, 150.0),
    ("7", "8", 0.0720, 0.0085, 250.0),
    ("8", "2", 0.0625, 0.0000, 250.0),
    ("8", "9", 0.1610, 0.0320, 250.0),
    ("9", "4", 0.0850, 0.0100, 250.0),
]

# Generator data: (name, bus, p_min, p_max, marginal_cost $/MWh)
GENERATORS = [
    ("gen_slack_0", "1", 10.0, 250.0, 20.0),
    ("gen_0", "2", 10.0, 300.0, 35.0),
    ("gen_1", "3", 10.0, 270.0, 50.0),
]

# Load data: (name, bus, p_load_base_MW)
LOADS = [
    ("load_0", "5", 90.0),
    ("load_1", "7", 100.0),
    ("load_2", "9", 125.0),
]

# ── Time series ────────────────────────────────────────────────────────
N_PERIODS = 24
TIME_KEYS = list(range(N_PERIODS))

# Diurnal load profile (same as ieee9_network.py)
hours = np.arange(N_PERIODS) % 24
profile = 0.6 + 0.4 * np.sin(np.pi * (hours - 4) / 16)
profile = np.clip(profile, 0.5, 1.0)


def _make_time_series(values_list):
    """Create Egret time_series dict. values must be a LIST (not dict).

    Egret's scale_ModelData_to_pu uses enumerate() on values, which only
    works correctly with lists. Using a dict causes keys to be scaled
    instead of values.
    """
    assert isinstance(values_list, list), f"values must be a list, got {type(values_list)}"
    return {"data_type": "time_series", "values": values_list}


def _make_cost_curve(marginal_cost, p_max):
    """Piecewise linear cost curve: 0 at 0 MW, marginal_cost * p_max at p_max."""
    return {
        "data_type": "cost_curve",
        "cost_curve_type": "piecewise",
        "values": [(0, 0.0), (p_max, marginal_cost * p_max)],
    }


# ── Build ModelData ────────────────────────────────────────────────────
bus_dict = {}
for b in BUSES:
    bus_dict[b] = {
        "base_kv": 345.0,
        "matpower_bustype": "ref" if b == "1" else "PQ",
        "vm": 1.0,
        "va": 0.0,
    }

gen_dict = {}
for name, bus, p_min, p_max, mc in GENERATORS:
    gen_dict[name] = {
        "bus": bus,
        "in_service": True,
        "generator_type": "thermal",
        "fuel": "gas" if mc > 20 else "nuclear",
        "p_min": p_min,
        "p_max": p_max,
        "fixed_commitment": _make_time_series([1] * N_PERIODS),
        "p_cost": _make_cost_curve(mc, p_max),
        "startup_cost": 0,
        "shutdown_cost": 0,
        "ramp_up_60min": p_max,
        "ramp_down_60min": p_max,
        "min_up_time": 1,
        "min_down_time": 1,
        "initial_status": 1,
        "initial_p_output": p_max * 0.5,
    }

branch_dict = {}
for i, (fb, tb, x, r, rating) in enumerate(LINES):
    branch_dict[f"line_{i}"] = {
        "from_bus": fb,
        "to_bus": tb,
        "in_service": True,
        "branch_type": "line",
        "reactance": x,
        "resistance": r,
        "rating_long_term": rating,
        "rating_short_term": rating,
        "rating_emergency": rating,
    }

load_dict = {}
for name, bus, p_base in LOADS:
    load_dict[name] = {
        "bus": bus,
        "in_service": True,
        "p_load": _make_time_series([float(p_base * profile[t]) for t in TIME_KEYS]),
    }

md = ModelData({
    "system": {
        "time_keys": TIME_KEYS,
        "time_period_length_minutes": 60,
        "baseMVA": 100.0,
        "reference_bus": "1",
        "reference_bus_angle": 0.0,
    },
    "elements": {
        "bus": bus_dict,
        "generator": gen_dict,
        "branch": branch_dict,
        "load": load_dict,
    },
})

# ── Solve SCED ─────────────────────────────────────────────────────────
print("Solving SCED (relaxed=True, 24 periods)...", flush=True)
try:
    result = solve_unit_commitment(md, "highs", relaxed=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Extract and display results ────────────────────────────────────────
total_cost = result.data["system"].get("total_cost", "N/A")
print(f"\nTotal cost: ${total_cost:,.2f}\n")

# Generator dispatch
print("── Generator Dispatch (MW) ──")
print(f"{'Hour':>4}", end="")
gen_names = list(result.data["elements"]["generator"].keys())
for gn in gen_names:
    print(f"  {gn:>14}", end="")
print(f"  {'Total Gen':>10}  {'Total Load':>10}")

def _get_ts_value(ts_data, t_idx):
    """Extract value from Egret time series (list-based) at time index t_idx."""
    if isinstance(ts_data, dict) and "values" in ts_data:
        vals = ts_data["values"]
        if isinstance(vals, list):
            return vals[t_idx] if t_idx < len(vals) else 0
        elif isinstance(vals, dict):
            return vals.get(t_idx, 0)
    elif isinstance(ts_data, (int, float)):
        return ts_data
    return 0

total_load_by_hour = {}
for t in TIME_KEYS:
    total_load = sum(
        _get_ts_value(load_data["p_load"], t)
        for load_data in result.data["elements"]["load"].values()
    )
    total_load_by_hour[t] = total_load

for t in TIME_KEYS:
    print(f"{t:>4}", end="")
    total_gen = 0
    for gn in gen_names:
        gen = result.data["elements"]["generator"][gn]
        val = _get_ts_value(gen.get("pg", {}), t)
        total_gen += val
        print(f"  {val:>14.2f}", end="")
    print(f"  {total_gen:>10.2f}  {total_load_by_hour[t]:>10.2f}")

# LMPs
print("\n── Locational Marginal Prices ($/MWh) ──")
print(f"{'Hour':>4}", end="")
for b in BUSES:
    print(f"  Bus {b:>2}", end="")
print()

lmp_ts = {}
for b in BUSES:
    bus_data = result.data["elements"]["bus"][b]
    lmp_ts[b] = bus_data.get("lmp", {})

for t in TIME_KEYS:
    print(f"{t:>4}", end="")
    for b in BUSES:
        val = _get_ts_value(lmp_ts[b], t)
        print(f"  {val:>6.2f}", end="")
    print()

# Line flows
print("\n── Line Flows (MW, positive = from→to) ──")
print(f"{'Hour':>4}", end="")
line_names = list(result.data["elements"]["branch"].keys())
for ln in line_names:
    br = result.data["elements"]["branch"][ln]
    print(f"  {ln}({br['from_bus']}-{br['to_bus']})", end="")
print()

flow_ts = {}
for ln in line_names:
    br = result.data["elements"]["branch"][ln]
    flow_ts[ln] = br.get("pf", {})

for t in [0, 6, 12, 18, 23]:  # sample hours
    print(f"{t:>4}", end="")
    for ln in line_names:
        val = _get_ts_value(flow_ts[ln], t)
        print(f"  {val:>14.2f}", end="")
    print()

# ── Validation checks ─────────────────────────────────────────────────
print("\n── Validation ──")
errors = []

# Check 1: Supply = Demand each hour
for t in TIME_KEYS:
    total_gen = sum(
        _get_ts_value(result.data["elements"]["generator"][gn].get("pg", {}), t)
        for gn in gen_names
    )
    if abs(total_gen - total_load_by_hour[t]) > 0.1:
        errors.append(f"Hour {t}: gen={total_gen:.2f} != load={total_load_by_hour[t]:.2f}")

# Check 2: LMPs exist for all buses
for b in BUSES:
    if not lmp_ts[b]:
        errors.append(f"Bus {b}: no LMP data")

# Check 3: Line flows respect thermal limits (with small tolerance)
for ln in line_names:
    br = result.data["elements"]["branch"][ln]
    rating = br["rating_long_term"]
    for t in TIME_KEYS:
        flow = abs(_get_ts_value(flow_ts[ln], t))
        if flow > rating + 0.1:
            errors.append(f"{ln} hour {t}: |flow|={flow:.2f} > rating={rating:.2f}")

# Check 4: Dispatch within generator limits
for gn in gen_names:
    gen = result.data["elements"]["generator"][gn]
    p_min = gen["p_min"]
    p_max = gen["p_max"]
    for t in TIME_KEYS:
        pg = _get_ts_value(gen.get("pg", {}), t)
        if pg < p_min - 0.1 or pg > p_max + 0.1:
            errors.append(f"{gn} hour {t}: pg={pg:.2f} outside [{p_min}, {p_max}]")

# Check 5: LMPs at reference bus should approximate marginal cost
ref_lmps = [_get_ts_value(lmp_ts["1"], t) for t in TIME_KEYS]
if ref_lmps:
    avg_ref_lmp = np.mean(ref_lmps)
    print(f"  Avg LMP at ref bus (1): ${avg_ref_lmp:.2f}/MWh")

# Check 6: LMP spatial variation (should differ when lines are loaded)
for t in [12]:  # check peak-ish hour
    lmps_at_t = [_get_ts_value(lmp_ts[b], t) for b in BUSES]
    lmp_range = max(lmps_at_t) - min(lmps_at_t)
    print(f"  LMP range at hour {t}: ${lmp_range:.2f}/MWh (>0 indicates congestion)")

if errors:
    print(f"\n  FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("\n  ALL CHECKS PASSED")
    print("\nSCED VALIDATION COMPLETE", flush=True)
