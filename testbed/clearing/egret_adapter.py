# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapter between ASSUME market orders and Egret ModelData.

Converts ASSUME orderbook + grid_data into an Egret ModelData dict
suitable for solve_unit_commitment(), and maps Egret results back
to ASSUME accepted/rejected orders with LMP-based pricing.

Key design: Egret time series values MUST be lists (not dicts).
Egret's scale_ModelData_to_pu uses enumerate() on values, which
iterates over keys when given a dict, corrupting the data.

Multi-part bids: When a unit submits multiple orders per time period
(e.g., RL strategies with Pmin + flex bids), each tranche becomes a
separate Egret sub-generator ({unit_id}__{part_index}). This preserves
the per-tranche price differentiation in the LP.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from egret.data.model_data import ModelData

logger = logging.getLogger(__name__)


def orders_to_model_data(
    orderbook: list[dict],
    market_products: list[tuple],
    grid_data: dict[str, pd.DataFrame],
    generator_params: dict[str, dict] | None = None,
    fix_commitment: bool = True,
) -> tuple[ModelData, dict[int, str]]:
    """Convert ASSUME orders + grid topology into Egret ModelData.

    Handles multi-part bids (e.g., Pmin + flex from RL strategies) by
    creating sub-generators for units with multiple orders per time period.
    Single-bid units retain their original unit_id as the Egret generator name.

    Args:
        orderbook: List of Order dicts (volume > 0 = supply, < 0 = demand).
        market_products: List of (start, end, only_hours) tuples.
        grid_data: Dict with 'buses', 'lines' DataFrames (from MarketConfig.param_dict).
        generator_params: Optional dict of Egret-specific generator params keyed by unit_id.
        fix_commitment: If True, all generators get fixed_commitment=1 (SCED mode).

    Returns:
        (ModelData, order_gen_map) where order_gen_map maps orderbook index
        to the Egret generator name that carries its dispatch.
    """
    if generator_params is None:
        generator_params = {}

    n_periods = len(market_products)
    time_keys = list(range(n_periods))

    # ── Map products to time index ─────────────────────────────────
    product_to_idx = {}
    for i, prod in enumerate(market_products):
        product_to_idx[(prod[0], prod[1])] = i

    # ── Buses ──────────────────────────────────────────────────────
    buses_df = grid_data["buses"]
    bus_dict = {}
    reference_bus = None
    for bus_name in buses_df.index:
        bus_dict[str(bus_name)] = {
            "base_kv": float(buses_df.at[bus_name, "v_nom"]) if "v_nom" in buses_df.columns else 345.0,
            "vm": 1.0,
            "va": 0.0,
        }
        if reference_bus is None:
            reference_bus = str(bus_name)

    # ── Branches ───────────────────────────────────────────────────
    lines_df = grid_data["lines"]
    branch_dict = {}
    for line_name in lines_df.index:
        row = lines_df.loc[line_name]
        branch_dict[str(line_name)] = {
            "from_bus": str(row["bus0"]),
            "to_bus": str(row["bus1"]),
            "in_service": True,
            "branch_type": "line",
            "reactance": float(row.get("x_pu_eff", row.get("x_pu", row.get("x", 0.01)))),
            "resistance": float(row.get("r_pu_eff", row.get("r_pu", row.get("r", 0.0)))),
            "rating_long_term": float(row["s_nom"]),
            "rating_short_term": float(row["s_nom"]),
            "rating_emergency": float(row["s_nom"]),
        }

    # ── Generators and Loads from orderbook ────────────────────────
    gen_dict = {}
    load_by_bus: dict[str, list[float]] = {}  # bus -> [load_t0, load_t1, ...]
    order_gen_map: dict[int, str] = {}  # orderbook index -> Egret gen name

    orders_df = pd.DataFrame(orderbook)

    # --- Supply orders: detect multi-part bids and create sub-generators ---
    supply_df = orders_df[orders_df["volume"] > 0]

    for unit_id in supply_df["unit_id"].unique():
        unit_orders = supply_df[supply_df["unit_id"] == unit_id]
        node = str(unit_orders.iloc[0]["node"])

        # Group orders by time period: t_idx -> [(df_index, volume, price), ...]
        period_groups: dict[int, list[tuple[int, float, float]]] = {}
        for df_idx, order in unit_orders.iterrows():
            t_idx = product_to_idx.get((order["start_time"], order["end_time"]))
            if t_idx is not None:
                if t_idx not in period_groups:
                    period_groups[t_idx] = []
                period_groups[t_idx].append(
                    (int(df_idx), float(order["volume"]), float(order["price"]))
                )

        max_parts = max((len(bids) for bids in period_groups.values()), default=1)
        gp = generator_params.get(unit_id, {})

        if max_parts == 1:
            # ── Single bid per period ──────────────────────────────
            volumes = [0.0] * n_periods
            prices = [0.0] * n_periods
            for t_idx, bids in period_groups.items():
                df_idx, vol, price = bids[0]
                volumes[t_idx] = vol
                prices[t_idx] = price
                order_gen_map[df_idx] = unit_id

            # Always use time-varying p_max and cost curves.
            # Every bid has a per-hour price and volume — pass them all
            # so Egret sees the correct capacity and cost each hour.
            p_max = max((v for v in volumes if v > 0), default=0)
            nonzero = [(p, v) for p, v in zip(prices, volumes) if v > 0]
            avg_price = sum(p for p, _ in nonzero) / max(1, len(nonzero))

            gen_dict[unit_id] = _make_gen_entry(
                node, p_max, avg_price, gp, fix_commitment, n_periods,
                p_max_ts=volumes,
                prices_ts=prices,
            )

        else:
            # ── Multi-part bids — create sub-generators per tranche ─
            for part in range(max_parts):
                sub_name = f"{unit_id}__{part}"
                volumes = [0.0] * n_periods
                prices = [0.0] * n_periods
                commitments = [0] * n_periods

                for t_idx, bids in period_groups.items():
                    # Sort by price within the period (cheapest tranche first)
                    sorted_bids = sorted(bids, key=lambda b: b[2])
                    if part < len(sorted_bids):
                        df_idx, vol, price = sorted_bids[part]
                        volumes[t_idx] = vol
                        prices[t_idx] = price
                        commitments[t_idx] = 1
                        order_gen_map[df_idx] = sub_name

                if not any(v > 0 for v in volumes):
                    continue

                p_max = max((v for v in volumes if v > 0), default=0)
                nonzero = [(p, v) for p, v in zip(prices, volumes) if v > 0]
                avg_price = sum(p for p, _ in nonzero) / max(1, len(nonzero))

                gen_dict[sub_name] = _make_gen_entry(
                    node, p_max, avg_price, gp, fix_commitment, n_periods,
                    commitments=commitments,
                    p_max_ts=volumes,
                    prices_ts=prices,
                )

    # --- Demand orders (unchanged — aggregate by bus) ---
    demand_df = orders_df[orders_df["volume"] < 0]

    for unit_id in demand_df["unit_id"].unique():
        unit_orders = demand_df[demand_df["unit_id"] == unit_id]
        node = str(unit_orders.iloc[0]["node"])

        if node not in load_by_bus:
            load_by_bus[node] = [0.0] * n_periods
        for _, order in unit_orders.iterrows():
            t_idx = product_to_idx.get((order["start_time"], order["end_time"]))
            if t_idx is not None:
                load_by_bus[node][t_idx] += abs(float(order["volume"]))

    # Build Egret load elements
    load_dict = {}
    for bus, p_loads in load_by_bus.items():
        load_dict[f"load_{bus}"] = {
            "bus": bus,
            "in_service": True,
            "p_load": {
                "data_type": "time_series",
                "values": p_loads,
            },
        }

    # ── System ─────────────────────────────────────────────────────
    # Infer period length from market products
    if n_periods > 0:
        p0 = market_products[0]
        period_minutes = int((p0[1] - p0[0]).total_seconds() / 60)
    else:
        period_minutes = 60

    system = {
        "time_keys": time_keys,
        "time_period_length_minutes": period_minutes,
        "baseMVA": 100.0,
        "reference_bus": reference_bus,
        "reference_bus_angle": 0.0,
    }

    model_data = ModelData({
        "system": system,
        "elements": {
            "bus": bus_dict,
            "generator": gen_dict,
            "branch": branch_dict,
            "load": load_dict,
        },
    })

    return model_data, order_gen_map


def _make_gen_entry(
    node: str,
    p_max: float,
    avg_price: float,
    gp: dict,
    fix_commitment: bool,
    n_periods: int,
    commitments: list[int] | None = None,
    p_max_ts: list[float] | None = None,
    prices_ts: list[float] | None = None,
) -> dict:
    """Build an Egret generator dict.

    Args:
        node: Bus name.
        p_max: Maximum capacity (scalar, used for cost curve and ramp limits).
        avg_price: Average marginal price (fallback for static cost curve).
        gp: Generator params override dict.
        fix_commitment: Whether to fix commitment to 1.
        n_periods: Number of time periods.
        commitments: Per-period commitment values (for sub-generators).
        p_max_ts: Per-period p_max time series (for sub-generators with
            varying capacity). If provided, overrides the scalar p_max
            for the Egret p_max field.
        prices_ts: Per-period bid prices. If provided alongside p_max_ts,
            builds a time-varying cost curve for accurate per-hour LMPs.
    """
    # Build cost curve: time-varying if per-period prices are available
    if prices_ts is not None and p_max_ts is not None:
        cost_curves = []
        for vol, price in zip(p_max_ts, prices_ts):
            if vol > 0:
                cost_curves.append({
                    "data_type": "cost_curve",
                    "cost_curve_type": "piecewise",
                    "values": [(0, 0.0), (vol, price * vol)],
                })
            else:
                # Uncommitted period — zero-cost placeholder
                cost_curves.append({
                    "data_type": "cost_curve",
                    "cost_curve_type": "piecewise",
                    "values": [(0, 0.0)],
                })
        p_cost = {"data_type": "time_series", "values": cost_curves}
    else:
        p_cost = {
            "data_type": "cost_curve",
            "cost_curve_type": "piecewise",
            "values": [(0, 0.0), (p_max, avg_price * p_max)],
        }

    entry = {
        "bus": node,
        "in_service": True,
        "generator_type": "thermal",
        "p_min": float(gp.get("p_min", 0.0)),
        "p_max": p_max,
        "p_cost": p_cost,
        "startup_cost": float(gp.get("startup_cost", 0)),
        "shutdown_cost": float(gp.get("shutdown_cost", 0)),
        "ramp_up_60min": float(gp.get("ramp_up_60min", p_max)),
        "ramp_down_60min": float(gp.get("ramp_down_60min", p_max)),
        "min_up_time": int(gp.get("min_up_time", 1)),
        "min_down_time": int(gp.get("min_down_time", 1)),
        "initial_status": int(gp.get("initial_status", 1)),
        "initial_p_output": float(gp.get("initial_p_output", 0.0)),
    }

    # Time-varying p_max for sub-generators
    if p_max_ts is not None:
        entry["p_max"] = {
            "data_type": "time_series",
            "values": p_max_ts,
        }

    if fix_commitment:
        if commitments is not None:
            entry["fixed_commitment"] = {
                "data_type": "time_series",
                "values": commitments,
            }
        else:
            entry["fixed_commitment"] = {
                "data_type": "time_series",
                "values": [1] * n_periods,
            }

    return entry


def model_results_to_orderbook(
    result: ModelData,
    orderbook: list[dict],
    market_products: list[tuple],
    order_gen_map: dict[int, str] | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Map Egret solve results back to ASSUME accepted/rejected orders.

    Args:
        result: ModelData returned by solve_unit_commitment().
        orderbook: Original ASSUME orderbook.
        market_products: List of (start, end, only_hours) tuples.
        order_gen_map: Maps orderbook index to Egret generator name.
            Required for multi-part bids where sub-generators were created.

    Returns:
        (accepted_orders, rejected_orders, meta_list, flows_dict)
    """
    n_periods = len(market_products)

    # ── Extract LMPs per bus per period ────────────────────────────
    lmps: dict[str, list[float]] = {}
    for bus_name, bus_data in result.data["elements"]["bus"].items():
        lmp_ts = bus_data.get("lmp", {})
        if isinstance(lmp_ts, dict) and "values" in lmp_ts:
            vals = lmp_ts["values"]
            if isinstance(vals, list):
                lmps[bus_name] = [float(v) for v in vals]
            else:
                lmps[bus_name] = [0.0] * n_periods
        else:
            lmps[bus_name] = [0.0] * n_periods

    # ── Extract dispatch per generator per period ──────────────────
    dispatch: dict[str, list[float]] = {}
    for gen_name, gen_data in result.data["elements"]["generator"].items():
        pg_ts = gen_data.get("pg", {})
        if isinstance(pg_ts, dict) and "values" in pg_ts:
            vals = pg_ts["values"]
            if isinstance(vals, list):
                dispatch[gen_name] = [float(v) for v in vals]
            else:
                dispatch[gen_name] = [0.0] * n_periods
        else:
            dispatch[gen_name] = [0.0] * n_periods

    # ── Extract line flows (already in MW from Egret PTDF solver) ───
    flows_dict: dict[tuple, float] = {}
    for line_name, line_data in result.data["elements"]["branch"].items():
        pf_ts = line_data.get("pf", {})
        if isinstance(pf_ts, dict) and "values" in pf_ts:
            vals = pf_ts["values"]
            for t in range(min(n_periods, len(vals) if isinstance(vals, list) else 0)):
                flow_val = float(vals[t]) if isinstance(vals, list) else 0.0
                flows_dict[(line_name, t)] = flow_val

    # ── Map products to time index ─────────────────────────────────
    product_to_idx = {}
    for i, prod in enumerate(market_products):
        product_to_idx[(prod[0], prod[1])] = i

    # ── Build accepted/rejected orders ─────────────────────────────
    accepted_orders = []
    rejected_orders = []

    for i, order in enumerate(orderbook):
        order = dict(order)  # copy
        unit_id = order["unit_id"]
        node = str(order["node"])
        t_idx = product_to_idx.get((order["start_time"], order["end_time"]))

        if t_idx is None:
            order["accepted_volume"] = 0.0
            order["accepted_price"] = 0.0
            rejected_orders.append(order)
            continue

        # Get LMP at this bus for this period
        bus_lmp = lmps.get(node, [0.0] * n_periods)
        lmp_val = bus_lmp[t_idx] if t_idx < len(bus_lmp) else 0.0

        if order["volume"] > 0:
            # Supply order: look up the correct Egret generator
            gen_name = order_gen_map.get(i, unit_id) if order_gen_map else unit_id
            gen_dispatch = dispatch.get(gen_name, [0.0] * n_periods)
            acc_vol = gen_dispatch[t_idx] if t_idx < len(gen_dispatch) else 0.0
            order["accepted_volume"] = acc_vol
            order["accepted_price"] = lmp_val
        elif order["volume"] < 0:
            # Demand order: accepted at load volume, priced at LMP
            order["accepted_volume"] = order["volume"]
            order["accepted_price"] = lmp_val
        else:
            order["accepted_volume"] = 0.0
            order["accepted_price"] = 0.0

        if abs(order["accepted_volume"]) > 0.01:
            accepted_orders.append(order)
        else:
            rejected_orders.append(order)

    # ── Build metadata ─────────────────────────────────────────────
    meta_list = []
    for i, prod in enumerate(market_products):
        duration_hours = (prod[1] - prod[0]).total_seconds() / 3600

        # Aggregate supply/demand per product
        supply_vol = sum(
            o["accepted_volume"] for o in accepted_orders
            if o["start_time"] == prod[0] and o["accepted_volume"] > 0
        )
        demand_vol = sum(
            abs(o["accepted_volume"]) for o in accepted_orders
            if o["start_time"] == prod[0] and o["accepted_volume"] < 0
        )

        # Use avg LMP across buses for the meta price
        all_lmps = [lmps[b][i] for b in lmps if i < len(lmps[b])]
        avg_price = sum(all_lmps) / max(1, len(all_lmps)) if all_lmps else 0.0

        # Per-bus LMPs for this period (always available, even for zero-dispatch buses)
        bus_lmps_t = {b: lmps[b][i] for b in lmps if i < len(lmps[b])}

        meta_list.append({
            "supply_volume": supply_vol,
            "demand_volume": demand_vol,
            "supply_volume_energy": supply_vol * duration_hours,
            "demand_volume_energy": demand_vol * duration_hours,
            "price": avg_price,
            "max_price": max(all_lmps) if all_lmps else 0.0,
            "min_price": min(all_lmps) if all_lmps else 0.0,
            "bus_lmps": bus_lmps_t,
            "node": None,
            "product_start": prod[0],
            "product_end": prod[1],
            "only_hours": prod[2] if len(prod) > 2 else None,
        })

    return accepted_orders, rejected_orders, meta_list, flows_dict
