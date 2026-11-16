# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Powerex corner-and-squeeze illustration on the IEEE 9-bus testbed.

Thin helpers that reuse the existing testbed machinery -- ``build_ieee9_network``
and the Egret ``clear_sced`` role -- to stage a strategic storage / long-term-
obligation player ("Powerex") on the canonical 9-bus system, with NO RL agents.
Everything is a deterministic SCED you can re-run after moving a few variables.

It is the network-grounded companion to the abstract three-node model in
``western_seams/markets/canonical_corner.py`` (writeup
``notes/powerex/powerex_writeup.tex``).  The mapping:

    3-node model            IEEE 9-bus illustration
    --------------------    --------------------------------------------
    node A (hydro @ lam)    bus 1  (co-located with the $20 slack unit)
    A--B--C, no A--C        congested mesh: line_1 (4->5) capped at 90 MW
    node C (scarce LT load) bus 5  (downstream of the constraint, ~$60 LMP)
    K_BC binding            the 4->5 corridor that isolates bus 5

Two mechanisms, mirroring the writeup's two rent layers:

* **Spatial corner** -- Powerex injects cheap hydro at bus 1 but the value is
  set at bus 5, a node where it owns no generator.  Its transmission position
  (a financial transmission right, FTR) lets it collect the congestion spread
  ``LMP_5 - LMP_1`` -- arbitraging nodes it is not physically connected to.
* **Intertemporal squeeze** -- a reservoir with a daily energy budget that it
  releases into the highest-price hours, arbitraging the diurnal spread that
  intertemporally-constrained competitors (ramp-limited thermals, short
  batteries) cannot span.

This module holds the logic; the notebook calls these functions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from ieee9_network import build_ieee9_network
from clearing.sced_clearing import clear_sced

logger = logging.getLogger(__name__)

# Congestion preset from scenarios.yaml (line_overrides_congested): line_1 is
# bus 4->5, line_5 is bus 7->8.  Capping 4->5 isolates bus 5 at a high LMP.
CONGESTED_LIMITS: dict[str, float] = {"line_1": 90.0, "line_5": 90.0}

# Where Powerex sits in the 9-bus analogy.
STORAGE_BUS = "1"   # hydro / reservoir, co-located with the cheap slack unit
LT_BUS = "5"        # long-term delivery obligation (the scarce DSW analog)

VOLL = 1000.0       # load bid price ($/MWh), matches presentation_charts


def build_network(periods: int = 24, congested: bool = True,
                  line_overrides: dict[str, float] | None = None):
    """Build the IEEE 9-bus and apply line-rating overrides for congestion."""
    net = build_ieee9_network(periods=periods)
    overrides = line_overrides if line_overrides is not None else (
        CONGESTED_LIMITS if congested else {})
    for line, s_nom in overrides.items():
        net.lines.at[line, "s_nom"] = float(s_nom)
    return net


@dataclass
class Dispatch:
    """Result of one ``clear_sced`` call over a set of hours."""

    net: object
    accepted: list
    rejected: list
    meta: list
    flows: dict
    hours: list[int]                      # snapshot indices, in product order
    capacities: dict[str, float]          # unit_id -> nameplate (for viz)
    _pos: dict[int, int] = field(init=False)

    def __post_init__(self):
        self._pos = {h: i for i, h in enumerate(self.hours)}

    def lmps(self, hour: int) -> dict[str, float]:
        return self.meta[self._pos[hour]].get("bus_lmps", {})

    def lmp(self, bus: str, hour: int) -> float:
        return float(self.lmps(hour).get(str(bus), self.meta[self._pos[hour]]["price"]))

    def dispatch(self, hour: int) -> dict[str, float]:
        t0 = self.meta[self._pos[hour]]["product_start"]
        return {o["unit_id"]: o.get("accepted_volume", 0.0)
                for o in self.accepted if o["start_time"] == t0 and o["volume"] > 0}

    def lmp_series(self, bus: str) -> pd.Series:
        return pd.Series({h: self.lmp(bus, h) for h in self.hours}).sort_index()


def clear(net, hours, extra_orders=None, capacities=None) -> Dispatch:
    """Run ``clear_sced`` for the given snapshot ``hours``.

    Args:
        net: PyPSA network from :func:`build_network`.
        hours: iterable of snapshot indices to clear.
        extra_orders: dict ``{hour_index: [order, ...]}`` of extra bids
            (Powerex storage / LT load).  Each order needs ``price``,
            ``volume`` (>0 supply, <0 demand), ``node``, ``unit_id``; time
            fields and ``bid_id`` are filled in here.
        capacities: optional ``{unit_id: nameplate_MW}`` for plotting; the
            base generators are added automatically.

    Returns:
        :class:`Dispatch`.
    """
    hours = list(hours)
    extra_orders = extra_orders or {}
    caps = {g: float(net.generators.at[g, "p_nom"]) for g in net.generators.index}
    caps.update(capacities or {})

    orderbook, products = [], []
    for h in hours:
        ts = net.snapshots[h]
        start = ts.to_pydatetime()
        end = start + timedelta(hours=1)
        products.append((start, end, None))
        for g, row in net.generators.iterrows():
            orderbook.append(dict(
                start_time=start, end_time=end, only_hours=None,
                price=float(row.marginal_cost), volume=float(row.p_nom),
                node=str(row.bus), unit_id=g, bid_id=f"{g}_{h}"))
        for load in net.loads.index:
            d = float(net.loads_t.p_set.at[ts, load])
            orderbook.append(dict(
                start_time=start, end_time=end, only_hours=None,
                price=VOLL, volume=-d, node=str(net.loads.at[load, "bus"]),
                unit_id=load, bid_id=f"{load}_{h}"))
        for o in extra_orders.get(h, []):
            orderbook.append(dict(
                start_time=start, end_time=end, only_hours=None,
                bid_id=f"{o['unit_id']}_{h}", **o))

    grid_data = {
        "buses": net.buses, "lines": net.lines,
        "generators": net.generators.rename(columns={"bus": "node", "p_nom": "max_power"}),
        "loads": net.loads.rename(columns={"bus": "node", "p_set": "min_power"}),
    }
    accepted, rejected, meta, flows = clear_sced(
        orderbook, products, grid_data, solver="highs")
    return Dispatch(net, accepted, rejected, meta, flows, hours, caps)


def get_hour_data(disp: Dispatch, hour: int):
    """Build nodal_plot inputs for one hour (supply/demand/served/lmp/flows).

    Mirrors the structure ``presentation_charts.ipynb`` feeds to
    ``plot_network_topology`` / ``plot_nodal_circlize``.
    """
    pos = disp._pos[hour]
    t0 = disp.meta[pos]["product_start"]
    net = disp.net
    all_buses = sorted(net.buses.index.tolist(), key=lambda x: int(x))

    supply_by_bus: dict[str, list] = {}
    for o in disp.accepted + disp.rejected:
        if o["start_time"] != t0 or o["volume"] <= 0:
            continue
        supply_by_bus.setdefault(str(o["node"]), []).append({
            "unit_id": o["unit_id"], "price": o["price"], "volume": o["volume"],
            "accepted_volume": o.get("accepted_volume", 0.0),
            "capacity": disp.capacities.get(o["unit_id"], o["volume"]),
        })
    for gens in supply_by_bus.values():
        gens.sort(key=lambda g: g["price"])

    demand_by_bus, demand_served = {}, {}
    for o in disp.accepted + disp.rejected:
        if o["start_time"] != t0 or o["volume"] >= 0:
            continue
        b = str(o["node"])
        demand_by_bus[b] = demand_by_bus.get(b, 0.0) + abs(o["volume"])
        demand_served[b] = demand_served.get(b, 0.0) + abs(o.get("accepted_volume", 0.0))

    bus_lmps = disp.lmps(hour)
    line_flows = {ln: f for (ln, t), f in disp.flows.items() if t == pos}
    return supply_by_bus, demand_by_bus, demand_served, bus_lmps, line_flows


# --------------------------------------------------------------------------
# Scene 1 -- spatial corner P&L
# --------------------------------------------------------------------------

def corner_pnl(disp: Dispatch, hour: int, *, lam: float, release_mw: float,
               Q_LT: float, P_LT: float, ftr_mw: float,
               storage_bus: str = STORAGE_BUS, lt_bus: str = LT_BUS,
               ftr_from: str = STORAGE_BUS, ftr_to: str = LT_BUS) -> dict:
    """Decompose Powerex's single-hour profit into its three layers.

    1. **Storage margin** -- the reservoir dispatched at bus ``storage_bus``
       earns ``(LMP_storage - lam) * release``.
    2. **LT obligation** -- deliver ``Q_LT`` at ``lt_bus`` for fixed ``P_LT``
       while the spot value of sourcing there is ``LMP_lt``:
       ``(P_LT - LMP_lt) * Q_LT``.
    3. **FTR rent** -- a financial transmission right on ``ftr_from -> ftr_to``
       (nodes where Powerex owns nothing) pays ``(LMP_to - LMP_from) * ftr_mw``.
    """
    lmp_s = disp.lmp(storage_bus, hour)
    lmp_lt = disp.lmp(lt_bus, hour)
    lmp_i = disp.lmp(ftr_from, hour)
    lmp_j = disp.lmp(ftr_to, hour)

    storage_margin = (lmp_s - lam) * release_mw
    lt_margin = (P_LT - lmp_lt) * Q_LT
    ftr_rent = (lmp_j - lmp_i) * ftr_mw
    spread = lmp_lt - lmp_s
    return {
        "lmp_storage": lmp_s, "lmp_lt": lmp_lt, "congestion_spread": spread,
        "storage_margin": storage_margin, "lt_margin": lt_margin,
        "ftr_rent": ftr_rent,
        "total": storage_margin + lt_margin + ftr_rent,
    }


# --------------------------------------------------------------------------
# Scene 2 -- intertemporal reservoir release
# --------------------------------------------------------------------------

def release_schedule(price_series: pd.Series, energy_budget: float,
                     power_cap: float, *, strategy: str = "peak") -> pd.Series:
    """Allocate a daily energy budget across hours.

    ``"peak"`` fills ``power_cap`` in the highest-price hours until the budget
    is exhausted (strategic concentration into scarcity).  ``"flat"`` spreads
    the budget evenly (the naive, price-taking benchmark).
    """
    hours = list(price_series.index)
    rel = pd.Series(0.0, index=hours)
    if strategy == "flat":
        per = min(power_cap, energy_budget / len(hours))
        rel[:] = per
        return rel
    remaining = energy_budget
    for h in price_series.sort_values(ascending=False).index:
        take = min(power_cap, remaining)
        rel[h] = take
        remaining -= take
        if remaining <= 1e-9:
            break
    return rel


def intertemporal_pnl(disp: Dispatch, release: pd.Series, *, lam: float,
                      bus: str = LT_BUS) -> dict:
    """Reservoir revenue at realized LMPs minus the water value of releases.

    Releases are valued at the delivery node ``bus`` (the LT/scarce node),
    consistent with Powerex selling its hydro into the downstream peak.
    """
    lmp = disp.lmp_series(bus)
    revenue = float((release * lmp).sum())
    water_cost = float(lam * release.sum())
    return {
        "energy_released": float(release.sum()),
        "revenue": revenue, "water_cost": water_cost,
        "profit": revenue - water_cost,
        "avg_capture_price": revenue / release.sum() if release.sum() > 0 else 0.0,
    }
