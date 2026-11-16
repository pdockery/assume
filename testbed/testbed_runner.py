# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
IEEE 9-bus testbed runner for ASSUME.

Exercises the full PyPSA → ASSUME workflow:
  pandapower.case9() → PyPSA Network → load_pypsa() → World.run()

Usage:
    python -m testbed.testbed_runner
    python testbed/testbed_runner.py
"""

import logging
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from dateutil import rrule as rr

# Ensure the repo root is on sys.path so `assume` and `testbed` are importable
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from assume import World
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.markets import clearing_mechanisms
from assume.scenario.loader_pypsa import load_pypsa
from testbed.clearing.sced_clearing import ScedClearingRole
from testbed.ieee9_network import assign_zones, build_ieee9_network, get_seam_lines

logger = logging.getLogger(__name__)

# Register Egret-based clearing mechanisms alongside built-in ones
clearing_mechanisms["egret_sced"] = ScedClearingRole


def run_testbed(
    market_mechanism: str = "pay_as_clear",
    periods: int = 24,
    db_uri: str = "",
    export_csv: str = "",
    line_overrides: dict | None = None,
) -> World:
    """Build and run the IEEE 9-bus testbed through the PyPSA→ASSUME pipeline.

    Args:
        market_mechanism: Clearing algorithm name (e.g., 'pay_as_clear',
            'nodal_clearing', 'complex_clearing', 'egret_sced').
        periods: Number of hourly snapshots.
        db_uri: Database URI for results (empty string = no DB).
        export_csv: Path for CSV export (empty string = no export).
        line_overrides: Optional dict of {line_name: {col: val}} to modify
            line parameters (e.g., s_nom for congestion scenarios).

    Returns:
        The World object after simulation (for inspection/testing).
    """
    # ── 1. Build PyPSA network from pandapower case9 ──────────────────
    network = build_ieee9_network(periods=periods)
    assign_zones(network)
    seam_lines = get_seam_lines(network)

    # Apply line overrides (e.g., reduce s_nom for congestion scenarios)
    if line_overrides:
        for line_name, overrides in line_overrides.items():
            for col, val in overrides.items():
                network.lines.loc[line_name, col] = val
                logger.info(f"Line override: {line_name} {col} = {val}")

    start = network.snapshots[0]
    end = network.snapshots[-1]

    logger.info(
        f"IEEE 9-bus network: {len(network.buses)} buses, "
        f"{len(network.generators)} gens, {len(network.loads)} loads, "
        f"{len(seam_lines)} seam lines"
    )

    # ── 2. Define market design ───────────────────────────────────────
    additional_fields = ["node", "max_power", "min_power"]
    if market_mechanism in ("nodal_clearing", "complex_clearing", "redispatch"):
        additional_fields.append("bid_type")

    param_dict: dict = {}
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

    # ── 3. Define bidding strategies ──────────────────────────────────
    default_pp_strategy = {
        mc.market_id: "powerplant_energy_naive" for mc in marketdesign
    }
    default_demand_strategy = {
        mc.market_id: "demand_energy_naive" for mc in marketdesign
    }

    bidding_strategies = {
        "power_plant": defaultdict(lambda: default_pp_strategy),
        "demand": defaultdict(lambda: default_demand_strategy),
        "storage": defaultdict(lambda: default_pp_strategy),
    }

    # ── 4. Create World and run via load_pypsa ────────────────────────
    world = World(
        database_uri=db_uri,
        export_csv_path=export_csv,
        log_level="INFO",
    )

    load_pypsa(
        world=world,
        scenario="ieee9_testbed",
        study_case=market_mechanism,
        network=network,
        marketdesign=marketdesign,
        bidding_strategies=bidding_strategies,
    )

    logger.info("Starting simulation...")
    world.run()
    logger.info("Simulation complete.")

    return world


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Run with Egret SCED clearing (agents bid into nodal LP dispatch)
    world = run_testbed(
        market_mechanism="egret_sced",
        export_csv="./testbed_results",
    )
