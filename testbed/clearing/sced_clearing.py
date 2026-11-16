# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Egret-based Security-Constrained Economic Dispatch (SCED) clearing role.

Solves an LP (relaxed UC with all generators committed) using Egret's
solve_unit_commitment with relaxed=True. Extracts LMPs from dual variables
for nodal pricing.

Usage in MarketConfig:
    clearing_mechanisms["egret_sced"] = ScedClearingRole
    MarketConfig(
        market_mechanism="egret_sced",
        additional_fields=["node"],
        param_dict={
            "grid_data": grid_data_dict,
            "solver": "highs",
            "generator_params": {...},  # optional
        },
    )
"""

from __future__ import annotations

import logging

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.markets.base_market import MarketRole
from egret.models.unit_commitment import solve_unit_commitment

from .egret_adapter import model_results_to_orderbook, orders_to_model_data
from .highs_compat import patch_egret_for_highs

logger = logging.getLogger(__name__)

# Apply HiGHS compatibility patches at import time
patch_egret_for_highs()


class ScedClearingRole(MarketRole):
    """Egret-based SCED clearing as an ASSUME MarketRole.

    Wraps the Egret LP solver so that ASSUME agents can bid into a
    nodal SCED market through the standard World simulation loop.

    Register with:
        world.clearing_mechanisms["egret_sced"] = ScedClearingRole
    """

    required_fields = ["node"]

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)
        if not self.grid_data:
            raise ValueError(
                f"Market '{marketconfig.market_id}': grid_data is required for SCED clearing."
            )
        self.solver = marketconfig.param_dict.get("solver", "highs")
        self.generator_params = marketconfig.param_dict.get("generator_params", {})

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict], dict[tuple, float]]:
        return clear_sced(
            orderbook,
            market_products,
            self.grid_data,
            solver=self.solver,
            generator_params=self.generator_params,
        )


def clear_sced(
    orderbook: Orderbook,
    market_products: list,
    grid_data: dict,
    solver: str = "highs",
    generator_params: dict | None = None,
) -> tuple[Orderbook, Orderbook, list[dict], dict]:
    """Run Egret SCED clearing on an ASSUME orderbook.

    This is a standalone function that can be used directly or via ScedClearingRole.

    Args:
        orderbook: ASSUME orderbook (list of Order dicts).
        market_products: List of (start, end, only_hours) tuples.
        grid_data: Dict with 'buses', 'lines' DataFrames.
        solver: Pyomo solver name (default "highs").
        generator_params: Optional Egret-specific generator parameters.

    Returns:
        (accepted_orders, rejected_orders, meta_list, flows_dict)
    """
    # 1. Convert ASSUME orders + grid → Egret ModelData
    model_data, order_gen_map = orders_to_model_data(
        orderbook,
        market_products,
        grid_data,
        generator_params=generator_params,
        fix_commitment=True,  # SCED: all units assumed online
    )

    # 2. Solve LP (continuous dispatch + duals for LMPs)
    logger.info("Solving Egret SCED (relaxed=True)...")
    md_results = solve_unit_commitment(model_data, solver, relaxed=True)

    total_cost = md_results.data["system"].get("total_cost", "N/A")
    logger.info(f"Egret SCED solved. Total cost: {total_cost}")

    # 3. Map Egret results back to ASSUME format
    return model_results_to_orderbook(
        md_results, orderbook, market_products, order_gen_map=order_gen_map,
    )
