# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Closed-form bidding heuristics for behavioral cloning warm-start.

Each heuristic maps market context (MC, forecast, capacity, demand, etc.)
to a target (bid_price, bid_volume) pair.  These are used to generate
supervised training data for the RL actor network before the RL loop begins.

Heuristics are registered in HEURISTIC_REGISTRY and selected via YAML:

    training:
      bc_warmstart:
        strategy: "forecast_markup"
        params: {}
        bc_steps: 500

All heuristic functions have the same signature:

    def heuristic(ctx: BidContext, **params) -> tuple[float, float]

    Returns (target_price, target_volume_fraction) where:
      - target_price is in $/MWh (raw, not scaled)
      - target_volume_fraction is in [0, 1] (fraction of max_power)
"""

from dataclasses import dataclass


@dataclass
class BidContext:
    """Market context available at bid time for a single generator and hour.

    All values are in physical units ($/MWh, MW), not scaled.
    The BC loop populates this from the unit's forecaster and network data.
    """

    mc: float                    # generator marginal cost ($/MWh)
    max_power: float             # generator nameplate capacity (MW)
    min_power: float             # generator minimum stable output (MW)
    forecast_price: float        # naive LMP forecast for this hour ($/MWh)
    total_demand: float          # total system demand this hour (MW)
    total_supply: float          # total available supply capacity (MW)
    own_capacity: float          # this generator's capacity (= max_power)
    competitor_mc: float         # MC of the next more expensive competitor ($/MWh)
    price_cap: float             # market price cap ($/MWh)
    hour: int                    # hour of day (0-23)
    cheaper_supply: float = 0.0  # competitor capacity priced below own MC (MW)


# ─── Forecast Markup (default) ─────────────────────────────────────────

def forecast_markup(ctx: BidContext, **_kw) -> tuple[float, float]:
    """Cournot markup (price) + residual demand (volume).

    Combines two dimensions of the optimal single-bid response from
    residual demand theory (Hortacsu & Puller 2008, Wolak 2003):

    **Price** — odds-ratio markup toward competitor MC:
        s_i = own_capacity / total_supply
        fraction = s_i / (1 - s_i)
        P* = MC + fraction * (competitor_mc - MC)

    **Volume** — offer residual demand (what the market needs from us):
        RD = total_demand - cheaper_supply
        volume_fraction = clamp(RD / max_power, 0, 1)

    Note: SFE theory (Holmberg 2008) predicts full-volume offers when
    generators submit multi-segment supply curves.  But this strategy
    submits a single (price, volume) bid, making it Cournot — where
    volume IS a strategic variable.  The RL agent will discover the
    true optimal volume through training; residual demand provides
    an informed starting point.

    When forecast < MC (off-peak), bid competitively: MC at full volume.

    For gen_0 (300 MW, MC=$35) at hour 12 (D=315 MW, cheaper=250 MW):
        Price: s=0.345, fraction=0.527, P* = $35 + 0.527*($50-$35) = $42.90
        Volume: RD = 315 - 250 = 65 MW → 65/300 = 21.7% → 65 MW
    """
    if ctx.forecast_price >= ctx.mc and ctx.competitor_mc > ctx.mc:
        # Price: Cournot odds-ratio markup
        s_i = ctx.own_capacity / max(ctx.total_supply, 1.0)
        fraction = min(1.0, s_i / max(1.0 - s_i, 0.01))
        bid_price = ctx.mc + fraction * (ctx.competitor_mc - ctx.mc)

        # Volume: residual demand — offer what the market needs from us
        residual_demand = max(ctx.total_demand - ctx.cheaper_supply, 0.0)
        volume_fraction = min(residual_demand / max(ctx.max_power, 1.0), 1.0)
    else:
        bid_price = ctx.mc
        volume_fraction = 1.0

    bid_price = max(ctx.mc, min(bid_price, ctx.price_cap))

    return bid_price, volume_fraction


# ─── Residual Demand Midpoint ───────────────────────────────────────────

def residual_demand_midpoint(
    ctx: BidContext, alpha: float = 1.0, **_kw,
) -> tuple[float, float]:
    """Bid halfway between MC and the counterfactual LMP.

    P* = MC + alpha * (LMP_counterfactual - MC) / 2

    The counterfactual LMP is the price that would prevail if this
    generator withdrew entirely.  Bidding at the midpoint captures
    half the available markup while staying well below the price
    that would trigger dispatch of the next competitor.

    This is the simplest strategy grounded in residual demand theory
    (Wolak 2003, Hortacsu & Puller 2008).

    Args:
        alpha: Market power factor in [0, 1].
               0 = bid at MC (competitive), 1 = full midpoint markup.
    """
    if ctx.forecast_price > ctx.mc:
        markup = alpha * (ctx.forecast_price - ctx.mc) / 2.0
        bid_price = ctx.mc + markup
    else:
        bid_price = ctx.mc

    bid_price = max(ctx.mc, min(bid_price, ctx.price_cap))
    volume_fraction = 1.0

    return bid_price, volume_fraction


# ─── Price-Cap Fraction ─────────────────────────────────────────────────

def price_cap_fraction(
    ctx: BidContext, fraction: float = 0.5, **_kw,
) -> tuple[float, float]:
    """Interpolate between own MC and competitor's MC.

    P* = MC + fraction * (MC_competitor - MC)

    The simplest possible markup rule.  fraction=0 is competitive,
    fraction=1 bids just at the competitor's cost (maximum safe markup).

    Args:
        fraction: Aggression parameter in [0, 1].
    """
    if ctx.competitor_mc > ctx.mc:
        bid_price = ctx.mc + fraction * (ctx.competitor_mc - ctx.mc)
    else:
        bid_price = ctx.mc

    bid_price = max(ctx.mc, min(bid_price, ctx.price_cap))
    volume_fraction = 1.0

    return bid_price, volume_fraction


# ─── Registry ───────────────────────────────────────────────────────────

HEURISTIC_REGISTRY: dict[str, callable] = {
    "forecast_markup": forecast_markup,
    "residual_demand_midpoint": residual_demand_midpoint,
    "price_cap_fraction": price_cap_fraction,
}


def get_heuristic(name: str) -> callable:
    """Look up a heuristic by name. Raises KeyError with available names."""
    if name not in HEURISTIC_REGISTRY:
        available = ", ".join(HEURISTIC_REGISTRY.keys())
        raise KeyError(
            f"Unknown heuristic '{name}'. Available: {available}"
        )
    return HEURISTIC_REGISTRY[name]
