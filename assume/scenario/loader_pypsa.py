# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta

import pandas as pd
import pypsa
from dateutil import rrule as rr

from assume import World
from assume.common.forecaster import (
    DemandForecaster,
    PowerplantForecaster,
    UnitForecaster,
)
from assume.common.market_objects import MarketConfig, MarketProduct

logger = logging.getLogger(__name__)


def load_pypsa(
    world: World,
    scenario: str,
    study_case: str,
    network: pypsa.Network,
    marketdesign: list[MarketConfig],
    bidding_strategies: dict[str, dict[str, str]],
    save_frequency_hours: int = 4,
):
    """
    This initializes a scenario from the given pypsa grid.
    One can load a grid from pypsa `import_from_csv_folder`, adjust its properties and add it to this function to create an energy market scenario from it.
    This is also compatible with netCDF, PyPower, PandaPower and HDF5 pypsa-compatible datasets.

    Args:
        world (World): the world to add this scenario to
        scenario (str): scenario name
        study_case (str): study case name
        network (pypsa.Network): pypsa Network from which the simulation properties and timeseries data is extracted
        marketdesign (list[MarketConfig]): description of the market design which will be used with the scenario

    Notes:
        - If a time-varying `generators_t.marginal_cost` table is present, per-generator
          marginal cost series are attached to `PowerplantForecaster.fuel_prices`.
          Otherwise the static `generator.marginal_cost` scalar is used.
    """
    index = network.snapshots
    # Handle both MultiIndex and DatetimeIndex
    if isinstance(index, pd.MultiIndex):
        datetime_index = pd.DatetimeIndex(index.get_level_values(1))
    else:
        datetime_index = pd.DatetimeIndex(index)
    
    # Infer frequency from the datetime index
    freq = pd.infer_freq(datetime_index)
    if freq is None:
        # If frequency cannot be inferred, calculate it from the time differences
        if len(datetime_index) > 1:
            time_diffs = datetime_index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                # Get the most common time difference
                most_common_diff = time_diffs.value_counts().index[0]
                
                # Map common Timedeltas to their frequency strings
                # These values come from pd.infer_freq() output
                timedelta_to_freq = {
                    pd.Timedelta(minutes=1): "min",
                    pd.Timedelta(minutes=15): "15min",
                    pd.Timedelta(minutes=30): "30min",
                    pd.Timedelta(hours=1): "h",
                    pd.Timedelta(days=1): "D",
                }
                
                freq_str = timedelta_to_freq.get(most_common_diff)
                
                if freq_str:
                    try:
                        datetime_index = pd.date_range(
                            start=datetime_index[0],
                            end=datetime_index[-1],
                            freq=freq_str
                        )
                    except Exception as e:
                        logger.warning(f"Could not create regular DatetimeIndex with frequency {freq_str}: {e}")
                else:
                    logger.warning(f"Could not determine standard frequency from time difference {most_common_diff}")
            else:
                raise ValueError("Cannot infer frequency from snapshots and insufficient data points")
        else:
            raise ValueError("Cannot infer frequency from snapshots and insufficient data points")
    else:
        # If frequency was successfully inferred, recreate with explicit freq to ensure it's set
        datetime_index = pd.date_range(
            start=datetime_index[0],
            end=datetime_index[-1],
            freq=freq
        )
    
    start = datetime_index[0]
    end = datetime_index[-1]
    simulation_id = f"{scenario}_{study_case}"
    logger.info(f"loading scenario {simulation_id}")

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_frequency_hours,
        simulation_id=simulation_id,
    )
    # setup eom market

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)

    network.generators.rename(
        columns={"bus": "node", "p_nom": "max_power"}, inplace=True
    )
    network.loads.rename(columns={"bus": "node", "p_set": "min_power"}, inplace=True)
    if "max_power" not in network.loads.columns:
        network.loads["max_power"] = 0
    grid_data = {
        "buses": network.buses,
        "lines": network.lines,
        "generators": network.generators,
        "loads": network.loads,
    }

    for market_config in marketdesign:
        market_config.param_dict["grid_data"] = grid_data
        world.add_market(mo_id, market_config)

    world.add_unit_operator("powerplant_operator")
    for _, generator in network.generators.iterrows():
        # Skip generators marked as "load" - they represent demand and are handled separately
        # in the loads section below. Including them would create duplicate demand.
        if hasattr(generator, 'carrier') and generator.carrier == 'load':
            continue
        
        if generator.name in network.generators_t["p_max_pu"].columns:
            av = network.generators_t["p_max_pu"][generator.name]
            # Reindex to match datetime_index (handles MultiIndex from PyPSA netCDF)
            if isinstance(av.index, pd.MultiIndex):
                av = av.reset_index(level=0, drop=True)
            av = av.reindex(datetime_index)
        else:
            av = 1

        # Retrieve marginal cost: prefer time-varying series if available in generators_t.marginal_cost
        # Otherwise derive synthetic cost from heat_rate * fuel_cost + variable_cost if attributes present
        # Final fallback to scalar generator.marginal_cost
        fuel_price_obj = generator.marginal_cost
        try:
            if hasattr(network.generators_t, "marginal_cost") and generator.name in network.generators_t.marginal_cost.columns:
                mc_series = network.generators_t.marginal_cost[generator.name]
                if isinstance(mc_series.index, pd.MultiIndex):
                    mc_series = mc_series.reset_index(level=0, drop=True)
                mc_series = mc_series.reindex(datetime_index)
                # Use the series only if it contains more than one timestamp and has at least one non-null value
                if len(mc_series) > 1 and not mc_series.isnull().all():
                    fuel_price_obj = mc_series
                    logger.debug(f"Using time-varying marginal_cost series for generator {generator.name}")
        except Exception as e:
            logger.warning(f"Failed to attach time-varying marginal_cost for {generator.name}: {e}. Falling back to scalar.")
        
        # If still scalar/zero and static attributes available, synthesize marginal cost
        if not isinstance(fuel_price_obj, pd.Series) or (isinstance(fuel_price_obj, pd.Series) and fuel_price_obj.sum() == 0):
            # Attempt to derive from heat_rate, fuel_cost, variable_cost
            synthetic_cost = 0.0
            try:
                # Heat rate (thermal efficiency inverse) and fuel cost
                heat_rate = getattr(generator, 'heat_rate', None)
                fuel_cost = getattr(generator, 'fuel_cost', None)
                if heat_rate is not None and fuel_cost is not None:
                    synthetic_cost += float(heat_rate) * float(fuel_cost)
                # Add variable cost if present
                variable_cost = getattr(generator, 'variable_cost', None)
                if variable_cost is not None:
                    synthetic_cost += float(variable_cost)
                # If synthetic cost derived, broadcast to time series
                if synthetic_cost > 0:
                    fuel_price_obj = pd.Series(synthetic_cost, index=datetime_index)
                    logger.debug(f"Derived synthetic marginal_cost={synthetic_cost:.2f} for {generator.name} from heat_rate/fuel_cost/variable_cost")
            except Exception as e:
                logger.debug(f"Could not derive synthetic cost for {generator.name}: {e}")

        unit_type = "power_plant"

        max_power = generator.max_power or 1000
        # if p_nom is not set, generator.p_nom_extendable must be
        ramp_up = generator.ramp_limit_start_up * max_power
        ramp_down = generator.ramp_limit_shut_down * max_power
        
        # Ensure min_operating_time and min_down_time are > 0
        # ASSUME requires these to be positive values
        min_operating_time = generator.min_up_time if generator.min_up_time and generator.min_up_time > 0 else 1
        min_down_time = generator.min_down_time if generator.min_down_time and generator.min_down_time > 0 else 1
        
        # Clamp min_power to max_power to avoid floating-point precision issues
        min_power = min(generator.p_nom_min, max_power)
        
        # Build fuel_prices mapping with both fuel type and generator-specific key
        fuel_prices_mapping = {}
        # Attach fuel type key if present
        if getattr(generator, "carrier", None):
            fuel_prices_mapping[generator.carrier] = fuel_price_obj
        # Generator-specific key always added to enable per-generator export even when carrier missing
        if isinstance(fuel_price_obj, pd.Series):
            gen_series = fuel_price_obj.reindex(datetime_index)
        else:
            # Broadcast scalar to full datetime_index
            try:
                scalar_val = float(fuel_price_obj)
            except Exception:
                scalar_val = 0.0
            gen_series = pd.Series(scalar_val, index=datetime_index)
        fuel_prices_mapping[generator.name] = gen_series

        world.add_unit(
            generator.name,
            unit_type,
            "powerplant_operator",
            {
                "min_power": min_power,
                "max_power": max_power,
                "bidding_strategies": bidding_strategies[unit_type][generator.name],
                "technology": "conventional",
                "node": generator.node,
                "efficiency": 1,  # do not use generator.efficiency as it is respected in marginal_cost,
                "fuel_type": generator.carrier,
                "ramp_up": ramp_up,
                "ramp_down": ramp_down,
                "min_operating_time": min_operating_time,
                "min_down_time": min_down_time,
            },
            PowerplantForecaster(
                datetime_index,
                # Provide both per-fuel and per-generator keys.
                fuel_prices=fuel_prices_mapping,
                availability=av,
            ),
        )

    world.add_unit_operator("demand_operator")
    for _, load in network.loads.iterrows():
        if load.name not in network.loads_t["p_set"].columns:
            # we have no load
            continue

        load_t = network.loads_t["p_set"][load.name]
        unit_type = "demand"

        # Reindex to match datetime_index (handles MultiIndex from PyPSA netCDF)
        if isinstance(load_t.index, pd.MultiIndex):
            load_t = load_t.reset_index(level=0, drop=True)
        load_t = load_t.reindex(datetime_index)

        # Negate load values: ASSUME requires demand to be negative (consumption)
        load_t_negated = -load_t

        world.add_unit(
            load.name,
            unit_type,
            "demand_operator",
            {
                "min_power": 0,
                "max_power": -load_t.max(),
                "bidding_strategies": bidding_strategies[unit_type][load.name],
                "technology": "demand",
                "node": load.node,
                "price": 1e3,
            },
            DemandForecaster(datetime_index, demand=load_t_negated),
        )

    # Note: Storage units require MinMaxChargeStrategy-based bidding strategies
    # We provide a default storage strategy (flexableEOMStorage) but users can override
    # by passing custom bidding_strategies for the "storage" unit type
    world.add_unit_operator("storage_operator")
    for _, storage_unit in network.storage_units.iterrows():
        # Skip expandable storage units (p_nom == 0)
        if storage_unit.p_nom == 0:
            continue
        
        # Reindex timeseries if available
        if storage_unit.name in network.storage_units_t["p_set"].columns:
            storage_t = network.storage_units_t["p_set"][storage_unit.name]
            # Reindex to match datetime_index (handles MultiIndex from PyPSA netCDF)
            if isinstance(storage_t.index, pd.MultiIndex):
                storage_t = storage_t.reset_index(level=0, drop=True)
            storage_t = storage_t.reindex(datetime_index)

        unit_type = "storage"
        max_power_charge = storage_unit.p_nom * storage_unit.p_min_pu
        max_power_discharge = storage_unit.p_nom * storage_unit.p_max_pu

        # Use provided bidding strategies or default to flexableEOMStorage
        storage_strategies = bidding_strategies[unit_type].get(
            storage_unit.name,
            {mc.market_id: "flexable_eom_storage" for mc in marketdesign}
        )

        # Determine storage technology from carrier if available, otherwise default to battery
        storage_tech = getattr(storage_unit, 'carrier', 'battery') or 'battery'

        world.add_unit(
            f"StorageTrader_{storage_unit.name}",
            unit_type,
            "storage_operator",
            {
                "max_power_charge": max_power_charge,
                "max_power_discharge": max_power_discharge,
                "efficiency_charge": storage_unit.efficiency_store,
                "efficiency_discharge": storage_unit.efficiency_dispatch,
                "initial_soc": storage_unit.state_of_charge_initial,
                "capacity": storage_unit.p_nom * storage_unit.max_hours,
                "bidding_strategies": storage_strategies,
                "technology": storage_tech,
                "emission_factor": getattr(storage_unit, 'emission_factor', 0) or 0,
                "node": storage_unit.bus,
            },
            UnitForecaster(datetime_index),
        )

    # Populate forecaster.forecasts for demand, availability, and prices
    # This enables export_world_to_csv to export time series data without helper cells
    for uo in world.unit_operators.values():
        for uid, unit in uo.units.items():
            f = getattr(unit, "forecaster", None)
            if f is None:
                continue
            # Skip if already populated
            if hasattr(f, "forecasts") and f.forecasts:
                continue
            forecasts = {}
            # Demand forecaster provides 'demand' attribute
            if hasattr(f, "demand"):
                forecasts["demand"] = f.demand
                forecasts["energy"] = f.demand  # Export compatibility
            # Availability for powerplants
            if hasattr(f, "availability"):
                forecasts["availability"] = f.availability
                forecasts[f"availability_{uid}"] = f.availability
            # Price series keyed by market
            if hasattr(f, "price") and isinstance(f.price, dict):
                for mk, series in f.price.items():
                    forecasts[f"price_{mk}"] = series
            # Exchange import/export
            if hasattr(f, "volume_export"):
                forecasts[f"{uid}_export"] = f.volume_export
            if hasattr(f, "volume_import"):
                forecasts[f"{uid}_import"] = f.volume_import
            if forecasts:
                f.forecasts = forecasts
                logger.debug(f"Populated forecasts for {uid}: {list(forecasts.keys())}")


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    scenario = "world_pypsa"
    study_case = "scigrid_de"
    # "pay_as_clear", "redispatch" or "nodal"
    market_mechanism = "complex_clearing"

    match study_case:
        case "ac_dc_meshed":
            network = pypsa.examples.ac_dc_meshed()
        case "scigrid_de":
            network = pypsa.examples.scigrid_de()
        case "storage_hvdc":
            network = pypsa.examples.storage_hvdc()
        case _:
            logger.info(f"invalid studycase: {study_case}")
            network = pd.DataFrame()

    study_case = f"{study_case}_{market_mechanism}"

    start = network.snapshots[0]
    end = network.snapshots[-1]
    marketdesign = [
        MarketConfig(
            "EOM",
            rr.rrule(rr.HOURLY, interval=1, dtstart=start, until=end),
            timedelta(hours=1),
            market_mechanism,
            [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
            additional_fields=["node", "max_power", "min_power", "bid_type"],
            maximum_bid_volume=1e9,
            maximum_bid_price=1e9,
            param_dict={"log_flows": True},
        )
    ]
    if market_mechanism == "redispatch":
        marketdesign.append(
            MarketConfig(
                "EOM",
                rr.rrule(
                    rr.HOURLY,
                    interval=1,
                    dtstart=start - timedelta(hours=0.5),
                    until=end,
                ),
                timedelta(hours=0.25),
                "pay_as_clear",
                [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1.5))],
                additional_fields=["node", "max_power", "min_power"],
                maximum_bid_volume=1e9,
                maximum_bid_price=1e9,
            )
        )
    default_strategies = {
        mc.market_id: (
            "powerplant_energy_naive_redispatch"
            if mc.market_mechanism == "redispatch"
            else "demand_energy_naive"
        )
        for mc in marketdesign
    }
    from collections import defaultdict

    bidding_strategies = {
        "power_plant": defaultdict(lambda: default_strategies),
        "demand": defaultdict(
            lambda: {mc.market_id: "demand_energy_naive" for mc in marketdesign}
        ),
        "storage": defaultdict(lambda: default_strategies),
    }

    load_pypsa(world, scenario, study_case, network, marketdesign, bidding_strategies)
    world.run()
