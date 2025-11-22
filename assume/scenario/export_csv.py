# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Export module for ASSUME framework.

This module provides functions to export simulation data from a World instance
back to CSV files in the format expected by loader_csv.py. It mirrors the structure
of the loader to ensure round-trip compatibility.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from assume.strategies import bidding_strategies as registered_strategies
from assume.units import Demand, PowerPlant, Storage
from assume.world import World

logger = logging.getLogger(__name__)


def create_strategy_class_to_key_mapping() -> dict[str, str]:
    """
    Create a reverse mapping from strategy class names to their registered keys.
    
    This maps class names (e.g., "NaiveSingleBidStrategy") to their registered keys 
    (e.g., "naive_eom") used in the bidding_strategies registry.
    
    Returns:
        dict[str, str]: Mapping from class name to strategy key.
    """
    strategy_class_to_key = {}
    for key, strategy_class in registered_strategies.items():
        class_name = strategy_class.__name__
        # If multiple keys map to same class, prefer the first one found
        if class_name not in strategy_class_to_key:
            strategy_class_to_key[class_name] = key
    
    logger.debug(f"Created strategy mapping with {len(strategy_class_to_key)} entries")
    return strategy_class_to_key


def export_unit_operators(
    world: World,
    output_dir: str | Path,
) -> pd.DataFrame | None:
    """
    Export unit operators to CSV.
    
    Args:
        world: The World instance containing unit operators.
        output_dir: Directory to save the CSV file.
        
    Returns:
        DataFrame of unit operators, or None if no operators exist.
    """
    if not world.unit_operators:
        logger.info("No unit operators to export")
        return None
    
    operators_data = []
    for operator_id, unit_operator in world.unit_operators.items():
        operator_dict = {
            "name": operator_id,
            # Add other operator-level attributes as needed
        }
        operators_data.append(operator_dict)
    
    if not operators_data:
        return None
    
    df = pd.DataFrame(operators_data)
    df.set_index("name", inplace=True)
    
    filepath = Path(output_dir) / "unit_operators.csv"
    df.to_csv(filepath)
    logger.info(f"✅ Exported unit_operators.csv ({len(df)} operators)")
    
    return df


def export_powerplant_units(
    world: World,
    output_dir: str | Path,
    strategy_mapping: dict[str, str],
) -> pd.DataFrame | None:
    """
    Export power plant units to CSV.
    
    Extracts PowerPlant units from all unit operators and exports them with
    all relevant attributes in the format expected by loader_csv.py.
    
    Args:
        world: The World instance containing units.
        output_dir: Directory to save the CSV file.
        strategy_mapping: Mapping from strategy class names to registered keys.
        
    Returns:
        DataFrame of powerplant units, or None if no units exist.
    """
    units_data = []
    
    for operator_id, unit_operator in world.unit_operators.items():
        for unit in unit_operator.units.values():
            # Only export PowerPlant units
            if not isinstance(unit, PowerPlant):
                continue
            
            unit_dict = {
                "name": unit.id,
                "unit_operator": operator_id,
                "technology": getattr(unit, "technology", None),
                "node": getattr(unit, "node", None),
                "max_power": getattr(unit, "max_power", None),
                "min_power": getattr(unit, "min_power", None),
                "efficiency": getattr(unit, "efficiency", None),
                "fuel_type": getattr(unit, "fuel_type", None),
                "emission_factor": getattr(unit, "emission_factor", None),
                "fixed_cost": getattr(unit, "fixed_cost", None),
                "variable_cost": getattr(unit, "variable_cost", None),
                "additional_cost": getattr(unit, "additional_cost", None),
                "cold_start_cost": getattr(unit, "cold_start_cost", None),
                "hot_start_cost": getattr(unit, "hot_start_cost", None),
                "warm_start_cost": getattr(unit, "warm_start_cost", None),
                "min_operating_time": getattr(unit, "min_operating_time", None),
                "min_down_time": getattr(unit, "min_down_time", None),
                "ramp_up": getattr(unit, "ramp_up", None),
                "ramp_down": getattr(unit, "ramp_down", None),
            }
            
            # Add bidding strategies for all markets this unit participates in
            if hasattr(unit, "bidding_strategies"):
                for market_id, strategy_instance in unit.bidding_strategies.items():
                    class_name = strategy_instance.__class__.__name__
                    strategy_key = strategy_mapping.get(class_name, class_name)
                    unit_dict[f"bidding_{market_id}"] = strategy_key
            
            units_data.append(unit_dict)
    
    if not units_data:
        logger.info("No powerplant units to export")
        return None
    
    df = pd.DataFrame(units_data)
    df.set_index("name", inplace=True)
    
    # Remove columns that are all NaN
    df = df.dropna(axis=1, how="all")
    
    filepath = Path(output_dir) / "powerplant_units.csv"
    df.to_csv(filepath)
    logger.info(f"✅ Exported powerplant_units.csv ({len(df)} units)")
    
    return df


def export_storage_units(
    world: World,
    output_dir: str | Path,
    strategy_mapping: dict[str, str],
) -> pd.DataFrame | None:
    """
    Export storage units to CSV.
    
    Args:
        world: The World instance containing units.
        output_dir: Directory to save the CSV file.
        strategy_mapping: Mapping from strategy class names to registered keys.
        
    Returns:
        DataFrame of storage units, or None if no units exist.
    """
    units_data = []
    
    for operator_id, unit_operator in world.unit_operators.items():
        for unit in unit_operator.units.values():
            # Only export Storage units
            if not isinstance(unit, Storage):
                continue
            
            unit_dict = {
                "name": unit.id,
                "unit_operator": operator_id,
                "technology": getattr(unit, "technology", None),
                "node": getattr(unit, "node", None),
                "max_power_charge": getattr(unit, "max_power_charge", None),
                "max_power_discharge": getattr(unit, "max_power_discharge", None),
                "efficiency_charge": getattr(unit, "efficiency_charge", None),
                "efficiency_discharge": getattr(unit, "efficiency_discharge", None),
                "max_soc": getattr(unit, "max_soc", None),
                "min_soc": getattr(unit, "min_soc", None),
                "initial_soc": getattr(unit, "initial_soc", None),
                "fixed_cost": getattr(unit, "fixed_cost", None),
                "additional_cost": getattr(unit, "additional_cost", None),
            }
            
            # Note: loader_csv expects negative values for charging
            if unit_dict["max_power_charge"] is not None:
                unit_dict["max_power_charge"] = -abs(unit_dict["max_power_charge"])
            if "min_power_charge" in dir(unit):
                unit_dict["min_power_charge"] = -abs(getattr(unit, "min_power_charge", 0))
            
            # Add bidding strategies
            if hasattr(unit, "bidding_strategies"):
                for market_id, strategy_instance in unit.bidding_strategies.items():
                    class_name = strategy_instance.__class__.__name__
                    strategy_key = strategy_mapping.get(class_name, class_name)
                    unit_dict[f"bidding_{market_id}"] = strategy_key
            
            units_data.append(unit_dict)
    
    if not units_data:
        logger.info("No storage units to export")
        return None
    
    df = pd.DataFrame(units_data)
    df.set_index("name", inplace=True)
    
    # Remove columns that are all NaN
    df = df.dropna(axis=1, how="all")
    
    filepath = Path(output_dir) / "storage_units.csv"
    df.to_csv(filepath)
    logger.info(f"✅ Exported storage_units.csv ({len(df)} units)")
    
    return df


def export_demand_units(
    world: World,
    output_dir: str | Path,
    strategy_mapping: dict[str, str],
) -> pd.DataFrame | None:
    """
    Export demand units to CSV.
    
    Args:
        world: The World instance containing units.
        output_dir: Directory to save the CSV file.
        strategy_mapping: Mapping from strategy class names to registered keys.
        
    Returns:
        DataFrame of demand units, or None if no units exist.
    """
    units_data = []
    
    for operator_id, unit_operator in world.unit_operators.items():
        for unit in unit_operator.units.values():
            # Only export Demand units
            if not isinstance(unit, Demand):
                continue
            
            unit_dict = {
                "name": unit.id,
                "unit_operator": operator_id,
                "technology": getattr(unit, "technology", "demand"),
                "node": getattr(unit, "node", None),
                "max_power": getattr(unit, "max_power", None),
                "min_power": getattr(unit, "min_power", None),
            }
            
            # Ensure negative values for demand (loader expects this)
            if unit_dict["max_power"] is not None:
                unit_dict["max_power"] = -abs(unit_dict["max_power"])
            if unit_dict["min_power"] is not None:
                unit_dict["min_power"] = -abs(unit_dict["min_power"])
            
            # Add bidding strategies
            if hasattr(unit, "bidding_strategies"):
                for market_id, strategy_instance in unit.bidding_strategies.items():
                    class_name = strategy_instance.__class__.__name__
                    strategy_key = strategy_mapping.get(class_name, class_name)
                    unit_dict[f"bidding_{market_id}"] = strategy_key
            
            units_data.append(unit_dict)
    
    if not units_data:
        logger.info("No demand units to export")
        return None
    
    df = pd.DataFrame(units_data)
    df.set_index("name", inplace=True)
    
    # Remove columns that are all NaN
    df = df.dropna(axis=1, how="all")
    
    filepath = Path(output_dir) / "demand_units.csv"
    df.to_csv(filepath)
    logger.info(f"✅ Exported demand_units.csv ({len(df)} units)")
    
    return df


def export_fuel_prices(
    world: World,
    output_dir: str | Path,
    generator_time_series: bool = True,
) -> pd.DataFrame | None:
    """
    Export fuel prices to CSV.

    Behaviour:
        1. If any fuel price is provided as a time series (length > 1), a time-indexed
           DataFrame is written with one row per simulation timestep and columns per fuel type.
           When multiple units provide a time series for the same fuel, their prices are averaged
           per timestep. Scalar prices are broadcast across all timesteps.
        2. If only scalar prices are available (legacy behaviour), a single-row DataFrame is
           written preserving backward compatibility with existing loader expectations.
        3. A 'co2' column is ensured (defaults to 0.0 when absent) for marginal cost calculations.

    Args:
        world: The World instance containing units with forecasters.
        output_dir: Directory to save the CSV file.

    Returns:
        DataFrame of fuel prices (time-indexed or single-row), or None if no fuel prices exist.
    """
    # Collect raw fuel price objects (scalar or Series) per fuel type
    fuel_price_objects: dict[str, list] = {}
    has_timeseries = False

    for _, unit_operator in world.unit_operators.items():
        for _, unit in unit_operator.units.items():
            if hasattr(unit, "forecaster") and hasattr(unit.forecaster, "fuel_prices"):
                for fuel_type, price_obj in unit.forecaster.fuel_prices.items():
                    fuel_price_objects.setdefault(fuel_type, []).append(price_obj)
                    # Detect presence of any time series (length > 1)
                    if hasattr(price_obj, "__len__") and not isinstance(price_obj, str):
                        try:
                            if len(price_obj) > 1:
                                has_timeseries = True
                        except Exception:
                            pass

    if not fuel_price_objects:
        logger.info("No fuel prices to export")
        return None

    # Ensure co2 presence (default 0.0 if absent)
    if "co2" not in fuel_price_objects:
        fuel_price_objects["co2"] = [0.0]

    if has_timeseries and hasattr(world, "index") and world.index is not None:
        # Time-indexed export
        export_index = world.index
        data = {}
        for fuel_type, objs in fuel_price_objects.items():
            series_list = []
            for obj in objs:
                # Convert to pandas Series indexed by export_index
                if isinstance(obj, pd.Series):
                    s = obj.copy()
                else:
                    # Scalar: broadcast across index
                    try:
                        scalar_val = float(obj)
                    except Exception:
                        scalar_val = 0.0
                    s = pd.Series(scalar_val, index=export_index)
                # Align MultiIndex if needed
                if isinstance(s.index, pd.MultiIndex):
                    s = s.reset_index(level=0, drop=True)
                s = s.reindex(export_index)
                series_list.append(s)
            # Average across all units for this fuel type
            if len(series_list) == 1:
                data[fuel_type] = series_list[0]
            else:
                df_tmp = pd.concat(series_list, axis=1)
                data[fuel_type] = df_tmp.mean(axis=1)
        df = pd.DataFrame(data, index=export_index)
        # Reorder columns: alphabetical with co2 last
        col_order = sorted([c for c in df.columns if c != "co2"]) + ["co2"]
        df = df[col_order]
        filepath = Path(output_dir) / "fuel_prices_df.csv"
        df.to_csv(filepath)
        logger.info(
            f"✅ Exported fuel_prices_df.csv as time series ({len(df)} timesteps × {len(df.columns)} fuel types)"
        )
        return df

    # Legacy single-row export (only scalar prices present)
    scalar_export: dict[str, float] = {}
    for fuel_type, objs in fuel_price_objects.items():
        # Average scalars / lists of scalars
        values = []
        for obj in objs:
            if isinstance(obj, pd.Series):
                # Treat single-value series or take mean
                if len(obj) == 1:
                    try:
                        values.append(float(obj.item()))
                    except Exception:
                        values.append(float(obj.iloc[0]))
                else:
                    values.extend([float(v) for v in obj.values if v is not None])
            else:
                try:
                    values.append(float(obj))
                except Exception:
                    if hasattr(obj, "value"):
                        values.append(float(obj.value))
        if values:
            scalar_export[fuel_type] = sum(values) / len(values)
    # Ensure co2 last
    if "co2" not in scalar_export:
        scalar_export["co2"] = 0.0
    df = pd.DataFrame([scalar_export])
    col_order = sorted([c for c in df.columns if c != "co2"]) + ["co2"]
    df = df[col_order]
    filepath = Path(output_dir) / "fuel_prices_df.csv"
    df.to_csv(filepath, index=False)
    logger.info(f"✅ Exported fuel_prices_df.csv (single row, {len(df.columns)} fuel types)")
    return df


def export_generator_fuel_prices(
    world: World,
    output_dir: str | Path,
) -> pd.DataFrame | None:
    """
    Export per-generator fuel price time series (one column per generator).

    Logic:
        - Only exports fuel prices for PowerPlant units (not demand, storage, or renewables without fuel costs)
        - Prefer a generator-specific key (unit.id) in forecaster.fuel_prices.
        - Fallback to fuel_type key if generator key missing.
        - Scalars are broadcast across the full simulation index to always produce
          a multi-row DataFrame (enabling time series export even without true variation).
        - Skip units that don't have meaningful fuel price data (renewables, storage, demand)

    Returns:
        DataFrame indexed by world.index (or derived hourly range) with generator columns.
    """
    if not hasattr(world, "unit_operators") or not world.unit_operators:
        logger.info("No unit operators present - generator-level fuel prices not exported")
        return None

    # Determine export index
    if hasattr(world, "index") and world.index is not None:
        export_index = world.index
    else:
        try:
            export_index = pd.date_range(start=world.start, end=world.end, freq="h")
        except Exception:
            logger.warning("Could not derive export index for generator-level fuel prices")
            return None

    from assume.units import PowerPlant
    
    data: dict[str, pd.Series] = {}
    for _, uo in world.unit_operators.items():
        for uid, unit in uo.units.items():
            # Only export fuel prices for PowerPlant units
            # Skip demand, storage, and units without fuel costs
            if not isinstance(unit, PowerPlant):
                continue
            
            # Skip renewables (they typically don't have fuel costs)
            fuel_type = getattr(unit, "fuel_type", None)
            if fuel_type in ["wind", "onwind", "offwind", "solar", "hydro"]:
                continue
                
            f = getattr(unit, "forecaster", None)
            price_obj = None
            if f is not None and hasattr(f, "fuel_prices") and isinstance(f.fuel_prices, dict):
                # Prefer generator-specific key
                if uid in f.fuel_prices:
                    price_obj = f.fuel_prices[uid]
                else:
                    fuel_key = getattr(unit, "fuel_type", None)
                    if fuel_key and fuel_key in f.fuel_prices:
                        price_obj = f.fuel_prices[fuel_key]
                    # Don't use fallback to first available key - this can cause wrong mappings
            
            # Attribute fallbacks for units that don't have forecaster fuel prices
            if price_obj is None:
                for attr in ("variable_cost", "marginal_cost", "fuel_cost"):
                    val = getattr(unit, attr, None)
                    if val is not None and val > 0:
                        price_obj = float(val)
                        break
            
            # Skip units without meaningful fuel price data
            if price_obj is None:
                continue
                
            # Construct series
            # Handle FastSeries, pd.Series, or scalars
            if isinstance(price_obj, pd.Series):
                series = price_obj.copy()
                if isinstance(series.index, pd.MultiIndex):
                    series = series.reset_index(level=0, drop=True)
                series = series.reindex(export_index)
            elif hasattr(price_obj, '__len__') and not isinstance(price_obj, (str, bytes)):
                # FastSeries or array-like: convert to pd.Series
                try:
                    series = pd.Series(price_obj, index=export_index)
                    # If index mismatch, try extracting values
                    if len(series) != len(export_index):
                        series = pd.Series(list(price_obj), index=export_index)
                except Exception:
                    # Fallback: try to get mean value and broadcast
                    try:
                        mean_val = float(pd.Series(price_obj).mean())
                        series = pd.Series(mean_val, index=export_index)
                    except Exception:
                        series = pd.Series(0.0, index=export_index)
            else:
                # Scalar: broadcast to full index
                try:
                    scalar_val = float(price_obj)
                except Exception:
                    scalar_val = 0.0
                series = pd.Series(scalar_val, index=export_index)
            data[uid] = series.astype(float)

    # Also export fuel-type columns as fallbacks for units without per-generator prices
    # Collect unique fuel types
    fuel_types = set()
    for uo in world.unit_operators.values():
        for uid, unit in uo.units.items():
            if not isinstance(unit, PowerPlant):
                continue
            fuel_type = getattr(unit, "fuel_type", None)
            if fuel_type and fuel_type not in ["wind", "onwind", "offwind", "solar", "hydro"]:
                fuel_types.add(fuel_type)
    
    # For each fuel type, try to get a representative price series
    for fuel_type in fuel_types:
        if fuel_type in data:  # Already have a column with this name
            continue
            
        # Find a unit with this fuel_type that has a price series
        for uo in world.unit_operators.values():
            for uid, unit in uo.units.items():
                if not isinstance(unit, PowerPlant):
                    continue
                if getattr(unit, "fuel_type", None) != fuel_type:
                    continue
                    
                f = getattr(unit, "forecaster", None)
                if f is not None and hasattr(f, "fuel_prices") and isinstance(f.fuel_prices, dict):
                    if fuel_type in f.fuel_prices:
                        price_obj = f.fuel_prices[fuel_type]
                        # Convert to series
                        if isinstance(price_obj, pd.Series):
                            series = price_obj.copy()
                            if isinstance(series.index, pd.MultiIndex):
                                series = series.reset_index(level=0, drop=True)
                            series = series.reindex(export_index)
                        elif hasattr(price_obj, '__len__') and not isinstance(price_obj, (str, bytes)):
                            try:
                                series = pd.Series(list(price_obj), index=export_index)
                            except Exception:
                                series = pd.Series(0.0, index=export_index)
                        else:
                            try:
                                series = pd.Series(float(price_obj), index=export_index)
                            except Exception:
                                series = pd.Series(0.0, index=export_index)
                        data[fuel_type] = series.astype(float)
                        break
            if fuel_type in data:
                break

    if not data:
        logger.info("No per-generator or fuel-type price series could be constructed")
        return None

    df = pd.DataFrame(data, index=export_index)
    filepath = Path(output_dir) / "fuel_prices_df.csv"
    df.to_csv(filepath)
    logger.info(
        f"✅ Exported fuel_prices_df.csv as per-generator time series ({len(df)} timesteps × {len(df.columns)} PowerPlant units with fuel costs)"
    )
    return df


def export_grid_data(
    world: World,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame | None]:
    """
    Export grid/network data (buses and lines) to CSV files.
    
    Extracts grid data from market configurations if available (e.g., from PyPSA networks).
    
    Args:
        world: The World instance containing markets with grid data.
        output_dir: Directory to save the CSV files.
        
    Returns:
        Dictionary with 'buses' and 'lines' DataFrames, or None if no grid data exists.
    """
    exported_grids = {
        "buses": None,
        "lines": None,
    }
    
    # Check if any market has grid_data in its param_dict
    for market_id, market_config in world.markets.items():
        if hasattr(market_config, "param_dict") and "grid_data" in market_config.param_dict:
            grid_data = market_config.param_dict["grid_data"]
            
            # Export buses
            if "buses" in grid_data and grid_data["buses"] is not None:
                buses_df = grid_data["buses"]
                if not buses_df.empty:
                    filepath = Path(output_dir) / "buses.csv"
                    buses_df.to_csv(filepath)
                    logger.info(f"✅ Exported buses.csv ({len(buses_df)} buses)")
                    exported_grids["buses"] = buses_df
            
            # Export lines
            if "lines" in grid_data and grid_data["lines"] is not None:
                lines_df = grid_data["lines"]
                if not lines_df.empty:
                    filepath = Path(output_dir) / "lines.csv"
                    lines_df.to_csv(filepath)
                    logger.info(f"✅ Exported lines.csv ({len(lines_df)} lines)")
                    exported_grids["lines"] = lines_df
            
            # We only need to export once if multiple markets share the same grid
            break
    
    if exported_grids["buses"] is None and exported_grids["lines"] is None:
        logger.info("No grid data found in markets (buses/lines not exported)")
    
    return exported_grids


def export_demand_timeseries(
    world: World,
    output_dir: str | Path,
) -> pd.DataFrame | None:
    """
    Export demand time series data to CSV.
    
    Extracts demand values from demand unit forecasters and creates a time-indexed
    DataFrame with columns for each demand unit.
    
    Args:
        world: The World instance containing demand units.
        output_dir: Directory to save the CSV file.
        
    Returns:
        DataFrame of demand time series, or None if no demand data exists.
    """
    demand_data = {}
    
    for operator_id, unit_operator in world.unit_operators.items():
        for unit_id, unit in unit_operator.units.items():
            # Only process Demand units
            if not isinstance(unit, Demand):
                continue
            
            # Extract demand time series from forecaster
            if hasattr(unit, "forecaster") and hasattr(unit.forecaster, "forecasts"):
                forecasts = unit.forecaster.forecasts
                if "energy" in forecasts:
                    demand_data[unit_id] = forecasts["energy"]
                elif "demand" in forecasts:
                    demand_data[unit_id] = forecasts["demand"]
    
    if not demand_data:
        logger.info("No demand time series data to export")
        return None
    
    # Create DataFrame from dict of series, ensuring we use the world's index
    # to keep time series aligned when values are array-like or FastSeries
    df = pd.DataFrame(index=world.index)
    for uid, series in demand_data.items():
        if isinstance(series, pd.Series):
            df[uid] = series.reindex(world.index)
        else:
            # array-like or FastSeries-like values
            try:
                df[uid] = pd.Series(series, index=world.index)
            except Exception:
                # Fall back to assigning directly and let pandas attempt alignment
                df[uid] = series

    # Ensure negative values (demand convention)
    df = -abs(df)
    
    filepath = Path(output_dir) / "demand_df.csv"
    df.to_csv(filepath)
    logger.info(f"✅ Exported demand_df.csv ({len(df)} timesteps × {len(df.columns)} demand units)")
    
    return df


def export_availability_timeseries(
    world: World,
    output_dir: str | Path,
) -> pd.DataFrame | None:
    """
    Export availability/capacity factor time series data to CSV.
    
    Extracts availability values (typically for renewable generators) from unit forecasters.
    
    Args:
        world: The World instance containing units.
        output_dir: Directory to save the CSV file.
        
    Returns:
        DataFrame of availability time series, or None if no availability data exists.
    """
    availability_data = {}
    
    for operator_id, unit_operator in world.unit_operators.items():
        for unit_id, unit in unit_operator.units.items():
            # Check for availability in forecaster
            if hasattr(unit, "forecaster") and hasattr(unit.forecaster, "forecasts"):
                forecasts = unit.forecaster.forecasts
                
                # Look for availability-related forecasts
                for key in ["availability", "capacity_factor", "p_max_pu"]:
                    if key in forecasts:
                        availability_data[unit_id] = forecasts[key]
                        break
    
    if not availability_data:
        logger.info("No availability time series data to export")
        return None
    
    # Create DataFrame from dict of series, using world.index for alignment
    df = pd.DataFrame(index=world.index)
    for uid, series in availability_data.items():
        if isinstance(series, pd.Series):
            df[uid] = series.reindex(world.index)
        else:
            try:
                df[uid] = pd.Series(series, index=world.index)
            except Exception:
                df[uid] = series
    
    filepath = Path(output_dir) / "availability_df.csv"
    df.to_csv(filepath)
    logger.info(f"✅ Exported availability_df.csv ({len(df)} timesteps × {len(df.columns)} units)")
    
    return df


def export_exchange_timeseries(
    world: World,
    output_dir: str | Path,
) -> pd.DataFrame | None:
    """
    Export exchange unit time series data to CSV.
    
    Extracts exchange values from exchange unit forecasters.
    
    Args:
        world: The World instance containing exchange units.
        output_dir: Directory to save the CSV file.
        
    Returns:
        DataFrame of exchange time series, or None if no exchange data exists.
    """
    exchange_data = {}
    
    for operator_id, unit_operator in world.unit_operators.items():
        for unit_id, unit in unit_operator.units.items():
            # Check if this is an exchange-type unit
            if hasattr(unit, "technology") and "exchange" in str(unit.technology).lower():
                if hasattr(unit, "forecaster") and hasattr(unit.forecaster, "forecasts"):
                    forecasts = unit.forecaster.forecasts
                    
                    # Look for exchange-related forecasts
                    for key in ["exchange", "import", "export", "energy"]:
                        if key in forecasts:
                            exchange_data[unit_id] = forecasts[key]
                            break
    
    if not exchange_data:
        logger.info("No exchange time series data to export")
        return None
    
    # Create DataFrame from dict of series, ensure world.index alignment
    df = pd.DataFrame(index=world.index)
    for uid, series in exchange_data.items():
        if isinstance(series, pd.Series):
            df[uid] = series.reindex(world.index)
        else:
            try:
                df[uid] = pd.Series(series, index=world.index)
            except Exception:
                df[uid] = series
    
    filepath = Path(output_dir) / "exchanges_df.csv"
    df.to_csv(filepath)
    logger.info(f"✅ Exported exchanges_df.csv ({len(df)} timesteps × {len(df.columns)} exchange units)")
    
    return df


def export_config(
    world: World,
    output_dir: str | Path,
    scenario_name: str = "exported_scenario",
    study_case_name: str = "base_case",
) -> dict:
    """
    Export configuration YAML file.
    
    Creates a config.yaml file based on the world's current configuration.
    
    Args:
        world: The World instance to export configuration from.
        output_dir: Directory to save the config file.
        scenario_name: Name of the scenario.
        study_case_name: Name of the study case.
        
    Returns:
        Dictionary containing the configuration.
    """
    # Build markets config from world.markets
    markets_config = {}
    for market_id, market_config in world.markets.items():
        markets_config[market_id] = {
            "operator": f"{market_id}_operator",
            "product_type": market_config.product_type,
            "products": [
                {
                    "duration": str(p.duration),
                    "count": p.count,
                    "first_delivery": str(p.first_delivery),
                }
                for p in market_config.market_products
            ],
            "opening_frequency": "1h",  # Default, adjust as needed
            "opening_duration": str(market_config.opening_duration),
            "market_mechanism": market_config.market_mechanism,
            "maximum_bid_volume": market_config.maximum_bid_volume,
            "maximum_bid_price": market_config.maximum_bid_price,
            "minimum_bid_price": market_config.minimum_bid_price,
            "volume_unit": market_config.volume_unit,
            "price_unit": market_config.price_unit,
        }
        
        if market_config.additional_fields:
            markets_config[market_id]["additional_fields"] = market_config.additional_fields
    
    # Build the configuration
    config = {
        study_case_name: {
            "start_date": world.start.strftime("%Y-%m-%d %H:%M"),
            "end_date": world.end.strftime("%Y-%m-%d %H:%M"),
            "time_step": "1h",  # Adjust based on your simulation
            "simulation_id": world.simulation_id or f"{scenario_name}_{study_case_name}",
            "save_frequency_hours": 48,
            "markets_config": markets_config,
        }
    }
    
    filepath = Path(output_dir) / "config.yaml"
    with open(filepath, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    
    logger.info("✅ Exported config.yaml")
    
    return config


def export_world_to_csv(
    world: World,
    output_dir: str | Path,
    scenario_name: str = "exported_scenario",
    study_case_name: str = "base_case",
    generator_time_series: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Export complete world state to CSV files.
    
    This is the main export function that orchestrates the export of all
    components from a World instance to CSV files in the format expected
    by loader_csv.py.
    
    Args:
        world: The World instance to export.
        output_dir: Base directory for exports. Will create subdirectory for scenario.
        scenario_name: Name for the scenario subfolder.
        study_case_name: Name for the study case in config.yaml.
        
    Returns:
        Dictionary of exported DataFrames keyed by file type.
    """
    # Create output directory structure
    scenario_dir = Path(output_dir) / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"EXPORTING WORLD TO CSV: {scenario_dir}")
    logger.info(f"{'='*80}\n")
    
    # Create strategy mapping once
    strategy_mapping = create_strategy_class_to_key_mapping()
    
    # Export all components
    exported_data = {}
    
    # Export unit operators
    unit_operators_df = export_unit_operators(world, scenario_dir)
    if unit_operators_df is not None:
        exported_data["unit_operators"] = unit_operators_df
    
    # Export units by type
    powerplant_df = export_powerplant_units(world, scenario_dir, strategy_mapping)
    if powerplant_df is not None:
        exported_data["powerplant_units"] = powerplant_df
    
    storage_df = export_storage_units(world, scenario_dir, strategy_mapping)
    if storage_df is not None:
        exported_data["storage_units"] = storage_df
    
    demand_df = export_demand_units(world, scenario_dir, strategy_mapping)
    if demand_df is not None:
        exported_data["demand_units"] = demand_df
    
    # Export fuel prices
    fuel_prices_df = None
    # If the user requests generator-level time series explicitly, call generator exporter
    if generator_time_series:
        fuel_prices_df = export_generator_fuel_prices(world, scenario_dir)
    else:
        # Try per-fuel export first for backward compatibility
        fuel_prices_df = export_fuel_prices(world, scenario_dir)
        # If only a single-row/legacy scalar export was emitted, provide per-generator series
        try:
            if fuel_prices_df is not None and getattr(fuel_prices_df, 'shape', (1, 0))[0] == 1:
                # Replace with per-generator time series if available
                gen_df = export_generator_fuel_prices(world, scenario_dir)
                if gen_df is not None and not gen_df.empty:
                    fuel_prices_df = gen_df
        except Exception:
            pass
    if fuel_prices_df is not None:
        exported_data["fuel_prices_df"] = fuel_prices_df
    
    # Export grid data (buses and lines) if available
    grid_data = export_grid_data(world, scenario_dir)
    if grid_data["buses"] is not None:
        exported_data["buses"] = grid_data["buses"]
    if grid_data["lines"] is not None:
        exported_data["lines"] = grid_data["lines"]
    
    # Export time series data
    demand_ts = export_demand_timeseries(world, scenario_dir)
    if demand_ts is not None:
        exported_data["demand_df"] = demand_ts
    
    availability_ts = export_availability_timeseries(world, scenario_dir)
    if availability_ts is not None:
        exported_data["availability_df"] = availability_ts
    
    exchange_ts = export_exchange_timeseries(world, scenario_dir)
    if exchange_ts is not None:
        exported_data["exchanges_df"] = exchange_ts
    
    # Export configuration
    export_config(world, scenario_dir, scenario_name, study_case_name)
    
    logger.info("")
    logger.info("="*80)
    logger.info("EXPORT COMPLETE")
    logger.info(f"Location: {scenario_dir}")
    logger.info(f"Files exported: {len(exported_data) + 1} (+ config.yaml)")
    logger.info(f"{'='*80}\n")
    
    return exported_data
