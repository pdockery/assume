import pandas as pd

from assume.common.forecast_initialisation import ForecastInitialisation


def test_co2_default_added_if_missing():
    # Create a small hourly index
    index = pd.date_range(start="2020-01-01", periods=3, freq="H")

    # Minimal powerplant_units DataFrame with required columns
    powerplant_units = pd.DataFrame(
        index=["pp1"],
        data={
            "efficiency": [0.4],
            "emission_factor": [0.5],
        },
    )

    # demand_units and market_configs minimal placeholders
    demand_units = pd.DataFrame(index=["d1"], data={"min_power": [-100], "max_power": [-10], "unit_operator": ["op1"]})
    market_configs = {"EOM": {"product_type": "energy", "opening_frequency": "HOURLY", "opening_duration": "1h", "market_mechanism": "pay_as_clear", "products": [{"duration": "1h","count": 24, "first_delivery":"1h"}]}}

    # Fuel prices without 'co2'
    fuel_prices = pd.DataFrame(index=index)
    fuel_prices["gas"] = 20

    fi = ForecastInitialisation(index=index, powerplants_units=powerplant_units, demand_units=demand_units, market_configs=market_configs, fuel_prices=fuel_prices)

    assert "co2" in fi.fuel_prices.columns
    # The co2 column should contain zeros (scalar or timeseries)
    assert (fi.fuel_prices["co2"] == 0).all()


def test_generator_specific_fuel_price_preferred():
    index = pd.date_range(start="2020-01-01", periods=3, freq="H")
    powerplant_units = pd.DataFrame(index=["pp1"], data={"fuel_type": ["gas"], "efficiency": [0.4], "emission_factor": [0.5]})
    demand_units = pd.DataFrame(index=["d1"], data={"min_power": [-100], "max_power": [-10], "unit_operator": ["op1"]})
    market_configs = {"EOM": {"product_type": "energy", "opening_frequency": "HOURLY", "opening_duration": "1h", "market_mechanism": "pay_as_clear", "products": [{"duration": "1h","count": 24, "first_delivery":"1h"}]}}

    # Two fuel price columns: gas (generic) and pp1 (generator-specific)
    fuel_prices = pd.DataFrame(index=index)
    fuel_prices["gas"] = 20.0
    fuel_prices["pp1"] = 25.0

    fi = ForecastInitialisation(index=index, powerplants_units=powerplant_units, demand_units=demand_units, market_configs=market_configs, fuel_prices=fuel_prices)

    # calculate marginal cost for the generator series
    gen_series = powerplant_units.loc["pp1"].copy()
    # The name of the series is used by calculate_marginal_cost to prefer a column matching the unit id
    gen_series.name = "pp1"
    mc = fi.calculate_marginal_cost(gen_series)

    # If generator-specific column is used, marginal cost should reflect fuel price 25 / efficiency
    expected_fc = fuel_prices["pp1"][0] / gen_series["efficiency"]
    assert (mc - expected_fc).abs().max() < 1e-6
