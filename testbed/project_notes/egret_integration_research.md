# Egret SCUC/SCED Integration into ASSUME — Research Notes

## 1. What Egret Is

Egret (Electrical Grid Research and Engineering Tools) is a Python/Pyomo library for
building power system optimization problems, developed by Sandia National Laboratories
under the Grid Parity Exchange project. It is not an application — it is a library of
optimization formulations that other tools call.

Egret provides reference implementations of multiple unit commitment MIP formulations
from the academic literature, all written in Pyomo. The key value proposition: you
choose a formulation by name, pass it generator/network/demand data, solve it, and
extract commitment decisions, dispatch levels, and LMPs from dual variables.

Prescient, Sandia's production cost model, uses Egret as its optimization backend for
both day-ahead RUC (Reliability Unit Commitment) and real-time SCED.

**Package**: `gridx-egret` on PyPI.
**License**: BSD-3-Clause.
**Repository**: https://github.com/grid-parity-exchange/Egret

---

## 2. The Foundational Papers

### 2a. Knueven, Ostrowski & Watson (2020a) — Formulation Comparison

"On Mixed-Integer Programming Formulations for the Unit Commitment Problem."
*INFORMS Journal on Computing* 32(4):857–876, 2020.
DOI: [10.1287/ijoc.2019.0944](https://pubsonline.informs.org/doi/10.1287/ijoc.2019.0944)

This paper systematically compares UC MIP formulations by LP relaxation tightness.
It establishes a dominance hierarchy: formulations with tighter LP relaxations
consistently solve faster because the branch-and-bound tree is smaller. All
formulations examined are implemented in Egret as reference Pyomo code.

Formulations compared (by label used in Egret):

| Label  | Source                                    | Notes                           |
|--------|-------------------------------------------|---------------------------------|
| Garver | Garver (1962)                             | Original 3-binary formulation   |
| CA     | Carrión & Arroyo (2006)                   | Widely used baseline            |
| MLR    | Morales-España, Latorre & Ramos (2013)    | "Tight and compact"             |
| OAV    | Ostrowski, Anjos & Vannelli (2012)        | Several variants (O, UD, etc.)  |
| ALS    | Atakan, Lulli & Sen (2018)                | Multi-period ramping            |
| KOW    | Knueven, Ostrowski & Watson (2018/2020)   | Tightest known for general case |

The KOW formulation (also called "Tight" in Egret) dominates all others in LP
relaxation quality and is the recommended default.

### 2b. Knueven, Ostrowski & Watson (2020b) — Startup Cost Formulations

"A Novel Matching Formulation for Startup Costs in Unit Commitment."
*Mathematical Programming Computation* 12:225–248, 2020.
DOI: [10.1007/s12532-020-00176-5](https://link.springer.com/article/10.1007/s12532-020-00176-5)

This paper addresses time-dependent startup costs (hot/warm/cold start), which are
critical for realistic ISO market modeling. Generators that have been offline longer
cost more to restart. The "matching" formulation provides a tighter LP relaxation for
the startup cost component than previous approaches. Also implemented in Egret.

### 2c. NREL Technical Report — Solving the UC Problem (2023)

"Solving the Unit Commitment Problem." NREL/TP-6A40-85006, 2023.
URL: https://docs.nrel.gov/docs/fy23osti/85006.pdf

Comprehensive survey of UC solution methods. References Egret/Prescient as the
state-of-the-art open-source implementation. Covers formulation tightening,
decomposition methods, and solver benchmarks.

### 2d. Tesfatsion & Battula (2020) — AMES V5.0 SCUC/SCED

"Analytical SCUC/SCED Optimization Formulation for AMES V5.0."
Economics Working Paper #20014, Iowa State University, 2020.
URL: https://ideas.repec.org/p/isu/genstf/202007020700001108.html

Provides the complete mathematical formulation for SCUC and SCED as used in the AMES
two-settlement market simulator. While AMES uses PSST (not Egret) as its solver, the
formulation is a useful reference for what constraints a US ISO market requires.

### 2e. pglib-uc Benchmark Library

"Benchmarks for the Unit Commitment Problem."
Repository: https://github.com/power-grid-lib/pglib-uc

Standardized UC test cases in JSON format, with reference implementations in both
Egret and PSST. These benchmarks are used by the power systems optimization community
to compare formulations and solvers. The JSON format defines the canonical data
schema for UC problems — generator parameters, network topology, demand profiles,
and reserve requirements.

---

## 3. Egret's Architecture

### 3a. ModelData — the central data object

Egret organizes all power system data in a `ModelData` object. This is a nested
dictionary structure (accessible via attribute-style access) with these top-level
keys:

```python
model_data = {
    "system": {
        "time_keys": [...],              # list of time period labels
        "time_period_length_minutes": 60  # or 5 for RT
    },
    "elements": {
        "generator": {
            "gen_1": {
                "bus": "bus_1",
                "generator_type": "thermal",
                "p_min": 50.0,                    # MW minimum output when committed
                "p_max": 350.0,                   # MW maximum output
                "ramp_up_60min": 200.0,           # MW/hr ramp up limit
                "ramp_down_60min": 200.0,         # MW/hr ramp down limit
                "startup_capacity": 50.0,         # MW output at startup
                "shutdown_capacity": 50.0,        # MW output at shutdown
                "min_up_time": 4,                 # hours must stay on after start
                "min_down_time": 4,               # hours must stay off after stop
                "startup_cost": [                 # time-dependent startup costs
                    (4, 5000.0),                  # (hours_offline, cost_$)
                    (10, 10000.0),                # warm start
                    (24, 20000.0),                # cold start
                ],
                "fixed_commitment": None,         # or 0/1 for SCED (fixed by RUC)
                "p_cost": {                       # piecewise-linear cost curve
                    "data_type": "cost_curve",
                    "cost_curve_type": "piecewise",
                    "values": [
                        (50.0, 1000.0),           # (MW, total_cost_$)
                        (200.0, 5000.0),
                        (350.0, 10000.0),
                    ]
                },
                "initial_status": 4,              # hours online (>0) or offline (<0) at t=0
                "initial_p_output": 200.0,        # MW output at t=0
            },
            ...
        },
        "bus": {
            "bus_1": {
                "p_load": 300.0,                  # MW load (or time-series dict)
                "va": 0.0,                        # reference angle (slack bus)
            },
            ...
        },
        "branch": {
            "line_1_2": {
                "from_bus": "bus_1",
                "to_bus": "bus_2",
                "reactance": 0.05,                # per-unit
                "rating_long_term": 500.0,        # MW thermal limit
            },
            ...
        },
    }
}
```

Time-varying data (load profiles, renewable availability, forecasts) is stored as
dicts keyed by time period within each element, using a `"data_type": "time_series"`
wrapper.

### 3b. The pglib-uc JSON format

The pglib-uc benchmark library defines the canonical JSON schema for UC problems.
Egret can load these directly. The format includes all standard generator parameters
(piecewise-linear costs, ramping, startup curves, min up/down times), DC network
model (buses, branches with reactance and flow limits), and time-varying demand
and renewable profiles. This is also the format used for the FERC unit commitment
benchmark instances.

### 3c. UC model generation

Egret builds Pyomo models via a model generator pattern:

```python
from egret.model_library.unit_commitment import uc_model_generator

# Select formulation components
formulation = {
    "status_vars": "garver_3bin_vars",         # or "CA_1bin_vars", etc.
    "power_vars": "basic_power_vars",
    "startup_costs": "KOW_startup_costs",      # from 2020b paper
    "production_costs": "ML_production_costs",  # or piecewise-linear
    "power_balance": "copperplate_power_balance",  # or "ptdf_power_flow"
    "reserve_requirement": "MLR_reserve_constraints",
    "ramp_limits": "KOW_ramp_limits",
}

# Build model
model = uc_model_generator.create_tight_unit_commitment_model(
    model_data,
    **formulation
)

# Solve
solver = pyo.SolverFactory("gurobi")  # or "appsi_highs", "cplex"
results = solver.solve(model, tee=False)
```

The key insight: Egret is compositional. You mix and match formulation components
(status variables, cost formulations, ramping constraints, network model) to build
exactly the UC model you need. The `create_tight_unit_commitment_model` convenience
function selects the tightest known formulation for each component.

### 3d. SCED (Economic Dispatch with fixed commitments)

For real-time dispatch, Egret fixes the binary commitment variables to the values
determined by the DA SCUC, then solves the remaining continuous dispatch problem:

```python
# After DA SCUC solve, extract commitments:
for gen in model_data.elements("generator"):
    gen["fixed_commitment"] = uc_solution[gen_name]  # 0 or 1 per time period

# Build SCED model (same infrastructure, commitments fixed)
sced_model = uc_model_generator.create_tight_unit_commitment_model(
    model_data_rt,  # updated with RT forecasts
    # ... formulation components ...
)
# Binary vars are now fixed → LP solve, much faster
# Duals on power balance = LMPs
```

### 3e. LMP extraction

LMPs come from dual variables of the power balance constraint. In Egret's DC-OPF
formulation:

- **Energy component**: dual of the system power balance constraint
- **Congestion component**: dual of branch flow limit constraints, propagated via
  PTDF (Power Transfer Distribution Factor) matrix
- **Loss component**: zero in DC approximation (losses modeled separately if needed)

```python
# After solving SCED:
for bus in model_data.elements("bus"):
    lmp[bus] = model.dual[model.power_balance[bus, t]]
```

This is structurally identical to how ASSUME's `complex_clearing.py` extracts
prices from `instance.dual[instance.energy_balance[node, t]]`. The mapping is
direct.

---

## 4. Prescient's Two-Settlement Architecture

Prescient implements the DA/RT simulation loop that we need to replicate in ASSUME.
Understanding its architecture is essential.

### 4a. The RUC–SCED cycle

Prescient runs two nested loops:

**Outer loop — RUC (Reliability Unit Commitment) = Day-Ahead**:
- Runs once per day (or configurable frequency)
- Solves SCUC using **day-ahead forecasts** for load, wind, solar
- Produces: unit commitment schedule (which generators on/off per hour)
- Optionally produces: pricing schedule (DA LMPs)
- Forecast horizon extends beyond the planning period (look-ahead) to avoid
  poor end-of-horizon decisions

**Inner loop — SCED (Security-Constrained Economic Dispatch) = Real-Time**:
- Runs every hour (or sub-hourly, e.g., every 5 minutes)
- Takes commitment decisions from the RUC as **fixed** (cannot change unit status)
- Solves economic dispatch using **real-time forecasts** (updated, less uncertain)
- Produces: dispatch levels for all committed generators + RT LMPs
- Can commit "fast-start" generators that weren't in the DA schedule

### 4b. Forecast divergence — how DA and RT differ

This is the mechanism for modeling uncertainty:

- **DA RUC** sees forecasted load and renewable profiles made ~12–36 hours ahead.
  These forecasts have significant uncertainty, especially for wind and solar.
- **RT SCED** sees updated forecasts made ~minutes to 1 hour ahead. These are
  much more accurate but the commitment decisions are already locked.

The gap between DA and RT forecasts is where the economic inefficiency lives:
- If DA over-forecasted wind → too few thermal units committed → RT prices spike
- If DA under-forecasted wind → too many thermal units committed → RT prices drop
  (and curtailment may occur)

Prescient models this by providing separate forecast time series for RUC and SCED
inputs. The user supplies DA forecasts (with deliberate errors or scenario draws)
and actual/RT values. Prescient runs the DA against the forecast and the RT against
actuals.

### 4c. Settlement

Prescient supports two-settlement accounting:
- Generators are paid DA LMP × DA scheduled quantity for DA awards
- RT deviations from DA schedule are settled at RT LMP
- The net payment = DA settlement + RT deviation settlement

This is the standard US ISO two-settlement mechanism (PJM, CAISO, SPP, etc.).

### 4d. Volume + Price Bids in Egret (Economic Withholding)

Egret natively supports volume+price bids through its generator `p_max` parameter. The
adapter (`clearing/egret_adapter.py`) sets each generator's `p_max` from the bid volume
rather than the nameplate capacity. This means:

- A generator offering 100 MW at $50 gets `p_max=100`, `p_cost` curve to $50
- The LP constraint is `0 ≤ dispatch ≤ 100` (bid volume, not nameplate)
- If the generator withholds capacity (offers less than nameplate), Egret sees a smaller
  generator and dispatches accordingly
- LMPs reflect the reduced supply: if withholding makes a more expensive generator
  marginal, LMPs rise at all buses

This is the key mechanism for the WithholdingStrategy: the RL agent controls both price
(action[0]) and volume (action[1]). The adapter translates the volume action into a
reduced `p_max`, and Egret's LP naturally handles the economic withholding without any
special-case logic.

**No Egret code changes required** — volume control is achieved entirely through the
adapter's data mapping. The generator cost curve is always a two-point piecewise linear
curve: `[(0, 0), (bid_volume, bid_volume * bid_price)]`.

---

## 5. ASSUME's Clearing Algorithm Interface

ASSUME clearing algorithms implement a `clear()` method on a `MarketRole` subclass:

```python
class SomeClearing(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        # Initialize from marketconfig.param_dict

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> tuple[Orderbook, Orderbook, list[dict], dict]:
        """
        Returns:
            accepted_orders:  orders with accepted_volume/price populated
            rejected_orders:  orders with accepted_volume=0
            meta:             list of dicts, one per product×node
            flows:            dict of network flows (optional)
        """
```

Each `Order` in the orderbook has:
- `bid_id`, `start_time`, `end_time`
- `volume` (positive=supply, negative=demand)
- `price` ($/MWh bid)
- `node` (bus or zone placement)
- `accepted_volume`, `accepted_price` (set by clearing)

The `meta` list contains one dict per product×node with `price`, `supply_volume`,
`demand_volume`, `product_start`, `product_end`, `node`.

The clearing algorithm is registered by name in `World.clearing_mechanisms` and
selected via `MarketConfig.market_mechanism`. Configuration parameters (solver,
grid_data, pricing_mechanism) are passed via `MarketConfig.param_dict`.

---

## 6. Integration Architecture: Egret as an ASSUME Clearing Algorithm

### 6a. The adapter pattern

Create a new clearing algorithm class `ScucScedClearingRole` that:

1. Receives ASSUME orderbook (supply bids with costs, demand bids with willingness-to-pay)
2. Converts bids into Egret `ModelData` format (generators with cost curves, loads)
3. Calls Egret's `create_tight_unit_commitment_model()` to build the Pyomo SCUC
4. Solves the MILP (Gurobi/HiGHS/CPLEX)
5. Extracts commitment decisions, dispatch, and LMPs from duals
6. Maps results back to ASSUME's orderbook format (accepted_volume, accepted_price)
7. Returns the 4-tuple (accepted, rejected, meta, flows)

### 6b. Data mapping: ASSUME Orders → Egret ModelData

| ASSUME Order field     | Egret ModelData field                           |
|------------------------|-------------------------------------------------|
| `volume` (positive)    | `generator.p_max`                               |
| `price`                | `generator.p_cost` (marginal cost → cost curve)  |
| `node`                 | `generator.bus`                                  |
| `volume` (negative)    | `bus.p_load` (aggregate demand at bus)            |
| `start_time/end_time`  | `system.time_keys`                               |

Additional generator parameters (ramp rates, min up/down, startup costs) must be
supplied outside the orderbook — either via `param_dict` in the `MarketConfig`, or
via a supplementary generator data table passed alongside the bids.

This is the key architectural decision: ASSUME's orderbook is bid-centric (price +
volume per timestep), while Egret's ModelData is unit-centric (full generator
parameters). The adapter must bridge this gap.

**Recommended approach**: Extend ASSUME's order format with an `additional_fields`
mechanism (already supported in MarketConfig) to carry UC parameters:
```python
MarketConfig(
    additional_fields=["ramp_up", "ramp_down", "min_up_time", "min_down_time",
                       "startup_cost", "min_power", "initial_status"],
    ...
)
```

Generators include these fields in their bids. The clearing algorithm extracts them
to populate Egret's ModelData.

### 6c. Two-settlement loop in ASSUME's World

ASSUME's `World` already supports multiple markets that clear in the same simulation.
The two-settlement architecture maps to:

**Market 1 — Day-Ahead (RUC/SCUC)**:
```python
MarketConfig(
    market_id="dam",
    market_mechanism="scuc_clearing",    # new Egret-based algorithm
    market_products=[MarketProduct(duration=timedelta(hours=1), count=24, ...)],
    param_dict={
        "grid_data": {...},
        "solver": "appsi_highs",
        "formulation": "tight",          # select Egret formulation
    }
)
```

**Market 2 — Real-Time (SCED)**:
```python
MarketConfig(
    market_id="rtm",
    market_mechanism="sced_clearing",    # Egret with fixed commitments
    market_products=[MarketProduct(duration=timedelta(minutes=5), count=12, ...)],
    param_dict={
        "grid_data": {...},
        "solver": "appsi_highs",
        "commitment_source": "dam",      # reference to DA market for commitments
    }
)
```

The RT clearing algorithm reads the DA commitment results and fixes generator
on/off status before solving dispatch. This coupling requires the DA market to
clear before the RT market within each simulation day — ASSUME's event loop
must be configured to enforce this sequencing.

### 6d. Forecast divergence in ASSUME

ASSUME's `UnitForecaster` already supports per-market forecasts:
```python
forecaster.market_prices["dam"]  # DA price forecast
forecaster.market_prices["rtm"]  # RT price forecast (updated)
```

For DA/RT divergence:
- DA agents bid using `forecaster.availability` based on DA renewable forecast
- RT agents bid using updated `forecaster.availability` with near-term forecast
- The gap between these forecasts drives price differences between DA and RT

This is the ASSUME-native way to model forecast uncertainty. Each agent's
`NaiveForecast` or custom forecaster provides different values for the DA and RT
market contexts.

### 6e. What must be built

| Component | Effort | Description |
|-----------|--------|-------------|
| `ScucClearingRole` | Large | New clearing algorithm: ASSUME orders → Egret ModelData → SCUC solve → results mapping |
| `ScedClearingRole` | Medium | Same pattern but with fixed commitments from DA; LP solve for speed |
| Egret ModelData adapter | Medium | Convert ASSUME bus/line/generator data to Egret's nested dict format |
| Extended order fields | Small | Add UC parameters (ramp, min up/down, startup cost) to order schema |
| DA→RT commitment coupling | Medium | Mechanism for RT market to read DA commitment decisions |
| Forecast divergence setup | Small | Configure separate forecasters for DA and RT market contexts |
| pglib-uc test case loader | Small | Load pglib-uc JSON benchmarks directly for validation |

---

## 7. Solver Considerations

Egret's UC formulations produce MILPs. Solver choice matters significantly:

| Solver     | License      | UC Performance | Notes                              |
|------------|-------------|----------------|------------------------------------|
| Gurobi     | Commercial  | Best           | Free academic license; fastest     |
| CPLEX      | Commercial  | Excellent      | IBM; comparable to Gurobi          |
| HiGHS      | Open source | Good           | ASSUME's default; handles UC well  |
| GLPK       | Open source | Slow           | Not recommended for UC             |

For the testbed (9-bus, 3 generators), any solver works. For the real WECC network
(50+ clustered buses, 100+ generators), Gurobi or HiGHS is recommended. ASSUME
already uses `appsi_highs` for complex clearing, so HiGHS is the path of least
resistance.

The SCED (real-time dispatch) is an LP after commitments are fixed — all solvers
handle this efficiently.

---

## 8. Validation Strategy

### 8a. Unit test: Egret standalone on IEEE 9-bus

Before integrating into ASSUME, verify Egret produces correct SCUC/SCED results
on the IEEE 9-bus with the 3 generators:
- Commit all 3 generators under high load
- Verify one generator decommits under low load
- Verify LMPs differ across buses when a line is congested
- Verify SCED with fixed commitments produces same dispatch as SCUC

### 8b. Integration test: Egret clearing in ASSUME

Run the IEEE 9-bus testbed with the new `ScucClearingRole`:
- Compare clearing prices with ASSUME's existing `complex_clearing.py` results
- They should match when UC constraints are non-binding (all units online)
- They should differ when commitment decisions change (unit startup/shutdown)

### 8c. Two-settlement test

Run DA SCUC → RT SCED on the testbed with deliberately different forecasts:
- DA forecast: high load (all 3 generators committed)
- RT actual: low load (one generator should have been off)
- Verify DA and RT LMPs differ
- Verify two-settlement accounting (DA payment + RT deviation)

### 8d. pglib-uc benchmark validation

Run Egret via the ASSUME adapter on a pglib-uc benchmark instance. Compare
objective value and solve time against published results to verify correctness.

---

## 9. Source Materials

### Papers

1. Knueven, B., Ostrowski, J., & Watson, J.-P. (2020). "On Mixed-Integer Programming
   Formulations for the Unit Commitment Problem." *INFORMS Journal on Computing*
   32(4):857–876. [DOI](https://pubsonline.informs.org/doi/10.1287/ijoc.2019.0944)

2. Knueven, B., Ostrowski, J., & Watson, J.-P. (2020). "A Novel Matching Formulation
   for Startup Costs in Unit Commitment." *Mathematical Programming Computation*
   12:225–248. [DOI](https://link.springer.com/article/10.1007/s12532-020-00176-5)

3. Tesfatsion, L., & Battula, S. (2020). "Analytical SCUC/SCED Optimization
   Formulation for AMES V5.0." Iowa State University Economics Working Paper #20014.
   [IDEAS](https://ideas.repec.org/p/isu/genstf/202007020700001108.html)

4. "Solving the Unit Commitment Problem." NREL/TP-6A40-85006, 2023.
   [NREL](https://docs.nrel.gov/docs/fy23osti/85006.pdf)

### Software

5. Egret — https://github.com/grid-parity-exchange/Egret
   PyPI: `gridx-egret`

6. Prescient — https://github.com/grid-parity-exchange/Prescient
   Docs: https://prescient.readthedocs.io/

7. pglib-uc benchmarks — https://github.com/power-grid-lib/pglib-uc

8. PSST — https://github.com/ames-market/psst

9. AMES V5.0 — https://github.com/ames-market/AMES-V5.0

10. ASSUME — https://github.com/assume-framework/assume

### Presentations

11. Knueven, B. "Mixed Integer Programming Formulations for the Unit Commitment
    Problem." FERC Technical Conference, 2020.
    [PDF](https://www.ferc.gov/sites/default/files/2020-09/W1-A-1-Knueven.pdf)

12. Knueven, B. "The EGRET Library of Unit Commitment and Economic Dispatch Models
    and Methods for Large-scale Linear OPF." 2021.
    [OSTI](https://www.osti.gov/servlets/purl/1826439)
