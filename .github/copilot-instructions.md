# ASSUME Framework - AI Agent Instructions

## Project Overview
ASSUME is an agent-based electricity market simulation framework that models European electricity markets using multi-agent systems and reinforcement learning. The codebase combines asynchronous agent communication (via mango-agents), market clearing mechanisms, bidding strategies, and optional deep reinforcement learning capabilities.

## Communication Style & Code Standards

### Be Concise - NOT VERBOSE
- Keep chat responses brief and to the point
- Avoid unnecessary explanations or process commentary
- State what you're doing, then do it
- Don't describe every step unless specifically asked

### Python Code Standards by Context

**Jupyter Notebooks (.ipynb)**:
- **Prioritize SIMPLICITY over robustness**
- Notebooks are for understanding and exploration, NOT production
- Minimal print statements - only expose critical information (results, key metrics)
- NO process steps, diagnosis messages, or verbose logging in outputs
- Simple, readable code over defensive programming
- Example: Print final result, not "Loading data...", "Processing...", "Done!"

**Python Modules (.py files)**:
- **Prioritize ROBUSTNESS**
- Include proper error handling, validation, type hints
- Comprehensive docstrings and logging where appropriate
- Production-quality code with edge case handling

### Working with Jupyter Notebooks

**Critical Tools**:
- Use `read_file` to retrieve notebook cell contents (specifying line ranges)
- Use `edit_notebook_file` to modify notebook cells (insert/edit/delete operations)
- **DO NOT use `copilot_getNotebookSummary`** - it does not work reliably
- Use `run_notebook_cell` to execute cells and verify changes

**Notebook Editing Pattern**:
1. Read notebook file to understand structure and get cell content
2. Use `edit_notebook_file` with proper cellId and editType
3. Keep cells focused and concise - avoid verbose outputs
4. Run modified cells to validate functionality

**Cell Output Philosophy**:
- Show final results, not intermediate progress
- Use simple f-strings, not elaborate formatted tables (unless truly needed)
- Example BAD: `print("Processing unit 1..."); print("Processing unit 2...")`
- Example GOOD: `print(f"Processed {len(units)} units")`

## Core Architecture Concepts

### Agent-Based Multi-Layer System
The framework has three distinct agent layers:
1. **World**: Central coordinator managing simulation lifecycle, container orchestration, and database connections
2. **Unit Operators**: Agents managing portfolios of generation/storage/demand units with bidding strategies  
3. **Market Operators**: Agents running market clearing mechanisms (pay-as-clear, nodal, redispatch, etc.)

Communication flows: `World` → `UnitsOperator` (opening messages) → Market bids → `MarketRole` (clearing) → `UnitsOperator` (acceptance feedback)

### Key Structural Pattern: Unit → Operator → Strategy
- **Units** (`assume/units/`): Physical assets (PowerPlant, Storage, Demand, Building) with technical constraints
- **UnitsOperator** (`assume/common/units_operator.py`): Portfolio manager bidding units into markets
- **Strategies**: Two types exist:
  - `UnitStrategy` in `assume/strategies/`: Per-unit bidding logic (e.g., `EnergyNaiveStrategy`)
  - `UnitOperatorStrategy` in `assume/strategies/portfolio_strategies.py`: Portfolio-level optimization (default: `DirectUnitOperatorStrategy` aggregates individual unit bids)

**Critical**: When units have learning strategies, they must use `RLUnitsOperator` instead of standard `UnitsOperator`. The framework automatically handles this via `adjust_unit_operator_for_learning()` when `learning_mode=True`.

### Bidding Strategy Assignment
Units specify bidding strategies per market via `bidding_type` in scenario CSVs. Strategy names follow the pattern `{unit_type}_{product_type}_{method}`:

**Common strategies** (defined in `assume/strategies/__init__.py` → `bidding_strategies` dict):
- `powerplant_energy_naive` / `demand_energy_naive` — simple marginal-cost/utility bidding
- `powerplant_energy_heuristic_flexable` — flexible dispatch with min up/down constraints
- `storage_energy_heuristic_flexable` — storage arbitrage
- `powerplant_energy_naive_redispatch` / `demand_energy_naive_redispatch` — redispatch market
- `powerplant_energy_naive_profile` — profile-based generation
- Full list: ~30 strategies covering powerplant, demand, storage, household, industry, exchange

**Deprecated names** (still work via `deprecated_bidding_strategies` dict): `naive_eom`, `flexable_eom`, `flexable_pos_crm`, etc. New code should use the new naming convention.

The `UnitsOperator.submit_bids()` method selects the appropriate strategy based on the market opening message.

## Database & Output System

### Database Schema (TimescaleDB/PostgreSQL or SQLite)
Results are stored in these main tables (all include `simulation` column for multi-run isolation):
- `market_meta`: Market clearing results (price, supply_volume, demand_volume per timestamp)
- `market_orders`: All submitted bids with acceptance status (unit_id, price, volume, accepted_price, accepted_volume)
- `market_dispatch`: Accepted dispatches aggregated by unit
- `unit_dispatch`: Detailed unit operational state (power output, costs, revenues)
- `rl_params`: Reinforcement learning metrics (reward, regret, profit, observations, actions)

**Connection patterns**:
```python
# SQLite (development)
db_uri = "sqlite:///./examples/local_db/assume_db.db"

# TimescaleDB (production with Grafana dashboards)
db_uri = "postgresql://assume:assume@localhost:5432/assume"
```

The `WriteOutput` role (`assume/common/outputs.py`) handles buffered writes with configurable frequency (`save_frequency_hours`) and buffer size (`outputs_buffer_size_mb`).

## Critical Development Workflows

### Running Simulations
```bash
# CLI approach (recommended for production)
assume -s example_01b -db "postgresql://assume:assume@localhost:5432/assume"

# Python approach (development/notebooks)
python examples/examples.py  # Edit example selection in __main__

# With Docker Compose (TimescaleDB + Grafana)
docker compose up -d  # Access Grafana at http://localhost:3000
```

### Testing Structure
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=assume --cov-report=html

# Skip slow integration tests
pytest -m "not slow"

# Run only learning tests (requires torch)
pytest -m require_learning

# Specific test file
pytest tests/test_complex_market_mechanisms.py -v
```

**Test fixture pattern**: Most tests use pytest fixtures for unit/market setup (see `tests/conftest.py` for shared mocks). Complex units (Building, SteamPlant) have dedicated fixture files.

### Adding New Unit Types
1. Create unit class inheriting from `BaseUnit` or specific support protocol (e.g., `SupportsMinMaxCharge` for storage)
2. Implement required methods: `calculate_min_max_power()`, `calculate_marginal_cost()`
3. Register in `assume/units/__init__.py` → `unit_types` dict
4. Create corresponding bidding strategy in `assume/strategies/`
5. Add forecaster support in `assume/common/forecaster.py` if needed (unit-specific forecaster classes: `PowerplantForecaster`, `DemandForecaster`, `UnitForecaster`, `StorageForecaster`, etc.)

Example from `assume/units/powerplant.py`:
```python
class PowerPlant(BaseUnit):
    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        # Fuel cost + CO2 cost + variable O&M
        fuel_price = self.forecaster.get_price("fuel_type")
        return (fuel_price + co2_price * emission_factor) / efficiency + var_cost
```

### Reinforcement Learning Integration
RL units require special setup:
1. Use learning strategies from `assume/strategies/learning_strategies.py` (e.g., `RLStrategy`, `StorageRLStrategy`)
2. Units automatically assigned to `RLUnitsOperator` when `learning_mode=True`
3. Learning role (`assume/reinforcement_learning/learning_role.py`) centrally manages:
   - Actor-critic networks (stored per unit_id in `rl_strats` dict)
   - Experience buffer collection
   - Training episodes via `run_learning()` in `assume/scenario/loader_csv.py`

**TensorBoard monitoring**:
```bash
tensorboard --logdir tensorboard  # View at http://localhost:6006
```

Query learning results:
```sql
SELECT datetime, unit, reward, regret, profit 
FROM rl_params 
WHERE simulation = 'example_02e_base' AND evaluation_mode = false
```

## Project-Specific Conventions

### Time Handling
- Always use `datetime` objects (not timestamps) for product tuples: `(start: datetime, end: datetime, only_hours: str)`
- Market products defined via `dateutil.rrule` for recurring market openings
- Index management via `FastIndex` class for efficient pandas operations

### Market Clearing Protocols
Markets implement `MarketMechanism` protocol with `clear(orderbook, market_products)` method returning `(accepted, rejected, meta, prices)`.

**Registered clearing algorithms** (`assume/markets/clearing_algorithms/__init__.py` → `clearing_mechanisms` dict):
- `pay_as_clear` — uniform pricing (PayAsClearRole)
- `pay_as_bid` — discriminatory pricing (PayAsBidRole)
- `pay_as_bid_contract` — bilateral contracts (PayAsBidContractRole)
- `complex_clearing` — Pyomo-based with network constraints (ComplexClearingRole)
- `pay_as_clear_complex_dmas` — DMAS complex clearing (ComplexDmasClearingRole)
- `redispatch` — redispatch market with PyPSA OPF (RedispatchMarketRole, requires pypsa)
- `nodal_clearing` — nodal/zonal LMP clearing with PyPSA OPF (NodalClearingRole, requires pypsa)

Custom clearing algorithms can be registered: `clearing_mechanisms["my_clearing"] = MyClearingRole`

Complex clearing uses Pyomo optimization:
```python
# See assume/markets/clearing_algorithms/complex_clearing_dcopf.py
model = pyo.ConcreteModel()
model.vars = pyo.Var(...)  # Decision variables
model.constraints = pyo.Constraint(...)  # Market rules
results = solver.solve(model)
```

### Nodal Clearing (NodalClearingRole)
Network-constrained clearing using PyPSA linear OPF (`assume/markets/clearing_algorithms/nodal_clearing.py`).

**Setup**: Pass grid topology via `MarketConfig.param_dict["grid_data"]` containing DataFrames: `buses`, `lines`, `generators`, `loads`, `storage_units`.

**Key behavior**:
- `required_fields = ["node", "max_power"]` — orders must specify bus node
- LMPs extracted from `network.buses_t.marginal_price` after PyPSA optimization
- Supports `pricing_mechanism`: `"pay_as_clear"` (default) or `"pay_as_bid"` via `param_dict`
- Supports zonal aggregation via `zones_identifier` in `param_dict`

**Grid utilities** (`assume/common/grid_utils.py`):
- `read_pypsa_grid(network, grid_dict)` — populates a PyPSA network from grid data dict
- `add_generators()`, `add_nodal_loads()`, `add_redispatch_generators()` — add components
- `calculate_network_meta()` — compute supply/demand volumes and nodal prices

### CSV Scenario Loading
Input structure under `examples/inputs/{scenario}/{study_case}/`:
- `config.yaml`: Market definitions, unit operators, simulation parameters
- `*_units.csv`: Unit technical parameters (powerplant_units.csv, storage_units.csv, demand_units.csv)
- `*_df.csv`: Timeseries data (fuel_prices_df.csv, availability_df.csv, demand_df.csv)

Load via: `load_scenario_folder(world, inputs_path, scenario, study_case)`

### Code Quality Standards
- SPDX license headers required in all files: `# SPDX-FileCopyrightText: ASSUME Developers` + `# SPDX-License-Identifier: AGPL-3.0-or-later`
- Ruff for linting/formatting (config in `pyproject.toml`)
- Type hints encouraged but not strictly enforced
- Pre-commit hooks configured: `pre-commit install`

## Common Pitfalls

1. **Database connection timing**: Always check `World.db_uri` exists before assuming database operations work (supports both SQLite and PostgreSQL)

2. **Learning mode unit operators**: Don't manually create `RLUnitsOperator` - use `adjust_unit_operator_for_learning()` or set `learning_mode=True` in world config

3. **Market product timing**: Products use `end_time` for clearing timestamp, but bids reference `start_time`. This asymmetry is intentional for hourly markets.

4. **Bidding strategy naming**: Strategy keys in `bidding_strategies` dict must match `market_id` exactly (case-sensitive)

5. **Forecaster data access**: Use `forecaster.get_price(fuel_type)` not direct dict access - handles fallback logic for missing keys

6. **PyPSA network integration**: Network-based clearing requires `pypsa` installation (`pip install 'assume-framework[network]'`) and grid topology in market config

## Installation & Dependencies

```bash
# Core installation
pip install assume-framework

# With reinforcement learning (PyTorch)
pip install 'assume-framework[learning]'

# With network clearing (PyPSA)
pip install 'assume-framework[network]'

# Development (all features + testing)
pip install -e ".[all]"
```

**Python version**: Requires 3.10+ (tested on 3.10, 3.11, 3.12, 3.13)

## Egret SCUC/SCED Integration (Planned)

This fork extends ASSUME with Sandia's Egret solver for security-constrained unit commitment (SCUC) and economic dispatch (SCED).

### Clearing Algorithms (to be registered)
- `scuc_clearing` — day-ahead SCUC via `ScucClearingRole` (MIP → fix binaries → LP for LMPs)
- `sced_clearing` — real-time SCED via `ScedClearingRole` (LP with fixed DA commitments)

### Egret API
```python
from egret.data.model_data import ModelData
from egret.models.unit_commitment import solve_unit_commitment

md_results = solve_unit_commitment(model_data, "highs", relaxed=False)
```
- Default formulation: `create_tight_unit_commitment_model` (garver_3bin, KOW, pan_guan_gentile, damcikurt, rajan_takriti, ptdf)
- LMPs: only available when `relaxed=True`, extracted from dual variables on `eq_p_balance`
- Results in returned ModelData: `generator[g]['pg']` (dispatch), `bus[b]['lmp']` (prices), `system['total_cost']`

### Data Adapter Pattern
- `orders_to_model_data()` — converts ASSUME orderbook + grid_data → Egret ModelData
- `model_results_to_orderbook()` — converts Egret results → ASSUME (accepted, rejected, meta, prices)
- UC parameters (ramp rates, min up/down, startup costs) passed via `MarketConfig.param_dict["generator_params"]`

### Two-Settlement Architecture
DA market (SCUC) produces commitment schedules; RT market (SCED) takes fixed commitments and solves for dispatch + RT LMPs.

## Documentation Resources
- User docs: https://assume.readthedocs.io
- API reference: Built from docstrings via Sphinx
- Example notebooks: `examples/notebooks/` (runnable in Colab)
- Paper: https://doi.org/10.1016/j.softx.2025.102176
