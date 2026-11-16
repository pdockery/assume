# ASSUME Integration Plan: Rebase + Egret SCUC/SCED + IEEE 9-Bus Testbed

**Date**: 2026-03-09
**Scope**: Rebase onto upstream, update copilot instructions, build IEEE 9-bus testbed, implement Egret-based SCUC/SCED clearing

---

## Current State

| Item | Value |
|------|-------|
| Local HEAD | `6224a7cd` (main) — "improvements to pypsa interoperability workflow" |
| Upstream HEAD | `656d8c82` (upstream/main) — "Fix learning convergence with heterogeneous units (#756)" |
| Divergence | 3 local commits ahead, 109 commits behind upstream |
| Local-only commits | `cf1a3ca5`, `3e92d476`, `6224a7cd` — PyPSA interoperability work |
| Unstaged changes | `loader_pypsa.py`, `powerplant.py`, `environment.yaml`, 2 notebooks |
| Untracked files | `.github/copilot-instructions.md`, `docs/egret_integration_research.md`, `docs/ieee9_testbed_plan.md`, `outputs/` |
| Stash | `stash@{0}`: WIP on main (pre-upstream-fetch state) |

---

## Phase 1 — Rebase onto Upstream

### 1a. Preparation

```bash
# Save working tree changes
git stash push -m "pre-rebase-working-tree" -- \
  assume/scenario/loader_pypsa.py \
  assume/units/powerplant.py \
  environment.yaml \
  examples/notebooks/02_automated_run_example.ipynb \
  examples/notebooks/03_custom_unit_example.ipynb

# Copy untracked files out of the way (git rebase won't touch them, but be safe)
# They'll stay in place — just noting they exist
```

### 1b. Rebase

```bash
# Rebase the 3 local commits onto upstream/main
git rebase upstream/main
```

**Expected conflicts** (files modified both locally and upstream):
- `assume/scenario/loader_pypsa.py` — local interop changes vs upstream refactors
- `assume/units/powerplant.py` — local changes vs upstream changes
- `environment.yaml` — local dependency additions vs upstream dependency updates

**Conflict resolution strategy**: For each conflict, preserve local functional intent
(frequency inference, marginal cost handling in `loader_pypsa.py`; powerplant changes)
on top of upstream's structural changes (renamed imports, new function signatures).

### 1c. Restore working tree

```bash
# After successful rebase, restore stashed changes
git stash pop
```

### 1d. Post-rebase verification

```bash
# Verify history is clean and linear
git log --oneline -10

# Confirm upstream changes are present
python -c "from assume.markets.clearing_algorithms import clearing_mechanisms; print(list(clearing_mechanisms.keys()))"
# Should include: nodal_clearing, redispatch

# Run tests
pytest tests/ -x -q
```

### 1e. Push

```bash
# Force-push the rebased branch (after confirming tests pass)
git push --force-with-lease origin main
```

---

## Phase 2 — Update Copilot Instructions

After rebase, `.github/copilot-instructions.md` needs updating to reflect upstream
changes. Specific updates required:

### 2a. Strategy naming

**Current (stale)**:
```
bidding_strategies = {
    "EOM": NaiveSingleBidStrategy(),
}
```

**Updated**: Reference new naming convention. Upstream now uses
`{unit_type}_{product_type}_{method}` pattern (e.g., `"powerplant_energy_naive"`,
`"demand_energy_naive"`, `"powerplant_energy_learning"`). Old names preserved in
`deprecated_bidding_strategies` dict for backward compatibility.

### 2b. Clearing mechanisms

**Add**: Document `nodal_clearing` (new on upstream) and its `NodalClearingRole`
class. It uses PyPSA linear OPF via `n.optimize()`, extracts LMPs from
`n.buses_t.marginal_price`. Requires `grid_data` in `param_dict`. Supports
`zones_identifier` for zonal aggregation.

### 2c. Forecasting

**Update**: `forecast_initialisation.py` renamed to `forecasts.py` upstream.
Update any references.

### 2d. Grid utilities

**Add**: Document `assume/common/grid_utils.py` (new) with `read_pypsa_grid()`
helper that populates a PyPSA network from a grid_data dict.

### 2e. Egret/SCUC section

**Add**: New section documenting the planned Egret integration — clearing algorithm
names (`scuc_clearing`, `sced_clearing`), the adapter pattern, and the two-settlement
DA/RT architecture.

---

## Phase 3 — IEEE 9-Bus Testbed

The full testbed plan is in `docs/ieee9_testbed_plan.md`. This section summarizes
the execution sequence with corrections from primary source verification.

### 3a. Retrieve network (Phase 0 of testbed plan)

```python
import pandapower.networks as pn
import pypsa

pp_net = pn.case9()  # Verified: case9() exists in pandapower
network = pypsa.Network()
network.import_from_pandapower_net(pp_net)
```

**Verified**: `pandapower.networks.case9()` loads from `case9.json` (PYPOWER origin,
Anderson & Fouad 1980). Returns 9 buses, 3 generators, 3 loads, 9 branches.

### 3b. Partition into zones (Phase 1)

Zone A: buses 1, 2, 4, 7, 8 (Gen 1, Gen 2, Load 7)
Zone B: buses 3, 5, 6, 9 (Gen 3, Load 5, Load 9)

This creates a price gradient: Zone A (cheaper, more generation diversity) vs
Zone B (single generator, higher marginal cost).

### 3c. Build ASSUME scenario inputs (Phase 2)

Generate CSV files under `data/testbed/ieee9/assume_scenario/`:
- `powerplant_units.csv` — 3 generators with varied marginal costs
- `demand_units.csv` — 3 loads
- `fuel_prices.csv` — flat prices for testbed
- `buses.csv`, `lines.csv` — from PyPSA export
- `config.yaml` — market definitions

**Note**: The testbed plan references `bidding_type: "naive_eom"`. After rebase,
use new naming: `"powerplant_energy_naive"` for generators, `"demand_energy_naive"`
for loads. The deprecated names still work but new code should use new names.

### 3d. Wire into ASSUME World (Phase 3)

Create `testbed/testbed_runner.py` that builds an ASSUME World. Key integration
points (all verified against upstream):

- `world.add_market_operator()` with `MarketConfig`
- `clearing_mechanisms` dict to register custom clearing roles
- `grid_data` in `param_dict` must contain: `buses` (DataFrame), `lines` (DataFrame),
  `generators` (DataFrame), `loads` (DataFrame)
- `NodalClearingRole` pattern: `self.grid_data = marketconfig.param_dict.get("grid_data")`

### 3e. Add seam agent (Phase 4)

Simplified `testbed_seam_strategy.py` with 4-dim observation space (zone prices +
time encoding). Verifies the seam scheduling concept before scaling to real WECC.

---

## Phase 4 — Egret SCUC/SCED Integration

This is the core engineering work. The research notes are in
`docs/egret_integration_research.md`. Below are the corrections and refined
implementation details from primary source verification.

### 4a. Corrections to research doc

The following items in `egret_integration_research.md` need correction based on
verified Egret source code:

1. **API entry point**: The research doc describes calling
   `create_tight_unit_commitment_model()` directly with formulation components.
   The actual recommended entry point is:

   ```python
   from egret.models.unit_commitment import solve_unit_commitment

   md_results = solve_unit_commitment(
       model_data,        # egret.data.ModelData
       solver,            # str ("gurobi", "cbc", "highs") or Pyomo solver
       relaxed=False,     # True for LP relaxation (needed for LMPs)
       return_model=True, # optionally return the Pyomo model
   )
   ```

   This is a high-level function that:
   - Creates the Pyomo model via `create_tight_unit_commitment_model` (default)
   - Solves with lazy PTDF iteration for transmission feasibility
   - Extracts ALL results (dispatch, commitment, LMPs, flows, costs) back into
     a new ModelData object
   - Returns the updated ModelData (and optionally the model + results)

2. **Formulation components (verified)**: The tight formulation uses:
   ```python
   ['garver_3bin_vars',
    'garver_power_vars',
    'garver_power_avail_vars',
    'pan_guan_gentile_KOW_generation_limits',
    'damcikurt_ramping',
    'KOW_production_costs_tightened',
    'rajan_takriti_UT_DT',
    'KOW_startup_costs',
    network_constraints]  # default: 'ptdf_power_flow'
   ```

3. **UCFormulation namedtuple fields (verified)**:
   `status_vars, power_vars, reserve_vars, generation_limits, ramping_limits,
   production_costs, uptime_downtime, startup_costs, network_constraints`

4. **LMP extraction**: The research doc says
   `model.dual[model.power_balance[bus, t]]`. The actual Egret code varies by
   network model:
   - `btheta_power_flow`: `dual[m.TransmissionBlock[mt].eq_p_balance[b]] / time_period_length_hours`
   - `ptdf_power_flow`: `PTDF.calculate_LMP(b, dual, b.eq_p_balance)` — **vectorized, preferred**
   - `power_balance_constraints`: `dual[m.PowerBalance[b,mt]] / time_period_length_hours`
   - `copperplate`: `dual[m.TransmissionBlock[mt].eq_p_balance] / time_period_length_hours`

   All LMP values are in $/MWh (Egret divides by `time_period_length_hours`).

   **Critical**: LMPs are only available when `relaxed=True`. The standard SCUC
   workflow for pricing is: solve MIP → fix binaries → re-solve as LP → extract duals.
   `solve_unit_commitment` handles this automatically when called with the appropriate
   options.

5. **Results in ModelData**: After `solve_unit_commitment()`, the returned ModelData
   contains (verified from `_save_uc_results`):
   - `generator[g]['pg']` — time series of dispatch (MW)
   - `generator[g]['commitment']` — time series of on/off status (0/1)
   - `generator[g]['production_cost']` — time series of production cost ($)
   - `bus[b]['lmp']` — time series of LMP ($/MWh) — **only when relaxed=True**
   - `branch[l]['pf']` — time series of line flows (MW)
   - `system['total_cost']` — total objective value ($)

6. **ModelData construction**: The research doc's nested dict format is correct.
   Verified `ModelData(source=dict)` constructor accepts the nested dict directly.
   Also verified: `ModelData.read(filename)` supports JSON, json.gz, MATPOWER (.m),
   Prescient (.dat), and pglib-uc formats.

7. **Available convenience formulations** (all take `model_data, network_constraints, relaxed, **kwargs`):
   - `create_tight_unit_commitment_model` — KOW tight (recommended default)
   - `create_compact_unit_commitment_model` — compact variant
   - `create_KOW_unit_commitment_model` — KOW matching startup costs
   - `create_CA_unit_commitment_model` — Carrión & Arroyo (widely used baseline)
   - `create_MLR_unit_commitment_model` — Morales-España, Latorre & Ramos
   - `create_ALS_unit_commitment_model` — Atakan, Lulli & Sen
   - `create_OAV_unit_commitment_model` — Ostrowski, Anjos & Vannelli (+ variants)
   - `create_super_tight_unit_commitment_model` — tightest available per component
   - `create_CHP_unit_commitment_model` — convex hull pricing formulation

### 4b. Install Egret

```bash
pip install gridx-egret
```

Dependencies: Pyomo (already in ASSUME), numpy. Solver: HiGHS (already ASSUME's
default via `highspy`/`appsi_highs`), Gurobi (optional, faster for larger instances).

### 4c. Standalone SCED validation script

Create `testbed/validate_egret_sced.py`. Start with SCED (economic dispatch LP)
because it's simpler — no binary commitment decisions, all generators assumed online.
This validates the Egret ModelData construction and LMP extraction before adding
SCUC complexity.

```python
from egret.data.model_data import ModelData
from egret.models.unit_commitment import solve_unit_commitment

# Build ModelData from IEEE 9-bus parameters
# All generators assumed committed (no UC binaries)
md = ModelData({
    "system": {
        "time_keys": [1, 2, ..., 24],
        "time_period_length_minutes": 60,
        "baseMVA": 100.0,
    },
    "elements": {
        "generator": { ... },  # 3 thermal generators, fixed_commitment=1
        "bus": { ... },         # 9 buses with loads
        "branch": { ... },      # 9 branches with reactance + limits
    }
})

# Solve SCED (LP relaxation with all units committed → economic dispatch)
md_results = solve_unit_commitment(md, "highs", relaxed=True, return_model=True)

# Extract results
for name, gen in md_results[0].elements("generator", generator_type="thermal"):
    print(f"{name}: dispatch={gen['pg']['values']}")

for name, bus in md_results[0].elements("bus"):
    if 'lmp' in bus:
        print(f"{name}: lmp={bus['lmp']['values']}")
```

**SCED validation checks**:
- Dispatch sums to total load each hour
- LMPs differ across buses when a branch is congested
- LMPs are uniform (equal to marginal unit price) when no congestion
- Line flows respect thermal limits
- `system['total_cost']` matches expected range

### 4d. ASSUME ↔ Egret data adapter

Create `testbed/clearing/egret_adapter.py` with two functions:

#### `orders_to_model_data()`

```
ASSUME concept           → Egret ModelData field
─────────────────────────────────────────────────
order["volume"] > 0      → generator with p_max = volume
order["price"]           → generator p_cost (marginal → piecewise cost curve)
order["node"]            → generator bus
order["volume"] < 0      → bus p_load (aggregate demand at bus)
market_products          → system time_keys
grid_data["lines"]       → branch entries (reactance, rating_long_term)
grid_data["buses"]       → bus entries
generator_params         → ramp rates, min up/down, startup costs, p_min, initial_status
```

**Key design decision**: UC parameters (ramp rates, min up/down times, startup costs)
cannot be encoded in ASSUME's standard order format. They come from
`MarketConfig.param_dict["generator_params"]` — a dict keyed by generator name
with the full Egret generator parameter set. This is populated by the scenario loader
from the `powerplant_units.csv` extended columns.

For SCED, generators have `fixed_commitment` = 1 (all online). UC params are
optional — only ramp rates affect the SCED dispatch.

#### `model_results_to_orderbook()`

```
Egret result                        → ASSUME output
───────────────────────────────────────────────────
generator[g]["pg"]                  → order["accepted_volume"]
bus[b]["lmp"]                       → order["accepted_price"] (per node per time)
branch[l]["pf"]                     → flows dict (for visualization)
system["total_cost"]                → meta list entries
```

### 4e. ScedClearingRole implementation (FIRST)

Create `testbed/clearing/sced_clearing.py`. SCED is implemented first because:
1. It's a pure LP (no MIP), simpler to debug
2. Validates the full adapter pipeline (orders → ModelData → solve → results)
3. Produces LMPs from dual variables directly
4. The same adapter code is reused by SCUC later

```python
from assume.markets.base_market import MarketRole
from egret.data.model_data import ModelData
from egret.models.unit_commitment import solve_unit_commitment

class ScedClearingRole(MarketRole):
    required_fields = ["node"]

    def __init__(self, marketconfig):
        super().__init__(marketconfig)
        self.grid_data = marketconfig.param_dict.get("grid_data")
        self.generator_params = marketconfig.param_dict.get("generator_params", {})
        self.solver = marketconfig.param_dict.get("solver", "highs")

    def clear(self, orderbook, market_products):
        # 1. Convert ASSUME orders + grid → Egret ModelData
        #    All generators get fixed_commitment=1 (assumed online)
        model_data = orders_to_model_data(
            orderbook, market_products,
            self.grid_data, self.generator_params,
            fix_commitment=True,  # SCED: all units committed
        )

        # 2. Solve LP (continuous dispatch + duals for LMPs)
        md_results = solve_unit_commitment(
            model_data, self.solver, relaxed=True
        )

        # 3. Map back to ASSUME format
        return model_results_to_orderbook(
            md_results, orderbook, market_products, self.grid_data
        )
```

### 4f. ScucClearingRole implementation (SECOND)

Create `testbed/clearing/scuc_clearing.py`. Builds on SCED by adding binary
commitment decisions, then re-solving as LP for LMPs.

```python
class ScucClearingRole(MarketRole):
    required_fields = ["node"]

    def __init__(self, marketconfig):
        super().__init__(marketconfig)
        self.grid_data = marketconfig.param_dict.get("grid_data")
        self.generator_params = marketconfig.param_dict.get("generator_params", {})
        self.solver = marketconfig.param_dict.get("solver", "highs")
        self.last_commitments = {}  # Shared with RT SCED

    def clear(self, orderbook, market_products):
        # 1. Convert ASSUME orders + grid → Egret ModelData
        model_data = orders_to_model_data(
            orderbook, market_products,
            self.grid_data, self.generator_params,
            fix_commitment=False,  # SCUC: solve for commitments
        )

        # 2. Solve SCUC (MIP for commitments)
        md_mip = solve_unit_commitment(model_data, self.solver, relaxed=False)

        # 3. Fix commitments, re-solve as LP for LMPs
        for name, gen in md_mip.elements("generator", generator_type="thermal"):
            model_data.data["elements"]["generator"][name]["fixed_commitment"] = \
                gen["commitment"]

        md_lmp = solve_unit_commitment(model_data, self.solver, relaxed=True)

        # 4. Store commitments for RT SCED market
        self.last_commitments = {
            name: gen["commitment"]
            for name, gen in md_mip.elements("generator", generator_type="thermal")
        }

        # 5. Map back to ASSUME format
        return model_results_to_orderbook(
            md_lmp, orderbook, market_products, self.grid_data
        )
```

### 4g. Register clearing algorithms

In the testbed runner (and eventually in ASSUME's `__init__.py`):

```python
from testbed.clearing.sced_clearing import ScedClearingRole
from testbed.clearing.scuc_clearing import ScucClearingRole

# Register alongside built-in mechanisms
clearing_mechanisms["sced_clearing"] = ScedClearingRole
clearing_mechanisms["scuc_clearing"] = ScucClearingRole
```

### 4h. Two-settlement DA/RT coupling

**Commitment transfer mechanism**: After DA SCUC clears, `ScucClearingRole` stores
commitments in `self.last_commitments`. The RT `ScedClearingRole` retrieves these
via a shared reference through the World object:

```python
# In testbed_runner.py:
scuc_role = ScucClearingRole(dam_config)
sced_role = ScedClearingRole(rtm_config)
sced_role._scuc_role = scuc_role  # Direct reference for testbed

# ScedClearingRole can optionally accept fixed commitments from SCUC:
class ScedClearingRole(MarketRole):
    def clear(self, orderbook, market_products):
        commitments = getattr(self, '_scuc_role', None)
        fix_all = commitments is None  # standalone SCED: all committed
        ...
```

For the testbed MVP, this direct coupling is appropriate. For production ASSUME
integration, refactor to use World-level storage or the messaging system.

---

## Execution Sequence

### Immediate (Phase 1 + 2)
1. **Rebase** — stash working tree, `git rebase upstream/main`, resolve conflicts, pop stash
2. **Verify** — run `pytest tests/ -x`, confirm `nodal_clearing` importable
3. **Update copilot-instructions.md** — new strategy names, nodal clearing, forecasts rename, grid_utils, Egret section
4. **Commit** — single commit for copilot instructions update

### Short-term (Phase 3: Testbed)
5. Install pandapower, gridx-egret
6. **Phase 0**: `testbed/retrieve_ieee9.py` — fetch + validate IEEE 9-bus
7. **Phase 1**: `testbed/partition_zones.py` — zone assignment + seam identification
8. **Phase 2**: `testbed/build_assume_inputs.py` — generate ASSUME CSV scenario
9. **Phase 3**: `testbed/testbed_runner.py` — wire into ASSUME World, validate basic clearing

### Medium-term (Phase 4: Egret Integration — SCED first, then SCUC)
10. ✅ **4c**: `testbed/validate_egret_sced.py` — standalone Egret SCED on IEEE 9-bus (LP, LMPs)
11. ✅ **4d**: `testbed/clearing/egret_adapter.py` — ASSUME ↔ Egret data mapping
12. ✅ **4e**: `testbed/clearing/sced_clearing.py` — SCED MarketRole (LP dispatch + LMPs)
12a. ✅ **4e-RL**: `testbed/learning_runner.py` — RL agents (MATD3) learn strategic bidding in SCED
12b. ✅ **4e-viz**: `testbed/inspect_sced_results.ipynb` + `testbed/inspect_rl_results.ipynb` — results visualization
12c. ✅ **4e-BC**: `testbed/strategies/heuristics.py` — BC warm-start heuristics (forecast_markup replaces broken Cournot)
12d. ✅ **4e-finding**: RL agent discovered physical withholding strategy (bids at MC, withholds 80%+ capacity → $50 LMP)
12e. ✅ **4e-SAC**: SAC algorithm implemented, entropy-only config (no additive noise)
12f. ✅ **4e-PPO**: PPO algorithm implemented (DistActor + CriticPPO + RolloutBuffer)
12g. ✅ **4e-ProfitStrategy**: New reward function — pure profit + terrain shaping + guardrails
12h. ✅ **4e-3algo**: Three-algorithm comparison configs in scenarios.yaml (MATD3/SAC/PPO)
12i. ⬜ **4e-run**: Run and compare three algorithms on uncongested IEEE 9-bus
12j. ⬜ **4e-congestion**: Add congestion scenario with ProfitStrategy + two RL agents
13. ⬜ **4f**: `testbed/clearing/scuc_clearing.py` — SCUC MarketRole (MIP commitment + LP LMPs)
14. ⬜ **4g**: Register clearing algorithms
15. ⬜ **4h**: Two-settlement DA (SCUC) / RT (SCED) with forecast divergence on testbed

### Validation Checkpoints

| Checkpoint | What passes | Blocking? |
|------------|------------|-----------|
| Rebase clean | `pytest tests/ -x` passes | Yes |
| IEEE 9-bus loads | 9 buses, 3 gens, 3 loads, PF converges | Yes |
| ASSUME basic clearing | pay_as_clear simulation runs, prices correct | Yes |
| Nodal visualization | Circlize + topology plots render with bus colors | Yes |
| Egret SCED standalone | LP solves, LMPs differ under congestion | ✅ Done |
| SCED in ASSUME | `ScedClearingRole.clear()` returns valid 4-tuple | ✅ Done |
| RL in SCED | RL agents learn strategic bidding above MC | ✅ Done |
| BC heuristics | Closed-form warm-start targets (forecast_markup) | ✅ Done |
| RL convergence | Agent finds profitable strategy in peak hours | ✅ Done — agent learned physical withholding ($35 bid, 15-55 MW of 300 MW → forces $50 LMP) |
| SCED visualization | Circlize + topology + bid evolution notebooks | ✅ Done |
| Egret SCUC standalone | MIP solves, generators commit/decommit | Yes |
| SCUC in ASSUME | `ScucClearingRole.clear()` returns valid 4-tuple | Yes |
| Two-settlement | DA and RT LMPs diverge under forecast uncertainty | No (stretch) |

---

## File Layout (after all phases)

```
assume/                        # ASSUME framework (rebased onto upstream)
├── .github/
│   └── copilot-instructions.md    # Updated with post-rebase conventions
├── docs/
│   ├── egret_integration_research.md
│   ├── ieee9_testbed_plan.md
│   └── integration_plan.md        # This file
├── assume/
│   └── markets/
│       └── clearing_algorithms/
│           ├── __init__.py         # Eventually registers scuc/sced
│           ├── scuc_clearing.py    # Production version (post-testbed)
│           └── sced_clearing.py    # Production version (post-testbed)
└── testbed/                       # IEEE 9-bus testbed (parallel track)
    ├── retrieve_ieee9.py
    ├── partition_zones.py
    ├── build_assume_inputs.py
    ├── testbed_runner.py
    ├── validate_egret.py
    ├── clearing/
    │   ├── egret_adapter.py
    │   ├── scuc_clearing.py
    │   └── sced_clearing.py
    └── agents/
        └── testbed_seam_strategy.py
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Rebase conflicts in `loader_pypsa.py` | High | Medium | Resolve manually; local changes are well-understood |
| Egret+HiGHS solver issues | Medium | High | Egret tested with CBC/Gurobi; HiGHS support via Pyomo's `appsi_highs` may need testing |
| ASSUME event loop ordering (DA before RT) | Medium | Medium | Verify with testbed; ASSUME markets clear sequentially within timestep |
| Egret ModelData → ASSUME orderbook mapping edge cases | Medium | Medium | Validate with known pglib-uc benchmark first |
| pandapower case9 → PyPSA conversion losing parameters | Low | Low | Validate bus/gen/branch counts after conversion |

---

## Dependencies

```
# Already in ASSUME
pyomo
pypsa
pandas
numpy

# New
gridx-egret     # Egret SCUC/SCED formulations (BSD-3)
pandapower      # IEEE 9-bus retrieval (for testbed only)
highspy         # HiGHS solver (likely already installed)
```
