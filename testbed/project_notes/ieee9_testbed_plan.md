# IEEE 9-Bus ASSUME Testbed — Project Plan for Claude Code

This plan creates a parallel development track: an IEEE 9-bus testbed that validates
the ASSUME market clearing + seam scheduling pipeline end-to-end, while the real
WECC network continues through phases 6–8 of the Snakemake pipeline.

This is a permanent engineering scaffold — the test rig for validating market
mechanics during initial development and for ongoing regression testing and
experimentation after the real WECC network is wired into ASSUME.

---

## Phase 0 — Retrieve the IEEE 9-Bus Network

### What the IEEE 9-bus is

The WSCC (Western Systems Coordinating Council) 9-bus, 3-machine system. Originally
from P.M. Anderson and A.A. Fouad, *Power System Control and Stability*, 1977. It
has 9 buses, 3 generators (buses 1, 2, 3), 3 loads (buses 5, 7, 9), and 9 branches
(6 lines + 3 transformers). Small enough to debug by hand, large enough to exhibit
congestion under the right loading conditions.

### How to retrieve it

There are three retrieval paths, in order of preference:

#### Option A — pandapower (recommended)

```python
import pandapower.networks as pn
import pypsa

# Load the IEEE 9-bus from pandapower's built-in library
pp_net = pn.case9()

# Convert to PyPSA
network = pypsa.Network()
network.import_from_pandapower_net(pp_net)
```

pandapower ships with `case9()` which returns the full network with generators,
loads, lines, transformers, and bus data. The PyPSA import preserves topology and
electrical parameters. This is the cleanest path because the data is already
validated and consistent.

**Install**: `pip install pandapower --break-system-packages`

#### Option B — PyPSA's built-in examples

PyPSA ships three example networks:
- `pypsa.examples.ac_dc_meshed`
- `pypsa.examples.scigrid_de`
- `pypsa.examples.storage_hvdc`

None of these is the IEEE 9-bus. Use Option A instead.

#### Option C — manual construction from textbook data

If neither pandapower nor the import method works, construct the network manually
from the Anderson & Fouad data. The bus/branch/generator tables are reproduced in
every power systems textbook and in the MATPOWER case file `case9.m`. This is the
fallback.

### Validation after retrieval

After loading the network, verify:

```python
assert len(network.buses) == 9
assert len(network.generators) == 3
assert len(network.loads) == 3
assert len(network.lines) + len(network.transformers) >= 9
# Run a basic power flow to check consistency
network.pf()
assert network.buses_t.v_mag_pu.notna().all().all()
```

### File output

Save the retrieved network to:
```
data/testbed/ieee9/network/    # PyPSA CSV export (buses.csv, lines.csv, etc.)
```

Use `network.export_to_csv_folder("data/testbed/ieee9/network/")`.

---

## Phase 1 — Partition into Two Zones

### Zone assignment

Split the 9 buses into two zones to simulate a seam:

| Zone     | Buses         | Generators | Loads    | Rationale                         |
|----------|---------------|------------|----------|-----------------------------------|
| Zone A   | 1, 2, 4, 7, 8| Gen 1, 2   | Load 7   | Upper-left cluster in the diagram |
| Zone B   | 3, 5, 6, 9   | Gen 3      | Load 5, 9| Lower-right cluster               |

This partition creates a 2-generator vs 1-generator asymmetry (Zone A has more
supply diversity) and a 1-load vs 2-load asymmetry (Zone B has more demand). That
imbalance naturally produces a price gradient — Zone B clears higher under most
conditions — which gives the seam agent something to learn.

### Seam interface

The tie-lines connecting Zone A and Zone B become the seam. Based on the 9-bus
topology, the candidate seam lines are:

- Line 4→6 (if it exists in the specific data version)
- Line 7→8→9 path crossing the zone boundary
- Transformer branches bridging the zones

Identify which branches span Zone A → Zone B after loading. These become PyPSA Links
(or are retained as Lines with the `seam=True` flag). For the MVP testbed, select the
single highest-capacity tie-line and model it as the seam Link, consistent with how
`assume_config.py`'s `_find_seam_links()` discovers seams in the real network.

### Implementation

```python
# Add zone assignment to buses
network.buses["zone"] = network.buses.index.map({
    "bus_1": "zone_a", "bus_2": "zone_a", "bus_4": "zone_a",
    "bus_7": "zone_a", "bus_8": "zone_a",
    "bus_3": "zone_b", "bus_5": "zone_b", "bus_6": "zone_b",
    "bus_9": "zone_b",
})

# Identify seam lines (bus0 in zone_a, bus1 in zone_b or vice versa)
seam_mask = (
    network.lines.bus0.map(network.buses.zone) !=
    network.lines.bus1.map(network.buses.zone)
)
seam_lines = network.lines[seam_mask]
```

### Output

Save zone-partitioned network to:
```
data/testbed/ieee9/network_zoned/
```

Also produce a `ba_group_testbed.csv` matching the format of
`config/ba_group_SCED_EIM.csv` so that `zones.py` can consume it:

```csv
ba_code,group,stage,year,market_type
zone_a,ZONE_A_MKT,testbed,,organized_rt
zone_b,ZONE_B_MKT,testbed,,organized_rt
```

---

## Phase 2 — Build ASSUME Scenario Inputs

ASSUME's CSV loader expects a specific directory layout. Create these files under
`data/testbed/ieee9/assume_scenario/`:

### 2a. `powerplant_units.csv`

One row per generator. Columns must match what `loader_csv.py` expects:

| Column             | Source                                    |
|--------------------|-------------------------------------------|
| `name`             | Generator name from PyPSA (e.g., "gen_1") |
| `technology`       | "nuclear", "gas", "coal" — assign varied  |
| `bidding_type`     | "naive_eom" for MVP (price-taking)        |
| `fuel_type`        | Matching technology                       |
| `emission_factor`  | Standard values per fuel                  |
| `max_power`        | p_nom from PyPSA generators (MW)          |
| `min_power`        | 0 or fraction of p_nom                    |
| `efficiency`       | 0.33–0.55 depending on tech               |
| `node`             | Bus name — CRITICAL for nodal clearing    |
| `unit_operator`    | "operator_zone_a" or "operator_zone_b"    |

Assign different marginal costs to create price separation:
- Zone A generators: ~$25/MWh and ~$35/MWh (cheaper zone)
- Zone B generator: ~$50/MWh (more expensive zone)

This guarantees a price gradient for the seam agent to exploit.

### 2b. `demand_units.csv`

One row per load:

| Column         | Source                           |
|----------------|----------------------------------|
| `name`         | Load name from PyPSA             |
| `technology`   | "demand"                         |
| `bidding_type` | "naive_da" or "naive_eom"        |
| `max_power`    | p_set from PyPSA loads (MW)      |
| `min_power`    | Same (inelastic demand for MVP)  |
| `node`         | Bus name                         |

### 2c. `fuel_prices.csv`

Time-indexed fuel prices. For the testbed, use flat prices:

```csv
datetime,uranium,gas,coal
2025-01-01 00:00,5.0,30.0,15.0
...
```

Hourly rows for the simulation period (e.g., one week = 168 rows).

### 2d. `buses.csv` and `lines.csv`

Export directly from the zoned PyPSA network. ASSUME's `complex_clearing.py`
reads these via `grid_data` in the MarketConfig's `param_dict`:

```python
grid_data = {
    "buses": network.buses,
    "lines": network.lines,
}
```

### 2e. `config.yaml` (scenario config)

```yaml
scenario:
  start_date: "2025-01-01 00:00"
  end_date: "2025-01-07 23:00"

markets:
  zone_a_eom:
    operator: "zone_a_operator"
    product_type: "energy"
    opening_frequency: "1h"
    opening_duration: "1h"
    market_mechanism: "pay_as_clear"
    products:
      - duration: "1h"
        count: 1
        first_delivery: "1h"

  zone_b_eom:
    operator: "zone_b_operator"
    product_type: "energy"
    opening_frequency: "1h"
    opening_duration: "1h"
    market_mechanism: "pay_as_clear"
    products:
      - duration: "1h"
        count: 1
        first_delivery: "1h"
```

---

## Phase 3 — Wire into ASSUME's World

### 3a. Create `testbed_runner.py`

This script assembles the ASSUME World from the IEEE 9-bus scenario. It is the
testbed equivalent of `assume_config.py`'s `build_world()`:

```python
from assume import World, MarketConfig, MarketProduct
from assume.common.forecasts import NaiveForecast
from datetime import timedelta
import pypsa

def build_testbed_world():
    """Build an ASSUME World from the IEEE 9-bus two-zone testbed."""
    world = World(...)

    # 1. Load the zoned PyPSA network
    network = pypsa.Network("data/testbed/ieee9/network_zoned/")

    # 2. Register two market zones
    #    Each zone gets a MarketConfig with pay_as_clear mechanism
    #    Pass grid_data (buses, lines) for network-constrained clearing

    # 3. Add generator units to appropriate zone's market
    #    Use NaiveForecast for marginal costs (flat price-taking)

    # 4. Add demand units to appropriate zone's market

    # 5. Add the seam agent
    #    Register SeamSchedulingStrategy on the tie-line Link
    #    Observation: price forecasts from both zones
    #    Action: MW schedule on the seam link

    # 6. Run
    world.run()
```

### 3b. Key integration points to audit

Before writing `build_testbed_world()`, read and understand these ASSUME internals:

1. **`assume/world.py`** — How `World` is instantiated, how markets and units are
   added via `world.add_market_operator()` and `world.add_unit_operator()`

2. **`assume/common/market_objects.py`** — The `MarketConfig` dataclass, especially:
   - `market_products: list[MarketProduct]`
   - `param_dict` — where `grid_data`, `solver`, `zones_identifier` go
   - `market_mechanism` — string name mapped to clearing function

3. **`assume/scenario/loader_csv.py`** — How CSV scenarios populate the World. This
   is the reference implementation for what `build_testbed_world()` must replicate.

4. **`assume/scenario/loader_pypsa.py`** — How PyPSA networks map to ASSUME. In
   particular, how generators become bidding units and how `generators_t.marginal_cost`
   maps to fuel_prices.

5. **`assume/markets/clearing_algorithms/complex_clearing.py`** — The `ComplexClearingRole`
   class. Understand how `grid_data` is consumed: it extracts `buses`, `lines`, builds
   the incidence matrix, and sets up per-node energy balance constraints whose duals
   become LMPs.

### 3c. Two-zone simultaneous clearing

This is the hardest integration point for the testbed and for the real project.
ASSUME's `World` runs a single event loop. Markets clear in sequence within each
timestep unless explicitly parallelized. For the MVP (both testbed and dissertation),
"simultaneous" means:

- Both markets open, collect bids, and clear within the same simulation timestep
- The seam agent submits its schedule to both markets *before* either clears
- Neither market sees the other's clearing results before its own clearing

Verify that ASSUME's event loop supports this. If markets clear sequentially within
a timestep, confirm that the order does not leak information (i.e., Zone A's clearing
price is not available to Zone B's bidders within the same timestep).

---

## Phase 4 — Add the Seam Agent

### 4a. Adapt `SeamSchedulingStrategy`

The existing stub at `western_seams/agents/seam_strategy.py` is designed for the real
WECC network. For the testbed, create a simplified version:

```
testbed/agents/testbed_seam_strategy.py
```

Observation space (4 dimensions for testbed, vs 6 for real):
- Zone A forecast price ($/MWh)
- Zone B forecast price ($/MWh)
- Hour-of-day sine
- Hour-of-day cosine

Action space: scalar ∈ [-1, 1], scaled to [-ATC, +ATC] MW on the seam link.

Reward: `spread_captured = (price_b - price_a) * scheduled_mw` when scheduling
A→B, or `(price_a - price_b) * |scheduled_mw|` when scheduling B→A. This is
consistent with the real project's reward design.

### 4b. Training loop

Use ASSUME's built-in RL integration with stable-baselines3. The testbed should:

1. Run 50–100 episodes (not 1000 — this is a smoke test)
2. Log cumulative reward per episode
3. Verify reward increases over episodes (learning signal exists)
4. Compare against a zero-schedule baseline (agent does nothing)

---

## Phase 5 — Research: Modifying ASSUME for Nodal Real-Time Markets

This phase is research, not code. The goal is to document what ASSUME supports today
vs what needs to change for a US-style 5-minute nodal real-time market, so that the
testbed can later be extended (or so the findings inform the dissertation's market
design choices).

### 5a. What ASSUME already supports for nodal pricing

ASSUME's `complex_clearing.py` already computes locational marginal prices (LMPs).
The mechanism:

1. The clearing algorithm builds a Pyomo MILP with per-node energy balance constraints
2. After solving the MILP (with binary variables for block/linked bids), it fixes
   binaries and re-solves as an LP
3. Dual variables of the energy balance constraints are extracted as nodal prices:
   ```python
   for node in self.nodes:
       market_clearing_prices[node] = {
           t: instance.dual[instance.energy_balance[node, t]]
           for t in instance.T
       }
   ```
4. When `zones_identifier` is set in `param_dict`, buses are aggregated into zones
   and the incidence matrix is zone-to-zone. When it is omitted, each bus is its own
   node, giving full nodal LMPs.

**Bottom line**: ASSUME can already produce bus-level LMPs through complex clearing.
The nodal pricing math is in place. What is missing is the surrounding infrastructure
for a US-style real-time market.

### 5b. What needs to change for 5-minute real-time market intervals

**Current state**: All ASSUME loaders and examples use hourly products:
```python
MarketProduct(duration=timedelta(hours=1), count=24, first_delivery=timedelta(hours=1))
```

**Required change**: Define sub-hourly products:
```python
MarketProduct(duration=timedelta(minutes=5), count=288, first_delivery=timedelta(minutes=5))
```

The `MarketProduct` dataclass accepts any `timedelta` for duration, so this is
architecturally supported but has never been tested. Risks:

- **Solver performance**: 288 timesteps × N buses × M generators creates a much
  larger MILP than 24 × N × M. For the 9-bus testbed this is fine. For a 50-bus
  clustered WECC network, benchmark solve times before committing to 5-min clearing.

- **Data pipeline**: `loader_csv.py` and `loader_pypsa.py` infer frequency from the
  datetime index. Sub-hourly time series must be explicitly 5-minute indexed. Fuel
  prices, demand profiles, and availability factors all need 5-min granularity (or
  interpolation logic).

- **Bidding strategies**: Current naive strategies bid a single price/volume pair.
  For 5-min markets, strategies may need intra-hour ramp awareness. For the testbed
  MVP, flat bids are fine — this is a structural concern for later.

### 5c. What needs to change for ramp rate constraints

US real-time markets enforce generator ramp limits (MW/min). ASSUME's complex
clearing does not currently include ramp constraints. Adding them requires:

```python
# In the Pyomo model (complex_clearing.py):
# For each generator g, consecutive timesteps t and t+1:
model.ramp_up = pyo.ConstraintList()
model.ramp_down = pyo.ConstraintList()
for g in generators:
    for t in range(len(timesteps) - 1):
        model.ramp_up.add(
            dispatch[g, t+1] - dispatch[g, t] <= ramp_rate_up[g] * dt_minutes
        )
        model.ramp_down.add(
            dispatch[g, t] - dispatch[g, t+1] <= ramp_rate_down[g] * dt_minutes
        )
```

This is a meaningful extension. For the testbed MVP, skip it. For the dissertation,
it matters only if the research question requires intra-hour dynamics. Since the
dissertation focuses on inter-market seam scheduling (hourly to sub-hourly), ramp
constraints are a Phase 2 enhancement.

### 5d. What needs to change for reserve products

US ISOs co-optimize energy and ancillary services (spinning reserve, regulation,
non-spinning reserve) in the real-time market. ASSUME supports multiple `product_type`
values (`"energy"`, `"capacity"`, `"heat"`) but does not co-optimize them within a
single clearing. Adding reserve co-optimization requires:

- A new `product_type: "reserve"` with its own MarketProduct
- Joint clearing across energy + reserve in a single Pyomo model
- Reserve requirement constraints (zonal or system-wide)

This is out of scope for both the testbed and the MVP. Note it as a limitation.

### 5e. Redispatch vs nodal: two paths to congestion management

ASSUME offers two architectures for network-aware market clearing:

1. **Complex clearing with nodal prices** — single-stage optimization. Bids clear
   subject to transmission constraints. LMPs emerge from duals. This is the US ISO
   model (PJM, CAISO, SPP, MISO).

2. **Simple clearing + redispatch** — two-stage. First, a copperplate merit-order
   clearing ignoring network constraints. Then, a redispatch stage (using PyPSA OPF)
   resolves congestion by re-dispatching generators. This is the European model
   (Germany, Nordic).

For the western seams project, the US ISO model (option 1) is the correct architecture.
Use `complex_clearing` with full nodal resolution (no `zones_identifier`). The
`redispatch.py` path is informative for understanding ASSUME's internals but is not
the target market design.

### 5f. Recommended testbed experiments

Once the IEEE 9-bus testbed runs, use it to validate these ASSUME behaviors before
scaling to the real network:

| Experiment | What it tests | How to configure |
|------------|--------------|------------------|
| Hourly copperplate clearing | Basic market clearing works | `simple.py`, no grid_data |
| Hourly nodal clearing | LMPs emerge from network constraints | `complex_clearing.py`, with grid_data, no zones_identifier |
| Hourly zonal clearing | Zone prices differ from nodal | `complex_clearing.py`, with zones_identifier="zone" |
| Two-zone simultaneous clearing | Both zones clear in same timestep, seam agent submits before clearing | Two MarketConfigs, seam agent registered |
| Seam agent learning | Reward increases over episodes | RL training with PPO, 50+ episodes |
| 5-min clearing (stretch) | Sub-hourly works at all | `MarketProduct(timedelta(minutes=5), ...)`, benchmark solver |

Run experiments in this order. Each validates a prerequisite for the next.

---

## Phase 6 — Egret SCUC/SCED Clearing Algorithm

See `docs/egret_integration_research.md` for full research notes, source materials,
and architectural analysis. This phase implements Egret as a new ASSUME clearing
algorithm, replacing the simple merit-order clearing with US ISO-style SCUC/SCED.

### 6a. Install and validate Egret standalone

```bash
pip install gridx-egret --break-system-packages
```

Write a standalone script `testbed/validate_egret.py` that:

1. Loads the IEEE 9-bus network from Phase 0
2. Converts it to Egret's `ModelData` format (nested dict with `system`, `elements`)
3. Populates generator parameters:
   - `p_min`, `p_max` from PyPSA generators
   - `ramp_up_60min`, `ramp_down_60min` (assign reasonable values: 50% of p_max/hr)
   - `min_up_time`, `min_down_time` (2–4 hours)
   - `startup_cost` with time-dependent stages: `[(4, hot), (10, warm), (24, cold)]`
   - `p_cost` as piecewise-linear cost curve derived from marginal costs
   - `initial_status` and `initial_p_output`
4. Populates bus loads from PyPSA loads (with a 24-hour demand profile)
5. Populates branches from PyPSA lines (reactance, thermal rating)
6. Calls `create_tight_unit_commitment_model()` to build the Pyomo SCUC
7. Solves with HiGHS (or Gurobi if available)
8. Extracts and prints: commitment schedule, dispatch per generator per hour, LMPs

Validation checks:
- All generators committed during peak hours
- At least one generator decommits during off-peak (set loads low enough)
- LMPs vary across buses when a line binding constraint is active
- Total generation = total load + losses (should balance within solver tolerance)

### 6b. Build the ASSUME → Egret data adapter

Create `testbed/clearing/egret_adapter.py` with:

```python
def orders_to_model_data(
    orderbook: Orderbook,
    market_products: list[MarketProduct],
    grid_data: dict,           # {"buses": DataFrame, "lines": DataFrame}
    generator_params: dict,    # {gen_name: {ramp_up, min_up_time, startup_cost, ...}}
) -> ModelData:
    """Convert ASSUME orderbook + grid to Egret ModelData."""
```

The adapter must handle:
- Supply orders → generator entries with cost curves derived from bid prices
- Demand orders → bus load entries (aggregate per bus per timestep)
- Grid topology → branch entries with reactance and thermal limits
- UC parameters → sourced from `generator_params` dict (passed via
  `MarketConfig.param_dict["generator_params"]`)
- Time series → map `MarketProduct` (start, end) tuples to Egret time keys

And the reverse mapping:

```python
def model_results_to_orderbook(
    model: pyo.ConcreteModel,
    model_data: ModelData,
    orderbook: Orderbook,
    market_products: list[MarketProduct],
) -> tuple[Orderbook, Orderbook, list[dict], dict]:
    """Extract Egret solution back into ASSUME's 4-tuple format."""
```

This must:
- Map generator dispatch levels back to `accepted_volume` on supply orders
- Extract LMPs from power balance duals → `accepted_price` per node per timestep
- Build `meta` list (one dict per product×node with price, volumes)
- Extract line flows → `flows` dict
- Separate accepted vs rejected orders based on dispatch > 0

### 6c. Implement `ScucClearingRole`

Create `testbed/clearing/scuc_clearing.py`:

```python
from assume.markets.base_market import MarketRole

class ScucClearingRole(MarketRole):
    def __init__(self, marketconfig):
        super().__init__(marketconfig)
        self.grid_data = marketconfig.param_dict.get("grid_data")
        self.generator_params = marketconfig.param_dict.get("generator_params", {})
        self.solver = marketconfig.param_dict.get("solver", "appsi_highs")
        self.formulation = marketconfig.param_dict.get("formulation", "tight")

    def clear(self, orderbook, market_products):
        # 1. Convert ASSUME orders → Egret ModelData
        model_data = orders_to_model_data(
            orderbook, market_products,
            self.grid_data, self.generator_params
        )

        # 2. Build SCUC Pyomo model (select formulation)
        model = create_tight_unit_commitment_model(model_data)

        # 3. Solve
        solver = pyo.SolverFactory(self.solver)
        results = solver.solve(model)

        # 4. Store commitment decisions for RT market
        self.last_commitments = extract_commitments(model, model_data)

        # 5. Map back to ASSUME format
        return model_results_to_orderbook(
            model, model_data, orderbook, market_products
        )
```

### 6d. Implement `ScedClearingRole`

Create `testbed/clearing/sced_clearing.py`:

```python
class ScedClearingRole(MarketRole):
    def __init__(self, marketconfig):
        super().__init__(marketconfig)
        self.grid_data = marketconfig.param_dict.get("grid_data")
        self.generator_params = marketconfig.param_dict.get("generator_params", {})
        self.commitment_source = marketconfig.param_dict.get("commitment_source")
        self.solver = marketconfig.param_dict.get("solver", "appsi_highs")

    def clear(self, orderbook, market_products):
        # 1. Get commitments from DA market
        commitments = self._get_da_commitments()

        # 2. Convert to Egret ModelData with fixed commitments
        model_data = orders_to_model_data(
            orderbook, market_products,
            self.grid_data, self.generator_params
        )
        for gen, status in commitments.items():
            model_data["elements"]["generator"][gen]["fixed_commitment"] = status

        # 3. Build and solve (LP since commitments fixed)
        model = create_tight_unit_commitment_model(model_data)
        solver = pyo.SolverFactory(self.solver)
        results = solver.solve(model)

        # 4. Map back to ASSUME format
        return model_results_to_orderbook(
            model, model_data, orderbook, market_products
        )
```

### 6e-bis. WithholdingStrategy for economic withholding via RL

The standard EnergyLearningStrategy uses 2 actions for a 2-part bid (Pmin price + flex
price), with volumes hardcoded to min_power and max_power - min_power. This limits
market power to price manipulation only.

In nodal SCED markets, **economic withholding** is the more powerful mechanism: offer
a reduced volume at a competitive price so that a more expensive generator becomes
marginal, raising the LMP.

The WithholdingStrategy (`testbed/strategies/withholding_strategy.py`) reinterprets
the 2 action dimensions as:
- action[0] → bid price (scaled by max_bid_price)
- action[1] → volume fraction (0 = offer nothing, 1 = offer max_power)

The agent submits a single bid per timestep with `bid_type="single"`. The Egret adapter
translates the volume into `p_max`, so the LP naturally handles withholding.

Key design decisions:
- **min_power=0** for all generators (no must-run minimum) — allows full volume control
- **Single RL agent** (gen_0 only) — simplifies learning dynamics
- **50 episodes, 24h periods** — configured via scenarios.yaml "withholding" scenario
- **Registered as** `powerplant_energy_withholding` in learning_runner.py

### 6f. Register new clearing algorithms in testbed runner

Update `testbed/testbed_runner.py` to register the Egret-based clearing:

```python
from testbed.clearing.scuc_clearing import ScucClearingRole
from testbed.clearing.sced_clearing import ScedClearingRole

world = World(
    clearing_mechanisms={
        **world.clearing_mechanisms,         # keep existing
        "scuc_clearing": ScucClearingRole,   # add SCUC
        "sced_clearing": ScedClearingRole,   # add SCED
    }
)
```

### 6f. Test: SCUC clearing on IEEE 9-bus

Run the testbed with `market_mechanism="scuc_clearing"` and verify:
- Clearing results match standalone Egret validation (Phase 6a)
- LMPs match `complex_clearing.py` when all generators are committed
- LMPs differ when commitment decisions change the merit order

---

## Phase 7 — Two-Settlement DA/RT Market

### 7a. Configure two markets in testbed runner

```python
# Day-Ahead market: hourly SCUC, clears once per day
dam_config = MarketConfig(
    market_id="dam",
    market_mechanism="scuc_clearing",
    market_products=[MarketProduct(duration=timedelta(hours=1), count=24, ...)],
    param_dict={
        "grid_data": grid_data,
        "generator_params": gen_params,
        "solver": "appsi_highs",
    }
)

# Real-Time market: 5-min or hourly SCED, clears every interval
rtm_config = MarketConfig(
    market_id="rtm",
    market_mechanism="sced_clearing",
    market_products=[MarketProduct(duration=timedelta(hours=1), count=1, ...)],
    param_dict={
        "grid_data": grid_data,
        "generator_params": gen_params,
        "solver": "appsi_highs",
        "commitment_source": "dam",
    }
)
```

### 7b. Implement DA → RT commitment coupling

The RT clearing algorithm needs the DA commitment schedule. Options:

**Option A — World-level storage**: After DA clears, store commitments in a shared
dict on the World object. RT clearing reads from it. Simplest approach.

**Option B — Market result query**: RT clearing queries the DA market's last results
via ASSUME's messaging system. More correct but harder to implement.

**Option C — File-based**: DA writes commitments to a parquet/CSV, RT reads. Simple
but couples via filesystem.

For the testbed, use Option A. For the dissertation, refactor to Option B.

### 7c. Implement forecast divergence

Create two forecast profiles for the IEEE 9-bus testbed:

```python
# DA forecast: load prediction made 24 hours ahead (with error)
da_load_forecast = actual_load * (1 + np.random.normal(0, 0.05, size=24))

# RT forecast: load prediction made 1 hour ahead (more accurate)
rt_load_forecast = actual_load * (1 + np.random.normal(0, 0.01, size=24))
```

Wire these into separate `NaiveForecast` instances for the DA and RT markets.
Generators bidding into the DA market see `da_load_forecast`; generators bidding
into the RT market see `rt_load_forecast`.

For renewables (when added): DA wind/solar forecast has 15–20% RMSE, RT has 5%.
This is the primary driver of DA/RT price divergence in real ISO markets.

### 7d. Test: two-settlement economics

Run a full day (24 DA hours, 24 RT hours) and verify:
- DA SCUC commits generators based on DA forecast
- RT SCED dispatches against actual/updated load with fixed DA commitments
- When DA over-committed (forecast load > actual), RT prices are low
- When DA under-committed (forecast load < actual), RT prices spike
- Two-settlement calculation: DA payment + RT deviation at RT price

### 7e. Seam agent in two-settlement context

The seam agent must now participate in both markets:
- Submit DA schedule (MW per hour for next day) into DA SCUC
- Submit RT adjustments (MW per interval) into RT SCED
- DA schedule = planned seam transfer, RT = actual transfer

The agent's observation space expands:
- DA price forecasts (both zones)
- RT price forecasts (both zones, updated)
- DA commitment status (which generators committed in each zone)
- Current transfer schedule from DA

The reward is now two-settlement: DA spread capture + RT deviation value.

---

## Phase 8 — Integration Checkpoint (Extended)

When the testbed produces all of the following, the initial validation is complete
and the real WECC network can be wired into ASSUME. The testbed remains useful for
ongoing regression testing and for experimenting with new market designs or clearing
algorithms without waiting for the full pipeline to run:

### Phases 0–5 (basic ASSUME integration):
- [ ] Two zones clear simultaneously with network constraints
- [ ] LMPs differ between zones due to congestion on seam lines
- [ ] Seam agent submits schedules before clearing, receives price signals after
- [ ] Agent reward increases over 50+ training episodes
- [ ] Agent outperforms zero-schedule baseline on cumulative spread captured

### Phases 6–7 (Egret SCUC/SCED + two-settlement):
- [ ] Egret SCUC produces valid commitment schedule on IEEE 9-bus
- [ ] SCUC LMPs match `complex_clearing.py` when all units committed
- [ ] SCUC decommits a generator under low-load conditions
- [ ] RT SCED runs with fixed DA commitments and produces different LMPs
- [ ] DA and RT prices diverge under forecast uncertainty
- [ ] Two-settlement accounting balances (DA payment + RT deviation)
- [ ] Seam agent participates in both DA and RT markets

At that point, swap in the real clustered WECC network from the Snakemake pipeline
(phases 6–8 outputs), replace the testbed CSVs with real generator/load data, and
the ASSUME integration carries over.

---

## File Layout

```
western-seams/
├── data/
│   └── testbed/
│       └── ieee9/
│           ├── network/              # Raw PyPSA export of IEEE 9-bus
│           ├── network_zoned/        # Zone-partitioned version
│           └── assume_scenario/      # ASSUME CSV inputs
│               ├── powerplant_units.csv
│               ├── demand_units.csv
│               ├── fuel_prices.csv
│               ├── buses.csv
│               ├── lines.csv
│               └── config.yaml
├── testbed/
│   ├── retrieve_ieee9.py             # Phase 0: fetch + validate
│   ├── partition_zones.py            # Phase 1: zone assignment + seam ID
│   ├── build_assume_inputs.py        # Phase 2: generate ASSUME CSVs
│   ├── testbed_runner.py             # Phase 3: build_world + run
│   ├── validate_egret.py             # Phase 6a: standalone Egret validation
│   ├── clearing/
│   │   ├── egret_adapter.py          # Phase 6b: ASSUME ↔ Egret data mapping
│   │   ├── scuc_clearing.py          # Phase 6c: SCUC MarketRole
│   │   └── sced_clearing.py          # Phase 6d: SCED MarketRole (fixed commits)
│   └── agents/
│       └── testbed_seam_strategy.py  # Phase 4: simplified seam agent
├── docs/
│   ├── ieee9_testbed_plan.md         # This file
│   └── egret_integration_research.md # Egret research notes + source materials
└── ...
```

---

## Dependencies

Verify these are installed before starting:

```bash
pip install pandapower --break-system-packages    # IEEE case retrieval
pip install assume-framework[all] --break-system-packages  # ASSUME + PyPSA + learning
pip install gridx-egret --break-system-packages   # Egret SCUC/SCED formulations
```

If `assume-framework` is installed from the local clone at `../assume/`, ensure
it is in editable mode (`pip install -e ../assume/[all]`).

---

## Sequencing Relative to Main Pipeline

This testbed is fully independent of the Snakemake pipeline. It does not read from
`data/raw/` or `data/processed/`. It can be developed in parallel with phases 6–8
(generator attachment, load attachment, renewable profiles) without conflicts.

The only shared code is:
- `western_seams/markets/assume_config.py` — the testbed runner is a simplified
  version of `build_world()`, and lessons learned should flow back into it
- `western_seams/agents/seam_strategy.py` — the testbed strategy is a simplified
  version; once validated, the real strategy's `calculate_bids()` method should be
  completed using the same patterns

When phases 6–8 complete and the real network has generators and loads, wire it into
ASSUME by updating `assume_config.build_world()` using patterns proven in
`testbed_runner.py`. The testbed continues to serve as a fast-running regression
suite and a sandbox for experimenting with new market designs, clearing algorithms,
or agent architectures without full pipeline overhead.
