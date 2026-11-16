# WSCC 9-Bus Teaching Notebooks — Orientation

Two **standalone teaching notebooks** on the WSCC / IEEE 9-bus network that make
nodal pricing, congestion rent, and market-seams ideas concrete. They are developed
here in the ASSUME testbed and published to a Google-Colab GitHub repo for students.
This file orients a fresh Claude Code session to the whole ecosystem.

---

## 1. The three locations (and how they connect)

| Role | Path | Notes |
|---|---|---|
| **Source / dev** (edit here) | `C:\Users\docke\Projects\ASSUME\assume\testbed\` | The notebooks + modules live here; runs in the `assume` conda env. This is the source of truth. |
| **Publish / Colab clone** | `C:\Users\docke\Projects\wscc9-illustrative-models\` (plural) | A git clone of `github.com/pdockery/wscc9-illustrative-model` (**singular** — cosmetic mismatch). Re-staged *from* the testbed; never edited directly. |
| **Academic source material** | `C:\Users\docke\Oxford\OneDrive - Nexus365\MSc Energy System\Thesis\notes\market_design_lab\` | The papers, the assignment, and the seams write-up the notebooks teach (see §7). |

Data flow: **edit in testbed → execute → re-stage into the repo clone → push → live on Colab.**

---

## 2. The files

**Notebooks** — a numbered learning pathway (refactored 2026-06-17 onto shared
libraries; see `project_wscc9_library_refactor`). Hundreds digit = difficulty tier
(100 fundamentals / 200 core issue / 300 advanced); last digit = track (congestion
x01, seams x02):
- `101_nodal_market_fundamentals.ipynb` — **issue-neutral fundamentals**: build the
  9-bus case, nodal LMPs + congestion rent, **transport (net-interchange) constraints
  + transfer rent**, **self-schedules**, the **three settlement ledgers**, and a
  **sandbox** (fully configurable per-bus gen/load — add gen to a load-only bus, cost
  & volume overrides — plus line ratings, a transfer cap, and a self-schedule).
- `112_three_ba_fundamentals.ipynb` — fundamentals: a **third balancing authority**
  and **single-node BAs** (co-located gen + load) on the three-BA fleet
  (`FLEET_3BA`/`LOADS_3BA`: cheap units at nodes 6/8, gen+load at 2/3). Issue-neutral;
  3-BA dispatch + per-BA settlement (rent in TOTAL) + autarky-vs-unified.
- `201_congestion_revenue_allocation.ipynb` — two BAs; how congestion (and transfer)
  revenue is **allocated** (Method 1/2, T1/T2).
- `202_market_seams.ipynb` — two market footprints; three seam issues (interference,
  accommodation, interchange) + risk + seam ledger.
- `212_two_market_transfer.ipynb` — seams track: **two markets on three BAs**.
  Market A = two non-electrically-connected BAs (nodes 2,3) coordinated by a
  **transfer** (a 2-node transport via `build_transport_network`, μ_T = transfer
  rent); Market B **wheels** it as a self-schedule (`exo`) and keeps the congestion
  rent (`wheeling_settlement`); + inefficient accommodation (Market B de-rates the
  3↔2 path lines, margin ladder). The Markets+/EDAM seam.
- `301_two_settlement_congestion.ipynb` — advanced: the moving day-ahead/real-time
  market boundary, exogenous→endogenous, a contingency + its accommodation; the 5-6
  interface is a **parallel double circuit** (`build_network(split_5_6=True)`). (Its §2
  boundary-map `lcol` was fixed 2026-06-18 to colour a monitored line by its BA only when
  that BA participates the active settlement — `LINE_BA.get(l) in market` — so BA-2 ties
  monitored day-ahead no longer render blue in the day-ahead panel.)
- `302_two_market_contingency.ipynb` — advanced seams: **two co-equal markets** and a
  **contingency in the other market** (the two-MARKET analog of 301). Market A {2,8,7,6,3}
  cheap exporter + monitored 5-6 circuit `line_2_ba1`; Market B {1,9,4,5} importer + the
  contingency circuit `line_2_ba2`. Market A is the **managing** market (clears with B's
  215 MW generic import as exo, so it can enforce N-1). Three arrangements — **blind**
  (post-ctg overload, $8,475), **accommodation** (ATC reservation, secure-but-costly,
  $9,713), **coordination/unified** (B's unit ramps, cheapest secure, $9,018; worth
  $695/h) — each rendered as the **four views** (Market A / Market B / Combined / Unified)
  via a `four_views()` helper. LODF 0.60, N-1 de-rate 46.9 (matches 301). + config map,
  formulation, per-market settlement, cross-market position ledger, contingency accounting.

**Config visualization (2026-06-18):** the moving-boundary/footprint map (colour buses by
market, lines by the monitoring market, annotate seams/transfer) is now present wherever
node-in-market config is editable: 202 §2 (cell 14, pre-existing) + its sandbox (cell
566e8617); 212 §1 (`config-map`) + its sandbox (`sandbox-map`); 302 §1 (`config-map`).

> **Downstream template (201/202/301):** they assume 101 was read (no PTDF print, no
> admittance plot) and never cross-reference the *other* issue notebook. Cell 1 =
> imports; Cell 2 = visible config + the two-panel composite; Cell 3 = issue
> formulation (own internal eq numbering). They were produced by **wrapper-binding**:
> each definition cell became a thin cell that binds the library function to the
> notebook's footprint `fp` (e.g. `ba_settlement = lambda res, loads:
> ra.ba_settlement(fp, res, loads)`), so the tuned issue cells kept working
> byte-for-byte. Edit methodology in the **libraries**, config in the notebook cells.

**Modules** — the four primitive modules plus four teaching libraries (all flat
top-level `.py`, fetched by the Colab bootstrap):

*Teaching libraries (2026-06-17):*
- `wscc9_model.py` — `DEFAULT_GEN_FLEET`/`DEFAULT_LOADS`, `FLEET_3BA`/`LOADS_3BA`
  (the 3-BA scenario for 112/212), `build_network(line_ratings, split_5_6=False)`,
  `build_transport_network(buses, ratings)` (a 2-bus transfer-only net for
  non-connected BAs; line dual = μ_T), `make_engine(gen_fleet=, loads=)`,
  `shift_factors`/`sf_table`, layout consts (`COORDS`/`RING_ORDER`/`ROTATION_DEG`/
  `CENTER_BUS`/`BUS_COLORS`).
- `footprints.py` — `Footprints` + `make(pt, defs, colors, manage=|monitored=,
  tie_label='tie'|'seam')`; N-way (2 or 3+ footprints; single-node BAs OK). Two
  partitions can coexist (e.g. a BA-level `fp_ba` and a market-level `fp_mkt`).
- `revenue_allocation.py` — `ba_settlement`, `allocate_congestion_rent` (M1/M2),
  `compare_methods`, `allocate_transfer_rent` (T1/T2), `transfer_ledger`,
  `autarky_vs_unified`, `position_ledger`, `wheeling_settlement` (transfer rent in
  the transport market + congestion rent in the wheeling market), `independent_clear`,
  `solve_with_transfer`; all take a `Footprints` first arg.
- `wscc9_figures.py` — `footprint_figure` (unifies CRA `example_figure` + seams
  `market_figure`; `highlight=`/`dim_buses=`/`exo_sched=`/`bus_colors=`/`bus_coords=`/
  `ring_order=`/`center_bus=` — the coord overrides let it render the 2-bus transport
  net), `composite_figure`, `transfer_figure`/`draw_net_dispatch`. Self-schedules drawn
  grey-at-LMP (price-taking).

*Primitive modules:*
- `ieee9_network.py` — builds the 9-bus case from `pandapower`'s built-in network.
- `seams_engine.py` — transparent PTDF DC-OPF (`scipy.optimize.linprog(method='highs')`).
  Key objects: `compute_ptdf`→shift factors, `MarketEngine`, `solve_engine_dispatch`
  →`EngineResult(lmp, energy_price, line_dual=μ, flow_own=Fᴹ)`, `physical_flows`
  (superposition `F_phys=ΣFᴹ`), `seam_dual_gap`, `susceptance_widths`, `to_supply_demand`;
  `solve_engine_dispatch(flow_offsets=…)` (2026-06-10, opt-in) enforces limits on
  `F^M+offset` — the §5 accommodation mechanism;
  `solve_engine_dispatch(interchange=(buses, limit))` (2026-06-11, opt-in, CRA §4)
  adds an EDAM-style net-interchange scheduling limit: the net export of the given
  bus set is held within ±limit, the signed dual returns as
  `EngineResult.interchange_dual` (with `interchange_mw`/`interchange_limit`),
  and every bus in the set carries the same +μ_T LMP term (per-footprint energy
  prices — EDAM's per-BAA λ_j structure). Default None: existing callers
  (the whole seams notebook) are unaffected.
  `solve_engine_dispatch(shed_price=…)` (2026-06-12, opt-in) adds a **power-balance
  relaxation (load shedding)**: an unserved-load variable 0 ≤ u_n ≤ d_n at every load
  bus, priced at `shed_price` in the objective (EDAM DTD §5.5/§11 ENUS-style;
  mechanically a virtual generator at the load bus with cost V). Flows/interchange use
  SERVED load (d−u); shed returns as `EngineResult.shed_by_bus`; LMP = V where shedding
  is interior; `total_cost` stays production-only. Default None = old behavior
  (an unservable engine raises). Both notebooks set `SHED_PRICE = 150` in the shared
  §2 EDIT cell and pass it to their §2 unified clear. **Display + adequacy (2026-06-12):**
  three `seams_engine` helpers — `shed_segments(result, demand_by_bus, shed_alpha=0.12)`
  → a `demand_segments` dict drawing the unserved load as a **faint tail in the SAME bus
  colour** (alpha 0.12 vs the 0.35 demand fill — the idle-generation convention; the user
  rejected grey); `served_by_bus` → `{bus: served}` for
  `plot_network_topology(demand_served_by_bus=…)` ('served/total MW' annotation, e.g.
  "Load: 60/90 MW"); and **`served_demand(result, demand_by_bus)` → full {bus: d−shed},
  which MUST be passed to `compute_ptdf_flows`** so the gen→load chords trace the SERVED
  dispatch — otherwise the tracer balances nominal demand against actual gen (unbalanced by
  the shed MW), drawing chords that overshoot each generator's dispatch and fill a shed
  bus's whole load arc. With served-demand tracing, each generator's chords match its
  dispatch and a shed bus receives only its served-MW of incoming chords (the faint tail
  stays chord-free). `nodal_plot` demand_segments gained per-segment `alpha`. Wired into the
  §2 network cell (387e5a8a), §2 circlize/settlement cell (20620f42), the combined figure
  (6147ea1a), and CRA's §3 `example_figure`. **Settlement is on SERVED load (d−u)**
  everywhere (§2 `settle` table gained a 'served (MW)' column; ba_settlement/_agg already net
  shed) — this is what keeps the model **revenue-adequate**: shed load pays nothing, so
  paid-by-load − paid-to-gen still equals Σ|μ|·F exactly; the shed penalty (SHED_PRICE×MW) is
  an objective term (value of lost load), not a settlement flow. Verified at bus 5 (lines 4-5
  & 5-6 each 30 MW): 60 served / 30 shed, λ₅=150, rent identity 7,221.2 = 7,221.2, Bus-1 chords
  = its 101 dispatch. **Note (2026-06-12): the committed §2 `LINE_RATINGS` is the user's
  3-line shed scenario `{line_4:40, line_1:30, line_2:30}`** (was `{line_4:40}`) — this diverges
  the shared `84b7e33d` from seams by that one knob value and makes §4-combined's transfer
  stream slack (R_T=0); §3 is unaffected (rebuilds with its own `SCN_B=line_4`).
- `nodal_plot.py` — all the figures (network topology + circlize/chord). See §6.

**Tooling:**
- `export_methodology_pdf.py` — code-free methodology PDFs (see §5).

---

## 3. Environment

Runs in the **`assume`** conda env (has pandapower / pypsa / scipy / pycirclize),
NOT `western-seams` (which lacks pandapower). Activate (Git-Bash style):

```bash
eval "$(/c/Users/docke/miniconda3/condabin/conda.bat shell.bash hook 2>/dev/null)" && conda activate assume
```

When running Python that prints unicode, prefix `PYTHONUTF8=1` (Windows console is cp1252).
The Bash tool's working directory resets between calls — `cd` into the testbed each time.

---

## 4. How the notebooks work (the model)

Both notebooks share one engine (`seams_engine.py`) and the same physical primitives.

**Network & fleet** (current base case, reworked 2026-06-08):
- 9 buses; lines `line_0`=1-4, `line_1`=4-5, `line_2`=5-6, `line_3`=3-6, `line_4`=6-7,
  `line_5`=7-8, `line_6`=8-2, `line_7`=8-9, `line_8`=9-4.
- Generators: `gen_slack_0` **$50** @ bus 1, `gen_0` **$35** @ bus 2, `gen_1` **$20**
  @ bus 3 (170 MW) — **bus 3 is the cheapest unit.** Loads: 90 @ bus 5, 100 @ bus 7,
  125 @ bus 9.

**Mechanism:** each *engine* solves a single-period DC-OPF (min Σ cᵢgᵢ s.t. energy
balance [dual `λ`] and activated line limits [duals `μ`]); the nodal LMP is
`λ_n = λ + Σ_m SF_{n,m}·μ_m`. Congestion rent on a line = `|μ|·|flow|`, and
`paid-by-load − paid-to-gen = Σ_m |μ_m|·F̄_m` (energy nets to zero).

The **formal derivation** (objective, energy balance `[λ]`, flow + activated-line limits
`[μ_m]` annotated at the constraint, Lagrangian with generator-bound terms + complementary
slackness + stationarity, nodal LMP) lives in the shared Section-2 `form-one-engine-dispatch`
markdown cell, **identical in both notebooks**; the CRA notebook then adds the congestion-rent
identity + allocation methodology in Section 3 (own eqs `(1)–(12)`; seams `(1)–(18)`,
with the rent identity eq (6) now in the shared §2 block of both).
The shared block opens with a **nomenclature table** (N buses, M monitored lines, E set
of market engines M∈E, R^M resources; lower-case m = line vs upper-case M = market vs
script M = line set; plain italic E = scheduled interchange MW, defined at point of use
in the seams notebook). Checked against EDAM §4: m = generalized-network-constraint index
matches EDAM exactly; EDAM's l is the TRANSFER index (so lines are m, never l); notebook
g_i ≡ EDAM's EN_{i,t} (signed energy schedule) restricted to supply. Consistency with
Hogan *Flowgate Rights and Wrongs* / CAISO EDAM is recorded in
`reference_lmp_formulation_literature`.

**The notebooks just relabel that one engine:**
- **CRA notebook:** the network is partitioned into two **balancing authorities** that
  co-optimise one dispatch but settle separately. `BA-1 = {2,8,7,6,3}` holds the cheap
  fleet (bus 3 $20 + bus 2 $35) → the **net exporter** (purple `#993AFF`); `BA-2 =
  {1,9,4,5}` has only the expensive bus-1 unit → the **net importer** (blue). Two tie lines:
  `line_2` (5-6), `line_7` (8-9). **Configurable line assignment (2026-06-11):** the §3
  config cell's **`BA_LINES` dict-of-lists sits right beside `BA_DEFS` and places EVERY
  line explicitly under a BA** (the user rejected a derived-rule+override version —
  config parameters must be *visible*): `line_2` under BA-1, `line_7` under BA-2; a
  line listed under neither is unassigned (grey, rent split 50/50); duplicate/unknown
  entries assert. Downstream code reads the derived `LINE_BA` {line: BA|None}: line
  colours, the rent ledger (`R_own`/`R_unassigned`, eqs 8/10 use
  $\mathcal M_a$/$\mathcal M_\emptyset$), and the independent-clear activated sets all
  follow it; the tie set for τ (eq 9) stays topological. **The old Scenario 3A (tie binds) was CUT 2026-06-11** (user:
  confusing and unnecessary) — one scenario remains, de-numbered ("The scenario"):
  the internal BA-1 line **`line_4` (6-7) at 40 MW** → prices separate in BA-2 with
  no tie overloaded. Two allocation rules compared: **Method 1** (managing BA keeps
  the rent) vs **Method 2** (rebate the cross-seam separation τ to the net-payer BA).
  **τ is constraint-based (2026-06-12, user directive):** eq (9) prices it as
  Σ over binding lines of (shift-factor difference across the tie) × μ — never the
  raw nodal price gap, which absorbs μ_T once §4's transfer constraint binds
  (`seam_separation` shows `dlam` / `dlam_cong` / `dlam_xfer`=±μ_T side by side;
  verified against EDAM DTD §10 price formation, where the congestion component of
  LMP is Σ SF·μ). The §3 method ledgers (`_ledger`/`revenue_table`) render side by
  side (2026-06-12): per BA MultiIndex columns Consumer / Generator / Area net + a
  TOTAL conservation column, cash-out negative; the bottom "Net position" row
  carries eq (12)'s two readings (Area net = A−N, Consumer = −(L−A)). The §4
  `transfer_ledger(res, p, loads, method, indep)` is the full **autarky-vs-
  constrained** comparison (the cell-37 layout): autarky block from
  `independent_clear` on the SAME ratings, production-cost rows, both revenue
  streams with **collected separate from allocated** (congestion collected per
  assigned BA, TOTAL = R; transfer collected at the seam, TOTAL column only),
  Final position / Delta vs autarky / **Pareto** rows; allocation knobs in the §4
  EDIT block — `CONGESTION_METHODS` and `TRANSFER_METHODS`, with the transfer
  methodologies defined in the editable `allocate_transfer_rent(res, loads,
  t_method)` block (the σ_a of eq 17): **T1** fixed `TRANSFER_SPLIT` shares
  (default 50/50 = EDAM, asserts sum-to-1), **T2** all to the net-payer BA (the
  transfer analogue of Method 2). The pure-seam cell tabulates one ledger per
  transfer rule (τ=0 ⇒ congestion methods coincide); the combined cell one per
  (congestion × transfer) pair. They echo the assignment's C_tot / R_tot
  framing (§7).
- **Seams notebook (rebuilt 2026-06-10):** the SAME partition as the CRA BAs, with
  neutral names — `Market A = {2,8,7,6,3}` cheap exporter (purple), `Market B =
  {1,9,4,5}` high-cost importer (blue); seams `line_2` (5-6), `line_7` (8-9). §3 also
  exposes `MONITORED_LINES` — each market's ℳ^M_act as **explicit per-market line
  lists** (2026-06-11: every line listed so the config is a visible parameter block;
  default = both-ends rule PLUS the seam lines assigned, `line_2`→Market A,
  `line_7`→Market B; unknown-line assert); engines and line colours derive from it
  (grey = monitored by neither or both).
  The interchange is a **self-schedule**, not "price-taking" (renamed 2026-06-11:
  it shifts each market's residual supply curve and moves both LMPs without setting
  one). Drawn at **bar height = local LMP, no bid price**: import = unpriced supply
  block at the sink bus (price=lmp in the supply entry), export = scheduled demand
  at the source bus. `market_figure` helper (end of §3, mirrors CRA `example_figure`)
  renders one [network + nodal-dispatch] set per market with the other footprint
  greyed — used for every dispatch scenario: §4 blind clears (`seams-blind-figs`,
  superposition network kept as the sum) and §6's configurable transaction
  (`a8b3de4b`: SOURCE_BUS '3' → SINK_BUS '9', E_TRADE 150, TRADE_RATINGS knob;
  the LPs carry the FULL PATH: A's clear serves the sink delivery (exo
  −E@sink, outside its footprint), B's receives the source injection (exo
  +E@source) — both network panels show the same 3→9 routing).
  **Four-leg settlement (`seams-trade-settle`, final 2026-06-11; supersedes
  Method A): each market settles the schedule across its FULL PATH at its
  own two LMPs** — buys the source @ λ^M_s (pays its gen), sells the
  delivery @ λ^M_k (A charges the trader; B bills the served load). The
  trader pays λ^A_k, is paid λ^B_s, keeps the CROSS-MARKET gap
  (λ^B_s−λ^A_k)E; each market keeps its own path spread (λ^M_k−λ^M_s)E as a
  congestion charge (negative = counterflow path). Telescope (unnumbered
  display in trade-md): λ^B_k−λ^A_s = A's congestion + trader margin + B's
  congestion. All prices ENDOGENOUS. Demo: default flat 20/50 → trader
  1,500, market terms 0; E_TRADE=125 → λ^A=35, trader 1,875; line_4=30 →
  λ^A_9=30.25: A keeps ~512/h, trader ~988/h; SOURCE_BUS '2' → trader
  unchanged (settles at delivery/receipt buses), A's term −238 (counterflow);
  line_4=20 → INFEASIBLE (teaching point). Charts: **every GREY bar draws at
  the OTHER market's LMP for that bus — which IS the price the transaction
  settles there** (the four grey heights across the two sets = the four
  settlement prices); dashed bus-LMP line drawn across split load bars too,
  so grey-vs-dashed = the gap at that bus.
  nodal_plot opt-ins: per-unit `'color'` on supply entries; `demand_segments`
  on plot_nodal_circlize/plot_combined_letter (+ LMP line across segmented
  loads). `market_figure(sup_dem=..., demand_segments=..., legend_note=...)`.
  The risk code's gap (λ^B_import−λ^A_export) IS the trader leg already —
  only eq-16 prose retied; the seam ledger's Trader row likewise.
  Escape `\$` when a matplotlib title/legend string
  contains two dollar signs (MathText). The sweep, risk section, and
  ledger stay on the seam pair 6→5. §9 sandbox: + TRADE_MW/TRADE_SOURCE/
  TRADE_SINK knobs (full-path, merged with the seam E_MW), trader pays/paid
  print + per-market congestion take, per-market figure sets with the §6
  grey conventions, a §8-style POSITIONS LEDGER vs autarky and unified
  (per market: consumers, gen REVENUE + gen PROFIT separately, CONGESTION
  REVENUE = settlement residual so each row closes (consumers = gen rev +
  congestion + trader); Trader profit = interchange + trade legs; unified
  row attributes pool rent to the footprint containing each binding line,
  seam lines split evenly — an allocation choice; autarky = same knobs minus
  every cross-boundary schedule, reservation kept; overloads printed above
  it), a COMBINED set
  (network = ACTUAL total flows — gens &
  loads counted once, since a schedule booked in both clears would
  double-count its path; each bus at its home LMP) and a UNIFIED-clearing
  set. The sandbox pf table's F_phys/overload are the actual view (≠
  F[A]+F[B] when a trade/tie schedule is booked); it REWIRES the
  §3 config dicts in place for its figures (re-run §3 to restore defaults).
  §4: blind
  autarky clears overload `line_4` (43.4 > 40) with zero interchange. §5: editable
  per-line reservations (unified footprint components × `ATC_MARGIN`) restore
  feasibility; margin ladder prices conservatism. §6 (base ratings), reframed
  2026-06-11 as **participant-initiated interchange**: no engine clears the seam, a
  participant commits E before the prices materialise. Perfect-foresight sweep 0–215
  across 6→5 (W(E) falls at the dual-gap rate); contract-path addendum (scheduled 49
  vs actual 87 MW on the 40 MW line at E=150) closes §6. **§7 risk — its OWN main
  section (promoted 2026-06-11; TWO-SIDED continuous uncertainty same day)**,
  eqs 16–18. Two uncertain numbers, one per market, two KINDS of risk:
  (1) B's marginal cost ~ N(posted $50, B_COST_SD²) — gen_slack_0 IS B's whole
  flat supply curve so the bell passes straight through to λ^B (smooth price
  risk; default sd 15.5 = the old 30/50/70 spread's sd); (2) A's load ~
  N(100, A_LOAD_SD²) — WHICH SIDE OF A'S STEP the schedule lands on: λ^A is a
  two-point {20,35} mixture with weight p35(E)=Pr[L_A+E>170] (quantity risk →
  price risk because the schedule leans the market against its step; default
  sd 15, 0 recovers the known staircase). Consequences: E[gap] GLIDES 30→15
  (23.2 @E=70, p35=45%), sd BULGES 15.5→17.2 at the edge, and γ=0.001
  retreats from the old corner 70 to E*=55 — "the corner is no longer a safe
  place to stand". γ ladder → E* = 210/55/25, E[W@E*] = 287/2,855/3,737;
  γ=0.005 unchanged (never near the edge; FOC 30≈0.005·240·25 still).
  Quadrature: GH-15 for cost × CLIPPED ±4σ 33-pt grid for load (GH-25 tails
  gave negative load → INFEASIBLE LP; clip + renormalize, np.maximum(...,0)
  guard); A cleared per load node, B per cost node, unified per pair; tails +
  fan quantiles from the ANALYTIC mixture cdf over lamA atoms (quadrature
  atoms too coarse for 5% tails). CE exactness caveat: exact iff A_LOAD_SD=0.
  Figures (reworked over 2026-06-12 across several user passes; the section is
  now just TWO figures): (1) dist triple — A = STACKED outcome-share band vs E
  w/ γ-coloured E* markers + p35 labels / B bell / **basis panel = the basis
  each of the three modelled traders holds AT ITS OWN E*(γ)** (gamma-coloured,
  shaded P(loss) 17%/4%/3% at E*=210/55/25; risk aversion steers away from A's
  expensive step — riskier the bigger the volume). Dotted mixture-component
  curves were tried then REMOVED (user: muddled w/o insight). (2) payoff FAN
  (5-95 + 25-75 bands) OVERLAID with the three CE_γ(E) curves + E* dots — the
  risk-adjusted value = mean minus a penalty for the fan's width, droops below
  the dashed mean and peaks earlier as γ rises (green CE plunges off the floor
  → freeze ylim via get_ylim before the γ loop, restore after). GAMMA_COLORS =
  [#2E86C1,#E67E22,#27AE60] + E_STAR{γ:E*} defined ONCE in seams-risk-code,
  shared by dist-fig + fan. **REMOVED along the way: the (a)+(b) two-panel
  Fig 2, the np.gradient panel (c), and the GAMMA_FOCUS knob** — fan carries
  the CE curves + peaks so the decomposition panels were redundant; md eq-18
  in-words + read cell narrate fan→distribution.
  PLANNED enhancement (separate
  notebook, per user 2026-06-11): linear upward-sloping supply curves (mean at
  current price, adjustable slope → quadratic cost, QP à la canonical_corner.py or
  fine-block LP approximation) + one-factor correlated stochastic loads (CV + single
  ρ knob) + Monte Carlo so the price distributions EMERGE. §8: seam ledger — B's
  consumers gain only under unified; A's consumers lose everywhere; Pareto needs side
  instruments.

**Config cells drive everything** — each section opens with an `# -- EDIT --` cell
(fleet, line ratings, footprints, interchange). Change numbers, re-run, watch prices/flows.

---

## 5. Producing the combined PDFs

`export_methodology_pdf.py` makes **code-free methodology PDFs** (markdown + figures +
result tables, no code cells). From the testbed in the `assume` env:

```bash
python export_methodology_pdf.py                          # both notebooks
python export_methodology_pdf.py seams_examples.ipynb     # just one
```

Pipeline per notebook: strip `print`/log stream output (keep `display_data` /
`execute_result`) and lift the first `# H1` into the document title → `nbconvert --to
latex --no-input` (drops all code) → patch the `.tex` (inject `\newcounter{none}` to
dodge the nbconvert+pandoc longtable bug; set the real `\title{}`) → `xelatex` ×2.
Outputs: `congestion_revenue_allocation_methodology.pdf`, `seams_examples_methodology.pdf`.

Tables render as real booktabs tables because each notebook's imports cell registers a
`text/latex` DataFrame repr (PDF only; Jupyter/Colab keep the HTML tables). The notebooks
must be **executed first** so the figures/tables are present in the outputs.

---

## 6. Google Colab ⇄ GitHub synchronization

The repo `pdockery/wscc9-illustrative-model` serves the notebooks to Colab with zero
install. Each Colab user clicks the README badge → **Runtime → Run all**; a
**`colab-bootstrap`** cell (the first code cell, now present in BOTH the testbed source and
the repo copies — added to the testbed 2026-06-09 so a plain file copy carries it) `pip
install`s `pypsa pandapower pycirclize` and `wget`s the three modules from
`raw.githubusercontent.com/pdockery/wscc9-illustrative-model/main/`. The same cell is a
no-op when run locally (its action is gated on `"google.colab" in sys.modules`).

**Re-stage procedure** (after editing notebooks/modules in the testbed):
1. **Execute the notebooks** so outputs are saved (the repo serves them with outputs visible
   on open) — `jupyter nbconvert --to notebook --execute --inplace <nb>` in the `assume` env.
2. Copy the 3 modules + both notebooks from the testbed into the repo clone. **The
   `colab-bootstrap` cell now lives in the testbed source too (added 2026-06-09), so a plain
   file copy carries it — no re-insert step.** (Optional: clear the bootstrap cell's harmless
   "Local run" stream output before committing.)
3. `git add` the changed files, `git commit`, `git push origin main` (push needs the sandbox
   disabled). If `origin` is ahead, `git merge --ff-only origin/main` first.

**Gotcha:** `git push` needs the **sandbox disabled** in this environment (`Recv failure`
otherwise); `git commit` works sandboxed. `gh` CLI is not installed; plain `git` with
cached credentials. Default branch `main`. `.gitignore` keeps `*.png/*.pdf/*.csv` out.

---

## 7. Academic resources (`…/market_design_lab/`)

The notebooks operationalise this coursework / thesis material:
- **`Congestion Revenue Allocation/`** — the source for the CRA notebook:
  - `part1.tex` / `part1.pdf`, `part2.tex` — the user's own write-up (the "assignment").
  - `interconnector_assignment.ipynb`, `single-node.ipynb`, `two-nodes.ipynb`,
    `meshed-network.ipynb` — the exploratory lab notebooks the teaching notebook distils.
  - Papers: `HOGAN-PrimerTransmissionBenefits-2018.pdf`,
    `Joskow-MerchantTransmissionInvestment-2005.pdf`, `The Competitive Effects of
    Transmission Capacity…pdf`, and the course PDF (`A16758S1 Energy Systems…`).
  - **Formulation sources** for the Section-2 dispatch/LMP block: `flow0800r.pdf` (Hogan,
    *Flowgate Rights and Wrongs*, 2000) and `ExtendedDay-AheadMarketDraftTechnicalDescription.pdf`
    (CAISO EDAM §10 price formation) — the notebook block is consistent with both; see
    `reference_lmp_formulation_literature`.
- **Seams material** (folder root): `Revised Seams Blog…docx`, `Seams Messaging…docx`,
  `Virtual Town Hall - Seams Script.docx`, `Seams Webpage - Revised Content.md`,
  `proposal.tex/pdf`, `meeting_notes.tex/pdf`. These back the seams notebook (the broader
  seams work also lives in the separate **western-seams** project / `project_edam_alignment`).
- `NOTETAKING_PROMPT.md` — a **separate** Claude Code prompt for live meeting-note-taking
  into `meeting_notes.tex`; unrelated to the notebooks.

> The published notebooks are **standalone**: do not add paper equation numbers, §/Part
> references, "the paper" citations, or cross-references between the two notebooks. Use
> each notebook's own internal equation numbering. (The README may mention the paper as
> context — that's repo-level, not notebook content.)

---

## 8. Editing the notebooks — rules & gotchas

- **Terminology (user directive 2026-06-11):** the circlize/chord figure is called the
  **"nodal dispatch diagram"** in headers, figure titles, and prose ("circlize / chord"
  may appear once as a parenthetical descriptor). Never describe a unit/market/load as
  **"dear"** — use "expensive" / "high-cost" / "expensive-to-serve". Both notebooks were
  scrubbed 2026-06-11; don't reintroduce either.
- **`Read` and `NotebookEdit` FAIL on these notebooks** (embedded figure outputs exceed
  the tool token limit). **Edit the `.ipynb` JSON directly** via Write-tool Python scripts
  (`json.load`/`dump`, `ensure_ascii=True`; match cells by their stable `id`).
- **The user hand-edits these notebooks in the IDE.** Before a large multi-cell pass,
  confirm they've **saved & stopped**, and **re-read the cells fresh** first — a whole-file
  `json.dump` reverts any unsaved cells, and their next save reverts your edits.
- **Changing the shared fleet silently breaks the tuned demos** (overload presence,
  per-engine feasibility). After any base-case change, **re-execute the whole notebook**
  (`jupyter nbconvert --to notebook --execute --inplace …`), not just the unified clearing,
  and visually check the figures (extract embedded PNGs and view them).
- All `nodal_plot.py` figure options are **opt-in** (default off) so other testbed
  notebooks are unaffected. The teaching notebooks turn on: `line_colors` (colour lines by
  managing BA/market; congestion red overrides; ties grey), `lmp_line` (dashed bus-LMP line
  over the marginal-cost fill), `bus_groups`/`group_colors` (operator outer band),
  `annotate_roles` + `axis_key` (the supply/demand/dispatch labels and the polar
  price/volume axis key). Transit buses (no gen/load) render as circled numbers.
- **Shared cells stay consistent (equations identical; prose may intentionally differ):**
  the §0–2 cells common to both notebooks (notably `form-one-engine-dispatch`, the one-engine
  Lagrangian/LMP derivation) are matched by `id`. The **formulation/equations must stay the
  same**; wording may diverge **only for a deliberate reason** (e.g. seam-vs-border
  terminology — see below). When you edit the *math or structure* of a shared cell, mirror it
  to the other notebook and re-stage both; don't let prose drift by accident.
- **Terminology: "seam" is reserved for the seams notebook (2026-06-14).** The CRA notebook
  must NOT call the BA-to-BA boundary a "seam" — it uses **"border"** / "cross-border"
  (local helper `border_separation`, dict key `R_border`, displayed column "Transfer rent
  collected (border)"). KEPT as shared infrastructure (still say "seam"): the `seams_engine`
  module name, the `seam_dual_gap` import, and the shared §2 csv outputs
  `seams_settlement_by_node.csv` / `seams_line_capacity_value.csv`. The shared formulation
  cell now says "cross-footprint schedule" (neutral, in both notebooks).
- **Sync status (2026-06-11 end of day): the shared §0–2 cells are byte-identical again**
  (only the title cell and the figure-save filenames differ, by design). Ported to seams:
  the index-scoping/nomenclature fix ($g_i$ row; $d^M_n$/$\mathrm{exo}^M_n$; $g_n$ defined
  in the rent cell) and the seams-side `## 2. Market engine and nodal dispatch` header
  back-ported to CRA; the configurable line assignment (CRA `BA_LINES`, seams explicit
  `MONITORED_LINES` lists with `line_2`→Market A, `line_7`→Market B); seams eq (10)/(11)
  quantifier corrected $\mathcal M^M_{\mathrm{int}}$→$\mathcal M^M_{\mathrm{act}}$ (the code
  reserves on MONITORED lines) and eq (12) exo→$\mathrm{exo}^M$; `market_figure` greys a
  line monitored only by the dimmed market. Both notebooks fully re-executed; ALL tuned
  demo numbers verified unchanged (§4 13,659 / 43.4-on-40; §5 13,745 + ladder — the new
  seam-line reservations are slack; §6 8,475 / staircase / trade 1,500; risk table matches
  its read cell). **NOT ported: CRA §4 transfer constraint** — the user has not reviewed it
  yet; port to seams only on user instruction (the engine `interchange` param is already
  available and opt-in).
- **2026-06-12 CRA-only pass (τ repricing + ledger reformat)** — all in §3/§4 cells, so
  the shared §0–2 byte-identical status holds with NO seams re-sync needed. Eq (9)'s τ
  now Σ(ΔSF)·μ over binding lines (`seam_separation`: `dlam_cong`, `dlam_xfer`); §3
  numbers unchanged (R=804.1, τ=1048.4); §4 pure-seam now genuinely M1≡M2 (τ=0), combined
  τ 2,618.5→584.5 with a per-tie gap-decomposition display (line_7: raw +17.51 = cong
  −4.75 + xfer +22.26 — the raw-gap τ even mis-signed that tie). Ledgers in the
  autarky-vs-unified side-by-side format (see §4 above). Verified against the EDAM Draft
  Technical Description (v1.3): §10 LMP = λ_j − Σ SF·μ, §3.3/§5.6 transfer/scheduling
  limits; the DTD is formulation-only — the 50/50 transfer-revenue split is EDAM policy
  (eq-17 words say so). Same-day second pass: §4 `transfer_ledger` rebuilt as the
  autarky-vs-constrained ledger with Pareto check and collected-vs-allocated rows, plus
  the `CONGESTION_METHODS`/`TRANSFER_SPLIT` knobs (see the CRA bullet in §4 above; eq 17
  is now A_T = σ_a·R_T). Headline demo numbers: pure-seam — autarky PC 12,750 →
  constrained 9,900, **BA-1 consumers fail Pareto (−600)** while BA-1 generators gain
  2,550 (price 20→35); combined — 13,658.5 → 11,056.1, Delta TOTAL 2,602.4 under both
  methods, every column Pareto-yes. TOTAL-column identity: positions sum to −(production
  cost), so the Delta-vs-autarky TOTAL is exactly the production-cost saving. §3
  `_ledger` reverted to congestion-only (dead rt param removed); closer item 3 refreshed
  (τ double-count resolved; knob refinement remains). Third pass (same day): transfer
  allocation became a true **methodology block** — `allocate_transfer_rent` with T1
  (fixed shares) / T2 (all to net-payer), `TRANSFER_METHODS` knob, `transfer_ledger`
  gained a `t_method` arg; pure-seam shows a ledger per transfer rule, combined a 2×2
  method grid. Grid corners worth knowing: pure-seam T2 deepens the exporter-consumer
  loss (−600 → −1,500); combined **M1×T2 puts BA-1 consumers exactly at autarky (Δ=0)**
  and **M2×T2 drops them below it (−584.5, the grid's one Pareto 'no')** — stacking both
  rebates on the net-payer over-strips the exporter. Delta TOTAL invariant (2,602.4 /
  2,850) — allocation is pure distribution. Fourth pass (user rewrite + figure fix):
  **§4 is now titled "Optimized transfers"** (user-authored intro: EDAM transport-layer
  framing — export/import pairs, per-BAA net transfer in the power balance, scheduling
  limits as generalized network constraints; I polished grammar and synced the
  title-cell TOC row, dropping its stale "split 50/50"). The net-dispatch inset is now
  ONE copy (was both panels) at the lower left of the network panel, with line scale
  lw = 0.8 + 4.2·min(1, Ē/TIE_CAP) (the old 1+11 scale rendered as a blob) — then
  redesigned as the **"Transfers" inset** (user): panel-title font, `flow/limit`
  interchange label in the network's flow-label style (red bold `120/120` + `$22.26`
  price label when binding), bubbles = BA total addressable market (gen capacity +
  load) at the network panel's exact bus-bubble scale (scatter s = max(3·MW, 200)),
  rect (0.005, 0.02, 0.19, 0.235) with an inside-the-frame shrink guard;
  `draw_net_dispatch(ax, res, ebar, engine)`. Re-executed clean; methodology PDF +
  Colab re-stage still pending.
- **2026-06-12 generalization pass (non-market areas + load shedding):** the §3 config
  builds **`AREAS`** = BA_DEFS plus a `'Non-market'` area when buses sit outside both
  BAs (none in this notebook — AREAS == BA_DEFS); `ba_of` now returns None (was a
  KeyError), `line_kind` gained `('boundary', None)` for BA↔non-market lines, and
  `ba_settlement` / `_ledger` / `_agg` / `autarky_vs_unified` / `independent_clear` /
  `transfer_ledger` all loop AREAS — a non-market area settles at LMP (its column keeps
  conservation) but is never allocated rent; τ/M^tie stay BA-to-BA. Eq (13) now sums
  over **∂N_1 (BA-1's boundary)** with ∂N_1 = M^tie when the BAs cover every bus
  (nomenclature row added); eq (7) words note the non-market column. **Load shedding:**
  `SHED_PRICE = 150` $/MWh knob in the shared §2 EDIT cell + a "Feasibility relaxation"
  note (unnumbered math) and u_n/V nomenclature rows in the shared formulation cell —
  both mirrored to seams byte-identical, BOTH notebooks re-executed, all tuned numbers
  verified unchanged (relaxation slack everywhere; §2 cells now print 'load shed: none').
  The §2 unified clear, `solve_with_transfer`, and the §4 autarky (`independent_clear(...,
  shed_price=SHED_PRICE)`) all carry the relaxation, so any reconfiguration that strands
  load behind a constraint sheds at $150 (LMP caps there) instead of crashing; §3's
  independent-clear demo deliberately stays shed-OFF so infeasibility remains a finding.
  `example_figure` greys unserved load as a `demand_segments` tail (+ legend patch);
  settlement throughout uses SERVED load (d−u). Smoke-tested: 50-cap/80-load → shed 30,
  λ=150; load behind a 10 MW wire → shed 88.3, λ=150, line dual −152.6; feasible case
  bit-identical with/without the knob; interchange+shed coexist (E=120, μ_T=−15).
- **Combined-figure title default:** `plot_combined_letter(title_right=…)` defaults to
  `"Nodal dispatch - - merit order, demand, PTDF gen->load"` (2026-06-09); the CRA 3A/3B
  `example_figure` keeps its own separate default.
- **Combined-figure filenames differ by design** (2026-06-10): CRA saves
  `cra_combined_figure.png/.pdf`, seams saves `seams_combined_figure.png/.pdf` — the
  one permitted difference (besides the title cell) in the otherwise byte-identical
  §0–2 cells.

---

*Cross-refs in the persistent memory: `project_congestion_revenue_allocation`,
`project_seams_teaching_notebook`, `project_wscc9_colab_repo`,
`reference_notebook_pdf_export`, `feedback_standalone_notebooks`.*
