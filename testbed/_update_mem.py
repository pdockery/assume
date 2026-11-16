# One-shot memory + CLAUDE.md updater for the 2026-06-10 seams rebuild.
import os

MEM = r'C:\Users\docke\.claude\projects\c--Users-docke-Projects\memory'
CL  = r'C:\Users\docke\Projects\ASSUME\assume\testbed\CLAUDE.md'


def edit(path, old, new, count=1):
    s = open(path, encoding='utf-8').read()
    n = s.count(old)
    assert n == count, (os.path.basename(path), old[:60], n)
    open(path, 'w', encoding='utf-8').write(s.replace(old, new, count))
    print('ok:', os.path.basename(path), '|', old[:50].replace(chr(10), ' '))


SEAMS_MEM = os.path.join(MEM, 'project_seams_teaching_notebook.md')

# R1: purpose
edit(SEAMS_MEM,
"""**Purpose:** hand-tunable WSCC/IEEE 9-bus examples illustrating the paper's three
results — Part 2 trading friction (seam dual gap), Part 3 dispatch interference
(combined-flow overload neither engine sees), Part 4 transfer/price-taking export.
No RL/agent; just config-cell-driven examples.""",
"""**Purpose (REBUILT 2026-06-10):** hand-tunable WSCC/IEEE 9-bus examples around
**three seams issues**: (1) dispatch interference (physical superposition of
uncoordinated flows), (2) inefficient accommodation of flows (exogenous ATC-style
per-line reservations for the other market's flows), (3) inefficient interchange
(participant-initiated, price-taking transfers facing the seam dual gap). Plus a
contract-path-vs-actual-flow addendum (the transport-layer seam) and a seam ledger.
No RL/agent; config-cell-driven. **Footprints now = the CRA BA partition with neutral
names**: Market A {2,8,7,6,3} cheap exporter (purple #993AFF), Market B {1,9,4,5}
dear importer (blue #2471A3); seam lines line_2 (5-6), line_7 (8-9). The earlier
A=WEM-red {1,2,4,7,8} / B=Markets+-blue {3,5,6,9} partition and its tunings
(SEAM_RATINGS line_1=110, sweep-from-E=50) are GONE.""")

# R2: base-case paragraph -> tuned numbers of the rebuild
edit(SEAMS_MEM,
"""**Base case synced to CRA (2026-06-08, [[project_congestion_revenue_allocation]]):** fleet
reworked to gen_slack_0 \\$50@bus1/250, gen_0 \\$35@bus2/300, gen_1 **\\$20@bus3/170** (bus 3
cheapest); unified Section-2 constraint `line_4=40` (6-7). Two seams-specific retunes
followed from the smaller bus-3 unit: (a) market **B** (bus 3) holds the cheapest gen yet
its 215 MW load exceeds the 170 MW unit, so B needs ≥45 MW import — the price-taking sweep
now starts at E=50 (E=0 was infeasible); (b) the Section-3 combined overload moved, so
`SEAM_RATINGS` tightens `line_1` (4-5) to **110** (< the −120 combined |F|) to restore an
overload neither engine sees. A stays gen-rich exporter / B load-heavy importer by
capacity/load, so the A→B framing holds. **Lesson: changing the shared fleet silently
breaks the tuned two-engine demos (overload presence, per-engine feasibility) — always
re-EXECUTE the whole notebook, not just the unified clearing, to catch it.**""",
"""**Tuned demo numbers (2026-06-10 rebuild; fleet gen_slack_0 \\$50@bus1/250, gen_0
\\$35@bus2/300, gen_1 \\$20@bus3/170; §2 constraint `line_4=40`):**
- **§4 interference** (line_4=40, E=0): blind autarky clears are each feasible, wire
  carries 43.4 > 40 on line_4 (A holds 40.0, B unknowingly adds 3.4) — interference
  with ZERO interchange is the headline. Costs: A 2,909 + B 10,750 = 13,659; unified
  (line_4=40) 8,941.
- **§5 accommodation** (same config): reservations seeded from the UNIFIED clearing's
  footprint flow components × ATC_MARGIN (editable dicts ACCOMMODATION_A/B; default
  gamma=1.25) on each market's internal lines → feasible, sum 13,745; margin ladder
  1.0/1.25/1.5/2.0 → 13,728/13,745/13,762/13,797. B's reservations look huge
  (−268.8 on line_0) because the unified reference has A serving everything —
  deliberate "capability withheld" lesson. Off-point staleness covered in prose only.
- **§6 interchange** (BASE ratings; seam pair 6→5 across line_2; sweep E 0–215,
  INTERCHANGE_CAP): unified 8,475; W(E) staircase 4,275→0, falls at exactly the
  dual-gap rate within each price plateau (gap 30 until bus-3 exhausts ~E=70, then 15);
  at E=215 B's price degenerates to 0 (no marginal unit) — noted in prose. Contract-path
  addendum re-imposes line_4=40 at E=150: per-market components ≤40, scheduled-view sum
  49.2, ACTUAL flow 86.7 — the schedule's seam-bus injections are a fiction.
- **§7 ledger** (base ratings; autarky/E=150/E=200/unified): under scheduled trade B's
  consumers gain NOTHING (import settles at B's own marginal price 50; surplus goes to
  A's gens + trader = gap×E); only unified passes savings to B load (50→35). A's
  consumers LOSE everywhere (20→35). Pareto fails without side instruments — stated.

**Durable lesson retained: changing the shared fleet/footprints silently breaks the
tuned demos — always re-EXECUTE the whole notebook and eyeball the figures.**""")

# R3: sections list
edit(SEAMS_MEM,
"""- `seams_examples.ipynb` — the notebook. Sections: 0 setup, 1 shift factors +
  slipperiness map, 2 unified clearing (topology + circlize + settlement/congestion
  tables + combined letter figure), 3 two-engine overload, 4 price-taking export +
  seam dual gap, 5 sandbox.""",
"""- `seams_examples.ipynb` — the notebook (49 cells, 2026-06-10). Sections: 0 setup,
  1 shift factors, 2 one-engine clearing (BYTE-IDENTICAL with CRA §0–2 except the
  title cell and the figure-save filenames: seams writes seams_combined_figure, CRA
  writes cra_combined_figure), 3 footprints+map, 4 dispatch interference,
  5 inefficient accommodation, 6 inefficient interchange (+contract-path addendum),
  7 seam ledger, 8 sandbox (knobs incl. ACC_MARGIN one-shot forecast accommodation).""")

# R4: engine flow_offsets
edit(SEAMS_MEM,
"`susceptance_widths`, `to_supply_demand`.",
"""`susceptance_widths`, `to_supply_demand`. **`flow_offsets` param added 2026-06-10**
  (opt-in, default None): per-line signed MW the engine accommodates — limits enforced
  on F^M+offset, i.e. asymmetric window [−F̄−off, F̄−off]; backs §5 and the sandbox.""")

# R5: formulation block
s = open(SEAMS_MEM, encoding='utf-8').read()
old_hdr = '(own eqs (1)–(11); §2 block ENRICHED + SHARED 2026-06-09)'
assert s.count(old_hdr) == 1
s = s.replace(old_hdr, '(own eqs (1)–(15) as of 2026-06-10; §2 block ENRICHED + SHARED 2026-06-09)')
i = s.index('§3 the flow superposition')
marker = 'no paper citations.'
j = s.index(marker, i) + len(marker)
s = s[:i] + """now §2 also carries the congestion-rent identity eq (6) (cell `ced2c9a5`, shared).
§4 the flow superposition eqs (7)–(9) (own flow, `F^phys=ΣF^M`, the non-implication);
§5 the accommodation eqs (10)–(11) (effective limit `−F̄≤F^M+F̂^{M'}≤F̄`, reservation
`F̂=γ·F^ref`); §6 the price-taking `exo` interchange, dual gap, and welfare loss eqs
(12)–(15) (`Δλ^{A,B}=λ^B−λ^A≠0`; `W(E)=C^A+C^B−C^uni≥0`, envelope reading: W falls at
the gap rate per scheduled MW). Each ties to the real function
(`solve_engine_dispatch`, `physical_flows`, `seam_dual_gap`). Paper-style annotations,
no paper citations.""" + s[j:]
open(SEAMS_MEM, 'w', encoding='utf-8').write(s)
print('ok: seams formulation block rewritten')

# R6: status
edit(SEAMS_MEM,
'**Status 2026-06-08:**',
"""**Status 2026-06-10:** rebuilt notebook executes clean end-to-end; figures checked
(footprint map, superposition overload 43/40 red, side-by-side dual-gap fields);
pushed as `0ec70f3`. **Planned next:** a SECOND seams notebook on EDAM vs Markets+
design differences (pricing/interface aggregation etc.), and jointly editing the
standalone seams paper notes (`Thesis/notes/standalone_seams_paper.tex` +
`methodology/seams_paper.tex`) with insights from the notebook build.

**Status 2026-06-08 (prior):**""")

# ---- colab repo memory
edit(os.path.join(MEM, 'project_wscc9_colab_repo.md'),
'**STATUS: re-synced 2026-06-09**',
"""**STATUS: re-synced 2026-06-10** (commit `0ec70f3`) — the REBUILT seams notebook
(three seams issues on the CRA BA footprints, neutral Market A/B names, §5
accommodation via new `seams_engine.flow_offsets`, §6 sweep + contract-path addendum,
§7 seam ledger) plus the re-executed CRA (fleet-cell BA tags stripped, combined figure
renamed cra_combined_figure) and `seams_engine.py`. See
[[project_seams_teaching_notebook]].

**STATUS: re-synced 2026-06-09**""")

# ---- CRA memory append
p = os.path.join(MEM, 'project_congestion_revenue_allocation.md')
s = open(p, encoding='utf-8').read()
s += """

**2026-06-10:** shared fleet cell's `(BA-1)/(BA-2)` comment tags stripped (membership
is a §3 concept; keeps the cell byte-identical with the rebuilt seams notebook, whose
footprints now equal the BA partition — [[project_seams_teaching_notebook]]). The
combined letter figure now saves as `cra_combined_figure.png/.pdf` (was
seams_combined_figure — name collision with the seams notebook)."""
open(p, 'w', encoding='utf-8').write(s)
print('ok: CRA memory appended')

# ---- feedback eq numbering
edit(os.path.join(MEM, 'feedback_standalone_notebooks.md'),
"""(CRA `(1)`–`(12)`, seams `(1)`–`(11)` — both continuous across their
  formulation sections; the shared one-engine §2 block is eqs (1)–(5), byte-identical in
  both,""",
"""(CRA `(1)`–`(12)`, seams `(1)`–`(15)` — both continuous across their
  formulation sections; the shared §0–2 block is eqs (1)–(6) incl. the congestion-rent
  identity, byte-identical in both,""")

# ---- MEMORY.md index
MD = os.path.join(MEM, 'MEMORY.md')
edit(MD,
'PTDF DC-OPF validated vs Egret; standalone formulation (own eqs 1-11; §2 one-engine block shared byte-identical with CRA); plotting conventions (2026-06-09)',
'PTDF DC-OPF validated vs Egret; REBUILT 2026-06-10: 3 seams issues (interference/accommodation/interchange) on the CRA BA footprints, neutral Market A/B names, own eqs 1-15, flow_offsets engine ext, seam ledger')
edit(MD,
'synced through 658cff5 (2026-06-09)',
'synced through 0ec70f3 (2026-06-10)')

# ---- CLAUDE.md
edit(CL,
"""- `seams_examples.ipynb` — two market **engines** sharing the grid; dispatch
  interference (combined-flow overload), price-taking export, the seam dual gap.""",
"""- `seams_examples.ipynb` — two market footprints sharing the grid; **three seams
  issues**: dispatch interference (superposition overload with zero interchange),
  inefficient accommodation (exogenous ATC-style reservations, `flow_offsets`), and
  inefficient interchange (price-taking transfer vs the seam dual gap; W(E) staircase);
  plus a contract-path-vs-actual-flow addendum and a seam ledger.""")
edit(CL,
'(superposition `F_phys=ΣFᴹ`), `seam_dual_gap`, `susceptance_widths`, `to_supply_demand`.',
"""(superposition `F_phys=ΣFᴹ`), `seam_dual_gap`, `susceptance_widths`, `to_supply_demand`;
  `solve_engine_dispatch(flow_offsets=…)` (2026-06-10, opt-in) enforces limits on
  `F^M+offset` — the §5 accommodation mechanism.""")
edit(CL,
"""- **Seams notebook:** two **independent engines** A (`WEM`, buses 1,2,4,7,8, red) and B
  (`Markets+`, buses 3,5,6,9, blue) each enforce only their own lines and treat the
  other's flow as zero. A scheduled **A→B interchange** (price-taking) demonstrates the
  combined-flow overload (`SEAM_RATINGS` tightens `line_1` to 110 MW) and the seam dual
  gap. NB market B's 215 MW load exceeds its 170 MW unit, so it needs ≥45 MW import (the
  price-taking sweep starts at E=50).""",
"""- **Seams notebook (rebuilt 2026-06-10):** the SAME partition as the CRA BAs, with
  neutral names — `Market A = {2,8,7,6,3}` cheap exporter (purple), `Market B =
  {1,9,4,5}` dear importer (blue); seams `line_2` (5-6), `line_7` (8-9). §4: blind
  autarky clears overload `line_4` (43.4 > 40) with zero interchange. §5: editable
  per-line reservations (unified footprint components × `ATC_MARGIN`) restore
  feasibility; margin ladder prices conservatism. §6 (base ratings): price-taking E
  sweep 0–215 across 6→5; W(E) falls at the dual-gap rate; contract-path addendum
  (scheduled 49 vs actual 87 MW on the 40 MW line at E=150). §7: seam ledger — B's
  consumers gain only under unified; A's consumers lose everywhere; Pareto needs side
  instruments.""")
edit(CL,
"""  `example_figure` keeps its own separate default.""",
"""  `example_figure` keeps its own separate default.
- **Combined-figure filenames differ by design** (2026-06-10): CRA saves
  `cra_combined_figure.png/.pdf`, seams saves `seams_combined_figure.png/.pdf` — the
  one permitted difference (besides the title cell) in the otherwise byte-identical
  §0–2 cells.""")

print('ALL DONE')
