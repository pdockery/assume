# ASSUME Egret Integration Project

## Current Status (2026-03-19)

### Completed
- **Phase 1: Rebase** — rebased onto upstream/main (`656d8c82`), 2 local PyPSA commits on top
- **Phase 2: Copilot instructions** — `.github/copilot-instructions.md` updated with strategy naming, nodal clearing, grid_utils, forecaster classes, Egret section
- **Phase 3: IEEE 9-bus testbed** — full pay_as_clear simulation pipeline working
  - `testbed/ieee9_network.py`: pandapower case9 → PyPSA with enriched generator data
  - `testbed/testbed_runner.py`: end-to-end runner using `load_pypsa()` workflow
  - Validated: clearing prices correct ($20 off-peak, $35 peak), supply=demand balanced
- **Phase 3 visualizations** — results inspection notebook + nodal visualization
  - `testbed/inspect_results.ipynb`: clearing prices, supply/demand, dispatch, merit order, economics
  - `testbed/nodal_plot.py`: pycirclize chord diagram + network topology diagram
- **Test validation** — 339 passed, 1 skipped (1 upstream failure unrelated to our changes)
- **Installed**: pandapower 3.4.0, gridx-egret 0.6.0, pypsa 1.1.2, pycirclize, torch 2.10.0+cpu, tensorboard

### Phase 4 — Egret Integration (SCED + RL + visualization done, SCUC next)
1. ✅ `testbed/validate_egret_sced.py` — standalone Egret SCED on IEEE 9-bus (LP, LMPs)
2. ✅ `testbed/clearing/egret_adapter.py` — ASSUME ↔ Egret data mapping (volume+price bids via p_max)
3. ✅ `testbed/clearing/sced_clearing.py` — SCED clearing (LP dispatch + LMPs from duals)
4. ✅ `testbed/test_sced_integration.py` — integration test (107 accepted orders, supply=demand balanced)
5. ✅ `testbed/learning_runner.py` — RL training with WithholdingStrategy (price + volume control)
6. ✅ `testbed/inspect_sced_results.ipynb` — SCED results notebook (scenario-driven, circlize + topology)
7. ✅ `testbed/inspect_rl_results.ipynb` — RL bidding behavior notebook (reward surface + training bids)
8. ✅ `testbed/strategies/withholding_strategy.py` — legacy RL strategy (two-regime reward)
9. ✅ `testbed/strategies/heuristics.py` — BC warm-start heuristics (forecast_markup, residual_demand_midpoint, price_cap_fraction)
10. ✅ **RL learns physical withholding** — agent discovered profitable strategy (see findings below)
11. ⬜ ScucClearingRole (MIP commitment + LP re-solve for LMPs)
12. ⬜ Two-settlement DA/RT coupling

### Phase 5 — Algorithm Comparison + Config Consolidation
1. ✅ **SAC algorithm** (`assume/reinforcement_learning/algorithms/sac.py`) — StochasticActor, auto-tuned alpha
2. ✅ **PPO algorithm** (`assume/reinforcement_learning/algorithms/ppo.py`) — DistActor, CriticPPO, RolloutBuffer
3. ✅ **Algorithm-agnostic learning_runner** — `--algorithm matd3|sac|ppo` flag, buffer factory
4. ✅ **Benchmark runner** (`testbed/benchmark_runner.py`) — Optuna HPO + multi-seed eval + analysis
5. ✅ **Config consolidation** — all RL hyperparameters in `scenarios.yaml` `training.learning_config`
6. ✅ **forecast_mode** — `"exclude_rl"` (counterfactual) vs `"include_all"` (all gens at MC)
7. ✅ **YAML→network pipeline** — `_apply_generator_overrides()` applies scenarios.yaml to PyPSA network

### Phase 5b — ProfitStrategy + Reward Redesign (2026-03-18/19)
1. ✅ **ProfitStrategy** (`testbed/strategies/profit_strategy.py`) — pure profit + terrain shaping + guardrails
2. ✅ **Terrain shaping design** — MC mountain (off-peak), competitor mountain + volume ridge (peak), annealing
3. ✅ **Three-algorithm configs** — `base_profit_matd3`, `base_profit_sac`, `base_profit_ppo` in scenarios.yaml
4. ✅ **SAC noise diagnosis** — additive noise incompatible with SAC (corrupts log-prob), entropy-only config
5. ✅ **Reward surface visualization** — analytical 200×200 grid in inspect_rl_results.ipynb + presentation_charts.ipynb
6. ✅ **Presentation charts** — clean 9-bus topology, 4-panel reward decomposition (profit/terrain/guards/total)
7. ⬜ **Run four algorithms** — compare MATD3/SAC/PPO/DQN convergence and final policies
8. ⬜ **Congestion scenarios** — update `congested` to ProfitStrategy, two RL agents

### Phase 5c — DQN + Action Discretization (2026-03-19)
1. ✅ **DQN algorithm** (`assume/reinforcement_learning/algorithms/dqn.py`) — Double DQN, epsilon-greedy, target network
2. ✅ **Discrete action space** — price × volume bins mapped to continuous bids via ActionDiscretizer
3. ✅ **Adaptive binning** — inverse-CDF density placement concentrating bins near known ridges
4. ✅ **Ridge auto-detection** — competitor MCs (price ridges) + residual demand (volume ridges) from network data
5. ✅ **Pipeline integration** — DiscreteActor wrapper, epsilon decay in training loop, PER support
6. ✅ **PPO initial experience fix** — zero-noise bug in `get_actions()` for PPO/zero-noise configs
7. ✅ **PPO LearningConfig fields** — `clip_ratio`, `entropy_coef`, `vf_coef`, `gae_lambda`, `ppo_max_grad_norm`
8. ⬜ **Long DQN training runs** — 500-1000 episodes with adaptive bins, compare to continuous algorithms
9. ⬜ **DQN without terrain shaping** — pure profit + guardrails (DQN doesn't need cliff smoothing)

### Phase 5d — Distributional Critic + Observation Redesign (2026-03-20)
1. ✅ **Distributional critic** (`DistributionalCriticTD3`) — 32-quantile QR critic for SAC + MATD3
2. ✅ **Quantile Huber loss** (`learning_utils.py`) — pairwise NxN QR loss with PER support
3. ✅ **Reward scaling** — `reward_scale=500` for distributional critics (replaces PopArt)
4. ✅ **SAC entropy floor** — `sac_alpha_min=0.01` prevents entropy collapse
5. ✅ **Residual demand observation** — 4th obs dim, scaled RD/max_power (Wolak 2003)
6. ✅ **Network scaled to 64×32** — from 32×16, more capacity for volume-vs-load mapping
7. ✅ **Per-scenario results_dir** — each scenario writes to its own directory
8. ✅ **2×2 algorithm comparison** — {MATD3, SAC} × {scalar, distributional} at 200 eps
9. ✅ **DQN ruled out** — Q-value lock-in prevents learning regardless of bin design
10. ⬜ **Ricker wavelet reward shaping** — bidirectional gradient at all edges (next session)
11. ⬜ **1000-episode runs** — full training for best algorithms

### Phase 6 — Multi-Agent + Congestion (Next)
1. ⬜ Update `congested` scenario with ProfitStrategy + same algorithm params
2. ⬜ Train two RL agents (gen_0 + gen_1) on congested network
3. ⬜ Analyze impact of congestion on strategic behavior (physical vs economic withholding)

### Not Started
- ScucClearingRole (MIP commitment + LP re-solve for LMPs)
- Two-settlement DA/RT coupling

### Next Steps
1. **Ricker wavelet shaping** — bidirectional gradient at MC, competitor MC, and residual demand edges
2. **1000-episode runs** — MATD3/SAC/DSAC/DMATD3 with Ricker wavelets
3. **Add congestion** — update `congested` scenario to ProfitStrategy, run with two RL agents
4. **Analyze results** — convergence, final policy, physical vs economic withholding
5. **ScucClearingRole** — MIP commitments, fix binaries, LP re-solve for LMPs
6. **Two-settlement DA/RT coupling**

### Key Finding — PPO Catastrophic Forgetting (2026-03-19)

PPO (200 episodes, ProfitStrategy) showed catastrophic forgetting:
- Reward peaked at ~$263 (eval, ep 45) then collapsed steadily to -$55 by ep 100
- On-policy: each update discards prior experience, no replay buffer retention
- Policy locked onto bad region and never recovered despite MC exploration resets
- **Conclusion**: PPO is poorly suited for this problem (narrow ridge, needs experience retention)

### Key Finding — DQN Ruled Out (2026-03-19/20)

DQN with discrete action bins was implemented with multiple binning strategies (uniform,
adaptive inverse-CDF, gradient-informed). All suffered from Q-value lock-in: the Q-network
locks onto early estimates and the greedy policy freezes. Terrain shaping worsened DQN by
rewarding the right volume at the wrong price. DQN is not viable for this problem.

### Key Finding — 2×2 Algorithm Comparison (2026-03-20)

| | Scalar Critic (PopArt) | Distributional Critic (reward_scale) |
|---|---|---|
| **MATD3** | $356 best / $320 final | $330 best / $311 final |
| **SAC** | $355 best / $254 final | $321 best / $319 final |

- MATD3 scalar is the best performer. At 1000 eps: $381 best, H12 profit $886/h (91% optimal)
- Distributional critics collapsed without reward scaling — fixed with reward_scale=500
- SAC entropy floor (`alpha_min=0.01`) prevents collapse but policy still drifts
- Two distinct strategies emerge: MATD3 bids above residual demand (sets own LMP);
  SAC physically withholds below residual demand (gen_1 sets LMP=$50)
- Residual demand observation (4th dim) improved volume targeting: H12 59→66MW for MATD3

### Key Finding — Volume Threshold Cliff (2026-03-20)

The reward landscape has a cliff in the volume dimension at residual demand:
- Below RD: agent is infra-marginal, gen_1 sets LMP=$50. Profit = $15 × vol.
- Above RD: agent is marginal, sets own LMP at bid price. Profit drops if bid < $50.
- Current sigmoid terrain shaping pulls volume toward RD but provides no gradient to
  prevent crossing. MATD3/DMATD3 cross the threshold; SAC/DSAC stay below (physical withholding).
- **Proposed fix**: Ricker wavelet shaping — peaks just before threshold, goes negative
  just after, providing bidirectional gradient at all edges (MC, competitor MC, residual demand).

### Major RL Finding — Physical Withholding (2026-03-12)

The RL agent (MATD3, 200 episodes) discovered physical withholding instead of economic withholding:
- **Price**: Bids at exactly $35 (MC) for most hours — no price markup
- **Volume**: Offers only 15-55 MW of 300 MW capacity during peak hours H10-H16
- By withholding volume, forces gen_1 ($50) dispatch → LMP = $50 at all buses
- **Caveat**: This finding was under exploration-starved config. May change with more exploration.

### Key Finding — Premature Convergence (2026-03-18/19)

Three SAC test runs with ProfitStrategy all showed premature convergence:
- Agent converges to BC warmstart region (~$42-43) instead of optimal ~$49.99
- SAC entropy auto-tuning drives exploration to zero before critic learns landscape
- Best result: physical withholding at 40MW/$44-46 → $600/h (vs optimal $975/h)
- **Root cause**: Entropy decay outpaces critic learning
- **Mitigation**: Larger buffer, more initial experience, critic warmup, higher initial entropy

### Key Finding — SAC + Additive Noise Incompatibility (2026-03-18)

Additive Gaussian noise (noise_sigma > 0) corrupts SAC's reparameterization trick:
- SAC computes log_prob from tanh(mean + std × internal_noise)
- External noise added AFTER this breaks the entropy computation
- SAC with additive noise goes deterministic while relying on external noise → degraded policy
- **Rule**: SAC must use noise_sigma=0.0. Entropy is the only exploration mechanism.

## Git State
```
Branch: main (rebased onto upstream/main)
Local commits:
  a1bcea31 — improvements to pypsa interoperability workflow
  c55ab7d1 — testing, understanding, and adapting for pypsa interoperability
Upstream HEAD: 656d8c82 (assume-framework/assume main, 2026-03-09)
Working tree: modified files (loader_pypsa.py, powerplant.py, 2 example notebooks)
Untracked: .github/copilot-instructions.md, testbed/, outputs/
Stash: stash@{0} = old pre-fetch WIP (can be dropped)
```

## Directory Layout
```
C:\Users\docke\Projects\ASSUME\
├── project/                              # This directory — outside git
│   ├── README.md                         # Level-set (this file)
│   ├── integration_plan.md               # Master plan
│   ├── egret_integration_research.md     # Egret API research notes
│   ├── ieee9_testbed_plan.md             # IEEE 9-bus testbed design
│   ├── rl_algorithm_comparison_plan.md   # SAC vs MATD3 vs PPO benchmark plan
│   └── continuation_prompt.md            # Context for resuming sessions
├── assume/                               # Git repo (assume-framework/assume fork)
│   ├── .github/copilot-instructions.md   # VS Code Copilot config
│   ├── assume/                           # Package source
│   │   └── reinforcement_learning/
│   │       └── algorithms/
│   │           ├── matd3.py              # MATD3 (upstream)
│   │           ├── sac.py                # SAC (new)
│   │           ├── ppo.py                # PPO (new)
│   │           └── dqn.py                # DQN (new) — discrete actions, adaptive bins
│   ├── tests/                            # Test suite
│   ├── examples/                         # Notebooks
│   ├── testbed/                          # IEEE 9-bus testbed
│   │   ├── ieee9_network.py              # pandapower → PyPSA builder
│   │   ├── testbed_runner.py             # End-to-end ASSUME simulation
│   │   ├── learning_runner.py            # RL training loop (MATD3/SAC/PPO)
│   │   ├── benchmark_runner.py           # Optuna HPO + multi-seed eval
│   │   ├── scenarios.yaml                # Scenario definitions + RL hyperparameters
│   │   ├── inspect_results.ipynb         # Basic results visualization
│   │   ├── inspect_sced_results.ipynb    # SCED scenario results
│   │   ├── inspect_rl_results.ipynb      # RL bidding behavior + reward surface
│   │   ├── presentation_charts.ipynb     # Publication-quality figures
│   │   ├── nodal_plot.py                 # Circlize + topology visualization
│   │   ├── strategies/                   # RL strategies + heuristics
│   │   │   ├── withholding_strategy.py   # Legacy (two-regime reward)
│   │   │   ├── profit_strategy.py        # Current (profit + terrain shaping)
│   │   │   └── heuristics.py             # BC warm-start heuristics
│   │   └── clearing/                     # Egret clearing roles
│   │       ├── egret_adapter.py          # ASSUME ↔ Egret mapping
│   │       ├── sced_clearing.py          # SCED (LP + LMPs)
│   │       └── highs_compat.py           # HiGHS monkey-patches
│   └── testbed_results/                  # Simulation output CSVs
│       ├── rl_episodes/                  # Per-episode CSV + bid_history.csv
│       ├── rl_final/                     # Final RL evaluation CSVs
│       ├── learned_strategies/           # Saved actor/critic params per episode
│       └── benchmark/                    # Algorithm comparison results
```

## Key Technical Decisions
1. **Rebase** (not merge) to stay linear with upstream
2. **SCED before SCUC** — validate LP dispatch + LMPs before adding MIP complexity
3. **Egret `solve_unit_commitment()`** with `relaxed=True` for SCED (LP + duals)
4. **LMP extraction** from dual variables (only when `relaxed=True`)
5. **pandapower `case9()`** for testbed (Anderson & Fouad 1980)
6. **ProfitStrategy** replaces WithholdingStrategy for new experiments (pure profit + terrain shaping)
7. **Three-algorithm comparison** with matched hyperparameters for fair evaluation
8. **SAC entropy-only** — no additive noise (incompatible with SAC's reparameterization)

## Conda Environment
```
Name: assume
Python: 3.12
Install: pip install -e ".[dev,test]"
Activate: conda activate assume
Key packages: pandapower 3.4.0, gridx-egret 0.6.0, pypsa 1.1.2, pycirclize, torch 2.10.0+cpu, optuna 4.7.0
```
