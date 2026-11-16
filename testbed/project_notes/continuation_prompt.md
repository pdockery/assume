# Continuation Prompt — ASSUME Egret Integration

Use this to resume work in a new session. Paste it as your first message.

---

## Context

I'm working on integrating Egret (Sandia's Pyomo-based SCUC/SCED solver) into the ASSUME electricity market simulation framework. This is a fork of assume-framework/assume. The goal is to train RL agents to learn strategic bidding on an IEEE 9-bus transmission network with nodal LMP pricing.

**Workspace:** `C:\Users\docke\Projects\ASSUME\assume` (git repo)
**Project docs:** `C:\Users\docke\Projects\ASSUME\project\` (outside git)
**Thesis methodology:** `C:\Users\docke\Oxford\OneDrive - Nexus365\MSc Energy System\Thesis\notes\methodology\2_methodology.tex`
**Conda env:** `assume` (Python 3.12) — activate via `C:/Users/docke/miniconda3/condabin/conda.bat run -n assume <cmd>`

### Key files to read first:
1. `C:\Users\docke\Projects\ASSUME\project\README.md` — current status and git state
2. `C:\Users\docke\Projects\ASSUME\project\integration_plan.md` — master plan
3. `C:\Users\docke\Projects\ASSUME\assume\testbed\scenarios.yaml` — all scenario configs + RL hyperparameters

### What's complete (Phases 1-5):
- Rebased onto upstream/main, IEEE 9-bus testbed, Egret SCED clearing, RL training pipeline
- Three algorithms: MATD3, SAC, PPO (all implemented and registered)
- Two strategies: WithholdingStrategy (legacy), ProfitStrategy (current)
- PopArt + PER, BC warm-start, benchmark runner, visualization notebooks
- Config consolidation: all hyperparameters in scenarios.yaml

### ProfitStrategy Design (current reward function):
- **Reward**: pure SCED profit + terrain shaping + guardrails
- **Terrain shaping** (anneals over training):
  - PEAK hours: Competitor Mountain (sigmoid × gaussian at $50 cliff) + Volume Ridge (bump at residual demand)
  - OFF-PEAK hours: MC Mountain (sigmoid × gaussian at $35 MC — guides competitive bidding when profit ≡ $0)
- **Guardrails** (always active): exponential penalties at 0-volume and 0-price
- **Action space**: price [0, price_cap], volume [0, max_power]
- Key params: `mountain_scale=100`, `mc_mountain_scale=100`, `mountain_sigma=3.0`, `mountain_tau=0.5`, `vol_ridge_tau=3.0`, `shaping_anneal=0.01`

### Four-Algorithm Comparison (ready to run):

| Config | Algorithm | Exploration | Key Params |
|--------|-----------|-------------|------------|
| `base_profit_matd3` | MATD3 | Additive noise + linear decay | noise_sigma=0.3, lr=0.0001 |
| `base_profit_sac` | SAC | Entropy only (NO additive noise) | alpha=0.2, target=-1.0, lr=0.0001 |
| `base_profit_ppo` | PPO | DistActor + entropy_coef | entropy=0.05, clip=0.2, lr=0.0003 |
| `base_profit_dqn` | DQN | Epsilon-greedy (discrete actions) | 60×50 adaptive bins, ε=1→0.05 over 300 eps |

MATD3/SAC/PPO share: ProfitStrategy, 1000 episodes, BC warmstart, PopArt, continuous actions.
DQN: ProfitStrategy, 1000 episodes, NO BC warmstart (argmax not differentiable), PER, adaptive discrete bins.

**CRITICAL**: SAC + additive noise is INCOMPATIBLE. Additive noise breaks SAC's log-probability computation. SAC must use noise_sigma=0.0.

### Algorithms ruled out:
- **PPO**: Catastrophic forgetting — on-policy discards experience, collapses after ~50 eps
- **DQN**: Q-value lock-in — greedy policy freezes regardless of exploration strategy or bin design

### Distributional Critic (DSAC / DMATD3):
- Quantile regression critic (32 quantiles) preserves bimodal reward structure at cliffs
- **Requires reward_scale** (~500) since no PopArt — raw $0-$950 rewards destabilize training
- **Entropy floor** (`sac_alpha_min=0.01`) prevents SAC/DSAC entropy collapse
- PopArt and distributional are mutually exclusive — distributional handles scale via QR loss + reward_scale

### Observation space (4 dimensions):
- [0] price forecast (scaled 0-1 within episode)
- [1] load forecast (scaled 0-1 within episode)
- [2] regime indicator (1.0 = peak, 0.0 = off-peak)
- [3] residual demand fraction (residual_demand / max_power)
  - Standard market info — any generator can estimate from ISO demand forecasts + published competitor capacity (Wolak 2003)

### 2×2 comparison results (200 eps, 64×32 network, 4-obs):

| | Scalar (PopArt) | Distributional (reward_scale=500) |
|---|---|---|
| **MATD3** | **$356 best** / $320 final | $330 / $311 |
| **SAC** | $355 / $254 | $321 / $319 |

- MATD3 at 1000 eps: $381 best, H12=$886/h (91% optimal)
- Two strategies emerge: MATD3 bids above residual demand (sets own LMP); SAC withholds below (gen_1 sets LMP=$50)
- Volume threshold cliff: sigmoid shaping pulls volume past threshold with no gradient to pull back

### Current problem: Volume threshold cliff
- Below residual demand: infra-marginal, LMP=$50. Above: marginal, LMP=bid price.
- Current sigmoid shaping provides no penalty for crossing the threshold.
- **Proposed fix**: Ricker wavelet shaping — peaks just before edge, goes negative after.
  `shaping = scale * delta * exp(-0.5 * delta²)` where `delta = (value - threshold) / width`
  Apply at: own MC ($35), competitor MC ($50), residual demand (hourly).
  MC mountain stays for off-peak safe-harbor. Wavelets are permanent (no anneal).

### What's next:
1. **Ricker wavelet reward shaping** — implement bidirectional gradient at all edges
2. **1000-episode runs** — MATD3/SAC/DSAC/DMATD3 with new shaping
3. **Add congestion** — update `congested` scenario, two RL agents
4. **Multi-agent** — train gen_0 + gen_1 simultaneously on congested network
5. **ScucClearingRole** — MIP commitments, fix binaries, LP re-solve for LMPs
6. **Two-settlement DA/RT** — forecast divergence between settlements

### Running training:
```bash
cd assume
# Single algorithm:
C:/Users/docke/miniconda3/condabin/conda.bat run -n assume python -m testbed.learning_runner --scenario base_profit_matd3
C:/Users/docke/miniconda3/condabin/conda.bat run -n assume python -m testbed.learning_runner --scenario base_profit_sac
C:/Users/docke/miniconda3/condabin/conda.bat run -n assume python -m testbed.learning_runner --scenario base_profit_ppo
C:/Users/docke/miniconda3/condabin/conda.bat run -n assume python -m testbed.learning_runner --scenario base_profit_dqn
# Benchmark (Optuna HPO):
C:/Users/docke/miniconda3/condabin/conda.bat run -n assume python -m testbed.benchmark_runner --scenario base_profit_matd3 --algorithms matd3
```

### IEEE 9-bus topology:
- Gen buses: 1 (slack, 250MW, $20 + peaker 50MW, $60), 2 (gen_0, 300MW, $35), 3 (gen_1, 270MW, $50)
- Load buses: 5 (90 MW), 7 (100 MW), 9 (125 MW)
- Transit buses: 4, 6, 8. Ring: 4-5-6-7-8-9-4, spurs: 1-4, 3-6, 2-8
- Total demand: 189-315 MW (diurnal). Residual demand (above slack 250MW): 0-65 MW
- All generators: min_power=0

### Key technical details:
- Egret entry point: `solve_unit_commitment(model_data, 'highs', relaxed=True)` for SCED
- **CRITICAL**: Egret time series values MUST be **lists** (not dicts)
- **HiGHS compat**: Patches in `clearing/highs_compat.py`
- ASSUME clearing interface: `clear(orderbook, market_products) → (accepted, rejected, meta, flows)`

### Novel findings:
- **No ASSUME forks or PRs combine RL with nodal/SCED clearing.** This work is novel.
- **RL discovered physical withholding** as a stable strategy under exploration-starved config.
- **SAC additive noise incompatibility** — not documented in ASSUME or standard RL literature for this context.
- **Cournot formula broken for electricity markets** — demand inelasticity makes standard formula degenerate.

### Standing instruction for this and all future sessions:
At the end of each working session, update these files with what was learned:
1. Memory files (`~/.claude/projects/.../memory/`) — topic files + MEMORY.md index
2. This continuation prompt (`project/continuation_prompt.md`)
3. Project README (`project/README.md`) — status, completed items, next steps
4. Integration plan (`project/integration_plan.md`) — checkmarks and next steps
5. Methodology LaTeX (`C:\Users\docke\Oxford\OneDrive - Nexus365\MSc Energy System\Thesis\notes\methodology\2_methodology.tex`) — new reward functions, algorithm details, findings

---

**Resume with:** "Read the project README, scenarios.yaml, and profit_strategy.py, then let's continue. Three algorithm configs ready. Next: run and compare MATD3/SAC/PPO, then add congestion."
