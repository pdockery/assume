# Plan: RL Algorithm Comparison Benchmark (MATD3 vs SAC vs PPO)

## Context

After 200 MATD3 episodes, the agent discovered **physical withholding** — bidding at MC ($35) but offering only 15-55 MW of 300 MW capacity during peak hours, forcing gen_1 ($50) to set the LMP. This is a stable equilibrium for MATD3 but may not be the globally optimal strategy. SAC's entropy-regularized exploration may discover **economic withholding** (higher price on larger volume) which the literature suggests is typically more profitable. The benchmark compares all three algorithms to see which strategies each discovers.

### Current Codebase State (2026-03-12) — What Changed Since Plan Was Written

Key changes by the user since our last session:
1. **`min_bid_price` reverted to 0.0** (was 35.0 in our session) — agent can bid below MC
2. **`max_bid_price` back to 100.0** (was 60.0) — wider action space
3. **BC warm-start added**: `_pretrain_actor_heuristic()` with `forecast_markup` heuristic targeting $42.90 bid
4. **`noise_sigma` reduced to 0.1** (was 0.3) — BC provides good start, so less noise needed
5. **`exploration_reset_interval` back to 0** — no periodic resets (BC start is sufficient)
6. **`initial_experience_episodes` back to 5** (was 15)
7. **New file**: `testbed/strategies/heuristics.py` — 3 BC heuristics (forecast_markup, residual_demand_midpoint, price_cap_fraction)
8. **Physical withholding result**: Agent learned volume withholding instead of price markup
9. **Visualization upgrades**: Nameplate capacity bars, three fill levels, bid labels on both plots
10. **`agent_scenarios.yaml`**: New agent-specific scenario config file

### Current Hyperparameters (learning_runner.py)
```
algorithm: matd3, lr: 0.0003, gradient_steps: 10, batch_size: 64,
gamma: 0.0, noise_sigma: 0.1, noise_dt: 1, train_freq: 12h,
min_bid_price: 0.0, max_bid_price: 100.0, action_noise_schedule: linear
```

### Current Scenario (scenarios.yaml withholding)
```
episodes: 200, min_bid_price: 0.0, max_bid_price: 100.0,
initial_experience_episodes: 5, exploration_reset_interval: 0,
bc_warmstart: {strategy: forecast_markup, bc_steps: 500, bc_lr: 0.01}
```

## Research Questions

1. **Does SAC find economic withholding** while MATD3 converges to physical withholding?
2. **Which strategy is more profitable** — small volume × high margin (physical) vs large volume × small margin (economic)?
3. **Does BC warm-start help or hurt** each algorithm? (confound to control for)
4. **Which algorithm converges fastest** to a profitable strategy?
5. **How robust is each algorithm** across random seeds? (MATD3 collapsed to MC in 50% of our runs)

## Implementation Phases

### Phase A: SAC Algorithm (~350 lines new code)

**1. StochasticActor** — `assume/reinforcement_learning/neural_network_architecture.py`
- Architecture: `obs_dim → 256 → 128` shared trunk (ReLU), then two heads: `mean_head(128 → act_dim)` + `log_std_head(128 → act_dim)`
- Sampling via reparameterization trick: `action = tanh(mean + std * epsilon)`
- Log-prob with squashing correction: `log_pi -= sum(log(1 - tanh(u)^2))`
- `forward(obs)` returns just `tanh(mean)` (deterministic, matches MLPActor interface for strategy compat)
- `sample(obs)` returns `(action, log_prob)` (used only inside `sac.update_policy()`)
- Register as `"stochastic"` in `algorithms/__init__.py`

**2. SAC class** — `assume/reinforcement_learning/algorithms/sac.py`
- Extends `RLAlgorithm`, mirrors TD3 structure
- Reuses `CriticTD3` (twin Q-networks) — no new critic needed
- No target actor (unlike TD3), but has target critics + Polyak update
- Auto-tuned entropy temperature `alpha`: learnable `log_alpha` parameter + optimizer, target entropy = `-act_dim`
- Actor loss: `mean(alpha * log_pi - min(Q1, Q2))`
- Critic loss: MSE with target `r + gamma * (min(Q1_targ, Q2_targ) - alpha * log_pi_next)`
- Alpha loss: `-log_alpha * (log_pi + target_entropy).detach()`
- No policy delay (updates actor every step, simpler than TD3)
- `save_params` / `load_params` include alpha state

**3. Registration**
- `learning_role.py` line ~474: add `elif algorithm == "sac": self.rl_algorithm = SAC(learning_role=self)`
- `base.py` LearningConfig: add `sac_alpha_auto: bool = True`, `sac_initial_alpha: float = 0.2`

### Phase B: PPO Adaptation (~400 lines cherry-picked + adapted)

**1. Cherry-pick from `upstream/drl-ppo`:**
- `RolloutBuffer` → add to `buffer.py` (stores obs, actions, rewards, log_probs; clears after each update)
- `DistActor` → add to `neural_network_architecture.py` (MLPActor + Normal distribution output)
- `CriticPPO` → add to `neural_network_architecture.py` (single value network)

**2. Adapt PPO class** — `assume/reinforcement_learning/algorithms/ppo.py`
- Rewrite constructor to match `__init__(self, learning_role)` pattern (upstream takes explicit args)
- Fix hardcoded `std=0.2` in DistActor → make configurable
- Standardize critic storage to strategy-attached pattern (upstream stores on learning_role)
- GAE advantage estimation + clipped surrogate objective + entropy bonus

**3. Registration**
- `learning_role.py`: add `elif algorithm == "ppo"`
- `base.py`: add `clip_ratio`, `vf_coef`, `entropy_coef`, `gae_lambda`, `ppo_action_std`
- `algorithms/__init__.py`: register `DistActor`, `CriticPPO`

### Phase C: Algorithm-Agnostic learning_runner (~50 lines changed)

**Files**: `testbed/learning_runner.py`, `testbed/scenarios.yaml`

- Add `algorithm` field to scenario YAML (default: "matd3")
- Buffer factory: `ReplayBuffer` for matd3/sac, `RolloutBuffer` for ppo
- PPO buffer reset after each episode (on-policy)
- PPO skips `episodes_collecting_initial_experience` (no replay buffer to fill)
- Action interface: PPO's DistActor returns `(action, dist)` — strategy wrapper unpacks
- BC warm-start (`_pretrain_actor_heuristic`) works for all algorithms (same actor interface)
- SAC doesn't need `_bias_actor_toward_midpoint` — entropy handles exploration naturally

### Phase D: Benchmark Runner (~900 lines new code)

**1. `testbed/benchmark_runner.py`** (~550 lines)
- Uses Optuna with TPE sampler + MedianPruner
- Per-algorithm hyperparameter search spaces:
  - **MATD3**: lr, noise_sigma, tau, gamma, gradient_steps, batch_size
  - **SAC**: lr, tau, gamma, gradient_steps, batch_size, alpha_auto, initial_alpha
  - **PPO**: lr, clip_ratio, entropy_coef, gae_lambda, gradient_steps, vf_coef
- **Controlled variables** (same across algorithms): BC warm-start (on/off), episodes, periods, scenario, min/max bid price
- Protocol: 50 trials/algorithm, 3 seeds/trial for tuning → best params → 10 seeds for eval
- Budget-matched: same total env steps (episodes x periods) across all algorithms
- CLI: `python -m testbed.benchmark_runner --scenario withholding --algorithms matd3 sac ppo`
- Phases: `--phase tune`, `--phase eval`, `--phase analyze` (can run separately)
- Optional `--no-bc` flag to run without behavioral cloning warm-start (controls for confound)

**2. `testbed/benchmark_analysis.py`** (~350 lines)
- Learning curves: IQM across seeds with IQR shading (one line per algorithm)
- Performance profiles: fraction of runs exceeding normalized score threshold
- Sample efficiency bars: episodes to reach 80%/90% of final performance
- Bid evolution heatmaps: hour-of-day vs episode, colored by bid markup %
- **Strategy classification**: auto-detect physical vs economic withholding per run
  - Physical: price near MC, volume < 50% capacity
  - Economic: price > MC + $5, volume > 50% capacity
- Hyperparameter importance: fANOVA from Optuna
- Probability of improvement: pairwise bootstrap comparison (P(SAC > MATD3), etc.)

**3. Result storage:**
```
testbed_results/benchmark/
├── tuning/{algorithm}_study.db     # Optuna SQLite per algorithm
├── eval/{algorithm}_seed_{N}/      # per-seed bid_history + rewards
├── best_params.json                # best hyperparameters per algorithm
└── summary.csv                     # aggregated comparison metrics
```

## Files Modified/Created

| File | Action | Lines |
|------|--------|-------|
| `algorithms/sac.py` | CREATE | ~350 |
| `algorithms/ppo.py` | CREATE (cherry-pick + adapt) | ~400 |
| `neural_network_architecture.py` | EDIT (add StochasticActor, DistActor, CriticPPO) | +80 |
| `algorithms/__init__.py` | EDIT (register new architectures) | +5 |
| `learning_role.py` | EDIT (add sac/ppo to factory) | +10 |
| `base.py` | EDIT (add SAC/PPO LearningConfig fields) | +15 |
| `buffer.py` | EDIT (add RolloutBuffer) | +120 |
| `testbed/learning_runner.py` | EDIT (algorithm-agnostic buffer/actions) | +50 |
| `testbed/benchmark_runner.py` | CREATE | ~550 |
| `testbed/benchmark_analysis.py` | CREATE | ~350 |
| `testbed/scenarios.yaml` | EDIT (add algorithm field) | +10 |
| **Total** | | **~1,940** |

## Key Design Decisions

1. **SAC actor interface**: `forward()` returns deterministic action (MLPActor compat), `sample()` returns (action, log_prob) for training only. No strategy code changes needed.
2. **PPO cherry-pick**: Reuse upstream's tested code but adapt constructor/interface to match current `base_algorithm.py`.
3. **Buffer selection in learning_runner**: Testbed-local, not in ASSUME core.
4. **Budget matching**: Total env steps = episodes x periods. PPO is less sample-efficient but may find better policies.
5. **BC warm-start is a controlled variable**: Benchmark runs with and without BC to measure its interaction with each algorithm.
6. **Existing heuristics module** (`strategies/heuristics.py`) reused as-is for BC warm-start across all algorithms.

## Dependencies

- `optuna` (pip install) — Bayesian HPO
- `rliable` (optional, pip install) — IQM/bootstrap statistics
- All other deps already installed

## Verification

1. **SAC unit test**: Run withholding scenario with `algorithm: "sac"`, verify positive rewards by episode 50
2. **PPO unit test**: Same scenario with `algorithm: "ppo"`, verify learning occurs
3. **Benchmark smoke test**: `--trials 3 --algorithms matd3 sac --eval-seeds 2` (fast, ~10 min)
4. **Full benchmark**: `--trials 50 --algorithms matd3 sac ppo --eval-seeds 10` (~hours)
5. **Strategy classification**: Verify auto-detection of physical vs economic withholding
6. **Analysis**: Verify learning curves, performance profiles, bid heatmaps render correctly
