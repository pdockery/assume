# PopArt + Prioritized Experience Replay Implementation Plan

## Why These Two Techniques

### Problem 1: Reward Scale Mismatch (→ PopArt)
We iterated through 8 different manual normalization approaches for the reward function. Each solved one problem while creating another:
- Normalizing by `max_power × price_cap`: peak rewards 10x smaller than off-peak
- Normalizing by `bid_volume`: agent games the normalizer by offering tiny volumes
- Fixed normalizer `max_power/3`: arbitrary constant, doesn't adapt across scenarios

**PopArt eliminates manual normalization entirely.** It maintains running mean/std of critic targets and adaptively rescales them, with a weight correction that preserves existing value estimates. The critic learns on normalized targets while the actor sees real-scale Q-values.

### Problem 2: Regime Frequency Imbalance (→ PER)
12 off-peak hours produce easy, high-magnitude gradient signals (bid MC + full volume = reward 1.0). 12 peak hours produce small, noisy signals (the actual strategic challenge). With uniform replay sampling, the critic over-fits the easy off-peak pattern and under-fits peak.

**PER samples high-TD-error transitions more frequently.** Peak hours where the agent bids wrong have high TD error and get replayed more, counteracting the frequency imbalance.

## Implementation Sequence

### Step 1: New Files (no dependencies)

**`assume/reinforcement_learning/sumtree.py`** — Binary tree for O(log n) proportional sampling
- `add(priority, data_idx)`, `update(tree_idx, priority)`, `get(s) → (idx, priority, data_idx)`
- `total()`, `min_priority()`

**`assume/reinforcement_learning/popart.py`** — PopArt output layer
- `PopArtLayer(nn.Module)` — replaces final Linear layer of each Q-network
- `normalize(targets)` — `(y - μ) / Σ`
- `denormalize(output)` — `output × Σ + μ`
- `update_stats(targets)` — EMA update + output-preserving weight correction:
  ```
  w_new = (Σ_old / Σ_new) × w_old
  b_new = (Σ_old × b_old + μ_old - μ_new) / Σ_new
  ```

### Step 2: Config (backward-compatible)

**`assume/common/base.py`** — Add to `LearningConfig`:
```python
# PER
per_enabled: bool = False
per_alpha: float = 0.6          # prioritization exponent
per_beta_start: float = 0.4    # initial IS correction
per_beta_anneal_steps: int = 100000

# PopArt
popart_enabled: bool = False
popart_beta: float = 3e-4      # EMA decay rate
```

### Step 3: PrioritizedReplayBuffer

**`assume/reinforcement_learning/buffer.py`** — New subclass of `ReplayBuffer`:
- Wraps SumTree for priority storage
- `sample()` returns `PrioritizedReplayBufferSamples` with IS `weights` and `tree_indices`
- `update_priorities(tree_indices, td_errors)` — called after critic update
- New items get `max_priority` (ensures they're sampled at least once)

### Step 4: CriticTD3 PopArt Integration

**`assume/reinforcement_learning/neural_network_architecture.py`**:
- `CriticTD3.__init__` accepts `use_popart`, `popart_beta`
- If enabled, final layer of each Q-network is `PopArtLayer` instead of `nn.Linear`
- Expose `self.q1_popart`, `self.q2_popart` references (or `None`)
- No changes to `forward()` — PopArtLayer's forward is just `nn.Linear.forward`

### Step 5: Algorithm Integration

**SAC `update_policy()`** changes:
1. **Critic targets**: Denormalize target critic output → compute raw target → update PopArt stats → normalize target for loss
2. **Critic loss**: If PER, multiply per-sample MSE by IS weights
3. **TD errors**: Extract from `|target_q - min(q1, q2)|` after loss
4. **Priority update**: Call `buffer.update_priorities(tree_indices, td_errors)`
5. **Actor loss**: Denormalize critic output before computing policy gradient

**MATD3 `update_policy()`** — Same pattern.

**PPO** — Not affected (on-policy, uses RolloutBuffer).

### Step 6: Buffer Creation Sites

Three locations conditionally create `PrioritizedReplayBuffer`:
1. `assume/scenario/loader_csv.py` (~line 1123)
2. `testbed/learning_runner.py` (~line 1254)
3. `testbed/_debug_patch.py` (~line 71)

### Step 7: Critic Construction Sites

Pass `use_popart` and `popart_beta` from `LearningConfig` to `CriticTD3`:
1. SAC `create_critics()`
2. MATD3 `create_critics()`

## Key Design Decisions

### Twin-Q with PopArt
Each Q-network gets its own independent PopArtLayer with separate running stats. Both see the same targets, so stats converge similarly. Independent stats prevent one network's correction from affecting the other.

### Target Critic PopArt
Target critics' PopArt buffers (μ, Σ) are Polyak-averaged along with the weights. This means target normalization tracks the main critic smoothly. When computing raw target values:
```
next_q_norm = target_critic(next_states, next_actions)
next_q_raw = target_critic.q1_popart.denormalize(min(next_q1_norm, next_q2_norm))
raw_target = reward + gamma * next_q_raw
```

### Multi-Agent TD Error for PER
Both SAC and MATD3 accumulate critic loss across agents before backward(). For PER priorities, collect per-agent TD errors, stack, and take `max(dim=0)` — samples surprising to ANY agent get high priority.

### PopArt + PER Interaction
Orthogonal features, no special handling needed:
- PER provides IS weights for the loss
- PopArt normalizes targets and denormalizes outputs
- TD errors for PER are computed in the normalized space (what the critic actually predicts)

## Testing Strategy
1. Unit test SumTree: add, get, update, boundary conditions
2. Unit test PopArtLayer: normalize/denormalize roundtrip, weight correction preserves output
3. Unit test PrioritizedReplayBuffer: sampling proportional to priority, IS weights correct
4. Integration: run `base` scenario with both enabled, verify convergence
