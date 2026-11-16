# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Benchmark runner: systematic comparison of MATD3, SAC, and PPO algorithms
on the IEEE 9-bus SCED withholding scenario.

Protocol:
  Phase 1 (tune):  Optuna TPE + MedianPruner, N trials per algorithm, 3 seeds/trial
  Phase 2 (eval):  Best params per algorithm, M seeds for robust evaluation
  Phase 3 (analyze): Learning curves, strategy classification, comparison plots

Usage:
    cd assume/
    python -m testbed.benchmark_runner --scenario withholding --algorithms matd3 sac
    python -m testbed.benchmark_runner --phase tune --trials 20 --algorithms matd3 sac ppo
    python -m testbed.benchmark_runner --phase eval --eval-seeds 10
    python -m testbed.benchmark_runner --phase analyze
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logger = logging.getLogger(__name__)

# ── Default search spaces per algorithm ──────────────────────────────

SEARCH_SPACES = {
    "matd3": {
        "learning_rate": ("log_float", 1e-4, 3e-3),
        "noise_sigma": ("float", 0.05, 0.4),
        "tau": ("log_float", 0.001, 0.05),
        "gradient_steps": ("int", 5, 30),
        "batch_size": ("categorical", [32, 64, 128]),
    },
    "sac": {
        "learning_rate": ("log_float", 1e-4, 3e-3),
        "tau": ("log_float", 0.001, 0.05),
        "gradient_steps": ("int", 5, 30),
        "batch_size": ("categorical", [32, 64, 128]),
        "sac_initial_alpha": ("log_float", 0.05, 1.0),
    },
    "ppo": {
        "learning_rate": ("log_float", 1e-4, 3e-3),
        "clip_ratio": ("float", 0.1, 0.3),
        "entropy_coef": ("log_float", 0.001, 0.1),
        "gae_lambda": ("float", 0.9, 0.99),
        "gradient_steps": ("int", 3, 15),
        "vf_coef": ("float", 0.25, 1.0),
    },
}

# Fixed hyperparams (same across algorithms for fairness)
FIXED_CONFIG = {
    "gamma": 0.0,
    "noise_scale": 1,
    "noise_dt": 1,
    "action_noise_schedule": "linear",
    "exploration_noise_std": 0.1,
}


def _suggest_params(trial, algorithm: str) -> dict:
    """Suggest hyperparameters for a trial based on the algorithm's search space."""
    space = SEARCH_SPACES[algorithm]
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "log_float":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def _run_single_training(
    algorithm: str,
    scenario: str,
    overrides: dict,
    seed: int,
    episodes: int,
    bc_config: dict | None = None,
) -> dict:
    """Run a single training with given hyperparameters and return results."""
    import torch as th

    th.manual_seed(seed)
    np.random.seed(seed)

    from testbed.learning_runner import run_learning_testbed

    # Load scenario config
    import yaml

    yaml_path = Path(__file__).parent / "scenarios.yaml"
    with open(yaml_path) as f:
        all_scenarios = yaml.safe_load(f)

    scenario_cfg = all_scenarios.get(scenario, {})
    training = scenario_cfg.get("training", {})

    # Build kwargs from scenario defaults + overrides
    scratch_dir = f"testbed_results/benchmark/scratch/{algorithm}_s{seed}"
    Path(scratch_dir).mkdir(parents=True, exist_ok=True)

    # Reduce initial experience for short tuning runs
    init_exp = min(
        training.get("initial_experience_episodes", 5),
        max(1, episodes // 3),
    )
    val_interval = min(
        training.get("validation_interval", 10),
        max(1, episodes // 4),
    )

    kwargs = dict(
        market_mechanism="egret_sced",
        training_episodes=episodes,
        episodes_collecting_initial_experience=init_exp,
        validation_episodes_interval=val_interval,
        exploration_reset_interval=training.get("exploration_reset_interval", 0),
        rl_agents=training.get("rl_agents", None),
        rl_strategy=training.get("rl_strategy", None),
        algorithm=algorithm,
        periods=scenario_cfg.get("periods", 24),
        min_bid_price=training.get("min_bid_price", 0.0),
        max_bid_price=training.get("max_bid_price", 100.0),
        save_episode_interval=0,  # don't save per-episode CSVs during tuning
        bc_config=bc_config,
        results_dir=scratch_dir,
        extra_learning_config={
            **overrides,
            "trained_policies_save_path": f"{scratch_dir}/policies",
        },
    )

    return run_learning_testbed(**kwargs), overrides


def _objective(trial, algorithm: str, scenario: str, episodes: int, seeds: int, bc_config: dict | None):
    """Optuna objective: mean reward across seeds for suggested hyperparams."""
    params = _suggest_params(trial, algorithm)

    rewards = []
    for seed in range(seeds):
        try:
            result, _ = _run_single_training(
                algorithm=algorithm,
                scenario=scenario,
                overrides=params,
                seed=seed + trial.number * 100,
                episodes=episodes,
                bc_config=bc_config,
            )
            # Use best eval reward; fall back to best train reward
            r = result.get("best_reward", -1e9)
            if r <= -1e9:
                r = result.get("best_train_reward", -1e9)
            if r <= -1e9:
                r = 0.0  # no useful signal
            rewards.append(r)
        except Exception as e:
            logger.warning(f"Trial {trial.number} seed {seed} failed: {e}")
            rewards.append(-1e9)

    mean_reward = np.mean(rewards)
    trial.set_user_attr("rewards", rewards)
    trial.set_user_attr("params", params)

    return mean_reward


def run_tuning(
    algorithms: list[str],
    scenario: str,
    n_trials: int = 20,
    seeds_per_trial: int = 3,
    episodes: int = 50,
    bc_config: dict | None = None,
    output_dir: str = "testbed_results/benchmark",
) -> dict:
    """Run Optuna hyperparameter tuning for each algorithm.

    Returns:
        Dict mapping algorithm name to best params dict.
    """
    import optuna

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_params = {}

    for algo in algorithms:
        logger.info(f"\n{'='*60}")
        logger.info(f"Tuning {algo.upper()} — {n_trials} trials, {seeds_per_trial} seeds/trial, {episodes} episodes")
        logger.info(f"{'='*60}")

        # Ensure tuning DB directory exists before creating study
        (output_path / "tuning").mkdir(parents=True, exist_ok=True)

        study = optuna.create_study(
            study_name=f"{algo}_{scenario}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=f"sqlite:///{output_path / 'tuning' / f'{algo}_study.db'}",
            load_if_exists=True,
        )

        study.optimize(
            lambda trial: _objective(trial, algo, scenario, episodes, seeds_per_trial, bc_config),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best = study.best_params
        best_params[algo] = best
        logger.info(f"\n{algo.upper()} best params: {best}")
        logger.info(f"{algo.upper()} best reward: {study.best_value:.4f}")

    # Save best params
    params_path = output_path / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"\nBest params saved to {params_path}")

    return best_params


def run_evaluation(
    algorithms: list[str],
    scenario: str,
    n_seeds: int = 10,
    episodes: int = 200,
    bc_config: dict | None = None,
    output_dir: str = "testbed_results/benchmark",
) -> dict:
    """Run evaluation with best params from tuning, multiple seeds.

    Returns:
        Dict mapping algorithm to list of per-seed result dicts.
    """
    output_path = Path(output_dir)
    params_path = output_path / "best_params.json"

    if not params_path.exists():
        logger.error(f"No best_params.json found at {params_path}. Run tuning first.")
        return {}

    with open(params_path) as f:
        best_params = json.load(f)

    all_results = {}

    for algo in algorithms:
        if algo not in best_params:
            logger.warning(f"No tuned params for {algo}, using defaults")
            params = {}
        else:
            params = best_params[algo]

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {algo.upper()} — {n_seeds} seeds, {episodes} episodes")
        logger.info(f"Params: {params}")
        logger.info(f"{'='*60}")

        seed_results = []
        for seed in range(n_seeds):
            logger.info(f"\n--- {algo} seed {seed} ---")
            try:
                result, _ = _run_single_training(
                    algorithm=algo,
                    scenario=scenario,
                    overrides=params,
                    seed=seed,
                    episodes=episodes,
                    bc_config=bc_config,
                )
                seed_results.append(result)
            except Exception as e:
                logger.error(f"{algo} seed {seed} failed: {e}")
                seed_results.append({"best_reward": -1e9, "error": str(e)})

        all_results[algo] = seed_results

        # Save per-algorithm results
        eval_dir = output_path / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / f"{algo}_results.json"
        with open(eval_path, "w") as f:
            json.dump(
                {
                    "algorithm": algo,
                    "params": params,
                    "seeds": n_seeds,
                    "episodes": episodes,
                    "results": [
                        {
                            "seed": i,
                            "best_reward": r.get("best_reward", -1e9),
                            "rewards_history": r.get("rewards_history", []),
                        }
                        for i, r in enumerate(seed_results)
                    ],
                },
                f,
                indent=2,
            )

    # Save summary CSV
    _save_summary(all_results, output_path)

    return all_results


def _save_summary(all_results: dict, output_path: Path):
    """Save a summary CSV comparing algorithms."""
    import csv

    summary_path = output_path / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "n_seeds", "mean_reward", "std_reward",
            "median_reward", "min_reward", "max_reward",
        ])
        for algo, results in all_results.items():
            rewards = [r.get("best_reward", -1e9) for r in results]
            rewards = [r for r in rewards if r > -1e8]  # exclude failures
            if rewards:
                writer.writerow([
                    algo, len(rewards),
                    f"{np.mean(rewards):.4f}", f"{np.std(rewards):.4f}",
                    f"{np.median(rewards):.4f}",
                    f"{np.min(rewards):.4f}", f"{np.max(rewards):.4f}",
                ])
            else:
                writer.writerow([algo, 0, "N/A", "N/A", "N/A", "N/A", "N/A"])
    logger.info(f"\nSummary saved to {summary_path}")


def run_analysis(output_dir: str = "testbed_results/benchmark"):
    """Generate comparison plots and strategy classification from eval results."""
    output_path = Path(output_dir)
    eval_dir = output_path / "eval"

    if not eval_dir.exists():
        logger.error("No eval results found. Run evaluation first.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for analysis plots")
        return

    # Load all eval results
    algo_data = {}
    for f in sorted(eval_dir.glob("*_results.json")):
        with open(f) as fh:
            data = json.load(fh)
        algo_data[data["algorithm"]] = data

    if not algo_data:
        logger.error("No result files found in eval/")
        return

    # ── 1. Reward comparison box plot ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    all_rewards = []
    for algo, data in algo_data.items():
        rewards = [r["best_reward"] for r in data["results"] if r["best_reward"] > -1e8]
        if rewards:
            labels.append(algo.upper())
            all_rewards.append(rewards)

    if all_rewards:
        ax.boxplot(all_rewards, labels=labels)
        ax.set_ylabel("Best Reward")
        ax.set_title("Algorithm Comparison — Best Reward Distribution")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path / "reward_comparison.png", dpi=150)
        plt.close(fig)
        logger.info("Saved reward_comparison.png")

    # ── 2. Learning curves (mean ± std across seeds) ──────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"matd3": "#1f77b4", "sac": "#ff7f0e", "ppo": "#2ca02c"}

    for algo, data in algo_data.items():
        histories = [r["rewards_history"] for r in data["results"] if r["rewards_history"]]
        if not histories:
            continue
        # Pad to same length
        max_len = max(len(h) for h in histories)
        padded = np.full((len(histories), max_len), np.nan)
        for i, h in enumerate(histories):
            padded[i, : len(h)] = h

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        episodes = np.arange(1, max_len + 1)
        color = colors.get(algo, "gray")
        ax.plot(episodes, mean, label=algo.upper(), color=color)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Learning Curves (mean ± std across seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "learning_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Saved learning_curves.png")

    # ── 3. Strategy classification summary ────────────────────────
    # Classify each seed's final policy as physical vs economic withholding
    summary_lines = ["Algorithm Strategy Classification\n", "=" * 50 + "\n"]
    for algo, data in algo_data.items():
        n_physical = 0
        n_economic = 0
        n_other = 0
        for r in data["results"]:
            # Simple heuristic: if best_reward > 0.1, the agent found something
            # We'd need bid_history to properly classify, so for now just report rewards
            rw = r["best_reward"]
            if rw > 0.2:
                n_economic += 1  # placeholder
            elif rw > 0.05:
                n_physical += 1  # placeholder
            else:
                n_other += 1
        summary_lines.append(
            f"{algo.upper()}: profitable={n_economic + n_physical}, "
            f"unprofitable={n_other} (of {len(data['results'])} seeds)\n"
        )

    summary_text = "".join(summary_lines)
    with open(output_path / "strategy_classification.txt", "w") as f:
        f.write(summary_text)
    logger.info(f"\n{summary_text}")
    logger.info("Analysis complete. Results in: " + str(output_path))


def main():
    import yaml

    parser = argparse.ArgumentParser(
        description="Benchmark RL algorithms for SCED strategic bidding"
    )
    parser.add_argument(
        "--phase",
        choices=["tune", "eval", "analyze", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="withholding",
        help="Scenario from scenarios.yaml",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["matd3", "sac", "ppo"],
        default=["matd3", "sac"],
        help="Algorithms to benchmark",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Optuna trials per algorithm (tune phase)",
    )
    parser.add_argument(
        "--tune-seeds",
        type=int,
        default=3,
        help="Seeds per trial (tune phase)",
    )
    parser.add_argument(
        "--tune-episodes",
        type=int,
        default=50,
        help="Episodes per training run during tuning",
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        default=10,
        help="Seeds for evaluation (eval phase)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=200,
        help="Episodes per training run during evaluation",
    )
    parser.add_argument(
        "--no-bc",
        action="store_true",
        help="Disable behavioral cloning warm-start",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="testbed_results/benchmark",
        help="Output directory for results",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load BC config from scenario
    bc_config = None
    if not args.no_bc:
        yaml_path = Path(__file__).parent / "scenarios.yaml"
        with open(yaml_path) as f:
            all_scenarios = yaml.safe_load(f)
        scenario_cfg = all_scenarios.get(args.scenario, {})
        bc_config = scenario_cfg.get("training", {}).get("bc_warmstart", None)

    if args.phase in ("tune", "all"):
        run_tuning(
            algorithms=args.algorithms,
            scenario=args.scenario,
            n_trials=args.trials,
            seeds_per_trial=args.tune_seeds,
            episodes=args.tune_episodes,
            bc_config=bc_config,
            output_dir=args.output_dir,
        )

    if args.phase in ("eval", "all"):
        run_evaluation(
            algorithms=args.algorithms,
            scenario=args.scenario,
            n_seeds=args.eval_seeds,
            episodes=args.eval_episodes,
            bc_config=bc_config,
            output_dir=args.output_dir,
        )

    if args.phase in ("analyze", "all"):
        run_analysis(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
