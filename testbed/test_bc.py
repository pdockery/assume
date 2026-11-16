"""Quick smoke test for behavioral cloning warm-start."""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

print("Importing...", flush=True)
from testbed.learning_runner import run_learning_testbed

print("Starting 2-episode BC test...", flush=True)
results = run_learning_testbed(
    training_episodes=2,
    episodes_collecting_initial_experience=1,
    validation_episodes_interval=5,
    exploration_reset_interval=0,
    rl_agents=["gen_0"],
    rl_strategy="powerplant_energy_withholding",
    periods=24,
    start_hour=0,
    min_bid_price=0.0,
    max_bid_price=100.0,
    db_uri="",
    results_dir="testbed_results/bc_test",
    bc_config={
        "strategy": "forecast_markup",
        "params": {},
        "bc_steps": 200,
        "bc_lr": 0.01,
    },
)
print(f"Done: {results['episodes']} episodes, best_reward={results['best_reward']:.4f}")
