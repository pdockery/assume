"""Check rl_strats keys vs exploration center keys."""
import sys
import logging
import traceback

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

try:
    import yaml
    import pandas as pd
    from testbed.learning_runner import (
        build_ieee9_network, assign_zones, _apply_generator_overrides,
        _compute_naive_lmp_forecast, _compute_exploration_centers,
        _apply_exploration_centers, setup_learning_world,
    )

    with open("testbed/scenarios.yaml") as f:
        cfg = yaml.safe_load(f)["base"]
    gen_params = cfg.get("generators", {})
    training = cfg["training"]

    network = build_ieee9_network(
        start="2025-01-01 00:00", periods=24, profile_cfg=cfg.get("profile"),
    )
    assign_zones(network)
    if gen_params:
        _apply_generator_overrides(network, gen_params)

    rl_agents_set = set(training.get("rl_agents", []))
    forecast = _compute_naive_lmp_forecast(
        network, pd.DatetimeIndex(network.snapshots),
        exclude_agents=rl_agents_set,
    )

    lc = {
        "learning_mode": True, "algorithm": "matd3", "actor_architecture": "mlp",
        "learning_rate": 0.0003, "training_episodes": 2,
        "episodes_collecting_initial_experience": 1,
        "validation_episodes_interval": 5, "train_freq": "12h",
        "min_bid_price": 0.0, "max_bid_price": 100.0,
        "trained_policies_save_path": "", "trained_policies_load_path": "",
        "replay_buffer_size": 50000, "early_stopping_threshold": 0.05,
        "exploration_noise_std": 0.1,
    }

    world = setup_learning_world(
        network, learning_config=lc, market_mechanism="egret_sced",
        price_forecast=forecast, generator_params=gen_params,
    )

    bc_cfg = training.get("bc_warmstart", {})
    centers = _compute_exploration_centers(
        network, bc_cfg, gen_params, price_forecast=forecast,
    )

    # Apply centers
    _apply_exploration_centers(world, centers)

    with open("strats_keys.log", "w") as f:
        f.write(f"centers keys: {list(centers.keys())}\n")
        f.write(f"rl_strats keys: {list(world.learning_role.rl_strats.keys())}\n\n")
        for uid, strat in world.learning_role.rl_strats.items():
            center = strat._exploration_center
            f.write(f"{uid}:\n")
            if center is None:
                f.write("  center = None (NOT SET!)\n")
            elif isinstance(center, list):
                for h, (pf, vf) in enumerate(center):
                    f.write(f"  H{h:02d}: price={pf:.3f} vol={vf:.3f}\n")
            else:
                f.write(f"  center = {center}\n")
    print("Done — see strats_keys.log")

except Exception:
    traceback.print_exc()
