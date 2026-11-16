"""Check what exploration centers are computed after the vol_frac fix."""
import sys
import logging
import traceback

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

try:
    import yaml
    import pandas as pd
    from testbed.learning_runner import (
        build_ieee9_network, assign_zones, _apply_generator_overrides,
        _compute_exploration_centers, _compute_naive_lmp_forecast,
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

    bc_cfg = training.get("bc_warmstart", {})
    centers = _compute_exploration_centers(
        network, bc_cfg, gen_params, price_forecast=forecast,
    )

    with open("centers_debug.log", "w") as out:
        for gen, hourly in centers.items():
            out.write(f"{gen}:\n")
            for h, (pf, vf) in enumerate(hourly):
                out.write(f"  H{h:02d}: price_frac={pf:.3f}  vol_frac={vf:.3f}\n")
    print("Done — see centers_debug.log")

except Exception:
    traceback.print_exc()
    with open("centers_debug.log", "w") as out:
        traceback.print_exc(file=out)
