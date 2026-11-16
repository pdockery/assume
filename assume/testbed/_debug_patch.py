"""Debug: why does calculate_bids return empty for episodes 6+?"""
import sys, os, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from collections import defaultdict
from testbed.learning_runner import setup_learning_world, build_ieee9_network, assign_zones

network = build_ieee9_network(start='2025-01-01 00:00', periods=24)
assign_zones(network)

lc = {
    'learning_mode': True, 'algorithm': 'sac', 'actor_architecture': 'stochastic',
    'learning_rate': 0.0003, 'training_episodes': 10,
    'episodes_collecting_initial_experience': 5, 'validation_episodes_interval': 5,
    'train_freq': '12h', 'gradient_steps': 2, 'batch_size': 64, 'gamma': 0.0,
    'device': 'cpu', 'noise_sigma': 0.1, 'noise_scale': 1, 'noise_dt': 1,
    'action_noise_schedule': 'linear', 'max_bid_price': 100.0, 'min_bid_price': 0.0,
    'trained_policies_save_path': 'testbed_results/test_debug',
    'trained_policies_load_path': '', 'replay_buffer_size': 50000,
    'early_stopping_threshold': 0.05, 'exploration_noise_std': 0.1,
}

def patch_debug(world, bid_log, episode_ref):
    rl_uids = set(world.learning_role.rl_strats.keys())
    for op in world.unit_operators.values():
        for uid, unit in op.units.items():
            if uid not in rl_uids:
                continue
            original = unit.calculate_bids

            def _make(orig, uid_val):
                call_count = [0]
                def patched(market_config, product_tuples, **kwargs):
                    call_count[0] += 1
                    bids = orig(market_config=market_config, product_tuples=product_tuples, **kwargs)
                    if call_count[0] <= 2:
                        # Check what's inside
                        has_strats = market_config.market_id in unit.bidding_strategies
                        strat = unit.bidding_strategies.get(market_config.market_id)
                        strat_type = type(strat).__name__ if strat else "None"
                        print(f"  DETAIL ep={episode_ref[0]} {uid_val} call#{call_count[0]}: "
                              f"has_strats={has_strats} strat={strat_type} "
                              f"n_bids={len(bids)} bids={bids[:1]}")
                    for bid in bids:
                        bid_log.append({
                            "episode": episode_ref[0],
                            "hour": bid["start_time"].hour if hasattr(bid["start_time"], "hour") else 0,
                            "unit_id": uid_val,
                            "bid_price": float(bid["price"]),
                            "bid_volume": float(bid["volume"]),
                        })
                    return bids
                return patched

            unit.calculate_bids = _make(original, uid)

all_bid_log = []
episode_ref = [1]

for episode in range(1, 8):
    episode_ref[0] = episode
    print(f"\n=== Episode {episode} ===")

    world = setup_learning_world(network, episode=episode, learning_config=lc)

    if episode == 1:
        world.learning_role.rl_algorithm.initialize_policy()
        from assume.reinforcement_learning.buffer import ReplayBuffer
        buffer = ReplayBuffer(
            buffer_size=50000,
            obs_dim=world.learning_role.rl_algorithm.obs_dim,
            act_dim=world.learning_role.rl_algorithm.act_dim,
            n_rl_units=len(world.learning_role.rl_strats),
            device=world.learning_role.device,
            float_type=world.learning_role.float_type,
        )
        inter_episodic_data = {
            "buffer": buffer, "actors_and_critics": None,
            "max_eval": defaultdict(lambda: -1e9), "all_eval": defaultdict(list),
            "avg_all_eval": [], "episodes_done": 0, "eval_episodes_done": 0,
        }
        world.learning_role.load_inter_episodic_data(inter_episodic_data)
    else:
        world.learning_role.load_inter_episodic_data(inter_episodic_data)

    patch_debug(world, all_bid_log, episode_ref)

    log_before = len(all_bid_log)
    world.run()
    log_after = len(all_bid_log)
    print(f"  Result: +{log_after - log_before} entries")

    inter_episodic_data = world.learning_role.get_inter_episodic_data()
    inter_episodic_data["episodes_done"] = episode

eps_seen = set(e["episode"] for e in all_bid_log)
print(f"\nTotal: {len(all_bid_log)} entries, episodes: {sorted(eps_seen)}")
