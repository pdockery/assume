"""Debug: trace the EXACT accepted_price the reward function receives."""
import sys, os, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import testbed.learning_runner as lr
lr.RL_GENERATORS = {"gen_0"}
lr.RL_STRATEGY = "powerplant_energy_withholding"

from testbed.ieee9_network import build_ieee9_network, assign_zones
from testbed.learning_runner import setup_learning_world, _track_rewards, _clamp_actions_nonnegative, _bias_actor_toward_mc
from collections import defaultdict
from assume.reinforcement_learning.buffer import ReplayBuffer
import numpy as np

network = build_ieee9_network(periods=6)
assign_zones(network)

lc = {
    'learning_mode': True, 'algorithm': 'matd3', 'actor_architecture': 'mlp',
    'learning_rate': 0.0001, 'training_episodes': 1,
    'episodes_collecting_initial_experience': 1, 'validation_episodes_interval': 5,
    'train_freq': '6h', 'gradient_steps': 1, 'batch_size': 64,
    'gamma': 0.95, 'device': 'cpu', 'noise_sigma': 0.1, 'noise_scale': 1,
    'noise_dt': 1, 'action_noise_schedule': 'linear',
    'max_bid_price': 100, 'min_bid_price': 0,
    'trained_policies_save_path': 'testbed_results/debug_test',
    'trained_policies_load_path': '', 'replay_buffer_size': 50000,
    'early_stopping_threshold': 0.05, 'exploration_noise_std': 0.2,
}

world = setup_learning_world(network, episode=1, learning_config=lc)
world.learning_role.rl_algorithm.initialize_policy()
_bias_actor_toward_mc(world)
_clamp_actions_nonnegative(world)

ied = {
    'buffer': ReplayBuffer(
        buffer_size=50000,
        obs_dim=world.learning_role.rl_algorithm.obs_dim,
        act_dim=world.learning_role.rl_algorithm.act_dim,
        n_rl_units=len(world.learning_role.rl_strats),
        device=world.learning_role.device,
        float_type=world.learning_role.float_type,
    ),
    'actors_and_critics': None,
    'max_eval': defaultdict(lambda: -1e9),
    'all_eval': defaultdict(list),
    'avg_all_eval': [],
    'episodes_done': 0,
    'eval_episodes_done': 0,
}
world.learning_role.load_inter_episodic_data(ied)

# Patch WithholdingStrategy.calculate_reward to print debug
for uid, strat in world.learning_role.rl_strats.items():
    orig = strat.calculate_reward
    def wrap(unit, marketconfig, orderbook, _o=orig, _u=uid):
        n = len(orderbook)
        mcp = orderbook[0]['accepted_price']
        s = orderbook[0]['start_time']
        mc = unit.calculate_marginal_cost(s, unit.outputs[marketconfig.product_type].at[s])
        total_acc = sum(o.get('accepted_volume', 0) for o in orderbook)
        total_vol = sum(o.get('volume', 0) for o in orderbook)
        bp = orderbook[0].get('price', 0)
        # Also check what the SCED meta says (check if we can access it)
        print(f"RW {_u} H{s.hour:02d} n_orders={n}: bid=${bp:.1f} vol_offered={total_vol:.0f} "
              f"vol_accepted={total_acc:.0f} accepted_price=${mcp:.2f} MC=${mc:.1f} "
              f"profit$={(mcp - mc) * total_acc:.0f}")
        return _o(unit, marketconfig, orderbook)
    strat.calculate_reward = wrap

# Also patch the SCED clearing to show what LMPs it produces
from testbed.clearing import egret_adapter
orig_results = egret_adapter.model_results_to_orderbook
def debug_results(result, orderbook, market_products, order_gen_map=None):
    accepted, rejected, meta, flows = orig_results(result, orderbook, market_products, order_gen_map=order_gen_map)
    for m in meta:
        h = m['product_start'].hour
        lmps = m.get('bus_lmps', {})
        avg = m['price']
        print(f"SCED H{h:02d}: avg_LMP=${avg:.2f} bus_LMPs={{{', '.join(f'{b}:${v:.2f}' for b,v in sorted(lmps.items()))}}}")
    return accepted, rejected, meta, flows
egret_adapter.model_results_to_orderbook = debug_results

rl = _track_rewards(world)
world.run()
print(f"\nRewards: {[round(r, 6) for r in rl]}")
