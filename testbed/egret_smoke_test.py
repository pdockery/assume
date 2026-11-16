"""Quick smoke test: Egret SCED (solve_unit_commitment, relaxed=True) with HiGHS."""
import sys
print("Starting smoke test...", flush=True)

from egret.data.model_data import ModelData
from egret.models.unit_commitment import solve_unit_commitment

# Patch: Egret 0.6.0 calls solver.reset() which HiGHS's LegacySolver doesn't have.
# Monkey-patch _lazy_ptdf_termination_cleanup to handle AttributeError gracefully.
import egret.models.unit_commitment as _uc_mod
_orig_cleanup = _uc_mod._lazy_ptdf_termination_cleanup
def _patched_cleanup(m, md, time_periods, solver, ptdf_options, prepend_str,
                     t_subset, persistent_solver, symbolic_solver_labels,
                     solver_tee, vars_to_load, solve_method_options,
                     duals, warmstart_loop, results):
    try:
        return _orig_cleanup(m, md, time_periods, solver, ptdf_options,
                             prepend_str, t_subset, persistent_solver,
                             symbolic_solver_labels, solver_tee, vars_to_load,
                             solve_method_options, duals, warmstart_loop, results)
    except AttributeError as e:
        if 'reset' in str(e):
            # solver doesn't have .reset() — skip it and continue
            results = _uc_mod._lazy_ptdf_solve(
                m, solver, persistent_solver, symbolic_solver_labels,
                solver_tee, vars_to_load, solve_method_options)
            if persistent_solver and duals and results is not None and vars_to_load is None:
                solver.load_duals()
            return results
        raise
_uc_mod._lazy_ptdf_termination_cleanup = _patched_cleanup

md = ModelData({
    'system': {
        'time_keys': [0],
        'time_period_length_minutes': 60,
        'baseMVA': 100.0,
        'reference_bus': 'b1',
        'reference_bus_angle': 0.0,
    },
    'elements': {
        'bus': {
            'b1': {'base_kv': 230, 'matpower_bustype': 'ref', 'vm': 1.0, 'va': 0.0},
            'b2': {'base_kv': 230, 'matpower_bustype': 'PQ', 'vm': 1.0, 'va': 0.0},
        },
        'generator': {
            'g1': {
                'bus': 'b1',
                'in_service': True,
                'generator_type': 'thermal',
                'p_min': 0.0,
                'p_max': 200.0,
                'fixed_commitment': {
                    'data_type': 'time_series',
                    'values': {0: 1},
                },
                'p_cost': {
                    'data_type': 'cost_curve',
                    'cost_curve_type': 'piecewise',
                    'values': [(0, 0), (200, 4000)],
                },
                'startup_cost': 0,
                'shutdown_cost': 0,
                'ramp_up_60min': 200.0,
                'ramp_down_60min': 200.0,
                'min_up_time': 1,
                'min_down_time': 1,
                'initial_status': 1,
                'initial_p_output': 100.0,
            },
        },
        'branch': {
            'l1': {
                'from_bus': 'b1', 'to_bus': 'b2',
                'in_service': True, 'branch_type': 'line',
                'reactance': 0.1, 'resistance': 0.01,
                'rating_long_term': 200, 'rating_short_term': 200,
                'rating_emergency': 200,
            },
        },
        'load': {
            'd1': {
                'bus': 'b2', 'in_service': True,
                'p_load': {
                    'data_type': 'time_series',
                    'values': {0: 100},
                },
            },
        },
    },
})

print("Solving SCED (relaxed=True)...", flush=True)
try:
    result = solve_unit_commitment(md, 'highs', relaxed=True)

    print(f"\nTotal cost: {result.data['system'].get('total_cost', 'N/A')}", flush=True)
    for name, gen in result.elements('generator'):
        pg = gen.get('pg', {})
        print(f"  {name}: pg={pg}", flush=True)
    for name, bus in result.elements('bus'):
        lmp = bus.get('lmp', None)
        if lmp is not None:
            print(f"  {name}: lmp={lmp}", flush=True)
    for name, branch in result.elements('branch'):
        pf = branch.get('pf', None)
        if pf is not None:
            print(f"  {name}: flow={pf}", flush=True)

    print("\nSMOKE TEST PASSED", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
