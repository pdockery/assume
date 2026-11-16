# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Monkey-patches for HiGHS + Egret 0.6.0 compatibility.

Two known issues:
1. HiGHS LegacySolver doesn't have solver.reset()
2. HiGHS LegacySolver doesn't support warmstart=True kwarg

Call patch_egret_for_highs() once at import time.
"""

_patched = False


def patch_egret_for_highs():
    """Apply monkey-patches to Egret for HiGHS compatibility. Safe to call multiple times."""
    global _patched
    if _patched:
        return
    _patched = True

    import egret.models.unit_commitment as _uc_mod

    _orig_ptdf_solve = _uc_mod._lazy_ptdf_solve

    def _patched_ptdf_solve(m, solver, persistent_solver, symbolic_solver_labels,
                            solver_tee, vars_to_load, solve_method_options=None):
        main_model = m.model()
        if solve_method_options is None:
            solve_method_options = {}
        if persistent_solver:
            results = solver.solve(main_model, tee=solver_tee, load_solutions=False,
                                   save_results=False, warmstart=True, **solve_method_options)
            solver.load_vars(vars_to_load)
        else:
            try:
                results = solver.solve(main_model, tee=solver_tee,
                                       symbolic_solver_labels=symbolic_solver_labels,
                                       load_solutions=False, warmstart=True,
                                       **solve_method_options)
            except TypeError:
                results = solver.solve(main_model, tee=solver_tee,
                                       symbolic_solver_labels=symbolic_solver_labels,
                                       load_solutions=False,
                                       **solve_method_options)
            m.solutions.load_from(results)
        return results

    _uc_mod._lazy_ptdf_solve = _patched_ptdf_solve

    _orig_cleanup = _uc_mod._lazy_ptdf_termination_cleanup

    def _patched_cleanup(m, md, time_periods, solver, ptdf_options, prepend_str,
                         t_subset, persistent_solver, symbolic_solver_labels,
                         solver_tee, vars_to_load, solve_method_options,
                         duals, warmstart_loop, results):
        try:
            return _orig_cleanup(m, md, time_periods, solver, ptdf_options, prepend_str,
                                 t_subset, persistent_solver, symbolic_solver_labels,
                                 solver_tee, vars_to_load, solve_method_options,
                                 duals, warmstart_loop, results)
        except (AttributeError, TypeError) as e:
            if "reset" in str(e) or "warmstart" in str(e):
                results = _patched_ptdf_solve(m, solver, persistent_solver,
                                              symbolic_solver_labels, solver_tee,
                                              vars_to_load, solve_method_options)
                if persistent_solver and duals and results is not None and vars_to_load is None:
                    solver.load_duals()
                return results
            raise

    _uc_mod._lazy_ptdf_termination_cleanup = _patched_cleanup
