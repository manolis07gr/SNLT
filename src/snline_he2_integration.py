"""snline_he2_integration.py — Wire Phase 4 He II NLTE into the snline pipeline.

This is the Phase 4 analog of snline_he1_integration.py: it adapts the He II
NLTE solver (`nlte_he2.py`) to the pipeline's `state` object conventions
and provides a CSV-friendly summary helper.

Architectural note (Phase 4 v2 coupling):
    Phase 4 owns the He II / He III split. It runs BEFORE Phase 3 in
    process_snapshot so that Phase 3 can read state.he2_diag['X_HeIII']
    and adjust its X_HII input accordingly. This replaces the original
    Phase 3 'follow_H' assumption (X_HeII = X_HII, X_HeIII = 0) with a
    physically-consistent three-state He chemistry.

    For typical SN regimes (T < 25000 K), X_HeIII via local Saha is
    essentially zero, and the Phase 3 result is unchanged. The coupling
    matters in hot shock-photoionized layers and in early IIn-dense CSM.

Public API
----------
add_he2_populations_to_state(state, ...) → state (in place)
    Computes He II populations, attaches `state.he2_levels` (10 × nzones)
    and `state.he2_diag` (dict with X_HeII, X_HeIII, β, τ, named_tau).

he2_summary_dict(state) → dict
    Flat dict of He II diagnostics suitable for batch_metrics.csv export.
"""
import numpy as np
import nlte_he2
import he2_atom as he2


# -----------------------------------------------------------------------------
def _derive_X_HII_for_He2(state):
    """Return per-zone X_HII array for the Phase 4 'photoeq_match' mode.

    Same lookup chain as the He I version: state.h_diag → state.diag.h →
    Saha fallback. Returns array of shape (nzones,) with values in [0, 1].
    """
    n_p = None; n_HI = None
    if hasattr(state, 'h_diag') and isinstance(state.h_diag, dict):
        n_p = state.h_diag.get('n_p', None)
        n_HI = state.h_diag.get('n_HI', None)
    if n_p is None and hasattr(state, 'diag'):
        d = getattr(state.diag, 'h', None) or getattr(state.diag, 'h_nlte', None)
        if isinstance(d, dict):
            n_p = d.get('n_p', None); n_HI = d.get('n_HI', None)
    if n_p is not None and n_HI is not None:
        n_tot = n_p + n_HI
        return np.clip(n_p / np.maximum(n_tot, 1.0), 0.0, 1.0)
    # If state directly carries X_HII (e.g. from snap), use it
    if hasattr(state, 'X_HII'):
        return np.clip(np.asarray(state.X_HII, dtype=float), 0.0, 1.0)
    # Saha fallback (rarely used in production; production_runner provides
    # state.h_diag or state.X_HII)
    KB, HPL, ME, EV = 1.380649e-16, 6.62607015e-27, 9.1093837015e-28, 1.602176634e-12
    chi_H = 13.6 * EV
    T = np.asarray(state.T, dtype=float)
    n_e = np.asarray(state.n_e, dtype=float)
    therm = (HPL * HPL / (2 * np.pi * ME * KB * T)) ** 1.5
    R = (1.0 / therm) * np.exp(-chi_H / (KB * T)) / np.maximum(n_e, 1.0)
    return R / (1.0 + R)


# -----------------------------------------------------------------------------
def _get_He_mass_fraction(state, default=0.245):
    if hasattr(state, 'X_He'):
        v = state.X_He
        if v is None:
            return default
        return v
    if hasattr(state, 'composition') and isinstance(state.composition, dict):
        if 'He' in state.composition:
            return state.composition['He']
    if hasattr(state, 'X') and isinstance(state.X, dict):
        if 'He' in state.X:
            return state.X['He']
    return default


# -----------------------------------------------------------------------------
def add_he2_populations_to_state(state,
                                   X_HeIII_mode='saha_local',
                                   X_HeIII_external=None,
                                   max_iter=120, tol=1e-3, damping=0.4,
                                   verbose=False,
                                   print_summary=True):
    """Compute He II NLTE populations and attach them to `state`.

    Parameters
    ----------
    state : object with .rho, .T, .n_e, .r, .v arrays per zone
        Same convention as add_he1_populations_to_state.
    X_HeIII_mode : str
        How Phase 4 determines X_HeIII per zone:
          'saha_local'   — local Saha at gas T, n_e (default; correct in
                            cool plateau, overestimates in hot NLTE conditions)
          'photoeq_match' — sets n_HeIII/n_HeII ∝ (X_HII/(1-X_HII))² as a
                            crude Wien-suppression scaling
          'zero'         — assume no He III (test mode for plateau-only)
    X_HeIII_external : array, optional
        Per-zone X_HeIII override. Skips X_HeIII_mode entirely.
    max_iter, tol, damping : NLTE iteration controls
    print_summary : if True, prints a one-line [he2_nlte] diagnostic.

    Side effects on state
    ---------------------
    state.he2_levels : ndarray (10, nzones), He II level populations [cm⁻³]
    state.he2_diag   : dict with X_HeII, X_HeIII, β, τ, named_tau, iterations
    state.he2_tau    : dict, line_name → per-zone Sobolev τ for each tracked
                       He II line (4686, 1640, 3203, 10124)

    Returns
    -------
    state (modified in place)
    """
    rho = np.asarray(state.rho, dtype=float)
    T   = np.asarray(state.T,   dtype=float)
    n_e = np.asarray(state.n_e, dtype=float)
    r   = np.asarray(state.r,   dtype=float)
    v   = np.asarray(state.v,   dtype=float)

    X_He = _get_He_mass_fraction(state, default=0.245)
    X_HII = _derive_X_HII_for_He2(state)

    # Phase 4 solver
    n_lev, diag = nlte_he2.he2_populations_nlte(
        rho=rho, T=T, n_e=n_e, r=r, v=v,
        X_He=X_He,
        X_HII_external=X_HII,
        X_HeIII_external=X_HeIII_external,
        v_turb_cms=20e5,
        max_iter=max_iter, tol=tol, damping=damping,
        ionization_split_mode=X_HeIII_mode,
        verbose=verbose)

    # Per-line Sobolev τ (already in diag['named_tau'] from the solver,
    # but we re-build a dict with the same shape as he1 for symmetry)
    line_tau = dict(diag.get('named_tau', {}))

    # Attach to state
    state.he2_levels = n_lev
    state.he2_diag = diag
    state.he2_tau = line_tau

    if print_summary:
        X_HeIII_med = float(np.median(diag['X_HeIII']))
        X_HeII_med  = float(np.median(diag['X_HeII']))
        n_HeII_med  = float(np.median(diag['n_HeII_total']))
        n_HeIII_med = float(np.median(diag['n_HeIII']))
        # Named-line τ medians (defensive against missing keys)
        def _tm(name):
            arr = line_tau.get(name, None)
            if arr is None: return float('nan')
            return float(np.median(arr))
        t_4686 = _tm('He_II_4686')
        t_1640 = _tm('He_II_1640')
        t_3203 = _tm('He_II_3203')
        t_10124 = _tm('He_II_10124')
        iters = diag.get('iterations', 0)
        print(f"[he2_nlte] X_HeIII={X_HeIII_med:.2e}, X_HeII={X_HeII_med:.3f}, "
              f"n_HeIII={n_HeIII_med:.2e}, n_HeII={n_HeII_med:.2e}, "
              f"τ_med(4686)={t_4686:.1e}, (1640)={t_1640:.1e}, "
              f"(3203)={t_3203:.1e}, (10124)={t_10124:.1e}  "
              f"[{iters} iters]")

    return state


# -----------------------------------------------------------------------------
def he2_summary_dict(state):
    """Return a flat dict of He II diagnostics suitable for CSV export.

    Columns produced (9 total, parallel to he1_summary_dict):
        he2_X_HeIII_med       — median X(He III) per zone
        he2_X_HeII_med        — median X(He II) per zone
        he2_n_HeII_med        — median n(He II) total [cm⁻³]
        he2_n_HeIII_med       — median n(He III) [cm⁻³]
        he2_tau_4686_med      — median τ_Sob for 4686 Å (4→3)
        he2_tau_1640_med      — median τ_Sob for 1640 Å (3→2, UV Lyα-analog)
        he2_tau_3203_med      — median τ_Sob for 3203 Å (5→3)
        he2_tau_10124_med     — median τ_Sob for 10124 Å (5→4, NIR)
        he2_iters             — solver convergence iterations

    Returns NaN dict if state lacks the expected attributes.
    """
    nan_dict = {k: float('nan') for k in
                ('he2_X_HeIII_med', 'he2_X_HeII_med',
                 'he2_n_HeII_med', 'he2_n_HeIII_med',
                 'he2_tau_4686_med', 'he2_tau_1640_med',
                 'he2_tau_3203_med', 'he2_tau_10124_med', 'he2_iters')}
    if not hasattr(state, 'he2_levels'):
        return nan_dict
    diag = state.he2_diag
    ltau = state.he2_tau
    def _med(arr_or_key, default=float('nan')):
        if arr_or_key is None: return default
        a = np.asarray(arr_or_key, dtype=float)
        if a.size == 0: return default
        return float(np.median(a))
    return {
        'he2_X_HeIII_med':    _med(diag.get('X_HeIII')),
        'he2_X_HeII_med':     _med(diag.get('X_HeII')),
        'he2_n_HeII_med':     _med(diag.get('n_HeII_total')),
        'he2_n_HeIII_med':    _med(diag.get('n_HeIII')),
        'he2_tau_4686_med':   _med(ltau.get('He_II_4686')),
        'he2_tau_1640_med':   _med(ltau.get('He_II_1640')),
        'he2_tau_3203_med':   _med(ltau.get('He_II_3203')),
        'he2_tau_10124_med':  _med(ltau.get('He_II_10124')),
        'he2_iters':          int(diag.get('iterations', 0)),
    }
