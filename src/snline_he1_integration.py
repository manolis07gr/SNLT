"""snline_he1_integration.py — Wire Phase 3 He I NLTE into the snline pipeline.

This module is the glue between the He I NLTE solver (`nlte_he1.py`) and the
pipeline's `state` object conventions. It:

    (1) Extracts the inputs the He I solver needs from the state
    (2) Determines X_HII for the 'follow_H' ionization split
    (3) Calls he1_populations_nlte()
    (4) Stores populations and diagnostics on the state
    (5) Optionally prints a brief diagnostic summary

Phase 3 ONLY deposits He I level populations. It does NOT yet emit He I
synthetic line profiles — that is Phase 5 (multi-line MC pipeline). For now,
the populations enable line-strength diagnostics (τ_Sob per line, 2³S
metastable buildup), and you can verify physics correctness via those
diagnostics without affecting the Hα MC output.

Public API
----------
add_he1_populations_to_state(state, ...) → state (in place)
    Computes He I populations, attaches `state.he1_levels` (11 × nzones) and
    `state.diag.he1` (dict). Reads X_He from snapshot composition if
    available, defaults to 0.245 otherwise.

derive_X_HII_for_HeI(state) → np.ndarray
    Convenience: pulls per-zone X_HII from the H NLTE results if available,
    or computes from Saha as a fallback.
"""
import numpy as np
import nlte_he1
import he1_atom as he


# -----------------------------------------------------------------------------
def derive_X_HII_for_HeI(state):
    """Return per-zone X_HII array for He I 'follow_H' ionization split.

    Tries, in order:
      1. state.h_diag['n_p'] / (state.h_diag['n_p'] + state.h_diag['n_HI'])
      2. state.diag.h['n_p'] etc. (if diag is a namespace-like object)
      3. Saha equilibrium from state.T and state.n_e as fallback
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
    # Fallback: rough Saha
    KB, HPL, ME, EV = 1.380649e-16, 6.62607015e-27, 9.1093837015e-28, 1.602176634e-12
    chi_H = 13.6 * EV
    T = np.asarray(state.T, dtype=float)
    n_e = np.asarray(state.n_e, dtype=float)
    therm = (HPL * HPL / (2 * np.pi * ME * KB * T)) ** 1.5
    R = (1.0 / therm) * np.exp(-chi_H / (KB * T)) / np.maximum(n_e, 1.0)
    return R / (1.0 + R)


# -----------------------------------------------------------------------------
def _get_He_mass_fraction(state, default=0.245):
    """Extract per-zone or scalar He mass fraction from state if available."""
    # Try common attribute names: state.X_He (array or scalar), state.composition['He']
    if hasattr(state, 'X_He'):
        v = state.X_He
        if v is None:
            return default
        return v
    if hasattr(state, 'composition') and isinstance(state.composition, dict):
        if 'He' in state.composition:
            return state.composition['He']
    # Some snapshots store as state.X with per-element fields
    if hasattr(state, 'X') and isinstance(state.X, dict):
        if 'He' in state.X:
            return state.X['He']
    return default


# -----------------------------------------------------------------------------
def add_he1_populations_to_state(state,
                                   eps_He_resonance=None,
                                   two_photon_decay=True,
                                   include_M1_decay=True,
                                   ionization_split_mode='follow_H',
                                   max_iter=120, tol=1e-3, damping=0.4,
                                   X_HeIII_external=None,
                                   verbose=False,
                                   print_summary=True):
    """Compute He I NLTE populations and attach them to `state`.

    The He I solver is called with these key choices:
      - two_photon_decay=True: A=51.3 s⁻¹ destruction of 2¹S (singlet 2γ)
      - include_M1_decay=True: A=1.27e-4 s⁻¹ for 2³S→1¹S (tiny, but exact)
      - ionization_split_mode='follow_H': X_HeII ≈ X_HII per-zone, the
                                            simplest credible approximation
                                            (Phase 4 will refine)

    Parameters
    ----------
    state : object with .rho, .T, .n_e, .r, .v arrays per zone
    eps_He_resonance : optional float, ε floor on 2¹P→1¹S (584 Å) Sobolev β.
                       Analog of H Lyα ε floor. Default None (no floor),
                       i.e. only 2γ destruction from 2¹S. Set to 1e-4 if you
                       want symmetric treatment with H Lyα.
    print_summary : if True, prints a one-line diagnostic to stdout.

    Side effects on state
    ---------------------
    state.he1_levels : ndarray (11, nzones), populations [cm⁻³] per He I level
    state.he1_diag   : dict with full diagnostics (β, τ, iters, etc.)
    state.he1_tau    : dict, line_name → per-zone Sobolev τ for each tracked
                       He I line (5876, 6678, 7065, 10830). For diagnostic
                       use until Phase 5 wires multi-line MC.

    Returns
    -------
    state (modified in place)
    """
    # Inputs
    rho = np.asarray(state.rho, dtype=float)
    T   = np.asarray(state.T,   dtype=float)
    n_e = np.asarray(state.n_e, dtype=float)
    r   = np.asarray(state.r,   dtype=float)
    v   = np.asarray(state.v,   dtype=float)

    X_He = _get_He_mass_fraction(state, default=0.245)
    X_HII = derive_X_HII_for_HeI(state)

    # ---- Phase 4 → Phase 3 coupling (v2) ----
    # When X_HeIII_external is provided (typically by Phase 4 He II NLTE
    # via state.he2_diag['X_HeIII']), the He I solver needs to know that
    # a fraction X_HeIII of total He has been promoted to He III and is
    # no longer in the He I/He II system. Two coupled adjustments are
    # needed (BOTH are required — adjusting only X_HII overcounts n_HeI
    # by exactly X_HeIII × n_He_total):
    #
    #   X_He_for_solver  = X_He × (1 − X_HeIII)
    #       ↳ reduces the total He available to the He I solver
    #
    #   X_HII_for_solver = (X_HII − X_HeIII) / (1 − X_HeIII)
    #       ↳ rescales X_HII to the He II/He I sub-system
    #
    # Verification with X_HII=1, X_HeIII=0.20:
    #   X_HII_for_solver = (1−0.20)/(1−0.20) = 1.00 (still fully ionized)
    #   X_He_for_solver  = X_He × 0.80
    #   Result: n_HeI = 0 × 0.80 × n_He = 0,  n_HeII = 1 × 0.80 × n_He
    #           (the other 0.20 × n_He is in He III, owned by Phase 4)
    # Verification with X_HeIII = 0 (cool plateau):
    #   Identity transform — bit-identical to no-coupling behavior.
    if X_HeIII_external is None and hasattr(state, 'he2_diag') and \
            isinstance(state.he2_diag, dict):
        X_HeIII_external = state.he2_diag.get('X_HeIII', None)
    if X_HeIII_external is not None:
        X_HeIII_arr = np.asarray(X_HeIII_external, dtype=float)
        if X_HeIII_arr.size == X_HII.size:
            # Clip X_HeIII to avoid singularity at X_HeIII = 1
            X_HeIII_arr = np.clip(X_HeIII_arr, 0.0, 0.999999)
            not_he3 = 1.0 - X_HeIII_arr
            X_HII_for_solver = np.clip(
                (X_HII - X_HeIII_arr) / np.maximum(not_he3, 1e-6),
                0.0, 1.0)
            # X_He may be scalar or per-zone; broadcasting handles both
            if np.isscalar(X_He):
                X_He_for_solver = X_He * not_he3
            else:
                X_He_arr = np.asarray(X_He, dtype=float)
                X_He_for_solver = X_He_arr * not_he3
            X_HeIII_med = float(np.median(X_HeIII_arr))
            if verbose or X_HeIII_med > 0.01:
                print(f"  [he1_nlte] coupling: X_HeIII from Phase 4 "
                      f"(median={X_HeIII_med:.2e}) → reducing both "
                      f"X_He and X_HII for the He I solver")
        else:
            X_HII_for_solver = X_HII
            X_He_for_solver = X_He
    else:
        X_HII_for_solver = X_HII
        X_He_for_solver = X_He

    # Phase 3 solver
    n_lev, diag = nlte_he1.he1_populations_nlte(
        rho=rho, T=T, n_e=n_e, r=r, v=v,
        X_He=X_He_for_solver,
        X_HII_external=X_HII_for_solver,
        v_turb_cms=20e5,
        max_iter=max_iter, tol=tol, damping=damping,
        ionization_split_mode=ionization_split_mode,
        two_photon_decay=two_photon_decay,
        eps_He_resonance=eps_He_resonance,
        include_M1_decay=include_M1_decay,
        verbose=verbose)

    # Per-line Sobolev τ for diagnostics (cheap; we already have populations).
    # Robust to duplicate STELLA shock radii (see velocity_grad).
    from velocity_grad import robust_dvdr
    dv_dr_eff = np.maximum(robust_dvdr(v, r), 20e5 / np.maximum(r, 1e-30))
    line_tau = {}
    for line in ('He_I_5876', 'He_I_6678', 'He_I_7065', 'He_I_10830'):
        line_tau[line] = nlte_he1.line_optical_depth(n_lev, line, T, dv_dr_eff)

    # Attach to state
    state.he1_levels = n_lev
    state.he1_diag = diag
    state.he1_tau = line_tau

    if print_summary:
        # LTE prediction for 2³S/1¹S at the median T, for NLTE comparison
        T_med = float(np.median(T))
        EV_KB = 1.602e-12 / 1.381e-16
        lte_ratio = 3.0 * np.exp(-19.82 * EV_KB / T_med)
        nlte_ratio = float(np.median(n_lev[1]) / np.maximum(np.median(n_lev[0]), 1e-30))
        nlte_lte = nlte_ratio / max(lte_ratio, 1e-300)
        tau_med_10830 = float(np.median(line_tau['He_I_10830']))
        tau_med_5876  = float(np.median(line_tau['He_I_5876']))
        tau_med_7065  = float(np.median(line_tau['He_I_7065']))
        tau_med_6678  = float(np.median(line_tau['He_I_6678']))
        n_HeI = float(np.median(diag['n_HeI_total']))
        n_HeII = float(np.median(diag['n_HeII']))
        iters = diag['iterations']
        print(f"[he1_nlte] n_HeI={n_HeI:.2e}, n_HeII={n_HeII:.2e}, "
              f"NLTE/LTE(2³S/1¹S)={nlte_lte:.1e}, "
              f"τ_med(10830)={tau_med_10830:.1e}, (5876)={tau_med_5876:.1e}, "
              f"(7065)={tau_med_7065:.1e}, (6678)={tau_med_6678:.1e}  "
              f"[{iters} iters]")

    return state


# -----------------------------------------------------------------------------
def he1_summary_dict(state):
    """Return a flat dict of He I diagnostics suitable for CSV export.

    Columns: he1_n_HeI_med, he1_n_HeII_med, he1_2_3S_frac, he1_NLTE_LTE_ratio,
             he1_tau_10830_med, he1_tau_5876_med, he1_tau_7065_med,
             he1_tau_6678_med, he1_iters
    """
    if not hasattr(state, 'he1_levels'):
        return {k: float('nan') for k in
                ('he1_n_HeI_med', 'he1_n_HeII_med', 'he1_2_3S_frac',
                 'he1_NLTE_LTE_ratio', 'he1_tau_10830_med', 'he1_tau_5876_med',
                 'he1_tau_7065_med', 'he1_tau_6678_med', 'he1_iters')}

    n_lev = state.he1_levels
    diag = state.he1_diag
    line_tau = state.he1_tau
    T = np.asarray(state.T, dtype=float)
    T_med = float(np.median(T))
    EV_KB = 1.602e-12 / 1.381e-16
    lte_ratio = 3.0 * np.exp(-19.82 * EV_KB / T_med)
    nlte_ratio = float(np.median(n_lev[1]) / np.maximum(np.median(n_lev[0]),
                                                          1e-30))
    return {
        'he1_n_HeI_med':       float(np.median(diag['n_HeI_total'])),
        'he1_n_HeII_med':      float(np.median(diag['n_HeII'])),
        'he1_2_3S_frac':       float(np.median(n_lev[1]) /
                                       max(np.median(diag['n_HeI_total']),
                                            1e-30)),
        'he1_NLTE_LTE_ratio':  nlte_ratio / max(lte_ratio, 1e-300),
        'he1_tau_10830_med':   float(np.median(line_tau['He_I_10830'])),
        'he1_tau_5876_med':    float(np.median(line_tau['He_I_5876'])),
        'he1_tau_7065_med':    float(np.median(line_tau['He_I_7065'])),
        'he1_tau_6678_med':    float(np.median(line_tau['He_I_6678'])),
        'he1_iters':           int(diag['iterations']),
    }
