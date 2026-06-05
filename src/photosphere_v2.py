"""photosphere_v2.py — Continuum-opacity-aware photosphere finder.

Replaces the τ_es = 2/3 truncation in stella_io with a proper τ_cont = 2/3
photosphere. For Thomson-thick IIn-regime snapshots the two coincide; for
plateau-phase (IIP-like) snapshots the continuum photosphere sits at the
H recombination front, outside the τ_es surface, and L_phot becomes self-
consistent with L_bol.

Public API
----------
  find_photosphere(snap_full, lam_ref_AA=6562.8, populations='saha') → dict
      Returns:
        R_phot_cont   : continuum photosphere radius [cm]
        T_phot_cont   : gas temperature there [K]
        idx_phot_cont : zone index of the photosphere
        R_phot_es     : Thomson-only photosphere (for comparison) [cm]
        idx_phot_es   : zone index of τ_es = 2/3 surface
        tau_es_full   : per-zone cumulative τ_es from outside
        tau_cont_full : per-zone cumulative τ_cont from outside (at lam_ref)
        tau_therm     : per-zone effective τ (= √(τ_abs(τ_abs+τ_es)))
        R_therm       : thermalization depth (τ_therm = 1) radius
        T_color       : gas T at thermalization depth (BB source temperature)
        channel_at_phot : dict of κ_es, κ_bf, κ_H-, κ_ff at the photosphere
                          (diagnostic — tells you which opacity defines R_phot)

  truncate_at_cont_photosphere(snap_full, **kwargs) → truncated snap
      Drop-in replacement for stella_io.truncate_to_photosphere().
      All zones with r ≥ R_phot_cont are kept.
"""
import numpy as np

import opacity as op

KB        = 1.380649e-16
HPL       = 6.62607015e-27
ME        = 9.1093837015e-28
MH        = 1.6735575e-24
C_LIGHT   = 2.99792458e10
SIGMA_T   = 6.6524587e-25
EV        = 1.602176634e-12


# ===========================================================================
# Saha-Boltzmann population estimator (used to seed κ_cont before NLTE runs)
# ===========================================================================
def saha_boltzmann_H_populations(T, n_e, X_H, rho, nlev=5):
    """Quick Saha-Boltzmann for HI level populations [cm⁻³].

    Returns (nlev, nzones) array; index 0 = n=1 ground state.

    Used as a seed before NLTE has converged. Sufficient for τ_cont = 2/3
    photosphere finding to ~10% accuracy (the photosphere is set mostly by
    ground-state H bf at λ < 3646 Å OR by H⁻ which depends on n_HI_ground;
    not very sensitive to excited-level NLTE departures).
    """
    T = np.asarray(T, dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    rho = np.asarray(rho, dtype=float)
    X_H = np.asarray(X_H, dtype=float) if np.ndim(X_H) else np.full_like(T, X_H)

    n_H_tot = X_H * rho / MH
    # Saha for H ionization
    chi_H = 13.6 * EV
    g_HII = 1.0
    g_HI = 2.0
    g_e = 2.0
    thermal_vol = (HPL * HPL / (2 * np.pi * ME * KB * T))**1.5
    # Saha: n_HII × n_e / n_HI = (g_HII × g_e / g_HI) × (1/thermal_vol) × exp(-chi/kT)
    # → X_HII = n_HII / n_H_tot
    # ratio = n_HII/n_HI:
    ratio = (g_HII * g_e / g_HI) * (1.0 / thermal_vol) * np.exp(-chi_H / (KB * T)) / np.maximum(n_e, 1.0)
    X_HII = ratio / (1.0 + ratio)
    n_HI = (1.0 - X_HII) * n_H_tot

    # Boltzmann for level populations: n(level k) / n_HI = g_k/Z(T) × exp(-E_k/kT)
    # E_k = -Ry/k² → E_k - E_1 = Ry × (1 - 1/k²)
    nzones = T.size
    n_per_lev = np.zeros((nlev, nzones))
    Z = np.zeros(nzones)
    for k in range(1, nlev + 1):
        g_k = 2 * k * k
        E_k = 13.6 * EV * (1.0 - 1.0 / (k * k))
        weight = g_k * np.exp(-E_k / (KB * T))
        n_per_lev[k - 1] = weight
        Z += weight
    Z = np.maximum(Z, 1e-30)
    for k in range(nlev):
        n_per_lev[k] *= n_HI / Z

    return n_per_lev


# ===========================================================================
# Cumulative-tau integrator
# ===========================================================================
def cumulative_tau_inward(r, kappa):
    """Compute cumulative optical depth integrated inward from r_outer.

    τ(r) = ∫_r^{r_outer} κ(r') dr'

    Parameters
    ----------
    r : (nzones,) radial coordinate [cm], assumed monotonically increasing
    kappa : (nzones,) opacity [cm⁻¹]

    Returns
    -------
    tau : (nzones,) optical depth integrated from outside; tau[-1] = 0
          (outer boundary), tau[0] = total τ across grid.
    """
    r = np.asarray(r, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    # trapezoidal integration of κ from r outward to r_outer
    # τ(r_i) = Σ_{j > i} (κ_j + κ_{j-1})/2 × (r_j - r_{j-1})
    # Implement by reversing, cumulative-summing increments, reversing back
    increments = 0.5 * (kappa[:-1] + kappa[1:]) * np.diff(r)
    # increments[k] is the contribution between r[k] and r[k+1]
    # τ(r_i) = sum of increments from i to N-1
    tau = np.zeros_like(r)
    tau[-1] = 0.0
    cumsum_from_right = np.cumsum(increments[::-1])[::-1]
    tau[:-1] = cumsum_from_right
    return tau


def find_index_at_tau(tau, tau_target=2.0/3.0):
    """Return the index of the deepest zone with tau ≥ tau_target."""
    above = np.where(tau >= tau_target)[0]
    if above.size == 0:
        # τ never reaches target — atmosphere is optically thin
        return 0
    return int(above[-1])    # innermost zone above target


# ===========================================================================
# Thermalization depth (Hummer 1973 sense)
# ===========================================================================
def find_thermalization_depth(r, kappa_abs, kappa_es, tau_eff_target=1.0):
    """Thermalization depth at which τ_eff = √(τ_abs × (τ_abs + τ_es)) ≈ 1.

    Hummer (1973). Photons emitted deeper than this are thermalized to T_e;
    photons emitted shallower escape without full thermalization. Used to
    define T_color (the BB source temperature seen from outside).

    Approximation: we use cumulative τ_abs and τ_es from outside, take
    τ_eff(r) = √(τ_abs(r) × (τ_abs(r) + τ_es(r))), find inner zone where
    τ_eff just exceeds tau_eff_target.
    """
    tau_abs = cumulative_tau_inward(r, kappa_abs)
    tau_es  = cumulative_tau_inward(r, kappa_es)
    tau_eff = np.sqrt(tau_abs * (tau_abs + tau_es))
    idx = find_index_at_tau(tau_eff, tau_target=tau_eff_target)
    return idx, tau_eff


# ===========================================================================
# Main API
# ===========================================================================
def find_photosphere(snap_full, lam_ref_AA=6562.8, populations='saha',
                      verbose=True):
    """Locate the continuum photosphere and thermalization depth.

    Parameters
    ----------
    snap_full : dict with keys 'r','rho','v','T','n_e','X_H' on the FULL
                snapshot grid (NOT truncated). All arrays length nzones.
    lam_ref_AA : reference wavelength for κ_cont evaluation. Default Hα.
                 For very late epochs with cool T_color, set this to a
                 wavelength below the Balmer edge (e.g. 5500 Å) so the
                 photosphere is in the visible continuum.
    populations : 'saha' (default) or 'use_snap_X_HII'. The latter uses
                  snap['X_HII'] if present (e.g. from upstream photoeq).

    Returns
    -------
    dict (see module docstring).
    """
    r   = np.asarray(snap_full['r'], dtype=float)
    rho = np.asarray(snap_full['rho'], dtype=float)
    T   = np.asarray(snap_full['T'], dtype=float)
    n_e = np.asarray(snap_full['n_e'], dtype=float)
    X_H = np.asarray(snap_full.get('X_H', np.full_like(r, 0.737)), dtype=float)

    # ----- estimate level populations -----
    if populations == 'use_snap_X_HII' and 'X_HII' in snap_full:
        X_HII = np.asarray(snap_full['X_HII'], dtype=float)
        n_HI_total = (1.0 - X_HII) * X_H * rho / MH
        # Boltzmann for level structure within HI
        nlev_use = 5
        n_per_lev = np.zeros((nlev_use, r.size))
        Z = np.zeros_like(T)
        for k in range(1, nlev_use + 1):
            g_k = 2 * k * k
            E_k = 13.6 * EV * (1.0 - 1.0 / (k * k))
            w = g_k * np.exp(-E_k / (KB * T))
            n_per_lev[k - 1] = w
            Z += w
        Z = np.maximum(Z, 1e-30)
        for k in range(nlev_use):
            n_per_lev[k] *= n_HI_total / Z
        n_p = X_HII * X_H * rho / MH
    else:
        n_per_lev = saha_boltzmann_H_populations(T, n_e, X_H, rho, nlev=5)
        # ground state and proton density from Saha
        n_HI = n_per_lev.sum(axis=0)
        n_H_tot = X_H * rho / MH
        n_p = np.maximum(n_H_tot - n_HI, 0.0)

    # ----- per-zone continuum opacity at reference wavelength -----
    # P1 #4 root-fix: H-rich gas uses the four H channels (default → results
    # byte-identical). For H-free (He-rich) gas, add helium free-free — the only
    # continuum channel that matters at optical/NIR λ (He bound-free thresholds
    # are deep in the EUV). n_HeII is estimated from the free-electron budget
    # (leading singly-ionized estimate, capped at the total He); this is
    # negligible for cold neutral He and grows where He is ionized (the
    # interaction phase), placing the H-free photosphere more correctly.
    cont_include = ['es', 'bf', 'H-', 'ff']
    he_kwargs = {}
    X_H_mean = float(np.mean(X_H)) if np.ndim(X_H) else float(X_H)
    if X_H_mean < 1.0e-3:
        MHE = 6.646e-24                       # g, mass of He atom
        X_He = snap_full.get('X_He', 0.98)
        X_He = (np.asarray(X_He, dtype=float) if np.ndim(X_He)
                else np.full_like(r, float(X_He)))
        n_He_tot = X_He * rho / MHE
        n_HeII_est = np.minimum(np.asarray(n_e, dtype=float), n_He_tot)
        cont_include.append('He-ff')
        he_kwargs = dict(n_HeII=n_HeII_est,
                         n_HeIII=np.zeros_like(n_HeII_est))
    kappa_total = op.kappa_cont_total(
        lam_ref_AA, T, n_e, n_p, n_per_lev,
        include=tuple(cont_include), **he_kwargs)
    kappa_es = op.kappa_thomson(n_e)
    kappa_abs = kappa_total - kappa_es     # absorption-only
    kappa_abs = np.maximum(kappa_abs, 0.0)

    # ----- cumulative τ from outside -----
    tau_cont = cumulative_tau_inward(r, kappa_total)
    tau_es   = cumulative_tau_inward(r, kappa_es)

    # ----- find τ_cont = 2/3 -----
    idx_cont = find_index_at_tau(tau_cont, tau_target=2.0/3.0)
    R_phot_cont = float(r[idx_cont])
    T_phot_cont = float(T[idx_cont])

    # ----- find τ_es = 2/3 (for comparison) -----
    idx_es = find_index_at_tau(tau_es, tau_target=2.0/3.0)
    R_phot_es = float(r[idx_es])

    # ----- thermalization depth (τ_eff = 1) → T_color -----
    idx_therm, tau_therm = find_thermalization_depth(
        r, kappa_abs, kappa_es, tau_eff_target=1.0)
    R_therm = float(r[idx_therm])
    T_color = float(T[idx_therm])

    # ----- channel breakdown at the photosphere zone -----
    z = idx_cont
    channel = dict(
        es  = float(kappa_es[z]),
        bf  = float(op.kappa_bf_H(lam_ref_AA, n_per_lev[:, z:z+1])[0]),
        Hm  = float(op.kappa_H_minus(lam_ref_AA, T[z:z+1],
                                      n_per_lev[0, z:z+1], n_e[z:z+1])[0]),
        ff  = float(op.kappa_ff_H(lam_ref_AA, T[z:z+1],
                                   n_e[z:z+1], n_p[z:z+1])[0]),
    )
    channel['total'] = sum(channel.values())

    result = dict(
        R_phot_cont   = R_phot_cont,
        T_phot_cont   = T_phot_cont,
        idx_phot_cont = idx_cont,
        R_phot_es     = R_phot_es,
        idx_phot_es   = idx_es,
        R_therm       = R_therm,
        T_color       = T_color,
        idx_therm     = idx_therm,
        tau_es_full   = tau_es,
        tau_cont_full = tau_cont,
        tau_therm     = tau_therm,
        channel_at_phot = channel,
        lam_ref_AA    = lam_ref_AA,
    )

    if verbose:
        print(f"[photosphere_v2] λ_ref = {lam_ref_AA:.0f} Å")
        print(f"  τ_cont = 2/3 at zone {idx_cont:3d}: "
              f"r = {R_phot_cont:.4e} cm, T = {T_phot_cont:.0f} K")
        print(f"  τ_es   = 2/3 at zone {idx_es:3d}: "
              f"r = {R_phot_es:.4e} cm  ({(R_phot_cont/R_phot_es - 1)*100:+.1f}% vs cont)")
        print(f"  τ_eff = 1  at zone {idx_therm:3d}: "
              f"R_therm = {R_therm:.4e} cm, T_color = {T_color:.0f} K")
        print(f"  Photosphere opacity breakdown [cm⁻¹]:")
        for k, v in channel.items():
            frac = (v / channel['total'] * 100) if channel['total'] > 0 else 0
            print(f"    κ_{k:5s} = {v:.3e}   ({frac:5.1f}% of total)")

    return result


def truncate_at_cont_photosphere(snap_full, lam_ref_AA=6562.8,
                                  populations='saha',
                                  r_phot=None,
                                  tau_target=2.0/3.0,
                                  t_color_legacy_tau_es=10.0,
                                  extra_keys=(),
                                  verbose=True):
    """Drop-in replacement for stella_io.truncate_to_photosphere().

    Truncates the FULL snapshot to zones above the *continuum-opacity*
    photosphere τ_cont(λ_ref) = tau_target, at the reference wavelength.
    For IIn-regime (Thomson-thick) snapshots this gives the same answer as
    the legacy τ_es=2/3 truncation. For IIP-plateau snapshots the surface
    moves outward to the H recombination front.

    Backward compatibility
    ----------------------
    Returns ALL keys that the legacy truncate_to_photosphere set, so that
    downstream code (`solve_photoionization_equilibrium`, etc.) reads its
    expected fields without modification. New diagnostic keys are also
    added with non-colliding names.

    Legacy keys returned (unchanged semantics):
        R_phot_inner, T_phot_inner, L_phot_inner       (now from τ_cont surface)
        n_zones_full, r_phot_used, tau_es_threshold_used
        T_color_at_tau_es (dict)                       (τ_es=10 heuristic preserved)
        T_color_thermalization                          (τ_es=10 value, legacy default)

    New diagnostic keys (don't collide with legacy):
        R_phot_cont, T_phot_cont, idx_phot_cont
        R_phot_es,   idx_phot_es                        (τ_es=2/3 for comparison)
        R_therm, T_color_hummer                         (Hummer τ_eff=1 definition)
        photosphere_kappa_breakdown                     (which channel dominates)
        photosphere_lam_ref_AA

    Parameters
    ----------
    snap_full : dict
        FULL untruncated snapshot from load_stella_snapshot().
    lam_ref_AA : float, default 6562.8 (Hα)
        Reference wavelength at which κ_cont is evaluated for τ_cont = 2/3.
    populations : {'saha', 'use_snap_X_HII'}
        How to estimate HI level populations for κ_bf. 'saha' uses local-T
        Saha equilibrium; 'use_snap_X_HII' uses snap['X_HII'] if present.
    r_phot : float or None
        If not None, override: use this radius as the photosphere. Mimics
        the legacy `r_phot` kwarg for the wind-extension code path.
    tau_target : float
        Photosphere defined as τ_cont(λ_ref) = tau_target. Default 2/3.
    t_color_legacy_tau_es : float
        For backward compatibility with the legacy T_color heuristic.
        Default 10.0 (matches old code's choice).
    extra_keys : tuple of str
        Additional keys to copy through under truncation (besides the
        standard r, v, rho, T, n_e, X_H, tau_es).
    """
    import numpy as np
    r = np.asarray(snap_full['r'])
    n_zones_full = r.size

    # ----- Find both photospheres at λ_ref -----
    phot = find_photosphere(snap_full, lam_ref_AA=lam_ref_AA,
                             populations=populations, verbose=False)

    # ----- Handle r_phot override (legacy wind-extension path) -----
    if r_phot is not None:
        # Caller passed explicit r_phot — use it, but still expose Phase 1
        # diagnostics for the matching radius
        idx_use = int(np.searchsorted(r, r_phot))
        if idx_use >= r.size:
            raise ValueError(f"r_phot={r_phot:.3e} is outside outer "
                             f"boundary r[-1]={r[-1]:.3e}")
        r_phot_used = float(r[idx_use])
        idx_cont = idx_use
    else:
        idx_cont = phot['idx_phot_cont']
        r_phot_used = phot['R_phot_cont']

    # ----- Build truncation mask (keep zones with r > r_phot_used) -----
    mask = r > r_phot_used
    n_keep = int(mask.sum())
    if n_keep < 5:
        raise ValueError(f"Truncation at r_phot={r_phot_used:.3e} cm leaves "
                         f"only {n_keep} zones; need at least 5.")

    # ----- Copy through per-zone arrays under truncation -----
    snap_out = {}
    for key, val in snap_full.items():
        if isinstance(val, np.ndarray) and val.ndim == 1 and val.size == n_zones_full:
            snap_out[key] = val[mask].copy()
        elif key in ('composition', 'comp') and isinstance(val, dict):
            snap_out[key] = {sp: arr[mask].copy() for sp, arr in val.items()
                              if isinstance(arr, np.ndarray)
                              and arr.size == n_zones_full}
        else:
            snap_out[key] = val

    # ----- LEGACY-COMPATIBLE photosphere properties -----
    T_phot = float(snap_full['T'][idx_cont])
    if 'L' in snap_full and idx_cont < n_zones_full:
        L_phot = float(np.abs(snap_full['L'][idx_cont]))
    else:
        SIGMA_SB = 5.6704e-5
        L_phot = 4.0 * np.pi * r_phot_used * r_phot_used * SIGMA_SB * T_phot**4

    snap_out['R_phot_inner']         = r_phot_used      # legacy
    snap_out['T_phot_inner']         = T_phot           # legacy
    snap_out['L_phot_inner']         = L_phot           # legacy
    snap_out['n_zones_full']         = n_zones_full     # legacy
    snap_out['r_phot_used']          = r_phot_used      # legacy
    snap_out['tau_es_threshold_used'] = tau_target      # legacy (semantic change: now τ_cont)

    # Sometimes downstream uses the bare names R_phot, T_phot, L_phot
    snap_out['R_phot'] = r_phot_used
    snap_out['T_phot'] = T_phot
    snap_out['L_phot'] = L_phot

    # ----- LEGACY T_color (τ_es=10 heuristic) for backward compatibility -----
    tau_full = np.asarray(snap_full['tau_es'], dtype=float)
    T_full   = np.asarray(snap_full['T'], dtype=float)
    T_color_at_tau = {}
    for tau_target_dict in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        idx = int(np.argmin(np.abs(tau_full - tau_target_dict)))
        T_color_at_tau[tau_target_dict] = float(T_full[idx])
    snap_out['T_color_at_tau_es']      = T_color_at_tau                # legacy dict
    snap_out['T_color_thermalization'] = T_color_at_tau[t_color_legacy_tau_es]  # legacy default

    # ----- NEW Phase 1 diagnostic keys (non-colliding names) -----
    snap_out['R_phot_cont']      = phot['R_phot_cont']
    snap_out['T_phot_cont']      = phot['T_phot_cont']
    snap_out['idx_phot_cont']    = phot['idx_phot_cont']
    snap_out['R_phot_es']        = phot['R_phot_es']
    snap_out['idx_phot_es']      = phot['idx_phot_es']
    snap_out['R_therm']          = phot['R_therm']
    snap_out['T_color_hummer']   = phot['T_color']           # Hummer τ_eff=1 definition
    snap_out['photosphere_kappa_breakdown'] = phot['channel_at_phot']
    snap_out['photosphere_lam_ref_AA'] = lam_ref_AA

    if verbose:
        c = phot['channel_at_phot']
        dom_ch = max(c.items(), key=lambda kv: kv[1] if kv[0] != 'total' else -1)
        print(f"[truncate_at_cont_photosphere] {n_zones_full} → {n_keep} zones "
              f"@ λ_ref = {lam_ref_AA:.0f} Å")
        print(f"  τ_cont = {tau_target} at zone {idx_cont}: "
              f"r = {r_phot_used:.3e} cm, T = {T_phot:.0f} K")
        print(f"  Compared to legacy τ_es = 2/3: "
              f"r = {phot['R_phot_es']:.3e} cm "
              f"({(r_phot_used/phot['R_phot_es']-1)*100:+.1f}% shift)")
        print(f"  Photosphere opacity: {dom_ch[0]} dominates "
              f"({dom_ch[1]/c['total']*100:.0f}% of κ_total)")
        print(f"  L_phot = {L_phot:.3e} erg/s, T_color (legacy τ_es=10) = "
              f"{snap_out['T_color_thermalization']:.0f} K, "
              f"T_color (Hummer τ_eff=1) = {snap_out['T_color_hummer']:.0f} K")

    return snap_out
