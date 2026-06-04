"""nlte_he2.py — Phase 4 NLTE solver for He II level populations.

Computes per-zone He II level populations (n = 1..10) by solving statistical
equilibrium with Sobolev escape probability for radiative line transfer,
analogous to nlte_he1.py but simpler (single spin system, hydrogenic).

The chemistry chain is:
    He I  (neutral)  ←→  He II (singly ionized)  ←→  He III (fully ionized)
    Phase 3              ↑ this module             ↑ computed from local
                         level populations         photoionization balance
                         here                      or external override

Public API
----------
he2_populations_nlte(rho, T, n_e, r, v, X_He, X_HII_external=None, ...)
    → (n_lev [NLEV × nzones], diagnostics_dict)

Inputs are the same arrays Phase 3 takes; X_HII_external tells us how much
of the hydrogen is ionized (used to estimate He III fraction via Saha, since
the He²⁺/He⁺ ratio depends on the same physics that sets H II/H I).

Phase 5 will consume the output `n_lev` to add He II line opacities (4686,
1640, 3203, 10124) to the multi-line MC.
"""
import numpy as np
import he2_atom as he2

# Physical constants
H_PLANCK = 6.62607015e-27
C_LIGHT  = 2.99792458e10
KB       = 1.380649e-16
ME       = 9.1093837015e-28
EV_TO_ERG = 1.602176634e-12


# -----------------------------------------------------------------------------
def _saha_he2_to_he3(T, n_e):
    """Compute n(He III) / n(He II) by local Saha balance.

    Returns the per-zone ratio R = n_HeIII / n_HeII assuming:
       - Standard LTE Saha equation
       - Ionization potential χ = 54.4228 eV (He II → He III)
       - g(He III) = 1, g(He II ground) = 2

    The ratio is bounded above to avoid overflow in deeply ionized zones.
    """
    T = np.asarray(T, dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    chi_eV = 54.4228
    chi_erg = chi_eV * EV_TO_ERG
    therm = (H_PLANCK * H_PLANCK / (2.0 * np.pi * ME * KB
                                      * np.maximum(T, 100.0))) ** 1.5
    # Saha: n_e × n(HeIII) / n(HeII) = (2 g_HeIII / g_HeII) × (1/therm) × exp(-χ/kT)
    g_ratio = 2.0 * 1.0 / 2.0    # = 1
    saha_ratio = (g_ratio / therm) * np.exp(-chi_erg
                                               / (KB * np.maximum(T, 100.0)))
    R = saha_ratio / np.maximum(n_e, 1.0)
    return np.clip(R, 0.0, 1.0e6)


# -----------------------------------------------------------------------------
def _sobolev_beta(n_l, n_u, g_l, g_u, A, lam_AA, dv_dr_eff):
    """Sobolev escape probability β = (1 − exp(−τ)) / τ.

    Identical structure to the He I solver. τ is computed with a 1 − LTE
    correction so that thermalized populations give τ ≈ 0 (no net trapping).
    """
    lam_cm = lam_AA * 1.0e-8
    # Population ratio factor (1 − g_l n_u / (g_u n_l))
    n_l_safe = np.maximum(n_l, 1.0e-30)
    pop_corr = 1.0 - (g_l * n_u) / (g_u * n_l_safe)
    pop_corr = np.clip(pop_corr, 0.0, 1.0)
    # Sobolev optical depth
    tau = (A * (lam_cm ** 3) / (8.0 * np.pi)) * (g_u / g_l) * n_l_safe \
            * pop_corr / np.maximum(dv_dr_eff, 1.0e-30)
    tau = np.maximum(tau, 1.0e-30)
    # β = (1 − e⁻ᵗ) / τ, with stable small-τ limit
    beta = np.where(tau < 1.0e-6, 1.0 - 0.5 * tau,
                    (1.0 - np.exp(-tau)) / tau)
    return beta, tau


# -----------------------------------------------------------------------------
def he2_populations_nlte(rho, T, n_e, r, v,
                          X_He=0.245,
                          X_HII_external=None,
                          X_HeIII_external=None,
                          v_turb_cms=20e5,
                          max_iter=120, tol=1e-3, damping=0.4,
                          ionization_split_mode='saha_local',
                          verbose=False):
    """Solve NLTE statistical equilibrium for He II level populations.

    Parameters
    ----------
    rho, T, n_e, r, v : arrays of length nzones
        Hydro/thermal state. T in K, n_e in cm⁻³, r in cm, v in cm/s.
    X_He : scalar or array
        He mass fraction. Defaults to 0.245 (Big-Bang abundance).
    X_HII_external : array, optional
        Per-zone H ionization fraction. Used for the local Saha estimate of
        He III/He II when ionization_split_mode='saha_local'. If None,
        defaults to a uniform value of 0.05.
    X_HeIII_external : array, optional
        Per-zone X_HeIII override. If provided, used directly and the local
        Saha is skipped. Useful for testing with prescribed values.
    ionization_split_mode : str
        'saha_local'   — compute X_HeIII from local Saha at the gas T, n_e
                          (default; correct for collisional ionization)
        'photoeq_match' — compute X_HeIII to match a hypothetical photon
                          flux producing the observed X_HII (uses
                          n_HeIII/n_HeII proportional to (n_HII/n_HI)²
                          since χ_HeII/χ_HI = 4)
        'zero'         — assume no He III (cool plateau approximation)
    max_iter, tol, damping : NLTE iteration controls
        Same convention as nlte_he1.py.

    Returns
    -------
    n_lev : ndarray (NLEV, nzones)
        He II level populations [cm⁻³]. n_lev[0] is ground (n=1).
    diag : dict
        Diagnostic information: X_HeII per zone, X_HeIII per zone, β and τ
        per transition, iteration count, named-line τ values.
    """
    rho = np.asarray(rho, dtype=float)
    T   = np.asarray(T,   dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    r   = np.asarray(r,   dtype=float)
    v   = np.asarray(v,   dtype=float)
    nzones = len(r)

    if np.isscalar(X_He):
        X_He = np.full(nzones, float(X_He))
    else:
        X_He = np.asarray(X_He, dtype=float)
        if X_He.size != nzones:
            X_He = np.full(nzones, float(np.mean(X_He)))

    m_He = 4.002602 * 1.66053906660e-24
    n_He_total = X_He * rho / m_He        # per-zone total He nuclei

    # ---- Step 1: determine X_HeIII per zone ----
    if X_HeIII_external is not None:
        X_HeIII = np.clip(np.asarray(X_HeIII_external, dtype=float), 0.0, 1.0)
        if X_HeIII.size != nzones:
            X_HeIII = np.full(nzones, float(np.mean(X_HeIII)))
    elif ionization_split_mode == 'zero':
        X_HeIII = np.zeros(nzones)
    elif ionization_split_mode == 'saha_local':
        # Need a first estimate of X_HeII first. Start with "all He is He II
        # except what Saha says is He III" and iterate one step.
        R_he32 = _saha_he2_to_he3(T, n_e)    # n_HeIII / n_HeII
        # Assume He I is negligible (Phase 4 handles He II↔He III chemistry;
        # the He I fraction comes from Phase 3's separate calculation).
        # Self-consistency between Phase 3 and Phase 4 is the next layer.
        X_HeIII = R_he32 / (1.0 + R_he32)
    elif ionization_split_mode == 'photoeq_match':
        # If H II / H I = X / (1-X), and χ(He II) = 4 × χ(H I), then by
        # the same radiation field the He III / He II ratio is
        # roughly proportional to (X_HII/(1-X_HII))² in the optically thin
        # limit (since photon supply at hν > 54.4 eV scales as the SED
        # squared compared to hν > 13.6 eV). This is a crude estimator
        # — Phase 5 may want to refine.
        if X_HII_external is None:
            X_HII_external = np.full(nzones, 0.05)
        X_HII = np.clip(np.asarray(X_HII_external, dtype=float), 1e-6, 0.999999)
        r_HII = X_HII / (1.0 - X_HII)
        # Scale by 1e-3 to reflect the harder UV requirement (rough)
        r_HeIII = 1.0e-3 * r_HII ** 2
        X_HeIII = r_HeIII / (1.0 + r_HeIII)
    else:
        raise ValueError(f"Unknown ionization_split_mode: {ionization_split_mode}")

    X_HeIII = np.clip(X_HeIII, 0.0, 0.999999)
    # Set X_HeII assuming He I fraction is determined separately (Phase 3).
    # Here, Phase 4 owns the He II / He III split; total He I is provided
    # via X_HeII_total being implicitly = 1 − X_HeIII − X_HeI. To keep
    # Phase 4 self-contained, default to X_HeI = 0 (all He is in II or III).
    # This is the standard "Phase 4 reads He II + III; Phase 3 reads He I
    # + II" convention; the two phases share He II as the coupling layer.
    X_HeII = 1.0 - X_HeIII
    n_HeII  = X_HeII  * n_He_total          # total He II nuclei per zone
    n_HeIII = X_HeIII * n_He_total

    # ---- Step 2: initial populations from LTE Saha-Boltzmann ----
    G = he2.G_LEV
    E = he2.E_LEV * EV_TO_ERG               # erg
    NLEV = he2.NLEV

    # Boltzmann distribution among He II levels, normalized to sum = n_HeII
    n_i = np.zeros((NLEV, nzones))
    for i in range(NLEV):
        n_i[i] = G[i] * np.exp(-E[i] / (KB * np.maximum(T, 100.0)))
    partition = n_i.sum(axis=0)
    for i in range(NLEV):
        n_i[i] = n_i[i] * n_HeII / np.maximum(partition, 1.0e-30)

    # ---- Step 3: velocity gradient for Sobolev ----
    dv_dr = np.gradient(v, r)
    dv_dr_eff = np.maximum(np.abs(dv_dr), v_turb_cms / np.maximum(r, 1.0e-30))

    # ---- Step 4: iteration loop ----
    transitions = he2.TRANSITIONS

    # Precompute collisional rates (T-dependent, populations-independent)
    C_exc = {}
    C_dex = {}
    for (i_l, i_u), (lam, f, A) in transitions.items():
        C_exc[(i_l, i_u)] = he2.collisional_rate_lu(i_l + 1, i_u + 1, T)
        C_dex[(i_l, i_u)] = he2.collisional_rate_ul(i_l + 1, i_u + 1, T)

    # Source term: recombination from He III into each level
    alpha_eff_arr = he2.alpha_eff(T)         # shape (NLEV, nzones) if T is array
    if np.ndim(T) == 0 or alpha_eff_arr.ndim == 1:
        S_lev = (alpha_eff_arr[:, None]
                 * n_e[None, :] * n_HeIII[None, :])
    else:
        S_lev = alpha_eff_arr * n_e[None, :] * n_HeIII[None, :]

    beta = {}
    tau  = {}

    converged_at = max_iter
    for it in range(max_iter):
        # Sobolev β per transition (uses current populations)
        for (i_l, i_u), (lam, f, A) in transitions.items():
            b, t = _sobolev_beta(n_i[i_l], n_i[i_u],
                                  G[i_l], G[i_u], A, lam, dv_dr_eff)
            beta[(i_l, i_u)] = b
            tau[(i_l, i_u)]  = t

        new_n_i = np.zeros_like(n_i)
        for z in range(nzones):
            M = np.zeros((NLEV, NLEV))
            # E1 radiative + collisional rates
            for (i_l, i_u), (lam, f, A) in transitions.items():
                R_lu = C_exc[(i_l, i_u)][z] * n_e[z]
                R_ul = A * beta[(i_l, i_u)][z] + C_dex[(i_l, i_u)][z] * n_e[z]
                M[i_l, i_l] -= R_lu
                M[i_u, i_l] += R_lu
                M[i_u, i_u] -= R_ul
                M[i_l, i_u] += R_ul

            # Conservation row: replace ground state's rate equation
            # (largest-population level — same trick as nlte_he1).
            M[0, :] = 1.0
            rhs = -S_lev[:, z].copy()
            rhs[0] = n_HeII[z]

            # Solve
            try:
                sol = np.linalg.solve(M, rhs)
                sol = np.maximum(sol, 0.0)
                new_n_i[:, z] = sol
            except np.linalg.LinAlgError:
                new_n_i[:, z] = n_i[:, z]    # keep previous if singular

        # Damping + convergence check
        delta_max = 0.0
        for i in range(NLEV):
            mask = new_n_i[i] > 0
            if mask.any():
                rel = np.abs(new_n_i[i, mask] - n_i[i, mask]) / np.maximum(
                    n_i[i, mask], 1.0e-30)
                if rel.size:
                    delta_max = max(delta_max, float(np.max(rel)))
        n_i = (1.0 - damping) * n_i + damping * new_n_i

        if delta_max < tol:
            converged_at = it + 1
            break

    # ---- Step 5: final β/τ, named lines ----
    beta_final = {}
    tau_final = {}
    for (i_l, i_u), (lam, f, A) in transitions.items():
        b, t = _sobolev_beta(n_i[i_l], n_i[i_u],
                              G[i_l], G[i_u], A, lam, dv_dr_eff)
        beta_final[(i_l, i_u)] = b
        tau_final[(i_l, i_u)]  = t

    named_tau = {name: tau_final[idx] for name, idx in he2.NAMED_LINES.items()}

    diag = {
        'iterations':       converged_at,
        'X_HeII':           X_HeII,
        'X_HeIII':          X_HeIII,
        'n_HeII_total':     n_HeII,
        'n_HeIII':          n_HeIII,
        'beta':             beta_final,
        'tau':              tau_final,
        'named_tau':        named_tau,
        'alpha_eff_n2_at_T': alpha_eff_arr[1] if alpha_eff_arr.ndim > 1
                              else alpha_eff_arr[1],
    }
    return n_i, diag


# -----------------------------------------------------------------------------
def line_optical_depth(n_lev, line_name, T, dv_dr_eff):
    """Standalone helper: compute τ_Sobolev for a named He II line.

    Parameters
    ----------
    n_lev : ndarray (NLEV, nzones)
        Output of he2_populations_nlte.
    line_name : str
        Key into he2_atom.NAMED_LINES (e.g. 'He_II_4686').
    T : array
        Gas temperature per zone [K] (unused in this function but kept
        for parallel structure with He I helper).
    dv_dr_eff : array
        Effective |dv/dr| including v_turb floor [s⁻¹].

    Returns
    -------
    tau : array, per-zone Sobolev optical depth for the requested line.
    """
    i_l, i_u = he2.NAMED_LINES[line_name]
    lam, f, A = he2.TRANSITIONS[(i_l, i_u)]
    G = he2.G_LEV
    _, tau = _sobolev_beta(n_lev[i_l], n_lev[i_u],
                            G[i_l], G[i_u], A, lam, dv_dr_eff)
    return tau
