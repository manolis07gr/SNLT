"""nlte_he1.py — He I NLTE level populations with Sobolev escape probabilities.

Solves multi-level rate equations for the 11-level He I atom defined in
`he1_atom.py`, producing per-zone populations n_HeI(level) suitable for
computing line opacities and emissivities at He I 5876, 6678, 7065, 10830 Å.

Public API
----------
  he1_populations_nlte(rho, T, n_e, r, v, ...) → (n_levels, diag)
      Returns per-zone level populations [cm⁻³] and diagnostics. Structurally
      parallel to `h_populations_nlte.h_populations_nlte`.

Physics
-------
  * 11-level structure: 1¹S, 2³S, 2¹S, 2³P, 2¹P, 3³S, 3¹S, 3³P, 3¹P, 3³D, 3¹D
  * Spontaneous emission with Sobolev escape probability β_(l,u) per line
  * Collisional excitation/de-excitation (Bray+2000 effective collision strengths)
  * Source terms from continuum:
      - Case-B recombination into each level (Porter+2012 α_eff)
      - Optional Case-A: direct photoionization from each level
  * Metastable level treatment:
      - 2³S → 1¹S: M1 spin-forbidden, A = 1.27e-4 s⁻¹ (effectively infinite lifetime)
      - 2¹S → 1¹S: 2-photon decay, A = 51.3 s⁻¹ (genuine destruction channel)
  * He I 2¹P → 1¹S (584 Å resonance) trapping via Sobolev β_(0,4)
  * He I 1¹S → 3¹P (537 Å) similarly trapped

Assumptions / simplifications
-----------------------------
  * He ionization split (HeI/HeII/HeIII) supplied externally (e.g. as X_HeII
    per-zone, defaulting to X_HeII = X_HII as first approximation if not given).
    Phase 4 (He II NLTE) will refine this.
  * No charge-exchange with H (only matters for very specific regimes).
  * Photoionization rate not yet coupled to the J_bar field; uses local LTE
    radiation field at the photosphere as in h_populations_nlte.
"""
import numpy as np
import he1_atom as he

KB      = 1.380649e-16
HPL     = 6.62607015e-27
C_LIGHT = 2.99792458e10
ME      = 9.1093837015e-28
EE      = 4.8032068e-10
EV      = 1.602176634e-12
MHE     = 6.6464764e-24       # He atomic mass


# ---------- Helper: estimate He ionization split ----------
def estimate_he_ionization(T, n_e, X_HII_external=None, mode='follow_H'):
    """Return (X_HeI, X_HeII, X_HeIII) as per-zone fractions of total He.

    mode = 'follow_H':
        X_HeII = X_HII (the simplest approximation; valid when H and He
                         have similar ionization conditions, often OK in
                         moderately ionized CSM)
        X_HeIII ≈ 0  (very hot regions only)
        X_HeI = 1 - X_HeII
    mode = 'saha_approx':
        Saha balance using local T, n_e for HeI ↔ HeII (no charge exchange).
        Overestimates HeII at low T.

    For Phase 4, this gets replaced by full HeII NLTE coupling.
    """
    T   = np.asarray(T, dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    if mode == 'follow_H' and X_HII_external is not None:
        X_HeII = np.clip(np.asarray(X_HII_external, dtype=float), 0.0, 1.0)
        X_HeIII = np.zeros_like(X_HeII)
        X_HeI = 1.0 - X_HeII - X_HeIII
        return X_HeI, X_HeII, X_HeIII
    # Saha He I ↔ He II
    chi = he.CHI_HeI
    g_HeII = 2.0; g_HeI = 1.0; g_e = 2.0
    thermal_vol = (HPL * HPL / (2 * np.pi * ME * KB * T)) ** 1.5
    R_ion = (g_HeII * g_e / g_HeI) * (1.0 / thermal_vol) * \
            np.exp(-chi / (KB * T)) / np.maximum(n_e, 1.0)
    X_HeII = R_ion / (1.0 + R_ion)
    X_HeIII = np.zeros_like(X_HeII)   # ignore at low T
    X_HeI = 1.0 - X_HeII - X_HeIII
    return X_HeI, X_HeII, X_HeIII


# ---------- Sobolev optical depth per transition ----------
def _sobolev_beta(n_l, n_u, g_l, g_u, A_ul, lam_AA, dv_dr):
    """Sobolev escape probability for one transition, per zone.

    τ_Sob = (c/4π) × (λ A_ul / dv/dr) × n_l (g_u/g_l) (1 − n_u g_l/(n_l g_u))
    β = (1 - exp(-τ)) / τ

    For broadcasting: all inputs zone-arrays, dv_dr zone-array, A, lam scalars.
    """
    n_l = np.asarray(n_l)
    n_u = np.asarray(n_u)
    dv_dr = np.asarray(dv_dr)
    lam_cm = lam_AA * 1e-8
    # The per-photon line cross-section integrated over freq, then converted
    # to Sobolev τ via population difference and v gradient:
    # τ = (h ν c/ 4π × dv/dr)^-1 × n_l × σ_lu × c   (standard Sobolev form)
    # Practical form: τ = sigma_int × (n_l - g_l/g_u × n_u) / dv_dr
    # where sigma_int has units of [cm² × Hz × velocity] / [s] etc.
    # Use the equivalent:
    #   τ = (A_ul λ³ / 8π) × (g_u/g_l) × n_l × (1 - LTE_ratio) / dv/dr
    pop_factor = np.maximum(n_l - (g_l / g_u) * n_u, 0.0)
    tau = (A_ul * lam_cm**3 / (8.0 * np.pi)) * (g_u / g_l) * \
          pop_factor / np.maximum(dv_dr, 1e-30)
    # Clip τ to avoid overflow in exp
    tau_safe = np.minimum(tau, 700.0)
    beta = np.where(tau > 1e-12,
                     (1.0 - np.exp(-tau_safe)) / np.maximum(tau, 1e-30),
                     1.0 - 0.5*tau)    # series expansion for small τ
    return beta, tau


# ---------- Main solver ----------
def he1_populations_nlte(rho, T, n_e, r, v,
                          X_He=0.245,            # mass fraction of He
                          X_HII_external=None,    # for HeII estimation
                          v_turb_cms=20e5,
                          max_iter=120, tol=1e-3, damping=0.4,
                          ionization_split_mode='follow_H',
                          two_photon_decay=True,
                          # Lyα analog: 2¹P → 1¹S at 584 Å, trapped.
                          # In the slow CSM wind this is heavily trapped just
                          # like H Lyα. We provide an ε floor analogous to
                          # Phase 2 for H. Default uses 2γ from 2¹S only
                          # (which is automatically present when
                          # two_photon_decay=True).
                          eps_He_resonance=None,
                          # Whether to include the spin-forbidden 2³S → 1¹S
                          # M1 decay (A = 1.27e-4 s⁻¹). Almost always
                          # negligible vs collisional and photoionization
                          # destruction of 2³S in real plasmas.
                          include_M1_decay=True,
                          verbose=False):
    """Multi-level He I NLTE populations.

    Returns
    -------
    n_levels : array, shape (NLEV=11, nzones), per-zone population [cm⁻³]
                in each He I level. Order matches he1_atom.LEVELS.
    diag : dict with keys:
        n_HeI_total, n_HeII, n_HeIII : per-zone densities [cm⁻³]
        beta : dict (l,u) → per-zone β
        tau : dict (l,u) → per-zone Sobolev τ
        2_3S_population : convenience reference (n_levels[1])
        iterations : number of solver iterations used
    """
    # Per-zone arrays
    T   = np.asarray(T, dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    rho = np.asarray(rho, dtype=float)
    r   = np.asarray(r, dtype=float)
    v   = np.asarray(v, dtype=float)
    nzones = T.size

    # Total helium density
    n_He_total = X_He * rho / MHE

    # Ionization split
    X_HeI, X_HeII, X_HeIII = estimate_he_ionization(
        T, n_e, X_HII_external=X_HII_external,
        mode=ionization_split_mode)
    n_HeI_total = X_HeI * n_He_total
    n_HeII = X_HeII * n_He_total
    n_HeIII = X_HeIII * n_He_total

    # Velocity gradient (forward diff with edge handling)
    dv_dr = np.zeros_like(r)
    dv_dr[1:-1] = (v[2:] - v[:-2]) / (r[2:] - r[:-2])
    dv_dr[0]    = (v[1] - v[0]) / (r[1] - r[0])
    dv_dr[-1]   = (v[-1] - v[-2]) / (r[-1] - r[-2])
    # Add turbulent broadening floor
    dv_dr_eff = np.maximum(np.abs(dv_dr), v_turb_cms / np.maximum(r, 1e-30))

    # ---- Atomic data shortcuts ----
    NLEV = he.NLEV
    G = he.G_LEV
    E = he.E_LEV
    transitions = he.TRANSITIONS

    # ---- Initialize populations to LTE Boltzmann within He I ----
    Z_HeI = np.zeros(nzones)
    n_i = np.zeros((NLEV, nzones))
    for k in range(NLEV):
        w_k = G[k] * np.exp(-E[k] / (KB * T))
        n_i[k] = w_k
        Z_HeI += w_k
    Z_HeI = np.maximum(Z_HeI, 1e-30)
    for k in range(NLEV):
        n_i[k] *= n_HeI_total / Z_HeI

    # ---- Collisional rate coefficients per transition (precomputed) ----
    C_exc = {}; C_dex = {}
    for (l, u) in transitions:
        C_exc[(l, u)] = he.collisional_excitation_HeI(l, u, T)
        C_dex[(l, u)] = he.collisional_deexcitation_HeI(u, l, T)

    # Also for the "dark" forbidden 2³S → 1¹S transition (collisional only;
    # M1 photon channel is so slow it's effectively absent)
    C_exc[(0, 1)] = he.collisional_excitation_HeI(0, 1, T)
    C_dex[(0, 1)] = he.collisional_deexcitation_HeI(1, 0, T)
    # 2¹S → 1¹S collisional
    C_exc[(0, 2)] = he.collisional_excitation_HeI(0, 2, T)
    C_dex[(0, 2)] = he.collisional_deexcitation_HeI(2, 0, T)

    # ---- Source terms from continuum (Case-B recombination) ----
    # Source S_i = α_eff_i(T) × n_e × n_HeII   [cm⁻³ s⁻¹]
    S = np.zeros((NLEV, nzones))
    for k in range(1, NLEV):   # skip ground (Case B)
        alpha_eff = he.alpha_eff_HeI(k, T)
        S[k] = alpha_eff * n_e * n_HeII

    # ---- Iteration loop ----
    A_2_3S_g = he.A_2_3S_TO_GROUND if include_M1_decay else 0.0
    A_2_1S_g_2gamma = he.A_2_1S_TO_GROUND_2GAMMA

    last_n = n_i.copy()
    for it in range(max_iter):
        # 1) compute Sobolev β for each E1 transition
        beta = {}
        tau  = {}
        for (l, u), (lam, f, A) in transitions.items():
            b, t = _sobolev_beta(n_i[l], n_i[u], G[l], G[u], A, lam, dv_dr_eff)
            beta[(l, u)] = b
            tau[(l, u)] = t

        # Apply ε floor for He I 2¹P → 1¹S resonance (584 Å) if requested.
        # Strong analog of H Lyα: this transition is heavily trapped in the
        # slow CSM and the local Sobolev β can underestimate destruction by
        # metal-line / continuum opacity in the FUV.
        if eps_He_resonance is not None and (0, 4) in beta:
            beta[(0, 4)] = np.maximum(beta[(0, 4)], float(eps_He_resonance))
        # Apply ε floor to 1¹S → 3¹P (537 Å) similarly if any later need
        if eps_He_resonance is not None and (0, 8) in beta:
            beta[(0, 8)] = np.maximum(beta[(0, 8)], float(eps_He_resonance))

        # 2) build rate matrix per zone and solve
        new_n_i = np.zeros_like(n_i)
        for z in range(nzones):
            M = np.zeros((NLEV, NLEV))
            # Add E1 transitions
            for (l, u), (lam, f, A) in transitions.items():
                # excitation l → u: collisional only at moderate density
                R_lu = C_exc[(l, u)][z] * n_e[z]
                # de-excitation u → l: A × β_escape + collisional
                R_ul = A * beta[(l, u)][z] + C_dex[(l, u)][z] * n_e[z]
                M[l, l] -= R_lu
                M[u, l] += R_lu
                M[u, u] -= R_ul
                M[l, u] += R_ul

            # Add metastable channels (NOT in TRANSITIONS dict since they're
            # not E1-like):
            # 2¹S → 1¹S two-photon decay (genuine destruction)
            if two_photon_decay:
                R_2_1S_to_g = A_2_1S_g_2gamma
                M[2, 2] -= R_2_1S_to_g
                M[0, 2] += R_2_1S_to_g

            # 2³S → 1¹S M1 (very slow but included for completeness)
            if include_M1_decay and A_2_3S_g > 0:
                R_2_3S_to_g = A_2_3S_g
                M[1, 1] -= R_2_3S_to_g
                M[0, 1] += R_2_3S_to_g

            # Collisional excitation from ground to 2³S and 2¹S
            # (these aren't in TRANSITIONS but are real)
            for (l, u) in [(0, 1), (0, 2)]:
                R_lu = C_exc[(l, u)][z] * n_e[z]
                R_ul = C_dex[(l, u)][z] * n_e[z]
                M[l, l] -= R_lu
                M[u, l] += R_lu
                M[u, u] -= R_ul
                M[l, u] += R_ul

            # Replace the rate equation of the most-populated level (ground
            # state, 1¹S, level 0) with the conservation equation
            #     Σ_i n_i = n_HeI_total
            # This is standard NLTE practice: the rate equation for the
            # largest-population level is the least-well-conditioned (its
            # diagonal element is dominated by tiny upward rates), and
            # replacing it with the exact constraint stabilizes the linear
            # solve. Without this substitution the matrix has κ ~ 10⁸ and
            # the singlet populations come out negative.
            M[0, :] = 1.0
            rhs = -S[:, z].copy()
            rhs[0] = n_HeI_total[z]

            try:
                sol = np.linalg.solve(M, rhs)
                sol = np.maximum(sol, 0.0)
                new_n_i[:, z] = sol
            except np.linalg.LinAlgError:
                # Fall back to LTE for singular systems (very low-density edge)
                w = G * np.exp(-E / (KB * T[z]))
                Zk = w.sum()
                new_n_i[:, z] = (n_HeI_total[z] / max(Zk, 1e-30)) * w

        # Damped update
        n_i = damping * new_n_i + (1.0 - damping) * n_i

        # Convergence check on the metastable 2³S level (the slowest-evolving)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.abs(n_i - last_n) / np.maximum(np.abs(last_n), 1e-30)
            rel_2_3S = np.nanmax(rel[1])
        if verbose and it % 10 == 0:
            print(f"  [nlte_he1] iter {it:3d}: max Δn_2³S/n = {rel_2_3S:.2e}, "
                  f"median n_2³S = {np.median(n_i[1]):.3e} cm⁻³")
        if rel_2_3S < tol and it > 5:
            break
        last_n = n_i.copy()

    # Final β values (with the same ε floor applied as inside the loop, so
    # the diagnostics returned reflect what was actually used in the solve)
    beta_final = {}
    tau_final = {}
    for (l, u), (lam, f, A) in transitions.items():
        b, t = _sobolev_beta(n_i[l], n_i[u], G[l], G[u], A, lam, dv_dr_eff)
        beta_final[(l, u)] = b
        tau_final[(l, u)] = t
    if eps_He_resonance is not None and (0, 4) in beta_final:
        beta_final[(0, 4)] = np.maximum(beta_final[(0, 4)],
                                         float(eps_He_resonance))
    if eps_He_resonance is not None and (0, 8) in beta_final:
        beta_final[(0, 8)] = np.maximum(beta_final[(0, 8)],
                                         float(eps_He_resonance))

    diag = {
        'n_HeI_total': n_HeI_total,
        'n_HeII': n_HeII,
        'n_HeIII': n_HeIII,
        'n_He_total': n_He_total,
        'X_HeI': X_HeI,
        'X_HeII': X_HeII,
        'beta': beta_final,
        'tau': tau_final,
        'iterations': it + 1,
        'metastable_2_3S': n_i[1],
        'metastable_2_1S': n_i[2],
        'two_photon_decay_active': two_photon_decay,
    }
    return n_i, diag


# ---------- Convenience: compute line-specific Sobolev τ ----------
def line_optical_depth(n_levels, line_name, T, dv_dr_eff):
    """Sobolev optical depth at a given He I line, per zone."""
    info = he.line_info(line_name)
    l = info['l']; u = info['u']
    A = info['A_ul']; lam = info['lam_AA']
    g_l = info['g_l']; g_u = info['g_u']
    n_l = n_levels[l]; n_u = n_levels[u]
    _, tau = _sobolev_beta(n_l, n_u, g_l, g_u, A, lam, dv_dr_eff)
    return tau


def line_emissivity(n_levels, line_name):
    """Line emissivity j_ul = (h ν / 4π) × n_u × A_ul × β [erg/s/cm³/sr].

    Optically-thin limit (β = 1) gives the canonical luminosity per volume.
    For Sobolev, multiply by β separately."""
    info = he.line_info(line_name)
    u = info['u']
    A = info['A_ul']; lam = info['lam_AA']
    h_nu = HPL * C_LIGHT / (lam * 1e-8)
    return h_nu * n_levels[u] * A / (4.0 * np.pi)
