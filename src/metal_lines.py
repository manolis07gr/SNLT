"""
metal_lines.py — C/O/Ne metal-line spectra (Stage 3 driver)
================================================================================

FUTURE_WORK P2 item 5. Produces per-line spectra entries (L_line, F_λ, F_norm,
EW, L_cont_band, …) for the C/O/Ne metal lines, using:
  • metal_ionization.compute_metal_ionization  → per-zone ion densities,
  • metal_atoms.line_emissivity                → recombination / CEL emissivity,
  • phase5_continuum.build_per_band_continuum   → per-band continuum (composition-
                                                  general BB; works at any λ),
  • an OPTICALLY-THIN expanding-envelope profile (boxcar-superposition).

The metal lines are optically thin (forbidden/intercombination CEL + permitted
recombination in a transparent CSM), so there is NO Sobolev/MC transport: the
emergent profile of a thin spherical shell expanding at v_z is a boxcar in
line-of-sight velocity (±v_z, flat), and the total profile is the
emissivity-weighted superposition over zones. L_line is the exact volume integral
of the emissivity.

The returned dict matches the He/H spectra contract (keys lambda_AA, F_lambda,
F_cont_lambda, F_norm, L_line, L_cont_band, EW, tau_med, lambda_rest, plus the
*_corrected mirrors and strength_mode), so phase5_runner can merge it straight
into the spectra dict and every downstream consumer (npz, regime, plots, movies)
picks the metal lines up dynamically.
"""
from __future__ import annotations
import numpy as np

import metal_atoms as ma
import metal_ionization as mi
import metal_nlte as mnlte

C_KMS = 2.99792458e5


def _shell_volumes(r):
    r = np.asarray(r, float)
    edge = np.empty(len(r) + 1)
    edge[1:-1] = 0.5 * (r[:-1] + r[1:])
    edge[0] = r[0]
    edge[-1] = r[-1] + (r[-1] - r[-2])
    return (4.0 / 3.0) * np.pi * (edge[1:] ** 3 - edge[:-1] ** 3)


def _trapz(y, x):
    return (np.trapezoid(y, x) if hasattr(np, 'trapezoid') else np.trapz(y, x))


def _gaussian_smooth(lam_grid, y, lam0_AA, sigma_kms):
    """Convolve a profile with a Gaussian of velocity width sigma_kms. Removes
    the discrete-shell boxcar staircase (thermal + turbulent broadening; for
    heavy metal ions the thermal width is small, so this is mostly a numerical
    de-stairstep). Conserves the integral."""
    lam_grid = np.asarray(lam_grid, float)
    if sigma_kms <= 0 or lam_grid.size < 3:
        return y
    sigma_lam = float(lam0_AA) * float(sigma_kms) * 1e5 / (C_KMS * 1e5)  # AA
    dlam = float(np.mean(np.diff(lam_grid)))
    if dlam <= 0 or sigma_lam <= 0:
        return y
    half = int(max(3, np.ceil(4.0 * sigma_lam / dlam)))
    kx = np.arange(-half, half + 1) * dlam
    g = np.exp(-0.5 * (kx / sigma_lam) ** 2)
    s = g.sum()
    if s <= 0:
        return y
    g = g / s
    return np.convolve(np.asarray(y, float), g, mode='same')


def thin_emission_profile(lam_grid, lam0_AA, v_cms, weights):
    """Optically-thin expanding-envelope line profile on `lam_grid`.

    Each zone (velocity v_z, weight w_z = emissivity·ΔV) contributes a boxcar of
    half-width v_z in line-of-sight velocity (isotropic emission Doppler-spread
    uniformly over [−v_z, +v_z]). Returns P(λ) normalised to unit integral
    (∫P dλ = 1), or a delta-like spike at line centre if there is no velocity.
    """
    lam_grid = np.asarray(lam_grid, float)
    vgrid_kms = (lam_grid / float(lam0_AA) - 1.0) * C_KMS
    vz_kms = np.abs(np.asarray(v_cms, float)) / 1e5
    w = np.asarray(weights, float)
    P = np.zeros_like(lam_grid)
    for z in range(len(w)):
        if w[z] <= 0.0:
            continue
        if vz_kms[z] <= 0.0:
            # no velocity → put weight in the central pixel
            P[np.argmin(np.abs(vgrid_kms))] += w[z]
            continue
        mask = np.abs(vgrid_kms) <= vz_kms[z]
        if mask.any():
            P[mask] += w[z] / (2.0 * vz_kms[z])
        else:
            P[np.argmin(np.abs(vgrid_kms))] += w[z]
    area = _trapz(P, lam_grid)
    if area > 0:
        P = P / area
    return P


_SIGMA_INT = 0.02654000     # π e² / (m_e c)  [cm² Hz]  (Sobolev line const)


def _resonance_beta(r, v, n_lower, f_lu, lam0_AA):
    """Per-zone Sobolev escape probability β(τ) for a resonance line whose lower
    level is the ion ground state. τ = σ_int·f_lu·λ·n_lower·t_exp, t_exp = r/|v|
    (homologous). β = (1−e^−τ)/τ → ~1 (thin) … ~1/τ (thick, self-absorbed).
    Returns (beta_per_zone, tau_per_zone)."""
    r = np.asarray(r, float)
    t_exp = r / np.maximum(np.abs(np.asarray(v, float)), 1e-30)
    tau = (_SIGMA_INT * float(f_lu) * (float(lam0_AA) * 1e-8)
           * np.asarray(n_lower, float) * t_exp)
    ts = np.maximum(tau, 1e-30)
    beta = np.where(tau > 1e-6,
                    (1.0 - np.exp(-np.minimum(ts, 700.0))) / ts,
                    1.0 - 0.5 * ts)
    return beta, tau


def _cloudy_emiss_on_grid(cloudy_entry, r):
    """Interpolate Cloudy's per-zone line emissivity (radius_cm, emiss) onto our
    radial grid `r` [erg cm^-3 s^-1]. Cloudy's grid (R_in + depth) and ours both
    increase outward. Outside Cloudy's range the emissivity is 0 (no extrapolation
    of line formation). Returns an array len(r), or None if unusable. (Item 2:
    makes the MC shape's formation region consistent with Cloudy's self-consistent
    ionization, rather than our cruder Tier-1 ladder.)"""
    if cloudy_entry is None:
        return None
    try:
        rc, ec = cloudy_entry
        rc = np.asarray(rc, float); ec = np.asarray(ec, float)
        m = np.isfinite(rc) & np.isfinite(ec)
        rc, ec = rc[m], ec[m]
        if rc.size < 2 or not np.any(ec > 0):
            return None
        order = np.argsort(rc)
        rc, ec = rc[order], ec[order]
        j = np.interp(np.asarray(r, float), rc, ec, left=0.0, right=0.0)
        j = np.where(np.isfinite(j) & (j > 0), j, 0.0)
        return j if np.any(j > 0) else None
    except Exception:
        return None


def pcygni_absorption_overlay(F_norm, lam_grid, lam0_AA, v_cms, tau_zone, P_emiss):
    """Overlay a photon-conserving Sobolev P-Cygni profile on a thermal-emission
    line (simplified SEI).

    A resonance line in an expanding wind shows a blue ABSORPTION trough (the
    approaching material in front of the star, Δv<0, absorbs continuum at τ(Δv))
    PLUS a scattered EMISSION (the absorbed continuum photons re-emerge). For pure
    scattering this pair is EW-neutral; the NET line strength is set only by the
    THERMAL emission already in `F_norm` (= 1 + thermal, scaled to Cloudy's
    L_line). So the absorbed continuum is exactly RE-EMITTED, preserving the net
    EW (= Cloudy's net emission — no double counting, no spurious net absorption):

        absorbed(Δv<0) = 1 − exp(−τ(|Δv|))          (continuum removed, blue only)
        L_abs          = ∫ absorbed dλ              (scattered photon budget)
        F_out          = F_norm − absorbed + L_abs·P_emiss

    where P_emiss is the unit-area (∫P dλ=1) emission shape, so the re-emission
    ∫L_abs·P_emiss dλ = L_abs cancels the absorption in EW. The blue trough
    survives for saturated lines (deep absorption, tall compensating emission near
    centre/red — the classic strong-wind C IV P-Cygni). `tau_zone` is the radial
    Sobolev τ per zone (= σ_int·f_lu·λ·n_lower·t_exp). Returns the modified F_norm."""
    lam_grid = np.asarray(lam_grid, float)
    F_out = np.asarray(F_norm, float).copy()
    P_emiss = np.asarray(P_emiss, float)
    dv_kms = (lam_grid / float(lam0_AA) - 1.0) * C_KMS          # >0 red, <0 blue
    vz = np.abs(np.asarray(v_cms, float)) / 1e5                 # zone |v| [km/s]
    tz = np.asarray(tau_zone, float)
    m = np.isfinite(vz) & np.isfinite(tz) & (vz > 0)
    if m.sum() < 2:
        return F_out
    vz = vz[m]; tz = tz[m]
    order = np.argsort(vz)
    vz, tz = vz[order], tz[order]
    absorbed = np.zeros_like(lam_grid)
    blue = dv_kms < 0.0
    tau_abs = np.interp(np.abs(dv_kms[blue]), vz, tz, left=tz[0], right=0.0)
    tau_abs = np.clip(tau_abs, 0.0, 700.0)
    absorbed[blue] = 1.0 - np.exp(-tau_abs)                     # continuum removed
    L_abs = float(_trapz(absorbed, lam_grid))                  # scattered budget
    F_out = F_out - absorbed + L_abs * P_emiss                 # photon-conserving
    return F_out


def _weighted_vquantile(vabs_kms, w, q=0.95):
    """Emissivity-weighted velocity quantile [km/s]: the |v| below which a
    fraction q of the total emission weight lies. Sets the line's half-width —
    metals are dominated by the dense inner shells (ρ²), NOT the fast outer
    ejecta, so this is far smaller than v_max."""
    vv = np.asarray(vabs_kms, float)
    ww = np.asarray(w, float)
    m = ww > 0
    if not m.any():
        return float(np.max(vv)) if vv.size else 0.0
    vv = vv[m]; ww = ww[m]
    order = np.argsort(vv)
    vv = vv[order]; cw = np.cumsum(ww[order])
    cw = cw / cw[-1]
    idx = min(int(np.searchsorted(cw, q)), len(vv) - 1)
    return float(vv[idx])


def _profile_hwhm_kms(lam_grid, P, lam0):
    """Half-width at half-maximum of an emission profile P(λ) [km/s], averaged
    over the two sides. Used by the boxy-width VALIDATION (#2): a shell-formed
    (flat-topped) line has HWHM comparable to the emitting-shell velocity (the
    half-max sits near the flat-top edge ≈ ±v_shell), modestly broadened by
    electron scattering. Returns nan if there is no clear peak."""
    P = np.asarray(P, float); lam = np.asarray(lam_grid, float)
    if P.size < 5 or not np.any(np.isfinite(P)):
        return float('nan')
    pk = float(np.nanmax(P))
    if not (pk > 0):
        return float('nan')
    half = 0.5 * pk
    dv = (lam / float(lam0) - 1.0) * C_KMS
    imax = int(np.nanargmax(P))
    li = imax
    while li > 0 and P[li] > half:
        li -= 1
    ri = imax
    while ri < P.size - 1 and P[ri] > half:
        ri += 1
    return float(0.5 * (abs(dv[li]) + abs(dv[ri])))


_H_PL = 6.62607015e-27
_C_CGS = 2.99792458e10


def mc_metal_profile(merged, lam0_AA, j_volume, lam_grid, n_packets=50000):
    """Transport an optically-thin metal-line emissivity through the SAME Monte-
    Carlo peel-off kernel the He lines use (proper velocity-field projection +
    MULTIPLE electron scattering), returning the UNIT-AREA line-flux SHAPE.

    The line is forced optically thin (no self-absorption) by setting the lower
    level population to zero; the emission is injected via the reconstructed
    upper-level density n_u = j_volume / (hν·A), so the kernel's volume
    emissivity (hν/4π)·n_u·A reproduces j_volume exactly (A and g cancel).

    Returns the kernel's TRANSPORTED line flux (F_total − F_cont) interpolated
    onto `lam_grid` and normalised to unit area (∫P dλ = 1) — the MC supplies the
    SHAPE only; the caller scales by L_line/(4π R²) and divides by the
    line-centre continuum (avoiding the kernel's own wide-window continuum
    baseline/slope). Returns None on any failure (caller falls back to boxcar).
    """
    try:
        import mc_multi_line as p5
        from peel_pipeline_abs import run_peel_pipeline_abs
        r = np.asarray(merged.r, float)
        snap = {'r': r,
                'v': np.asarray(merged.v, float),
                'rho': np.asarray(merged.rho, float),
                'T': np.asarray(merged.T, float),
                'n_e': np.asarray(merged.n_e, float)}
        nz = r.size
        A_ul = 1.0e8                       # arbitrary (cancels: n_u ∝ 1/A, j ∝ A)
        nu0 = _C_CGS / (float(lam0_AA) * 1e-8)
        n_u = np.asarray(j_volume, float) / (_H_PL * nu0 * A_ul)
        emissivity = (_H_PL * nu0 / (4.0 * np.pi)) * n_u * A_ul   # = j_volume/4π
        dV = _shell_volumes(r)
        L_line = float(np.sum(4.0 * np.pi * emissivity * dV))
        if not (np.isfinite(L_line) and L_line > 0):
            return None
        R_phot = float(_get(merged, 'R_phot_cm', 'R_phot', default=r[-1]))
        T_phot = float(_get(merged, 'T_phot', 'T_color', default=snap['T'][-1]))
        params = {'R_phot': R_phot, 'T_phot': T_phot, 'v_turb_kms': 20.0,
                  'L_line': L_line, 'n_lower': np.zeros(nz), 'n_upper': n_u,
                  'emissivity': emissivity, 'f_vol': 1.0}
        Lp = _get(merged, 'L_phot', default=None)
        if Lp is not None:
            params['L_phot'] = float(Lp)
        line = p5.HeLine(float(lam0_AA), A_ul, 1.0, 1.0, name='metal')
        band_AA = (float(lam_grid[0]), float(lam_grid[-1]))
        out = run_peel_pipeline_abs(
            snap, params, line=line, n_packets=n_packets, nbins=len(lam_grid),
            band_AA=band_AA, source_padding_AA=500.0, v_turb_kms=20.0,
            n_steps_peel=40, calibration='f_cont_bb', max_events=2000,
            cont_line_emission=True, line_scatter_redistribution='aa_prd',
            seed=42, verbose=False)
        # kernel returns (lam_out, F_norm, F_cont_abs, F_total_abs, info); the
        # transported LINE flux is F_total − F_cont (the volumetric channel).
        lam_out = np.asarray(out[0], float)
        F_line_native = np.asarray(out[3], float) - np.asarray(out[2], float)
        P = np.interp(lam_grid, lam_out, F_line_native, left=0.0, right=0.0)
        P = np.where(np.isfinite(P) & (P > 0.0), P, 0.0)   # emission only
        area = _trapz(P, lam_grid)
        if not (np.isfinite(area) and area > 0.0):
            return None
        return P / area
    except Exception:
        return None


def _get(state, *names, default=None):
    for n in names:
        if isinstance(state, dict) and n in state:
            return state[n]
        if hasattr(state, n):
            return getattr(state, n)
    return default


def _composition(state, key):
    """Fetch a per-zone metal mass fraction (e.g. 'c12') from state.composition
    or a direct attribute. Returns None if absent."""
    comp = _get(state, 'composition')
    if isinstance(comp, dict) and key in comp:
        return np.asarray(comp[key], float)
    val = _get(state, key)
    return None if val is None else np.asarray(val, float)


def compute_metal_lines(state, snap=None, win_kms: float = 5000.0,
                        n_pix: int = 501, smooth_kms: float = 25.0,
                        mc_profile: bool = True, n_packets_mc: int = 50000,
                        use_cloudy: bool = False, verbose: bool = True):
    """Compute spectra entries for all C/O/Ne metal lines.

    Returns (spectra, ions) where `spectra` is {line_name: dict(...)} and `ions`
    is the metal-ionization result (for diagnostics). Returns ({}, {}) if the
    required hydro/composition fields are unavailable.

    Strength tiers (best available wins, per line):
      Tier-2 Cloudy  — `use_cloudy=True` and Cloudy reports the line: a
                       self-consistent photoionization + NLTE + resonance-RT
                       ABSOLUTE L_line (overrides the emissivity-integral value;
                       the MC profile SHAPE is retained and rescaled).
      Tier-1 CHIANTI — authoritative NLTE collisional emissivity (metal_nlte).
      provisional    — metal_atoms collisional/recombination emissivity.
    """
    import phase5_continuum as p5c

    # ---- gather fields: prefer the (truncated, zone-consistent) `state`/merged,
    #      fall back to the snap dict. Hydro MUST be zone-consistent with the
    #      composition, so everything is sourced the same way and length-checked.
    def g(*names, default=None):
        val = _get(state, *names, default=None)
        if val is None and isinstance(snap, dict):
            val = _get(snap, *names, default=None)
        return default if val is None else val

    r = g('r'); v = g('v'); T = g('T'); n_e = g('n_e'); rho = g('rho')
    if any(x is None for x in (r, v, T, n_e, rho)):
        if verbose:
            print("[metal] missing hydro fields (r/v/T/n_e/rho) — metal lines skipped.")
        return {}, {}
    r = np.asarray(r, float); v = np.asarray(v, float); T = np.asarray(T, float)
    n_e = np.asarray(n_e, float); rho = np.asarray(rho, float)
    nz = T.shape[0]
    R_phot = float(_get(state, 'R_phot_cm', 'R_phot', default=r[-1]))
    T_phot = float(_get(state, 'T_phot', 'T_color', default=T[-1]))
    # photospheric bolometric, for the energy-conservation ceiling on line L
    _SIGMA_SB = 5.670374419e-5
    _Lp = _get(state, 'L_phot', default=None)
    L_phot_metal = (float(_Lp) if (_Lp and np.isfinite(_Lp))
                    else 4.0 * np.pi * R_phot ** 2 * _SIGMA_SB * T_phot ** 4)
    _L_MAX_FRAC = 0.1   # no single SN line exceeds ~10% of L_bol (energy partition)
    def _scalar(x, default=0.0):
        """Coerce a scalar-or-array field to a representative scalar (array →
        its max, i.e. the total cumulative-from-outside τ_es)."""
        if x is None:
            return float(default)
        a = np.asarray(x, float)
        return float(np.max(np.abs(a))) if a.size else float(default)

    L_X_brems = _scalar(g('L_X_brems', default=0.0))
    T_shock = _scalar(g('T_shock', default=0.0))
    tau_es = _scalar(g('tau_es_total', 'tau_es', default=0.0))

    # metal mass fractions (STELLA isotopes) — from state.composition or snap
    def _comp(key):
        x = _composition(state, key)
        if x is None and isinstance(snap, dict):
            x = _composition(snap, key)
        if x is None:
            return None
        x = np.asarray(x, float)
        if x.shape[0] != nz:           # zone mismatch (e.g. untruncated) → drop
            if verbose:
                print(f"[metal] {key}: length {x.shape[0]} != {nz} zones — "
                      f"element skipped (composition not aligned to photosphere).")
            return None
        return x

    X_C = _comp('c12'); X_O = _comp('o16'); X_Ne = _comp('ne20')
    if all(x is None for x in (X_C, X_O, X_Ne)):
        if verbose:
            print("[metal] no aligned C/O/Ne mass fractions — metal lines skipped.")
        return {}, {}

    # ---- ionization ----
    ions = mi.compute_metal_ionization(
        T, n_e, r, rho, R_phot, T_phot, X_C=X_C, X_O=X_O, X_Ne=X_Ne,
        L_X_brems=L_X_brems, T_shock=T_shock, verbose=verbose)

    # ---- L_cont_band per metal band (standard window; for energy bookkeeping
    #      and the EW-reliability guard only — the PROFILE uses its own grid) ----
    cont = p5c.build_per_band_continuum(
        state, line_wavelengths=ma.METAL_LINES_AA,
        win_kms=5000.0, n_pix=101, verbose=False)

    dV = _shell_volumes(r)
    v_kms = np.abs(v) / 1e5

    # ---- Tier-2: Cloudy absolute line luminosities (self-consistent
    #      photoionization + NLTE + resonance-line RT) AND per-zone emissivity.
    #      One run for all lines; overrides the emissivity-integral L_line per
    #      line when reported, and (Item 2) supplies the line-formation profile
    #      that weights the MC shape.
    cloudy_L = {}
    cloudy_emiss = {}
    if use_cloudy:
        try:
            import metal_cloudy as mcl
            if mcl.cloudy_available():
                cloudy_L, cloudy_emiss = mcl.metal_line_luminosities(
                    state, snap, verbose=verbose)
            elif verbose:
                print("[metal] --metal-cloudy set but Cloudy executable not found "
                      "— using CHIANTI/provisional.")
        except Exception as e:
            if verbose:
                print(f"[metal] Cloudy Tier-2 unavailable ({e}) — "
                      "using CHIANTI/provisional.")
            cloudy_L = {}; cloudy_emiss = {}

    use_chianti = mnlte.chianti_available()
    if verbose:
        print(f"[metal] line emissivities: "
              + ("CHIANTI NLTE (authoritative) for collisional lines; "
                 if use_chianti else "ChiantiPy/XUVTOP not found — PROVISIONAL "
                 "fallback for all lines; ")
              + "recombination lines provisional.")
    spectra = {}
    rows = []
    for name, d in ma.METAL_LINES.items():
        # Recombination lines scale with the RECOMBINING (parent) ion density —
        # the stage above the emitter (e.g. C III λ4650 ← C³⁺='C_IV'). CEL and
        # resonance lines use their own ion. The CHIANTI/collisional path below
        # always wants the emitter ion, so keep that separate.
        n_ion_emit = ions.get(d['ion'])
        n_ion = ions.get(d.get('recomb_parent', d['ion']))
        if n_ion is None or n_ion_emit is None:
            continue
        lam0 = float(d['lam_AA'])
        # emissivity: CHIANTI NLTE (authoritative) when available + line is
        # collisional/in-CHIANTI, else the provisional metal_atoms emissivity.
        emiss_pi = None
        if use_chianti and name in mnlte.CHIANTI_LINES:
            emiss_pi, _ntr = mnlte.line_emissivity_per_ion(name, T, n_e)
        if emiss_pi is not None:
            # CHIANTI is a per-EMITTER-ion NLTE emissivity → use the emitter density
            j = 4.0 * np.pi * np.asarray(emiss_pi, float) * np.asarray(n_ion_emit, float)
            data_source = 'chianti'
        else:
            # metal_atoms: recomb lines use the parent (n_ion), CEL/resonance use
            # the emitter (n_ion == n_ion_emit when no recomb_parent is set)
            j = ma.line_emissivity(name, T, n_e, n_ion)    # erg/s/cm³ (provisional)
            data_source = 'prov'
        # resonance-line optical depth: τ_zone = σ_int·f_lu·λ·n_lower·t_exp for a
        # resonance line whose lower level is the ion ground state (n_lower≈n_ion).
        # Used two ways: (a) β(τ) dims the emissivity-integral absolute to the
        # emergent level in the Tier-1 fallback (no Cloudy); (b) τ_zone drives the
        # P-Cygni blue-absorption overlay (Item 1). Thin lines (forbidden /
        # intercombination, f_lu≈0) get τ≈0 and are unchanged.
        flu = ma.f_lu(name)
        resonance_beta_med = 1.0
        tau_zone = None
        j_abs = j                                      # emissivity for the absolute
        if flu > 1e-4:
            beta_res, tau_zone = _resonance_beta(r, v, n_ion, flu, lam0)
            j_abs = j * beta_res                       # β-escape dimmed (fallback)
            resonance_beta_med = float(np.median(beta_res[np.asarray(n_ion) > 0])
                                       if np.any(np.asarray(n_ion) > 0) else 1.0)
        L_emiss = float(np.sum(j_abs * dV))
        if not (np.isfinite(L_emiss) and L_emiss > 0):
            continue
        # --- absolute L_line: Cloudy Tier-2 ONLY for genuine RESONANCE lines
        #     (C IV 1549, f_lu>1e-4), where Cloudy's unique contribution — the
        #     self-consistent resonance-line RT — actually matters. For
        #     intercombination / forbidden / recombination lines (C III] 1909,
        #     [O III], [Ne III], …) the CHIANTI NLTE emissivity is authoritative
        #     AND temporally stable, so they are NOT exposed to Cloudy's occasional
        #     per-epoch thermal-bistability flicker. Else: emissivity integral.
        has_cloudy = (flu > 1e-4 and name in cloudy_L
                      and np.isfinite(cloudy_L[name]) and cloudy_L[name] > 0)
        if has_cloudy:
            L_line = float(cloudy_L[name])
            data_source = 'cloudy'
        else:
            L_line = L_emiss
        # energy-conservation ceiling (mirrors metal_cloudy): an over-bright thin
        # provisional/CHIANTI resonance line — or any value slipping through —
        # cannot exceed L_MAX_FRAC·L_phot. Cap + flag (keeps the time series from
        # spiking on the over-bright thin estimate when Cloudy is unavailable).
        energy_capped = False
        if L_phot_metal > 0 and L_line > _L_MAX_FRAC * L_phot_metal:
            L_line = _L_MAX_FRAC * L_phot_metal
            energy_capped = True
        # --- SHAPE-weighting emissivity j_shape (Item 2): Cloudy's per-zone line
        #     emissivity (self-consistent ionization) when available, else our
        #     emissivity. For the Tier-1 fallback the β-dimmed j_abs is the shape
        #     (suppresses the optically-thick inner zones → τ~1 surface).
        j_shape = None
        if has_cloudy and name in cloudy_emiss:
            j_shape = _cloudy_emiss_on_grid(cloudy_emiss[name], r)
        if j_shape is None:
            j_shape = j_abs
        w = j_shape * dV
        if not np.any(w > 0):
            w = j * dV
        # PER-LINE velocity window from the EMISSIVITY-weighted velocity (the
        # line is set by the dense inner shells, not the fast outer ejecta), so
        # the profile is neither clipped (too-narrow window) nor a spike in an
        # empty window (too-wide v_max window).
        v95 = _weighted_vquantile(v_kms, w, 0.95)
        win_line = float(np.clip(1.6 * v95, 4000.0, 150000.0))
        n_pix_line = int(np.clip(n_pix, 301, 1201))
        dlam = lam0 * win_line * 1e5 / (C_KMS * 1e5)      # AA half-width
        lam_grid = np.linspace(lam0 - dlam, lam0 + dlam, n_pix_line)
        # continuum AT LINE CENTRE (scalar): the symmetric line must be
        # normalised to a single reference, NOT the per-pixel sloped BB
        # continuum (which would distort the profile across a wide window).
        F_cont0 = float(np.atleast_1d(
            p5c.diluted_bb_F_lambda(T_phot, R_phot, np.array([lam0])))[0])
        L_cont_band = float(cont.get(name, {}).get('L_cont_band', np.nan))
        # PROFILE: route the emissivity through the He Monte-Carlo peel-off
        # (proper velocity-field projection + MULTIPLE electron scattering) for a
        # realistic line shape; fall back to the analytic optically-thin boxcar
        # (+ single-scatter Thomson) if the MC kernel is unavailable or fails.
        # unit-area line SHAPE P(λ): MC peel-off (velocity field + multiple
        # electron scattering) when available, else the analytic boxcar.
        P = None
        if mc_profile:
            P = mc_metal_profile(state, lam0, j_shape, lam_grid, n_packets=n_packets_mc)
        if P is not None:
            profile_method = 'MC'
        else:
            P = thin_emission_profile(lam_grid, lam0, v, w)   # ∫P dλ = 1
            P = _gaussian_smooth(lam_grid, P, lam0, smooth_kms)
            profile_method = 'boxcar'
        # ---- VALIDATION (#2): boxy-width vs emitting-shell velocity. The flat-top
        # half-width of a shell-formed line should track the velocity of the gas
        # that actually emits it (the emission-measure-weighted median |v|), NOT
        # the fast forward shock. A profile HWHM far from that velocity signals an
        # artificially narrow (clipped window) or broad (mis-placed emissivity)
        # shape rather than physics. Diagnostic only — never alters L or the shape.
        v_shell_emit = float(_weighted_vquantile(v_kms, w, 0.50))   # km/s
        v_prof_hwhm = _profile_hwhm_kms(lam_grid, P, lam0)          # km/s
        width_ratio = (v_prof_hwhm / v_shell_emit
                       if (np.isfinite(v_prof_hwhm) and v_shell_emit > 0)
                       else float('nan'))
        width_ok = bool(np.isfinite(width_ratio) and 0.4 <= width_ratio <= 6.0)
        # normalise the line to MY line-centre continuum (clean, symmetric;
        # avoids the kernel's wide-window continuum-slope artifact)
        F_norm = 1.0 + (L_line / (4.0 * np.pi * R_phot ** 2)) * P \
            / max(F_cont0, 1e-300)
        # electron scattering: the MC already did it; only the boxcar needs it
        if profile_method == 'boxcar' and tau_es > 0.0:
            try:
                import line_rt_escape as _ep
                F_norm = _ep.thomson_multiscatter(lam_grid, F_norm, lam0,
                                                  T_phot, tau_es)
            except Exception:
                pass
        # continuum-suppression flags (continuum-only criteria): when the band
        # continuum is Wien-negligible (UV line at a cold photosphere, e.g.
        # C IV 1549 @ 3387 K) the F_norm/EW are meaningless — quote L_line.
        cont_suppressed = bool(
            not np.isfinite(L_cont_band) or L_cont_band <= 0.0
            or not np.isfinite(F_cont0) or F_cont0 <= 0.0
            or L_line > 50.0 * L_cont_band)
        # Item 1 — P-Cygni blue absorption for resonance lines: overlay the
        # Sobolev trough exp(−τ(Δv<0)) on the continuum, ON TOP of which the
        # emission rides (photon budget unchanged). Only when there is a real
        # continuum to absorb (skip if cont-suppressed — no continuum → no
        # P-Cygni, the line is pure emission there).
        pcygni_applied = False
        if (flu > 1e-4 and tau_zone is not None and not cont_suppressed
                and np.nanmax(tau_zone) > 1e-3):
            F_norm = pcygni_absorption_overlay(F_norm, lam_grid, lam0, v,
                                               tau_zone, P)
            pcygni_applied = True
        F_cont = np.full_like(lam_grid, F_cont0)
        F_line = (np.asarray(F_norm, float) - 1.0) * F_cont0   # line flux (both paths)
        F_lam = F_cont * F_norm
        EW = float(-_trapz(F_norm - 1.0, lam_grid))
        # full EW-reliability guard: cont-suppression OR an EW that fills the band
        # (artifact). Flag + NaN the EW so it does not propagate downstream.
        band_span = float(lam_grid[-1] - lam_grid[0])
        ew_unreliable = bool(cont_suppressed or abs(EW) > 0.5 * band_span)
        EW_out = float('nan') if ew_unreliable else EW
        sp = dict(
            lambda_AA=lam_grid, F_lambda=F_lam, F_cont_lambda=F_cont,
            F_norm=F_norm, F_line_only=F_line,
            L_line=L_line, L_cont_band=L_cont_band, EW=EW_out,
            tau_med=0.0, lambda_rest=lam0,
            ew_unreliable=ew_unreliable,
            v95_kms=v95,
            # boxy-width validation (#2): emitting-shell velocity, profile HWHM,
            # their ratio, and a pass flag (shell-formed lines: HWHM ~ v_shell).
            v_shell_emit_kms=v_shell_emit,
            v_prof_hwhm_kms=v_prof_hwhm,
            width_ratio=width_ratio, width_ok=width_ok,
            # metal lines are first-principles (no anchor/correction):
            L_line_corrected=L_line, EW_corrected=EW_out,
            F_norm_corrected=F_norm.copy(),
            # peak F is meaningless when the continuum is suppressed (UV/cold);
            # NaN it so the txt doesn't show a 1e7 F/F_cont — the shape is in
            # F_norm and the PNG renders it peak-normalised.
            peak_F_corrected=(float('nan') if ew_unreliable
                              else float(np.max(F_norm))),
            strength_mode=f"metal-{d['mech']}-{data_source}",
            data_source=({'cloudy': 'Cloudy-Tier2',
                          'chianti': 'CHIANTI-NLTE'}.get(data_source,
                                                         'provisional')),
            L_emiss=L_emiss,            # the emissivity-integral value (pre-override)
            profile_method=profile_method,
            resonance_beta=resonance_beta_med,
            pcygni=pcygni_applied,      # Item 1: P-Cygni blue absorption overlaid
            energy_capped=energy_capped,  # L_line hit the L_MAX_FRAC·L_phot ceiling
            shape_source=('cloudy' if (has_cloudy and name in cloudy_emiss
                                       and j_shape is not j_abs) else data_source),
            provisional=(data_source not in ('chianti', 'cloudy')),
        )
        spectra[name] = sp
        rows.append((name, d['mech'], data_source, profile_method,
                     resonance_beta_med, L_line, EW_out, ew_unreliable,
                     pcygni_applied))

    if verbose and rows:
        n_mc = sum(1 for r in rows if r[3] == 'MC')
        n_cl = sum(1 for r in rows if r[2] == 'cloudy')
        print(f"[metal] line strengths (Cloudy = Tier-2 self-consistent absolute, "
              f"CHIANTI = NLTE emissivity, prov = provisional; "
              f"{n_cl}/{len(rows)} Cloudy, profile: {n_mc}/{len(rows)} MC peel-off):")
        for nm, mech, src, prof, rbeta, L, ew, unrel, pcyg in rows:
            ew_str = ("EW=cont-suppressed (quote L_line)" if unrel
                      else f"EW={ew:+.2f} AA")
            src_tag = {'cloudy': 'CLOUDY ', 'chianti': 'CHIANTI'}.get(src, 'prov   ')
            thick = f" thick(β≈{rbeta:.1e})" if rbeta < 0.95 else ""
            pcyg_str = " +P-Cygni" if pcyg else ""
            print(f"        {nm:14s} {mech:4s} [{src_tag}] prof={prof:6s} "
                  f"L_line={L:.3e} erg/s  {ew_str}{thick}{pcyg_str}")
        # boxy-width validation (#2): profile HWHM vs the emitting-shell velocity
        print(f"[metal] boxy-width check (HWHM vs emitting-shell v50; shell-formed "
              f"lines: ratio ~0.5-1.5, the half-max on the boxy shoulder):")
        for nm in (r[0] for r in rows):
            s = spectra[nm]
            vs = s.get('v_shell_emit_kms', float('nan'))
            vh = s.get('v_prof_hwhm_kms', float('nan'))
            wr = s.get('width_ratio', float('nan'))
            flag = "ok" if s.get('width_ok') else "⚠ CHECK"
            print(f"        {nm:14s} v_shell={vs:7.0f}  HWHM={vh:7.0f} km/s  "
                  f"ratio={wr:4.2f}  [{flag}]")
    return spectra, ions


def save_metal_png(spectra, path, epoch_d=None, T_phot=None, R_phot=None):
    """Dedicated multi-panel figure of the metal-line profiles for one snapshot
    ({out_prefix}_metal_lines.png). One panel per metal line present in `spectra`,
    F/F_cont vs Δv, coloured by element. Returns True if written, False if no
    metal lines are present. Non-fatal on any plotting error."""
    names = [n for n in ma.METAL_LINES if n in spectra]
    if not names:
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        sp = spectra[name]
        lam = np.asarray(sp['lambda_AA'], float)
        lam0 = float(sp['lambda_rest'])
        dv = (lam / lam0 - 1.0) * C_KMS
        Fn = np.asarray(sp.get('F_norm_corrected', sp['F_norm']), float)
        d = ma.METAL_LINES[name]
        ax.axvline(0.0, ls=':', color='grey', lw=0.6)
        ew = sp.get('EW', float('nan'))
        if sp.get('ew_unreliable') or not np.isfinite(ew):
            # continuum-suppressed (e.g. UV line, cold photosphere): F_norm is
            # meaningless → show the peak-normalised LINE SHAPE instead.
            Fl = np.asarray(sp.get('F_line_only', Fn - 1.0), float)
            pk = float(np.max(np.abs(Fl)))
            ax.plot(dv, Fl / pk if pk > 0 else Fl,
                    color=ma.line_color(name), lw=1.3)
            ax.set_ylabel('line flux (peak-norm.)', fontsize=8)
            ew_str = "EW: cont-suppressed (shape only)"
        else:
            ax.plot(dv, Fn, color=ma.line_color(name), lw=1.3)
            ax.axhline(1.0, ls=':', color='grey', lw=0.8)
            ax.set_ylabel('F / F_cont', fontsize=8)
            ew_str = f"EW={ew:+.1f} Å"
        mech_tag = d['mech'] + (', P-Cyg' if sp.get('pcygni') else '')
        ax.set_title(f"{ma.METAL_PRETTY.get(name, name)}  ({mech_tag})\n"
                     f"L={sp['L_line']:.2e}  {ew_str}", fontsize=9)
        ax.set_xlabel('Δv [km/s]', fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(names), nrows * ncols):     # hide empty panels
        axes[j // ncols][j % ncols].axis('off')
    sup = "Metal lines (C/O/Ne) — first-principles emissivity; ATOMIC DATA PROVISIONAL"
    bits = []
    if epoch_d is not None:
        bits.append(f"epoch = {float(epoch_d):.1f} d")
    if T_phot is not None:
        bits.append(f"T_phot = {float(T_phot):.0f} K")
    if R_phot is not None:
        bits.append(f"R_phot = {float(R_phot):.2e} cm")
    if bits:
        sup += "   |   " + ", ".join(bits)
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
