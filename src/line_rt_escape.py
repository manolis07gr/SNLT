"""
line_rt_escape.py — Saturated-line radiative transfer via a continuum-pumped
two-level escape-probability (EP) source function.
================================================================================

FUTURE_WORK P1 item 3.  The legacy Phase-5 path applies the Sobolev escape
probability β(τ)=(1−e^−τ)/τ ONCE per zone to the volume emissivity and then, for
optically-thick lines (τ_med ≥ 1), rescales the result by the *empirical*
Hα-anchored factor R_flat = L_prod(Hα)/L_phase5(Hα).  That single-shot kernel
cannot represent saturated transport (τ ≫ 1): the line source function is set in
one pass and the thick-line escape is borrowed from Hα rather than computed.

CORRECTED ROLE (after numeric verification — see commit notes / handover).
A numerical check showed two things:

  (1) The escape-probability luminosity built from the ACTUAL He-NLTE populations,
      L = Σ 4π (1−e^−τ_S) (j_ν/χ_ν) ΔV / (λ₀ t_exp), is ALGEBRAICALLY IDENTICAL to
      the existing single-shot β luminosity Σ hν n_u A_ul β ΔV (ratio 1.0000).
      I.e. the single-shot β value is ALREADY the first-principles escape-
      probability luminosity for the given populations — it was the empirical
      Hα-anchored R_flat MULTIPLIER applied on top of it that lacked justification.

  (2) A closed-form continuum-PUMPED two-level source, used as a REPLACEMENT,
      over-pumps a cool recombination-fed thick line by ~10²–10³× (it pins the
      source toward the diluted photospheric continuum and silently drops the
      recombination cascade). So pumped-EP-as-replacement is NOT valid for the
      recombination-dominated He lines.

Therefore the validated --saturated-rt behaviour is:
  • LUMINOSITY: drop the empirical R_flat for thick He lines and keep the bare
    single-shot β escape luminosity (= first-principles EP luminosity, verified).
    `escape_probability_luminosity()` reproduces it from S_L=j/χ as a cross-check.
  • SHAPE: apply MULTIPLE electron scattering (`thomson_multiscatter`) — the
    deterministic, photon-conserving generalisation of the existing single-scatter
    broadening — to the thick-line profile (broadens wings, suppresses peak,
    leaves L and EW unchanged).  This is the "+Thomson MC" part.

The continuum-pumped source function (`ep_source_function`, `compute_line_ep`)
is retained as a DIAGNOSTIC and a foundation for a future iterated-J̄/ALI solver,
but is NOT used to rescale recombination-line luminosity (the ~10³× hazard above).
A genuinely-below-single-shot saturation suppression requires the nonlocal
iterated-J̄ (ALI) route, which is intentionally out of scope here.

Physics
-------
Two-level atom in the Sobolev approximation with an external (photospheric)
continuum.  The local line mean intensity closes as

    J̄ = (1 − β) S_L + β_c I_c                                            (1)

where β(τ_S)=(1−e^−τ_S)/τ_S is the Sobolev escape probability, I_c=B_ν(T_phot) is
the photospheric continuum brightness at the line frequency, and
β_c = W(r)·β is the probability that core continuum radiation penetrates to and is
absorbed in the resonance zone (W(r)=½[1−√(1−(R_phot/r)²)] is the geometric
dilution of the photosphere).  Combined with the two-level source function
S_L = (1−ε) J̄ + ε B_ν(T), ε being the per-scatter collisional destruction
probability, (1) closes in CLOSED FORM:

    S_L = [ ε B_ν(T) + (1−ε) β_c I_c ] / [ ε + (1−ε) β ]                  (2)

Limits (all unit-testable without the pipeline):
  • ε → 1            : S_L → B_ν(T)                    (collisionally thermalised)
  • β → 1 (τ → 0)    : S_L → ε B + (1−ε) W I_c          (diluted continuum pump)
  • β → 0 (τ → ∞)    : S_L → B_ν(T)                     (thermalised; trapping)

Emergent line luminosity.  In the Sobolev picture the frequency-integrated line
opacity is χ_int = τ_S/(λ₀ t_exp), and a fraction β of locally produced line
energy escapes, so the escaping luminosity per unit volume is 4π β χ_int S_L =
4π (1−e^−τ_S) S_L / (λ₀ t_exp).  Hence

    L_line = Σ_zones 4π (1 − e^−τ_S) S_L ΔV / (λ₀ t_exp)                  (3)

Limits of (3):
  • τ → 0 : (1−e^−τ_S) → τ_S, and χ_int S_L → (hν₀/4π) n_u A_ul, so
            L_line → Σ hν₀ n_u A_ul ΔV — the optically-thin recombination
            luminosity (identical to the legacy thin value: thin lines are
            unchanged, factor ≈ 1, preserving the validated thin behaviour).
  • τ → ∞ : (1−e^−τ_S) → 1, L_line → Σ 4π S_L ΔV/(λ₀ t_exp) — saturated, set by
            the surface brightness S_L, NOT by the empirical Hα anchor.

This module is OPT-IN.  Nothing here runs unless the caller (phase5_runner, gated
by the production_runner --saturated-rt flag) invokes it.  It does not touch the
Sobolev profile-shape gate, the production-Hα path, or any optically-thin line.

All quantities are cgs.  S_L, I_c, B_ν are specific intensities
[erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹]; L_line is [erg s⁻¹].
"""
from __future__ import annotations
import numpy as np

# Physical constants (cgs) — identical to formal_line_profile.py
C  = 2.99792458e10          # cm/s
H  = 6.62607015e-27         # erg s
KB = 1.380649e-16           # erg/K


def planck_nu(nu, T):
    """Planck function B_ν(T) [erg/s/cm²/sr/Hz]. Robust to overflow.

    Matches formal_line_profile.planck_nu exactly (duplicated so this module is
    importable/testable standalone, without an import cycle through phase5).
    """
    T = np.maximum(np.asarray(T, float), 1.0)
    x = np.minimum(H * float(nu) / (KB * T), 700.0)
    return (2.0 * H * float(nu) ** 3 / C ** 2) / np.expm1(x)


def dilution_W(r, R_phot_cm):
    """Geometric dilution factor of a photosphere of radius R_phot seen at r.

        W(r) = ½ [ 1 − √(1 − (R_phot/r)²) ]

    W = ½ at the photosphere (r = R_phot) and → ¼ (R_phot/r)² far out. r below
    R_phot is clamped to the photosphere (W = ½); the line-forming envelope is
    outside the continuum photosphere so this clamp is only a numerical guard.
    """
    r = np.asarray(r, float)
    R = float(R_phot_cm)
    x = np.clip((R / np.maximum(r, 1e-30)) ** 2, 0.0, 1.0)
    return 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - x, 0.0)))


def sobolev_beta(tau_S):
    """Sobolev escape probability β(τ)=(1−e^−τ)/τ, with the standard thin-limit
    series 1−τ/2 for τ→0 and overflow-safe exponent clipping."""
    tau = np.asarray(tau_S, float)
    tau_safe = np.maximum(tau, 1e-30)
    tau_exp = np.minimum(tau_safe, 700.0)
    return np.where(tau > 1e-6,
                    (1.0 - np.exp(-tau_exp)) / tau_safe,
                    1.0 - 0.5 * tau_safe)


def _shell_volumes(r):
    """Spherical-shell volumes ΔV for a 1-D radial grid (matches
    formal_line_profile._shell_volumes)."""
    r = np.asarray(r, float)
    edge = np.empty(len(r) + 1)
    edge[1:-1] = 0.5 * (r[:-1] + r[1:])
    edge[0] = r[0]
    edge[-1] = r[-1] + (r[-1] - r[-2])
    return (4.0 / 3.0) * np.pi * (edge[1:] ** 3 - edge[:-1] ** 3)


def collisional_eps(n_e, T, A_ul, q_ul=None, g_u=None, collision_strength=5.0):
    """Per-scatter collisional destruction probability ε = C_ul / (A_ul + C_ul).

    C_ul = n_e · q_ul is the collisional de-excitation rate [s⁻¹].  If a
    de-excitation rate coefficient q_ul [cm³/s] is supplied (e.g. from
    he1_atom.collisional_deexcitation_HeI), it is used directly; otherwise q_ul is
    built from an effective collision strength Υ via the standard
    q_ul = 8.629e-6 · Υ / (g_u √T).  This is the SAME ε convention as
    formal_line_profile.collisional_destruction_eps (β-independent, so β is not
    double-counted in eqn (2)).
    """
    n_e = np.asarray(n_e, float)
    T = np.maximum(np.asarray(T, float), 1.0)
    if q_ul is not None:
        q = np.asarray(q_ul, float)
    else:
        g_u = 1.0 if g_u is None else float(g_u)
        q = 8.629e-6 * float(collision_strength) / (g_u * np.sqrt(T))
    C_ul = n_e * q
    return C_ul / (np.asarray(A_ul, float) + C_ul)


# ---------------------------------------------------------------------------
# VALIDATED, USED BY THE PIPELINE (--saturated-rt)
# ---------------------------------------------------------------------------

def escape_probability_luminosity(tau_S, n_lower, n_upper, A_ul, g_l, g_u,
                                  lam0_cm, r, v=None, t_exp=None):
    """First-principles escape-probability line luminosity from the ACTUAL
    (He-NLTE) populations [erg/s].

        L = Σ 4π (1 − e^−τ_S) S_L^NLTE ΔV / (λ₀ t_exp),
        S_L^NLTE = (2hν³/c²) / [ (g_u n_l)/(g_l n_u) − 1 ]   (two-level source)

    This is algebraically identical to the single-shot Σ hν n_u A_ul β ΔV — it is
    provided so the pipeline can DROP the empirical R_flat and assert that the
    bare single-shot value is the first-principles escape luminosity (and to
    cross-check that identity numerically). Returns the luminosity; falls back to
    the n_u-based form where the population inversion guard trips.
    """
    tau_S = np.asarray(tau_S, float)
    n_l = np.asarray(n_lower, float)
    n_u = np.maximum(np.asarray(n_upper, float), 1e-300)
    r = np.asarray(r, float)
    if t_exp is None:
        v = np.asarray(v, float)
        t_exp = float(np.median(r / np.maximum(np.abs(v), 1e-30)))
    nu0 = C / float(lam0_cm)
    ratio = (g_u * n_l) / (g_l * n_u)
    # two-level source function; guard the (ratio-1) denominator near inversion
    denom = ratio - 1.0
    S_nlte = np.where(np.abs(denom) > 1e-6,
                      (2.0 * H * nu0 ** 3 / C ** 2) / denom, 0.0)
    one_minus = 1.0 - np.exp(-np.minimum(np.maximum(tau_S, 0.0), 700.0))
    dV = _shell_volumes(r)
    L_ep = float(np.sum(4.0 * np.pi * one_minus * S_nlte * dV
                        / (float(lam0_cm) * float(t_exp))))
    # robust fallback: the identical n_u-based single-shot form
    beta = sobolev_beta(tau_S)
    L_ss = float(np.sum(H * nu0 * n_u * float(A_ul) * beta * dV))
    return L_ep if np.isfinite(L_ep) and L_ep > 0 else L_ss


def thomson_multiscatter(lam_AA, F_norm, lam0_AA, T_e, tau_es, n_kernel=301):
    """Multiple electron-scattering redistribution of a line profile ("Thomson
    MC", deterministic & photon-conserving).

    Generalises formal_line_profile.electron_scatter_broaden (single scatter) to
    the multiple-scattering regime of dense CSM. A photon escaping a medium of
    electron-scattering depth τ_es undergoes a mean N̄ scatters — N̄ ≈ τ_es for
    τ_es ≲ 1 (single-scatter probability) growing to the random-walk N̄ ≈ τ_es²
    for τ_es ≳ 1 — each imprinting a thermal Doppler kick of 1-D dispersion
    σ_e = √(kT_e/m_e). The cumulative kick is a Gaussian of width σ_tot = σ_e √N̄.
    The scattered fraction f = 1 − e^−τ_es is convolved; the unscattered fraction
    keeps the core. The convolution preserves ∫(F_norm−1)dλ, so L_line and EW are
    UNCHANGED — only the profile shape (broadened wings, suppressed peak), which
    is the saturated-line electron-scattering signature in IIn/Ibn spectra.

    Operates in wavelength space; σ_tot is mapped to Δλ via λ₀ σ_tot/c. Returns
    the redistributed F_norm (same grid). No-op for τ_es ≤ 0.
    """
    F_norm = np.asarray(F_norm, float)
    tau_es = float(tau_es)
    if tau_es <= 0.0 or F_norm.size < 3:
        return F_norm
    ME = 9.1093837e-28
    sigma_e_kms = np.sqrt(KB * max(float(T_e), 1.0) / ME) / 1e5
    N_bar = tau_es + tau_es * tau_es          # ≈ τ_es (thin) → τ_es² (thick)
    sigma_tot_kms = sigma_e_kms * np.sqrt(max(N_bar, 1.0))
    sigma_lam = float(lam0_AA) * sigma_tot_kms * 1e5 / C   # AA
    dlam = float(np.mean(np.diff(np.asarray(lam_AA, float))))
    if dlam <= 0 or sigma_lam <= 0:
        return F_norm
    half = n_kernel // 2
    kx = (np.arange(n_kernel) - half) * dlam
    g = np.exp(-0.5 * (kx / sigma_lam) ** 2)
    g_sum = g.sum()
    if g_sum <= 0:
        return F_norm
    g /= g_sum
    excess = F_norm - 1.0
    scattered = np.convolve(excess, g, mode='same')
    f_sc = 1.0 - np.exp(-tau_es)
    return 1.0 + (1.0 - f_sc) * excess + f_sc * scattered


# ---------------------------------------------------------------------------
# DIAGNOSTIC ONLY — continuum-pumped two-level source.
# NOT used to rescale recombination-line luminosity (over-pumps by ~10³×; see
# the module docstring). Retained for inspection and as an ALI foundation.
# ---------------------------------------------------------------------------

def ep_source_function(tau_S, T, n_e, r, R_phot_cm, T_phot, nu0, A_ul,
                       q_ul=None, g_u=None, collision_strength=5.0):
    """[DIAGNOSTIC] Continuum-pumped two-level Sobolev source function S_L, eqn (2).

    Parameters
    ----------
    tau_S : per-zone Sobolev optical depth (array)
    T, n_e : per-zone gas temperature [K], electron density [cm⁻³] (arrays)
    r : per-zone radius [cm] (array)
    R_phot_cm, T_phot : photospheric radius [cm] and temperature [K] (scalars)
    nu0 : line rest frequency [Hz] (scalar)
    A_ul : Einstein A [s⁻¹] (scalar)
    q_ul : optional per-zone collisional de-excitation coefficient [cm³/s]
    g_u, collision_strength : fallback for ε when q_ul is not supplied

    Returns
    -------
    dict with keys 'S_L' (erg/s/cm²/Hz/sr), 'beta', 'beta_c', 'eps', 'W'
    (all per-zone arrays), so callers can inspect the regime.
    """
    tau_S = np.asarray(tau_S, float)
    T = np.maximum(np.asarray(T, float), 1.0)
    beta = sobolev_beta(tau_S)
    W = dilution_W(r, R_phot_cm)
    B_local = planck_nu(nu0, T)               # local thermal source
    I_c = float(planck_nu(nu0, max(float(T_phot), 1.0)))  # photospheric brightness
    eps = collisional_eps(n_e, T, A_ul, q_ul=q_ul, g_u=g_u,
                          collision_strength=collision_strength)
    beta_c = W * beta                          # core-continuum penetration
    num = eps * B_local + (1.0 - eps) * beta_c * I_c
    den = eps + (1.0 - eps) * beta
    den = np.where(den > 1e-300, den, 1e-300)
    S_L = num / den
    return {'S_L': S_L, 'beta': beta, 'beta_c': beta_c, 'eps': eps, 'W': W}


def line_luminosity_ep(tau_S, S_L, r, lam0_cm, t_exp):
    """Emergent saturated-line luminosity from the EP source function, eqn (3).

        L_line = Σ 4π (1 − e^−τ_S) S_L ΔV / (λ₀ t_exp)   [erg/s]

    Exact in the thin (→ Σ hν n_u A_ul ΔV) and thick (→ Σ 4π S_L ΔV/(λ₀ t_exp))
    limits.
    """
    tau_S = np.asarray(tau_S, float)
    S_L = np.asarray(S_L, float)
    one_minus = 1.0 - np.exp(-np.minimum(np.maximum(tau_S, 0.0), 700.0))
    dV = _shell_volumes(r)
    pref = 4.0 * np.pi / (float(lam0_cm) * float(t_exp))
    return float(np.sum(pref * one_minus * S_L * dV))


def compute_line_ep(tau_S, T, n_e, r, v, R_phot_cm, T_phot, lam0_cm, A_ul,
                    q_ul=None, g_u=None, collision_strength=5.0, t_exp=None):
    """Convenience wrapper: EP source function + emergent luminosity for one line.

    Returns (L_line_EP [erg/s], diag dict).  t_exp defaults to median(r/v)
    (homologous expansion time), matching the rest of the pipeline.
    """
    r = np.asarray(r, float)
    if t_exp is None:
        v = np.asarray(v, float)
        t_exp = float(np.median(r / np.maximum(np.abs(v), 1e-30)))
    nu0 = C / float(lam0_cm)
    src = ep_source_function(tau_S, T, n_e, r, R_phot_cm, T_phot, nu0, A_ul,
                             q_ul=q_ul, g_u=g_u,
                             collision_strength=collision_strength)
    L = line_luminosity_ep(tau_S, src['S_L'], r, lam0_cm, t_exp)
    diag = dict(src)
    diag['t_exp'] = t_exp
    diag['L_line_EP'] = L
    return L, diag
