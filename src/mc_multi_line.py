"""
mc_multi_line.py — Phase 5 multi-line MC driver for synthetic He profiles
================================================================================

Given a populated state object (after Phase 3 + Phase 4), produce
synthetic F_λ(λ) for each of the 8 He lines using single-shot NLTE
populations (no Phase 5 ↔ Phase 3 iteration in this version).

Reads from these state attributes (populated by Phase 3/4 wrappers):
    state.he1_tau     — dict {line_name: per-zone τ array}, set by
                         snline_he1_integration.add_he1_populations_to_state
    state.he1_levels  — (11, n_zones) per-zone He I level populations
    state.he2_tau     — dict {line_name: per-zone τ array}, set by
                         snline_he2_integration.add_he2_populations_to_state
    state.he2_levels  — (n_he2_lev, n_zones) He II level populations

This module is organized in three layers:

  Layer 1 — Factory functions
    Extract per-line (tau_zone, S_zone, lambda_0) triplets from state.

  Layer 2 — MC driver
    Per-line Sobolev peel-off MC. Two backends:
    (a) Reference implementation here, volumetric emission only.
    (b) Callback to user's existing peel-off kernel via peel_callback.

  Layer 3 — Driver entry point
    compute_phase5_spectra(state, lines=None, n_packets=50000, ...).

Output structure:
    spectra[line_name] = {
        'lambda_AA':     array (n_pix,)
        'F_lambda':      array (n_pix,) erg/s/cm²/AA — total flux
        'F_line_only':   array (n_pix,) — emission contribution only
        'F_cont_lambda': array (n_pix,) — continuum baseline
        'F_norm':        array (n_pix,) — F_lambda / F_cont
        'L_line':        float — integrated escaped line luminosity (erg/s)
        'L_cont_band':   float — band-integrated continuum luminosity (erg/s)
        'EW':            float — equivalent width (AA, negative for emission)
        'lambda_rest':   float — line rest wavelength (AA)
        'tau_med':       float — median Sobolev τ (cross-check vs Phase 3/4)
    }
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Callable

# Local imports
import phase5_continuum
from phase5_continuum import HE_LINES_AA, C_LIGHT
try:
    import formal_line_profile as _flp_p5
    _HAVE_FORMAL_P5 = True
except Exception:
    _HAVE_FORMAL_P5 = False


# Atomic data: per-line (lam0_AA, A_ul_s, g_l, g_u, lower_index, upper_index)
# Level indices follow the standard 11-level He I model used by Phase 3:
#   0=1¹S, 1=2³S, 2=2¹S, 3=2³P, 4=2¹P, 5=3³S, 6=3¹S, 7=3³P, 8=3¹P, 9=3³D, 10=3¹D
# For He II (hydrogenic), indices are 0=n=1, 1=n=2, 2=n=3, 3=n=4, 4=n=5.
# Verify against nlte_he1.LEVELS / nlte_he2.LEVELS if the wrapper uses
# different indexing; mismatch will give wrong emission strengths.
HE_LINE_DATA = {
    # name           lam0(AA)   A_ul       g_l   g_u  idx_lower  idx_upper
    'He_I_10830':  (10830.34,  1.0216e7,    3,    9,     1,         3),
    'He_I_5876':   (5875.62,   7.0648e7,    9,   15,     3,         9),
    'He_I_7065':   (7065.19,   1.5462e7,    9,    3,     3,         5),
    'He_I_6678':   (6678.15,   6.3705e7,    3,    5,     4,        10),
    'He_II_1640':  (1640.42,   6.5450e8,    8,   18,     1,         2),
    'He_II_4686':  (4685.71,   1.7724e8,   18,   32,     2,         3),
    'He_II_3203':  (3203.10,   3.2244e7,   18,   50,     2,         4),
    'He_II_10124': (10123.6,   2.0090e7,   32,   50,     3,         4),
}

# H I lines (Balmer + Paschen). Indices follow the H NLTE solver convention
# where state.h_levels has shape (nlev, n_zones) with row i = principal
# quantum number n=(i+1). With default nlte_levels=5, only transitions
# involving n ≤ 5 can be computed; Hδ (n=6) and Pγ (n=6) are out of range
# and SKIPPED unless the H NLTE solver is rerun with nlev=6.
#
# A_ul values from NIST Atomic Spectra Database (Wiese & Fuhr 2009).
# g_n = 2n² for hydrogen.
H_LINE_DATA = {
    # name        lam0(AA)   A_ul       g_l   g_u   idx_lower  idx_upper
    'Halpha':   (6562.81,   4.4101e7,    8,   18,      1,         2),
    'Hbeta':    (4861.33,   8.4193e6,    8,   32,      1,         3),
    'Hgamma':   (4340.47,   2.5304e6,    8,   50,      1,         4),
    'Palpha':   (18751.01,  8.9860e6,   18,   32,      2,         3),
    'Pbeta':    (12818.08,  2.2008e6,   18,   50,      2,         4),
}

# Unified line catalog — used by extract_line_inputs to route per-species.
# Species inferred from line name prefix: 'He_I_*' → HeI, 'He_II_*' → HeII,
# everything else (Halpha, Hbeta, ...) → HI.
LINE_CATALOG = {**HE_LINE_DATA, **H_LINE_DATA}


def _species_of(line_name: str) -> str:
    """Return 'HeI', 'HeII', or 'HI' for a line name in LINE_CATALOG."""
    if line_name.startswith('He_II_'):
        return 'HeII'
    if line_name.startswith('He_I_'):
        return 'HeI'
    return 'HI'


# ============================================================================
# Layer 1: factory functions
# ============================================================================

def extract_line_inputs(state, line_name: str) -> dict:
    """Extract per-zone (tau, n_l, n_u, λ_rest, A_ul, ...) for one He line.

    Reads from:
        state.he1_tau / state.he2_tau  — per-line τ dict
        state.he1_levels / state.he2_levels — level populations array

    Falls back to state.he1_diag['line_tau'] etc. if the dict-style storage
    is used instead (alternate naming convention).
    """
    if line_name not in LINE_CATALOG:
        raise ValueError(f"Unknown line: {line_name}. "
                         f"Available: {list(LINE_CATALOG.keys())}")

    lam0, A_ul, g_l, g_u, idx_l, idx_u = LINE_CATALOG[line_name]
    species = _species_of(line_name)
    is_he2 = (species == 'HeII')

    # --- Per-zone level populations: select the right level array by species
    if species == 'HeII':
        levels_attr = 'he2_levels'
    elif species == 'HeI':
        levels_attr = 'he1_levels'
    else:   # 'HI'
        levels_attr = 'h_levels'
    levels = getattr(state, levels_attr, None)
    if levels is None:
        # Try a couple of alternate names
        for alt in (levels_attr.replace('_levels', '_n_lev'),
                     levels_attr.replace('_levels', '_lev')):
            levels = getattr(state, alt, None)
            if levels is not None:
                break
    if levels is None:
        raise ValueError(f"Cannot find level populations for {line_name}. "
                         f"Expected state.{levels_attr} (set by Phase "
                         f"{'4' if species == 'HeII' else ('3' if species == 'HeI' else '2')} "
                         f"wrapper / production_runner hook).")
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 2:
        raise ValueError(f"state.{levels_attr} has ndim {levels.ndim}, "
                         f"expected 2 (n_levels, n_zones).")
    if idx_l >= levels.shape[0] or idx_u >= levels.shape[0]:
        raise ValueError(f"{line_name} needs level indices ({idx_l}, {idx_u}) "
                         f"but state.{levels_attr} has only {levels.shape[0]} levels. "
                         f"Re-run with higher nlte_levels.")
    n_lower = levels[idx_l]
    n_upper = levels[idx_u]

    # --- Per-zone Sobolev τ: read pre-computed (He) or compute on the fly (H) ---
    if species in ('HeI', 'HeII'):
        tau_zone = _find_tau(state, line_name, is_he2)
        if tau_zone is None:
            raise ValueError(f"Cannot find per-zone τ for {line_name}. "
                             f"Expected state.he{2 if is_he2 else 1}_tau.")
    else:   # 'HI'
        # H NLTE solver only stores τ for Lyα; compute Sobolev τ on the fly
        # for Balmer / Paschen using level populations and v(r) gradient.
        tau_zone = _compute_sobolev_tau_h(state, n_lower, n_upper,
                                            A_ul, g_l, g_u, lam0)

    # --- Line source function S_l (per zone) ---
    S_zone = _construct_source_function(n_lower, n_upper, g_l, g_u, lam0)

    # --- B_lu from A_ul (cgs Einstein relation) ---
    nu0 = C_LIGHT / (lam0 * 1e-8)
    H_PL = 6.62607015e-27
    B_lu = (g_u / g_l) * (C_LIGHT**2 / (2.0 * H_PL * nu0**3)) * A_ul

    return {
        'lambda_rest':   lam0,
        'tau_zone':      tau_zone,
        'S_zone':        S_zone,
        'n_lower_zone':  n_lower,
        'n_upper_zone':  n_upper,
        'A_ul':          A_ul,
        'B_lu':          B_lu,
        'g_l':           g_l,
        'g_u':           g_u,
        'is_he2':        is_he2,
        'species':       species,
    }


def _compute_sobolev_tau_h(state, n_lower, n_upper, A_ul, g_l, g_u, lam0_AA):
    """Compute per-zone Sobolev τ for a hydrogen line.

    τ_Sobolev = (πe²/m_e c) × f_lu × λ × n_l × (1 − g_l n_u / g_u n_l) / |dv/dr|

    The leading constant in cgs is σ₀ = 0.02654 cm² Hz; combined with λ
    (in cm) it gives the integrated line cross-section in cm²·cm·Hz/(cm/s).
    """
    # f_lu from A_ul (Einstein relation, same formula as HeLine)
    ME = 9.1093837015e-28
    EE = 4.8032068e-10
    prefac = ME * C_LIGHT / (8.0 * np.pi**2 * EE**2)   # ≈ 1.499e-7
    lam_cm = lam0_AA * 1e-8
    f_lu = prefac * lam_cm**2 * (g_u / g_l) * A_ul

    # Velocity gradient |dv/dr| per zone, finite difference from state.r, v
    r = np.asarray(getattr(state, 'r'), dtype=float)
    v = np.asarray(getattr(state, 'v'), dtype=float)
    if r.shape != v.shape:
        raise ValueError(f"r and v shape mismatch in _compute_sobolev_tau_h")
    dv_dr = np.gradient(v, r)
    dv_dr = np.abs(dv_dr)
    dv_dr = np.maximum(dv_dr, 1e-30)   # avoid /0 in stagnant zones

    # σ_lam = π e² / (m_e c) × f_lu (in cm² × Hz = cm²/s)
    SIGMA0 = np.pi * EE**2 / (ME * C_LIGHT)
    # Sobolev tau: σ_lam × f_lu × λ × (n_l - (g_l/g_u) n_u) / dv/dr
    n_diff = np.maximum(n_lower - (g_l / g_u) * n_upper, 0.0)
    tau = SIGMA0 * f_lu * lam_cm * n_diff / dv_dr
    return tau


def _find_tau(state, line_name: str, is_he2: bool):
    """Look up per-zone τ in any of several possible storage conventions."""
    # Convention A: state.he1_tau / state.he2_tau as dict (production wrapper)
    base = 'he2_tau' if is_he2 else 'he1_tau'
    obj = getattr(state, base, None)
    if isinstance(obj, dict) and line_name in obj:
        return np.asarray(obj[line_name], dtype=float)
    # Convention B: state.he1_diag['line_tau'][line_name]
    diag_name = 'he2_diag' if is_he2 else 'he1_diag'
    diag = getattr(state, diag_name, None)
    if isinstance(diag, dict):
        if 'line_tau' in diag and isinstance(diag['line_tau'], dict):
            if line_name in diag['line_tau']:
                return np.asarray(diag['line_tau'][line_name], dtype=float)
    return None


def _construct_source_function(n_l, n_u, g_l, g_u, lam0):
    """S_l = (2hc²/λ⁵) × 1 / [(g_u n_l) / (g_l n_u) − 1].

    Returns S in erg/s/cm²/AA/sr. Robust to underflow.
    """
    H_PL = 6.62607015e-27
    lam_cm = lam0 * 1e-8
    prefac = (2.0 * H_PL * C_LIGHT**2 / lam_cm**5) * 1e-8   # /AA
    n_u_safe = np.maximum(n_u, 1e-30)
    n_l_safe = np.maximum(n_l, 1e-30)
    denom = (g_u * n_l_safe) / (g_l * n_u_safe) - 1.0
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return prefac / denom


# ============================================================================
# Line-object factory for peel_pipeline_abs integration
# ============================================================================

class HeLine:
    """Lightweight line object compatible with peel_pipeline_abs / sn_mc_voigt_peel.

    Matches the HalphaLine convention (lam0_cm, nu0, f_lu, g_l, g_u attributes).
    Constructed on the fly per He line from HE_LINE_DATA, so the user does not
    need to add the 8 He lines to lines.LINE_LIB.
    """
    __slots__ = ('lam0_cm', 'nu0', 'f_lu', 'g_l', 'g_u', 'A_ul', 'name')

    def __init__(self, lam0_AA, A_ul, g_l, g_u, name=''):
        self.lam0_cm = lam0_AA * 1e-8
        self.nu0 = C_LIGHT / self.lam0_cm
        self.A_ul = A_ul
        self.g_l = g_l
        self.g_u = g_u
        self.name = name
        # f_lu from A_ul: A_ul = (8π²e²ν²/m_e c³) × (g_l/g_u) × f_lu
        # → f_lu = m_e c³ A_ul / (8π²e²ν²) × (g_u/g_l)
        # In cgs: m_e c / (8π²e²) ≈ 1.4992e-7 s ⇒ f_lu = λ²/2 × (g_u/g_l) × A_ul × (m_e c / 8π²e²)
        # Easier form: f_lu = (m_e c λ² / 8π²e²) × (g_u/g_l) × A_ul
        ME = 9.1093837015e-28
        EE = 4.8032068e-10
        prefac = ME * C_LIGHT / (8.0 * np.pi**2 * EE**2)   # ≈ 1.499e-7 cgs
        self.f_lu = prefac * self.lam0_cm**2 * (g_u / g_l) * A_ul


def make_he_line(line_name: str) -> HeLine:
    """Return a HeLine instance for one of the 8 named He lines."""
    if line_name not in LINE_CATALOG:
        raise ValueError(f"Unknown line: {line_name}")
    lam0, A_ul, g_l, g_u, _, _ = LINE_CATALOG[line_name]
    return HeLine(lam0, A_ul, g_l, g_u, name=line_name)


# ============================================================================
# peel_pipeline_abs adapter
# ============================================================================

def peel_pipeline_abs_callback(state, line_inputs: dict,
                                 lambda_grid: np.ndarray,
                                 n_packets: int = 50000,
                                 calibration: str = 'theoretical_ew',
                                 n_steps_peel: int = 40,
                                 max_events: int = 2000,
                                 verbose: bool = False) -> dict:
    """Adapter that calls peel_pipeline_abs.run_peel_pipeline_abs for one He line.

    This is the recommended production peel_callback for compute_phase5_spectra.
    It uses the existing MC kernel (the same one that handles Hα) for each
    He line, so behavior is consistent with the rest of the pipeline.

    Args:
        state: pipeline state with snap-like attributes (r, v, rho, T, n_e)
               and photosphere fields (R_phot_cm, T_phot, L_phot)
        line_inputs: output of extract_line_inputs() for the line
        lambda_grid: target wavelength grid (n_pix,)
        n_packets: MC packets
        calibration: 'f_cont_bb' (recommended for τ_es > 100 / STELLA),
                     'theoretical_ew', or 'absolute'
        n_steps_peel: peel-off step count (passed to kernel)
        max_events: max scatter events per packet
        verbose: pass-through to kernel

    Returns:
        {'F_lambda': flux on lambda_grid, 'L_line_emitted': line luminosity}

    Requires:
        peel_pipeline_abs module importable (i.e. running from the snline
        source tree where peel_pipeline_abs.py lives).
    """
    # Lazy import — only attempted when this callback is actually used
    try:
        from peel_pipeline_abs import run_peel_pipeline_abs
    except ImportError as e:
        raise ImportError(
            "peel_pipeline_abs not importable. Either (a) run from the snline "
            "source dir, or (b) use the reference Sobolev MC by passing "
            "peel_callback=None to compute_phase5_spectra.") from e

    lam0_AA = line_inputs['lambda_rest']
    A_ul = line_inputs['A_ul']
    H_PL = 6.62607015e-27
    nu0 = C_LIGHT / (lam0_AA * 1e-8)

    # Per-zone UNATTENUATED line emissivity (rate at which excited atoms decay
    # per unit volume per steradian, assuming all photons escape):
    # j_l (erg/s/cm³/sr) = (hν / 4π) × n_u × A_ul
    n_u = line_inputs['n_upper_zone']
    j_unatt = (H_PL * nu0 / (4.0 * np.pi)) * n_u * A_ul

    # CRITICAL: apply Sobolev escape probability β(τ) = (1 - exp(-τ)) / τ.
    # The kernel needs the ESCAPING luminosity, not the unattenuated volume
    # emission rate. For τ = 3e6 (He I 10830), only β ≈ 3e-7 of the local
    # decay luminosity actually escapes; the rest is reabsorbed locally.
    # Without this correction, L_line is over-estimated by 1/β (up to 10^7
    # for the thickest He I lines), the kernel saturates max_events trying
    # to attenuate, and F_norm blows up to absurd values.
    #
    # This matches how the Hα pipeline passes L_line (h_populations_nlte
    # already returns the escaping value via its proper RT treatment).
    tau_zone = line_inputs['tau_zone']
    tau_safe = np.maximum(tau_zone, 1e-30)
    tau_for_exp = np.minimum(tau_safe, 700.0)   # avoid exp underflow
    beta_zone = np.where(tau_zone > 1e-6,
                          (1.0 - np.exp(-tau_for_exp)) / tau_safe,
                          1.0 - 0.5 * tau_safe)

    # β-corrected per-zone emissivity (what actually escapes locally)
    emissivity = j_unatt * beta_zone

    # Volume integral → total escaping line luminosity
    r = np.asarray(state.r, dtype=float)
    r_edge = np.zeros(len(r) + 1)
    r_edge[1:-1] = 0.5 * (r[:-1] + r[1:])
    r_edge[0] = max(r[0] - (r[1] - r[0]) / 2.0, 1e-30)
    r_edge[-1] = r[-1] + (r[-1] - r[-2]) / 2.0
    V_zone = (4.0 / 3.0) * np.pi * (r_edge[1:]**3 - r_edge[:-1]**3)
    L_line = float(np.sum(4.0 * np.pi * emissivity * V_zone))

    # Build snap dict (the kernel expects these keys)
    snap = {
        'r':   np.asarray(state.r, dtype=float),
        'v':   np.asarray(state.v, dtype=float),
        'rho': np.asarray(state.rho, dtype=float),
        'T':   np.asarray(state.T, dtype=float),
        'n_e': np.asarray(state.n_e, dtype=float),
    }

    # Build params dict
    R_phot = _safe_get(state, 'R_phot_cm', _safe_get(state, 'R_phot', 1e15))
    T_phot = _safe_get(state, 'T_phot', _safe_get(state, 'T_color', 1e4))
    L_phot = _safe_get(state, 'L_phot', None)
    v_turb_kms = _safe_get(state, 'v_turb_kms', 20.0)
    v_turb_grid = _safe_get(state, 'v_turb_kms_grid', None)

    params = {
        'R_phot':            float(R_phot),
        'T_phot':            float(T_phot),
        'v_turb_kms':        float(v_turb_kms),
        'L_line':            L_line,
        'n_lower':           line_inputs['n_lower_zone'],
        'n_upper':           line_inputs['n_upper_zone'],
        'emissivity':        emissivity,
        # f_vol MUST be present and >= 1e-6 or peel_pipeline_abs silently
        # skips the volumetric channel (line 229 of that module), giving
        # a continuum-only spectrum with F_norm ≡ 1 everywhere. The fraction
        # itself doesn't affect per-packet weighting (which is set by
        # L_line/L_cont_band internally); it's just a gate.
        'f_vol':             1.0,
    }
    if L_phot is not None:
        params['L_phot'] = float(L_phot)
    if v_turb_grid is not None:
        params['v_turb_kms_grid'] = np.asarray(v_turb_grid, dtype=float)

    # Build line object on the fly (HalphaLine-compatible)
    he_line = make_he_line(_line_name_from_inputs(line_inputs))

    # Output wavelength window: matched to lambda_grid
    band_AA = (float(lambda_grid[0]), float(lambda_grid[-1]))
    nbins = len(lambda_grid)

    # Call the existing MC kernel
    lam_out, F_norm, F_cont_abs, F_total_abs, info = run_peel_pipeline_abs(
        snap, params, line=he_line,
        n_packets=n_packets, nbins=nbins,
        band_AA=band_AA, source_padding_AA=500.0,
        v_turb_kms=v_turb_kms,
        n_steps_peel=n_steps_peel,
        calibration=calibration,
        max_events=max_events,
        cont_line_emission=True,
        line_scatter_redistribution='aa_prd',
        seed=42, verbose=verbose)

    # CRITICAL: the kernel reports F_total_abs and F_cont_abs in
    # erg/s/cm²/AA at D=1 cm (kernel uses D=1 internally; see peel_pipeline_abs
    # line 156). My phase5_continuum.F_cont_lambda is at R=R_phot. Mixing
    # them gives a (R_phot)² ≈ 3e+30 normalization error.
    #
    # The kernel also returns F_norm directly, which is dimensionless and
    # is the actual F/F_cont a distant observer sees (it normalizes using
    # the kernel's own self-consistent F_cont_baseline). We pass F_norm
    # back, and the caller (compute_phase5_spectra) uses it directly when
    # present, bypassing the phase5_continuum baseline divide.
    F_norm_resampled = np.interp(lambda_grid, lam_out, F_norm,
                                   left=1.0, right=1.0)
    # F_line_only in the kernel's units (D=1 cm) — for absolute-units
    # cross-checks only; not used in normalization.
    F_line_only = F_total_abs - F_cont_abs
    F_line_resampled = np.interp(lambda_grid, lam_out, F_line_only,
                                   left=0.0, right=0.0)

    return {
        'F_lambda':       F_line_resampled,   # at D=1 cm, kernel units
        'F_norm':         F_norm_resampled,   # dimensionless, ready to plot
        'L_line_emitted': L_line,
        'F_total_native': F_total_abs,
        'F_cont_native':  F_cont_abs,
        'lam_native':     lam_out,
    }


def _safe_get(obj, name, default):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _line_name_from_inputs(line_inputs: dict) -> str:
    """Recover the canonical name from line_inputs by matching lambda_rest."""
    lam = line_inputs['lambda_rest']
    for name, data in LINE_CATALOG.items():
        if abs(data[0] - lam) < 0.5:
            return name
    return 'UNKNOWN'


# ============================================================================
# Layer 2: reference MC driver (pure-NumPy Sobolev peel-off)
# ============================================================================

def reference_sobolev_peel_off(state, line_inputs: dict,
                                 lambda_grid: np.ndarray,
                                 n_packets: int = 50000,
                                 rng: Optional[np.random.Generator] = None,
                                 emit_mode: str = 'volumetric') -> dict:
    """Reference Sobolev peel-off MC for a single line.

    For each emission zone:
      1. Emit n_packets isotropically with luminosity = 4πj_l × V_zone
         where j_l = (h ν / 4π) × n_u × A_ul (line emissivity)
      2. For each packet, choose direction μ = cos(θ) ∈ [-1, 1].
      3. The packet's observer-frame wavelength is
            λ_obs = λ_rest × (1 + μ v(r)/c)
      4. Optical depth to escape along that direction:
            τ_los = Σ over zones intersected, weighted by Sobolev τ at each
         (Approximation: use the local zone τ along the line of sight; for
         shock_sanity-flag epochs this is brittle but for monotonic v(r)
         it's exact.)
      5. Contribute  L_packet × exp(−τ_los)  to the F_λ bin nearest λ_obs.

    NOT a substitute for the user's existing peel-off kernel (which includes
    aa_prd redistribution, n_steps_peel, etc.). This is a validation tool.

    Args:
        state: pipeline state (need r, v, rho, T arrays in zone structure)
        line_inputs: output of extract_line_inputs() for this line
        lambda_grid: wavelength grid to evaluate F_λ on (n_pix,)
        n_packets: total packets across all zones
        emit_mode: 'volumetric' (per-zone emissivity) or 'photosphere' (surface)

    Returns:
        dict with 'F_lambda' and 'L_line_emitted' (cross-check vs NLTE).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    r = np.asarray(state.r, dtype=float)
    v = np.asarray(state.v, dtype=float)
    rho = np.asarray(state.rho, dtype=float) if hasattr(state, 'rho') else None
    n_zones = len(r)

    tau_zone = line_inputs['tau_zone']
    S_zone = line_inputs['S_zone']
    n_u = line_inputs['n_upper_zone']
    A_ul = line_inputs['A_ul']
    lam0 = line_inputs['lambda_rest']

    H_PL = 6.62607015e-27
    nu0 = C_LIGHT / (lam0 * 1e-8)
    # Per-zone line emissivity j_l = (hν / 4π) × n_u × A_ul  in erg/s/cm³/sr
    j_l = (H_PL * nu0 / (4.0 * np.pi)) * n_u * A_ul

    # Per-zone volume (shells)
    r_edge = np.zeros(n_zones + 1)
    r_edge[1:-1] = 0.5 * (r[:-1] + r[1:])
    r_edge[0] = r[0] - (r[1] - r[0]) / 2.0
    r_edge[-1] = r[-1] + (r[-1] - r[-2]) / 2.0
    V_zone = (4.0 / 3.0) * np.pi * (r_edge[1:]**3 - r_edge[:-1]**3)

    # Per-zone line luminosity (sr-integrated): 4π j_l V
    L_zone = 4.0 * np.pi * j_l * V_zone   # erg/s per zone
    L_line_total = L_zone.sum()

    if L_line_total <= 0:
        # No emission — return zero spectrum
        return {'F_lambda': np.zeros_like(lambda_grid),
                'L_line_emitted': 0.0}

    # Distribute packets by L_zone weighting
    pkt_per_zone = np.round(n_packets * L_zone / L_line_total).astype(int)
    pkt_per_zone = np.maximum(pkt_per_zone, 1) * (L_zone > 0)
    total_pkts = pkt_per_zone.sum()
    L_per_packet_zone = L_zone / np.maximum(pkt_per_zone, 1)

    F_lambda = np.zeros_like(lambda_grid)
    dlam = lambda_grid[1] - lambda_grid[0]

    for iz in range(n_zones):
        if pkt_per_zone[iz] == 0:
            continue
        n_p = pkt_per_zone[iz]
        # Sample direction cosines uniformly in [-1, 1]
        mu = rng.uniform(-1.0, 1.0, n_p)
        v_los = mu * v[iz]   # cm/s
        # Observer-frame wavelength (Doppler from emitting zone toward observer)
        lam_obs = lam0 * (1.0 + v_los / C_LIGHT)
        # Sobolev escape probability β = (1 - exp(-τ)) / τ
        # For monotonic v(r) the photon resonates once (at its emission point);
        # after that the velocity gradient Doppler-shifts it out of resonance
        # so no further line absorption occurs. β accounts for the local
        # thermalization within the resonance zone.
        t = tau_zone[iz]
        if t > 1e-6:
            beta = (1.0 - np.exp(-t)) / t      # standard Sobolev β
        else:
            beta = 1.0 - 0.5 * t                # series expansion for thin
        # Per-packet contribution: emission rate × escape probability
        L_pkt = np.full(n_p, L_per_packet_zone[iz] * beta)   # erg/s
        # Bin into lambda grid
        idx = np.floor((lam_obs - lambda_grid[0]) / dlam).astype(int)
        valid = (idx >= 0) & (idx < len(lambda_grid))
        np.add.at(F_lambda, idx[valid], L_pkt[valid])

    # Convert binned L (erg/s per pixel) → F_λ (erg/s/cm²/AA at R_phot surface)
    # Per-pixel L_pkt is in erg/s; divide by 4π R_phot² × dλ to get F_λ
    R_phot = getattr(state, 'R_phot_cm', None) or getattr(state, 'R_phot', 1e15)
    F_lambda = F_lambda / (4.0 * np.pi * R_phot**2 * dlam)

    return {
        'F_lambda': F_lambda,
        'L_line_emitted': L_line_total,
    }


# ============================================================================
# Layer 3: top-level driver
# ============================================================================

def _formal_line_result(state, line_name, line_inputs, lam_grid, F_cont_lam,
                        electron_scatter=True):
    """Compute one line's emergent profile via the Sobolev source-function
    formal solution (formal_line_profile), returning a result dict compatible
    with the MC peel callback ('F_lambda', 'F_norm', 'L_line_emitted').

    Each line uses its own n_lower/n_upper and atomic data; the scattering
    source S_L = (1-eps) J_bar + eps B is built from the diluted continuum at
    that line's wavelength. No Hα anchoring, no empirical R-factor.
    """
    lam0 = float(line_inputs['lambda_rest'])
    dv = (np.asarray(lam_grid, float) / lam0 - 1.0) * (C_LIGHT / 1e5)  # km/s
    R_phot = float(getattr(state, 'R_phot_cm',
                           np.asarray(state.r, float)[0]))
    T_phot = float(getattr(state, 'T_phot', 6000.0))
    tau_es_env = float(getattr(state, 'tau_es_total', 0.0) or 0.0)
    res = _flp_p5.profile_for_line(
        line_name, lam0, float(line_inputs['A_ul']),
        float(line_inputs['g_l']), float(line_inputs['g_u']),
        np.asarray(state.r, float), np.asarray(state.v, float),
        np.asarray(state.T, float), np.asarray(state.n_e, float),
        np.asarray(line_inputs['n_lower_zone'], float),
        np.asarray(line_inputs['n_upper_zone'], float),
        R_phot=R_phot, T_phot=T_phot, vgrid_kms=dv,
        tau_es_env=tau_es_env, electron_scatter=electron_scatter)
    F_norm = np.asarray(res['F_norm'], float)
    return {'F_lambda': F_norm * np.asarray(F_cont_lam, float),
            'F_norm': F_norm,
            'L_line_emitted': float(res['L_line'])}


def compute_phase5_spectra(state,
                            lines: Optional[list] = None,
                            n_packets: int = 50000,
                            win_kms: float = 5000.0,
                            n_pix: int = 501,
                            peel_callback: Optional[Callable] = None,
                            include_es_wings: bool = False,
                            profile_method: str = 'mc',
                            verbose: bool = True) -> dict:
    """Compute synthetic F_λ profiles for the 8 He lines.

    Args:
        state: pipeline state with state.he1_diag and state.he2_diag populated
        lines: list of line names to compute (default: all 8)
        n_packets: MC packets per line
        win_kms: ± velocity window per line in km/s
        n_pix: pixels per line
        peel_callback: optional callback to user's existing peel-off kernel.
            Signature: peel_callback(state, line_inputs, lambda_grid,
                                       n_packets) -> {'F_lambda': array}
            If None, uses the reference Sobolev MC in this module.
        include_es_wings: pass-through to continuum module (Phase 5b)
        verbose: print per-line diagnostics

    Returns:
        spectra dict (see module docstring)
    """
    if lines is None:
        lines = list(HE_LINES_AA.keys())

    # Build per-band continuum once
    cont = phase5_continuum.build_per_band_continuum(
        state, line_wavelengths={ln: HE_LINES_AA[ln] for ln in lines},
        win_kms=win_kms, n_pix=n_pix,
        include_es_wings=include_es_wings, verbose=verbose)

    spectra = {}
    if verbose:
        print(f"\n[phase5] Computing {len(lines)} line profiles "
              f"({n_packets} packets each)")

    for line_name in lines:
        try:
            line_inputs = extract_line_inputs(state, line_name)
        except (ValueError, NotImplementedError) as e:
            if verbose:
                print(f"[phase5] {line_name}: SKIPPED ({e})")
            continue

        c = cont[line_name]
        lam_grid = c['lambda_AA']
        F_cont_lam = c['F_cont_lambda']

        # Run MC (either user kernel or reference) — or the source-function
        # formal solution (profile_method='formal'), which gives each line its
        # OWN rest-peaked P-Cygni and luminosity from its own S_L and tau_S,
        # with no Hα anchoring.
        if profile_method == 'formal' and _HAVE_FORMAL_P5:
            result = _formal_line_result(state, line_name, line_inputs,
                                         lam_grid, F_cont_lam)
        elif peel_callback is not None:
            result = peel_callback(state, line_inputs, lam_grid, n_packets)
        else:
            result = reference_sobolev_peel_off(
                state, line_inputs, lam_grid, n_packets=n_packets)

        F_line = result['F_lambda']
        # Use kernel-supplied F_norm if available (peel_pipeline_abs_callback
        # returns this; it's the dimensionless F/F_cont the kernel computes
        # using its OWN self-consistent baseline at D=1 cm — avoids the
        # R_phot² unit-mismatch between kernel-units F_line and our R_phot
        # phase5_continuum F_cont).
        if 'F_norm' in result and result['F_norm'] is not None:
            F_norm = np.asarray(result['F_norm'], dtype=float)
            # Build a self-consistent F_total in phase5_continuum's units
            # (R_phot, erg/s/cm²/AA) for users who want absolute flux:
            #   F_total_at_Rphot = F_norm × F_cont_at_Rphot
            F_total = F_norm * F_cont_lam
        else:
            # Reference Sobolev MC path: F_line is already in R_phot units
            # (same as F_cont_lam from phase5_continuum), so this works.
            F_total = F_cont_lam + F_line
            F_norm = F_total / np.maximum(F_cont_lam, 1e-30)

        # Equivalent width (negative for emission convention)
        dlam = lam_grid[1] - lam_grid[0]
        EW = -np.sum((F_norm - 1.0)) * dlam   # AA

        tau_med = float(np.median(line_inputs['tau_zone']))
        spectra[line_name] = {
            'lambda_AA':     lam_grid,
            'F_lambda':      F_total,
            'F_line_only':   F_line,
            'F_cont_lambda': F_cont_lam,
            'F_norm':        F_norm,
            'L_line':        result.get('L_line_emitted', 0.0),
            'L_cont_band':   c['L_cont_band'],
            'EW':            EW,
            'lambda_rest':   c['lambda_rest'],
            'tau_med':       tau_med,
        }
        if verbose:
            peak = F_norm.max()
            peak_v = (lam_grid[F_norm.argmax()] - c['lambda_rest']) / c['lambda_rest'] * C_LIGHT / 1e5
            print(f"[phase5] {line_name:<12s}: τ_med={tau_med:.2e}, "
                  f"L_line={result.get('L_line_emitted',0):.3e}, "
                  f"EW={EW:>+7.2f} AA, peak F={peak:.2f} @ {peak_v:+.0f} km/s")

    return spectra


# ============================================================================
# Self-test with synthetic data
# ============================================================================

if __name__ == "__main__":
    import types

    print("\nPhase 5 multi-line driver self-test")
    print("="*78)
    print("Construct a synthetic IIn-like state with hand-set He populations,")
    print("then verify that the reference MC reproduces sensible profiles.\n")

    # Synthetic state: 50-zone IIn-like CSM
    n_z = 50
    r = np.linspace(2e14, 1.7e15, n_z)
    v = np.linspace(300e5, 5000e5, n_z)
    rho = 1e-13 * (2e14 / r)**2
    T = np.full(n_z, 1e4)
    n_e = 1e10 * (2e14 / r)**2

    # Synthetic He I populations. Target: L_line(10830) ~ 1e+40 erg/s after
    # β ~ 3e-7 — i.e. volume-integrated 4π j V ~ 3e+46 → n_u × V_total ~ 1.7e+51.
    # With V_total ~ 2e+46 cm³ for this synthetic state, that gives
    # n_u_avg ~ 8e+4 cm⁻³ for the He I 10830 upper level (2³P).
    # The other levels are scaled to give EW ratios of order unity for the
    # strongest lines (10830 > 5876 > 7065 > 6678).
    n_2_3S_target = 4e+7    # 2³S metastable, populated by He II recomb
    n_2_3P_target = 1e+5    # 2³P, upper level of 10830
    n_3_3D_target = 5e+4    # 3³D, upper level of 5876
    n_3_3S_target = 1e+4    # 3³S, upper level of 7065
    n_2_1P_target = 5e+3    # 2¹P
    n_3_1D_target = 1e+3    # 3¹D, upper level of 6678

    # Per-zone scaling: drop as r⁻² (proportional to density)
    r_scale = (r[0] / r)**2
    n_2_3S = n_2_3S_target * r_scale
    n_2_3P = n_2_3P_target * r_scale
    n_3_3D = n_3_3D_target * r_scale
    n_3_3S = n_3_3S_target * r_scale
    n_2_1P = n_2_1P_target * r_scale
    n_3_1D = n_3_1D_target * r_scale

    # Per-line Sobolev τ (these are typical day-0.5 IIn values, ~3e+06 for 10830)
    line_tau = {
        'He_I_10830':  3.1e+06 * np.ones(n_z),
        'He_I_5876':   1.8e+06 * np.ones(n_z),
        'He_I_7065':   2.5e+05 * np.ones(n_z),
        'He_I_6678':   1.4e-03 * np.ones(n_z),
    }
    line_n_lower = {
        'He_I_10830': n_2_3S,
        'He_I_5876':  n_2_3P,
        'He_I_7065':  n_2_3P,
        'He_I_6678':  n_2_1P,
    }
    line_n_upper = {
        'He_I_10830': n_2_3P,
        'He_I_5876':  n_3_3D,
        'He_I_7065':  n_3_3S,
        'He_I_6678':  n_3_1D,
    }

    # He II analogues — scaled to give realistic L_line for He II 4686
    n_HeII_lvl_2_target = 1e+5
    n_HeII_lvl_3_target = 5e+4
    n_HeII_lvl_4_target = 1e+4
    n_HeII_lvl_5_target = 1e+3
    n_HeII_lvl_2 = n_HeII_lvl_2_target * r_scale
    n_HeII_lvl_3 = n_HeII_lvl_3_target * r_scale
    n_HeII_lvl_4 = n_HeII_lvl_4_target * r_scale
    n_HeII_lvl_5 = n_HeII_lvl_5_target * r_scale

    line_tau_he2 = {
        'He_II_1640':  5.2e+04 * np.ones(n_z),
        'He_II_4686':  8.8e+01 * np.ones(n_z),
        'He_II_3203':  9.5e+00 * np.ones(n_z),
        'He_II_10124': 1.1e+00 * np.ones(n_z),
    }
    line_n_lower_he2 = {
        'He_II_1640':  n_HeII_lvl_2,
        'He_II_4686':  n_HeII_lvl_3,
        'He_II_3203':  n_HeII_lvl_3,
        'He_II_10124': n_HeII_lvl_4,
    }
    line_n_upper_he2 = {
        'He_II_1640':  n_HeII_lvl_3,
        'He_II_4686':  n_HeII_lvl_4,
        'He_II_3203':  n_HeII_lvl_5,
        'He_II_10124': n_HeII_lvl_5,
    }

    state = types.SimpleNamespace()
    state.r = r; state.v = v; state.rho = rho; state.T = T; state.n_e = n_e
    state.T_phot = 14000.0
    state.R_phot_cm = 1.7e15
    state.L_phot = 4e42
    state.tau_es_total = 25.0
    state.he1_diag = {'line_tau': line_tau,
                       'line_n_lower': line_n_lower,
                       'line_n_upper': line_n_upper}
    state.he2_diag = {'line_tau': line_tau_he2,
                       'line_n_lower': line_n_lower_he2,
                       'line_n_upper': line_n_upper_he2}

    spectra = compute_phase5_spectra(state, n_packets=20000, verbose=True)

    print(f"\n[phase5] Generated {len(spectra)} line spectra:")
    for name, sp in spectra.items():
        print(f"   {name:<12s}  λ_grid: [{sp['lambda_AA'][0]:.1f}, "
              f"{sp['lambda_AA'][-1]:.1f}]  ({len(sp['lambda_AA'])} pix), "
              f"max F_norm = {sp['F_norm'].max():.2f}")
