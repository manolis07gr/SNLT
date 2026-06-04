"""
phase5_continuum.py — Per-band continuum for synthetic He line profiles
================================================================================

For each of the 8 He lines (4 He I + 4 He II), the MC kernel needs:
  (a) L_cont_src — the continuum luminosity in the source band, in erg/s,
      to set the per-packet luminosity for continuum-emission packets.
  (b) F_cont_baseline(λ) — the per-pixel continuum F_λ, in erg/s/cm²/AA, used
      to normalize the synthetic line profile F_norm = F_λ / F_cont(λ).
  (c) lambda_grid — the per-line wavelength grid the peel-off will populate.

This module computes those three quantities at each He line's rest wavelength,
using a diluted-blackbody continuum (the same model the rest of the pipeline
uses for Hα) but evaluated at the LINE'S wavelength rather than at the Hα
band. The factor-of-7 wavelength span across the 8 He lines (1640 Å in the UV
to 10830 Å in the NIR) means that the Wien suppression at 1640 and the
Rayleigh-Jeans growth at 10830 must be handled correctly per-line.

Optional refinements (controllable via include_es_wings flag):
  - Electron scattering wing broadening: a Lorentzian wing around the BB
    photosphere that reduces F_cont(λ) near the edges of the per-line window.
    Computed using the τ_es column already in state.tau_es_total.

This module is designed to interoperate with the existing photosphere_v2
output (T_phot, R_phot, L_phot, τ_es_total) WITHOUT modifying that module.
It reads what photosphere_v2 produces and extends it with per-band evaluation.

API:
  cont_dict = build_per_band_continuum(state, line_wavelengths,
                                        win_kms=5000.0, n_pix=501,
                                        include_es_wings=False)

  Returns: dict keyed by line name with sub-dict
    {'lambda_AA': array, 'F_cont_lambda': array (erg/s/cm²/AA),
     'L_cont_band': float (erg/s), 'baseline_far_wing': float}
"""
from __future__ import annotations
import numpy as np

# Physical constants (cgs)
H_PLANCK   = 6.62607015e-27
C_LIGHT    = 2.99792458e+10
K_BOLTZ    = 1.380649e-16

# Line rest wavelengths (Å in vacuum). Includes the 8 He lines (Phase 5
# original target) and 5 H I Balmer/Paschen lines (Phase 5 H extension).
# Hδ (n=6) and Pγ (n=6) require nlte_levels ≥ 6 in the H NLTE solver, so
# they are intentionally omitted under the default nlte_levels=5.
HE_LINES_AA = {
    'He_II_1640':  1640.42,    # 3 → 2 (UV; analog of Lyα)
    'He_II_3203':  3203.10,    # 5 → 3
    'He_II_4686':  4685.71,    # 4 → 3 (classic IIn diagnostic)
    'He_II_10124': 10123.6,    # 5 → 4
    'He_I_5876':   5875.62,    # 2³P → 3³D (D₃)
    'He_I_6678':   6678.15,    # 2¹P → 3¹D
    'He_I_7065':   7065.19,    # 2³P → 3³S
    'He_I_10830':  10830.34,   # 2³S → 2³P (the metastable trap line)
    # H Balmer
    'Halpha':      6562.81,    # n=2 → 3
    'Hbeta':       4861.33,    # n=2 → 4
    'Hgamma':      4340.47,    # n=2 → 5
    # H Paschen (NIR)
    'Palpha':     18751.01,    # n=3 → 4
    'Pbeta':      12818.08,    # n=3 → 5
}
# Backward-compat alias; new code should use LINES_AA below.
LINES_AA = HE_LINES_AA


def planck_lambda(T_kelvin: float, lambda_AA: np.ndarray) -> np.ndarray:
    """Planck function B_λ(T) in erg/s/cm²/AA/sr.

    Args:
        T_kelvin: blackbody temperature in K
        lambda_AA: wavelength array in Angstrom (any shape)

    Returns:
        B_λ in erg/s/cm²/AA/sr, same shape as lambda_AA
    """
    lambda_cm = lambda_AA * 1e-8   # convert to cm
    # B_λ = (2hc² / λ⁵) × 1 / (exp(hc/λkT) - 1), in erg/s/cm²/cm/sr
    # Multiply by 1e-8 to convert /cm to /AA
    x = H_PLANCK * C_LIGHT / (lambda_cm * K_BOLTZ * T_kelvin)
    # Numerical safety: clip exponent to avoid overflow
    x = np.clip(x, 1e-30, 700.0)
    B_per_cm = (2.0 * H_PLANCK * C_LIGHT**2 / lambda_cm**5) / np.expm1(x)
    return B_per_cm * 1e-8     # erg/s/cm²/AA/sr


def diluted_bb_F_lambda(T_kelvin: float, R_phot_cm: float,
                         lambda_AA: np.ndarray,
                         dilution_W: float = 0.5) -> np.ndarray:
    """Source-frame (4πR² × π B_λ) flux per Angstrom, evaluated at distance R.

    This is the F_λ that an observer at the photosphere surface (R = R_phot)
    sees from a diluted blackbody photosphere. It matches the convention used
    by the rest of the snline pipeline for setting the continuum baseline
    in the peel-off output.

    Returns: F_λ in erg/s/cm²/AA at R = R_phot.
    """
    B_lam = planck_lambda(T_kelvin, lambda_AA)   # erg/s/cm²/AA/sr
    # Emergent flux at the surface of a diluted-BB photosphere
    # F_λ = W × π × B_λ × (R_phot/R_obs)²  with R_obs = R_phot here
    return dilution_W * np.pi * B_lam


def build_per_band_continuum(state,
                              line_wavelengths: dict = None,
                              win_kms: float = 5000.0,
                              n_pix: int = 501,
                              dilution_W: float = 0.5,
                              include_es_wings: bool = False,
                              verbose: bool = False) -> dict:
    """Compute per-band continuum L_cont and F_cont for each He line.

    Args:
        state: pipeline state object with the following required attributes:
            T_phot     — photosphere color temperature (K)
            R_phot_cm  — photosphere radius (cm)
            L_phot     — bolometric photosphere luminosity (erg/s)
            [optional] tau_es_total — electron scattering optical depth
        line_wavelengths: dict mapping line name → λ_rest (Å). If None,
            uses HE_LINES_AA for all 8 He lines.
        win_kms: ± velocity window per line, in km/s (default 5000)
        n_pix: number of wavelength pixels per line (default 501 → ~20 km/s)
        dilution_W: BB dilution factor (default 0.5, geometric for opt-thick)
        include_es_wings: add electron-scattering Lorentzian wings (Phase 5b)
        verbose: print per-line diagnostics

    Returns:
        dict { line_name: {'lambda_AA': array,           # (n_pix,)
                            'F_cont_lambda': array,       # (n_pix,) erg/s/cm²/AA
                            'L_cont_band': float,         # erg/s
                            'baseline_far_wing': float,   # erg/s/cm²/AA
                            'T_phot_used': float,
                            'lambda_rest': float} }
    """
    if line_wavelengths is None:
        line_wavelengths = HE_LINES_AA

    # Pull required quantities from state
    T_phot = _get_attr(state, 'T_phot', None)
    R_phot = _get_attr(state, 'R_phot_cm', None)
    if T_phot is None or R_phot is None:
        # Fallbacks: try alternate naming
        T_phot = _get_attr(state, 'T_color', T_phot)
        R_phot = _get_attr(state, 'R_phot', R_phot)
    if T_phot is None or R_phot is None:
        raise ValueError("state must have T_phot and R_phot_cm (or T_color, R_phot)")

    if verbose:
        print(f"[phase5_cont] T_phot = {T_phot:.0f} K, R_phot = {R_phot:.3e} cm, "
              f"W = {dilution_W:.2f}")
        print(f"[phase5_cont] {'line':<12s} {'λ_rest':>10s} {'L_cont_band':>14s} "
              f"{'F_cont(λ_0)':>14s}  x=hc/λkT")
        print(f"[phase5_cont] {'-'*12} {'-'*10} {'-'*14} {'-'*14}  ------")

    results = {}
    for name, lam0 in line_wavelengths.items():
        # Build per-line wavelength grid: ±win_kms in line-of-sight Doppler
        dlam_max = lam0 * win_kms * 1e5 / C_LIGHT   # AA
        lam_grid = np.linspace(lam0 - dlam_max, lam0 + dlam_max, n_pix)

        # Per-pixel F_λ from diluted BB
        F_lam = diluted_bb_F_lambda(T_phot, R_phot, lam_grid, dilution_W)

        # Optional electron-scattering wing broadening (Phase 5b)
        if include_es_wings:
            tau_es = _get_attr(state, 'tau_es_total', 0.0)
            F_lam = _apply_es_wing(F_lam, lam_grid, lam0, tau_es, T_phot)

        # Band-integrated luminosity: L_cont = 4π R² × ∫ F_λ dλ over the band
        # (this matches the pipeline's existing per-band L_cont_src convention)
        dlam = lam_grid[1] - lam_grid[0]
        L_cont_band = 4.0 * np.pi * R_phot**2 * np.sum(F_lam) * dlam

        # The "far-wing" baseline used by peel-off normalization in the
        # current pipeline corresponds to F_λ evaluated near the band edge,
        # well outside the line core. With pure BB this is just F_λ(λ_0)
        # (BB is locally flat over ±5000 km/s for most He lines).
        F_baseline = float(F_lam[0])  # take the blue edge as the far-wing value

        # Wien diagnostic
        x_wien = H_PLANCK * C_LIGHT / (lam0 * 1e-8 * K_BOLTZ * T_phot)

        results[name] = {
            'lambda_AA':         lam_grid,
            'F_cont_lambda':     F_lam,
            'L_cont_band':       L_cont_band,
            'baseline_far_wing': F_baseline,
            'T_phot_used':       T_phot,
            'lambda_rest':       lam0,
            'x_wien':            x_wien,
        }
        if verbose:
            print(f"[phase5_cont] {name:<12s} {lam0:>10.1f} "
                  f"{L_cont_band:>14.3e} {F_lam[n_pix//2]:>14.3e}  "
                  f"x = {x_wien:5.2f}")

    return results


def _apply_es_wing(F_lam: np.ndarray, lam_grid: np.ndarray, lam0: float,
                    tau_es: float, T_phot: float) -> np.ndarray:
    """Electron scattering wing broadening of the BB continuum.

    Simple model: convolve F_λ with a Lorentzian whose width is set by the
    electron thermal velocity v_th = sqrt(kT/m_e) at T_phot, multiplied by
    sqrt(τ_es) to account for multiple scattering. For τ_es < 1 the wing is
    a small perturbation; for τ_es ≳ a few (typical SN photosphere) it
    broadens F_cont significantly near the line center.

    This is a known-good approximation for stellar/SN atmospheres but isn't
    rigorous — for paper-grade per-band continuum at the level of CMFGEN
    comparisons, the full Compton redistribution kernel would be needed.
    Provided here as an opt-in (Phase 5b).
    """
    if tau_es < 0.1:
        return F_lam   # no significant broadening

    M_E = 9.1093837015e-28   # electron mass (g)
    v_th = np.sqrt(K_BOLTZ * T_phot / M_E)   # cm/s (typically ~5500 km/s at T=1e4)
    sigma_v = v_th * np.sqrt(max(tau_es, 1.0))   # multiply by sqrt(τ_es)
    sigma_lam = lam0 * sigma_v / C_LIGHT          # in same units as lam_grid (AA)

    # Lorentzian kernel (FWHM = 2 sigma_lam for this approximation)
    half_width = 5.0 * sigma_lam
    kern_grid = np.arange(-half_width, half_width + 1e-9,
                           lam_grid[1] - lam_grid[0])
    kernel = (sigma_lam / np.pi) / (kern_grid**2 + sigma_lam**2)
    kernel /= kernel.sum()
    F_smooth = np.convolve(F_lam, kernel, mode='same')
    return F_smooth


def _get_attr(obj, name, default=None):
    """Safe getattr that handles SimpleNamespace, dict, and None."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ============================================================================
# Self-test: verify Wien/RJ scaling across the 8 He lines
# ============================================================================
if __name__ == "__main__":
    import types

    print("\nPhase 5 continuum self-test")
    print("="*78)
    print("Two regimes: day 0.5 (T_phot ≈ 14000 K, IIn) and day 040 (T_phot ≈ 5300 K, plateau)")
    print()

    for T, R, label in [(14000.0, 1.7e15, "day 0.5 (hot, IIn)"),
                         (5300.0, 1.6e15, "day 040 (cool, plateau)")]:
        print(f"--- {label} ---")
        state = types.SimpleNamespace(T_phot=T, R_phot_cm=R, L_phot=1e42,
                                       tau_es_total=2.0)
        cont = build_per_band_continuum(state, verbose=True, win_kms=5000.0,
                                         n_pix=501, dilution_W=0.5)
        print()

        # Quick sanity check: F_cont at 1640 vs 10830 ratio should track BB
        F_1640 = cont['He_II_1640']['baseline_far_wing']
        F_10830 = cont['He_I_10830']['baseline_far_wing']
        print(f"  F_cont(1640) / F_cont(10830) = {F_1640/F_10830:.3e}")
        print(f"  Expected from B_λ ratio:       "
              f"{(planck_lambda(T, np.array([1640.0]))[0] / planck_lambda(T, np.array([10830.0]))[0]):.3e}")
        print()
