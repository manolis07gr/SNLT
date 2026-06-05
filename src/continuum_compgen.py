"""
continuum_compgen.py — Composition-general continuum guard + He-budget diagnostics
================================================================================

FUTURE_WORK P1 item 4.  The per-band continuum (phase5_continuum.build_per_band_
continuum) is a pure diluted blackbody L_cont = 4π R² ∫ W π B_λ(T_phot) dλ. It was
built and validated for H-rich gas, where the τ_cont = 2/3 photosphere sits at the
warm H-recombination front. For H-FREE (He-rich, C-series) ejecta the continuum
opacity is Thomson-only, so photosphere_v2 places the photosphere deep in the cold
dense CSM: T_phot drops to ~2000-4000 K and R_phot shrinks, and L_cont_band
collapses unphysically (Wien suppression of B_λ × small R²). That collapsed
L_cont_band then corrupts the EW estimate L_corr / L_cont_band (the quantity the
trust ceiling says to prefer for thick-line EWs).

This module provides a COMPOSITION-GENERAL, energy-conserving guard that does NOT
assume hydrogen and does NOT depend on re-deriving the photosphere:

  1. Composition switch  — mean X_H classifier (H-free below x_h_thresh = 1e-3).
  2. Continuum-collapse detector — the diluted-BB photosphere cannot radiate more
     than L_phot, so if its bolometric 4π R² σ T_phot⁴ falls far below the model's
     L_phot the (T_phot, R_phot) surface is an interior cold layer, not the real
     emitting region.  Flag it.
  3. Energy-conserving continuum floor — replace the collapsed L_cont_band with the
     value at the color temperature T_floor that satisfies 4π R² σ T_floor⁴ = L_phot
     (per line, via the Planck ratio at that wavelength).  This is composition-
     general: it is driven by the radiated luminosity, not by any opacity/abundance
     assumption.
  4. Energy-conservation check — Σ L_line must not exceed L_phot.
  5. He decrement diagnostic — He-NLTE line ratios referenced to He I 10830 (the
     strongest He I line), with no external/empirical anchor (per the chosen
     "first-principles He-NLTE" budget policy).

All quantities cgs.  The guard NEVER touches the profile SHAPE (gate-authoritative)
— it only corrects the per-line continuum LEVEL L_cont_band used for energy
bookkeeping and EW-from-continuum estimates.  It is OPT-IN (phase5_runner, behind
--he-budget) and additionally AUTO-enabled when the gas is H-free.
"""
from __future__ import annotations
import numpy as np

H_PL = 6.62607015e-27
C = 2.99792458e10
KB = 1.380649e-16
SIGMA_SB = 5.670374419e-5          # erg s^-1 cm^-2 K^-4

X_H_FREE_THRESH = 1.0e-3           # ⟨X_H⟩ below this → H-free (closes P0 #2 too)
COLLAPSE_FRACTION = 0.1            # L_bb_bol < this × L_phot → collapsed


def planck_lambda(T, lam_AA):
    """Planck B_λ(T) [erg/s/cm²/AA/sr] (matches phase5_continuum.planck_lambda)."""
    lam_cm = np.asarray(lam_AA, float) * 1e-8
    x = np.clip(H_PL * C / (lam_cm * KB * max(float(T), 1.0)), 1e-30, 700.0)
    return (2.0 * H_PL * C ** 2 / lam_cm ** 5) / np.expm1(x) * 1e-8


def mean_X_H(obj, default=None):
    """Mean hydrogen mass fraction from a state/merged/snap (X_H per-zone or
    scalar). Returns default if no composition is attached."""
    for attr in ('X_H', 'x_h'):
        val = None
        if isinstance(obj, dict) and attr in obj:
            val = obj[attr]
        elif hasattr(obj, attr):
            val = getattr(obj, attr)
        if val is not None:
            arr = np.asarray(val, float)
            if arr.size:
                return float(np.mean(arr))
    return default


def mean_X_He(obj, default=None):
    """Mean helium mass fraction from a state/merged/snap (X_He per-zone or
    scalar). Returns default if no composition is attached."""
    for attr in ('X_He', 'x_he'):
        val = None
        if isinstance(obj, dict) and attr in obj:
            val = obj[attr]
        elif hasattr(obj, attr):
            val = getattr(obj, attr)
        if val is not None:
            arr = np.asarray(val, float)
            if arr.size:
                return float(np.mean(arr))
    return default


def is_h_free(x_h, thresh=X_H_FREE_THRESH):
    """True if the mean hydrogen fraction marks an H-free model."""
    return (x_h is not None) and np.isfinite(x_h) and (x_h < thresh)


def element_present(x_elem, thresh=X_H_FREE_THRESH):
    """True if the element's mean mass fraction is above the 'present' floor.
    Returns True when composition is unknown (x_elem is None) — i.e. do NOT
    downgrade a line for missing composition data."""
    if x_elem is None or not np.isfinite(x_elem):
        return True
    return x_elem >= thresh


def bb_bolometric(T_phot, R_phot):
    """Bolometric luminosity of a (full, undiluted) blackbody photosphere."""
    return 4.0 * np.pi * float(R_phot) ** 2 * SIGMA_SB * float(T_phot) ** 4


def color_temperature_floor(R_phot, L_phot):
    """Color temperature T_floor s.t. 4π R² σ T_floor⁴ = L_phot."""
    R = float(R_phot)
    if R <= 0 or not np.isfinite(L_phot) or L_phot <= 0:
        return None
    return (float(L_phot) / (4.0 * np.pi * R ** 2 * SIGMA_SB)) ** 0.25


def detect_continuum_collapse(T_phot, R_phot, L_phot, fraction=COLLAPSE_FRACTION):
    """Detect the unphysical cold/compact continuum collapse.

    Returns dict(collapsed, L_bb_bol, L_phot, ratio, T_phot, T_floor). 'collapsed'
    is True when the diluted-BB photosphere's bolometric output is < fraction×L_phot
    (i.e. the (T_phot,R_phot) surface radiates far less than the model actually
    emits — a sign the photosphere was placed too deep/cold)."""
    out = {'collapsed': False, 'L_bb_bol': np.nan, 'L_phot': L_phot,
           'ratio': np.nan, 'T_phot': T_phot, 'T_floor': None}
    if not (np.isfinite(L_phot) and L_phot > 0 and np.isfinite(T_phot)
            and np.isfinite(R_phot) and T_phot > 0 and R_phot > 0):
        return out
    L_bb = bb_bolometric(T_phot, R_phot)
    out['L_bb_bol'] = L_bb
    out['ratio'] = L_bb / L_phot
    out['T_floor'] = color_temperature_floor(R_phot, L_phot)
    out['collapsed'] = bool(out['ratio'] < fraction)
    return out


def guarded_L_cont_band(L_cont_band, lam0_AA, T_phot, T_floor):
    """Energy-conserving continuum floor for one line: rescale the collapsed
    L_cont_band from the (cold) T_phot to the color-temperature floor T_floor via
    the Planck ratio at the line wavelength. Same R, dilution and band, so only the
    Wien/RJ factor at that λ changes. Returns the floored L_cont_band."""
    if T_floor is None or not np.isfinite(L_cont_band):
        return L_cont_band
    b_phot = float(planck_lambda(T_phot, lam0_AA))
    b_floor = float(planck_lambda(T_floor, lam0_AA))
    if b_phot <= 0:
        return L_cont_band
    return float(L_cont_band) * (b_floor / b_phot)


def apply_continuum_guard(spectra, T_phot, R_phot, L_phot, verbose=True):
    """Apply the energy-conserving continuum guard to every line's L_cont_band.

    Mutates spectra in place: stores 'L_cont_band_raw' (original) and overwrites
    'L_cont_band' with the floored value when a collapse is detected. Returns the
    collapse-info dict. No-op (raw preserved) when no collapse. Profile SHAPE is
    untouched. Lines are otherwise left exactly as-is."""
    info = detect_continuum_collapse(T_phot, R_phot, L_phot)
    if not info['collapsed']:
        return info
    T_floor = info['T_floor']
    for name, sp in spectra.items():
        if not isinstance(sp, dict):
            continue
        Lc = sp.get('L_cont_band', None)
        lam0 = sp.get('lambda_rest', None)
        if Lc is None or lam0 is None or not np.isfinite(Lc):
            continue
        sp['L_cont_band_raw'] = Lc
        sp['L_cont_band'] = guarded_L_cont_band(Lc, lam0, T_phot, T_floor)
        sp['continuum_guarded'] = True
    if verbose:
        print(f"[phase5b/compgen] continuum-collapse guard ENGAGED: "
              f"L_bb(T_phot={T_phot:.0f}K,R_phot)={info['L_bb_bol']:.2e} is "
              f"{info['ratio']:.1e}× L_phot={L_phot:.2e}. Floored L_cont_band to "
              f"the energy-conserving color temperature T_floor={T_floor:.0f} K "
              f"(profile shapes untouched).")
    return info


def energy_conservation_check(spectra, L_phot, verbose=True):
    """Σ L_line(_corrected) vs L_phot. Warn if the lines radiate more than the
    bolometric photospheric luminosity (a physical impossibility → over-estimate).
    Returns dict(sum_L_line, L_phot, ok, ratio)."""
    tot = 0.0
    for name, sp in spectra.items():
        if not isinstance(sp, dict):
            continue
        L = sp.get('L_line_corrected', sp.get('L_line', None))
        if L is not None and np.isfinite(L) and L > 0:
            tot += float(L)
    ok = (not (np.isfinite(L_phot) and L_phot > 0)) or (tot <= L_phot)
    ratio = (tot / L_phot) if (np.isfinite(L_phot) and L_phot > 0) else np.nan
    if verbose and not ok:
        print(f"[phase5b/compgen] ENERGY WARNING: Σ L_line = {tot:.3e} erg/s "
              f"exceeds L_phot = {L_phot:.3e} ({ratio:.2f}×). Thick-line absolute "
              f"L is over-estimated; quote with the factor-2 caveat.")
    return {'sum_L_line': tot, 'L_phot': L_phot, 'ok': bool(ok), 'ratio': ratio}


def he_decrement_diagnostic(spectra, reference='He_I_10830', verbose=True):
    """First-principles He decrement: per-line He-NLTE L ratios referenced to the
    strongest He I line (default He I 10830). No external/empirical anchor — this
    is a CONSISTENCY diagnostic on the He-NLTE absolute values. Returns
    {line: L/L_ref}. Flags He I lines stronger than the reference (unusual)."""
    ref = spectra.get(reference, None)
    if not isinstance(ref, dict):
        return {}
    L_ref = ref.get('L_line_corrected', ref.get('L_line', None))
    if not (L_ref and np.isfinite(L_ref) and L_ref > 0):
        return {}
    ratios = {}
    for name, sp in spectra.items():
        if not (name.startswith('He_') and isinstance(sp, dict)):
            continue
        L = sp.get('L_line_corrected', sp.get('L_line', None))
        if L is not None and np.isfinite(L):
            ratios[name] = float(L) / float(L_ref)
    if verbose and ratios:
        print(f"[phase5b/compgen] He decrement (L / L[{reference}], "
              f"first-principles He-NLTE, no anchor):")
        for nm in sorted(ratios, key=lambda k: -ratios[k]):
            flag = '  <-- He I > 10830 (check)' if (
                nm.startswith('He_I_') and nm != reference
                and ratios[nm] > 1.0) else ''
            print(f"           {nm:12s} {ratios[nm]:8.3e}{flag}")
    return ratios
