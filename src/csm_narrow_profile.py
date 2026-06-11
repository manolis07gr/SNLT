#!/usr/bin/env python3
"""csm_narrow_profile.py — narrow P-Cygni component from the UNSHOCKED CSM.

WHY (observational driver): real Type Icn (SN 2019hgp, 2021csp) and IIn show
*narrow / intermediate-width P-Cygni* lines from the slow, dense, photoionized
wind AHEAD of the forward shock (the unshocked CSM) — superposed on the broad
interaction-shell emission the pipeline already computes. Our boxy profiles
capture the broad shell but not this narrow core. This module supplies the
narrow component so the TOTAL profile = broad interaction shell + narrow CSM.

PHYSICS: the unshocked CSM is a ~constant-velocity wind at v_wind (~100-300 km/s
for IIn H-rich winds; ~1000 km/s for the Icn He/C winds). Photoionized, an
(optically-thin-ish) expanding wind shell gives a NARROW ~flat-topped emission of
projected half-width v_wind. For a resonance/permitted line the near side
(blueshifted, projected in front of the photospheric disk) also ABSORBS the
continuum → a blue absorption notch = the classic P-Cygni. Forbidden lines
(f_lu≈0) get pure narrow emission, no absorption.

This is a SHAPE generator only — it takes the narrow-component luminosity / depth
as inputs (the caller sets them from the unshocked-CSM emission measure and the
line optical depth). It is ADDITIVE on the F/F_cont=1 continuum so the integrator
stays linear:  F_total(λ) = 1 + broad_excess(λ) + narrow_excess(λ).

Self-contained: numpy (+ scipy.special.erf if available; falls back to a numpy
erf). Nothing in the live pipeline imports this yet — it is wired in only after
validation, additively, so existing lines are unchanged when the narrow
component is off (amp_em=0, depth_abs=0).
"""
from __future__ import annotations
import numpy as np

try:                                    # smooth flat-top edges via erf
    from scipy.special import erf as _erf
except Exception:                       # numpy-only fallback (Abramowitz-Stegun 7.1.26)
    def _erf(x):
        x = np.asarray(x, float)
        s = np.sign(x); ax = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                    - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
        return s * y


def narrow_csm_component(v_kms, v_wind, amp_em, depth_abs=0.0,
                         therm_kms=None, abs_blue_frac=1.0):
    """Narrow CSM profile excess on F/F_cont=1 (ADDITIVE: return is F-1).

    Parameters
    ----------
    v_kms : array         line-of-sight velocity grid [km/s] (0 = line centre)
    v_wind : float        unshocked-CSM (wind) velocity [km/s]; sets the width
    amp_em : float        peak height of the narrow EMISSION above the continuum
                          (in F/F_cont units; caller sets from the narrow-component
                           luminosity / continuum)
    depth_abs : float     depth of the blue ABSORPTION notch (0..1; 0 = pure
                          emission, for forbidden lines). For resonance lines the
                          caller sets this from the line optical depth (≈1-exp(-τ)).
    therm_kms : float     thermal/instrumental smoothing of the flat-top edges
                          [km/s]; default max(0.12*v_wind, 20).
    abs_blue_frac : float fraction of v_wind over which the blue trough extends
                          (1 = full [-v_wind,0]).

    Returns
    -------
    excess : ndarray      F(v) - 1, same shape as v_kms. EMISSION is positive,
                          ABSORPTION negative; add to the broad-profile excess.
    """
    v = np.asarray(v_kms, float)
    vw = abs(float(v_wind))
    if vw <= 0 or (amp_em <= 0 and depth_abs <= 0):
        return np.zeros_like(v)
    sig = float(therm_kms) if therm_kms else max(0.12 * vw, 20.0)
    rt2 = np.sqrt(2.0) * sig

    # --- narrow ~flat-topped EMISSION over [-vw, +vw] (thin expanding wind shell)
    # 0.5*[erf((vw - v)/√2σ) + erf((vw + v)/√2σ)] is a smoothed boxcar of half-width vw
    emis = 0.5 * (_erf((vw - v) / rt2) + _erf((vw + v) / rt2))
    emis = amp_em * np.clip(emis, 0.0, None)

    # --- blue ABSORPTION (resonance lines): continuum ATTENUATION exp(-tau(v))
    # over the blueshifted wind in front of the disk. Modelling it as attenuation
    # (not an additive notch) is what lets the trough dip BELOW the continuum when
    # the line is optically thick and the local emission is modest — the real
    # P-Cygni behaviour. depth_abs is the max fractional absorption (≈1-exp(-tau)).
    absn = np.zeros_like(v)
    if depth_abs > 0.0:
        va = vw * float(abs_blue_frac)
        box = np.clip(0.5 * (_erf((0.0 - v) / rt2) + _erf((va + v) / rt2)), 0.0, 1.0)
        tau0 = -np.log(max(1.0 - float(np.clip(depth_abs, 0.0, 0.999)), 1e-6))
        absn = 1.0 - np.exp(-tau0 * box)          # in [0, depth_abs]

    return emis - absn


def wind_velocity_from_state(v_kms_zone, w_emission=None, q=0.10):
    """Estimate the unshocked-CSM (wind) velocity from a snapshot: the LOW end of
    the emitting velocity field — the q-quantile of |v| over the emitting zones
    (the slow outer wind), as opposed to the fast shell. Robust default for
    setting v_wind when the explicit CSM wind speed is not threaded through.
    """
    v = np.abs(np.asarray(v_kms_zone, float))
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return 0.0
    if w_emission is not None:
        w = np.asarray(w_emission, float)
        m = np.isfinite(w) & (w > 0)
        if m.any():
            vv = v[m] if v.shape == w.shape else v
            order = np.argsort(vv)
            cw = np.cumsum((w[m][order] if v.shape == w.shape else np.ones_like(vv)))
            cw = cw / cw[-1]
            return float(vv[order][min(int(np.searchsorted(cw, q)), vv.size - 1)])
    return float(np.quantile(v, q))
