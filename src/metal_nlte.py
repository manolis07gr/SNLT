"""
metal_nlte.py — CHIANTI-backed NLTE emissivities for the C/O/Ne metal lines
================================================================================

FUTURE_WORK P2 item 5 (Tier 1 upgrade). Replaces the PROVISIONAL collisional
emissivities in metal_atoms with authoritative, NLTE values from the CHIANTI
atomic database via ChiantiPy.

For each collisionally-excited metal line, CHIANTI's `ion.emiss()` returns the
per-ion emissivity

    emiss = (hν / 4π) · n_upper^frac · A_ul        [erg s⁻¹ sr⁻¹ per ion]

where n_upper^frac is the upper-level population FRACTION from CHIANTI's full
multi-level statistical-equilibrium solve (authoritative level energies,
A-values, electron + proton collision strengths, cascades, density / n_crit — all
built in). The volume emissivity is then

    j = 4π · emiss(T, n_e) · n_ion                  [erg s⁻¹ cm⁻³]

with n_ion from the photoionization-equilibrium ion balance (metal_ionization).
So the LINE PHYSICS becomes air-tight; the remaining modelling choice is the ion
balance (still our photoionization solver — a Cloudy Tier-2 step would close that).

Requirements (runtime): ChiantiPy installed and the CHIANTI database reachable
via the XUVTOP environment variable. If either is absent, `chianti_available()`
returns False and the caller falls back to the provisional metal_atoms emissivity
— the pipeline never breaks; it just uses the best data available.

Covered lines (collisional, in CHIANTI): C IV 1549, C III] 1909, [O I] 6300,
[O III] 5007, [Ne III] 3869. (C III 4647 is a pure recombination-cascade line,
not in CHIANTI's collisional emissivity — it stays provisional until a
recombination database / Cloudy is added.)
"""
from __future__ import annotations
import os
import numpy as np

# our metal-line name -> (CHIANTI ion id, central λ [Å], match/sum tolerance [Å])
# The tolerance sums close multiplet components (e.g. the C IV 1548+1550 doublet,
# the C III] 1907+1909 intercombination pair) into our single named line, while
# staying narrow enough to exclude neighbours (e.g. [O III] 4959 from 5007).
CHIANTI_LINES = {
    'C_IV_1549':   ('c_4',  1549.05, 4.0),   # 1548.2 + 1550.8 resonance doublet
    'C_III_1909':  ('c_3',  1907.70, 4.0),   # 1906.7 + 1908.7 intercombination
    'O_I_6300':    ('o_1',  6300.30, 2.0),   # [O I] 6300
    'O_III_5007':  ('o_3',  5006.84, 2.0),   # [O III] 5007
    'Ne_III_3869': ('ne_3', 3868.76, 2.0),   # [Ne III] 3869
}

# CHIANTI collision data validity guard (extrapolation outside is unreliable, but
# the out-of-range zones are low-emission and contribute negligibly).
_T_MIN, _T_MAX = 1.0e3, 1.0e8
_NE_MIN, _NE_MAX = 1.0, 1.0e16

_AVAIL_CACHE = None


def chianti_available():
    """True if ChiantiPy is importable AND the CHIANTI database (XUVTOP) is set."""
    global _AVAIL_CACHE
    if _AVAIL_CACHE is not None:
        return _AVAIL_CACHE
    ok = False
    try:
        if os.environ.get('XUVTOP'):
            import ChiantiPy.core as _ch  # noqa: F401
            ok = True
    except Exception:
        ok = False
    _AVAIL_CACHE = ok
    return ok


def line_emissivity_per_ion(name, T, n_e):
    """CHIANTI NLTE per-ion emissivity [erg/s/sr per ion] for `name`, summed over
    the matched transition(s), evaluated at per-zone (T, n_e).

    Returns (emiss_per_zone array, n_matched_transitions) or (None, 0) if the line
    is not CHIANTI-backed or anything fails. Multiply the result by `n_ion` and 4π
    to get the volume emissivity [erg/s/cm³].
    """
    if name not in CHIANTI_LINES:
        return None, 0
    ion_name, lam0, tol = CHIANTI_LINES[name]
    try:
        import ChiantiPy.core as ch
        T = np.clip(np.asarray(T, float), _T_MIN, _T_MAX)
        n_e = np.clip(np.asarray(n_e, float), _NE_MIN, _NE_MAX)
        z = ch.ion(ion_name, temperature=T, eDensity=n_e)
        z.emiss()
        wvl = np.asarray(z.Emiss['wvl'], float)
        em = np.asarray(z.Emiss['emiss'], float)      # (n_trans, n_zones) or (n_trans,)
        if em.ndim == 1:
            em = em[:, None]
        sel = np.abs(wvl - lam0) <= tol
        if not sel.any():
            return None, 0
        emiss = em[sel].sum(axis=0)                    # (n_zones,) erg/s/sr per ion
        emiss = np.where(np.isfinite(emiss), emiss, 0.0)
        return emiss, int(sel.sum())
    except Exception:
        return None, 0
