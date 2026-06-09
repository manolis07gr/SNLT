"""
velocity_grad.py — robust velocity gradient |dv/dr| for Sobolev optical depths
================================================================================

STELLA snapshots of CSM-interaction models place **coincident (duplicate) radial
zones at shocks** (the reverse/forward shock fronts). `np.gradient(v, r)` — and
any centred finite difference `(v[i+1]-v[i-1])/(r[i+1]-r[i-1])` — then divides by
a zero radius spacing and returns **NaN / inf** at those zones. The usual floor
`np.maximum(dv_dr, floor)` does NOT repair this (`max(NaN, x) == NaN`), so the
NaN poisons every Sobolev optical depth that divides by |dv/dr| and propagates
through the H / He NLTE solves and the production-Hα luminosity — turning the
whole snapshot's line output into NaN (seen in the H-free C-series interaction
phase, e.g. C4 days 10–50, which have velocity reversals + duplicate radii).

`robust_dvdr` computes |dv/dr| but, at any zone where the finite difference is
non-finite, falls back to the **local homologous gradient |v|/r** (= 1/t_exp for
free expansion — always finite and physically the right scale), then applies the
turbulent-broadening floor. Default behaviour on a clean (strictly monotonic)
grid is identical to `np.maximum(|np.gradient(v,r)|, floor)`.
"""
from __future__ import annotations
import numpy as np


def robust_dvdr(v, r, v_turb_cms=None, min_floor=1.0e-30):
    """Return |dv/dr| [1/s], finite everywhere.

    v, r : zone arrays (cm/s, cm). r may contain duplicate / non-monotonic values
           (STELLA shock zones); those give a non-finite finite-difference, which
           is replaced by the homologous estimate |v|/r.
    v_turb_cms : if given, the floor is max(min_floor, v_turb / (r[-1]-r[0])),
                 the standard microturbulent Sobolev floor; else min_floor.
    """
    v = np.asarray(v, float)
    r = np.asarray(r, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.abs(np.gradient(v, r))
    homol = np.abs(v) / np.maximum(np.abs(r), 1.0e-30)
    dv_dr = np.where(np.isfinite(g), g, homol)
    dv_dr = np.where(np.isfinite(dv_dr), dv_dr, 0.0)   # belt-and-suspenders
    floor = float(min_floor)
    if v_turb_cms is not None and r.size > 1:
        span = float(r[-1] - r[0])
        if span > 0:
            floor = max(min_floor, float(v_turb_cms) / span)
    return np.maximum(dv_dr, floor)
