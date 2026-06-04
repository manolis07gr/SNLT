"""diagnose_shock_emergence.py — quantify when the fast shock shell's
outer edge emerges above the photosphere across the early epochs.

CORRECTION to an earlier, wrong diagnostic: the location of the single
fastest zone is NOT the right metric. The velocity field is a thin spike
(slow inner ejecta → fast shell ~5000 km/s → slow outer CSM). The spike
PEAK stays below R_phot at both day 20 and day 30. What changes is whether
the OUTER WING of that fast shell pokes above the photosphere.

The right metric: how much above-photosphere material exceeds a velocity
threshold. We report, per epoch:
  - above-phot v_max
  - number of above-phot zones with v > v_thresh
  - mass of above-phot material with v > v_thresh
  - fraction of above-phot Hα emissivity (case-B proxy n_e^2) from v > v_thresh

Usage:
    python diagnose_shock_emergence.py \
        mesa.day010_post_Lbol_max.data \
        mesa.day020_post_Lbol_max.data \
        mesa.day030_post_Lbol_max.data \
        mesa.day040_post_Lbol_max.data \
        --v-thresh 2000
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import stella_io

M_P = 1.6726e-24


def analyze(path, v_thresh_kms):
    snap = stella_io.load_stella_snapshot(path, verbose=False)
    trunc = stella_io.truncate_to_photosphere(snap, verbose=False)
    R_phot = trunc['R_phot_inner']

    r = snap['r']
    v = snap['v']            # cm/s
    rho = snap['rho']

    above = r > R_phot
    v_above = v[above]
    r_above = r[above]
    rho_above = rho[above]

    # shell volume per zone (thin-shell)
    dr = np.gradient(r_above)
    dV = 4.0 * np.pi * r_above**2 * np.abs(dr)
    dm = rho_above * dV

    v_kms = v_above / 1e5
    fast = v_kms > v_thresh_kms

    # Hα emissivity proxy: case-B ~ n_e^2 ~ (rho/m_p)^2 (ignoring ionization
    # detail; just a weighting to see where the LINE luminosity concentrates)
    if 'n_e' in snap:
        n_e_above = snap['n_e'][above]
    else:
        n_e_above = rho_above / M_P
    emis = n_e_above**2 * dV
    emis_frac_fast = emis[fast].sum() / max(emis.sum(), 1e-30)

    return {
        'epoch': snap.get('epoch_d'),
        'R_phot': R_phot,
        'n_above': int(above.sum()),
        'v_above_max': float(v_kms.max()) if v_kms.size else 0.0,
        'n_fast': int(fast.sum()),
        'mass_fast_frac': float(dm[fast].sum() / max(dm.sum(), 1e-30)),
        'emis_fast_frac': float(emis_frac_fast),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('snapshots', nargs='+')
    ap.add_argument('--v-thresh', type=float, default=2000.0,
                    help='velocity threshold for "fast" material, km/s')
    args = ap.parse_args(argv)

    print(f"\nShock-shell emergence above photosphere  (v_thresh = "
          f"{args.v_thresh:.0f} km/s)\n")
    hdr = (f"{'epoch':>7s}  {'R_phot[cm]':>11s}  {'n_above':>7s}  "
           f"{'v_above_max':>11s}  {'n_fast':>6s}  {'mass_fast':>9s}  "
           f"{'emis_fast':>9s}")
    print(hdr)
    print('-' * len(hdr))
    rows = []
    for p in args.snapshots:
        try:
            d = analyze(p, args.v_thresh)
        except Exception as e:
            print(f"  {os.path.basename(p)}: ERROR {e}")
            continue
        rows.append(d)
        ep = d['epoch'] if d['epoch'] is not None else float('nan')
        print(f"{ep:>7.1f}  {d['R_phot']:>11.3e}  {d['n_above']:>7d}  "
              f"{d['v_above_max']:>9.0f}km/s  {d['n_fast']:>6d}  "
              f"{d['mass_fast_frac']:>8.1%}  {d['emis_fast_frac']:>8.1%}")

    # Identify emergence epoch (first epoch where n_fast > 0)
    emerged = [r for r in rows if r['n_fast'] > 0]
    if emerged and rows[0]['n_fast'] == 0:
        first = emerged[0]
        print(f"\nFast shell first emerges above photosphere at epoch "
              f"{first['epoch']:.1f}d "
              f"({first['n_fast']} zones, {first['mass_fast_frac']:.1%} of "
              f"above-phot mass, {first['emis_fast_frac']:.1%} of emissivity).")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
