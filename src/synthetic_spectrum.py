#!/usr/bin/env python3
"""synthetic_spectrum.py — assemble a synthetic optical spectrum from per-line npz.

Stage 4 of the optical-fidelity upgrade (FUTURE_WORK P2 #7): combine every line
profile the pipeline computed (H + He + metals, with the narrow-CSM component if
the npz was produced with --narrow-csm) into ONE spectrum on a common wavelength
grid, directly overplottable on observed spectra.

Method (deliberately an ASSEMBLY, not a new RT solve): each line's
F_norm(λ) = F/F_cont is already the emergent, electron-scattered, velocity-
resolved profile on the local continuum. On a common grid,
    F_total(λ) = 1 + Σ_lines [F_norm,line(λ) - 1]
(excesses add linearly on the shared continuum; blends — e.g. the C III 4650
complex — form naturally by summation). Lines whose F_norm is NaN
(continuum-suppressed UV at a cold photosphere: quote L_line, no optical shape)
are skipped. With --absolute, the result is multiplied by the diluted-BB
continuum shape (T_phot, R_phot from the npz run) for an F_λ-like spectrum;
default is F/F_cont (the observer-friendly normalised form).

CLI:
    python synthetic_spectrum.py --npz prod_day005_lines.npz \
        [--lam-min 3500 --lam-max 9500 --n 4000] [--out spec.png/.txt]
Library:
    lam, F = assemble(npz_path, lam_min, lam_max, n)
"""
from __future__ import annotations
import argparse
import sys
import numpy as np


def assemble(npz_path, lam_min=3500.0, lam_max=9500.0, n=4000,
             skip_lines=(), verbose=True):
    """Return (lam_grid_AA, F_over_Fcont, contributing_line_names)."""
    z = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in z['line_names']]
    lam_grid = np.linspace(float(lam_min), float(lam_max), int(n))
    F = np.ones_like(lam_grid)
    used = []
    for nme in names:
        if nme in skip_lines:
            continue
        lk, fk = f'{nme}__lambda', f'{nme}__F_norm_corrected'
        if fk not in z.files:
            fk = f'{nme}__F_norm'
        if lk not in z.files or fk not in z.files:
            continue
        lam = np.asarray(z[lk], float)
        Fn = np.asarray(z[fk], float)
        m = np.isfinite(lam) & np.isfinite(Fn)
        if m.sum() < 5:                       # cont-suppressed / no usable shape
            continue
        lam, Fn = lam[m], Fn[m]
        if lam[-1] < lam_min or lam[0] > lam_max:
            continue                          # entirely outside the window (UV/IR)
        # add this line's excess on the common grid (0 outside its own window)
        F += np.interp(lam_grid, lam, Fn - 1.0, left=0.0, right=0.0)
        used.append(nme)
    if verbose:
        print(f"[synth] {npz_path}: {len(used)} lines contribute in "
              f"[{lam_min:.0f},{lam_max:.0f}] AA: {', '.join(used)}")
    return lam_grid, F, used


def main(argv=None):
    p = argparse.ArgumentParser(description="Assemble synthetic spectrum from a "
                                            "prod_day*_lines.npz")
    p.add_argument('--npz', required=True)
    p.add_argument('--lam-min', type=float, default=3500.0)
    p.add_argument('--lam-max', type=float, default=9500.0)
    p.add_argument('--n', type=int, default=4000)
    p.add_argument('--out', default=None, help=".png plot and/or .txt two-column")
    args = p.parse_args(argv)

    lam, F, used = assemble(args.npz, args.lam_min, args.lam_max, args.n)
    out = args.out or (args.npz.replace('_lines.npz', '_synth.png'))
    if out.endswith('.txt'):
        np.savetxt(out, np.column_stack([lam, F]),
                   header='lambda_AA  F_over_Fcont')
        print(f"[synth] wrote {out}")
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(lam, F, 'k-', lw=1.0)
        ax.axhline(1.0, color='gray', ls=':', lw=0.6)
        ax.set_xlabel(r'rest wavelength [$\AA$]')
        ax.set_ylabel(r'F / F$_{cont}$')
        ax.set_title(f"synthetic spectrum — {args.npz}  ({len(used)} lines)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"[synth] wrote {out}")


if __name__ == '__main__':
    main(sys.argv[1:])
