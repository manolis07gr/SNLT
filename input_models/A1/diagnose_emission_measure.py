"""diagnose_emission_measure.py — localize why the production Hα luminosity
exceeds the photospheric luminosity in the cool, neutral IIP regime.

Hypothesis under test
---------------------
The production path uses a case-B recombination emissivity
    eps_rec = n_e * n_p * alpha_Halpha(T) * h*nu * dV
For a *photoionized* CSM (IIn) n_p ~ n_e and this is a faithful recombination
measure. For a cool, mostly-NEUTRAL IIP envelope, n_e (set partly by metals
and residual ionization) can greatly exceed the true proton density
n_p = n_HII. If the code evaluates the recombination term with n_p ~ n_e
(or with an over-estimated n_p), the emission measure n_e*n_p is inflated by
the factor n_e / n_HII, over-producing Hα.

This script computes, over the above-photosphere zones the pipeline actually
uses, the Halpha recombination luminosity under three assumptions and
compares them to:
  - L_phot (the photospheric luminosity; a hard physical ceiling)
  - the known production value  L ~ 1.21e42 erg/s  (the suspect output)
  - the known Phase-5 value      L ~ 5.5e39 erg/s   (the physical-looking output)

  (A) n_p = n_e                       -> what caseb_hr does if n_p defaults to n_e
  (B) n_p = x_Saha(T,n_e) * n_H        -> physically-correct proton density (pure-H Saha)
  (C) collisional-from-n=2 proxy       -> upper bound on the Lyman-trapping channel
                                          (only if it can be estimated; else skipped)

It then ranks the per-zone contributions of the dominant term so we can see
whether the luminosity is concentrated in a few dense inner zones just above
the photosphere (the IIP "pseudo-shell" assignment).

Usage:
    python diagnose_emission_measure.py mesa.day050_post_Lbol_max.data
    python diagnose_emission_measure.py mesa.day050_post_Lbol_max.data --top 15

Uses only stella_io + physical constants. No external pipeline modules, so the
recombination measure here is INDEPENDENT of the production code path — that is
the point: it is a clean external check.
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import stella_io

# ---- physical constants (cgs) ----
ME   = 9.1093837e-28
KB   = 1.380649e-16
HPL  = 6.62607015e-27
CC   = 2.99792458e10
MP   = 1.6726219e-24
EV   = 1.602176634e-12
CHI_H = 13.5984 * EV            # H ionization potential
LAM_HA = 6562.81e-8            # cm
NU_HA  = CC / LAM_HA
HNU_HA = HPL * NU_HA           # ~3.03e-12 erg


def alpha_halpha_eff(T):
    """Case-B effective recombination coefficient for Hα photons [cm^3/s].

    Normalized to 1.17e-13 at 1e4 K with a ~T^-0.94 dependence (standard
    case-B fit, accurate to ~10% over 5000-20000 K). The exact value cancels
    in the (A)/(B) ratio — it only matters for the absolute comparison to
    L_phot, where ~10% is irrelevant to a 200x discrepancy.
    """
    return 1.17e-13 * (np.maximum(T, 1.0) / 1.0e4) ** (-0.942)


def saha_x_HII(T, n_e):
    """Pure-hydrogen Saha ionization fraction x = n_HII / n_H given T and the
    free-electron density n_e.

    Solves  n_HII * n_e / n_HI = S(T)   with
        S(T) = (2 pi m_e kT / h^2)^1.5 * (2 g_II / g_I) * exp(-chi/kT)
    g_I = 2 (H I ground), g_II = 1 (proton)  ->  2 g_II/g_I = 1.
    Treating n_e as given (from the snapshot), x solves
        x / (1 - x) = S / n_e   =>   x = (S/n_e) / (1 + S/n_e).
    """
    kT = KB * np.maximum(T, 1.0)
    S = (2.0 * np.pi * ME * kT / (HPL * HPL)) ** 1.5 * np.exp(-CHI_H / kT)
    ratio = S / np.maximum(n_e, 1e-30)
    x = ratio / (1.0 + ratio)
    return np.clip(x, 0.0, 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('snapshot')
    ap.add_argument('--top', type=int, default=12,
                    help='how many top-contributing zones to list')
    ap.add_argument('--prod-L', type=float, default=1.21e42,
                    help='known production L_line for comparison')
    ap.add_argument('--phase5-L', type=float, default=5.5e39,
                    help='known Phase-5 physical L_line for comparison')
    args = ap.parse_args(argv)

    snap = stella_io.load_stella_snapshot(args.snapshot, verbose=False)
    trunc = stella_io.truncate_to_photosphere(snap, verbose=False)

    r   = trunc['r']
    v   = trunc['v']
    T   = trunc['T']
    rho = trunc['rho']
    n_e = trunc['n_e']
    X_H = trunc.get('X_H', 0.638 * np.ones_like(r))
    if np.ndim(X_H) == 0:
        X_H = float(X_H) * np.ones_like(r)
    L_phot = trunc['L_phot_inner']
    R_phot = trunc['R_phot_inner']

    n_H = rho * X_H / MP
    dr  = np.gradient(r)
    dV  = 4.0 * np.pi * r * r * np.abs(dr)

    alpha = alpha_halpha_eff(T)
    x_HII = saha_x_HII(T, n_e)
    n_p_saha = x_HII * n_H

    # Per-zone emissivities [erg/s]
    eps_A = n_e * n_e      * alpha * HNU_HA * dV   # (A) n_p = n_e
    eps_B = n_e * n_p_saha * alpha * HNU_HA * dV   # (B) n_p = Saha proton density

    L_A = float(eps_A.sum())
    L_B = float(eps_B.sum())

    print("=" * 78)
    print(f" Hα recombination emission-measure diagnostic")
    print(f" snapshot: {args.snapshot}")
    print("=" * 78)
    print(f" epoch            : {trunc.get('epoch_d')} d")
    print(f" zones above phot : {len(r)}")
    print(f" R_phot           : {R_phot:.3e} cm")
    print(f" L_phot (ceiling) : {L_phot:.3e} erg/s")
    print(f" T range          : {T.min():.0f} - {T.max():.0f} K")
    print(f" n_e range        : {n_e.min():.2e} - {n_e.max():.2e} cm^-3")
    print(f" x_HII (Saha)     : {x_HII.min():.2e} - {x_HII.max():.2e}  (median {np.median(x_HII):.2e})")
    print(f" <n_e / n_p_Saha> : {np.median(n_e / np.maximum(n_p_saha, 1e-30)):.2e}  (median over zones)")
    print()
    print(" Halpha recombination luminosity:")
    print(f"   (A) n_p = n_e          : L = {L_A:.3e} erg/s   = {L_A / L_phot:.2f} x L_phot")
    print(f"   (B) n_p = Saha proton  : L = {L_B:.3e} erg/s   = {L_B / L_phot:.3f} x L_phot")
    print()
    print(f"   overcount factor (A/B) : {L_A / max(L_B, 1e-30):.1f}x")
    print()
    print(" Comparison to known pipeline outputs:")
    print(f"   production L_line  ~ {args.prod_L:.2e}  -> matches (A)? ratio A/prod = {L_A/args.prod_L:.2f}")
    print(f"   Phase-5  L_line    ~ {args.phase5_L:.2e}  -> matches (B)? ratio B/ph5  = {L_B/args.phase5_L:.2f}")
    print()

    # Rank zones by the dominant (A) term
    order = np.argsort(eps_A)[::-1]
    cumfrac = np.cumsum(eps_A[order]) / max(eps_A.sum(), 1e-30)
    n_for_90 = int(np.searchsorted(cumfrac, 0.90) + 1)
    print(f" Concentration of term (A): top {n_for_90} zones carry 90% of the emission.")
    print()
    print(f" Top {args.top} zones by (A) emissivity:")
    print(f"   {'idx':>4} {'r[cm]':>11} {'r/Rphot':>8} {'v[km/s]':>8} {'T[K]':>7} "
          f"{'n_e':>10} {'x_HII':>9} {'n_e/n_p':>9} {'L_A_frac':>9}")
    for k in range(min(args.top, len(r))):
        i = order[k]
        ratio_np = n_e[i] / max(n_p_saha[i], 1e-30)
        print(f"   {i:>4d} {r[i]:>11.3e} {r[i]/R_phot:>8.3f} {v[i]/1e5:>8.0f} "
              f"{T[i]:>7.0f} {n_e[i]:>10.2e} {x_HII[i]:>9.2e} {ratio_np:>9.1e} "
              f"{eps_A[i]/max(eps_A.sum(),1e-30):>9.1%}")
    print()
    print("=" * 78)
    print(" INTERPRETATION")
    print("=" * 78)
    if L_A > 0.5 * L_phot and L_B < 0.1 * L_phot:
        print(" CONFIRMED: the n_p = n_e recombination measure (A) is unphysical")
        print(" (> 0.5 L_phot), while the proper-proton measure (B) is physical")
        print(" (< 0.1 L_phot). The over-luminosity is the n_e-vs-n_p overcount in")
        print(" the cool neutral envelope. Fix: feed the true n_p = x_HII * n_H to")
        print(" the production emissivity (the Phase-5 kernel already does this).")
    elif L_A > L_B * 10:
        print(f" The n_p=n_e measure (A) exceeds the proper measure (B) by "
              f"{L_A/max(L_B,1e-30):.0f}x.")
        print(" The n_e-vs-n_p overcount is a major contributor, though the")
        print(" absolute match to the production value may need the collisional /")
        print(" Lyα-trapping channel too. Recommend checking n_p source in")
        print(" snline_autoparams.compute_emissivity for the IIP branch.")
    else:
        print(" (A) and (B) are comparable -> the n_e-vs-n_p overcount is NOT the")
        print(" dominant cause. The production over-luminosity must come from")
        print(" another term (collisional excitation / Lyα-trapped n_2 feeding")
        print(" n_3, or the emission-model selection). Next probe: dump the")
        print(" per-zone n_2, n_3 from the NLTE solve and compare n_3*A_32*hν*dV.")
    print("=" * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
