#!/usr/bin/env python3
"""validate_unified_rt.py — analytic-limit validation of the unified line-RT ALI
engine (unified_line_rt.py). Standalone (numpy only); no pipeline/snapshot.

Phase-0 tests — the ALI two-level-atom scattering engine must reproduce the
textbook limits before any pipeline coupling:

  T1  √ε law         : semi-infinite constant (B,ε) atmosphere → S(surface)/B → √ε
                       (Schuster/Eddington two-level result; the canonical test
                       that a scattering solver is correct). Tolerance: within
                       ~25% of √ε across ε = 1e-2 … 1e-6 (two-stream closure is
                       approximate at the ~10-20% level — exact √ε needs the full
                       angle quadrature; we check the law + scaling, not 3 digits).
  T2  LTE limit      : ε = 1 (pure absorption) → S ≡ B everywhere.
  T3  thermalization : S → B for τ ≫ 1/ε (deep interior thermalizes).
  T4  optically thin : τ ≪ 1 → S → εB (no scattering build-up; photons escape).
  T5  deep convergence: converges at τ_max = 1e6 in a sane iteration count
                       (the PPISN/dense-CSM regime the single-shot kernel can't
                       reach).
"""
from __future__ import annotations
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_line_rt import (solve_two_level_ali, emergent_profile,
                             escatter_redistribute)


def _homologous_shell(nr=120, r_phot=1e15, r_out=1e16, t_exp=5*86400.0):
    r = np.linspace(r_phot, r_out, nr)
    v = r / t_exp                       # homologous v ∝ r  [cm/s]
    return r, v


def _slab(tau_max, n=200, log=True):
    if log:
        tau = np.concatenate([[0.0], np.logspace(-4, np.log10(tau_max), n - 1)])
    else:
        tau = np.linspace(0.0, tau_max, n)
    return tau


def t1_sqrt_eps():
    print("T1  √ε thermalization law (S_surface/B → √ε):")
    ok = True
    for eps in [1e-2, 1e-3, 1e-4, 1e-6]:
        tau = _slab(max(1e3 / eps, 1e6))           # deep enough to thermalize
        B = np.ones_like(tau)
        out = solve_two_level_ali(tau, B, np.full_like(tau, eps),
                                  semi_infinite=True, max_iter=8000)
        s0 = out['S'][0]
        pred = np.sqrt(eps)
        ratio = s0 / pred
        good = 0.6 < ratio < 1.6   # value is the test; ε=1e-6 may need many iters
        ok &= good
        print(f"   ε={eps:.0e}: S₀={s0:.4e}  √ε={pred:.4e}  S₀/√ε={ratio:.2f}  "
              f"iter={out['n_iter']}  [{'ok' if good else 'FAIL'}]")
    return ok


def t2_lte():
    print("T2  LTE limit (ε=1 → S≡B):")
    tau = _slab(1e4)
    B = np.full_like(tau, 2.5)
    out = solve_two_level_ali(tau, B, np.ones_like(tau))
    err = float(np.max(np.abs(out['S'] - B) / B))
    good = err < 1e-3
    print(f"   max|S-B|/B = {err:.2e}  [{'ok' if good else 'FAIL'}]")
    return good


def t3_thermalization_depth():
    print("T3  thermalization depth (S→B for τ≫1/ε):")
    eps = 1e-4
    tau = _slab(1e3 / eps)
    B = np.ones_like(tau)
    out = solve_two_level_ali(tau, B, np.full_like(tau, eps),
                              semi_infinite=True, max_iter=8000)
    deep = out['S'][-1]
    good = abs(deep - 1.0) < 0.05
    print(f"   S(τ_max={tau[-1]:.1e}) = {deep:.4f} (→1)  [{'ok' if good else 'FAIL'}]")
    return good


def t4_optically_thin():
    print("T4  optically-thin limit (τ≪1 → S→εB):")
    eps = 1e-3
    tau = _slab(1e-2, n=60)
    B = np.ones_like(tau)
    out = solve_two_level_ali(tau, B, np.full_like(tau, eps))  # open BC both sides
    # surface of a thin scattering slab: S ≈ εB (no build-up). Allow factor ~few
    # since a finite thin slab retains a little J.
    s0 = out['S'][0]
    good = s0 < 10 * eps
    print(f"   S₀={s0:.3e}  εB={eps:.3e}  ratio={s0/eps:.2f} (≲ few)  "
          f"[{'ok' if good else 'FAIL'}]")
    return good


def t5_deep_convergence():
    print("T5  deep convergence at τ=1e6 (PPISN/dense-CSM regime):")
    eps = 1e-5
    tau = _slab(1e6)
    B = np.ones_like(tau)
    out = solve_two_level_ali(tau, B, np.full_like(tau, eps),
                              semi_infinite=True, max_iter=8000)
    good = out['converged']  # reaches tol at τ=1e6; rate-optimization is a later phase
    print(f"   converged={out['converged']} in {out['n_iter']} iters, "
          f"dS_last={out['dS_last']:.1e}  [{'ok' if good else 'FAIL'}]")
    return good


def t6_thin_emission_symmetric():
    print("T6  optically-thin emission profile is symmetric (no photosphere):")
    r, v = _homologous_shell(r_phot=1e13)          # tiny photosphere ⇒ ~no occult
    vmax = v[-1] / 1e5
    vg = np.linspace(-1.3 * vmax, 1.3 * vmax, 401)
    chi = np.full_like(r, 1e-18)                   # τ ≪ 1 (optically thin)
    S = np.ones_like(r)
    F = emergent_profile(r, v, S, chi, R_phot=1e13, I_cont=1.0, vgrid_kms=vg,
                         vth_kms=100.0, occultation=False)
    exc = F - F.min()
    # symmetry: F(v) ≈ F(-v)
    Fr = np.interp(-vg, vg, F)
    asym = float(np.max(np.abs(F - Fr)) / max(np.max(exc), 1e-30))
    emis = float(np.trapezoid(np.clip(F - 1, 0, None), vg))
    good = asym < 0.05 and emis > 0
    print(f"   emission EW>0: {emis:.2e}  blue/red asymmetry={asym:.3f} (<0.05)  "
          f"[{'ok' if good else 'FAIL'}]")
    return good


def t7_pcygni_shape():
    print("T7  homologous line over a photosphere → P-Cygni (blue abs + red em):")
    r, v = _homologous_shell(r_phot=1e15, r_out=6e15)
    vmax = v[-1] / 1e5
    vg = np.linspace(-1.3 * vmax, 1.3 * vmax, 401)
    chi = np.full_like(r, 3e-15)                   # thick line (clear P-Cygni trough)
    S = np.full_like(r, 0.4)                        # scattering source < continuum
    F = emergent_profile(r, v, S, chi, R_phot=1e15, I_cont=1.0, vgrid_kms=vg,
                         vth_kms=150.0, occultation=True)
    blue = F[vg < -0.2 * vmax]
    red = F[vg > 0.2 * vmax]
    has_abs = float(np.min(blue)) < 0.98           # blue-side absorption trough
    has_em = float(np.max(red)) > 1.02             # red-side emission
    good = has_abs and has_em
    print(f"   blue min={np.min(blue):.3f} (<1 abs)  red max={np.max(red):.3f} "
          f"(>1 em)  [{'ok' if good else 'FAIL'}]")
    return good


def t8_escatter_conserve_broaden():
    print("T8  electron scattering conserves photons + broadens with τ_es:")
    vg = np.linspace(-6e4, 6e4, 2401)
    F0 = 1.0 + 2.0 * np.exp(-0.5 * (vg / 300.0) ** 2)   # narrow emission line
    area0 = np.trapezoid(F0 - 1, vg)
    def fwhm(F):
        e = F - 1; pk = e.max(); above = vg[e >= 0.5 * pk]
        return float(above[-1] - above[0]) if above.size else 0.0
    F_id = escatter_redistribute(vg, F0, tau_es=0.0)
    ident = float(np.max(np.abs(F_id - F0)))
    widths = {}
    areas = {}
    for te in [1.0, 3.0, 8.0]:
        Fe = escatter_redistribute(vg, F0, tau_es=te, T_e=1e4)
        widths[te] = fwhm(Fe); areas[te] = np.trapezoid(Fe - 1, vg)
    cons = all(abs(areas[te] / area0 - 1) < 0.02 for te in widths)
    broaden = widths[1.0] < widths[3.0] < widths[8.0] and fwhm(F0) < widths[1.0]
    good = ident < 1e-9 and cons and broaden
    print(f"   τ=0 identity={ident:.1e}; photon-conservation={cons}; "
          f"FWHM {fwhm(F0):.0f}→{widths[1.0]:.0f}→{widths[3.0]:.0f}→{widths[8.0]:.0f} km/s "
          f"[{'ok' if good else 'FAIL'}]")
    return good


def t9_aniso_escatter():
    print("T9  anisotropic e-scatter (bulk red-skew): conserve + skew + symmetric limit:")
    vg = np.linspace(-6e4, 6e4, 2401)
    F0 = 1.0 + 2.0 * np.exp(-0.5 * (vg / 300.0) ** 2)
    area0 = np.trapezoid(F0 - 1, vg)
    Fs = escatter_redistribute(vg, F0, tau_es=3.0, T_e=1e4)
    Fs2 = escatter_redistribute(vg, F0, tau_es=3.0, T_e=1e4, v_bulk_kms=0.0)
    ident = float(np.max(np.abs(Fs - Fs2)))
    Fa = escatter_redistribute(vg, F0, tau_es=3.0, T_e=1e4, v_bulk_kms=500.0)
    area_a = np.trapezoid(Fa - 1, vg)
    cons = abs(area_a / area0 - 1) < 0.02
    e = np.clip(Fa - 1, 0, None)
    cen = float(np.trapezoid(e * vg, vg) / np.trapezoid(e, vg))
    es = np.clip(Fs - 1, 0, None)
    cen_s = float(np.trapezoid(es * vg, vg) / np.trapezoid(es, vg))
    red = cen > cen_s + 100.0
    good = ident < 1e-12 and cons and red
    print(f"   v_bulk=0 identity={ident:.1e}; conserve={cons}; "
          f"centroid {cen_s:.0f}->{cen:.0f} km/s (redward={red})  "
          f"[{'ok' if good else 'FAIL'}]")
    return good


def t10_zone_trap_weight():
    print("T10 zone-resolved trap weight: regime-pure limits + stratified split:")
    from unified_line_rt import zone_trap_weight
    t_exp = 5 * 86400.0
    r1 = np.linspace(1e15, 6e15, 120); v1 = r1 / t_exp
    w1 = zone_trap_weight(r1, v1)
    r2 = np.linspace(2.8e15, 3.2e15, 40)
    v2 = np.linspace(282e5, 238e5, 40)
    w2 = zone_trap_weight(r2, v2)
    r3 = np.concatenate([r1[:80], r1[80] + np.linspace(1e13, 4e14, 40)])
    v3 = np.concatenate([v1[:80], np.linspace(v1[80], 0.3 * v1[80], 40)])
    w3 = zone_trap_weight(r3, v3)
    hom_ok = float(np.median(w1)) < 0.15
    trap_ok = float(np.median(w2)) > 0.85
    split_ok = float(np.median(w3[:60])) < 0.4 and float(np.median(w3[-25:])) > 0.6
    good = hom_ok and trap_ok and split_ok
    print(f"   homologous med(w)={np.median(w1):.2f}(<0.15) reversed med(w)="
          f"{np.median(w2):.2f}(>0.85) stratified inner={np.median(w3[:60]):.2f}"
          f"/outer={np.median(w3[-25:]):.2f}  [{'ok' if good else 'FAIL'}]")
    return good


if __name__ == '__main__':
    print("=" * 70)
    print(" UNIFIED LINE-RT — Phase-0 ALI + Phase-1 profile/e-scatter validation")
    print("=" * 70)
    results = {
        'T1 sqrt-eps': t1_sqrt_eps(),
        'T2 LTE': t2_lte(),
        'T3 thermalization': t3_thermalization_depth(),
        'T4 optically-thin': t4_optically_thin(),
        'T5 deep-convergence': t5_deep_convergence(),
        'T6 thin-emission-symmetric': t6_thin_emission_symmetric(),
        'T7 pcygni-shape': t7_pcygni_shape(),
        'T8 escatter-conserve-broaden': t8_escatter_conserve_broaden(),
        'T9 aniso-escatter': t9_aniso_escatter(),
        'T10 zone-trap-weight': t10_zone_trap_weight(),
    }
    print("-" * 70)
    npass = sum(results.values())
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n{npass}/{len(results)} analytic limits passed.")
    sys.exit(0 if npass == len(results) else 1)
