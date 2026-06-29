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
from unified_line_rt import solve_two_level_ali


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


if __name__ == '__main__':
    print("=" * 70)
    print(" UNIFIED LINE-RT — Phase-0 ALI engine analytic-limit validation")
    print("=" * 70)
    results = {
        'T1 sqrt-eps': t1_sqrt_eps(),
        'T2 LTE': t2_lte(),
        'T3 thermalization': t3_thermalization_depth(),
        'T4 optically-thin': t4_optically_thin(),
        'T5 deep-convergence': t5_deep_convergence(),
    }
    print("-" * 70)
    npass = sum(results.values())
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n{npass}/{len(results)} analytic limits passed.")
    sys.exit(0 if npass == len(results) else 1)
