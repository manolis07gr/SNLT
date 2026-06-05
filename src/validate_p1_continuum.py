"""
validate_p1_continuum.py — Analytic-limit checks for P1 #4 (composition-general
continuum guard + He-budget diagnostics). Imported by validate_p1_physics.py;
can also be run standalone:  python validate_p1_continuum.py
"""
from __future__ import annotations
import numpy as np
import continuum_compgen as cg


def run(check):
    """Run all P1 #4 checks via the shared `check(name, cond, detail)` callback."""
    SIGMA_SB = cg.SIGMA_SB

    # 1. color_temperature_floor roundtrips: 4πR²σT_floor⁴ == L_phot
    print("color_temperature_floor energy roundtrip")
    R, L = 3.0e14, 1.0e42
    Tf = cg.color_temperature_floor(R, L)
    L_back = cg.bb_bolometric(Tf, R)
    check(f"  T_floor={Tf:.0f}K → 4πR²σT⁴/L_phot = {L_back / L:.4f}",
          abs(L_back / L - 1.0) < 1e-3)

    # 2. collapse detection: cold/compact triggers, warm/consistent does not
    print("detect_continuum_collapse")
    cold = cg.detect_continuum_collapse(2500.0, 3.0e14, 1.0e42)
    warm = cg.detect_continuum_collapse(8000.0, 1.2e15,
                                        cg.bb_bolometric(8000.0, 1.2e15))
    check(f"  cold/compact collapsed={cold['collapsed']} (ratio={cold['ratio']:.1e})",
          cold['collapsed'] is True)
    check(f"  warm/consistent collapsed={warm['collapsed']} (ratio={warm['ratio']:.2f})",
          warm['collapsed'] is False)

    # 3. guarded_L_cont_band raises a collapsed continuum (T_floor > T_phot),
    #    and more strongly in the UV (Wien) than the NIR.
    print("guarded_L_cont_band flooring")
    Lc = 1.0e36
    g_nir = cg.guarded_L_cont_band(Lc, 10830.0, 2500.0, Tf)
    g_uv = cg.guarded_L_cont_band(Lc, 1640.0, 2500.0, Tf)
    check(f"  NIR ×{g_nir / Lc:.2e}, UV ×{g_uv / Lc:.2e}  (both >1, UV ≫ NIR)",
          g_nir > Lc and g_uv > g_nir)

    # 4. apply_continuum_guard mutates spectra, preserves raw, no-op when warm
    print("apply_continuum_guard")
    spec = {'He_I_10830': {'lambda_rest': 10830.0, 'L_cont_band': 1e36,
                           'L_line': 1e40},
            'He_II_1640': {'lambda_rest': 1640.0, 'L_cont_band': 1e30,
                           'L_line': 1e34}}
    info = cg.apply_continuum_guard(spec, 2500.0, 3.0e14, 1.0e42, verbose=False)
    raised = spec['He_I_10830']['L_cont_band'] > spec['He_I_10830']['L_cont_band_raw']
    check(f"  collapsed→ L_cont raised & raw preserved ({raised})",
          info['collapsed'] and raised
          and spec['He_I_10830']['L_cont_band_raw'] == 1e36)
    spec2 = {'X': {'lambda_rest': 5000.0, 'L_cont_band': 1e38, 'L_line': 1e36}}
    info2 = cg.apply_continuum_guard(spec2, 8000.0, 1.2e15,
                                     cg.bb_bolometric(8000.0, 1.2e15),
                                     verbose=False)
    check(f"  warm→ no-op (L_cont unchanged={spec2['X']['L_cont_band'] == 1e38})",
          (not info2['collapsed']) and spec2['X']['L_cont_band'] == 1e38)

    # 5. energy_conservation_check flags Σ L_line > L_phot
    print("energy_conservation_check")
    over = cg.energy_conservation_check(
        {'a': {'L_line': 8e42}, 'b': {'L_line': 8e42}}, 1.0e43, verbose=False)
    under = cg.energy_conservation_check(
        {'a': {'L_line': 1e41}}, 1.0e43, verbose=False)
    check(f"  Σ>L_phot flagged not-ok ({not over['ok']}), Σ<L_phot ok ({under['ok']})",
          (not over['ok']) and under['ok'])

    # 6. composition switch
    print("composition switch (is_h_free / element_present)")
    check("  X_H=1e-5 → H-free, X_H=0.7 → not; element_present(None)=True",
          cg.is_h_free(1e-5) and (not cg.is_h_free(0.7))
          and cg.element_present(None) and (not cg.element_present(1e-5)))

    # 7. He free-free opacity (P1 #4 root-fix): Z² scaling, vanishes for neutral.
    # NOTE: kappa_ff_* expect PER-ZONE arrays for T/n_e/n_ion (as the pipeline
    # always passes), so use 1-element arrays here.
    print("kappa_ff_He (helium free-free)")
    try:
        import opacity as _op
        lam = 6562.8
        T = np.array([1.0e4]); ne = np.array([1.0e10]); n = np.array([1.0e10])
        z = np.zeros(1)
        kH = float(np.atleast_1d(_op.kappa_ff_H(lam, T, ne, n)).ravel()[0])
        kHe2 = float(np.atleast_1d(_op.kappa_ff_He(lam, T, ne, n, z)).ravel()[0])
        kHe3 = float(np.atleast_1d(_op.kappa_ff_He(lam, T, ne, z, n)).ravel()[0])
        kHe0 = float(np.atleast_1d(_op.kappa_ff_He(lam, T, ne, z, z)).ravel()[0])
        check(f"  He II ff == H ff ({kHe2 / kH:.3f}), He III = 4× ({kHe3 / kH:.3f}), "
              f"neutral→0 ({kHe0:.0e})",
              abs(kHe2 / kH - 1.0) < 1e-6 and abs(kHe3 / kH - 4.0) < 1e-6
              and kHe0 == 0.0)
        # default kappa_cont_total (no He-ff) is unchanged → H-rich byte-identical
        base = _op.kappa_cont_total(lam, T, ne, n, np.zeros((5, 1)))
        withhe = _op.kappa_cont_total(lam, T, ne, n, np.zeros((5, 1)),
                                      include=('es', 'bf', 'H-', 'ff', 'He-ff'),
                                      n_HeII=n)
        check(f"  default total excludes He-ff; +He-ff raises κ "
              f"({float(withhe[0]) > float(base[0])})",
              float(withhe[0]) > float(base[0]))
    except Exception as e:
        check(f"  opacity He-ff import/run ({e})", False)


if __name__ == '__main__':
    fails = []

    def _check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)
    print("P1 #4 continuum/budget checks")
    run(_check)
    import sys
    sys.exit(1 if fails else 0)
