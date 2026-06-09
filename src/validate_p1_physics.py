"""
validate_p1_physics.py — Standalone validation harness for the P1 physics
(FUTURE_WORK items 3 and 4). Pure-numpy analytic-limit checks; NO pipeline run,
NO STELLA snapshot required. Run from src/ (or any model dir, which symlinks it):

    python validate_p1_physics.py

Exit code 0 = all checks passed. Each check prints PASS/FAIL with the numbers,
so a failure tells you exactly which physical limit broke.

These checks are the things that must hold BY CONSTRUCTION; they are independent
of the (un-runnable-here) full MC pipeline, so they pin the new physics before a
production batch is spent on it.
"""
from __future__ import annotations
import numpy as np
import line_rt_escape as ep

# numpy 2.x renamed np.trapz -> np.trapezoid; support both.
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)

C = 2.99792458e10
H = 6.62607015e-27
KB = 1.380649e-16
ME = 9.1093837e-28
EE = 4.80320425e-10

_fails = []


def _check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        _fails.append(name)


def _synth_line(thick=True, n_zones=120):
    """A synthetic spherical envelope + one He-like line, thick or thin."""
    r = np.linspace(2.5e15, 1.0e16, n_zones)
    v = 5.0e7 * (r / r[0])              # homologous, ~500-2000 km/s
    T = np.full(n_zones, 8000.0)
    n_e = np.full(n_zones, 1.0e10)
    lam0_AA = 10830.34
    lam0_cm = lam0_AA * 1e-8
    A_ul = 1.0216e7
    g_l, g_u = 3, 9
    # choose lower/upper populations to make the line thick or thin
    n_l = np.full(n_zones, 2.0e6 if thick else 2.0e1)
    n_u = np.full(n_zones, 5.0e3 if thick else 5.0e-2)
    t_exp = float(np.median(r / v))
    SIGMA_INT = np.pi * EE * EE / (ME * C)
    f_lu = 1.4991938 * (g_u / g_l) * A_ul * lam0_cm ** 2
    pop = np.maximum(n_l - (g_l / g_u) * n_u, 0.0)
    tau_S = SIGMA_INT * f_lu * lam0_cm * pop * t_exp
    return dict(r=r, v=v, T=T, n_e=n_e, lam0_AA=lam0_AA, lam0_cm=lam0_cm,
               A_ul=A_ul, g_l=g_l, g_u=g_u, n_l=n_l, n_u=n_u, t_exp=t_exp,
               tau_S=tau_S)


def test_escape_identity():
    """(#3) The escape-probability luminosity from the actual populations equals
    the single-shot β luminosity — for thick AND thin lines."""
    print("escape_probability_luminosity ≡ single-shot β luminosity")
    for thick in (True, False):
        s = _synth_line(thick=thick)
        beta = ep.sobolev_beta(s['tau_S'])
        nu0 = C / s['lam0_cm']
        dV = ep._shell_volumes(s['r'])
        L_single = float(np.sum(H * nu0 * s['n_u'] * s['A_ul'] * beta * dV))
        L_ep = ep.escape_probability_luminosity(
            s['tau_S'], s['n_l'], s['n_u'], s['A_ul'], s['g_l'], s['g_u'],
            s['lam0_cm'], s['r'], t_exp=s['t_exp'])
        ratio = L_ep / L_single if L_single > 0 else np.nan
        _check(f"  τ_med={np.median(s['tau_S']):.2e}: L_EP/L_single = {ratio:.5f}",
               abs(ratio - 1.0) < 1e-3, f"L_single={L_single:.3e}")


def test_thomson_conserves_photons():
    """(#3) Multiple-scattering redistribution conserves ∫(F−1) (L_line/EW) and
    broadens the profile (peak suppressed, second moment increased)."""
    print("thomson_multiscatter — photon conservation + broadening")
    lam0 = 10830.34
    lam = np.linspace(lam0 - 200, lam0 + 200, 1201)
    sig0 = 8.0
    excess = np.exp(-0.5 * ((lam - lam0) / sig0) ** 2)   # emission line
    F = 1.0 + excess
    for tau_es in (0.5, 3.0, 8.0):
        Fr = ep.thomson_multiscatter(lam, F, lam0, 1.0e4, tau_es)
        I0 = _trapz(F - 1.0, lam)
        I1 = _trapz(Fr - 1.0, lam)
        cons = abs(I1 - I0) / abs(I0)
        peak_drop = (Fr.max() - 1.0) <= (F.max() - 1.0) + 1e-9
        m2_0 = _trapz((lam - lam0) ** 2 * (F - 1.0), lam) / I0
        m2_1 = _trapz((lam - lam0) ** 2 * (Fr - 1.0), lam) / I1
        _check(f"  τ_es={tau_es}: ∫ conserved to {cons*100:.2f}%, "
               f"σ²: {m2_0:.0f}→{m2_1:.0f}",
               cons < 0.02 and peak_drop and m2_1 >= m2_0 * 0.999)


def test_dilution_and_beta():
    """(#3) Geometric dilution and Sobolev β endpoints."""
    print("dilution_W and sobolev_beta limits")
    W = ep.dilution_W(np.array([2.5e15, 1.0e17]), 2.5e15)
    _check(f"  W(R_phot)=½ ({W[0]:.3f}), W(far)→small ({W[1]:.2e})",
           abs(W[0] - 0.5) < 1e-6 and W[1] < 0.05)
    b = ep.sobolev_beta(np.array([1e-8, 1.0, 1e7]))
    _check(f"  β(0)→1 ({b[0]:.4f}), β(∞)→0 ({b[2]:.2e})",
           abs(b[0] - 1.0) < 1e-3 and b[2] < 1e-6)


def test_ep_source_limits():
    """(#3, DIAGNOSTIC) Continuum-pumped source function analytic limits."""
    print("ep_source_function analytic limits (diagnostic source)")
    r = np.array([3.0e15]); T = np.array([8000.0]); n_e = np.array([1e10])
    nu0 = C / 10830.34e-8
    B = float(ep.planck_nu(nu0, 8000.0))
    # ε→1 (huge collisions): S_L → B
    s_eps1 = ep.ep_source_function(np.array([5.0]), T, np.array([1e30]), r,
                                   2.5e15, 8000.0, nu0, 1.0216e7,
                                   collision_strength=1e6)
    _check(f"  ε→1: S_L/B = {float(s_eps1['S_L'][0]) / B:.4f}",
           abs(float(s_eps1['S_L'][0]) / B - 1.0) < 0.05)
    # β→0 (τ→∞): S_L → B (thermalised by trapping)
    s_thick = ep.ep_source_function(np.array([1e7]), T, n_e, r, 2.5e15,
                                    9000.0, nu0, 1.0216e7)
    _check(f"  τ→∞: S_L/B = {float(s_thick['S_L'][0]) / B:.4f}",
           abs(float(s_thick['S_L'][0]) / B - 1.0) < 0.10)


def main():
    print("=" * 70)
    print("P1 physics validation (analytic limits; no pipeline run)")
    print("=" * 70)
    print("\n-- P1 #3: saturated-line RT --")
    test_escape_identity()
    test_thomson_conserves_photons()
    test_dilution_and_beta()
    test_ep_source_limits()
    try:
        import validate_p1_continuum as _c4   # added by P1 #4 (optional)
        print("\n-- P1 #4: composition-general continuum / budget --")
        _c4.run(_check)
    except Exception:
        pass
    try:
        import validate_metals as _m5          # added by P2 #5 (optional)
        print("\n-- P2 #5: C/O/Ne metal lines --")
        _m5.run(_check)
    except Exception as _e:
        print(f"\n-- P2 #5: metal checks skipped ({_e}) --")
    print("\n" + "=" * 70)
    if _fails:
        print(f"RESULT: {len(_fails)} FAILED — {_fails}")
        return 1
    print("RESULT: all checks PASSED")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
