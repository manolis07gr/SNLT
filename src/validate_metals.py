"""
validate_metals.py — Analytic-limit checks for P2 #5 (C/O/Ne metal lines).
Imported by validate_p1_physics.py; also runnable standalone:
    python validate_metals.py

These check the PHYSICS (emissivity limits, n_crit suppression, ion-ladder
normalisation, ionization monotonicity) — NOT the provisional atomic NUMBERS,
which must be verified against CHIANTI/Cloudy separately.
"""
from __future__ import annotations
import numpy as np
import metal_atoms as ma
import metal_ionization as mi


def run(check):
    one = np.array([1.0])

    # 1. CEL n_crit limits: low-density ∝ n_e; high-density plateau (thermalised)
    print("CEL emissivity n_crit limits (low- & high-density)")
    name = 'O_III_5007'
    T = np.array([1.0e4])
    n_ion = one
    nc = float(np.atleast_1d(ma.critical_density(name, T))[0])
    ne_lo = np.array([nc * 1e-4])
    ne_hi = np.array([nc * 1e4])
    j_lo = float(ma.cel_emissivity(name, T, ne_lo, n_ion)[0])
    j_lo2 = float(ma.cel_emissivity(name, T, ne_lo * 2.0, n_ion)[0])
    j_hi = float(ma.cel_emissivity(name, T, ne_hi, n_ion)[0])
    j_hi2 = float(ma.cel_emissivity(name, T, ne_hi * 2.0, n_ion)[0])
    lin = abs(j_lo2 / j_lo - 2.0) < 0.02            # doubles with n_e (low ρ)
    plateau = abs(j_hi2 / j_hi - 1.0) < 0.02        # n_e-independent (high ρ)
    check(f"  n_crit({name})={nc:.2e}; low-ρ ∝n_e ({j_lo2/j_lo:.2f}), "
          f"high-ρ plateau ({j_hi2/j_hi:.2f})", lin and plateau)

    # 2. recombination emissivity scales ∝ n_e·n_ion
    print("recombination emissivity scaling")
    jr1 = float(ma.recomb_emissivity('C_IV_1549', T, one, one)[0])
    jr2 = float(ma.recomb_emissivity('C_IV_1549', T, 2.0 * one, 3.0 * one)[0])
    check(f"  ∝ n_e·n_ion ratio = {jr2/jr1:.2f} (expect 6.0)",
          abs(jr2 / jr1 - 6.0) < 1e-6)

    # 3. ion-ladder fractions are normalised and physical
    print("ion_ladder_fractions normalisation")
    Tz = np.full(4, 1.0e4)
    ne = np.full(4, 1.0e8)
    gam = {'C_I': np.full(4, 1e-6), 'C_II': np.full(4, 1e-7),
           'C_III': np.full(4, 1e-8)}
    fr = ma.ion_ladder_fractions('C', gam, Tz, ne)
    tot = sum(fr.values())
    check(f"  Σ fractions = {float(np.mean(tot)):.4f} (expect 1.0), all ≥0",
          np.allclose(tot, 1.0, atol=1e-6)
          and all(float(np.min(f)) >= -1e-12 for f in fr.values()))

    # 4. stronger radiation field → higher ionization (more C IV)
    print("ionization monotonic in radiation field")
    g_cold = mi.gamma_unit_rates('C_III', 8000.0, 0.0, 0.0, 1e15)
    g_hot = mi.gamma_unit_rates('C_III', 30000.0, 0.0, 0.0, 1e15)
    check(f"  Γ_unit(C_III): hot {g_hot:.2e} > cold {g_cold:.2e}",
          g_hot > g_cold >= 0.0)
    g_xray = mi.gamma_unit_rates('C_III', 8000.0, 1e44, 1e9, 1e15)
    check(f"  shock X-ray adds ionization: {g_xray:.2e} > BB-only {g_cold:.2e}",
          g_xray > g_cold)

    # 5. number density from mass fraction
    print("number_density")
    n = ma.number_density('O', np.array([0.5]), np.array([2.0e-15]))
    expect = 0.5 * 2.0e-15 / ma.M_ATOM['O']
    check(f"  n_O = {float(n[0]):.3e} (expect {expect:.3e})",
          abs(float(n[0]) / expect - 1.0) < 1e-9)

    # 6. full ionization pass: fractions sum to 1 per element, n_ion ≤ n_elem
    print("compute_metal_ionization end-to-end")
    nz = 6
    Tz = np.linspace(5e3, 1.5e4, nz)
    ne = np.full(nz, 1.0e9)
    r = np.linspace(2e15, 1e16, nz)
    rho = np.full(nz, 1e-15)
    ions = mi.compute_metal_ionization(
        Tz, ne, r, rho, R_phot=1.5e15, T_phot=1.2e4,
        X_C=np.full(nz, 1e-3), X_O=np.full(nz, 0.5), X_Ne=np.full(nz, 0.1),
        L_X_brems=1e43, T_shock=1e9, verbose=False)
    n_O_tot = ma.number_density('O', np.full(nz, 0.5), rho)
    o_sum = ions['O_I'] + ions['O_II'] + ions['O_III']
    check(f"  Σ O stages / n_O_tot = {float(np.mean(o_sum/n_O_tot)):.4f} (expect 1.0)",
          np.allclose(o_sum, n_O_tot, rtol=1e-4))


if __name__ == '__main__':
    fails = []

    def _check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            fails.append(name)
    print("P2 #5 metal-line physics checks")
    run(_check)
    import sys
    sys.exit(1 if fails else 0)
