"""probe_emis_inproc.py — call derive_parameters_from_state directly and read
the emission decomposition straight out of the returned dict.

This bypasses every failure mode of the in-pipeline env-var dump: no
SNLINE_DUMP_EMIS, no --no-iter branch, no bytecode cache, no question of which
internal return path runs. We call the function the SAME WAY production_runner
does (emission_model='caseb_hr_attenuated_split'), then read params['emissivity'],
params['n_p'], params['n_lower'/'n_upper'], params['tau_es'] — the arrays it
already returns — and recompute the recombination escape factor P_esc per zone.

Goal: confirm L_line ~ 1.21e42 reproduces here, and show whether the excess
lives in P_esc -> 1 (escape collapse) or in n_p.

Usage:
    python probe_emis_inproc.py mesa.day050_post_Lbol_max.data
"""
from __future__ import annotations
import argparse, sys
import numpy as np
from snline_autoparams import derive_parameters_from_state


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('snapshot')
    p.add_argument('--band-lo', type=float, default=6200.0)
    p.add_argument('--band-hi', type=float, default=6950.0)
    p.add_argument('--top', type=int, default=12)
    args = p.parse_args(argv)

    # Build the state/snap exactly as the pipeline does — reuse the pipeline's
    # own loader and state constructor so this is byte-identical to production.
    from production_runner import load_snapshot
    from snapshot_analyzer import analyze_snapshot
    # MATCH PRODUCTION: --no-shock-xray  => include_shock_xray=False
    snap = load_snapshot(args.snapshot, fmt='stella', verbose=True,
                         include_shock_xray=False)
    state = analyze_snapshot(snap)

    # --- test the contamination hypothesis directly ---
    # zone(s) the photoeq floor clamped to ~1e4 K have over-ionized n_e baked in.
    # Show, BEFORE touching anything, the n_e jump at the floored zone, then
    # report what L_line would be if that zone's n_e is restored to a value
    # consistent with its (un-floored) neighbors.
    Tf = np.asarray(snap['T'], dtype=float)
    Ts = np.asarray(snap.get('T_stella', snap['T']), dtype=float)
    ne0 = np.asarray(snap['n_e'], dtype=float)
    floored = np.where((Tf >= 0.999e4) & (Ts < Tf))[0]
    print(f"\n[hypothesis-test] photoeq-floored zones: {list(floored)}")
    for i in floored:
        nb = i + 1 if i + 1 < len(ne0) else i - 1
        print(f"  zone {i}: T={Tf[i]:.0f}->{Ts[i]:.0f}K  n_e={ne0[i]:.3e}  "
              f"(neighbor zone {nb}: T={Tf[nb]:.0f}K n_e={ne0[nb]:.3e}, "
              f"ratio n_e[{i}]/n_e[{nb}]={ne0[i]/ne0[nb]:.1f}x)")

    params = derive_parameters_from_state(
        state, snap, line_name='Halpha',
        wavelength_band_AA=(args.band_lo, args.band_hi),
        populations_mode='nlte', nlte_levels=5,
        emission_model='caseb_hr_attenuated_split',
        f_wing=0.02,
        ionization_mode='saha',
        f_HI_max=1.0,
        J_bar_Ha_abs=None,
        verbose=False,
    )

    L_line = float(params['L_line'])
    emis = np.asarray(params['emissivity'])
    n_p = np.asarray(params['n_p'])
    n_e = np.asarray(snap['n_e'])[:len(n_p)] if 'n_e' in snap else None
    n_2 = np.asarray(params.get('n_lower'))
    n_3 = np.asarray(params.get('n_upper'))
    tau_es = np.asarray(params.get('tau_es'))
    r = np.asarray(snap['r'])[:len(n_p)]
    T = np.asarray(snap['T'])[:len(n_p)]

    print("=" * 74)
    print(f" in-process probe — model=caseb_hr_attenuated_split")
    print("=" * 74)
    print(f" L_line (returned)   : {L_line:.4e} erg/s")
    print(f" sum(emissivity)     : {emis.sum():.4e} erg/s   (should match L_line)")
    print(f" L_phot              : {float(state.L_phot):.4e} erg/s")
    print(f" L_line / L_phot     : {L_line/float(state.L_phot):.3f}")
    print()
    # recompute the recombination escape factor P_esc the model used
    alpha = 1.17e-13 * (T / 1e4) ** -0.942
    HPL = 6.62607015e-27; CC = 2.99792458e10; LAM = 6562.81e-8
    nu = CC / LAM
    drz = np.empty_like(r); drz[:-1] = np.diff(r); drz[-1] = drz[-2]
    dV = 4 * np.pi * r * r * drz
    eps_recomb_gross = n_p * (n_e if n_e is not None else n_p) * alpha * HPL * nu * dV
    # the model emissivity divided by the gross recomb gives P_esc*kernel_factor
    with np.errstate(divide='ignore', invalid='ignore'):
        eff_factor = np.where(eps_recomb_gross > 0, emis / eps_recomb_gross, 0.0)
    print(f" sum eps_recomb_gross (n_p·n_e·α·hν·dV): {eps_recomb_gross.sum():.4e} erg/s")
    print(f" implied mean (P_esc × kernel) = L_line/gross = {L_line/max(eps_recomb_gross.sum(),1e-30):.3f}")
    print()
    order = np.argsort(emis)[::-1]
    print(f" Top {args.top} emitting zones:")
    print(f"   {'idx':>4} {'r/Rp':>6} {'T':>6} {'n_e':>9} {'n_p':>9} {'tau_es':>8} "
          f"{'P_esc*K':>8} {'L_frac':>7}")
    Rp = float(state.R_phot)
    for k in range(min(args.top, len(r))):
        i = order[k]
        print(f"   {i:>4d} {r[i]/Rp:>6.3f} {T[i]:>6.0f} "
              f"{(n_e[i] if n_e is not None else 0):>9.2e} {n_p[i]:>9.2e} "
              f"{tau_es[i]:>8.2f} {eff_factor[i]:>8.3f} {emis[i]/max(emis.sum(),1e-30):>7.1%}")
    print("=" * 74)
    print(" zone-0 contamination check (T, n_e, rho vs neighbors):")
    rho_arr = np.asarray(snap['rho'])[:len(n_p)]
    for i in range(min(5, len(r))):
        print(f"   zone {i}: r/Rp={r[i]/float(state.R_phot):.4f}  "
              f"T={T[i]:.1f} K  n_e={(n_e[i] if n_e is not None else 0):.3e}  "
              f"rho={rho_arr[i]:.3e}")
    print(f"   state.T_phot = {float(state.T_phot):.1f} K   "
          f"snap['T_phot_inner'] = {snap.get('T_phot_inner', 'n/a')}")
    print(f"   snap['T'][0] (raw zone array) = {np.asarray(snap['T'])[0]:.1f} K")
    print("=" * 74)
    # --- decisive test: re-solve floored-zone n_e PROPERLY (Saha at its true
    #     STELLA temperature, not a neighbor-copy), re-derive, and check the
    #     transported L_line. If it lands ~5e39 (Phase-5), the kernel transports
    #     a clean emissivity to the physical value and the re-solve is the fix.
    if len(floored) > 0:
        ME=9.1093837e-28; KB=1.380649e-16; HPL=6.62607015e-27
        CHI=13.5984*1.602176634e-12; MP=1.6726219e-24
        def saha_ne(T, n_H):
            # n_HII*n_e/n_HI = S ; assume H-only donor => n_e=n_HII, solve quadratic
            kT=KB*T
            S=(2*np.pi*ME*kT/HPL**2)**1.5*np.exp(-CHI/kT)
            # x = n_HII/n_H : x^2/(1-x) = S/n_H
            a=S/max(n_H,1e-30)
            x=(-a+np.sqrt(a*a+4*a))/2.0
            return np.clip(x,0,1)*n_H
        snap2 = dict(snap)
        ne_fixed = np.asarray(snap['n_e'], dtype=float).copy()
        T_fixed = np.asarray(snap['T'], dtype=float).copy()
        rho_arr2 = np.asarray(snap['rho'], dtype=float)
        XH = np.asarray(snap.get('X_H', 0.638*np.ones_like(ne_fixed)), dtype=float)
        if np.ndim(XH)==0: XH=float(XH)*np.ones_like(ne_fixed)
        for i in floored:
            n_H_i = XH[i]*rho_arr2[i]/MP
            T_true = Ts[i]
            ne_resolved = saha_ne(T_true, n_H_i)
            T_fixed[i] = T_true
            ne_fixed[i] = ne_resolved
            print(f"  re-solve zone {i}: T->{T_true:.0f}K  n_H={n_H_i:.2e}  "
                  f"n_e: {np.asarray(snap['n_e'])[i]:.3e} -> {ne_resolved:.3e} (Saha)")
        snap2['n_e'] = ne_fixed
        snap2['T'] = T_fixed
        state2 = analyze_snapshot(snap2)
        params2 = derive_parameters_from_state(
            state2, snap2, line_name='Halpha',
            wavelength_band_AA=(args.band_lo, args.band_hi),
            populations_mode='nlte', nlte_levels=5,
            emission_model='caseb_hr_attenuated_split', f_wing=0.02,
            ionization_mode='saha', f_HI_max=1.0, J_bar_Ha_abs=None, verbose=False)
        L2 = float(params2['L_line'])
        print(f" L_line with floored zone re-solved (Saha): {L2:.4e} erg/s "
              f"({L2/float(state.L_phot):.4f} x L_phot)")
        print(f"   original {L_line:.3e} | neighbor-swap gave ~5e38 | Phase-5 ~5.5e39")
        print("=" * 74)

        # --- zone-0-exclusion variant: drop the photosphere boundary zone(s)
        #     entirely from the line emissivity. That zone IS the tau=2/3
        #     surface (its emission is continuum, not line); the line should
        #     form in the zones above it. This removes the knife-edge n_e
        #     dependence. Report the resulting L_line from the SAME model.
        # We exclude by setting the floored zone's n_e (hence n_e*n_p) to 0,
        # which zeroes its recombination emissivity contribution.
        snap3 = dict(snap)
        ne_excl = np.asarray(snap['n_e'], dtype=float).copy()
        T_excl = np.asarray(snap['T'], dtype=float).copy()
        for i in floored:
            ne_excl[i] = 0.0          # drop this zone from the emissivity
            T_excl[i] = Ts[i]         # (T value now irrelevant to emis, but un-floor it)
        snap3['n_e'] = ne_excl
        snap3['T'] = T_excl
        state3 = analyze_snapshot(snap3)
        params3 = derive_parameters_from_state(
            state3, snap3, line_name='Halpha',
            wavelength_band_AA=(args.band_lo, args.band_hi),
            populations_mode='nlte', nlte_levels=5,
            emission_model='caseb_hr_attenuated_split', f_wing=0.02,
            ionization_mode='saha', f_HI_max=1.0, J_bar_Ha_abs=None, verbose=False)
        L3 = float(params3['L_line'])
        # also report the top emitting zone now, to see where emission sits
        emis3 = np.asarray(params3['emissivity'])
        top3 = int(np.argmax(emis3))
        print(f" L_line with photosphere zone EXCLUDED: {L3:.4e} erg/s "
              f"({L3/float(state.L_phot):.5f} x L_phot)")
        print(f"   now top zone = {top3} at r/Rp={r[top3]/float(state.R_phot):.3f}, "
              f"carries {emis3[top3]/max(emis3.sum(),1e-30):.1%}")
        print(f"   reference: Phase-5 transported Hα ~5.5e39 (0.005 x L_phot)")
        print("=" * 74)
    return 0


if __name__ == '__main__':
    sys.exit(main())
