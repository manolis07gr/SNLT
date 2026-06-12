"""
phase5_runner.py — Phase 5 high-level driver and output writer
================================================================================

One-call entry point for Phase 5: takes a populated state object (post Phase
3 + Phase 4) and produces all Phase 5 deliverables:
    {out_prefix}_he.npz   — per-line spectra (lambda, F_lambda, F_norm, etc.)
    {out_prefix}_he.png   — 8-panel figure of all He profiles
    {out_prefix}_he.txt   — summary table (L_line, EW, peak F, peak v per line)

Designed as a single-line hook for production_runner.py:

    if args.he_lines:
        from phase5_runner import run_phase5_for_state
        he_spectra = run_phase5_for_state(
            state, snap, n_packets=args.he_lines_n_packets,
            calibration=args.he_lines_calibration,
            out_prefix=out_prefix, verbose=args.verbose,
        )
        result['he_spectra'] = he_spectra

The runner respects the existing convention that hydrodynamic arrays (r, v,
rho, T, n_e) live in `snap` dict while NLTE diagnostics (he1_levels, he1_tau,
he2_levels, he2_tau) are attached to `state`. The wrapper merges the two
into a SimpleNamespace that mc_multi_line.compute_phase5_spectra can consume.
"""
from __future__ import annotations
import os
import types
import numpy as np
try:
    import formal_line_profile as _flp_p5
    _HAVE_FORMAL_P5 = True
except Exception:
    _HAVE_FORMAL_P5 = False


def run_phase5_for_state(state, snap, n_packets: int = 50000,
                          win_kms: float = 5000.0, n_pix: int = 501,
                          use_existing_peel_kernel: bool = True,
                          calibration: str = 'theoretical_ew',
                          lines: list = None,
                          out_prefix: str = None,
                          make_png: bool = True,
                          production_halpha: dict = None,
                          profile_method: str = 'mc',
                          lock: bool = False,
                          saturated_rt: bool = False,
                          he_budget: bool = False,
                          metal_lines: bool = False,
                          metal_cloudy: bool = False,
                          narrow_csm: bool = False,
                          verbose: bool = True) -> dict:
    """Run Phase 5 (multi-line MC) on a state populated by Phases 3 + 4.

    Args:
        state: PhysicalState (or SimpleNamespace) with he1_levels, he1_tau,
               he2_levels, he2_tau attributes set by Phase 3/4 wrappers.
        snap:  hydrodynamic snapshot dict with r, v, rho, T, n_e keys plus
               R_phot_cm, T_phot, L_phot (or T_color, R_phot fallbacks).
        n_packets: MC packets per line (default 50,000)
        win_kms: ± velocity window per line (default 5000 km/s)
        n_pix: pixels per line (default 501)
        use_existing_peel_kernel:
            True  — call peel_pipeline_abs.run_peel_pipeline_abs for each line
                    (uses the same kernel as Hα; gives P-Cygni structure).
            False — use the reference Sobolev MC in mc_multi_line (emission
                    only, no P-Cygni; faster, useful for debugging/QA).
        calibration: 'theoretical_ew' (default, matches Hα auto-selection
                     for L_line/L_cont < 0.3), 'f_cont_bb', 'absolute'
        lines: list of line names to compute (default: all 8)
        out_prefix: file path prefix for outputs. If None, no files written.
        make_png: whether to render the 8-panel summary figure
        verbose: print per-line diagnostics

    Returns:
        spectra dict (see mc_multi_line.compute_phase5_spectra docstring)
    """
    import mc_multi_line as p5

    # Build the merged state expected by Phase 5
    merged = _build_merged_state(state, snap)
    _validate_state_for_phase5(merged)

    # Adaptive velocity window: a FIXED ±win_kms clips broad, blueshifted
    # dense-CSM lines (v_max can far exceed 5000 km/s) AND starves the F/F_cont
    # baseline — the far-wing mean used to normalise the profile then falls
    # INSIDE the line, pinning the apparent continuum well below 1. Widen to the
    # emission-measure-weighted (n_e^2·dV) 95th-percentile velocity of the
    # line-forming gas (× a margin), so the window captures the full line plus a
    # genuine continuum shoulder. Never NARROWER than the requested win_kms, so
    # slow/homologous models (A1/B1-like) are unchanged; capped for sanity.
    try:
        _v = np.abs(np.asarray(merged.v, float)) / 1e5
        _ne = np.asarray(merged.n_e, float)
        _r = np.asarray(merged.r, float)
        _edge = np.empty(_r.size + 1)
        _edge[1:-1] = 0.5 * (_r[:-1] + _r[1:])
        _edge[0] = _r[0]; _edge[-1] = _r[-1] + (_r[-1] - _r[-2])
        _dV = (4.0 / 3.0) * np.pi * (_edge[1:] ** 3 - _edge[:-1] ** 3)
        _w = _ne ** 2 * _dV
        _w = np.where(np.isfinite(_w) & (_w > 0), _w, 0.0)
        if _w.sum() > 0 and _v.size > 1:
            _o = np.argsort(_v)
            _cv = np.cumsum(_w[_o]); _cv = _cv / _cv[-1]
            _v95 = float(_v[_o][min(int(np.searchsorted(_cv, 0.95)), _v.size - 1)])
            win_kms_eff = float(np.clip(1.6 * _v95, win_kms, 25000.0))
        else:
            win_kms_eff = win_kms
    except Exception:
        win_kms_eff = win_kms
    # keep the spectral resolution (~km/s per pixel) roughly constant as the
    # window widens, capped for performance.
    if win_kms_eff > win_kms + 1.0 and win_kms > 0:
        n_pix = int(np.clip(round(n_pix * win_kms_eff / win_kms), n_pix, 1201))
    if verbose and win_kms_eff > win_kms + 1.0:
        print(f"[phase5] adaptive velocity window ±{win_kms_eff:.0f} km/s "
              f"(emission-measure v95-driven; requested ±{win_kms:.0f}), n_pix={n_pix} "
              f"— captures the full broad/blueshifted line + a continuum shoulder.")
    win_kms = win_kms_eff

    # Sobolev-validity gate: the per-line formal source-function solution applies
    # only to homologous ejecta (IIP-like). For a dense, non-homologous CSM
    # (IIn-like) it is the wrong regime (the lines are recombination emission +
    # electron scattering), so fall back to the MC kernel (with the empirical
    # correction). This keeps both regimes physical.
    requested_method = profile_method   # what the user asked for (drives strengths)
    if profile_method == 'formal' and _HAVE_FORMAL_P5 and not lock:
        try:
            sv = _flp_p5.sobolev_validity(np.asarray(merged.r, float),
                                          np.asarray(merged.v, float))
            if not sv['valid']:
                if verbose:
                    print(f"[phase5] formal source-function SKIPPED — {sv['reason']}.")
                    print(f"[phase5] Using MC kernel for the SHAPE; line STRENGTHS "
                          f"still come from the per-line recombination budget "
                          f"(no empirical anchor).")
                profile_method = 'mc'        # shape only; strengths handled below
        except Exception:
            pass

    # Choose the peel callback
    if use_existing_peel_kernel:
        def peel_callback(s, line_inputs, lam_grid, n_pkt):
            return p5.peel_pipeline_abs_callback(
                s, line_inputs, lam_grid, n_packets=n_pkt,
                calibration=calibration, verbose=False)
    else:
        peel_callback = None    # use reference Sobolev MC

    # Run the MC
    spectra = p5.compute_phase5_spectra(
        merged, lines=lines, n_packets=n_packets,
        win_kms=win_kms, n_pix=n_pix,
        peel_callback=peel_callback, profile_method=profile_method,
        verbose=verbose)

    # ---- Phase 5b: line strengths ----
    # The STRENGTH method is keyed to what the user REQUESTED, independent of
    # whether the gate downgraded the profile SHAPE to MC. So a IIn run with
    # --line-profile-method formal gets MC shapes (correct for the regime) but
    # per-line recombination-budget strengths (no empirical Hα anchor).
    if requested_method == 'formal':
        _apply_recombination_budget(spectra, merged, production_halpha,
                                    verbose=verbose, saturated_rt=saturated_rt)
        correction_factor = None
    else:
        # ---- Phase 5b-empirical: Hα-anchored systematic correction (legacy) ----
        # The He NLTE solvers do not yet accept external J_bar arrays, so Phase 5
        # uses a single-shot Sobolev β per line. The empirical Hα anchor scales
        # that to the validated production Hα. Conservative legacy default.
        correction_factor = _compute_empirical_correction(spectra, production_halpha)
        if correction_factor is not None and verbose:
            _label = _correction_label()
            print(f"[phase5b] H\u03b1-anchored correction factor R = "
                  f"{_label} = {correction_factor:.4f}")
            print(f"[phase5b] Applying R to L_line, EW, (F_norm - 1) for all "
                  f"non-H\u03b1 lines.")
        _apply_empirical_correction(spectra, correction_factor)

    # ---- Phase 5b-compgen (P1 #4): composition-general continuum guard ----
    # Corrects the per-line continuum LEVEL (energy conservation) so that the
    # L_corr / L_cont_band EW estimate is physical for H-free / cold-compact
    # photospheres where the diluted-BB L_cont_band collapses (Wien). NEVER
    # touches the profile shape. Opt-in via --he-budget; AUTO when X_H < 1e-3.
    try:
        import continuum_compgen as _cg
        _x_h = _cg.mean_X_H(merged)
        _h_free = _cg.is_h_free(_x_h)
        if he_budget or _h_free:
            if verbose and _h_free and not he_budget:
                print(f"[phase5b/compgen] H-free gas (⟨X_H⟩={_x_h:.1e}) — "
                      f"composition-general continuum guard auto-enabled.")
            _T_phot = float(getattr(merged, 'T_phot', float('nan')))
            _R_phot = float(getattr(merged, 'R_phot_cm', float('nan')))
            _L_phot = float(getattr(merged, 'L_phot', float('nan')))
            _cg.apply_continuum_guard(spectra, _T_phot, _R_phot, _L_phot,
                                      verbose=verbose)
            _cg.energy_conservation_check(spectra, _L_phot, verbose=verbose)
            _cg.he_decrement_diagnostic(spectra, verbose=verbose)
    except Exception as _cg_exc:
        if verbose:
            print(f"[phase5b/compgen] guard skipped: {_cg_exc}")

    # ---- Continuum renormalization to the EMERGENT continuum (observer
    #      convention). At high τ_es the directly-escaping continuum is well
    #      below the un-attenuated BB, so BB-normalized He profiles sit at F<1
    #      in line-free regions (while the emergent-normalized H/production lines
    #      sit at 1.0) — an inconsistent baseline that reads as a spurious
    #      sub-continuum "deficit". Divide each H/He line by its clean far-wing
    #      median (now reliably continuum thanks to the adaptive window) so every
    #      line's continuum reads 1.0. L_line is untouched; the profile EW is
    #      recomputed relative to the emergent continuum.
    _renormalize_to_emergent_continuum(spectra, verbose=verbose)

    # ---- Phase 5c (P2 #5): C/O/Ne metal lines (opt-in --metal-lines) ----
    # First-principles emissivity integrals (recombination + collisional, with
    # the n_crit correction) on photoionization-equilibrium ion densities. The
    # metal lines are merged into the spectra dict so the npz/regime/plots/movies
    # pick them up dynamically. Atomic data is PROVISIONAL (verify vs CHIANTI).
    if metal_lines:
        try:
            import metal_lines as _ml
            metal_spectra, _metal_ions = _ml.compute_metal_lines(
                merged, snap=snap, use_cloudy=metal_cloudy,
                narrow_csm=narrow_csm, verbose=verbose)
            for _name, _sp in metal_spectra.items():
                spectra[_name] = _sp
            if metal_spectra and out_prefix is not None and make_png:
                try:
                    _ml.save_metal_png(
                        spectra, f"{out_prefix}_metal_lines.png",
                        epoch_d=(snap.get('epoch_d') if isinstance(snap, dict)
                                 else None),
                        T_phot=getattr(merged, 'T_phot', None),
                        R_phot=getattr(merged, 'R_phot_cm', None))
                except Exception as _mp:
                    print(f"[phase5c] metal PNG render failed: {_mp}")
            if verbose and metal_spectra:
                print(f"[phase5c] merged {len(metal_spectra)} metal lines "
                      f"(C/O/Ne) into the output"
                      + (f"; wrote {out_prefix}_metal_lines.png"
                         if (out_prefix and make_png) else "") + ".")
        except Exception as _ml_exc:
            if verbose:
                print(f"[phase5c] metal lines skipped: {_ml_exc}")

    # Save outputs if requested
    if out_prefix is not None:
        _save_phase5_npz(spectra, f"{out_prefix}_lines.npz")
        _save_phase5_txt(spectra, f"{out_prefix}_lines.txt", state, snap,
                          production_halpha=production_halpha,
                          correction_factor=correction_factor)
        if make_png:
            try:
                _save_phase5_png(spectra, f"{out_prefix}_lines.png", state, snap,
                                  production_halpha=production_halpha,
                                  correction_factor=correction_factor)
            except Exception as e:
                # Plotting is non-essential; log and continue
                print(f"[phase5] WARNING: PNG render failed: {e}")
        if verbose:
            print(f"[phase5] Saved {out_prefix}_lines.npz, .txt"
                  + (", .png" if make_png else ""))

    return spectra


# ============================================================================
# Continuum renormalization (emergent-continuum / observer convention)
# ============================================================================

def _renormalize_to_emergent_continuum(spectra, frac=0.82, verbose=True):
    """Rescale each H/He line's F_norm / F_norm_corrected so its line-free
    far-wing continuum reads 1.0 (observer convention).

    At high \u03c4_es the He lines (normalized to the un-attenuated diluted-BB) sit at
    F<1 in continuum regions while the emergent-normalized H/production lines sit
    at 1.0 \u2014 an inconsistent, misleading baseline. We estimate the emergent
    continuum from the median of |\u0394v| > frac\u00b7window (now reliably continuum given
    the adaptive window) and divide it out. The shape is unchanged; L_line is NOT
    touched; the profile-EW (sp['EW']/'EW_corrected') is recomputed relative to
    the corrected continuum so it matches the displayed F/F_cont. Metal lines
    (already line-centre-normalized to 1.0) are skipped. Skips a line whose
    far-wing is too contaminated (continuum estimate not finite/positive, or the
    line fills the window).
    """
    _C = 2.99792458e5
    _trz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    n_fixed = 0
    for ln, sp in spectra.items():
        # skip metals (line-centre normalized, already continuum=1) and anything
        # without a profile.
        if str(sp.get('strength_mode', '')).startswith('metal'):
            continue
        if 'F_norm' not in sp or 'lambda_AA' not in sp:
            continue
        lam = np.asarray(sp['lambda_AA'], float)
        lam0 = float(sp.get('lambda_rest', lam[len(lam) // 2]))
        if lam.size < 5 or not (lam0 > 0):
            continue
        dv = (lam / lam0 - 1.0) * _C
        win = max(abs(float(dv.min())), abs(float(dv.max())))
        if win <= 0:
            continue
        fw = np.abs(dv) > frac * win
        if fw.sum() < 3:
            continue
        Fref = np.asarray(sp.get('F_norm_corrected', sp['F_norm']), float)
        cont = float(np.median(Fref[fw]))
        if not (np.isfinite(cont) and cont > 0):
            continue
        # guard: if the "continuum" is itself a large departure from the line
        # peak (i.e. the window is line-filled and the far wing is not real
        # continuum), skip rather than mis-scale.
        if abs(cont - 1.0) < 1e-3:
            continue
        for key in ('F_norm', 'F_norm_corrected'):
            if key in sp:
                sp[key] = np.asarray(sp[key], float) / cont
        # recompute profile EW relative to the now-unit continuum
        Fc = np.asarray(sp.get('F_norm_corrected', sp['F_norm']), float)
        ew = float(-_trz(Fc - 1.0, lam))
        if 'EW_corrected' in sp:
            sp['EW_corrected'] = ew
        sp['EW'] = ew
        sp['cont_renorm'] = cont           # the emergent/BB continuum ratio applied
        n_fixed += 1
    if verbose and n_fixed:
        print(f"[phase5] continuum renormalized to the emergent continuum for "
              f"{n_fixed} H/He lines (far-wing \u2192 1.0; observer convention).")
    return spectra


# ============================================================================
# Phase 5b-empirical: H\u03b1-anchored correction
# ============================================================================

def _correction_label():
    """Return human-readable name of the current empirical correction mode.

    Reads SNLINE_R_MODE env var; defaults to 'lline'. This avoids stashing
    metadata inside the `spectra` dict (which is iterated over by the
    correction-apply step and would crash on non-dict entries).
    """
    import os as _os
    mode = _os.environ.get('SNLINE_R_MODE', 'lline').lower()
    return 'EW_prod / EW_phase5' if mode == 'ew' else 'L_prod / L_phase5'


def _compute_empirical_correction(spectra: dict, production_halpha: dict):
    """Compute the Hα-anchored empirical correction factor R.

    Default: R = L_line(production Hα) / L_line(Phase 5 Hα).
    This is the ratio of total integrated line luminosity — clean of the
    EW-cancellation pathology that arises when a P-Cygni profile has
    comparable emission and absorption (their EW contributions partially
    cancel, making R_EW artificially small even when the line is bright).
    L_line is conserved and equally meaningful for emission, absorption,
    and mixed profiles.

    The legacy EW-based correction (R = EW_prod / EW_p5) is still
    available by setting the environment variable SNLINE_R_MODE=ew.

    Returns None if either input is missing or out of physical range,
    in which case the correction step is skipped (raw Phase 5 values
    pass through).
    """
    import os as _os
    if not isinstance(production_halpha, dict):
        return None
    if 'Halpha' not in spectra:
        return None

    mode = _os.environ.get('SNLINE_R_MODE', 'lline').lower()

    if mode == 'ew':
        # Legacy EW-based correction
        EW_prod = production_halpha.get('EW', None)
        if EW_prod is None or not np.isfinite(EW_prod) or EW_prod == 0:
            return None
        EW_p5 = spectra['Halpha'].get('EW', None)
        if EW_p5 is None or not np.isfinite(EW_p5) or EW_p5 == 0:
            return None
        R = EW_prod / EW_p5
    else:
        # Default: L_line-based correction (more robust for P-Cygni)
        L_prod = production_halpha.get('L_line', None)
        if L_prod is None or not np.isfinite(L_prod) or L_prod <= 0:
            return None
        L_p5 = spectra['Halpha'].get('L_line', None)
        if L_p5 is None or not np.isfinite(L_p5) or L_p5 <= 0:
            return None
        R = L_prod / L_p5

    # Sanity: physical R is typically in (0.1, 5.0) for well-matched kernels.
    # Wider range (0.01, 50) handles edge cases. Outside that range we treat
    # the correction as unreliable and pass through raw values.
    if not (0.01 < abs(R) < 50.0):
        return None
    return float(R)


def _ion_density(merged, ion_key, n_e):
    """Return the recombining-ion number density array for a budget, or None.

    n_p   : proton density (H lines). For H-line *ratios* this cancels, so any
            consistent H-ion proxy is fine; prefer a real n_HII if present.
    n_HeII : He+  (feeds He I recombination lines)   — needs He-NLTE state.
    n_HeIII: He++ (feeds He II recombination lines)   — needs He-NLTE state.
    """
    import numpy as _np
    if ion_key == 'n_p':
        for a in ('n_HII', 'n_p', 'n_proton', 'nHII'):
            if hasattr(merged, a):
                return _np.asarray(getattr(merged, a), float)
        return _np.asarray(n_e, float)            # cancels in H-line ratios
    if ion_key == 'n_HeII':
        for a in ('n_HeII', 'nHeII'):
            if hasattr(merged, a):
                return _np.asarray(getattr(merged, a), float)
        hl = getattr(merged, 'he2_levels', None)   # He+ level populations
        if hl is not None:
            hl = _np.asarray(hl, float)
            return hl.sum(axis=0) if hl.ndim == 2 else hl
        return None
    if ion_key == 'n_HeIII':
        for a in ('n_HeIII', 'nHeIII'):
            if hasattr(merged, a):
                return _np.asarray(getattr(merged, a), float)
        return None
    return None


def _thomson_shape_thick_he(sp, merged, Fn):
    """Apply multiple electron-scattering redistribution to a thick He line's
    corrected profile (the '+Thomson MC' part of --saturated-rt). Photon-
    conserving: L_line and EW are unchanged, only the shape. Returns the (possibly
    redistributed) Fn; silently returns Fn unchanged on any failure."""
    import numpy as _np
    try:
        import line_rt_escape as _ep
        tau_es = float(getattr(merged, 'tau_es_total', 0.0) or 0.0)
        if tau_es <= 0.0:
            return Fn
        T_phot = float(getattr(merged, 'T_phot',
                               _np.asarray(merged.T, float)[-1]))
        lam_grid = _np.asarray(sp.get('lambda_AA'), float)
        lam0 = sp.get('lambda_rest', None)
        if lam0 is None or lam_grid.size != _np.asarray(Fn).size:
            return Fn
        return _ep.thomson_multiscatter(lam_grid, Fn, float(lam0), T_phot, tau_es)
    except Exception:
        return Fn


def _apply_recombination_budget(spectra, merged, production_halpha, verbose=True,
                                saturated_rt=False):
    """Set per-line strengths from a first-principles recombination budget,
    replacing the empirical Hα-anchored R-factor.

        L(line) = L_prod_Hα * budget(line) / budget(Hα)

    The Hα ABSOLUTE scale is the validated full-RT-NLTE production value; the
    RATIO budget(line)/budget(Hα) is the physical recombination ratio, in which
    the absolute escape/over-counting cancels (both gross integrals scale
    together). For H lines this is the case-B Balmer decrement and is solid. For
    He lines it uses provisional recombination coefficients and the He-NLTE ion
    densities (n_HeII/n_HeIII); if those are unavailable (no He-NLTE in the run)
    the line falls back to the flat Hα anchor with a flag. He I lines also carry
    a collisional channel not in the recomb-only coefficient — flagged 'prov'.
    """
    import numpy as _np
    import formal_line_profile as _flp
    if not isinstance(production_halpha, dict) or 'Halpha' not in spectra:
        _apply_empirical_correction(spectra, None); return
    L_prod = production_halpha.get('L_line', None)
    L_p5_Ha = spectra['Halpha'].get('L_line', None)
    if not (L_prod and _np.isfinite(L_prod) and L_prod > 0
            and L_p5_Ha and L_p5_Ha > 0):
        _apply_empirical_correction(spectra, None); return
    R_flat = L_prod / L_p5_Ha
    r = _np.asarray(merged.r, float); T = _np.asarray(merged.T, float)
    ne = _np.asarray(merged.n_e, float)
    lamHa = spectra['Halpha'].get('lambda_rest', 6562.8)
    budget_Ha, _ = _flp.line_recombination_luminosity(
        'Halpha', lamHa, r, T, ne, _ion_density(merged, 'n_p', ne))
    rows = []
    for name, sp in spectra.items():
        L_p5 = sp.get('L_line', None); lam = sp.get('lambda_rest', None)
        tau_med = sp.get('tau_med', None)
        coeff = _flp.RECOMB_COEFF.get(name)
        is_H = (coeff is not None and coeff[2] == 'n_p')
        is_He = name.startswith('He')
        if name == 'Halpha':
            factor, mode = R_flat, 'reference'
        elif is_H and lam is not None and budget_Ha and budget_Ha > 0 and L_p5 and L_p5 > 0:
            # H lines: recombination budget on the validated Hα scale. The ratio
            # budget(line)/budget(Hα) is the case-B decrement; absolute escape
            # cancels. n_p cancels in the ratio so this is robust.
            b_line, _ = _flp.line_recombination_luminosity(
                name, lam, r, T, ne, _ion_density(merged, 'n_p', ne))
            factor = (L_prod * b_line / budget_Ha) / L_p5
            mode = 'budget(H)'
        elif is_He and L_p5 and L_p5 > 0:
            # He lines: the He-NLTE solvers (he1_nlte/he2_nlte) already compute
            # the ionization AND excitation (incl. the collisional channel) from
            # first principles using the STELLA state — that L_line is the
            # physical value, NOT something to override with a recombination-only
            # budget. Optically-thin He lines are exact (single-shot kernel);
            # optically-thick He lines have the single-shot escape over-estimate,
            # for which the Hα-anchored R is the interim escape correction.
            if tau_med is not None and tau_med >= 1.0:
                # Optically-thick He line. Default: the interim Hα-anchored
                # R_flat escape correction (~factor-2). With --saturated-rt, DROP
                # the empirical anchor: the bare single-shot β luminosity is the
                # first-principles escape-probability luminosity (verified
                # identity, line_rt_escape.escape_probability_luminosity), so
                # factor = 1.0. The SHAPE then gets multiple electron scattering
                # applied below (photon-conserving; L/EW unchanged).
                if saturated_rt:
                    factor, mode = 1.0, 'He-NLTE(thick,EP-esc)'
                else:
                    factor, mode = R_flat, 'He-NLTE(thick,esc-corr)'
            else:
                factor, mode = 1.0, 'He-NLTE(thin,exact)'
        elif L_p5 and L_p5 > 0:
            factor, mode = R_flat, 'anchor(fallback)'
        else:
            factor, mode = 1.0, 'raw'
        if factor is None or not _np.isfinite(factor):
            factor, mode = 1.0, 'raw'
        sp['EW_corrected'] = sp['EW'] * factor
        sp['L_line_corrected'] = sp['L_line'] * factor
        Fn = 1.0 + (_np.asarray(sp['F_norm']) - 1.0) * factor
        # --saturated-rt: multiple electron-scattering redistribution of the
        # thick-line SHAPE (photon-conserving — L_line/EW already set above are
        # unchanged; only the profile is broadened / peak suppressed).
        if saturated_rt and mode == 'He-NLTE(thick,EP-esc)':
            Fn = _thomson_shape_thick_he(sp, merged, Fn)
        sp['F_norm_corrected'] = Fn
        sp['peak_F_corrected'] = float(_np.max(Fn))
        sp['strength_mode'] = mode
        rows.append((name, mode, sp['L_line_corrected']))
    if verbose:
        print("[phase5b] line strengths — H: recombination budget on validated "
              "Hα scale (empirical anchor REMOVED); He: He-NLTE first-principles:")
        for nm, md, L in rows:
            print(f"           {nm:12s} {md:22s} L_line={L:.3e} erg/s")
        if any(m.startswith('He-NLTE') for _, m, _ in rows):
            if any(m == 'He-NLTE(thick,EP-esc)' for _, m, _ in rows):
                print("[phase5b] NOTE: He strengths from the he1/he2 NLTE solvers; "
                      "optically-thick He lines (--saturated-rt, P1 #3) use the bare "
                      "single-shot β escape luminosity (= first-principles "
                      "escape-probability value; empirical Hα anchor REMOVED) with "
                      "multiple-electron-scattering applied to the profile SHAPE.")
            else:
                print("[phase5b] NOTE: He strengths come from the he1/he2 NLTE "
                      "solvers (ionization + excitation incl. collisional, from the "
                      "STELLA state). Optically-thick He lines carry a single-shot "
                      "escape correction (~factor-2); pass --saturated-rt to drop "
                      "the empirical anchor (bare β escape + Thomson MC shape).")


def _apply_empirical_correction(spectra: dict, R):
    """Apply R as a multiplicative scaling to non-Hα lines' EW/L_line.

    The F_norm profile is scaled around 1.0: F_norm_corr − 1 = R × (F_norm − 1)
    so the continuum baseline is preserved and only the line departure is scaled.

    Adds keys 'EW_corrected', 'L_line_corrected', 'F_norm_corrected',
    'peak_F_corrected' to each spectrum dict. If R is None, sets corrected
    values equal to raw values (no-op).
    """
    if R is None:
        for name, sp in spectra.items():
            sp['EW_corrected'] = sp['EW']
            sp['L_line_corrected'] = sp['L_line']
            sp['F_norm_corrected'] = np.asarray(sp['F_norm']).copy()
            sp['peak_F_corrected'] = float(np.max(sp['F_norm']))
        return

    for name, sp in spectra.items():
        if name == 'Halpha':
            # Hα IS the reference. Its corrected EW must equal production
            # by construction. Use R = 1 for Hα (no double-counting).
            # The raw Phase 5 Hα value remains in 'EW'; 'EW_corrected'
            # equals production Hα EW.
            sp['EW_corrected'] = sp['EW'] * R
            sp['L_line_corrected'] = sp['L_line'] * R
            Fn_corr = 1.0 + (np.asarray(sp['F_norm']) - 1.0) * R
            sp['F_norm_corrected'] = Fn_corr
            sp['peak_F_corrected'] = float(np.max(Fn_corr))
        else:
            sp['EW_corrected'] = sp['EW'] * R
            sp['L_line_corrected'] = sp['L_line'] * R
            Fn_corr = 1.0 + (np.asarray(sp['F_norm']) - 1.0) * R
            sp['F_norm_corrected'] = Fn_corr
            sp['peak_F_corrected'] = float(np.max(Fn_corr))


# ============================================================================
# State assembly
# ============================================================================

def _build_merged_state(state, snap) -> types.SimpleNamespace:
    """Merge `state` (NLTE diagnostics) and `snap` (hydro) into one namespace."""
    merged = types.SimpleNamespace()

    # Hydro from snap
    for key in ('r', 'v', 'rho', 'T', 'n_e'):
        if key in snap:
            setattr(merged, key, np.asarray(snap[key], dtype=float))
        elif hasattr(state, key):
            setattr(merged, key, np.asarray(getattr(state, key), dtype=float))
        else:
            raise ValueError(f"Phase 5: cannot find '{key}' in snap or state")

    # Photosphere fields (from snap, state, or fallback)
    for src_key, dest_key in [('R_phot_cm', 'R_phot_cm'),
                                ('R_phot',    'R_phot_cm'),
                                ('T_phot',    'T_phot'),
                                ('T_color',   'T_phot'),
                                ('L_phot',    'L_phot'),
                                ('tau_es_total', 'tau_es_total'),
                                ('v_turb_kms', 'v_turb_kms'),
                                ('v_turb_kms_grid', 'v_turb_kms_grid')]:
        val = None
        if snap is not None and src_key in (snap if isinstance(snap, dict) else {}):
            val = snap[src_key]
        elif hasattr(state, src_key):
            val = getattr(state, src_key)
        if val is not None and not hasattr(merged, dest_key):
            setattr(merged, dest_key, val)

    # NLTE diagnostics from state
    for key in ('he1_levels', 'he1_diag', 'he1_tau',
                'he2_levels', 'he2_diag', 'he2_tau',
                'h_levels', 'h_diag', 'h_tau'):
        if hasattr(state, key):
            setattr(merged, key, getattr(state, key))

    # Composition (P1 #4: composition-general continuum guard / H-free switch)
    for key in ('X_H', 'X_He'):
        if isinstance(snap, dict) and key in snap:
            setattr(merged, key, np.asarray(snap[key], float))
        elif hasattr(state, key):
            setattr(merged, key, np.asarray(getattr(state, key), float))

    # Metal composition dict + shock-X-ray params (P2 #5: metal lines). The
    # composition dict is truncated zone-consistently with the hydro by
    # stella_io.truncate_to_photosphere.
    if isinstance(snap, dict):
        comp = snap.get('composition')
        if isinstance(comp, dict):
            merged.composition = comp
        if 'T_shock' in snap:
            merged.T_shock = snap['T_shock']
        if 'L_X_brems' in snap:
            # Apply the SAME smooth interior-shock attenuation that photoionize_csm
            # uses for the H/He field: a shock buried inside R_phot has its hard
            # X-rays partially reprocessed by the overlying optically-thick gas, so
            # only a fraction f_xray_escape = exp(−Δτ_es) reaches the line-forming
            # CSM. Scaling by this CONTINUOUS factor (rather than the old binary
            # interior/exterior gate) is what stops the metal high-ion lines
            # (C III↔C IV, [O III], [Ne III]) flickering epoch-to-epoch as the
            # shock front wobbles across R_phot at the breakout epochs — the ion
            # balance now varies smoothly with the (smooth) shock depth. Also cap
            # the escaping X-rays at L_phot (energy conservation — the ionizing
            # flux can't exceed L_bol). Falls back to the binary include_shock_xray
            # flag for snapshots produced before f_xray_escape was stored.
            _pip = snap.get('photoionization_params') or {}
            _lx = float(snap['L_X_brems'])
            _fx = _pip.get('f_xray_escape', None)
            if _fx is None:
                _fx = 1.0 if bool(_pip.get('include_shock_xray', True)) else 0.0
            _lx *= float(_fx)
            _lp = float(getattr(merged, 'L_phot', 0.0) or 0.0)
            if _lp <= 0:
                _Rp = float(getattr(merged, 'R_phot_cm',
                                    getattr(merged, 'R_phot', 0.0)) or 0.0)
                _Tp = float(getattr(merged, 'T_phot', 0.0) or 0.0)
                if _Rp > 0 and _Tp > 0:
                    _lp = 4.0 * np.pi * _Rp ** 2 * 5.670374419e-5 * _Tp ** 4
            if _lp > 0 and _lx > _lp:
                _lx = _lp
            merged.L_X_brems = _lx

    return merged


def _validate_state_for_phase5(merged):
    """Sanity check that required Phase 2/3/4 outputs are present."""
    have_he1 = (hasattr(merged, 'he1_levels')
                and hasattr(merged, 'he1_tau'))
    have_he2 = (hasattr(merged, 'he2_levels')
                and hasattr(merged, 'he2_tau'))
    have_h = hasattr(merged, 'h_levels')
    if not (have_he1 or have_he2 or have_h):
        raise RuntimeError(
            "Phase 5 requires at least one of: Phase 2 (H NLTE; "
            "state.h_levels), Phase 3 (--he1-nlte; state.he1_levels/he1_tau), "
            "Phase 4 (--he2-nlte; state.he2_levels/he2_tau). None found.")
    # Warn for partial setup
    if not have_he1:
        print("[phase5] WARNING: state.he1_levels missing — He I lines will "
              "be skipped.")
    if not have_he2:
        print("[phase5] WARNING: state.he2_levels missing — He II lines will "
              "be skipped.")
    if not have_h:
        print("[phase5] WARNING: state.h_levels missing — H lines (Hα, Hβ, "
              "Hγ, Pα, Pβ) will be skipped. (production_runner Phase 5 hook "
              "attaches this from params['populations_diag']['n_levels'].)")


# ============================================================================
# Output writers
# ============================================================================

def _save_phase5_npz(spectra: dict, path: str):
    """Save all per-line arrays to a single .npz."""
    payload = {}
    line_names = list(spectra.keys())
    payload['line_names'] = np.array(line_names)
    for name in line_names:
        sp = spectra[name]
        payload[f"{name}__lambda"]      = sp['lambda_AA']
        payload[f"{name}__F_lambda"]    = sp['F_lambda']
        payload[f"{name}__F_line_only"] = sp.get('F_line_only',
                                                   np.zeros_like(sp['F_lambda']))
        payload[f"{name}__F_cont"]      = sp['F_cont_lambda']
        payload[f"{name}__F_norm"]      = sp['F_norm']
        # Phase 5b empirical correction (set even when R is None: equal to raw)
        if 'F_norm_corrected' in sp:
            payload[f"{name}__F_norm_corrected"] = sp['F_norm_corrected']
    # Scalar summary arrays — include corrected versions when present
    scalar_keys = ['L_line', 'L_cont_band', 'EW', 'lambda_rest', 'tau_med']
    for sk in scalar_keys:
        payload[sk] = np.array([spectra[ln].get(sk, np.nan)
                                  for ln in line_names])
    if any('EW_corrected' in spectra[ln] for ln in line_names):
        payload['EW_corrected'] = np.array(
            [spectra[ln].get('EW_corrected', spectra[ln]['EW'])
             for ln in line_names])
        payload['L_line_corrected'] = np.array(
            [spectra[ln].get('L_line_corrected', spectra[ln]['L_line'])
             for ln in line_names])
        payload['peak_F_corrected'] = np.array(
            [spectra[ln].get('peak_F_corrected', float(np.max(spectra[ln]['F_norm'])))
             for ln in line_names])
    np.savez(path, **payload)


def _save_phase5_txt(spectra: dict, path: str, state, snap,
                       production_halpha: dict = None,
                       correction_factor: float = None):
    """Save a human-readable summary table grouped by species.

    Layout: top section = H I lines (with optional production-Hα cross-check
    in a header line), middle = He I, bottom = He II. When an empirical
    Hα-anchored correction has been applied (correction_factor is not None),
    additional EW_corr and L_corr columns show the corrected values.
    """
    import mc_multi_line as p5
    epoch = snap.get('epoch_d', '?') if isinstance(snap, dict) else '?'
    T_phot = (snap.get('T_phot') if isinstance(snap, dict) else None) or \
             getattr(state, 'T_phot', None) or getattr(state, 'T_color', None)
    C = 2.998e5

    # Group lines by species (metals routed to their own section — P2 #5)
    by_species = {'HI': [], 'HeI': [], 'HeII': [], 'metal': []}
    for name in spectra:
        if name.startswith(('C_', 'O_', 'Ne_')) and not name.startswith('He_'):
            by_species['metal'].append(name)
            continue
        sp = by_species.get(p5._species_of(name))
        if sp is not None:
            sp.append(name)

    prod_Ha = production_halpha if isinstance(production_halpha, dict) else None
    R = correction_factor
    has_corr = (R is not None) or \
        any('L_line_corrected' in spectra[ln] for ln in spectra)

    with open(path, 'w') as f:
        f.write(f"# Phase 5 multi-line summary "
                f"(β-corrected single-shot RT + Hα-anchored empirical correction)\n")
        f.write(f"# Snapshot: {snap.get('source_file', '?') if isinstance(snap, dict) else '?'}\n")
        f.write(f"# Epoch: {epoch} d   T_phot: {T_phot} K\n")
        if prod_Ha:
            f.write(f"#\n")
            f.write(f"# Production Hα (RT-NLTE iterated, gold-standard):\n")
            f.write(f"#   L_line = {prod_Ha['L_line']:.4e} erg/s   "
                    f"peak F = {prod_Ha['peak_F']:.3f} @ Δv = "
                    f"{prod_Ha['peak_dv']:+.1f} km/s   EW = {prod_Ha['EW']:+.2f} Å\n")
        if has_corr:
            f.write(f"#\n")
            if R is not None:
                f.write(f"# Phase 5b empirical Hα-anchored correction:\n")
                _label_txt = _correction_label()
                f.write(f"#   R = {_label_txt}(Hα) = {R:.4f}\n")
                f.write(f"#   Applied to L_line, EW, (F_norm−1) of ALL lines as\n")
                f.write(f"#   uniform scaling. For Hα: when R is L-based, the\n")
                f.write(f"#   corrected Hα L_line matches production exactly; when\n")
                f.write(f"#   R is EW-based, the corrected Hα EW matches production.\n")
                f.write(f"#   Default R = L_prod / L_phase5 avoids EW-cancellation\n")
                f.write(f"#   pathology in P-Cygni profiles with comparable\n")
                f.write(f"#   absorption and emission contributions. Set env var\n")
                f.write(f"#   SNLINE_R_MODE=ew for legacy EW-based correction.\n")
            else:
                f.write(f"# Phase 5b strengths (no empirical anchor):\n")
                f.write(f"#   H lines  = per-line recombination budget on the validated\n")
                f.write(f"#              production-Hα scale (case-B decrement).\n")
                f.write(f"#   He lines = he1/he2 NLTE first-principles; optically-thick\n")
                f.write(f"#              He lines carry a single-shot escape correction.\n")
                f.write(f"#   Quote the L_corr / EW_corr columns (NOT L_raw/EW_raw).\n")
                f.write(f"#   CAVEAT: for saturated lines (τ≫1, common in the IIn\n")
                f.write(f"#   regime) the single-shot PROFILE SHAPE is unreliable, so\n")
                f.write(f"#   peak_F_corr and profile-integrated EW are factor-of-few;\n")
                f.write(f"#   use the production Hα for the Hα shape.\n")
        f.write(f"#\n")
        if has_corr:
            f.write(f"# species  line          λ_rest[Å]   τ_med     "
                    f"L_raw[erg/s]      L_corr[erg/s]     "
                    f"EW_raw[Å]   EW_corr[Å]   peak F_corr   peak Δv[km/s]\n")
        else:
            f.write(f"# species  line          λ_rest[Å]   τ_med     "
                    f"L_line[erg/s]    L_cont_band[erg/s]   "
                    f"EW[Å]     peak F   peak Δv[km/s]\n")
        f.write(f"# {'-'*135 if has_corr else '-'*118}\n")

        for species_label, header_comment in [
                ('HI',   '# --- H I (Balmer + Paschen) ---'),
                ('HeI',  '# --- He I ---'),
                ('HeII', '# --- He II ---'),
                ('metal', '# --- Metal lines C/O/Ne (PROVISIONAL atomic data; '
                          'quote L_line, verify vs CHIANTI) ---')]:
            if not by_species[species_label]:
                continue
            f.write(f"{header_comment}\n")
            for name in by_species[species_label]:
                sp = spectra[name]
                lam0 = sp['lambda_rest']
                Fn = sp.get('F_norm_corrected', sp['F_norm'])
                lam = sp['lambda_AA']
                idx_peak = int(np.argmax(Fn))
                peak_F = float(Fn[idx_peak])
                peak_dv = (lam[idx_peak] - lam0) / lam0 * C
                if has_corr:
                    f.write(f"  {species_label:<7s} {name:<13s} {lam0:>10.2f}  "
                            f"{sp['tau_med']:>9.2e}  "
                            f"{sp['L_line']:>15.4e}  "
                            f"{sp['L_line_corrected']:>15.4e}  "
                            f"{sp['EW']:>+9.2f}  {sp['EW_corrected']:>+10.2f}  "
                            f"{peak_F:>11.3f}  {peak_dv:>+10.1f}\n")
                else:
                    f.write(f"  {species_label:<7s} {name:<13s} {lam0:>10.2f}  "
                            f"{sp['tau_med']:>9.2e}  "
                            f"{sp['L_line']:>15.4e}  {sp['L_cont_band']:>17.4e}  "
                            f"{sp['EW']:>+8.2f}  {peak_F:>8.3f}  {peak_dv:>+10.1f}\n")


def _save_phase5_png(spectra: dict, path: str, state, snap,
                       production_halpha: dict = None,
                       correction_factor: float = None):
    """Render a 3-row × 5-col figure with corrected profiles + raw overlay.

    When correction_factor R is supplied, each panel plots the R-corrected
    F_norm in solid line, with the raw (uncorrected) F_norm shown as a
    light dashed line for transparency. Panel titles show the corrected EW.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mc_multi_line as p5

    epoch = snap.get('epoch_d', '?') if isinstance(snap, dict) else '?'
    T_phot = (snap.get('T_phot') if isinstance(snap, dict) else None) or \
             getattr(state, 'T_phot', None) or 0.0

    # Display order — keep slots empty if a line wasn't computed
    h_order   = ['Halpha', 'Hbeta', 'Hgamma', 'Palpha', 'Pbeta']
    he1_order = ['He_I_5876',  'He_I_6678',  'He_I_7065',  'He_I_10830', None]
    he2_order = ['He_II_1640', 'He_II_3203', 'He_II_4686', 'He_II_10124', None]

    fig, axes = plt.subplots(3, 5, figsize=(20, 10.5))
    plt.subplots_adjust(hspace=0.42, wspace=0.30, top=0.90)
    has_corr = (correction_factor is not None) or \
        any('L_line_corrected' in spectra[ln] for ln in spectra)

    C = 2.998e5
    for row, line_set, color, row_label in [
            (0, h_order,   '#2ca02c', 'H I'),
            (1, he1_order, '#1f77b4', 'He I'),
            (2, he2_order, '#d62728', 'He II')]:
        for col, ln in enumerate(line_set):
            ax = axes[row, col]
            if ln is None:
                ax.set_visible(False)
                continue
            if ln not in spectra:
                ax.text(0.5, 0.5, f"{ln}\n(no data)", ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, alpha=0.4)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            sp = spectra[ln]
            lam0 = sp['lambda_rest']
            dv = (sp['lambda_AA'] - lam0) / lam0 * C
            # Plot corrected (solid) and raw (light dashed) when available
            Fn_corr = sp.get('F_norm_corrected', sp['F_norm'])
            if has_corr:
                _corr_lbl = (f'×R={correction_factor:.2f}'
                             if correction_factor is not None else 'corrected')
                ax.plot(dv, sp['F_norm'], '--', color=color, lw=0.7, alpha=0.4,
                        label='raw (single-shot)')
                ax.plot(dv, Fn_corr, '-', color=color, lw=1.5,
                        label=_corr_lbl)
            else:
                ax.plot(dv, sp['F_norm'], '-', color=color, lw=1.5)
            ax.axhline(1.0, color='k', ls=':', lw=0.5, alpha=0.5)
            ax.axvline(0.0, color='k', ls=':', lw=0.5, alpha=0.5)
            display_name = ln.replace('_', ' ')
            EW_show = sp.get('EW_corrected', sp['EW'])
            L_show = sp.get('L_line_corrected', sp['L_line'])
            ax.set_title(f"{display_name}   τ={sp['tau_med']:.1e}\n"
                          f"L={L_show:.2e} erg/s   EW={EW_show:+.2f} Å",
                          fontsize=9)
            ax.set_xlabel('Δv [km/s]', fontsize=9)
            ax.set_ylabel('F / F$_{cont}$', fontsize=9)
            # x-range follows the DATA: auto-detect the line extent (where the
            # profile departs from the unit continuum) + 25% margin, falling back
            # to the full window. The fixed ±5000 used to CLIP the now-adaptive
            # wide windows (lines reaching ±10000+).
            try:
                _exc = np.abs(np.asarray(Fn_corr, float) - 1.0)
                _pk = float(np.nanmax(_exc)) if _exc.size else 0.0
                _m = _exc > max(0.05 * _pk, 0.02)
                if _pk > 0.05 and _m.sum() >= 5:
                    _lo = float(dv[_m][0]); _hi = float(dv[_m][-1])
                    _w = max(_hi - _lo, 1.0)
                    _lo -= 0.25 * _w; _hi += 0.25 * _w
                    ax.set_xlim(max(_lo, float(dv.min())),
                                min(_hi, float(dv.max())))
                else:
                    ax.set_xlim(float(dv.min()), float(dv.max()))
            except Exception:
                ax.set_xlim(float(dv.min()), float(dv.max()))
            ax.grid(alpha=0.3)
            if ln == 'Halpha' and has_corr:
                ax.legend(loc='upper right', fontsize=7, framealpha=0.7)

    # Suptitle: header summary + production-Hα + correction factor
    R_phot = (snap.get('R_phot_cm') if isinstance(snap, dict) else None) or \
             getattr(state, 'R_phot_cm', 0.0) or getattr(state, 'R_phot', 0.0)
    line1 = (f"Phase 5 + 5b-empirical multi-line H + He profiles   "
             f"epoch = {epoch} d, T_phot = {T_phot:.0f} K, "
             f"R_phot = {R_phot:.2e} cm")
    sub_lines = []
    if production_halpha:
        sub_lines.append(
            f"Production Hα: L={production_halpha.get('L_line', float('nan')):.2e} erg/s, "
            f"peak F = {production_halpha.get('peak_F', float('nan')):.2f}, "
            f"EW = {production_halpha.get('EW', float('nan')):+.1f} Å")
    if has_corr:
        if correction_factor is not None:
            _label_fig = _correction_label()
            sub_lines.append(
                f"Empirical correction: R = {_label_fig}(Hα) = {correction_factor:.4f}  "
                f"(solid = corrected, dashed = raw)")
        else:
            sub_lines.append(
                "Strengths: H = recombination budget (no anchor), He = NLTE  "
                "(solid = corrected, dashed = raw single-shot)")
    sup = line1 + ('\n' + '   |   '.join(sub_lines) if sub_lines else '')
    fig.suptitle(sup, fontsize=10.5, fontweight='bold', y=0.995)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    print("Phase 5 runner self-test")
    print("="*78)

    # Build a synthetic IIn state by hand (no real snapshot loaded)
    import types
    n_z = 50
    r = np.linspace(2e14, 1.7e15, n_z)
    v = np.linspace(300e5, 5000e5, n_z)
    rho = 1e-13 * (2e14 / r) ** 2
    T = np.full(n_z, 1e4)
    n_e = 1e10 * (2e14 / r) ** 2

    snap = {'r': r, 'v': v, 'rho': rho, 'T': T, 'n_e': n_e,
            'R_phot_cm': 1.7e15, 'T_phot': 14000.0, 'L_phot': 4e42,
            'epoch_d': 0.5}

    state = types.SimpleNamespace()
    r_scale = (r[0] / r) ** 2
    he1_lev = np.zeros((11, n_z))
    he1_lev[0]  = 1.94e+01 * r_scale
    he1_lev[1]  = 4.0e+07 * r_scale
    he1_lev[3]  = 1.0e+05 * r_scale
    he1_lev[4]  = 5.0e+03 * r_scale
    he1_lev[5]  = 1.0e+04 * r_scale
    he1_lev[9]  = 5.0e+04 * r_scale
    he1_lev[10] = 1.0e+03 * r_scale
    state.he1_levels = he1_lev
    state.he1_tau = {
        'He_I_10830': 3.1e+06 * np.ones(n_z),
        'He_I_5876':  1.8e+06 * np.ones(n_z),
        'He_I_7065':  2.5e+05 * np.ones(n_z),
        'He_I_6678':  1.4e-03 * np.ones(n_z),
    }
    he2_lev = np.zeros((10, n_z))
    he2_lev[1] = 1.0e+05 * r_scale
    he2_lev[2] = 5.0e+04 * r_scale
    he2_lev[3] = 1.0e+04 * r_scale
    he2_lev[4] = 1.0e+03 * r_scale
    state.he2_levels = he2_lev
    state.he2_tau = {
        'He_II_1640':  5.2e+04 * np.ones(n_z),
        'He_II_4686':  8.8e+01 * np.ones(n_z),
        'He_II_3203':  9.5e+00 * np.ones(n_z),
        'He_II_10124': 1.1e+00 * np.ones(n_z),
    }

    print("Running with reference Sobolev MC (use_existing_peel_kernel=False)")
    spectra = run_phase5_for_state(
        state, snap,
        n_packets=20000,
        use_existing_peel_kernel=False,
        out_prefix='/tmp/phase5_test',
        verbose=True)
    print(f"\nWrote /tmp/phase5_test_he.npz, .txt, .png ({len(spectra)} lines)")
