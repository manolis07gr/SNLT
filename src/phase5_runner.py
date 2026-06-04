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
                                    verbose=verbose)
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


def _apply_recombination_budget(spectra, merged, production_halpha, verbose=True):
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
            print("[phase5b] NOTE: He strengths come from the he1/he2 NLTE solvers "
                  "(ionization + excitation incl. collisional, from the STELLA state). "
                  "Optically-thick He lines carry a single-shot escape correction "
                  "(~factor-2); confirm against data when available.")


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

    # Group lines by species
    by_species = {'HI': [], 'HeI': [], 'HeII': []}
    for name in spectra:
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
                ('HeII', '# --- He II ---')]:
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
            ax.set_xlim(-5000, 5000)
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
