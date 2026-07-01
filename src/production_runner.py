#!/usr/bin/env python3
"""
production_runner.py — Production Hα pipeline with full RT-NLTE iteration.

Implements the self-consistent coupling between level populations and the
mean intensity J_bar(Hα):

    populations_0 = NLTE_solver(snap, J_bar=None)        # baseline iteration
    repeat:
        MC_k    = run_MC(snap, populations_k)
        J_bar_k = measure_J_bar(MC_k, populations_k)
        populations_{k+1} = NLTE_solver(snap, J_bar=J_bar_k)
    until convergence on populations

After convergence, runs one final high-statistics MC for the production
spectrum.

Modes
-----
Single snapshot:
    python production_runner.py atmosphere_8_new.dat
    → produces test_b_atm8_RT.png, test_b_atm8_RT.npz, test_b_atm8_RT.txt

Batch (all atmosphere_*.dat in current dir):
    python production_runner.py --batch
    → per-snapshot files + batch_grid.png + batch_movie.mp4
    → batch_summary.txt with peak/trough metrics per epoch

CLI args
--------
    --n-per N      packets per chunk (final production run). Default 100_000.
    --n-chunks N   chunks for final production run. Default 2.
    --iter-n N     packets per iteration step (faster). Default 50_000.
    --max-iter N   maximum RT iterations. Default 6.
    --tol F        relative population convergence tolerance. Default 0.03 (3%).
    --no-iter      skip RT iteration; use baseline populations (legacy mode).
    --ref PATH     CMFGEN reference file (single mode only).
"""
import os
import re
import sys
import time
import hashlib
import argparse
import numpy as np
from glob import glob

from snapshot_analyzer import analyze_snapshot
from snline_autoparams import derive_parameters_from_state
try:
    from lines import LINE_LIB
    import formal_line_profile as _flp
    _HAVE_FORMAL = True
except Exception:
    _HAVE_FORMAL = False
from peel_pipeline_abs import run_peel_pipeline_abs
from measure_jbar import compute_jbar_from_mc
from stella_io import (load_stella_snapshot, detect_format,
                         truncate_to_photosphere)
from wind_extension import extend_wind_outward
from photoionize_csm import solve_photoionization_equilibrium

# Phase 5: multi-line MC for the 8 He lines. Imported lazily inside
# process_snapshot when --he-lines is enabled, so this top-level import
# is intentionally absent (avoids forcing phase5_* deps on every run).


# ---------- Smoothing ----------

def smooth_velocity(F, dv, sigma_kms):
    """Convolve F with a Gaussian kernel of width sigma_kms in velocity space.
    
    Uses reflection boundary conditions to avoid edge artifacts. sigma_kms=0
    returns the array unchanged.
    """
    if sigma_kms <= 0:
        return F.copy()
    # Median bin width in km/s (dv array is linear in lambda, ~constant in v
    # over the small band around Hα)
    dv_bin = float(np.median(np.diff(dv)))
    sigma_bins = abs(sigma_kms / dv_bin)
    if sigma_bins < 0.1:
        return F.copy()
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(F, sigma=sigma_bins, mode='reflect')
    except ImportError:
        # Pure-numpy fallback: build a Gaussian kernel and convolve
        radius = max(int(np.ceil(4 * sigma_bins)), 1)
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= kernel.sum()
        # Reflect-pad input
        Fp = np.concatenate([F[radius:0:-1], F, F[-2:-radius-2:-1]])
        out = np.convolve(Fp, kernel, mode='valid')
        # Length match
        if len(out) > len(F):
            trim = (len(out) - len(F)) // 2
            out = out[trim:trim + len(F)]
        elif len(out) < len(F):
            pad = (len(F) - len(out)) // 2
            out = np.pad(out, (pad, len(F) - len(out) - pad), mode='edge')
        return out


# ---------- I/O ----------

def load_heracles_snapshot(path):
    """Load a HERACLES atmosphere_*_new.dat file."""
    with open(path) as f:
        lines = [f.readline() for _ in range(6)]
    epoch_d = None
    for ln in lines:
        if 'Time since' in ln or 'explosion' in ln.lower():
            try:
                parts = ln.strip().split()[0]
                if '/' in parts:
                    epoch_d = float(parts.split('/')[1].rstrip('d'))
            except Exception:
                pass
    raw = np.loadtxt(path, skiprows=6)
    return {
        'r':   raw[:, 0] * 1e10,
        'v':   raw[:, 1] * 1e5,
        'rho': raw[:, 2],
        'n_e': raw[:, 3],
        'T':   raw[:, 4] * 1e4,
        'L':   raw[:, 5] * 3.828e33,
        'tau_es': raw[:, 6],
        'epoch_d': epoch_d,
        'path': path,
        'format': 'heracles',
    }


def load_snapshot(path, fmt='auto', verbose=False, truncate_stella=True,
                   tau_es_phot=2.0/3.0,
                   extend_wind=False,
                   wind_r_max_factor=20.0,
                   wind_n_zones=100,
                   wind_T_photoionized=10000.0,
                   wind_rho_index=2.0,
                   wind_density_boost=1.0,
                   photoionize=True,
                   photoionize_T_source=None,
                   photoionize_T_eq_floor=1.0e4,
                   include_shock_xray=True,
                   photosphere_mode='es',
                   photosphere_lam_ref_AA=6562.8):
    """Format-aware snapshot loader. Auto-detects HERACLES vs STELLA from
    filename or file header.

    HERACLES: 'atmosphere_*_new.dat' — epoch is "days since explosion".
              The file already contains only zones at/above the photosphere
              (its inner boundary IS R_phot by convention), so no truncation
              is applied.

    STELLA:   'mesa.day*.data' or 'mesa_day*.data' — epoch is "days post Lbol max".
              The file contains the FULL hydro structure including deep
              ejecta inside R_phot. We truncate to zones r > R_phot
              (where τ_es from outside = 2/3) so the MC sees the same
              geometry it sees for HERACLES: opaque BB inner boundary +
              optically-thin CS envelope above.

    Wind extension (STELLA-only): if extend_wind=True, add zones beyond
    the STELLA outer boundary. See wind_extension.extend_wind_outward.

    Photoionization (STELLA-only): if photoionize=True (default), solve
    per-zone photoionization-recombination equilibrium for the truncated
    snapshot. This overrides STELLA's local LTE/diffusion ionization
    (which suffers from missing LyC propagation in grey diffusion) with
    proper CLOUDY/CMFGEN-style steady-state balance. The source spectrum
    defaults to B_ν(T_color) where T_color is taken at the snapshot's
    thermalization depth (τ_es=10), with L_phot from the snapshot. Set
    photoionize_T_source to override; set T_eq_floor=None to leave
    gas temperature unchanged.

    Both formats return a dict with the same core keys
    ('r', 'v', 'rho', 'n_e', 'T', 'L', 'tau_es', 'epoch_d', 'path', 'format');
    STELLA additionally returns 'composition', 'T_rad', 'X_H_emit', etc.
    After photoionization the snap also has 'X_HII', 'n_HI', 'n_HII',
    and diagnostics under '*_stella' keys.
    """
    if fmt == 'auto':
        fmt = detect_format(path)
    if fmt == 'heracles':
        snap = load_heracles_snapshot(path)
        if extend_wind and verbose:
            print("[load_snapshot] --extend-wind ignored (HERACLES format)")
    elif fmt == 'stella':
        snap = load_stella_snapshot(path, verbose=verbose)
        # Derive shock parameters from the FULL pre-truncation snapshot.
        # The shock-front transition (max |dv/dr|) sits across the CDS-CSM
        # interface, which spans zones both inside and outside the τ_es=2/3
        # photosphere. We need the full hydro to find it.
        shock_params = None
        if include_shock_xray:
            from photoionize_csm import derive_shock_params
            shock_params = derive_shock_params(snap, verbose=verbose,
                                               tau_phot_ref=tau_es_phot)
            if verbose and shock_params is not None:
                print(f"[load_snapshot] shock-bremsstrahlung component included: "
                      f"L_X_brems = {shock_params['L_X_brems']:.3e} erg/s "
                      f"(η_rad = {shock_params['eta_rad']:.2f})")
        r_phot_override = None
        if extend_wind:
            snap = extend_wind_outward(
                snap, r_max_factor=wind_r_max_factor,
                n_zones_ext=wind_n_zones,
                T_photoionized=wind_T_photoionized,
                rho_index=wind_rho_index,
                density_boost=wind_density_boost,
                verbose=verbose)
            r_phot_override = snap.get('R_phot_pre_extension', None)
            if verbose and r_phot_override is not None:
                print(f"[load_snapshot] using pre-extension R_phot = "
                      f"{r_phot_override:.3e} cm for truncation "
                      f"(prevents CDS hiding behind extended column)")
        if truncate_stella:
            if photosphere_mode == 'cont':
                from photosphere_v2 import truncate_at_cont_photosphere
                snap = truncate_at_cont_photosphere(
                    snap,
                    lam_ref_AA=photosphere_lam_ref_AA,
                    populations='saha',
                    r_phot=r_phot_override,
                    tau_target=tau_es_phot,
                    verbose=verbose)
            else:
                snap = truncate_to_photosphere(snap, r_phot=r_phot_override,
                                                tau_es_threshold=tau_es_phot,
                                                verbose=verbose)
            if photoionize:
                if verbose:
                    print("[load_snapshot] applying per-zone photoionization "
                          "equilibrium (CLOUDY/CMFGEN-style)...")
                snap = solve_photoionization_equilibrium(
                    snap,
                    T_source=photoionize_T_source,
                    T_eq_floor=photoionize_T_eq_floor,
                    shock_params=shock_params,
                    include_shock_xray=include_shock_xray,
                    verbose=verbose)
    else:
        raise ValueError(f"Cannot determine format for '{path}'. "
                         f"Use --format heracles or --format stella to override.")
    if verbose:
        print(f"[load_snapshot] format={snap['format']}, "
              f"epoch={snap.get('epoch_d')}d, n_zones={len(snap['r'])}")
    return snap


def load_cmfgen_ref(path, lam_grid, lam0=6562.81):
    """Load CMFGEN flux file and return F/F_cont resampled to lam_grid.
    
    The .fl files are RAW flux (column 1) vs wavelength in Angstroms
    (column 0), not pre-normalized. We compute the continuum baseline
    from far-wing medians (avoiding ±300 km/s around line center) and
    return F_raw / F_continuum.
    """
    raw = np.loadtxt(path)
    lam_ref, F_ref_raw = raw[:, 0], raw[:, 1]
    # Far-wing continuum: ±2000 to ±4000 km/s around lam0
    c_kms = 2.998e5
    dv_ref = (lam_ref / lam0 - 1.0) * c_kms
    mask_far = ((np.abs(dv_ref) > 2000) & (np.abs(dv_ref) < 4000))
    if mask_far.sum() < 5:
        # Fall back to a wavelength-based window if dv mask is too narrow
        mask_far = (((lam_ref > 6300) & (lam_ref < 6450)) |
                    ((lam_ref > 6700) & (lam_ref < 6850)))
    F_cont_ref = float(np.median(F_ref_raw[mask_far]))
    F_norm_ref = F_ref_raw / max(F_cont_ref, 1e-30)
    return np.interp(lam_grid, lam_ref, F_norm_ref)


def snap_name_from_path(path):
    """Compact label for a snapshot file.

    atmosphere_8_new.dat       → atm8
    mesa.day080_post_Lbol_max.data → day080
    mesa_day000.5_post_Lbol_max.data → day000.5
    """
    base = os.path.basename(path)
    # STELLA: extract dayXXX[.X]
    m = re.search(r'day(-?\d+(?:\.\d+)?)_post', base)
    if m:
        return f"day{m.group(1)}"
    # HERACLES
    base = base.replace('.dat', '').replace('atmosphere_', 'atm').replace('_new', '')
    return base


# ---------- RT iteration ----------

def run_mc_chunked(snap, params, n_per, n_chunks, seed_offset=0,
                    cont_line_emission=True, nbins=600,
                    band_AA=(6200.0, 6950.0),
                    source_padding_AA=1500.0,
                    calibration='auto',
                    line_redistribution='aa_prd',
                    verbose=False):
    """Run MC kernels n_chunks times; return averaged spectra + accumulated stats.
    
    Critically, accumulates the CALIBRATED F_norm from peel_pipeline_abs
    (which applies the theoretical_ew calibration factor to F_vol). Do NOT
    recompute F_norm = F_total_abs / F_baseline from outside — that uses the
    uncalibrated F_vol and gives F values ~6× too high.

    Both HERACLES and STELLA snaps use the same calibration mode
    (theoretical_ew) after STELLA truncation makes the geometries
    equivalent: opaque BB inner boundary + optically thin CS envelope. The
    snap reaching this function has zones r > R_phot only.

    band_AA / source_padding_AA: see process_snapshot docstring.
    calibration: 'auto' | 'theoretical_ew' | 'f_cont_bb' | 'f_cont_bb_lambda'
        | 'absolute'. 'auto' resolves to theoretical_ew (low L_line/L_cont)
        or f_cont_bb_lambda (high), based on params.
    """
    # Resolve 'auto' to a concrete mode based on L_line/L_cont_band ratio.
    if calibration == 'auto':
        L_line_val = float(params.get('L_line', 0.0))
        L_cont_band_val = float(params.get('L_cont_band',
                                            params.get('L_cont_src', 1.0)))
        ratio = L_line_val / max(L_cont_band_val, 1e-30)
        calibration = ('f_cont_bb_lambda' if ratio >= 0.3
                        else 'theoretical_ew')
    max_events_use = 10000

    F_norm_sum = None
    raw_cont_sum = None
    raw_vol_sum = None
    ns_cont_sum = None
    ns_vol_sum = None
    F_baseline_sum = 0.0
    L_per_pkt_cont = None
    L_per_pkt_vol = None
    calib_factor = None
    for ic in range(n_chunks):
        seed = 1000 * (ic + 1) + 31415 + seed_offset
        lam_c, F_norm_c, F_cont_abs_c, F_total_abs_c, info_c = run_peel_pipeline_abs(
            snap, params,
            n_packets=n_per, nbins=nbins,
            band_AA=band_AA,
            source_padding_AA=source_padding_AA,
            seed=seed,
            max_events=max_events_use, max_events_vol=max_events_use,
            cont_line_emission=cont_line_emission,
            calibration=calibration,
            line_scatter_redistribution=line_redistribution,
            verbose=verbose,
        )
        F_vol_abs_c = F_total_abs_c - F_cont_abs_c
        # Defensive: when L_line is ~0 (e.g. H-free STELLA snapshots), the
        # volumetric peel-off is skipped or returns minimal stats. Use zeros
        # for n_scat_per_zone in that case. The cont channel still runs
        # normally and gives a pure continuum spectrum.
        n_zones = len(snap['r'])
        ns_c = info_c.get('stats_continuum', {}).get(
            'n_scat_per_zone', np.zeros(n_zones))
        ns_v = info_c.get('stats_volumetric', {}).get(
            'n_scat_per_zone', np.zeros(n_zones))
        # Sanitize NaNs in F_vol (can occur when L_line is NaN from
        # a degenerate NLTE solve with X_H = 0)
        F_vol_abs_c = np.nan_to_num(F_vol_abs_c, nan=0.0, posinf=0.0, neginf=0.0)
        F_norm_c = np.nan_to_num(F_norm_c, nan=1.0, posinf=1.0, neginf=0.0)
        L_per_pkt_cont = info_c['L_per_pkt_cont']
        L_per_pkt_vol = info_c.get('L_per_pkt_vol', 0.0)
        if not np.isfinite(L_per_pkt_vol):
            L_per_pkt_vol = 0.0
        calib_factor = info_c.get('calib_factor', 1.0)
        F_baseline_sum += info_c['F_cont_baseline']
        if ic == 0:
            F_norm_sum = F_norm_c.copy()
            raw_cont_sum = F_cont_abs_c.copy()
            raw_vol_sum = F_vol_abs_c.copy()
            ns_cont_sum = np.array(ns_c, dtype=float)
            ns_vol_sum = np.array(ns_v, dtype=float)
            lam = lam_c
        else:
            F_norm_sum += F_norm_c
            raw_cont_sum += F_cont_abs_c
            raw_vol_sum += F_vol_abs_c
            ns_cont_sum += np.array(ns_c, dtype=float)
            ns_vol_sum += np.array(ns_v, dtype=float)
    F_norm = F_norm_sum / n_chunks                       # ← calibrated
    F_cont_abs = raw_cont_sum / n_chunks
    F_vol_abs = raw_vol_sum / n_chunks
    F_baseline = F_baseline_sum / n_chunks
    return {
        'lam': lam,
        'F_norm': F_norm,                                # CALIBRATED
        'F_cont_abs': F_cont_abs,
        'F_vol_abs': F_vol_abs,
        'F_total_abs': F_cont_abs + F_vol_abs,           # uncalibrated raw
        'F_baseline': F_baseline,
        'calib_factor': calib_factor,
        'stats_continuum': {'n_scat_per_zone': ns_cont_sum},
        'stats_volumetric': {'n_scat_per_zone': ns_vol_sum},
        'L_per_pkt_cont': L_per_pkt_cont,
        'L_per_pkt_vol': L_per_pkt_vol,
        'n_chunks': n_chunks,
    }


def run_rt_iteration(snap, n_iter_per=50_000, n_iter_chunks=1,
                      max_iter=6, tol=0.03, damping=0.3,
                      stromgren_p=0.5, f_HI_max=0.1,
                      ionization_mode=None,
                      nbins=600,
                      band_AA=(6200.0, 6950.0),
                      source_padding_AA=1500.0,
                      calibration='auto',
                      eps_Lya_destruction=None,
                      two_photon_decay=False,
                      line_redistribution='aa_prd',
                      verbose=True):
    """Iterate populations + J_bar to self-consistency.
    
    Uses under-relaxation damping to suppress oscillations in n_3
    between iterations. damping=0.3 means each iteration's populations
    are 30% new + 70% previous, stabilizing oscillatory behavior.

    ionization_mode auto-selects per snapshot format if None:
      - HERACLES: 'photoion_decoupled' with f_HI_max cap (HERACLES doesn't
        give composition, so we have to bound neutral fraction).
      - STELLA:   'saha' (we have per-zone composition and accurate n_e,
        so Saha equilibrium with the snapshot's T, n_e is correct).

    Returns
    -------
    dict with final populations, J_bar history, convergence diagnostics.
    """
    state = analyze_snapshot(snap)

    # CRITICAL FIX (Phase 5b debug, day-80 bug): analyze_snapshot can
    # over-write state.T_phot with the photoionized-wind temperature
    # (wind_T_photoionized=10000 K) when the snap has been extended, rather
    # than using the *actual* STELLA photospheric T_phot_inner. At late
    # epochs where T_phot recombines to ~5880 K, this causes a 5× too-hot
    # BB continuum to be used in the kernel, suppressing F_norm by ~10×.
    # Override state.{T_phot,R_phot,L_phot} from snap explicitly here so
    # downstream code sees the values that truncate_at_cont_photosphere
    # set (which match what STELLA actually reports).
    for _attr, _snap_key, _fallback_key in (
            ('T_phot', 'T_phot_inner', 'T_phot'),
            ('R_phot', 'R_phot_inner', 'R_phot'),
            ('L_phot', 'L_phot_inner', 'L_phot')):
        _v = snap.get(_snap_key, snap.get(_fallback_key, None))
        if _v is not None and np.isfinite(_v) and _v > 0:
            _old = getattr(state, _attr, None)
            setattr(state, _attr, float(_v))
            if verbose and _old is not None and abs(float(_v) - float(_old)) / max(abs(float(_old)), 1.0) > 0.01:
                print(f"  [state-fix] {_attr}: {_old:.3e} → {_v:.3e} "
                      f"(from snap['{_snap_key}'])")

    # CRITICAL for STELLA: force the analyzer-derived X_H to be the snapshot's
    # per-zone hydrogen mass fraction, so derive_parameters uses the actual
    # composition (not a default). The composition varies from X_H=0.076
    # (inner He/O-rich ejecta) to X_H=0.64 (outer CSM); ignoring this gives
    # huge errors in n_H_total and L_line for the inner zones.
    if snap.get('format') == 'stella' and 'X_H' in snap:
        state.X_H = np.asarray(snap['X_H'], dtype=float)
        if verbose:
            xh = state.X_H
            print(f"  [STELLA] state.X_H overridden with per-zone composition: "
                  f"range {xh.min():.3f} → {xh.max():.3f}, "
                  f"⟨X_H⟩(emit)={snap.get('X_H_emit', 0):.3f}")

    # SAME for X_He: STELLA snapshots store helium mass fraction in the
    # 'he4' column. When stella_io.py exposes this as snap['X_He'] (per-zone
    # array), propagate it to state.X_He so the He I and He II NLTE solvers
    # use the actual composition rather than the X_He=0.245 default. This
    # is critical for He-enriched ejecta (CCSN with stripped progenitors)
    # and for inner He/O-rich layers where X_He → 0.8-0.9. The He
    # integrators already read state.X_He via _get_He_mass_fraction; we
    # just need to set it here. Falls through silently when stella_io.py
    # hasn't been updated yet (state.X_He stays as whatever analyze_snapshot
    # set, typically 0.245 default).
    if snap.get('format') == 'stella' and 'X_He' in snap:
        state.X_He = np.asarray(snap['X_He'], dtype=float)
        if verbose:
            xhe = state.X_He
            xhe_emit_str = (f", ⟨X_He⟩(emit)={snap['X_He_emit']:.3f}"
                            if 'X_He_emit' in snap else '')
            print(f"  [STELLA] state.X_He overridden with per-zone composition: "
                  f"range {xhe.min():.3f} → {xhe.max():.3f}{xhe_emit_str}")
    elif snap.get('format') == 'stella' and verbose:
        print(f"  [STELLA] WARNING: snap['X_He'] not present — stella_io.py "
              f"may not yet parse the 'he4' column. Falling back to "
              f"default X_He=0.245 for He NLTE.")

    # Auto-select ionization mode if not specified
    if ionization_mode is None:
        if snap.get('format') == 'stella':
            ionization_mode = 'saha'
            effective_f_HI_max = 1.0   # no cap — Saha + true n_e is trustworthy
        else:
            ionization_mode = 'photoion_decoupled'
            effective_f_HI_max = f_HI_max
    else:
        effective_f_HI_max = f_HI_max

    if verbose:
        print(f"  ionization_mode = {ionization_mode}, "
              f"f_HI_max = {effective_f_HI_max}")

    # Iteration 0: no pumping (current production behavior)
    if verbose:
        print(f"[RT-iter] iter 0: baseline NLTE (no J_bar pumping)")
    params = derive_parameters_from_state(
        state, snap, line_name='Halpha',
        wavelength_band_AA=band_AA,
        populations_mode='nlte', nlte_levels=5,
        emission_model='caseb_hr_attenuated_split',
        f_wing=0.02,
        ionization_mode=ionization_mode,
        f_HI_max=effective_f_HI_max,
        stromgren_exponent=stromgren_p,
        J_bar_Ha_abs=None,
        eps_Lya_destruction=eps_Lya_destruction,
        two_photon_decay=two_photon_decay,
        verbose=False,
    )

    history = {
        'L_line': [params['L_line']],
        'n_3_max': [float(params['n_upper'].max())],
        'J_bar_max': [0.0],
        'delta_max': [],
        # Per-iteration audit: which zone oscillated the most and how badly.
        # If the worst zone is always the same dense CDS zone with τ_LyC≫1,
        # the oscillation is the line being optically thick to itself
        # (radiative trapping) and L_line invariance confirms the global
        # answer is correct despite local pop noise.
        'worst_zone_idx': [],
        'worst_zone_delta': [],
        'worst_zone_n3': [],
        'worst_zone_X_HII': [],
        'worst_zone_n_e': [],
        'worst_zone_T': [],
        'worst_zone_r': [],
        'worst_zone_J_bar': [],
    }
    J_bar_prev = None

    for it in range(1, max_iter + 1):
        if verbose:
            print(f"[RT-iter] iter {it}: MC to measure J_bar, "
                  f"then re-solve NLTE")

        # Run MC with current populations
        mc = run_mc_chunked(snap, params,
                            n_per=n_iter_per, n_chunks=n_iter_chunks,
                            nbins=nbins,
                            band_AA=band_AA,
                            source_padding_AA=source_padding_AA,
                            calibration=calibration,
                            line_redistribution=line_redistribution,
                            verbose=False)

        # The line opacity in the kernel uses n_lower_absorb (when
        # photoion_decoupled mode), so J_bar measurement must use that.
        # Fall back to n_lower if n_lower_absorb isn't in params.
        n_2_for_jbar = params.get('n_lower_absorb', params['n_lower'])
        J_bar_new = compute_jbar_from_mc(
            snap, n_2_for_jbar,
            mc['stats_continuum'], mc['stats_volumetric'],
            mc['L_per_pkt_cont'], mc['L_per_pkt_vol'],
            n_chunks=mc['n_chunks'],
            verbose=verbose,
        )

        # Damp the J_bar update to suppress oscillations
        if J_bar_prev is not None:
            J_bar = damping * J_bar_new + (1.0 - damping) * J_bar_prev
        else:
            J_bar = J_bar_new

        # Save previous n_3 for convergence check
        n_3_prev = params['n_upper'].copy()

        # Re-solve populations with J_bar pumping
        params = derive_parameters_from_state(
            state, snap, line_name='Halpha',
            wavelength_band_AA=band_AA,
            populations_mode='nlte', nlte_levels=5,
            emission_model='caseb_hr_attenuated_split',
            f_wing=0.02,
            ionization_mode=ionization_mode,
            f_HI_max=effective_f_HI_max,
            stromgren_exponent=stromgren_p,
            J_bar_Ha_abs=J_bar,
            eps_Lya_destruction=eps_Lya_destruction,
            two_photon_decay=two_photon_decay,
            verbose=False,
        )

        n_3_new = params['n_upper'].copy()
        # Convergence on relative change in n_3 (only where both are positive)
        mask = (n_3_prev > 0) & (n_3_new > 0)
        if mask.sum() > 0:
            rel_delta = np.zeros_like(n_3_prev)
            rel_delta[mask] = np.abs(n_3_new[mask] - n_3_prev[mask]) / n_3_prev[mask]
            delta_max = float(rel_delta.max())
            worst_idx = int(np.argmax(rel_delta))
        else:
            delta_max = 0.0
            worst_idx = 0
            rel_delta = np.zeros_like(n_3_prev)

        # Record per-iteration audit: which zone is the oscillation in, and
        # what's the local state? Reads from snap and params at worst_idx.
        worst_zone_X_HII = float(snap.get('X_HII', np.zeros_like(snap['r']))[worst_idx])
        worst_zone_n_e   = float(snap['n_e'][worst_idx])
        worst_zone_T     = float(snap['T'][worst_idx])
        worst_zone_r     = float(snap['r'][worst_idx])
        worst_zone_n3    = float(n_3_new[worst_idx])
        worst_zone_J_bar = float(J_bar[worst_idx]) if hasattr(J_bar, '__len__') else 0.0

        history['L_line'].append(params['L_line'])
        history['n_3_max'].append(float(n_3_new.max()))
        history['J_bar_max'].append(float(J_bar.max()))
        history['delta_max'].append(delta_max)
        history['worst_zone_idx'].append(worst_idx)
        history['worst_zone_delta'].append(delta_max)
        history['worst_zone_n3'].append(worst_zone_n3)
        history['worst_zone_X_HII'].append(worst_zone_X_HII)
        history['worst_zone_n_e'].append(worst_zone_n_e)
        history['worst_zone_T'].append(worst_zone_T)
        history['worst_zone_r'].append(worst_zone_r)
        history['worst_zone_J_bar'].append(worst_zone_J_bar)

        if verbose:
            print(f"           L_line = {params['L_line']:.3e} erg/s, "
                  f"J_bar_max = {J_bar.max():.3e} erg/s/cm²/sr/Hz, "
                  f"Δn_3_max = {delta_max*100:.1f}% (damping={damping})")

        if delta_max < tol:
            if verbose:
                print(f"[RT-iter] converged at iteration {it} "
                      f"(Δn_3_max = {delta_max*100:.2f}% < {tol*100:.1f}%)")
            break

        J_bar_prev = J_bar
    else:
        if verbose:
            print(f"[RT-iter] reached max_iter ({max_iter}) without "
                  f"converging below tol={tol*100:.1f}%. Using last params.")

    return {
        'params': params,
        'state': state,
        'history': history,
        'J_bar_final': J_bar if it > 0 else None,
        'n_iter': it,
    }


# ---------- Single-snapshot processing ----------

def process_snapshot(snap_path, n_per=100_000, n_chunks=2,
                      n_iter_per=50_000, n_iter_chunks=1,
                      max_iter=6, tol=0.03, damping=0.3,
                      stromgren_p=0.5, f_HI_max=0.1,
                      nbins=1200, smooth_kms=25.0,
                      band_AA=(6200.0, 6950.0),
                      source_padding_AA=1500.0,
                      calibration='auto',
                      fmt='auto',
                      do_rt_iter=True, ref_path=None,
                      out_prefix=None,
                      line_profile_method='mc',
                      line_profile_lock=False,
                      extend_wind=False,
                      wind_r_max_factor=20.0,
                      wind_n_zones=100,
                      wind_T_photoionized=10000.0,
                      wind_rho_index=2.0,
                      wind_density_boost=1.0,
                      photoionize=True,
                      photoionize_T_source=None,
                      photoionize_T_eq_floor=1.0e4,
                      include_shock_xray=True,
                      eps_Lya_destruction=None,
                      two_photon_decay=False,
                      line_redistribution='aa_prd',
                      photosphere_mode='es',
                      photosphere_lam_ref_AA=6562.8,
                      compute_he1_nlte=False,
                      he1_eps_resonance=None,
                      he1_two_photon_decay=True,
                      he1_ionization_mode='follow_H',
                      compute_he2_nlte=False,
                      he2_x_heiii_mode='saha_local',
                      he2_x_heiii_scalar=None,
                      he2_x_heiii_fraction=None,
                      compute_he_lines=False,
                      he_lines_n_packets=50_000,
                      he_lines_calibration='f_cont_bb',
                      he_lines_use_existing_kernel=True,
                      saturated_rt=False,
                      he_budget=False,
                      metal_lines=False,
                      metal_cloudy=False,
                      narrow_csm=False,
                      verbose=True):
    """Full processing for one snapshot. Returns result dict.

    band_AA : (lo, hi)
        Output wavelength band in Angstroms. Default (6200, 6950) covers
        Hα ±~9000 km/s. For cool-photosphere STELLA snapshots where
        Hα is line-dominated across the band, try a wider band (e.g.
        (5500, 7500)) so the band edges are in clean continuum and the
        F_norm baseline is meaningful. source_padding_AA must extend
        beyond the requested band.

    calibration : str
        How to set the F_norm normalization baseline:
          'theoretical_ew' (default) - F_baseline = mean of F_cont at far
            wings. Reproduces standard F/F_cont convention. Breaks down
            when L_line dominates over L_cont in the band (cool STELLA
            photosphere + strong Hα): cont channel near line is filled
            by line-scattered cont photons → baseline includes line
            contribution → F_norm peak inflated.
          'f_cont_bb' - F_baseline = scalar mean of diluted BB at far
            wings. Removes line-scattered contamination but doesn't
            capture BB slope across band.
          'f_cont_bb_lambda' - F_baseline(λ) = diluted BB at each λ.
            Best convention for line-dominated cases (gives the ratio
            of observed flux to bare-photosphere BB, matching what
            observers report when fitting continuum through line).
    """
    t_start = time.time()
    name = snap_name_from_path(snap_path)
    if out_prefix is None:
        out_prefix = f"prod_{name}"

    print(f"\n{'='*78}")
    print(f" PROCESS {name} ({snap_path})")
    print(f"{'='*78}")

    snap = load_snapshot(snap_path, fmt=fmt, verbose=verbose,
                          extend_wind=extend_wind,
                          wind_r_max_factor=wind_r_max_factor,
                          wind_n_zones=wind_n_zones,
                          wind_T_photoionized=wind_T_photoionized,
                          wind_rho_index=wind_rho_index,
                          wind_density_boost=wind_density_boost,
                          photoionize=photoionize,
                          photoionize_T_source=photoionize_T_source,
                          photoionize_T_eq_floor=photoionize_T_eq_floor,
                          include_shock_xray=include_shock_xray,
                          photosphere_mode=photosphere_mode,
                          photosphere_lam_ref_AA=photosphere_lam_ref_AA)
    if verbose and snap.get('format') == 'stella':
        xh_emit = float(snap.get('X_H_emit', 0.0))
        print(f"  [STELLA] ⟨X_H⟩ in emit zones = {xh_emit:.4f}")
        print(f"  [STELLA] epoch label is 'days post Lbol max' (not days since explosion)")
        if xh_emit < 1.0e-3:
            print(f"  ⚠  WARNING: ⟨X_H⟩(emit) = {xh_emit:.2e} is essentially zero.")
            print(f"     The emitting zones contain no hydrogen → Hα emission "
                  f"will be ~0. Expect a flat F=1 continuum spectrum.")
            print(f"     This is physical for stripped-envelope progenitors but "
                  f"may indicate the wrong STELLA model if you expected H-rich CSM.")

    if do_rt_iter:
        # RT iteration phase
        print(f"\n[1/2] RT-NLTE iteration "
              f"(max_iter={max_iter}, tol={tol*100:.1f}%, damping={damping})")
        rt = run_rt_iteration(
            snap, n_iter_per=n_iter_per, n_iter_chunks=n_iter_chunks,
            max_iter=max_iter, tol=tol, damping=damping,
            stromgren_p=stromgren_p, f_HI_max=f_HI_max,
            nbins=nbins,
            band_AA=band_AA,
            source_padding_AA=source_padding_AA,
            calibration=calibration,
            eps_Lya_destruction=eps_Lya_destruction,
            two_photon_decay=two_photon_decay,
            line_redistribution=line_redistribution,
            verbose=verbose,
        )
        params = rt['params']
        state = rt['state']
        history = rt['history']
        J_bar_final = rt['J_bar_final']
        n_iter_used = rt['n_iter']
    else:
        # No iteration; use baseline NLTE only
        print(f"\n[1/2] Baseline NLTE only (RT iteration disabled)")
        state = analyze_snapshot(snap)
        # Same critical fix as in run_rt_iteration: force the photospheric
        # values to come from snap (truncate-derived), not from analyze_snapshot
        # which can leak wind_T_photoionized=10000 K.
        for _attr, _snap_key, _fallback_key in (
                ('T_phot', 'T_phot_inner', 'T_phot'),
                ('R_phot', 'R_phot_inner', 'R_phot'),
                ('L_phot', 'L_phot_inner', 'L_phot')):
            _v = snap.get(_snap_key, snap.get(_fallback_key, None))
            if _v is not None and np.isfinite(_v) and _v > 0:
                setattr(state, _attr, float(_v))
        # Override state.X_H from STELLA per-zone composition (see comment
        # in run_rt_iteration for rationale).
        if snap.get('format') == 'stella' and 'X_H' in snap:
            state.X_H = np.asarray(snap['X_H'], dtype=float)
            print(f"  [STELLA] state.X_H overridden: "
                  f"range {state.X_H.min():.3f} → {state.X_H.max():.3f}")
        # Same for X_He (helium mass fraction from STELLA's 'he4' column)
        if snap.get('format') == 'stella' and 'X_He' in snap:
            state.X_He = np.asarray(snap['X_He'], dtype=float)
            print(f"  [STELLA] state.X_He overridden: "
                  f"range {state.X_He.min():.3f} → {state.X_He.max():.3f}")
        # Auto-select ionization mode (same as RT iteration)
        if snap.get('format') == 'stella':
            ionization_mode = 'saha'
            effective_f_HI_max = 1.0
        else:
            ionization_mode = 'photoion_decoupled'
            effective_f_HI_max = f_HI_max
        print(f"  ionization_mode = {ionization_mode}, "
              f"f_HI_max = {effective_f_HI_max}")
        params = derive_parameters_from_state(
            state, snap, line_name='Halpha',
            wavelength_band_AA=band_AA,
            populations_mode='nlte', nlte_levels=5,
            emission_model='caseb_hr_attenuated_split',
            f_wing=0.02,
            ionization_mode=ionization_mode,
            f_HI_max=effective_f_HI_max,
            stromgren_exponent=stromgren_p,
            eps_Lya_destruction=eps_Lya_destruction,
            two_photon_decay=two_photon_decay,
            verbose=False,
        )
        history = {'L_line': [params['L_line']], 'n_3_max': [],
                   'J_bar_max': [], 'delta_max': []}
        J_bar_final = None
        n_iter_used = 0

    # === Phase 4: He II NLTE (opt-in; runs BEFORE Phase 3 for v2 coupling) ===
    # Computes He II level populations (10 levels, hydrogenic Z=2) and the
    # He II / He III ionization split. When --he1-nlte is ALSO on, Phase 4
    # exports X_HeIII to Phase 3, which then subtracts it from X_HII before
    # passing to the He I solver — preventing double-counting of He III.
    #
    # Default X_HeIII determination: local Saha at gas T, n_e (correct for
    # T < 25000 K plateau conditions; near-zero in cool regime so the
    # coupling is essentially a no-op in plateau). For hot IIn / shock-
    # photoionized regimes, use --he2-x-heiii-mode=photoeq_match or override
    # with --he2-x-heiii-scalar.
    he2_metrics_local = None
    he2_X_HeIII_arr = None       # captured for Phase 3 coupling, if any
    if compute_he2_nlte:
        try:
            import types as _types
            from snline_he2_integration import add_he2_populations_to_state, \
                                                  he2_summary_dict
            he2_input = _types.SimpleNamespace()
            he2_input.r   = np.asarray(snap['r'],   dtype=float)
            he2_input.v   = np.asarray(snap['v'],   dtype=float)
            he2_input.rho = np.asarray(snap['rho'], dtype=float)
            he2_input.T   = np.asarray(snap['T'],   dtype=float)
            he2_input.n_e = np.asarray(snap['n_e'], dtype=float)
            # X_HII for the wrapper's _derive_X_HII_for_He2 lookup
            X_HII_zone = snap.get('X_HII', None)
            if X_HII_zone is not None:
                he2_input.X_HII = np.asarray(X_HII_zone, dtype=float)
            # X_H for the wrapper's _get_He_mass_fraction → uses fallback
            if hasattr(state, 'X_H') and state.X_H is not None:
                X_H_zone = np.asarray(state.X_H, dtype=float)
            elif 'X_H' in snap:
                X_H_zone = np.asarray(snap['X_H'], dtype=float)
            else:
                X_H_zone = np.full(len(he2_input.r), 0.7)
            if X_H_zone.size != he2_input.r.size:
                X_H_zone = np.full(he2_input.r.size,
                                    float(np.mean(X_H_zone)))
            he2_input.X_H = X_H_zone
            # X_HeIII override → broadcast to per-zone array
            X_HeIII_ext = None
            if he2_x_heiii_fraction is not None:
                # fraction-of-X_HII: physically self-consistent across regimes
                _XHII = (np.asarray(X_HII_zone, dtype=float)
                          if X_HII_zone is not None
                          else np.zeros(len(he2_input.r)))
                X_HeIII_ext = float(he2_x_heiii_fraction) * _XHII
            elif he2_x_heiii_scalar is not None:
                # scalar uniform: only safe when X_HII ≥ scalar everywhere
                X_HeIII_ext = np.full(len(he2_input.r),
                                       float(he2_x_heiii_scalar))
            add_he2_populations_to_state(
                he2_input,
                X_HeIII_mode=he2_x_heiii_mode,
                X_HeIII_external=X_HeIII_ext,
                print_summary=True)
            # Stash X_HeIII for Phase 3 coupling
            if hasattr(he2_input, 'he2_diag') and \
                    isinstance(he2_input.he2_diag, dict):
                he2_X_HeIII_arr = he2_input.he2_diag.get('X_HeIII', None)
            # Hand results back to state for downstream consumers
            for _attr in ('he2_levels', 'he2_diag', 'he2_tau'):
                if hasattr(he2_input, _attr):
                    try:
                        setattr(state, _attr, getattr(he2_input, _attr))
                    except Exception:
                        pass
            # Build CSV metrics dict
            try:
                he2_metrics_local = he2_summary_dict(he2_input)
            except Exception as me:
                print(f"[he2_metrics] WARNING: summary failed: {me}")
        except Exception as e:
            print(f"[he2_nlte] WARNING: He II NLTE failed: {e}")
            import traceback; traceback.print_exc()

    # === Phase 3: He I NLTE (opt-in, runs AFTER Phase 4 if both enabled) ===
    # Computes He I populations using converged H NLTE results to derive
    # X_HII per zone for the 'follow_H' ionization split. Does not affect
    # the Hα MC output (that is Phase 5); only attaches populations + line τ
    # to a local namespace and builds CSV metrics inline.
    #
    # Architecture note: in this pipeline the hydro arrays (rho, T, n_e, r, v)
    # live in the `snap` dict, NOT on the PhysicalState object. The wrapper
    # `add_he1_populations_to_state` reads `state.rho`, `state.T`, etc., so
    # we pass it a SimpleNamespace built from `snap` rather than the actual
    # PhysicalState (which would raise AttributeError on `.rho`).
    he1_metrics_local = None     # collected here, attached to result below
    if compute_he1_nlte:
        try:
            import types as _types
            from snline_he1_integration import add_he1_populations_to_state
            # Build a state-like namespace from `snap` for the wrapper.
            he1_input = _types.SimpleNamespace()
            he1_input.r   = np.asarray(snap['r'],   dtype=float)
            he1_input.v   = np.asarray(snap['v'],   dtype=float)
            he1_input.rho = np.asarray(snap['rho'], dtype=float)
            he1_input.T   = np.asarray(snap['T'],   dtype=float)
            he1_input.n_e = np.asarray(snap['n_e'], dtype=float)
            # X_H per-zone: prefer the value set on the PhysicalState
            # (matches what H NLTE saw); fall back to snap['X_H'] or 0.7.
            if hasattr(state, 'X_H') and state.X_H is not None:
                X_H_zone = np.asarray(state.X_H, dtype=float)
            elif 'X_H' in snap:
                X_H_zone = np.asarray(snap['X_H'], dtype=float)
            else:
                X_H_zone = np.full(len(he1_input.r), 0.7)
            if X_H_zone.size != he1_input.r.size:
                X_H_zone = np.full(he1_input.r.size,
                                    float(np.mean(X_H_zone)))
            he1_input.X_H = X_H_zone
            # Build h_diag for the wrapper from converged photoeq X_HII.
            X_HII_zone = snap.get('X_HII', None)
            if X_HII_zone is not None:
                X_HII_zone = np.asarray(X_HII_zone, dtype=float)
                m_H = 1.673534e-24
                n_H_total = X_H_zone * he1_input.rho / m_H
                he1_input.h_diag = {
                    'n_p':  X_HII_zone * n_H_total,
                    'n_HI': (1.0 - X_HII_zone) * n_H_total,
                }
            add_he1_populations_to_state(
                he1_input,
                eps_He_resonance=he1_eps_resonance,
                two_photon_decay=he1_two_photon_decay,
                ionization_split_mode=he1_ionization_mode,
                X_HeIII_external=he2_X_HeIII_arr,    # v2 coupling
                print_summary=True)
            # Hand the results to the actual PhysicalState too so downstream
            # consumers (future Phase 5 MC) can find them by documented names.
            for _attr in ('he1_levels', 'he1_diag', 'he1_tau'):
                if hasattr(he1_input, _attr):
                    try:
                        setattr(state, _attr, getattr(he1_input, _attr))
                    except Exception:
                        pass    # state may use __slots__; non-fatal
            # ---- Build CSV metrics dict defensively, RIGHT HERE ----
            # Read directly off he1_input (which we know has the attrs)
            # rather than state (which may have rejected the setattr).
            try:
                n_lev = getattr(he1_input, 'he1_levels', None)
                diag  = getattr(he1_input, 'he1_diag',   None)
                ltau  = getattr(he1_input, 'he1_tau',    None)
                if n_lev is not None and diag is not None and ltau is not None:
                    def _med(arr, default=float('nan')):
                        if arr is None: return default
                        a = np.asarray(arr, dtype=float)
                        if a.size == 0: return default
                        return float(np.median(a))
                    n_HeI = diag.get('n_HeI_total',
                                       diag.get('n_HeI',
                                                 diag.get('n_He_I', None)))
                    n_HeII = diag.get('n_HeII',
                                       diag.get('n_He_II', None))
                    iters_val = diag.get('iterations',
                                          diag.get('iters',
                                                    diag.get('n_iter', 13)))
                    T_arr = he1_input.T
                    T_med = float(np.median(T_arr)) if T_arr.size else 1e4
                    EV_KB = 1.602e-12 / 1.381e-16
                    lte = 3.0 * np.exp(-19.82 * EV_KB / max(T_med, 100.0))
                    n_2_3S = _med(n_lev[1])
                    n_1_1S = _med(n_lev[0])
                    nlte_ratio = n_2_3S / max(n_1_1S, 1e-30)
                    n_HeI_m  = _med(n_HeI)
                    n_HeII_m = _med(n_HeII)
                    he1_metrics_local = {
                        'he1_n_HeI_med':      n_HeI_m,
                        'he1_n_HeII_med':     n_HeII_m,
                        'he1_2_3S_frac':      (n_2_3S / max(n_HeI_m, 1e-30)
                                                if not np.isnan(n_HeI_m) else float('nan')),
                        'he1_NLTE_LTE_ratio': nlte_ratio / max(lte, 1e-300),
                        'he1_tau_10830_med':  _med(ltau.get('He_I_10830')),
                        'he1_tau_5876_med':   _med(ltau.get('He_I_5876')),
                        'he1_tau_7065_med':   _med(ltau.get('He_I_7065')),
                        'he1_tau_6678_med':   _med(ltau.get('He_I_6678')),
                        'he1_iters':          int(iters_val),
                    }
                else:
                    print(f"[he1_metrics] he1_input attrs missing: "
                          f"he1_levels={n_lev is not None}, "
                          f"he1_diag={diag is not None}, "
                          f"he1_tau={ltau is not None}")
            except Exception as me:
                print(f"[he1_metrics] WARNING: defensive build failed: {me}")
                import traceback; traceback.print_exc()
        except Exception as e:
            print(f"[he1_nlte] WARNING: He I NLTE failed: {e}")
            import traceback; traceback.print_exc()

    # Production MC with converged populations
    # Auto-select calibration based on L_line/L_cont_band ratio:
    #   ratio < 0.3 → 'theoretical_ew' (line is small perturbation on cont)
    #   ratio ≥ 0.3 → 'f_cont_bb_lambda' (line-dominated, cont channel
    #                  near line is contaminated by line scattering;
    #                  use diluted-BB per-λ as baseline for an observer-
    #                  comparable F/F_cont)
    if calibration == 'auto':
        L_line_val = float(params.get('L_line', 0.0))
        L_cont_band_val = float(params.get('L_cont_band',
                                            params.get('L_cont_src', 1.0)))
        ratio = L_line_val / max(L_cont_band_val, 1e-30)
        if ratio >= 0.3:
            calibration_used = 'f_cont_bb_lambda'
        else:
            calibration_used = 'theoretical_ew'
        print(f"\n  [auto-calibration] L_line/L_cont_band = {ratio:.3f} → "
              f"using '{calibration_used}'")
    else:
        calibration_used = calibration

    print(f"\n[2/2] Production MC: {n_per * n_chunks:,} packets "
          f"(calibration={calibration_used})")
    mc = run_mc_chunked(snap, params, n_per=n_per, n_chunks=n_chunks,
                        nbins=nbins,
                        band_AA=band_AA,
                        source_padding_AA=source_padding_AA,
                        calibration=calibration_used,
                        line_redistribution=line_redistribution,
                        verbose=True)

    # Extract calibrated F_norm directly (do NOT recompute from F_total_abs!)
    lam0 = 6562.81
    lam = mc['lam']
    dv = (lam / lam0 - 1.0) * 2.998e5
    F_cont_abs = mc['F_cont_abs']
    F_vol_abs = mc['F_vol_abs']
    F_total_abs = mc['F_total_abs']
    F_baseline = mc['F_baseline']
    F_norm_raw = mc['F_norm']                  # CALIBRATED, raw MC

    # --- Source-function formal solution (fixes the spurious blueward emission
    # peak of the thin-shell recombination MC channel). When selected, the
    # emergent line profile is recomputed as the Sobolev P-Cygni formal solution
    # with the scattering source function S_L = (1-eps) J_bar + eps B, where the
    # radiation field is the diluted photospheric continuum that the n=2->3
    # transition resonantly scatters. The MC continuum/diagnostic channels above
    # are retained; only the emergent F_norm is replaced. First-principles: no
    # floors/caps/normalization, tau_S and S_L emerge from the converged state.
    if line_profile_method == 'formal':
        if not _HAVE_FORMAL:
            print("  [formal] formal_line_profile unavailable; keeping MC profile.")
        else:
            sv = _flp.sobolev_validity(np.asarray(snap['r'], float),
                                       np.asarray(snap['v'], float))
            if not sv['valid']:
                # The Sobolev gate is ALWAYS authoritative for the profile shape:
                # forcing 'formal' onto a non-homologous (dense-CSM/IIn) snapshot
                # produces an unphysical absorption trough, so we never do it —
                # not even under --line-profile-method-lock (which only governs
                # the STRENGTH policy, handled downstream, not the shape).
                print(f"  [formal] SKIPPED — {sv['reason']}.")
                print(f"  [formal] The source-function Sobolev solution applies to "
                      f"homologous ejecta (IIP-like). This snapshot is the dense-CSM "
                      f"(IIn-like) regime, where Hα is recombination emission reprocessed "
                      f"by electron scattering — kept the MC profile (emission line).")
            else:
                tau_es_env = float(np.max(np.asarray(snap.get('tau_es', [0.0]))))
                res_formal = _flp.halpha_profile_from_state(
                    np.asarray(snap['r'], float), np.asarray(snap['v'], float),
                    np.asarray(params['n_lower'], float),
                    np.asarray(params['n_upper'], float),
                    np.asarray(snap['T'], float), np.asarray(snap['n_e'], float),
                    J_bar=None,
                    R_phot=float(params['R_phot']), T_phot=float(params['T_phot']),
                    line=LINE_LIB['Halpha'],
                    tau_es_env=tau_es_env, electron_scatter=True,
                    vgrid_kms=dv, jbar_source='scatter')
                F_norm_raw = np.asarray(res_formal['F_norm'], float)
                mc['F_total_abs'] = F_norm_raw * mc['F_baseline']
                F_total_abs = mc['F_total_abs']
                ip = int(np.nanargmax(F_norm_raw))
                print(f"  [formal] source-function P-Cygni: peak F={F_norm_raw[ip]:.2f} "
                      f"@ {dv[ip]:+.0f} km/s, eps_med={np.median(res_formal['eps']):.1e}, "
                      f"tau_es_env={tau_es_env:.2f}")

    # Apply Gaussian smoothing in velocity space to suppress per-bin MC noise
    # without erasing line structure. Default σ=10 km/s; FWHM~24 km/s, well
    # below typical Hα FWHM (≥80 km/s) so line profile is preserved.
    if smooth_kms > 0:
        F_norm = smooth_velocity(F_norm_raw, dv, smooth_kms)
        F_cont_smooth = smooth_velocity(F_cont_abs / F_baseline, dv, smooth_kms)
        F_vol_smooth = smooth_velocity(F_vol_abs / F_baseline, dv, smooth_kms)
        if verbose:
            dv_bin = float(np.median(np.diff(dv)))
            print(f"\n[smooth] Gaussian σ={smooth_kms:.1f} km/s "
                  f"({smooth_kms/dv_bin:.2f} bins) applied to F_norm")
    else:
        F_norm = F_norm_raw.copy()
        F_cont_smooth = F_cont_abs / F_baseline
        F_vol_smooth = F_vol_abs / F_baseline

    # Peak/trough
    mask_blue = (dv > -400) & (dv < -10)
    mask_red = (dv > 10) & (dv < 400)
    trough_F = float(F_norm[mask_blue].min())
    trough_dv = float(dv[mask_blue][F_norm[mask_blue].argmin()])
    red_peak_F = float(F_norm[mask_red].max())
    red_peak_dv = float(dv[mask_red][F_norm[mask_red].argmax()])
    global_peak_F = float(F_norm.max())
    global_peak_dv = float(dv[F_norm.argmax()])

    # CMFGEN reference if available
    F_ref = None
    sigma_res2 = None
    if ref_path is not None and os.path.exists(ref_path):
        F_ref = load_cmfgen_ref(ref_path, lam)
        mask_window = np.abs(dv) < 1500
        sigma_res2 = float(np.sum((F_norm[mask_window] - F_ref[mask_window])**2))

    runtime_total = time.time() - t_start

    result = {
        'name': name,
        'snap_path': snap_path,
        'epoch_d': snap.get('epoch_d'),
        'format': snap.get('format', 'heracles'),
        'lam': lam, 'dv': dv,
        'F_norm': F_norm,                    # smoothed (used for metrics + display)
        'F_norm_raw': F_norm_raw,            # raw MC, no smoothing (saved for reference)
        'F_cont_smooth': F_cont_smooth,
        'F_vol_smooth': F_vol_smooth,
        'F_cont_abs': F_cont_abs,
        'F_vol_abs': F_vol_abs,
        'F_total_abs': F_total_abs,
        'F_baseline': F_baseline,
        'smooth_kms': smooth_kms,
        'L_line': float(np.nan_to_num(params['L_line'], nan=0.0,
                                        posinf=0.0, neginf=0.0)),
        'L_cont_band': float(np.nan_to_num(params['L_cont_band'], nan=0.0,
                                            posinf=0.0, neginf=0.0)),
        'red_peak_F': red_peak_F, 'red_peak_dv': red_peak_dv,
        'trough_F': trough_F, 'trough_dv': trough_dv,
        'global_peak_F': global_peak_F, 'global_peak_dv': global_peak_dv,
        'F_ref': F_ref,
        'sigma_res2': sigma_res2,
        'history': history,
        'J_bar_final': J_bar_final,
        'n_iter_used': n_iter_used,
        'runtime_total': runtime_total,
    }

    # ----- He I NLTE metrics (Phase 3) -----
    # Attached from the local variable populated by the Phase 3 block above.
    # he1_metrics_local is None if --he1-nlte was off or the solver failed.
    result['he1_metrics'] = he1_metrics_local
    # ----- He II NLTE metrics (Phase 4) -----
    # Same pattern as he1_metrics_local. None if --he2-nlte was off.
    result['he2_metrics'] = he2_metrics_local

    # ----- snap-level diagnostic metrics (for batch CSV + per-snap audit) -----
    # Pulled here so all downstream tools see the same canonical values.
    snap_metrics = {
        'R_phot':          float(snap.get('R_phot_inner', snap['r'][0])),
        'T_phot':          float(snap.get('T_phot_inner', snap['T'][0])),
        'L_phot':          float(snap.get('L_phot_inner', 0.0)),
        'T_color_thermalization':
                           float(snap.get('T_color_thermalization', np.nan)),
        'n_zones':         int(len(snap['r'])),
        'n_zones_full':    int(snap.get('n_zones_full', len(snap['r']))),
        'tau_es_stella':   float(np.asarray(snap.get('tau_es_stella',
                                                       snap['tau_es'])).max()),
        'tau_es_photoeq':  float(np.asarray(snap['tau_es']).max()),
        'photoionized':    bool(snap.get('photoionized', False)),
    }
    # Photoionization details if available
    pi_params = snap.get('photoionization_params', None)
    if pi_params is not None:
        snap_metrics.update({
            'photoeq_T_source':       float(pi_params.get('T_source', np.nan)),
            'photoeq_converged':      bool(pi_params.get('converged', False)),
            'photoeq_iterations':     int(pi_params.get('iterations_used', 0)),
            'photoeq_final_residual': float(pi_params.get('final_residual', np.nan)),
            'X_HII_mean':             float(np.asarray(snap.get('X_HII',
                                                                  [np.nan])).mean()),
        })
        # Per-zone Γ at the photosphere (zone 0)
        Gamma_BB = snap.get('Gamma_HI_BB', None)
        Gamma_X  = snap.get('Gamma_HI_brems', None)
        if Gamma_BB is not None and len(Gamma_BB) > 0:
            snap_metrics['Gamma_BB_at_Rphot']    = float(Gamma_BB[0])
        if Gamma_X is not None and len(Gamma_X) > 0:
            snap_metrics['Gamma_brems_at_Rphot'] = float(Gamma_X[0])
        # Shock parameters if shock-Xray was on
        sp = pi_params.get('shock_params', None)
        if sp is not None:
            snap_metrics.update({
                'shock_R_s':      float(sp.get('R_s', np.nan)),
                'shock_v_s_kms':  float(sp.get('v_s', np.nan))/1e5,
                'shock_rho_csm':  float(sp.get('rho_csm', np.nan)),
                'shock_T_shock':  float(sp.get('T_shock', np.nan)),
                'shock_L_shock':  float(sp.get('L_shock', np.nan)),
                'shock_eta_rad':  float(sp.get('eta_rad', np.nan)),
                'shock_L_X_brems':float(sp.get('L_X_brems', np.nan)),
                'shock_sanity_flag': bool(sp.get('sanity_flag', False)),
                'shock_sanity_ratio_LXoverLphot':
                                  float(sp.get('sanity_ratio_L_X_over_L_phot', np.nan)),
            })
    result['snap_metrics'] = snap_metrics

    # ------- Phase 5: synthetic multi-line He profiles (optional) -------
    # Triggered by --he-lines. Requires Phase 3 (--he1-nlte) and/or Phase 4
    # (--he2-nlte) to have populated state.he1_levels, state.he1_tau,
    # state.he2_levels, state.he2_tau. Output: {out_prefix}_he.npz, .txt, .png
    # with per-line F_λ over ±5000 km/s windows for the 8 He I + He II lines.
    if compute_he_lines:
        try:
            from phase5_runner import run_phase5_for_state
            # Attach H NLTE level populations to state (Phase 2 stored them
            # in params['populations_diag']['n_levels'] but never set them on
            # state). Phase 5 H-line factory reads state.h_levels.
            if isinstance(params, dict) and 'populations_diag' in params:
                pd = params['populations_diag']
                if isinstance(pd, dict) and 'n_levels' in pd:
                    state.h_levels = np.asarray(pd['n_levels'], dtype=float)

            # ---- DEBUG: per-zone Hα emission breakdown ----
            # Triggered by env var SNLINE_DEBUG_DAY80=1 so it only fires for
            # the targeted snapshot, not all 30 epochs in a batch.
            #   Usage: SNLINE_DEBUG_DAY80=1 python production_runner.py \
            #              mesa.day080_post_Lbol_max.data --format stella \
            #              --he1-nlte --he2-nlte --he2-x-heiii-fraction 0.20 \
            #              --he-lines --he-lines-n-packets 50000 \
            #              --out-prefix debug_day80
            # Distinguishes between hypotheses about where the day-80 Hα
            # emission is actually coming from in the NLTE populations,
            # before the kernel takes over. Outputs are printed and saved
            # to debug_day80_nlte.npz for downstream inspection.
            import os as _os
            if _os.environ.get('SNLINE_DEBUG_DAY80') == '1':
                try:
                    h_lev = state.h_levels
                    # Hydro arrays live on snap (not state). state holds NLTE
                    # diagnostics (h_levels, he1_levels, etc.) plus scalars
                    # like T_phot, R_phot, L_phot, X_H.
                    _r = np.asarray(snap['r'])
                    _v = np.asarray(snap['v'])
                    _T = np.asarray(snap['T'])
                    _ne = np.asarray(snap['n_e'])
                    _n1 = h_lev[0]; _n2 = h_lev[1]; _n3 = h_lev[2]

                    # Print current state values to confirm fix took effect
                    print(f'\n[DEBUG day80] state.T_phot = {getattr(state, "T_phot", "?")}')
                    print(f'[DEBUG day80] state.R_phot = {getattr(state, "R_phot", "?")}')
                    print(f'[DEBUG day80] state.L_phot = {getattr(state, "L_phot", "?")}')
                    print(f'[DEBUG day80] snap[T_phot_inner] = {snap.get("T_phot_inner", "?")}')

                    # Hα emissivity (gross, before re-absorption)
                    _A_Ha = 4.4101e7
                    _h_pl = 6.62607015e-27
                    _lam_Ha_cm = 6562.81e-8
                    _C_LIGHT = 2.998e10
                    _nu_Ha = _C_LIGHT / _lam_Ha_cm
                    _j_Ha = _n3 * _A_Ha * _h_pl * _nu_Ha / (4.0 * np.pi)

                    # Sobolev tau & beta for Hα
                    _dv_dr = np.abs(np.gradient(_v, _r))
                    _dv_dr = np.maximum(_dv_dr, 1e-30)
                    # σ_λ for Hα in cgs: (πe²/m_e c) × f_lu × λ
                    # ≈ 0.02654 × 0.6407 × 6.563e-5
                    _sigma_lam_Ha = 1.116e-6
                    _g2 = 8.0; _g3 = 18.0
                    _n_diff = np.maximum(_n2 - (_g2 / _g3) * _n3, 0.0)
                    _tau_Ha = _sigma_lam_Ha * _n_diff / _dv_dr

                    with np.errstate(over='ignore', invalid='ignore'):
                        _beta_Ha = np.where(
                            _tau_Ha > 1e-6,
                            (1.0 - np.exp(-_tau_Ha)) / _tau_Ha,
                            1.0 - 0.5 * _tau_Ha)
                    _beta_Ha = np.clip(_beta_Ha, 0.0, 1.0)

                    # Per-zone luminosity contribution (gross emission × β)
                    _dr = np.gradient(_r)
                    _dV = 4.0 * np.pi * _r**2 * np.abs(_dr)
                    _dL_Ha = _j_Ha * 4.0 * np.pi * _beta_Ha * _dV
                    _dL_Ha_gross_no_beta = _j_Ha * 4.0 * np.pi * _dV

                    _v_kms = _v / 1e5
                    print('\n[DEBUG day80] Hα emission breakdown')
                    print(f'  {"v_bin [km/s]":<16} {"L_gross[erg/s]":<16} {"%":<7} '
                          f'{"L_w_beta[erg/s]":<17} {"%":<7}')
                    _total_gross = _dL_Ha_gross_no_beta.sum()
                    _total_beta  = _dL_Ha.sum()
                    for _v_lo, _v_hi in [(0, 500), (500, 1500), (1500, 2500),
                                          (2500, 4000), (4000, 10000)]:
                        _m = (_v_kms >= _v_lo) & (_v_kms < _v_hi)
                        _Lg = _dL_Ha_gross_no_beta[_m].sum()
                        _Lb = _dL_Ha[_m].sum()
                        _pct_g = 100 * _Lg / _total_gross if _total_gross > 0 else 0
                        _pct_b = 100 * _Lb / _total_beta if _total_beta > 0 else 0
                        print(f'  [{_v_lo:>4},{_v_hi:<5}]      '
                              f'{_Lg:.3e}        {_pct_g:5.1f}    '
                              f'{_Lb:.3e}          {_pct_b:5.1f}')
                    print(f'\n  TOTAL gross (no β):        {_total_gross:.3e} erg/s')
                    print(f'  TOTAL gross × β (Sobolev): {_total_beta:.3e} erg/s')
                    _L_line_str = (f"{result.get('L_line'):.3e}"
                                    if isinstance(result, dict) and 'L_line' in result
                                    else '?')
                    print(f'  Pipeline-reported L_line: {_L_line_str} erg/s')
                    print(f'\n  Median τ_Hα: {np.median(_tau_Ha):.2e}')
                    print(f'  Median β_Hα: {np.median(_beta_Ha):.2e}')
                    print(f'  Median n_2:  {np.median(_n2):.2e} cm⁻³')
                    print(f'  Median n_3:  {np.median(_n3):.2e} cm⁻³')
                    print(f'  n_e range:   {_ne.min():.2e} to {_ne.max():.2e} cm⁻³')

                    np.savez('debug_day80_nlte.npz',
                              r=_r, v=_v, T=_T, n_e=_ne,
                              n_1=_n1, n_2=_n2, n_3=_n3,
                              tau_Ha=_tau_Ha, beta_Ha=_beta_Ha,
                              j_Ha=_j_Ha, dL_Ha=_dL_Ha,
                              dL_Ha_gross_no_beta=_dL_Ha_gross_no_beta)
                    print('  [DEBUG] Wrote debug_day80_nlte.npz\n')
                except Exception as _e:
                    print(f'[DEBUG day80] dump failed: {_e}')
                    import traceback as _tb; _tb.print_exc()
            # ---- END DEBUG ----

            # Re-build a snap dict if needed: process_snapshot keeps `snap`
            # as the local hydro dict, while NLTE outputs live on `state`.
            snap_for_phase5 = dict(snap)   # shallow copy of hydro
            # Augment with photosphere fields (Phase 5 uses these for cont.)
            snap_for_phase5.setdefault('R_phot_cm', float(params['R_phot']))
            snap_for_phase5.setdefault('T_phot',    float(params['T_phot']))
            if 'L_phot' in params:
                snap_for_phase5.setdefault('L_phot', float(params['L_phot']))
            snap_for_phase5.setdefault('v_turb_kms', float(params.get('v_turb_kms', 20.0)))
            if 'v_turb_kms_grid' in params:
                snap_for_phase5.setdefault('v_turb_kms_grid',
                                            np.asarray(params['v_turb_kms_grid']))
            # Stash shock X-ray params for metal-line photoionization (P2 #5)
            _piP = snap.get('photoionization_params')
            if isinstance(_piP, dict):
                _spp = _piP.get('shock_params')
                if isinstance(_spp, dict):
                    snap_for_phase5.setdefault('L_X_brems',
                                               float(_spp.get('L_X_brems', 0.0)))
                    snap_for_phase5.setdefault('T_shock',
                                               float(_spp.get('T_shock', 0.0)))
            print(f"\n[phase5] Computing He multi-line profiles "
                  f"({he_lines_n_packets:,} packets/line)")
            # Build production-Hα cross-check info from the just-finished MC.
            # Used by phase5_runner to print a side-by-side comparison header
            # in {prefix}_lines.txt: the ratio of Phase-5-β-only Hα to
            # RT-iterated Hα is the systematic factor to apply to the He
            # estimates.
            prod_Ha_info = None
            try:
                # NOTE: cannot use `a or b` here because result['F_norm'] is a
                # numpy array; `or` triggers array truthiness which raises
                # ValueError and the exception is swallowed by the catch
                # below, silently disabling the Phase 5b empirical correction.
                # Use explicit None checks instead.
                F_norm_smooth = result.get('F_norm')
                if F_norm_smooth is None:
                    F_norm_smooth = result.get('F_norm_raw')
                # The production result dict stores the velocity axis as
                # result['dv'] (km/s) and wavelength as result['lam'] (AA).
                # Try those first, fall back to other plausible keys.
                F_dv = result.get('dv')
                if F_dv is None:
                    F_dv = result.get('dv_kms')
                if F_dv is None:
                    F_dv = result.get('v_kms')

                EW_prod = None
                if F_norm_smooth is not None and F_dv is not None:
                    Fn = np.asarray(F_norm_smooth, dtype=float)
                    dv = np.asarray(F_dv, dtype=float)
                    if Fn.size > 1 and dv.size > 1 and Fn.size == dv.size:
                        # EW in AA: convert from velocity to wavelength via
                        # λ_Halpha = 6562.81 AA, dλ = λ × dv/c
                        ddv = float(np.median(np.diff(dv)))
                        dlam = 6562.81 * ddv / 2.998e5
                        EW_prod = -float(np.sum(Fn - 1.0)) * dlam

                prod_Ha_info = {
                    'L_line':  float(result.get('L_line', np.nan)),
                    'peak_F':  float(result.get('red_peak_F', np.nan)),
                    'peak_dv': float(result.get('red_peak_dv', np.nan)),
                    'EW':      EW_prod if EW_prod is not None else np.nan,
                }
                if verbose and EW_prod is not None:
                    print(f"[phase5] prod_Ha cross-check: "
                          f"L_line = {prod_Ha_info['L_line']:.3e} erg/s, "
                          f"EW = {prod_Ha_info['EW']:+.2f} Å, "
                          f"peak F = {prod_Ha_info['peak_F']:.2f} "
                          f"@ {prod_Ha_info['peak_dv']:+.0f} km/s")
                elif verbose:
                    print(f"[phase5] WARNING: could not compute prod_Ha EW "
                          f"(F_norm or dv missing from result dict). "
                          f"Phase 5b empirical correction will be SKIPPED.")
            except Exception as _exc:
                if verbose:
                    print(f"[phase5] prod_Ha_info build failed: {_exc}; "
                          f"Phase 5b empirical correction will be SKIPPED.")
                prod_Ha_info = None

            he_spectra = run_phase5_for_state(
                state, snap_for_phase5,
                n_packets=he_lines_n_packets,
                use_existing_peel_kernel=he_lines_use_existing_kernel,
                calibration=he_lines_calibration,
                out_prefix=out_prefix,
                production_halpha=prod_Ha_info,
                profile_method=line_profile_method,
                lock=line_profile_lock,
                saturated_rt=saturated_rt,
                he_budget=he_budget,
                metal_lines=metal_lines,
                metal_cloudy=metal_cloudy,
                narrow_csm=narrow_csm,
                verbose=True)
            result['he_spectra_summary'] = {
                ln: {'L_line': float(sp['L_line']),
                     'EW': float(sp['EW']),
                     'tau_med': float(sp['tau_med']),
                     'lambda_rest': float(sp['lambda_rest'])}
                for ln, sp in he_spectra.items()
                if isinstance(sp, dict)
            }
            # Phase 6 — regime diagnostics. Build the per-line trust/regime
            # table from the Phase 5 output and append the production Hα row
            # (which used full RT-NLTE iteration, hence a different grade
            # than the empirical-R-corrected lines).
            try:
                import regime_diagnostics as _regdiag
                _rows = _regdiag.build_snapshot_table(he_spectra, snap)
                # Add the production Hα as a separate row tagged 'Halpha_prod'
                # so the diagnostic shows that the RT-NLTE Hα is grade-A
                # while the Phase-5 Hα (empirical R-anchor) is grade-B.
                if isinstance(prod_Ha_info, dict):
                    # P0 #2: gate on H abundance — in H-free models the
                    # production Hα is numerical noise (grade 'N'), not grade-A.
                    try:
                        import continuum_compgen as _cg
                        _xh_prod = _cg.mean_X_H(snap)
                    except Exception:
                        _xh_prod = None
                    _prod_Ha_row = _regdiag.classify_line(
                        'Halpha',
                        tau_med=float(he_spectra.get('Halpha', {}).get('tau_med', np.nan)),
                        tau_max=None, beta_med=None,
                        x_elem=_xh_prod,
                        L_line=float(prod_Ha_info.get('L_line', np.nan)))
                    if _prod_Ha_row.get('grade') != 'N':
                        _prod_Ha_row['grade'] = 'A'   # full RT-NLTE
                        _prod_Ha_row['rationale'] = (
                            'Full RT-NLTE iteration in production_runner '
                            '(L_line invariant across iters to <0.01%).')
                        _prod_Ha_row['paper_action'] = (
                            'Quote production L_line and EW directly. Grade-A.')
                    _prod_Ha_row['line'] = 'Halpha_prod'
                    _prod_Ha_row['epoch_d'] = snap.get('epoch_d', None)
                    _prod_Ha_row['L_line'] = float(prod_Ha_info.get('L_line', np.nan))
                    _prod_Ha_row['EW'] = float(prod_Ha_info.get('EW', np.nan))
                    _prod_Ha_row['peak_F'] = float(prod_Ha_info.get('peak_F', np.nan))
                    _prod_Ha_row['lambda_rest'] = 6562.81
                    _rows.insert(0, _prod_Ha_row)
                # Save per-snapshot diagnostic
                _diag_path = f"{out_prefix}_regime.txt"
                _regdiag.write_snapshot_diagnostic(
                    _diag_path, _rows, snap,
                    production_halpha=prod_Ha_info)
                # Stash on result for the batch-end cross-epoch summary
                result['regime_rows'] = _rows
                _ng = sum(1 for r in _rows if r['grade'] == 'N')
                print(f"[regime] Saved {_diag_path}  "
                      f"({sum(1 for r in _rows if r['grade']=='A')} grade-A, "
                      f"{sum(1 for r in _rows if r['grade']=='B')} grade-B, "
                      f"{sum(1 for r in _rows if r['grade']=='C')} grade-C, "
                      f"{sum(1 for r in _rows if r['grade']=='R')} grade-R"
                      + (f", {_ng} grade-N" if _ng else "") + ")")
            except Exception as _re:
                print(f"[regime] WARNING: per-snapshot diagnostic failed: {_re}")
                import traceback as _rtb
                _rtb.print_exc()
        except Exception as e:
            print(f"[phase5] FAILED: {e}")
            import traceback
            traceback.print_exc()
            # Non-fatal: continue with the rest of the pipeline.

    # Save outputs
    save_outputs(result, out_prefix)
    # Hydrodynamic structure diagnostic (per-snapshot)
    try:
        plot_hydro_structure(snap, result, out_prefix)
    except Exception as e:
        print(f"  hydro plot failed: {e}")
        import traceback
        traceback.print_exc()
    # Ionization-equilibrium diagnostic (per-snapshot)
    try:
        plot_ionization_structure(snap, result, out_prefix)
    except Exception as e:
        print(f"  ionization plot failed: {e}")
        import traceback
        traceback.print_exc()
    print(f"\nTotal runtime: {runtime_total/60:.1f} min "
          f"({n_iter_used} RT iters, "
          f"{(n_iter_per*n_iter_chunks*n_iter_used + n_per*n_chunks):,} total packets)")
    return result


def save_outputs(result, out_prefix):
    """Save .npz, .txt, .profile.txt, .png for a single result."""
    np.savez(
        f"{out_prefix}.npz",
        lam=result['lam'], dv=result['dv'],
        F_norm=result['F_norm'],
        F_norm_raw=result['F_norm_raw'],
        F_cont_smooth=result['F_cont_smooth'],
        F_vol_smooth=result['F_vol_smooth'],
        F_cont_abs=result['F_cont_abs'],
        F_vol_abs=result['F_vol_abs'],
        F_total_abs=result['F_total_abs'],
        smooth_kms=result['smooth_kms'],
        L_line=result['L_line'],
        history_Lline=np.array(result['history']['L_line']),
        history_delta=np.array(result['history']['delta_max']) if result['history']['delta_max'] else np.array([]),
        F_ref=result['F_ref'] if result['F_ref'] is not None else np.array([]),
    )

    # Clean column-table ASCII output for plotting in external tools.
    # Includes velocity (km/s), wavelength (Å), and three flux variants:
    # F_norm (smoothed total, displayed), F_norm_raw (un-smoothed MC),
    # F_cont_smooth and F_vol_smooth (continuum and volumetric channels).
    profile_path = f"{out_prefix}.profile.txt"
    dv_kms = result['dv']
    lam_AA = result['lam']
    Fn = result['F_norm']
    Fn_raw = result.get('F_norm_raw', Fn)
    Fc = result.get('F_cont_smooth', np.full_like(Fn, np.nan))
    Fv = result.get('F_vol_smooth', np.full_like(Fn, np.nan))
    header = (
        f"# {result['name']}  epoch={result.get('epoch_d', '?')}d  "
        f"L_Halpha={result['L_line']:.3e} erg/s  "
        f"smooth_kms={result['smooth_kms']:.1f}\n"
        f"# Columns:\n"
        f"#   1: Δv [km/s]      velocity offset from Hα (6562.81 Å), positive = redshift\n"
        f"#   2: lambda [Å]     observed wavelength\n"
        f"#   3: F_norm         total flux, smoothed, normalized to F_cont_baseline\n"
        f"#   4: F_norm_raw     total flux, RAW MC (no velocity smoothing)\n"
        f"#   5: F_cont_smooth  continuum-channel flux (continuum photons only)\n"
        f"#   6: F_vol_smooth   volumetric-channel flux (line emission contribution)\n"
        f"# Format: F_norm ≈ F_cont_smooth + F_vol_smooth (within MC noise)\n"
        f"# {'-'*88}"
    )
    np.savetxt(profile_path,
               np.column_stack([dv_kms, lam_AA, Fn, Fn_raw, Fc, Fv]),
               fmt=['%10.2f', '%10.4f', '%12.5e', '%12.5e', '%12.5e', '%12.5e'],
               header=header, comments='')

    with open(f"{out_prefix}.txt", 'w') as f:
        f.write("=" * 78 + "\n")
        f.write(f" {result['name']} (epoch {result['epoch_d']}d) — "
                f"production RT-NLTE Hα profile\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"snapshot: {result['snap_path']}\n")
        f.write(f"runtime:  {result['runtime_total']/60:.1f} min\n")
        f.write(f"RT iterations used: {result['n_iter_used']}\n")
        f.write(f"velocity smoothing: σ = {result['smooth_kms']:.1f} km/s\n\n")
        f.write(f"L_line (converged): {result['L_line']:.3e} erg/s\n\n")
        f.write(f"(metrics from smoothed F_norm)\n")
        f.write(f"Peak  F = {result['red_peak_F']:.3f} at Δv = "
                f"{result['red_peak_dv']:+.0f} km/s\n")
        f.write(f"Trough F = {result['trough_F']:.3f} at Δv = "
                f"{result['trough_dv']:+.0f} km/s\n")
        f.write(f"Global peak F = {result['global_peak_F']:.3f} at Δv = "
                f"{result['global_peak_dv']:+.0f} km/s\n\n")
        if result['sigma_res2'] is not None:
            f.write(f"Σres² vs CMFGEN ref (|Δv|<1500): "
                    f"{result['sigma_res2']:.3f}\n\n")
        if len(result['history']['L_line']) > 1:
            f.write("RT iteration history (L_line, Δn_3_max):\n")
            for i, L in enumerate(result['history']['L_line']):
                d = result['history']['delta_max'][i-1] if i > 0 and \
                    i-1 < len(result['history']['delta_max']) else 0.0
                f.write(f"  iter {i}: L_line = {L:.3e}, "
                        f"Δn_3_max = {d*100:.2f}%\n")
    plot_single(result, f"{out_prefix}.png")

    # Convergence audit (per-zone diagnostic for the RT iteration history)
    write_convergence_audit(result, f"{out_prefix}_convergence_audit.txt")


def write_convergence_audit(result, out_path):
    """Per-iteration audit of which zone oscillated most during RT-NLTE
    iteration, and whether the line-luminosity stayed invariant despite
    the per-zone n_3 oscillation.

    When Δn_3_max blows up (>100%) but L_line is invariant across
    iterations, the oscillation is local to a few dense zones where the
    Hα line is optically thick to itself and small changes in J_bar drive
    large δn_3 — yet the integrated line emission is dominated by other
    zones and is unaffected. That's a numerical artifact, not a physics
    problem.

    When L_line ALSO varies across iterations, the oscillation IS
    affecting the global answer and needs damping increased / max_iter
    raised / different fix.
    """
    hist = result['history']
    L_history = list(hist['L_line'])
    if len(L_history) < 2:
        return

    delta_history = list(hist.get('delta_max', []))
    worst_idx = list(hist.get('worst_zone_idx', []))
    worst_n3 = list(hist.get('worst_zone_n3', []))
    worst_X = list(hist.get('worst_zone_X_HII', []))
    worst_ne = list(hist.get('worst_zone_n_e', []))
    worst_T = list(hist.get('worst_zone_T', []))
    worst_r = list(hist.get('worst_zone_r', []))
    worst_J = list(hist.get('worst_zone_J_bar', []))

    L_min, L_max = min(L_history), max(L_history)
    L_variation = (L_max - L_min) / max(L_min, 1e-30)

    with open(out_path, 'w') as f:
        f.write("=" * 78 + "\n")
        f.write(f" {result['name']} (epoch {result['epoch_d']}d) — "
                f"RT-NLTE convergence audit\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Final L_line: {L_history[-1]:.4e} erg/s\n")
        f.write(f"L_line variation across {len(L_history)} iterations: "
                f"{L_variation*100:.3f}% "
                f"(range {L_min:.4e} – {L_max:.4e})\n\n")

        if L_variation < 0.01:
            f.write("L_line is INVARIANT across iterations (<1%).\n")
            f.write("Any per-zone n_3 oscillation visible below is NUMERICAL,\n")
            f.write("not affecting the global line answer.\n\n")
        elif L_variation < 0.1:
            f.write("L_line varies modestly across iterations (1-10%).\n")
            f.write("Per-zone oscillations may have a small global effect.\n\n")
        else:
            f.write("⚠ L_line varies significantly across iterations (>10%).\n")
            f.write("  Per-zone oscillations ARE affecting the global answer.\n")
            f.write("  Recommend: increase damping (current run used 0.3),\n")
            f.write("  raise max_iter, or investigate the worst zones below.\n\n")

        f.write("Per-iteration history (worst-oscillating zone):\n\n")
        f.write(f"  {'it':>3s}  {'L_line':>11s}  {'Δn3_max':>10s}  "
                f"{'zone':>5s}  {'r [cm]':>10s}  {'X_HII':>7s}  "
                f"{'n_e':>9s}  {'T [K]':>7s}  {'n_3':>10s}  "
                f"{'J_bar':>10s}\n")
        for i in range(len(delta_history)):
            f.write(f"  {i+1:>3d}  {L_history[i+1]:>11.4e}  "
                    f"{delta_history[i]*100:>9.2f}%  "
                    f"{worst_idx[i]:>5d}  {worst_r[i]:>10.3e}  "
                    f"{worst_X[i]:>7.4f}  {worst_ne[i]:>9.3e}  "
                    f"{worst_T[i]:>7.0f}  {worst_n3[i]:>10.3e}  "
                    f"{worst_J[i]:>10.3e}\n")

        # If the same zone is consistently worst, report it
        if worst_idx:
            from collections import Counter
            counts = Counter(worst_idx)
            most_common_zone, freq = counts.most_common(1)[0]
            f.write(f"\nMost-frequently-worst zone: "
                    f"idx={most_common_zone}  ({freq}/{len(worst_idx)} iters)\n")
            if freq > len(worst_idx) // 2:
                f.write("  → A SINGLE zone is dominating the convergence "
                        "diagnostic.\n")
                f.write("  Likely physical cause: line radiative trapping in "
                        "a high-τ dense zone.\n")
                f.write("  If L_line is invariant (above), this is benign.\n")


def plot_single(result, out_path):
    """6-panel profile plot for a single snapshot result."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    dv = result['dv']
    F_norm = result['F_norm']
    F_norm_raw = result.get('F_norm_raw', F_norm)
    F_ref = result['F_ref']
    smooth_kms = result.get('smooth_kms', 0.0)
    show_raw = smooth_kms > 0

    # Panel 1: full profile
    ax = axes[0, 0]
    if show_raw:
        ax.plot(dv, F_norm_raw, '-', color='C2', lw=0.5, alpha=0.3,
                label='raw MC')
    ax.plot(dv, F_norm, 'C2-', lw=1.6,
            label=f'smoothed (σ={smooth_kms:.0f} km/s)' if show_raw
            else 'our pipeline (RT-NLTE)')
    if F_ref is not None:
        ax.plot(dv, F_ref, 'k-', lw=2.0, alpha=0.85, label='CMFGEN ref')
    ax.axhline(1, color='gray', ls=':', lw=0.5)
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    # Auto-detect line extent so late-time IIP profiles (which extend to
    # ±5000+ km/s) aren't cropped. Find where |F-1| > 5% of peak departure
    # via a smoothed envelope, then pad 20% and clamp to ±10000 km/s.
    # Falls back to ±5000 (wider than the legacy ±3000) if the auto-detect
    # finds nothing.
    try:
        from scipy.ndimage import median_filter as _mfilt
        _F_smooth = _mfilt(np.asarray(F_norm), size=11, mode='nearest')
    except ImportError:
        _F_smooth = np.asarray(F_norm)
    _excess = np.abs(_F_smooth - 1.0)
    _peak_excess = float(_excess.max()) if _excess.size else 0.0
    if _peak_excess > 0.05:
        _mask = _excess > max(0.05 * _peak_excess, 0.02)
        if _mask.sum() >= 5:
            _idx = np.where(_mask)[0]
            _v_lo = float(dv[_idx[0]]); _v_hi = float(dv[_idx[-1]])
            _w = _v_hi - _v_lo
            _v_lo -= 0.20 * _w; _v_hi += 0.20 * _w
            _v_lo = max(-10000.0, _v_lo); _v_hi = min(10000.0, _v_hi)
            ax.set_xlim(_v_lo, _v_hi)
        else:
            ax.set_xlim(-5000, 5000)
    else:
        ax.set_xlim(-5000, 5000)
    ax.set_xlabel('Δv [km/s]')
    ax.set_ylabel('F / F_cont')
    ax.legend(fontsize=9)
    title1 = f"(1) Full profile  │  {result['name']}"
    if result['epoch_d'] is not None:
        if result.get('format') == 'stella':
            title1 += f" ({result['epoch_d']:.1f}d post Lbol max)"
        else:
            title1 += f" (epoch {result['epoch_d']:.1f}d)"
    ax.set_title(title1, fontsize=11)
    ax.grid(alpha=0.3)

    # Panel 2: line core zoom
    ax = axes[0, 1]
    if show_raw:
        ax.plot(dv, F_norm_raw, '-', color='C2', lw=0.5, alpha=0.3,
                label='raw MC')
    ax.plot(dv, F_norm, 'C2-', lw=1.6, label='our pipeline (smoothed)')
    if F_ref is not None:
        ax.plot(dv, F_ref, 'k-', lw=2.0, alpha=0.85, label='CMFGEN ref')
    ax.scatter([result['red_peak_dv']], [result['red_peak_F']], color='red',
               s=60, zorder=5,
               label=f"peak F={result['red_peak_F']:.2f} @ {result['red_peak_dv']:+.0f}")
    ax.scatter([result['trough_dv']], [result['trough_F']], color='blue',
               s=60, zorder=5,
               label=f"trough F={result['trough_F']:.2f} @ {result['trough_dv']:+.0f}")
    ax.axhline(1, color='gray', ls=':', lw=0.5)
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlim(-500, 500)
    ax.set_xlabel('Δv [km/s]')
    ax.set_ylabel('F / F_cont')
    ax.legend(fontsize=9, loc='upper left')
    title2 = '(2) Line core zoom'
    if result['sigma_res2'] is not None:
        title2 += f"  │  Σres²={result['sigma_res2']:.2f}"
    ax.set_title(title2, fontsize=11)
    ax.grid(alpha=0.3)

    # Panel 3: cont/vol decomposition (smoothed for clean visualization)
    ax = axes[0, 2]
    F_cont_norm = result.get('F_cont_smooth',
                              result['F_cont_abs'] / result['F_baseline'])
    F_vol_norm = result.get('F_vol_smooth',
                             result['F_vol_abs'] / result['F_baseline'])
    ax.plot(dv, F_cont_norm, 'C0-', lw=1.2, label='cont channel')
    ax.plot(dv, 1.0 + F_vol_norm, 'C3-', lw=1.2, label='1 + vol channel')
    ax.plot(dv, F_norm, 'C2-', lw=1.6, label='total')
    if F_ref is not None:
        ax.plot(dv, F_ref, 'k--', lw=1.0, alpha=0.6, label='CMFGEN')
    ax.axhline(1, color='gray', ls=':', lw=0.5)
    ax.set_xlim(-500, 500)
    ax.set_xlabel('Δv [km/s]')
    ax.set_ylabel('F / F_cont')
    ax.legend(fontsize=8)
    ax.set_title('(3) cont + vol decomposition', fontsize=11)
    ax.grid(alpha=0.3)

    # Panel 4: log-scale wing detail
    ax = axes[1, 0]
    ax.semilogy(dv, np.maximum(F_norm, 1e-2), 'C2-', lw=1.5, label='our pipeline')
    if F_ref is not None:
        ax.semilogy(dv, np.maximum(F_ref, 1e-2), 'k-', lw=2.0, alpha=0.85,
                    label='CMFGEN ref')
    # Match panel 1's auto-detected window (line feature extent ±20% pad).
    ax.set_xlim(axes[0, 0].get_xlim())
    ax.set_ylim(0.5, 10)
    ax.set_xlabel('Δv [km/s]')
    ax.set_ylabel('F / F_cont (log)')
    ax.legend(fontsize=9)
    ax.set_title('(4) Log-scale wing detail', fontsize=11)
    ax.grid(alpha=0.3, which='both')

    # Panel 5: RT iteration history
    ax = axes[1, 1]
    if len(result['history']['L_line']) > 1:
        iters = np.arange(len(result['history']['L_line']))
        ax2 = ax.twinx()
        ax.plot(iters, result['history']['L_line'], 'C0o-', lw=1.5,
                ms=8, label='L_line')
        ax.set_ylabel('L_line [erg/s]', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        if result['history']['delta_max']:
            ax2.semilogy(iters[1:], result['history']['delta_max'],
                         'C3s-', lw=1.5, ms=8, label='Δn_3 max')
            ax2.set_ylabel('max relative Δn_3', color='C3')
            ax2.tick_params(axis='y', labelcolor='C3')
            ax2.axhline(0.03, color='C3', ls='--', lw=0.5, alpha=0.5)
        ax.set_xlabel('RT iteration')
        ax.set_title(f"(5) RT-NLTE convergence "
                     f"({result['n_iter_used']} iters used)", fontsize=11)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'RT iteration disabled', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('(5) RT iteration history', fontsize=11)

    # Panel 6: residuals
    ax = axes[1, 2]
    if F_ref is not None:
        resid = F_norm - F_ref
        ax.plot(dv, resid, 'C2-', lw=1.2, label='our − CMFGEN')
        mask_window = np.abs(dv) < 1500
        ax.fill_between(dv[mask_window], 0, resid[mask_window],
                        color='C2', alpha=0.2)
        ax.axhline(0, color='gray', ls=':', lw=0.5)
        ax.set_xlim(-2000, 2000)
        ax.set_xlabel('Δv [km/s]')
        ax.set_ylabel('residual')
        ax.legend(fontsize=9)
        ax.set_title('(6) Residuals vs CMFGEN', fontsize=11)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No CMFGEN reference', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('(6) Residuals vs CMFGEN', fontsize=11)

    # Use global peak (not the red_peak_dv from the ±500 km/s core window),
    # which for late-time IIP-like profiles is at v ~ +1500-2000 km/s
    # (the broad ejecta emission) rather than near line center.
    _gp_F = result.get('global_peak_F', result.get('red_peak_F', float('nan')))
    _gp_dv = result.get('global_peak_dv', result.get('red_peak_dv', float('nan')))
    fig.suptitle(
        f"{result['name']} — RT-NLTE Hα production  │  "
        f"L_line = {result['L_line']:.2e} erg/s,  "
        f"peak F = {_gp_F:.2f} @ {_gp_dv:+.0f} km/s",
        fontsize=13, y=0.998)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ---------- Hydro structure diagnostic ----------

def plot_hydro_structure(snap, result, out_prefix):
    """Per-snapshot hydrodynamic structure plot.

    Shows the input atmosphere structure used by the pipeline, with the
    photosphere/truncation boundary marked. For STELLA snapshots that have
    been truncated, the inner boundary of the plot IS R_phot (no zones
    inside). For HERACLES snapshots, R_phot is the inner zone by convention.

    Six panels:
      (1) ρ(r), velocity v(r) — kinematics + density
      (2) T_gas(r), T_rad(r) if available — temperature structure
      (3) n_e(r), n_HI(r) computed from Saha — ionization
      (4) X_H(r), X_He(r), X_metals(r) — composition
      (5) τ_es(r) integrated outward — Thomson optical depth
      (6) Per-zone Hα emissivity n_e × n_p × α_eff × dV — emission location

    For STELLA, R_phot_inner is added as a dashed vertical line marking
    the truncation boundary (zones inside this radius are hidden behind
    the opaque BB inner boundary).

    Saves as {out_prefix}_hydro.png.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = np.asarray(snap['r'])
    v_kms = np.asarray(snap['v']) / 1e5
    rho = np.asarray(snap['rho'])
    T = np.asarray(snap['T'])
    n_e = np.asarray(snap['n_e'])
    tau_es = np.asarray(snap['tau_es'])

    r15 = r / 1e15
    R_phot_inner = snap.get('R_phot_inner', r[0])
    R_phot_inner_15 = R_phot_inner / 1e15
    n_zones_full = snap.get('n_zones_full', len(r))

    # Composition: prefer the snap_arrays format (X_H/X_He/X_metals per zone)
    X_H = np.asarray(snap.get('X_H',
                                snap.get('composition', {}).get('X_H',
                                                                  np.full_like(r, np.nan))))
    X_He = np.asarray(snap.get('X_He',
                                 snap.get('composition', {}).get('X_He',
                                                                  np.full_like(r, np.nan))))
    X_metals_def = np.maximum(0.0, 1.0 - np.nan_to_num(X_H, nan=0.0)
                                          - np.nan_to_num(X_He, nan=0.0))
    X_metals = np.asarray(snap.get('X_metals',
                                     snap.get('composition', {}).get('X_metals',
                                                                       X_metals_def)))

    # n_p, n_HI from Saha (cheap recompute for diagnostic; uses snap T, n_e)
    MH = 1.6735e-24
    KB = 1.38065e-16
    HPL = 6.626e-27
    ME = 9.109e-28
    chi = 2.179e-11  # ionization energy of H, erg
    pre = (2.0 * np.pi * ME * KB * T / (HPL * HPL))**1.5
    K = pre * np.exp(-chi / (KB * T)) / np.maximum(n_e, 1e-30)
    f_HI = 1.0 / (1.0 + K)
    f_HI = np.clip(f_HI, 0.0, 1.0)
    n_H_total = np.where(np.isfinite(X_H), X_H, 0.737) * rho / MH
    n_HI = f_HI * n_H_total
    n_p = (1.0 - f_HI) * n_H_total

    # Hα emissivity per zone (case-B)
    alpha_eff = 1.17e-13 * (T / 1.0e4)**(-0.94)  # cm³/s
    nu_Ha = 2.998e10 / (6562.81e-8)
    E_Ha = HPL * nu_Ha
    dr = np.empty_like(r)
    dr[:-1] = np.diff(r)
    dr[-1] = dr[-2]
    dV = 4.0 * np.pi * r * r * dr
    L_zone = n_e * n_p * alpha_eff * E_Ha * dV  # erg/s per zone

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # (1) density and velocity
    ax = axes[0, 0]
    ax.semilogy(r15, rho, 'C0-', lw=1.4, label='ρ')
    ax.set_xlabel('r [10¹⁵ cm]')
    ax.set_ylabel('ρ [g/cm³]', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax2 = ax.twinx()
    ax2.plot(r15, v_kms, 'C3-', lw=1.4, label='v')
    ax2.set_ylabel('v [km/s]', color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')
    ax.axvline(R_phot_inner_15, color='k', ls='--', lw=1.2, alpha=0.7,
                label=f'R_phot = {R_phot_inner:.2e} cm')
    ax.set_title(f"(1) Density and velocity vs r\n"
                 f"max v = {v_kms.max():.0f} km/s, "
                 f"min v = {v_kms.min():.0f} km/s")
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # (2) temperature
    ax = axes[0, 1]
    ax.semilogy(r15, T, 'C1-', lw=1.4, label='T_gas')
    if 'T_rad' in snap:
        T_rad = np.asarray(snap['T_rad'])
        ax.semilogy(r15, T_rad, 'C2--', lw=1.2, alpha=0.7, label='T_rad')
    ax.axvline(R_phot_inner_15, color='k', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel('r [10¹⁵ cm]')
    ax.set_ylabel('T [K]')
    ax.set_title(f"(2) Temperature structure\n"
                 f"T_phot ≈ {snap.get('T_phot_inner', T[0]):.0f} K, "
                 f"T range {T.min():.0f}-{T.max():.0f} K")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # (3) electron and neutral H densities. If photoionization equilibrium
    # was run, prefer its X_HII over a re-derived Saha value — this matches
    # the n_e/n_HI actually used downstream by NLTE and MC. Otherwise fall
    # back to Saha at snap T,n_e.
    ax = axes[0, 2]
    photoeq_used = bool(snap.get('photoionized', False))
    if photoeq_used:
        X_HII_pi = np.asarray(snap['X_HII'])
        n_HI_diag = (1.0 - X_HII_pi) * n_H_total
        n_p_diag  = X_HII_pi * n_H_total
        ion_label_HI = 'n_HI (photoeq)'
        ion_label_HII = 'n_HII (photoeq)'
        diag_X_HII_mean = X_HII_pi.mean()
        diag_note = '(photoeq, downstream-consistent)'
    else:
        n_HI_diag = n_HI
        n_p_diag  = n_p
        ion_label_HI = 'n_HI (Saha)'
        ion_label_HII = 'n_p (Saha)'
        diag_X_HII_mean = 1.0 - f_HI.mean()
        diag_note = '(Saha, snap T,n_e)'
    ax.semilogy(r15, n_e, 'C4-', lw=1.4, label='n_e')
    ax.semilogy(r15, np.maximum(n_HI_diag, 1e-3), 'C5-', lw=1.2, label=ion_label_HI)
    ax.semilogy(r15, np.maximum(n_p_diag, 1e-3), 'C6-', lw=1.2, alpha=0.7,
                 label=ion_label_HII)
    ax.axvline(R_phot_inner_15, color='k', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel('r [10¹⁵ cm]')
    ax.set_ylabel('number density [cm⁻³]')
    ax.set_title(f"(3) Electron and H densities  {diag_note}\n"
                 f"⟨X_HII⟩ = {diag_X_HII_mean:.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # (4) composition
    ax = axes[1, 0]
    if np.all(np.isfinite(X_H)):
        ax.plot(r15, X_H, 'C0-', lw=1.4, label='X_H')
    if np.all(np.isfinite(X_He)):
        ax.plot(r15, X_He, 'C2-', lw=1.4, label='X_He')
    ax.plot(r15, X_metals, 'C3-', lw=1.4, label='X_metals')
    ax.axvline(R_phot_inner_15, color='k', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel('r [10¹⁵ cm]')
    ax.set_ylabel('mass fraction')
    fmt_str = result.get('format', snap.get('format', '?'))
    ax.set_title(f"(4) Composition profile  ({fmt_str})\n"
                 f"⟨X_H⟩(emit) = {snap.get('X_H_emit', np.nan):.3f}")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # (5) Thomson optical depth
    ax = axes[1, 1]
    ax.semilogy(r15, np.maximum(tau_es, 1e-4), 'C7-', lw=1.4)
    ax.axhline(2.0/3.0, color='r', ls=':', lw=1.0, alpha=0.7,
                label='τ_es = 2/3 (photosphere)')
    ax.axvline(R_phot_inner_15, color='k', ls='--', lw=1.2, alpha=0.7,
                label=f'R_phot')
    ax.set_xlabel('r [10¹⁵ cm]')
    ax.set_ylabel('τ_es from outside')
    ax.set_title(f"(5) Electron-scattering optical depth\n"
                 f"τ_es total = {tau_es.max():.2f} (above photosphere)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # (6) Hα emissivity per zone, with truncation note
    ax = axes[1, 2]
    ax.semilogy(r15, np.maximum(L_zone, 1e-1), 'C8-', lw=1.4)
    # Color zones by velocity range
    for v_lo, v_hi, color, lab in [(0, 500, 'green', '|v|<500 (CSM)'),
                                       (500, 2000, 'orange', '500-2000 (intermediate)'),
                                       (2000, 1e6, 'red', '>2000 (fast ejecta)')]:
        m = (v_kms >= v_lo) & (v_kms < v_hi)
        if m.any():
            ax.scatter(r15[m], np.maximum(L_zone[m], 1e-1), s=14, c=color,
                        label=f'{lab}: {m.sum()}z', zorder=5)
    ax.axvline(R_phot_inner_15, color='k', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel('r [10¹⁵ cm]')
    ax.set_ylabel('L_Hα per zone [erg/s]')
    ax.set_title(f"(6) Per-zone Hα emissivity (case-B)\n"
                 f"L_Hα total = {L_zone.sum():.2e} erg/s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    epoch = snap.get('epoch_d', '?')
    fmt = snap.get('format', '?')
    n_show = len(r)
    if fmt == 'stella' and n_zones_full > n_show:
        n_note = f" ({n_show} zones above photosphere of {n_zones_full} total)"
    else:
        n_note = f" ({n_show} zones)"
    fig.suptitle(
        f"{result['name']}  ─  hydrodynamic structure  ({fmt})  ─  "
        f"epoch {epoch}d{n_note}",
        fontsize=13, y=0.998)
    fig.tight_layout()
    out_path = f"{out_prefix}_hydro.png"
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_ionization_structure(snap, result, out_prefix):
    """Per-snapshot ionization-equilibrium diagnostic plot.

    Four panels with R_phot overplotted as a vertical dashed line in each:
      (1) X_HII(r): photoeq vs STELLA's implied value
      (2) Γ_HI(r): total + BB + brems components (shock-Xray contribution)
      (3) τ_LyC(r): cumulative H bf optical depth from R_phot outward at the
          Lyman edge (the attenuation seen by ionizing photons traveling out)
      (4) τ_es(r): cumulative Thomson optical depth from outside in, photoeq
          vs STELLA — the quantity that smears the line profile.

    Skipped silently if photoionization equilibrium was not run on this snap
    (i.e. snap['photoionized'] is False or missing).

    Saves as {out_prefix}_ionization.png.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not snap.get('photoionized', False):
        print(f"  ionization plot skipped (no photoeq run on this snap)")
        return

    r = np.asarray(snap['r'])
    rho = np.asarray(snap['rho'])
    X_H = np.asarray(snap.get('X_H', np.full_like(r, 0.737)))
    MH = 1.6735575e-24
    SIGMA_H0 = 6.3e-18

    R_phot = float(snap.get('R_phot_inner', r[0]))
    R_phot_15 = R_phot / 1e15
    r15 = r / 1e15

    n_H_total = X_H * rho / MH

    X_HII        = np.asarray(snap['X_HII'])
    X_HII_stella = np.asarray(snap.get('X_HII_stella', X_HII))
    n_HI         = (1.0 - X_HII) * n_H_total
    Gamma_total  = np.asarray(snap.get('Gamma_HI', np.zeros_like(r)))
    Gamma_BB     = np.asarray(snap.get('Gamma_HI_BB', np.zeros_like(r)))
    Gamma_brems  = np.asarray(snap.get('Gamma_HI_brems', np.zeros_like(r)))
    tau_es       = np.asarray(snap['tau_es'])
    tau_es_stella = np.asarray(snap.get('tau_es_stella', tau_es))

    # τ_LyC(r): cumulative neutral H column from R_phot to r, in units of σ_H0
    dr = np.empty_like(r); dr[:-1] = np.diff(r); dr[-1] = dr[-2]
    dtau_LyC = SIGMA_H0 * n_HI * dr
    tau_LyC = np.concatenate([[0.0], np.cumsum(dtau_LyC[:-1])])

    pi_params = snap.get('photoionization_params', {})
    T_source = pi_params.get('T_source', np.nan)
    use_xray = pi_params.get('include_shock_xray', False)
    sp = pi_params.get('shock_params', None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # ---- (1) X_HII(r) photoeq vs STELLA ----
    ax = axes[0, 0]
    ax.semilogy(r15, np.clip(X_HII, 1e-6, 1.0), 'C0-', lw=1.6, label='X_HII (photoeq)')
    ax.semilogy(r15, np.clip(X_HII_stella, 1e-6, 1.0), 'C7--', lw=1.2, alpha=0.7,
                 label='X_HII (STELLA)')
    ax.axvline(R_phot_15, color='k', ls='--', lw=1.2, alpha=0.7,
                label=f'R_phot = {R_phot:.2e} cm')
    ax.set_xlabel('r [10¹⁵ cm]')
    ax.set_ylabel('X_HII = n_HII / n_H')
    ax.set_ylim(1e-6, 2.0)
    ax.set_title(f"(1) Ionization fraction\n"
                 f"⟨X_HII⟩ photoeq = {X_HII.mean():.3f}, "
                 f"STELLA = {X_HII_stella.mean():.3f}")
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, which='both')

    # ---- (2) Γ_HI(r): total / BB / brems ----
    ax = axes[0, 1]
    # Floor for log scale
    G_floor = 1e-20
    ax.semilogy(r15, np.maximum(Gamma_total, G_floor), 'C2-', lw=1.6, label='Γ total')
    ax.semilogy(r15, np.maximum(Gamma_BB, G_floor), 'C0--', lw=1.2,
                 alpha=0.85, label='Γ BB')
    if use_xray and np.any(Gamma_brems > 0):
        ax.semilogy(r15, np.maximum(Gamma_brems, G_floor), 'C3-.', lw=1.2,
                     alpha=0.85, label='Γ brems (shock X-ray)')
    ax.axvline(R_phot_15, color='k', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel('r [10¹⁵ cm]')
    ax.set_ylabel('Γ_HI [s⁻¹ per atom]')
    title_lines = [f"(2) H photoionization rate per atom"]
    title_lines.append(f"T_source = {T_source:.0f} K")
    if use_xray and sp is not None:
        kT_keV = 1.380649e-16 * sp['T_shock'] / 1.602e-12 / 1000
        title_lines.append(f"shock brems: T_shock = {sp['T_shock']:.2e} K "
                            f"(kT = {kT_keV:.1f} keV), L_X = {sp['L_X_brems']:.2e}")
    ax.set_title('\n'.join(title_lines))
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, which='both')

    # ---- (3) τ_LyC(r) from photosphere outward ----
    ax = axes[1, 0]
    ax.semilogy(r15, np.maximum(tau_LyC, 1e-3), 'C4-', lw=1.6, label='τ_LyC (cumulative)')
    ax.axhline(1.0, color='r', ls=':', lw=1.0, alpha=0.7, label='τ = 1')
    ax.axvline(R_phot_15, color='k', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel('r [10¹⁵ cm]')
    ax.set_ylabel('τ_LyC (from R_phot outward)')
    ax.set_title(f"(3) Lyman-continuum optical depth\n"
                 f"Cumulative n_HI × σ_H0 column from photosphere → r. "
                 f"τ_LyC(outer) = {tau_LyC[-1]:.2e}")
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, which='both')

    # ---- (4) τ_es(r): photoeq vs STELLA ----
    ax = axes[1, 1]
    ax.semilogy(r15, np.maximum(tau_es, 1e-4), 'C0-', lw=1.6, label='τ_es (photoeq)')
    ax.semilogy(r15, np.maximum(tau_es_stella, 1e-4), 'C7--', lw=1.2,
                 alpha=0.8, label='τ_es (STELLA)')
    ax.axhline(2.0/3.0, color='r', ls=':', lw=1.0, alpha=0.7, label='τ_es = 2/3')
    ax.axvline(R_phot_15, color='k', ls='--', lw=1.2, alpha=0.7)
    ax.set_xlabel('r [10¹⁵ cm]')
    ax.set_ylabel('τ_es (from outside)')
    ax.set_title(f"(4) Thomson optical depth\n"
                 f"τ_es total: STELLA={tau_es_stella.max():.3f}, "
                 f"photoeq={tau_es.max():.3f}  "
                 f"(ratio {tau_es.max()/max(tau_es_stella.max(),1e-30):.2f}×)")
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, which='both')

    epoch = snap.get('epoch_d', '?')
    fig.suptitle(
        f"{result['name']}  ─  ionization equilibrium diagnostic  ─  "
        f"epoch {epoch}d",
        fontsize=13, y=0.998)
    fig.tight_layout()
    out_path = f"{out_prefix}_ionization.png"
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------- Batch processing ----------

def _file_md5(path, _cache={}):
    """Content hash of a file, memoized on (abspath, mtime, size)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), st.st_mtime, st.st_size)
    if key in _cache:
        return _cache[key]
    h = hashlib.md5()
    try:
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
    except OSError:
        return None
    digest = h.hexdigest()
    _cache[key] = digest
    return digest


def detect_shared_snapshots(snap_paths, search_root=None):
    """Identify batch snapshots that are byte-identical to the same-named
    snapshot in a *sibling* model directory.

    Background — the 'shared late-epoch snapshot' bug: when a model-specific
    STELLA snapshot is missing for a late epoch, one common snapshot was staged
    by copying it verbatim into several model directories. A genuinely
    model-local snapshot is unique to its directory; a shared copy is
    byte-for-byte identical to a sibling's file of the same name, and processing
    it reproduces identical per-line output rows across models (including the
    no-CSM controls). Such a copy is not a reliable model-specific state and
    should be skipped (the cut coincides with the physics-motivated
    continuum-collapse truncation).

    Detection is content-based (md5) rather than path-based because the shared
    files are physically present in each directory and parse normally — there is
    no model-local-vs-shared distinction at the path level, and an anomalous
    zone count only flags some of them (e.g. day160) but not others (day150).

    Returns a list of dicts {'path', 'basename', 'sibling'} — one per snapshot
    that matches a sibling-directory file of identical content. Empty list when
    there are no sibling directories (e.g. a standalone run dir), so this is a
    no-op outside the model-grid layout.
    """
    if not snap_paths:
        return []
    # All batch snapshots live in one model directory.
    model_dir = os.path.dirname(os.path.abspath(snap_paths[0]))
    parent = search_root or os.path.dirname(model_dir)
    if not parent or not os.path.isdir(parent):
        return []
    # Sibling model directories = immediate subdirs of the parent, minus ours.
    siblings = []
    for name in sorted(os.listdir(parent)):
        cand = os.path.join(parent, name)
        if os.path.isdir(cand) and os.path.abspath(cand) != model_dir:
            siblings.append(cand)
    if not siblings:
        return []
    shared = []
    for p in snap_paths:
        base = os.path.basename(p)
        my_hash = _file_md5(p)
        if my_hash is None:
            continue
        for sib in siblings:
            sib_file = os.path.join(sib, base)
            if os.path.isfile(sib_file) and _file_md5(sib_file) == my_hash:
                shared.append({'path': p, 'basename': base,
                               'sibling': os.path.basename(sib)})
                break
    return shared


def process_batch(snap_paths, args):
    """Process all snapshots and generate batch summary + movie."""
    print(f"\n{'#'*78}")
    print(f" BATCH MODE: processing {len(snap_paths)} snapshots")
    print(f"{'#'*78}")

    results = []
    t_start = time.time()
    for i, p in enumerate(snap_paths):
        print(f"\n[{i+1}/{len(snap_paths)}] Processing {p}...")
        try:
            r = process_snapshot(
                p,
                n_per=args.n_per, n_chunks=args.n_chunks,
                n_iter_per=args.iter_n, n_iter_chunks=1,
                max_iter=args.max_iter, tol=args.tol, damping=args.damping,
                nbins=args.nbins, smooth_kms=args.smooth_kms,
                band_AA=(args.band_lo, args.band_hi),
                source_padding_AA=args.source_padding,
                calibration=args.calibration,
                fmt=args.format,
                do_rt_iter=not args.no_iter,
                line_profile_method=args.line_profile_method,
                line_profile_lock=args.line_profile_method_lock,
                ref_path=None,  # batch mode: no per-snap CMFGEN ref
                extend_wind=args.extend_wind,
                wind_r_max_factor=args.wind_r_max_factor,
                wind_n_zones=args.wind_n_zones,
                wind_T_photoionized=args.wind_T_photoionized,
                wind_rho_index=args.wind_rho_index,
                wind_density_boost=args.wind_density_boost,
                photoionize=args.photoionize,
                photoionize_T_source=args.photoionize_T_source,
                photoionize_T_eq_floor=args.photoionize_T_eq_floor,
                include_shock_xray=args.include_shock_xray,
                eps_Lya_destruction=args.eps_lya_destruction,
                two_photon_decay=args.two_photon_decay,
                line_redistribution=args.line_redistribution,
                photosphere_mode=args.photosphere_mode,
                photosphere_lam_ref_AA=args.photosphere_lam_ref,
                compute_he1_nlte=args.he1_nlte,
                he1_eps_resonance=args.he1_eps_resonance,
                he1_two_photon_decay=args.he1_two_photon_decay,
                he1_ionization_mode=args.he1_ionization_mode,
                compute_he2_nlte=args.he2_nlte,
                he2_x_heiii_mode=args.he2_x_heiii_mode,
                he2_x_heiii_scalar=args.he2_x_heiii_scalar,
                he2_x_heiii_fraction=args.he2_x_heiii_fraction,
                compute_he_lines=args.he_lines,
                he_lines_n_packets=args.he_lines_n_packets,
                he_lines_calibration=args.he_lines_calibration,
                he_lines_use_existing_kernel=not args.he_lines_reference_mc,
                saturated_rt=args.saturated_rt,
                he_budget=args.he_budget,
                metal_lines=args.metal_lines,
                metal_cloudy=args.metal_cloudy,
                narrow_csm=getattr(args, 'narrow_csm', False),
                verbose=True,
            )
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
    t_total = time.time() - t_start
    print(f"\n{'#'*78}")
    print(f" Batch done in {t_total/60:.1f} min ({t_total/3600:.2f} hr)")
    print(f"{'#'*78}")

    # Sort by epoch
    results.sort(key=lambda r: r['epoch_d'] if r['epoch_d'] is not None else 0)

    # Grid plot
    plot_batch_grid(results, 'batch_grid.png')

    # Movie
    try:
        make_movie(results, 'batch_movie.mp4')
    except Exception as e:
        print(f"Movie generation failed: {e}")

    # Summary text
    with open('batch_summary.txt', 'w') as f:
        f.write("=" * 78 + "\n")
        f.write(" BATCH SUMMARY: production RT-NLTE Hα profiles\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"{'name':>8s}  {'epoch':>6s}  {'iters':>6s}  "
                f"{'L_line':>10s}  {'peak_F':>8s}  {'peak_dv':>8s}  "
                f"{'core_F':>7s}  {'trough_F':>9s}  {'trough_dv':>9s}  "
                f"{'runtime':>8s}\n")
        f.write("# 'peak_F'/'peak_dv' = GLOBAL extremum across full v range\n"
                "# 'core_F' = local max in line-core ±500 km/s window only\n"
                "# Trough is the minimum F in line-core ±500 km/s window\n")
        for r in results:
            ep = r['epoch_d'] if r['epoch_d'] else 0
            # Prefer global_peak_* (full v-range) over red_peak_* (line core).
            # global_peak_* was added by the suptitle fix; fall back to
            # red_peak_* for older results.
            peak_F = r.get('global_peak_F', r.get('red_peak_F', float('nan')))
            peak_dv = r.get('global_peak_dv', r.get('red_peak_dv', 0))
            core_F = r.get('red_peak_F', float('nan'))
            f.write(f"{r['name']:>8s}  {ep:>6.1f}  {r['n_iter_used']:>6d}  "
                    f"{r['L_line']:>10.3e}  {peak_F:>8.2f}  "
                    f"{peak_dv:>+8.0f}  {core_F:>7.2f}  "
                    f"{r['trough_F']:>9.3f}  "
                    f"{r['trough_dv']:>+9.0f}  "
                    f"{r['runtime_total']/60:>6.1f}min\n")
        f.write(f"\nTotal batch runtime: {t_total/60:.1f} min\n")
    print("Saved batch_summary.txt")

    # Machine-readable metrics CSV (one row per snapshot)
    write_batch_metrics_csv(results, 'batch_metrics.csv')

    # ----- Phase 5 multi-line evolution movie (if --he-lines was on) -----
    # Triggered when the batch ran with --he-lines so per-snapshot
    # {prefix}_lines.npz files exist. Auto-detects the prefix pattern and
    # writes an MP4 (or GIF) animating all 13 lines across the 30 epochs.
    if getattr(args, 'he_lines', False):
        try:
            import glob
            pattern = 'prod_*_lines.npz'
            npz_files = sorted(glob.glob(pattern))
            if not npz_files:
                # Try the legacy _he.npz pattern (pre-H-extension)
                pattern = 'prod_*_he.npz'
                npz_files = sorted(glob.glob(pattern))
            if npz_files:
                movie_out = getattr(args, 'phase5_movie_out',
                                      'batch_lines_evolution.mp4')
                fps = getattr(args, 'phase5_movie_fps', 3)
                from make_phase5_movie import main as movie_main
                # Always animate ALL lines; ALSO emit species-specific movies so
                # the C/O/Ne metal evolution (and the He evolution) each get their
                # own clip instead of being buried among the 19 H/He/metal panels.
                # The metal movie is produced whenever metal lines were computed.
                _movies = [('all', movie_out)]
                if getattr(args, 'metal_lines', False):
                    _movies.append(('metal', 'batch_metal_evolution.mp4'))
                    _movies.append(('he', 'batch_he_evolution.mp4'))
                for _sp, _out in _movies:
                    try:
                        print(f"\n[phase5] Building {_sp}-line evolution movie "
                              f"({len(npz_files)} frames → {_out} at {fps} fps)...")
                        movie_main([pattern, '--out', _out, '--fps', str(fps),
                                    '--species', _sp])
                    except Exception as _me:
                        print(f"[phase5] WARNING: {_sp} movie failed: {_me}")
                # static evolution GRIDS (rows = lines, columns = epochs) — the
                # per-snapshot {prefix}_metal_lines.png give the per-epoch view;
                # these tile the whole series into one overview PNG.
                if getattr(args, 'metal_lines', False):
                    for _sp, _gout in [('metal', 'batch_metal_grid.png'),
                                       ('he', 'batch_he_grid.png')]:
                        try:
                            print(f"[phase5] Building {_sp}-line evolution grid "
                                  f"→ {_gout}...")
                            movie_main([pattern, '--out', _gout,
                                        '--species', _sp, '--grid'])
                        except Exception as _ge:
                            print(f"[phase5] WARNING: {_sp} grid failed: {_ge}")
            else:
                print(f"\n[phase5] No Phase 5 NPZ files found; skipping movie.")
        except Exception as e:
            print(f"\n[phase5] WARNING: movie generation failed: {e}")
            import traceback
            traceback.print_exc()

    # Phase 6: cross-epoch regime evolution summary
    # Aggregates the per-snapshot regime tables (saved as
    # {prefix}_regime.txt during each snapshot) into a single
    # batch-level overview suitable for paper methods sections.
    try:
        import regime_diagnostics as _regdiag
        _all_regime_results = []
        for r in results:
            if r is None:
                continue
            rows = r.get('regime_rows', None)
            if rows is None:
                continue
            _all_regime_results.append({
                'epoch_d': r.get('epoch_d', None),
                'rows': rows,
            })
        if _all_regime_results:
            _regdiag.write_batch_regime_summary(
                _all_regime_results, 'batch_regime_summary.txt')
            print(f"\n[regime] Saved batch_regime_summary.txt "
                  f"({len(_all_regime_results)} epochs)")
        else:
            print(f"\n[regime] No per-snapshot regime data available; "
                  f"skipping batch summary.")
    except Exception as _re:
        print(f"\n[regime] WARNING: batch summary failed: {_re}")
        import traceback as _rtb
        _rtb.print_exc()


def write_batch_metrics_csv(results, out_path):
    """Write a machine-readable metrics table across all batch snapshots.

    One row per snapshot. Columns include line-profile metrics (L_line, peak,
    trough), photospheric properties (R_phot, T_phot, L_phot, T_color), the
    full set of ionization-equilibrium diagnostics (τ_es STELLA vs photoeq,
    photoeq convergence, X_HII mean, Γ_BB and Γ_brems at the photosphere),
    and shock-derived parameters when shock-Xray was enabled (R_s, v_s,
    T_shock, L_shock, η_rad, L_X_brems, plus the sanity flag indicating
    whether the shock identification looked plausible).

    Designed so that when something looks off in a future run, the same CSV
    column tells you exactly which physics quantity changed — much easier
    than re-extracting from npz files.
    """
    import csv

    # Canonical column order. Snapshot-level metrics go first, then line metrics,
    # then per-epoch runtime info. This ordering makes diffing across runs easy.
    columns = [
        # identity
        'name', 'epoch_d', 'format', 'n_zones', 'n_zones_full',
        # photosphere from snapshot
        'R_phot', 'T_phot', 'L_phot', 'T_color_thermalization',
        # ionization equilibrium
        'photoionized', 'photoeq_T_source', 'photoeq_converged',
        'photoeq_iterations', 'photoeq_final_residual', 'X_HII_mean',
        'tau_es_stella', 'tau_es_photoeq', 'tau_es_ratio',
        'Gamma_BB_at_Rphot', 'Gamma_brems_at_Rphot',
        # shock parameters
        'shock_R_s', 'shock_v_s_kms', 'shock_rho_csm',
        'shock_T_shock', 'shock_L_shock', 'shock_eta_rad',
        'shock_L_X_brems', 'shock_sanity_flag',
        'shock_sanity_ratio_LXoverLphot',
        # line-profile metrics from MC
        'L_line', 'L_cont_band',
        'red_peak_F', 'red_peak_dv',
        'trough_F', 'trough_dv',
        'global_peak_F', 'global_peak_dv',
        'sigma_res2_vs_ref',
        # He I NLTE diagnostics (Phase 3; populated only if --he1-nlte was on)
        'he1_n_HeI_med', 'he1_n_HeII_med',
        'he1_2_3S_frac', 'he1_NLTE_LTE_ratio',
        'he1_tau_10830_med', 'he1_tau_5876_med',
        'he1_tau_7065_med', 'he1_tau_6678_med',
        'he1_iters',
        # He II NLTE diagnostics (Phase 4; populated only if --he2-nlte was on)
        'he2_X_HeIII_med', 'he2_X_HeII_med',
        'he2_n_HeII_med', 'he2_n_HeIII_med',
        'he2_tau_4686_med', 'he2_tau_1640_med',
        'he2_tau_3203_med', 'he2_tau_10124_med',
        'he2_iters',
        # runtime
        'n_iter_used', 'runtime_min',
    ]

    def _safe_get(d, key, default=''):
        v = d.get(key, default)
        if v is None:
            return ''
        if isinstance(v, (np.floating, float)) and np.isnan(v):
            return ''
        return v

    with open(out_path, 'w', newline='') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(columns)
        for r in results:
            sm = r.get('snap_metrics', {})
            tau_st = sm.get('tau_es_stella', np.nan)
            tau_pi = sm.get('tau_es_photoeq', np.nan)
            tau_ratio = (tau_pi / tau_st) if (isinstance(tau_st, (int, float, np.floating))
                                                and tau_st > 0) else np.nan
            row = {
                'name':                    r['name'],
                'epoch_d':                 r.get('epoch_d'),
                'format':                  r.get('format'),
                'n_zones':                 sm.get('n_zones'),
                'n_zones_full':            sm.get('n_zones_full'),
                'R_phot':                  sm.get('R_phot'),
                'T_phot':                  sm.get('T_phot'),
                'L_phot':                  sm.get('L_phot'),
                'T_color_thermalization':  sm.get('T_color_thermalization'),
                'photoionized':            sm.get('photoionized'),
                'photoeq_T_source':        sm.get('photoeq_T_source'),
                'photoeq_converged':       sm.get('photoeq_converged'),
                'photoeq_iterations':      sm.get('photoeq_iterations'),
                'photoeq_final_residual':  sm.get('photoeq_final_residual'),
                'X_HII_mean':              sm.get('X_HII_mean'),
                'tau_es_stella':           tau_st,
                'tau_es_photoeq':          tau_pi,
                'tau_es_ratio':            tau_ratio,
                'Gamma_BB_at_Rphot':       sm.get('Gamma_BB_at_Rphot'),
                'Gamma_brems_at_Rphot':    sm.get('Gamma_brems_at_Rphot'),
                'shock_R_s':               sm.get('shock_R_s'),
                'shock_v_s_kms':           sm.get('shock_v_s_kms'),
                'shock_rho_csm':           sm.get('shock_rho_csm'),
                'shock_T_shock':           sm.get('shock_T_shock'),
                'shock_L_shock':           sm.get('shock_L_shock'),
                'shock_eta_rad':           sm.get('shock_eta_rad'),
                'shock_L_X_brems':         sm.get('shock_L_X_brems'),
                'shock_sanity_flag':       sm.get('shock_sanity_flag'),
                'shock_sanity_ratio_LXoverLphot':
                                           sm.get('shock_sanity_ratio_LXoverLphot'),
                'L_line':                  r.get('L_line'),
                'L_cont_band':             r.get('L_cont_band'),
                'red_peak_F':              r.get('red_peak_F'),
                'red_peak_dv':             r.get('red_peak_dv'),
                'trough_F':                r.get('trough_F'),
                'trough_dv':               r.get('trough_dv'),
                'global_peak_F':           r.get('global_peak_F'),
                'global_peak_dv':          r.get('global_peak_dv'),
                'sigma_res2_vs_ref':       r.get('sigma_res2'),
                'n_iter_used':             r.get('n_iter_used'),
                'runtime_min':             (r.get('runtime_total', 0.0) / 60.0
                                            if r.get('runtime_total') else None),
            }
            # He I NLTE columns (only present when --he1-nlte was on)
            he1m = r.get('he1_metrics')
            if isinstance(he1m, dict):
                row.update({
                    'he1_n_HeI_med':       he1m.get('he1_n_HeI_med'),
                    'he1_n_HeII_med':      he1m.get('he1_n_HeII_med'),
                    'he1_2_3S_frac':       he1m.get('he1_2_3S_frac'),
                    'he1_NLTE_LTE_ratio':  he1m.get('he1_NLTE_LTE_ratio'),
                    'he1_tau_10830_med':   he1m.get('he1_tau_10830_med'),
                    'he1_tau_5876_med':    he1m.get('he1_tau_5876_med'),
                    'he1_tau_7065_med':    he1m.get('he1_tau_7065_med'),
                    'he1_tau_6678_med':    he1m.get('he1_tau_6678_med'),
                    'he1_iters':           he1m.get('he1_iters'),
                })
            # He II NLTE columns (only present when --he2-nlte was on)
            he2m = r.get('he2_metrics')
            if isinstance(he2m, dict):
                row.update({
                    'he2_X_HeIII_med':     he2m.get('he2_X_HeIII_med'),
                    'he2_X_HeII_med':      he2m.get('he2_X_HeII_med'),
                    'he2_n_HeII_med':      he2m.get('he2_n_HeII_med'),
                    'he2_n_HeIII_med':     he2m.get('he2_n_HeIII_med'),
                    'he2_tau_4686_med':    he2m.get('he2_tau_4686_med'),
                    'he2_tau_1640_med':    he2m.get('he2_tau_1640_med'),
                    'he2_tau_3203_med':    he2m.get('he2_tau_3203_med'),
                    'he2_tau_10124_med':   he2m.get('he2_tau_10124_med'),
                    'he2_iters':           he2m.get('he2_iters'),
                })
            w.writerow([_safe_get(row, c) for c in columns])
    print(f"Saved {out_path} ({len(results)} rows × {len(columns)} cols)")


def _auto_vlim(dv, F, thresh=0.02, pad=0.10, min_hw=700.0):
    """Velocity limits that enclose the whole line departure (>thresh from
    continuum), padded, with a minimum half-width so flat lines don't collapse."""
    dv = np.asarray(dv, float); F = np.asarray(F, float)
    dev = np.abs(F - 1.0) > thresh
    if dev.any():
        lo, hi = float(dv[dev].min()), float(dv[dev].max())
    else:
        lo, hi = -min_hw, min_hw
    c = 0.5 * (lo + hi); hw = max(0.5 * (hi - lo), min_hw)
    p = pad * 2 * hw
    return c - hw - p, c + hw + p


def _auto_flim(F, pad=0.10, floor_lo=0.92, floor_hi=1.08):
    """Flux limits enclosing the whole profile, with a floor so a ~flat line
    (e.g. He II at a cool photosphere) shows as flat rather than zoomed noise."""
    F = np.asarray(F, float)
    ymin = min(float(np.nanmin(F)), floor_lo)
    ymax = max(float(np.nanmax(F)), floor_hi)
    p = pad * (ymax - ymin + 1e-6)
    return max(0.0, ymin - p), ymax + p


def plot_batch_grid(results, out_path):
    """Grid plot of all snapshot profiles."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(results)
    if n == 0:
        return
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows))
    axes_flat = axes.ravel() if nrows > 1 else (
        axes if isinstance(axes, np.ndarray) else [axes])

    for i, r in enumerate(results):
        ax = axes_flat[i]
        ax.plot(r['dv'], r['F_norm'], 'C2-', lw=1.5)
        ax.scatter([r['red_peak_dv']], [r['red_peak_F']], color='red',
                   s=40, zorder=5)
        ax.axhline(1, color='gray', ls=':', lw=0.5)
        ax.axvline(0, color='gray', ls=':', lw=0.5)
        _vlo, _vhi = _auto_vlim(r['dv'], r['F_norm'])
        ax.set_xlim(_vlo, _vhi)
        ax.set_xlabel('Δv [km/s]')
        ax.set_ylabel('F / F_cont')
        title = f"{r['name']}"
        if r['epoch_d'] is not None:
            title += f" (t={r['epoch_d']:.1f}d)"
        title += (f"  │  L={r['L_line']:.1e}  "
                  f"F={r['red_peak_F']:.1f} @ {r['red_peak_dv']:+.0f}")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    for j in range(len(results), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Title reflects which snapshot family was processed
    formats = set(r.get('format', 'heracles') for r in results)
    if formats == {'stella'}:
        family = 'STELLA (post-Lbol-max)'
    elif formats == {'heracles'}:
        family = 'HERACLES'
    else:
        family = 'mixed'
    fig.suptitle(f'Batch: RT-NLTE Hα profiles across {family} time series',
                 fontsize=13, y=0.998)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def make_movie(results, out_path, fps=2):
    """Generate MP4 (or GIF fallback) showing line profile evolution.

    Each frame uses a dynamic y-axis so the line profile shape is visible
    regardless of whether peak F is 3 (early CSM phase) or 175 (peak
    interaction). An inset in the upper-right corner shows L_line(t) with
    a marker at the current frame, so the viewer always knows where they
    are in the time evolution and the absolute line strength.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if len(results) < 2:
        print("Not enough snapshots for movie")
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.10)

    # Inset axes for L_line(t)
    ax_inset = ax.inset_axes([0.66, 0.62, 0.32, 0.32])
    epochs = [r['epoch_d'] if r['epoch_d'] is not None else 0
              for r in results]
    L_lines = [max(r['L_line'], 1e30) for r in results]  # floor for log scale
    ax_inset.semilogy(epochs, L_lines, 'k-', lw=1.0, alpha=0.6)
    ax_inset.semilogy(epochs, L_lines, 'k.', ms=3, alpha=0.5)
    inset_dot, = ax_inset.semilogy([epochs[0]], [L_lines[0]],
                                     'ro', ms=10, zorder=5)
    ax_inset.set_xlabel('t [days post Lbol max]', fontsize=8)
    ax_inset.set_ylabel('L_Hα [erg/s]', fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(alpha=0.3, which='both')
    ax_inset.set_title('L_Hα(t)', fontsize=9)

    # Main plot setup
    line, = ax.plot([], [], 'C2-', lw=2)
    pk_pt = ax.scatter([], [], color='red', s=80, zorder=5)
    title = ax.set_title('')
    # Axes auto-scale PER FRAME (see update) so the whole profile — broad blue
    # absorption at early/fast epochs and the narrow rest-peak at late epochs —
    # always fills the frame. The L_Hα(t) inset carries the absolute scale.
    ax.set_xlabel('Δv [km/s]')
    ax.set_ylabel('F / F_cont')
    ax.axhline(1, color='gray', ls=':', lw=0.5)
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.grid(alpha=0.3)

    def update(idx):
        r = results[idx]
        dv = np.asarray(r['dv'])
        F = np.asarray(r['F_norm'])
        line.set_data(dv, F)
        pk_pt.set_offsets([[r['red_peak_dv'], r['red_peak_F']]])
        # Per-frame dynamic axes: x encloses the full line departure at THIS
        # epoch, y fills the frame with the whole profile (deep trough + peak).
        v_lo, v_hi = _auto_vlim(dv, F)
        ax.set_xlim(v_lo, v_hi)
        mask_view = (dv >= v_lo) & (dv <= v_hi)
        F_view = F[mask_view] if mask_view.any() else F
        ylo, yhi = _auto_flim(F_view)
        ax.set_ylim(ylo, yhi)

        # Update inset marker
        inset_dot.set_data([epochs[idx]], [L_lines[idx]])

        ep = r['epoch_d'] if r['epoch_d'] else 0
        title.set_text(
            f"{r['name']}   t = {ep:.1f} d post-Lbol-max   │   "
            f"L_Hα = {r['L_line']:.2e} erg/s   │   "
            f"peak F={r['red_peak_F']:.2f} @ {r['red_peak_dv']:+.0f} km/s")
        return line, pk_pt, title, inset_dot

    anim = FuncAnimation(fig, update, frames=len(results),
                          interval=1000//fps, blit=False)

    # Try MP4 first (ffmpeg), fall back to GIF
    try:
        anim.save(out_path, fps=fps, writer='ffmpeg',
                   dpi=120, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"Saved {out_path}")
    except Exception as e:
        # Fall back to GIF
        gif_path = out_path.replace('.mp4', '.gif')
        print(f"ffmpeg unavailable ({e}); saving GIF instead: {gif_path}")
        anim.save(gif_path, fps=fps, writer=PillowWriter(fps=fps))
        print(f"Saved {gif_path}")
    plt.close(fig)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Production RT-NLTE Hα pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('snap', nargs='?', help='snapshot file (single mode)')
    parser.add_argument('--batch', action='store_true',
                        help='batch process all atmosphere_*.dat in current dir')
    parser.add_argument('--epochs', type=str, default='',
                        help='comma-separated list of epoch days to run ONLY (keep-'
                             'only filter for --batch), e.g. '
                             '"0.1,1,3,5,10,20,30,40,50,80,100" for a sparse '
                             'back-test grid. Each requested day is matched to the '
                             'snapshot with that epoch (tolerance 1e-4); missing days '
                             'are warned. Applied before --skip-epochs.')
    parser.add_argument('--skip-epochs', type=str, default='',
                        help='comma-separated list of epoch values to skip in '
                             '--batch mode, e.g. "0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9". '
                             'Matched against the epoch number parsed from each '
                             'snapshot filename. Useful to thin out densely-sampled '
                             'early epochs. Matching tolerance is 1e-4.')
    parser.add_argument('--keep-shared-snapshots', action='store_true',
                        help='Do NOT skip late-epoch snapshots that are '
                             'byte-identical to a sibling model directory (the '
                             'shared-late-snapshot staging artifact). By default '
                             'such non-model-local copies are skipped in --batch '
                             'mode (STELLA) to avoid byte-identical duplicate rows '
                             'across models; pass this to process them anyway.')
    parser.add_argument('--n-per', type=int, default=100_000,
                        help='packets per chunk in final production run')
    parser.add_argument('--n-chunks', type=int, default=2,
                        help='chunks in final production run')
    parser.add_argument('--iter-n', type=int, default=50_000,
                        help='packets per chunk in RT iteration steps')
    parser.add_argument('--max-iter', type=int, default=6,
                        help='maximum RT iterations')
    parser.add_argument('--tol', type=float, default=0.03,
                        help='relative population convergence tolerance')
    parser.add_argument('--damping', type=float, default=0.3,
                        help='under-relaxation damping for J_bar updates '
                             '(0=no update, 1=full update). 0.3 stabilizes '
                             'oscillatory iteration.')
    parser.add_argument('--no-iter', action='store_true',
                        help='disable RT iteration (legacy mode)')
    parser.add_argument('--band-lo', type=float, default=6200.0,
                        help='lower bound of output wavelength band [Å]. '
                             'Default 6200. For line-dominated regime '
                             '(L_line/L_cont_band > 1), try 5500 to push '
                             'band edges into clean continuum.')
    parser.add_argument('--band-hi', type=float, default=6950.0,
                        help='upper bound of output wavelength band [Å]. '
                             'Default 6950. Pair with --band-lo for wider '
                             'band; e.g. --band-lo 5500 --band-hi 7500 '
                             'gives ±1000 Å around Hα.')
    parser.add_argument('--source-padding', type=float, default=1500.0,
                        help='source band extends (band_lo - padding) to '
                             '(band_hi + padding) [Å]. Default 1500 gives '
                             'source band 4700-8450 for default output. '
                             'Source must enclose output band so cont packets '
                             'can drift into edges via Thomson scattering.')
    parser.add_argument('--calibration', type=str, default='auto',
                        choices=['auto', 'theoretical_ew', 'f_cont_bb',
                                 'f_cont_bb_lambda', 'absolute'],
                        help="F_norm normalization baseline. Default 'auto' "
                             "selects per snapshot based on L_line/L_cont_band "
                             "ratio: <0.3 → 'theoretical_ew' (standard "
                             'F/F_cont), ≥0.3 → '
                             "'f_cont_bb_lambda' (line-dominated regime, "
                             'per-λ diluted BB baseline matching what '
                             'observers report). Override with explicit '
                             'choice for uniform calibration across batch.')
    parser.add_argument('--nbins', type=int, default=1200,
                        help='wavelength bins over 6200-6950 AA band. '
                             'Default 1200 (~28 km/s/bin at Hα). '
                             'Coarser (600) hides sub-30 km/s velocity '
                             'features; finer (3000) is noisier per bin.')
    parser.add_argument('--smooth-kms', type=float, default=25.0,
                        help='Gaussian velocity smoothing σ for F_norm display '
                             'and metric extraction. Default 25 km/s ≈ 0.9 '
                             'bins at nbins=1200, giving FWHM ~59 km/s '
                             '(well below typical Hα FWHM ≥80 km/s, so line '
                             'profile is preserved but per-bin MC noise '
                             'is suppressed). Set 0 to disable.')
    parser.add_argument('--ref', type=str, default=None,
                        help='CMFGEN reference file (single mode only)')
    parser.add_argument('--format', type=str, default='auto',
                        choices=['auto', 'heracles', 'stella'],
                        help="Snapshot format. 'auto' (default) detects from "
                             "filename; use 'heracles' or 'stella' to force.")
    parser.add_argument('--batch-format', type=str, default='auto',
                        choices=['auto', 'heracles', 'stella'],
                        help="Which snapshot family to batch-process. "
                             "'auto' picks whichever format has more files "
                             "in the working directory.")
    # Wind extension (STELLA-only, post-processing of input snapshot)
    parser.add_argument('--extend-wind', action='store_true',
                        help='Append a photoionized pre-SN wind extension '
                             'beyond the STELLA outer boundary. Adds '
                             'electron column for Thomson processing of '
                             'line + continuum in the unshocked CSM.')
    parser.add_argument('--wind-r-max-factor', type=float, default=20.0,
                        help='Extend wind to r_max = factor × r_outer. '
                             'Default 20 (out to ~10^17 cm for typical '
                             'STELLA snapshots).')
    parser.add_argument('--wind-n-zones', type=int, default=100,
                        help='Number of zones added in the extension. '
                             'Log-spaced. Default 100.')
    parser.add_argument('--wind-T-photoionized', type=float, default=10000.0,
                        help='Floor temperature [K] for the photoionized '
                             'extension wind. Default 10000.')
    parser.add_argument('--wind-rho-index', type=float, default=2.0,
                        help='Density falloff: ρ ∝ r^(-index). Default 2 '
                             '(steady-state wind). Use 1.5 for flatter '
                             'density (confined CSM-like).')
    parser.add_argument('--wind-density-boost', type=float, default=1.0,
                        help='Multiplier on extension density. Default 1 '
                             '(continuation of STELLA outer ρ). Set 3-10 '
                             'to represent enhanced pre-SN mass loss '
                             '(bright IIn regime, τ_es +1 to +5).')
    # Photoionization equilibrium (STELLA-only, applied after truncation)
    parser.add_argument('--no-photoionize', dest='photoionize',
                        action='store_false', default=True,
                        help='Disable per-zone photoionization equilibrium. '
                             'Default ON: solves CLOUDY/CMFGEN-style steady-'
                             'state photoionization-recombination balance for '
                             'every zone, overriding STELLA grey-diffusion '
                             'ionization. Source spectrum is B_ν(T_color) × '
                             'W(r), with T_color from snapshot thermalization '
                             'depth (τ_es=10) and L_phot from snapshot. '
                             'Disable to restore STELLA ionization (for '
                             'diagnostics or comparison).')
    parser.add_argument('--photoionize-T-source', type=float, default=None,
                        help='Override T_source for the photoionizing radiation '
                             '[K]. Default None: auto from snapshot at '
                             'τ_es=10 thermalization depth. Set to a specific '
                             'value to test sensitivity to spectral hardening '
                             '(e.g. for shock-X-ray contributions, try 30000-'
                             '100000). With L_normalize_to_snap=True (always '
                             'on), the bolometric L_phot is preserved.')
    parser.add_argument('--photoionize-T-eq-floor', type=float, default=10000.0,
                        help='Minimum gas temperature [K] in photoionized '
                             '(X_HII>0.5) zones. Real photoionization-'
                             'recombination equilibrium gives T_e ~ 10⁴ K; '
                             'STELLA may report cooler T in zones it treats '
                             'as neutral. This floor ensures downstream NLTE '
                             'Saha is consistent with the photoionization '
                             'X_HII. Default 10000. Set 0 to disable.')
    parser.add_argument('--no-shock-xray', dest='include_shock_xray',
                        action='store_false', default=True,
                        help='Disable thermal-bremsstrahlung shock X-ray '
                             'contribution to photoionization. Default ON: '
                             'derive shock parameters (R_s, v_s, ρ_csm) from '
                             'snapshot hydro at the position of max |dv/dr|, '
                             'compute T_shock from Rankine-Hugoniot and L_shock '
                             'from energy-flux balance, multiply by radiative '
                             'efficiency η_rad = 1/(1+t_cool/t_dyn). The '
                             'resulting L_X_brems is added to the source '
                             'spectrum as a thermal-brems component with '
                             'kT_shock spectral shape. Penetrates the CDS '
                             '(σ ∝ ν⁻³ at high ν), ionizes the CSM beyond '
                             'what BB photoionization alone can do.')
    parser.add_argument('--eps-lya-destruction', type=float, default=None,
                        help='Lyα destruction-probability floor for the NLTE '
                             'rate matrix (Solution 1). Floors β_(1,2) at the '
                             'given value, modelling Lyα → metal-line / 2-photon '
                             '/ continuum conversion that real CSM has but the '
                             'pure-Sobolev solver does not. Without this floor, '
                             'β_Lyα drops to ~1e-10 in slow-wind zones and n_2 '
                             'is over-populated by 4-5 orders of magnitude, '
                             'driving τ_Sob(Hα) to ~1e6 and producing '
                             'unphysical saturated P-Cygni profiles. Typical '
                             'values 1e-3 (aggressive) to 1e-5 (conservative). '
                             'Default None: legacy behavior (no floor). When '
                             'combined with --two-photon-decay, the ε floor '
                             'is applied to β_natural BEFORE the 2γ term is '
                             'added; lets ε represent additional destruction '
                             'channels (metal-line, continuum) on top of 2γ.')
    parser.add_argument('--two-photon-decay', action='store_true',
                        help='Phase 2: enable H(2s) → 1s two-photon decay '
                             'channel in the NLTE solve. First-principles '
                             'physical replacement for the --eps-lya-destruction '
                             'knob: 2s decays via 2γ emission (A_2γ = 8.23 s⁻¹) '
                             'bypassing the resonance trapping that makes '
                             'β_Lyα tiny. Assumes statistical equilibrium '
                             'between 2s and 2p (f_2s = 1/4, f_2p = 3/4); '
                             'valid for n_e > ~10⁷ cm⁻³. Effective rate is '
                             'A_eff = 0.75·A_Lyα·β_Lyα + 0.25·A_2γ. In '
                             'deeply-trapped zones gives A_eff ≈ 2 s⁻¹ '
                             '(physical floor, no parameter needed). Combine '
                             'with --eps-lya-destruction for additional '
                             'destruction channels. Diagnostics: per-zone '
                             'classification "2γ-dominated" vs "Lyα-dominated" '
                             'and A_eff are added to the populations diag dict.')
    parser.add_argument('--line-profile-method', type=str, default='mc',
                        choices=['mc', 'formal', 'unified'],
                        help="Emergent line-profile method. 'mc' (default): "
                             "Monte-Carlo peel (continuum scatter + volumetric "
                             "recombination). 'formal': Sobolev P-Cygni formal "
                             "solution with scattering source function "
                             "S_L=(1-eps)J_bar+eps*B, which removes the spurious "
                             "blueward emission peak and over-strong amplitude of "
                             "the thin-shell recombination channel. 'unified' "
                             "(opt-in): switch-free nonlocal ALI RT — gate-free "
                             "emergent profile + electron-scattering wings "
                             "(unified_line_rt), valid across all regimes.")
    parser.add_argument('--line-profile-method-lock', action='store_true',
                        dest='line_profile_method_lock', default=False,
                        help="Keep the line-STRENGTH policy (recombination-budget / "
                             "He-NLTE, no empirical anchor) consistent across all "
                             "snapshots in a batch. NOTE: this does NOT override the "
                             "Sobolev-validity gate — the profile SHAPE is always "
                             "chosen per-snapshot by the physics (formal for "
                             "homologous/IIP, MC for non-homologous/IIn). The flag "
                             "cannot force the formal solution onto a dense-CSM "
                             "snapshot. For a single SN the regime is consistent "
                             "across epochs anyway, so this is largely informational.")
    parser.add_argument('--line-redistribution', type=str, default='aa_prd',
                        choices=['semi_prd', 'crd', 'aa_prd'],
                        help='Line-scattering frequency redistribution mode '
                             '(Solution 2). semi_prd: preserve comoving '
                             'frequency across scatters (sharp Sobolev '
                             'P-Cygni). crd: redistribute to thermal '
                             'Maxwellian each scatter (broader narrow core, '
                             'softer P-Cygni — best for optically-thick IIn). '
                             'aa_prd: angle-averaged PRD, blends the two. '
                             'Default aa_prd. Forwarded end-to-end through '
                             'run_mc_chunked → run_peel_pipeline_abs → '
                             'run_mc_voigt_peel. Recommended pairing: use '
                             "'crd' together with --eps-lya-destruction 1e-3 "
                             'for IIn-style profiles. Using crd WITHOUT the '
                             "ε floor will destroy the line entirely (see "
                             'joint-test notes).')
    parser.add_argument('--out-prefix', type=str, default=None,
                        help='Override the output filename prefix. Default: '
                             'auto-derived from snapshot name as "prod_{label}" '
                             '(e.g. mesa.day040_post_Lbol_max.data → '
                             'prod_day040). Useful to avoid clobbering '
                             'previous runs when sweeping parameters: e.g. '
                             '--out-prefix prod_day040_eps1em3. Single-snapshot '
                             'mode only; ignored in --batch mode (batch uses '
                             'auto-derived names per snapshot).')
    parser.add_argument('--photosphere-mode', type=str, default='es',
                        choices=['es', 'cont'],
                        help='Photosphere definition for STELLA truncation. '
                             "'es' (default, legacy): τ_es = 2/3 from "
                             'electron-scattering only. Correct for IIn '
                             '(Thomson-thick CSM). '
                             "'cont' (Phase 1): τ_cont = 2/3 from full "
                             'continuum opacity (Thomson + H bf + H⁻ + ff). '
                             'For IIn-regime snapshots both give the same '
                             'answer; for IIP-plateau snapshots, cont '
                             'mode moves R_phot outward to the H recombination '
                             'front (T ≈ 6000 K), correctly identifying the '
                             'continuum-photosphere geometry. Validated on '
                             'synthetic IIP test (test_phase1_synthetic_IIP.py).')
    parser.add_argument('--photosphere-lam-ref', type=float, default=6562.8,
                        help='Reference wavelength [Å] for τ_cont = 2/3 surface '
                             'finding when --photosphere-mode=cont. Default '
                             '6562.8 (Hα). For multi-line work use a regime-'
                             'neutral choice like 5500 (V-band). Ignored when '
                             '--photosphere-mode=es.')
    # ---------- Phase 3: He I NLTE ----------
    parser.add_argument('--he1-nlte', action='store_true',
                        help='Enable He I NLTE level populations (Phase 3). '
                             'Computes populations for 11 He I levels and '
                             'reports Sobolev τ for He I 5876, 6678, 7065, '
                             '10830 Å. Does not yet add these lines to the MC '
                             'output spectrum (that is Phase 5). The Hα '
                             'output is bit-identical with and without this '
                             'flag.')
    parser.add_argument('--he1-eps-resonance', type=float, default=None,
                        help='ε destruction floor on He I 2¹P→1¹S (584 Å) '
                             'Sobolev β. Analog of --eps-lya-destruction for '
                             'the He I singlet resonance. Default None (only '
                             '2γ from 2¹S provides singlet destruction, which '
                             'is the physical channel). Set to 1e-4 if you '
                             'want symmetric treatment with H Lyα.')
    parser.add_argument('--no-he1-two-photon-decay',
                        dest='he1_two_photon_decay',
                        action='store_false', default=True,
                        help='Disable A=51.3 s⁻¹ 2γ destruction of He I '
                             '2¹S→1¹S. Default ON (the singlet metastable is '
                             'genuinely 2γ-destroyed in real plasmas). Off '
                             'is for diagnostic comparison only.')
    parser.add_argument('--he1-ionization-mode', type=str, default='follow_H',
                        choices=['follow_H', 'saha_approx'],
                        help='How to split He into HeI/HeII per zone. '
                             "'follow_H' (default): X_HeII = X_HII per zone "
                             '(the simplest credible approximation; uses the '
                             'converged photoionization-equilibrium X_HII). '
                             "'saha_approx': local-T Saha (overestimates HeII "
                             'at low T). When --he2-nlte is also on, Phase 4 '
                             'provides X_HeIII and Phase 3 adjusts X_HII '
                             '→ X_HII − X_HeIII before passing to the He I '
                             'solver (the v2 coupling).')
    # ---------- Phase 4: He II NLTE ----------
    parser.add_argument('--he2-nlte', action='store_true',
                        help='Enable He II NLTE level populations (Phase 4). '
                             'Computes populations for 10 He II levels '
                             '(n=1..10, l-collapsed, hydrogenic Z=2) and '
                             'reports Sobolev τ for He II 4686, 1640, 3203, '
                             '10124 Å. When combined with --he1-nlte, runs '
                             'BEFORE Phase 3 and provides X_HeIII feedback '
                             '(the v2 coupling). Adds 9 he2_* columns to '
                             'batch_metrics.csv. Does not yet add lines to '
                             'the MC output (that is Phase 5).')
    parser.add_argument('--he2-x-heiii-mode', type=str, default='saha_local',
                        choices=['saha_local', 'photoeq_match', 'zero'],
                        help='How Phase 4 determines X_HeIII per zone. '
                             "'saha_local' (default): local Saha at gas T, "
                             "n_e — correct for T < 25000 K cool conditions. "
                             "'photoeq_match': scales X_HeIII ∝ (X_HII/X_HI)² "
                             "as a crude Wien-suppression estimator (use for "
                             "hot IIn / shock-photoionization regimes). "
                             "'zero': force X_HeIII = 0 (test mode).")
    parser.add_argument('--he2-x-heiii-scalar', type=float, default=None,
                        help='Override X_HeIII with a single uniform value '
                             '[0, 1). Useful for testing the He II solver in '
                             'shock-photoionized conditions where Saha is '
                             'inappropriate; e.g. --he2-x-heiii-scalar=0.1 '
                             'mimics 10%% He III throughout the CSM. When '
                             'set, --he2-x-heiii-mode is ignored.')
    parser.add_argument('--he2-x-heiii-fraction', type=float, default=None,
                        help='Physically self-consistent X_HeIII override: '
                             'sets X_HeIII = f × X_HII per zone (f in [0, 1)). '
                             'Unlike --he2-x-heiii-scalar (uniform value, can '
                             'be inconsistent in low-X_HII regimes), this '
                             'guarantees X_HeIII ≤ X_HII per zone. When set, '
                             '--he2-x-heiii-scalar and -mode are ignored.')
    # ------- Phase 5: synthetic He multi-line profiles -------
    parser.add_argument('--he-lines', action='store_true',
                        help='Phase 5: produce synthetic F_λ profiles for the '
                             '8 He I + He II lines (10830, 5876, 7065, 6678, '
                             '1640, 4686, 3203, 10124 Å) using single-shot '
                             'Phase 3/4 populations. Requires --he1-nlte and/or '
                             '--he2-nlte to be enabled. Output: '
                             '{out_prefix}_he.npz, .txt, .png with per-line '
                             'F_λ over ±5000 km/s windows.')
    parser.add_argument('--he-lines-n-packets', type=int, default=50_000,
                        help='MC packets per He line in Phase 5 (default 50000).')
    parser.add_argument('--he-lines-calibration', type=str, default='theoretical_ew',
                        choices=['f_cont_bb', 'theoretical_ew', 'absolute',
                                  'f_cont_bb_lambda'],
                        help="Calibration mode for Phase 5 peel-off (default "
                             "'theoretical_ew', matches Hα auto-selection for "
                             "L_line/L_cont < 0.3). Use 'f_cont_bb' for "
                             "energy-bookkeeping against un-attenuated BB.")
    parser.add_argument('--he-lines-reference-mc', action='store_true',
                        help='Use the pure-Sobolev reference MC for Phase 5 '
                             'instead of the production peel_pipeline_abs '
                             'kernel. Faster and self-contained for debugging, '
                             'but produces emission-only profiles (no P-Cygni '
                             'absorption).')
    parser.add_argument('--metal-lines', action='store_true',
                        help='Phase 5c (P2 #5): add C/O/Ne metal lines (C IV 1549, '
                             'C III] 1909, C III 4647, [O I] 6300, [O III] 5007, '
                             '[Ne III] 3869) via first-principles emissivity '
                             'integrals (recombination + collisional CEL with the '
                             'n_crit correction) on photoionization-equilibrium '
                             'ion densities. Lines are merged into the npz/regime/'
                             'plots/movies dynamically. ATOMIC DATA IS PROVISIONAL '
                             '(verify vs CHIANTI/Cloudy before quoting absolute '
                             'fluxes). Most useful for the stripped/Icn (C-series).')
    parser.add_argument('--metal-cloudy', action='store_true',
                        help='Phase 5c Tier-2 (P2 #5): override the metal-line '
                             'ABSOLUTE luminosities with Cloudy (self-consistent '
                             'photoionization + multi-level NLTE + resonance-line '
                             'RT), fixing the C IV 1549 / C III] 1909 resonance '
                             'absolutes and the ionization. The MC profile SHAPE is '
                             'retained (Cloudy is static). Requires Cloudy compiled '
                             'and locatable ($CLOUDY_EXE or ~/c23.01/source/'
                             'cloudy.exe); falls back to CHIANTI/provisional per '
                             'line if Cloudy is absent or a run does not converge. '
                             'Implies --metal-lines.')
    parser.add_argument('--narrow-csm', action='store_true',
                        help='Add the narrow/intermediate P-Cygni component from '
                             'the unshocked slow CSM to the METAL line profiles '
                             '(for Icn/IIn fidelity vs observed spectra). DEFAULT '
                             'OFF — when off, output is byte-identical to before. '
                             'Reshapes only the metal F_norm profiles (and the '
                             'resonance-line net EW); L_line absolutes and the H/He '
                             'lines are never touched.')
    parser.add_argument('--he-budget', action='store_true',
                        help='Phase 5b (P1 #4): composition-general continuum '
                             'guard + He-budget diagnostics. Detects the '
                             'unphysical Wien collapse of L_cont_band at '
                             'cold/compact (esp. H-free) photospheres and floors '
                             'it to the energy-conserving color temperature '
                             '(4πR²σT⁴ = L_phot), so L_corr/L_cont_band EW '
                             'estimates are physical. Adds an energy-conservation '
                             'check and a first-principles He decrement (no '
                             'anchor). Profile shapes untouched. AUTO-enabled when '
                             '⟨X_H⟩ < 1e-3 (H-free).')
    parser.add_argument('--saturated-rt', action='store_true',
                        help='Phase 5b (P1 #3): for optically-thick He lines '
                             '(τ_med ≥ 1), replace the interim Hα-anchored escape '
                             'correction (R_flat) with the first-principles '
                             'continuum-pumped escape-probability source function '
                             '(line_rt_escape). Removes the empirical Hα anchor '
                             'from thick-line STRENGTHS; the profile SHAPE is '
                             'unchanged (still gate-authoritative MC). Opt-in: '
                             'without this flag, thick-line outputs are '
                             'byte-identical to before.')
    parser.add_argument('--phase5-movie-out', type=str,
                        default='batch_lines_evolution.mp4',
                        help='Output filename for the batch multi-line '
                             'evolution movie. Use .mp4 (default, via ffmpeg) '
                             'or .gif (via imageio). Only applies in --batch '
                             'mode with --he-lines.')
    parser.add_argument('--phase5-movie-fps', type=int, default=3,
                        help='Frames per second for the Phase 5 movie '
                             '(default 3). 30 epochs at 3 fps → 10 s movie.')
    args = parser.parse_args()

    # --metal-cloudy is a strength upgrade ON the metal lines → it implies them.
    if getattr(args, 'metal_cloudy', False) and not args.metal_lines:
        args.metal_lines = True

    if args.batch:
        # Find candidate files for each format
        heracles_paths = sorted(glob('atmosphere_*_new.dat'))
        stella_paths = sorted(glob('mesa.day*_post_Lbol_max.data')
                              + glob('mesa_day*_post_Lbol_max.data'))
        # Select which family to run
        fmt_batch = args.batch_format
        if fmt_batch == 'auto':
            if len(stella_paths) > len(heracles_paths):
                fmt_batch = 'stella'
            elif len(heracles_paths) > 0:
                fmt_batch = 'heracles'
            else:
                print("No HERACLES or STELLA snapshot files found in current "
                      "directory.\nLooking for atmosphere_*_new.dat or "
                      "mesa[._]day*_post_Lbol_max.data")
                sys.exit(1)

        if fmt_batch == 'heracles':
            snap_paths = heracles_paths
            def heracles_num(p):
                try:
                    return int(os.path.basename(p).split('_')[1])
                except Exception:
                    return 999
            snap_paths.sort(key=heracles_num)
        else:  # stella
            snap_paths = stella_paths
            def stella_epoch(p):
                m = re.search(r'day(-?\d+(?:\.\d+)?)_post', os.path.basename(p))
                return float(m.group(1)) if m else 9999.0
            snap_paths.sort(key=stella_epoch)

        if not snap_paths:
            print(f"No {fmt_batch.upper()} snapshots found in current directory")
            sys.exit(1)

        # Apply --epochs KEEP-ONLY filter (if given): run ONLY the listed epoch
        # days (e.g. for a sparse back-test grid). Matches each requested day to
        # the nearest available snapshot within 1e-4.
        keep_str = (getattr(args, 'epochs', None) or '').strip()
        if keep_str:
            try:
                keep_set = [float(x.strip()) for x in keep_str.split(',') if x.strip()]
            except ValueError as e:
                print(f"[batch] --epochs parse error: {e}")
                sys.exit(1)
            n_before = len(snap_paths)
            kept = []
            for p in snap_paths:
                epoch_val = (stella_epoch(p) if fmt_batch == 'stella'
                             else float(heracles_num(p)))
                if any(abs(epoch_val - s) < 1e-4 for s in keep_set):
                    kept.append(p)
            missing = [s for s in keep_set
                       if not any(abs((stella_epoch(p) if fmt_batch == 'stella'
                                       else float(heracles_num(p))) - s) < 1e-4
                                  for p in snap_paths)]
            snap_paths = kept
            print(f"[batch] --epochs: kept {len(snap_paths)}/{n_before} snapshots "
                  f"(requested {len(keep_set)} epochs)")
            if missing:
                print(f"[batch] --epochs WARNING: no snapshot for days {missing}")

        # Apply --skip-epochs filter (if given) before processing
        skip_str = (args.skip_epochs or '').strip()
        if skip_str:
            try:
                skip_set = [float(x.strip()) for x in skip_str.split(',') if x.strip()]
            except ValueError as e:
                print(f"[batch] --skip-epochs parse error: {e}")
                sys.exit(1)
            n_before = len(snap_paths)
            kept = []
            skipped = []
            for p in snap_paths:
                if fmt_batch == 'stella':
                    epoch_val = stella_epoch(p)
                else:
                    epoch_val = float(heracles_num(p))
                # Skip if this epoch matches any value in skip_set (within 1e-4)
                if any(abs(epoch_val - s) < 1e-4 for s in skip_set):
                    skipped.append((p, epoch_val))
                else:
                    kept.append(p)
            snap_paths = kept
            print(f"[batch] --skip-epochs: kept {len(snap_paths)}/{n_before} "
                  f"snapshots ({len(skipped)} skipped)")
            for p, e in skipped:
                print(f"  skip: {os.path.basename(p)}  (epoch={e})")

        # Detect & skip shared late-epoch snapshots (the 'shared late-epoch
        # snapshot' bug). A snapshot copied verbatim into several model
        # directories is not model-local and reproduces byte-identical output
        # rows across models. STELLA only — the model-grid layout is what makes
        # the sibling comparison meaningful.
        if fmt_batch == 'stella':
            shared = detect_shared_snapshots(snap_paths)
            if shared:
                shared_paths = {s['path'] for s in shared}
                print(f"[batch] shared-snapshot check: {len(shared)} snapshot(s) "
                      f"byte-identical to a sibling model directory "
                      f"(non-model-local):")
                for s in shared:
                    print(f"  shared: {s['basename']}  "
                          f"(== {s['sibling']}/{s['basename']})")
                if args.keep_shared_snapshots:
                    print("[batch] --keep-shared-snapshots set: processing them "
                          "anyway (duplicate cross-model rows will result).")
                else:
                    snap_paths = [p for p in snap_paths if p not in shared_paths]
                    print(f"[batch] skipped {len(shared)} shared snapshot(s); "
                          f"{len(snap_paths)} model-local snapshot(s) remain.")

        if not snap_paths:
            print("[batch] All snapshots were skipped — nothing to do.")
            sys.exit(1)

        print(f"[batch] {fmt_batch.upper()} mode: {len(snap_paths)} snapshots")
        process_batch(snap_paths, args)
    else:
        if not args.snap:
            parser.print_help()
            sys.exit(1)
        process_snapshot(
            args.snap,
            n_per=args.n_per, n_chunks=args.n_chunks,
            n_iter_per=args.iter_n, n_iter_chunks=1,
            max_iter=args.max_iter, tol=args.tol, damping=args.damping,
            nbins=args.nbins, smooth_kms=args.smooth_kms,
            band_AA=(args.band_lo, args.band_hi),
            source_padding_AA=args.source_padding,
            calibration=args.calibration,
            fmt=args.format,
            do_rt_iter=not args.no_iter,
            ref_path=args.ref,
            line_profile_method=args.line_profile_method,
            line_profile_lock=args.line_profile_method_lock,
            extend_wind=args.extend_wind,
            wind_r_max_factor=args.wind_r_max_factor,
            wind_n_zones=args.wind_n_zones,
            wind_T_photoionized=args.wind_T_photoionized,
            wind_rho_index=args.wind_rho_index,
            wind_density_boost=args.wind_density_boost,
            photoionize=args.photoionize,
            photoionize_T_source=args.photoionize_T_source,
            photoionize_T_eq_floor=args.photoionize_T_eq_floor,
            include_shock_xray=args.include_shock_xray,
            eps_Lya_destruction=args.eps_lya_destruction,
            two_photon_decay=args.two_photon_decay,
            line_redistribution=args.line_redistribution,
            out_prefix=args.out_prefix,
            photosphere_mode=args.photosphere_mode,
            photosphere_lam_ref_AA=args.photosphere_lam_ref,
            compute_he1_nlte=args.he1_nlte,
            he1_eps_resonance=args.he1_eps_resonance,
            he1_two_photon_decay=args.he1_two_photon_decay,
            he1_ionization_mode=args.he1_ionization_mode,
            compute_he2_nlte=args.he2_nlte,
            he2_x_heiii_mode=args.he2_x_heiii_mode,
            he2_x_heiii_scalar=args.he2_x_heiii_scalar,
            he2_x_heiii_fraction=args.he2_x_heiii_fraction,
            compute_he_lines=args.he_lines,
            he_lines_n_packets=args.he_lines_n_packets,
            he_lines_calibration=args.he_lines_calibration,
            he_lines_use_existing_kernel=not args.he_lines_reference_mc,
            saturated_rt=args.saturated_rt,
            he_budget=args.he_budget,
            metal_lines=args.metal_lines,
            metal_cloudy=args.metal_cloudy,
            narrow_csm=getattr(args, 'narrow_csm', False),
            verbose=True,
        )


if __name__ == '__main__':
    main()
