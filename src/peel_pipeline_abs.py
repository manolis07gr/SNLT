"""
peel_pipeline_abs.py — Option B: absolute physical units throughout.

No free calibration parameters. Each packet carries its physical
luminosity weight; spectrum is in erg/s/AA. Final F/F_cont normalization
uses the analytic photospheric BB baseline computed from R_phot, T_phot.

Architecture:
  1. Continuum channel emits N_cont packets across [λ_lo_src, λ_hi_src].
     Total source luminosity in this band:
        L_cont_src = ∫ L_λ(T_phot, R_phot) dλ over [lo, hi]
     Each packet carries L_per_pkt_cont = L_cont_src / N_cont.

  2. Volumetric channel emits N_vol packets at line center with thermal
     width. Total line luminosity is L_line. Each packet carries
        L_per_pkt_vol = L_line / N_vol.

  3. The kernel adds L_per_pkt × (1/4π) × exp(-τ_to_obs) per peel event.
     Output spectrum has units of [erg/s/sr] integrated over each bin's
     wavelength range. To get F_λ at observer at distance D:
        F_λ(λ_obs) [erg/s/cm²/AA] = spectrum[bin] / (D² × Δλ_bin)

  4. Continuum baseline (analytic):
        F_λ_cont_BB(λ) = π B_λ(T_phot) × (R_phot/D)²
     This is the unattenuated photospheric flux at observer.

  5. F_norm = F_total / F_λ_cont_BB at each bin, ready for comparison
     to CMFGEN's F/F_cont.

D drops out because both numerator and denominator scale as 1/D², so
F_norm is D-independent. We use D=1 cm internally with no consequence.
"""
import sys
sys.path.insert(0, '/home/claude')
import numpy as np
import time

from sn_mc_voigt_peel import run_mc_voigt_peel
from lines import LINE_LIB

# Physical constants (CGS)
H_PLANCK = 6.62607015e-27   # erg·s
C_LIGHT = 2.99792458e10     # cm/s
KB = 1.380649e-16           # erg/K


def planck_lambda(lam_AA, T):
    """Planck function B_λ(T) in erg/s/cm²/AA/sr.
    
    B_λ(T) = (2hc²/λ⁵) × 1/(exp(hc/λkT) - 1)
    Convert to per-AA via factor 1e-8 (1 AA = 1e-8 cm).
    """
    lam_cm = np.asarray(lam_AA) * 1e-8
    x = H_PLANCK * C_LIGHT / (lam_cm * KB * T)
    # Avoid overflow for large x (e.g. far UV)
    with np.errstate(over='ignore'):
        denom = np.expm1(x)
    B_per_cm = (2.0 * H_PLANCK * C_LIGHT**2 / lam_cm**5) / denom
    # Convert from erg/s/cm²/cm/sr to erg/s/cm²/AA/sr
    return B_per_cm * 1e-8


def integrated_planck_band(T, lam_lo_AA, lam_hi_AA, n_quad=200):
    """∫ π B_λ(T) dλ over [lam_lo, lam_hi] in erg/s/cm²/(per source area)."""
    lam_grid = np.linspace(lam_lo_AA, lam_hi_AA, n_quad)
    Bs = planck_lambda(lam_grid, T)   # erg/s/cm²/AA/sr
    pi_B = np.pi * Bs   # × π gives flux from a flat source per cm²
    return np.trapezoid(pi_B, lam_grid)   # erg/s/cm²


SIG_SB = 5.670374419e-5    # Stefan-Boltzmann constant erg/s/cm²/K⁴


def L_cont_in_band(L_phot, T, lam_lo_AA, lam_hi_AA, n_quad=200):
    """Photospheric continuum luminosity in [lam_lo, lam_hi] AA, consistent
    with snline_autoparams.continuum_luminosity_per_A.
    
    Defined as L_phot × (Planck-band fraction of σT⁴):
        L_cont(band) = L_phot × ∫ π B_λ(T) dλ / (σ T⁴)
    
    This accounts for the fact that the snapshot's emergent L_phot is 
    typically much less than 4π R² σ T⁴ (the perfect-BB Stefan-Boltzmann
    value), because radiation diffuses through opacity and emerges with
    a softer spectrum than a pure BB at T_phot. Using L_phot×Planck-shape
    gives a per-packet luminosity matching production's calibration.
    """
    if L_phot <= 0 or T <= 0:
        return 0.0
    integrated_pi_B = integrated_planck_band(T, lam_lo_AA, lam_hi_AA, n_quad)
    return L_phot * integrated_pi_B / (SIG_SB * T**4)


def run_peel_pipeline_abs(snap, params, line=None,
                          n_packets=300_000, nbins=600,
                          band_AA=(6200.0, 6950.0),
                          source_padding_AA=1500.0,
                          v_turb_kms=None,
                          n_steps_peel=250,
                          step_dr_frac=3e-3,
                          calibration='theoretical_ew',
                          max_events=2000,
                          max_events_vol=None,
                          cont_line_emission=True,
                          line_scatter_redistribution='aa_prd',
                          seed=42, verbose=True):
    """Absolute-units peel-off pipeline.
    
    Returns: (lam, F_norm, F_cont_abs, F_total_abs, info)
    where F_norm = F_total_abs / F_cont_BB_baseline, ready for comparison
    to CMFGEN F/F_cont output.

    Parameters
    ----------
    calibration : str
        How to set the volumetric channel scale.
        
        'absolute' (no calibration): F_vol scale follows L_line/(4π D²)
            from energy conservation. The peel-off MC's per-packet
            yield ratio (cont vs vol) determines F_vol/F_cont ratio.
            For dense winds this OVER-PREDICTS the line by ~3-4× vs
            cone-MC and CMFGEN — peel-off counts per-event peels which
            are higher per packet than cone-MC's per-final-escape count.
        
        'theoretical_ew' (default): rescale F_vol so integrated F_vol over
            the output band equals the theoretical EW = (L_line/L_cont_band)
            × out_band. This matches production's cone-MC implicit
            calibration via scale_vol = (L_line/L_cont_band) × (out_band/
            src_band). Gives line strength comparable to production and
            CMFGEN for HERACLES-like atmospheres (τ_es ~ 30, cont packets
            escape freely). BREAKS DOWN for very thick atmospheres where
            cont packets don't escape directly (the far-wing F_cont
            baseline becomes meaningless).
        
        'f_cont_bb' (recommended for τ_es > 100 / STELLA): normalize by the
            diluted blackbody continuum F_cont_BB = π × B_λ(T_phot) × 
            (R_phot/D)². Physically meaningful regardless of how many cont
            packets actually escape — uses the un-attenuated photospheric
            BB as the reference continuum, which is the right convention
            for F/F_cont when comparing against detailed radiative transfer
            codes that handle diffusion correctly in thick atmospheres.

        'production': rescale F_vol so integrated F_vol over band matches
            production's effective EW × scaling. Requires a production
            reference run.
    """
    if line is None:
        line = LINE_LIB['Halpha']
    if v_turb_kms is None:
        v_turb_kms = params['v_turb_kms']
    # Per-zone v_turb (preferred over scalar). If params doesn't have it,
    # fall back to None — kernel will use scalar v_turb_kms.
    v_turb_kms_grid = params.get('v_turb_kms_grid', None)

    R_phot = float(params['R_phot'])
    T_phot = float(params['T_phot'])
    L_phot = float(params.get('L_phot', 4 * np.pi * R_phot**2 * SIG_SB * T_phot**4))
    D = 1.0   # cm; arbitrary, drops out of F_norm

    # Source band wavelength range
    lam_lo_src = band_AA[0] - source_padding_AA
    lam_hi_src = band_AA[1] + source_padding_AA
    src_band_AA = lam_hi_src - lam_lo_src

    # Total continuum luminosity in source band [erg/s], consistent with
    # snline_autoparams definition (L_phot × Planck-band fraction of σT⁴).
    # This matches what production's scale_vol formula uses for L_cont_band.
    L_cont_src = L_cont_in_band(L_phot, T_phot, lam_lo_src, lam_hi_src)

    # Total volumetric line luminosity carried by the packets. Prefer the
    # explicit physical EMERGENT (escaping) luminosity when the caller supplies
    # it (production and the He-line callback set params['L_line_emit'] =
    # Σ hν n_u A_ul β(τ) V, the source-function × Sobolev-escape luminosity).
    # Fall back to the legacy kept/gross case-B integral. This sets only the
    # luminosity the packets carry; params['emissivity'] still sets WHERE they
    # are born.
    if params.get('L_line_emit') is not None:
        L_line = float(params['L_line_emit'])
    else:
        L_line = float(params.get('L_line_kept', params['L_line']))

    # Per-packet luminosities
    if n_packets > 0:
        L_per_pkt_cont = L_cont_src / n_packets
        L_per_pkt_vol = L_line / n_packets
    else:
        L_per_pkt_cont = 0.0
        L_per_pkt_vol = 0.0

    if verbose:
        print(f"[abs] R_phot = {R_phot:.3e} cm, T_phot = {T_phot:.0f} K")
        print(f"[abs] L_cont_src ({lam_lo_src:.0f}-{lam_hi_src:.0f} AA) = {L_cont_src:.3e} erg/s")
        print(f"[abs] L_line = {L_line:.3e} erg/s, ratio L_line/L_cont = {L_line/max(L_cont_src,1e-30):.3e}")
        print(f"[abs] per-packet luminosities: cont = {L_per_pkt_cont:.3e}, vol = {L_per_pkt_vol:.3e}")

    common = dict(
        v_turb_kms=v_turb_kms,
        v_turb_kms_grid=v_turb_kms_grid,
        n_packets=n_packets, step_dr_frac=step_dr_frac,
        wavelength_range_AA=band_AA,
        source_padding_AA=source_padding_AA,
        nbins=nbins, max_events=max_events,
        tau_es_trim=None,
        frac_volumetric=params.get('f_vol', 0.5),
        line_scatter_redistribution=line_scatter_redistribution,
        photosphere_line_boundary='transmit',
    )
    eps_dest = params.get('eps_dest_per_zone', 0.0) or 0.0

    # Photoion-decoupled mode: if params has 'n_lower_absorb', use it as
    # the line-opacity carrier in the kernel (controls absorption along
    # peel-to-observer paths and along random-walk-step tau-marching).
    # The emission physics (emissivity array) was already computed using
    # the Saha-based n_lower in derive_parameters_from_state, so emission
    # comes from the inner ionized shell while absorption happens in the
    # outer wind where slow CSM has elevated n_2 from photoion balance.
    n_lower_for_kernel = params.get('n_lower_absorb', params['n_lower'])
    if verbose and 'n_lower_absorb' in params:
        scale_med = float(np.median(
            params['n_lower_absorb'] / np.maximum(params['n_lower'], 1e-30)))
        print(f"[abs] photoion-decoupled mode: n_lower_absorb / n_lower "
              f"median = {scale_med:.3e} (>1 means more outer-wind absorption)")

    if verbose:
        print(f"[abs] continuum peel-off, N={n_packets}, L_per_pkt={L_per_pkt_cont:.3e}")
    t0 = time.time()
    lam, raw_cont, st_c = run_mc_voigt_peel(
        snap['r'], snap['v'], snap['rho'], snap['T'], snap['n_e'],
        n_lower_for_kernel, params['n_upper'], line,
        emit_mode='continuum', seed=seed, eps_dest=eps_dest,
        n_steps_peel=n_steps_peel,
        packet_weight_cont=L_per_pkt_cont, packet_weight_vol=0.0,
        cont_line_emission=cont_line_emission,
        **common)
    t_cont = time.time() - t0

    if verbose:
        print(f"[abs] volumetric peel-off, N={n_packets}, L_per_pkt={L_per_pkt_vol:.3e}")
    t0 = time.time()
    if L_line > 0 and params.get('f_vol', 0) >= 1e-6:
        # Vol channel may need higher max_events when injecting deep-zone
        # emission (large τ_es): random walk needs ~τ² scatters to escape.
        common_vol = dict(common)
        if max_events_vol is not None:
            common_vol['max_events'] = int(max_events_vol)
        _, raw_vol, st_v = run_mc_voigt_peel(
            snap['r'], snap['v'], snap['rho'], snap['T'], snap['n_e'],
            n_lower_for_kernel, params['n_upper'], line,
            emit_mode='volumetric', emissivity=params['emissivity'],
            eps_dest=0.0, seed=seed + 1,
            n_steps_peel=n_steps_peel,
            packet_weight_cont=0.0, packet_weight_vol=L_per_pkt_vol,
            **common_vol)
    else:
        raw_vol = np.zeros_like(raw_cont)
        st_v = {}
    t_vol = time.time() - t0

    # raw_cont and raw_vol units after kernel:
    #   spectrum[bin] = sum L_per_pkt × (1/4π) × exp(-τ)  [erg/s, per bin]
    # 
    # F_λ at observer at distance D:
    #   F_λ[bin] [erg/s/cm²/AA] = spectrum[bin] / (D² × Δλ_bin)
    delta_lam = (band_AA[1] - band_AA[0]) / nbins
    F_cont_abs = raw_cont / (D * D * delta_lam)
    F_vol_abs = raw_vol / (D * D * delta_lam)
    F_total_abs = F_cont_abs + F_vol_abs

    # Continuum baseline for F/F_cont normalization:
    # 
    # The OBSERVED continuum (after wind attenuation, Thomson, etc.) is
    # what F_cont_abs computes. To get F_norm = F/F_cont matching CMFGEN's
    # convention, we use the MC-self-derived continuum at far wings (where
    # line absent and continuum is well-defined).
    # 
    # The crucial point: F_cont_abs and F_vol_abs are in the SAME physical
    # units (both erg/s/cm²/AA) because both kernels carry per-packet
    # luminosity weights. So F_total_abs preserves the absolute L_line/L_cont
    # physical relationship. Dividing by F_cont_abs's far-wing mean gives
    # F_norm consistent with CMFGEN's convention without losing any line
    # strength information.
    lam0_AA = line.lam0_cm * 1e8
    v_mask_far = 4000.0 / 3e5
    mask_fw = (lam < lam0_AA * (1 - v_mask_far)) | (lam > lam0_AA * (1 + v_mask_far))
    if np.any(mask_fw) and F_cont_abs[mask_fw].sum() > 0:
        F_cont_baseline_peel = float(np.mean(F_cont_abs[mask_fw]))
    else:
        F_cont_baseline_peel = 1.0

    # Diluted BB continuum at distance D (always computed). For very thick
    # winds where cont packets can't escape directly (e.g. τ_es >> 100), the
    # far-wing F_cont_abs is unreliable — most cont photons get trapped and
    # killed by max_events. F_cont_BB is the PHYSICAL un-attenuated reference
    # and is the right normalization in that regime.
    F_cont_BB = np.pi * planck_lambda(lam, T_phot) * (R_phot / D)**2
    F_cont_BB_baseline = float(np.mean(F_cont_BB[mask_fw])) if np.any(mask_fw) else 1.0

    # Choose which baseline to use for F_norm
    if calibration == 'f_cont_bb':
        # Use diluted-BB reference (single scalar mean over far wings).
        # Robust for highly-scattering atmospheres (STELLA-style) where
        # cont packets don't escape directly.
        F_cont_baseline = F_cont_BB_baseline
    elif calibration == 'f_cont_bb_lambda':
        # Use diluted-BB reference per wavelength (ndarray).
        # Right convention for line-dominated cases where cont channel
        # near line is contaminated by line-scattered continuum photons.
        # F_norm(λ) = F_total_abs(λ) / F_cont_BB(λ) gives the ratio of
        # observed flux to bare-photosphere BB, comparable to what
        # observers report (continuum interpolated through line region).
        F_cont_baseline = F_cont_BB
    else:
        # Default: use the MC-measured far-wing baseline (works when escape
        # rate is reasonable, e.g. HERACLES with τ_es ~ 30).
        F_cont_baseline = F_cont_baseline_peel

    # Sanity check: if peel-output baseline is wildly different from BB
    # baseline (>20× smaller), the atmosphere is too thick for the
    # max_events cap — warn the user.
    if F_cont_baseline_peel > 0 and F_cont_BB_baseline > 0:
        ratio = F_cont_baseline_peel / F_cont_BB_baseline
        if ratio < 0.05:
            print(f"[abs] WARNING: peel far-wing baseline is {ratio:.1%} of BB "
                  f"baseline. Cont packets likely hitting max_events cap "
                  f"(τ_es too high). Using calibration={calibration} "
                  f"({'F_cont_BB' if calibration=='f_cont_bb' else 'F_cont_peel — may be unreliable'}).")

    F_norm_uncalibrated = F_total_abs / F_cont_baseline

    # Apply calibration to volumetric channel.
    # The peel-off MC strictly conserves L_line/(4π D²) for the volumetric
    # channel (verified in clean-wind test), but the per-packet yield in
    # peel-off (sum over emission + scatter peel events) is systematically
    # different from cone-MC's (single final-escape count). For absolute
    # F_vol_norm to match production's cone-MC calibration, we anchor the
    # integrated F_vol over the output band to the theoretical EW expected
    # from L_line/L_cont:
    #     ∫F_vol over band = (L_line / L_cont_band) × out_band
    # where L_cont_band = autoparams' L_cont over output band. This is
    # exactly the production formula scale_vol × out_band. Doing this in
    # peel-off gives line strength matching production and CMFGEN.
    F_vol_norm_uncalibrated = F_vol_abs / F_cont_baseline
    EW_peel = float(np.trapezoid(F_vol_norm_uncalibrated, lam))
    L_cont_band = float(params.get('L_cont_band', L_cont_src * (band_AA[1]-band_AA[0])/src_band_AA))
    if L_cont_band > 0 and EW_peel > 0:
        if params.get('L_line_emit') is not None:
            _L_for_ew = float(params['L_line_emit'])
        elif params.get('L_line_kept') is not None:
            _L_for_ew = float(params['L_line_kept'])
        else:
            _L_for_ew = float(params['L_line'])
        EW_theoretical = _L_for_ew / L_cont_band * (band_AA[1] - band_AA[0])
    else:
        EW_theoretical = 0.0

    if calibration == 'theoretical_ew':
        calib_factor = EW_theoretical / max(EW_peel, 1e-30) if EW_peel > 0 else 1.0
    elif calibration == 'absolute':
        calib_factor = 1.0
    elif calibration == 'f_cont_bb':
        # F_cont_BB-based mode: the volumetric channel is already in the
        # right physical units (erg/s/cm²/AA at distance D); dividing by
        # F_cont_BB_baseline gives F_vol_norm directly. No further scaling.
        calib_factor = 1.0
    elif calibration == 'f_cont_bb_lambda':
        # Per-λ BB-based mode: same as f_cont_bb but with wavelength-
        # dependent denominator. No further scaling needed.
        calib_factor = 1.0
    else:
        calib_factor = 1.0

    F_vol_norm_calibrated = F_vol_norm_uncalibrated * calib_factor
    F_cont_norm = F_cont_abs / F_cont_baseline
    F_norm = F_cont_norm + F_vol_norm_calibrated

    if verbose:
        print(f"[abs] F_cont_baseline_peel (far wing): {F_cont_baseline_peel:.3e} erg/s/cm²/AA")
        print(f"[abs] F_cont_BB_baseline (diluted BB):  {F_cont_BB_baseline:.3e} erg/s/cm²/AA")
        # F_cont_baseline can be scalar (theoretical_ew, f_cont_bb) or
        # array (f_cont_bb_lambda). Report sensibly for both.
        if np.isscalar(F_cont_baseline) or (hasattr(F_cont_baseline, 'ndim') and F_cont_baseline.ndim == 0):
            print(f"[abs] using baseline:                   {float(F_cont_baseline):.3e} (mode={calibration})")
        else:
            bmean = float(np.mean(F_cont_baseline))
            bmin = float(np.min(F_cont_baseline))
            bmax = float(np.max(F_cont_baseline))
            print(f"[abs] using baseline:                   {bmean:.3e} mean "
                  f"(range {bmin:.3e}-{bmax:.3e}, per-λ, mode={calibration})")
        print(f"[abs] EW peel uncalibrated:        {EW_peel:.2f} AA")
        print(f"[abs] EW theoretical (L_line/L_cont × out_band): {EW_theoretical:.2f} AA")
        print(f"[abs] calibration mode: {calibration}, factor = {calib_factor:.4e}")

    # F_cont_BB is already computed above for normalization. Kept in info
    # dict for diagnostic plotting.

    info = dict(
        L_cont_src=L_cont_src,
        L_cont_band=L_cont_band,
        L_line=L_line,
        L_per_pkt_cont=L_per_pkt_cont,
        L_per_pkt_vol=L_per_pkt_vol,
        R_phot=R_phot, T_phot=T_phot,
        F_cont_baseline=F_cont_baseline,
        F_cont_BB=F_cont_BB,
        F_cont_abs=F_cont_abs,
        F_vol_abs=F_vol_abs,
        F_total_abs=F_total_abs,
        F_norm_uncalibrated=F_norm_uncalibrated,
        F_cont_norm=F_cont_norm,
        F_vol_norm=F_vol_norm_calibrated,
        EW_peel_uncalibrated=EW_peel,
        EW_theoretical=EW_theoretical,
        calib_factor=calib_factor,
        calibration_mode=calibration,
        wall_time_continuum=t_cont,
        wall_time_volumetric=t_vol,
        stats_continuum=st_c, stats_volumetric=st_v,
    )
    return lam, F_norm, F_cont_abs, F_total_abs, info
