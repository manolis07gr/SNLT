"""
stella_io.py — Loader for STELLA radiation-transport snapshot files.

STELLA snapshot files contain SN ejecta + CSM structure with full composition.
File format (one snapshot per file):
    Line 1:  ' days post max Lbol    <epoch_d>'
    Line 2:  '              zones    <n_zones>'
    Line 3:  ' inner boundary mass   <mass_g>   <mass_Msun>'
    Line 4:  '          total mass   <mass_g>   <mass_Msun>'
    Line 5:  blank
    Line 6:  column header line (38 fields)
    Line 7:  blank
    Lines 8+: data rows (1 zone index + 12 physical + 21 composition + 3 transport = 37 numeric cols)

Differences from HERACLES `atmosphere_*_new.dat`:
- Epoch is "days post Lbol max" (not "days since explosion").
- ~400 zones (vs 80 in HERACLES).
- Has full composition (h1 through ni56 mass fractions) per zone.
- Has both gas temperature and radiation temperature (can decouple in outer ejecta).
- Has matter Rosseland tau (not pure electron-scattering tau).
- Luminosity column may be negative (sign convention for outward flux).

This loader returns a dict in HERACLES schema so the rest of the pipeline
(analyze_snapshot, derive_parameters_from_state, etc.) works unchanged:

    {
        'r':      ndarray, cm
        'v':      ndarray, cm/s
        'rho':    ndarray, g/cm³
        'n_e':    ndarray, 1/cm³
        'T':      ndarray, K  (matter/gas temperature)
        'L':      ndarray, erg/s  (absolute value taken)
        'tau_es': ndarray, dimensionless  (COMPUTED from n_e × σ_T integrated outward)
        'epoch_d': float, days
        'path':    str
        'format':  'stella'

        # Extras specific to STELLA:
        'composition': {'h1': ndarray, 'he4': ndarray, ...}
        'T_rad':       ndarray, K  (radiation temperature; may differ from T in outer zones)
        'tau_stella':  ndarray  (the file's Rosseland tau, kept for reference)
        'X_H_emit':    float  (mass-weighted ⟨X_H⟩ over zones with tau_es < 1, for the pipeline)
    }

The 'epoch_d' here is "days post Lbol max" — the pipeline uses it for sorting and
display only, so the convention difference vs HERACLES doesn't affect physics.
"""
import os
import re
import numpy as np

# Column indices in the data array (0-indexed, after skipping the leading zone-id col)
# The data array has 37 numeric columns total. Column 0 is zone index.
COL_MASS_CELL = 1
COL_M_CENTER = 2
COL_R_CENTER = 3
COL_V_CENTER = 4
COL_RHO = 5
COL_P_RAD = 6
COL_T_GAS = 7
COL_T_RAD = 8
COL_OPACITY = 9
COL_TAU = 10
COL_M_OUTER = 11
COL_R_OUTER = 12
# Composition columns 13-33: 21 species
COMPOSITION_COLS = {
    'h1': 13, 'he3': 14, 'he4': 15, 'c12': 16, 'n14': 17, 'o16': 18,
    'ne20': 19, 'na23': 20, 'mg24': 21, 'si28': 22, 's32': 23, 'ar36': 24,
    'ca40': 25, 'ti44': 26, 'cr48': 27, 'cr60': 28, 'fe52': 29, 'fe54': 30,
    'fe56': 31, 'co56': 32, 'ni56': 33,
}
COL_LUMINOSITY = 34
COL_N_BAR = 35
COL_N_E = 36

SIGMA_T = 6.6525e-25  # Thomson cross-section, cm²


def parse_epoch_from_filename(path):
    """Extract epoch in days from a STELLA filename.

    Accepts both 'mesa.day080_post_Lbol_max.data' and
    'mesa_day080_post_Lbol_max.data' conventions. Also handles fractional
    days like 'mesa.day000.5_post_Lbol_max.data' → 0.5.

    Returns
    -------
    float or None if the filename doesn't match the expected pattern.
    """
    name = os.path.basename(path)
    m = re.search(r'day(-?\d+(?:\.\d+)?)_post', name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def compute_tau_es(r_cm, n_e):
    """Cumulative electron-scattering optical depth from outer boundary inward.

    Returns array of same length as r_cm. tau_es[i] is the optical depth
    of the layer EXTERIOR to zone i (i.e. between r[i] and r[-1]).
    tau_es[-1] = 0.

    Uses trapezoidal integration of (n_e × σ_T) dr.
    """
    n_zones = len(r_cm)
    tau_es = np.zeros(n_zones)
    # Integrate from outside in
    for i in range(n_zones - 2, -1, -1):
        # Trapezoidal element between zone i and zone i+1
        dr = r_cm[i+1] - r_cm[i]
        ne_avg = 0.5 * (n_e[i] + n_e[i+1])
        tau_es[i] = tau_es[i+1] + ne_avg * SIGMA_T * dr
    return tau_es


def load_stella_snapshot(path, verbose=False):
    """Load a STELLA snapshot file and return a HERACLES-schema dict.

    See module docstring for the returned dict layout.
    """
    # Parse header (first 4 lines have metadata)
    epoch_d = None
    n_zones = None
    with open(path) as f:
        for i, ln in enumerate(f):
            if i >= 4:
                break
            ln = ln.strip()
            if 'days post' in ln.lower() and 'lbol' in ln.lower():
                # Last token is the epoch value
                parts = ln.split()
                try:
                    epoch_d = float(parts[-1])
                except (ValueError, IndexError):
                    pass
            elif ln.lower().startswith('zones'):
                parts = ln.split()
                try:
                    n_zones = int(parts[-1])
                except (ValueError, IndexError):
                    pass

    # Fall back to filename if header didn't parse
    if epoch_d is None:
        epoch_d = parse_epoch_from_filename(path)

    # Data starts on line 8 (index 7)
    raw = np.loadtxt(path, skiprows=7)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]

    if n_zones is not None and raw.shape[0] != n_zones:
        if verbose:
            print(f"[stella_io] warning: header says {n_zones} zones but found "
                  f"{raw.shape[0]} data rows")

    # Extract columns into HERACLES schema
    r = raw[:, COL_R_CENTER]
    v = raw[:, COL_V_CENTER]
    rho = raw[:, COL_RHO]
    T_gas = raw[:, COL_T_GAS]
    T_rad = raw[:, COL_T_RAD]
    L_signed = raw[:, COL_LUMINOSITY]
    n_e = raw[:, COL_N_E]
    tau_stella = raw[:, COL_TAU]

    # Sort by radius (defensive — STELLA files are usually in order, but check)
    if not np.all(np.diff(r) > 0):
        order = np.argsort(r)
        r = r[order]
        v = v[order]
        rho = rho[order]
        T_gas = T_gas[order]
        T_rad = T_rad[order]
        L_signed = L_signed[order]
        n_e = n_e[order]
        tau_stella = tau_stella[order]
        raw = raw[order]

    # Compute electron-scattering tau from n_e × σ_T
    tau_es = compute_tau_es(r, n_e)

    # Composition (per zone) — keyed by species name
    composition_species = {sp: raw[:, col] for sp, col in COMPOSITION_COLS.items()}

    # Per-zone mass fractions in the format expected by snapshot_analyzer's
    # derive_composition(): keys 'X_H', 'X_He', 'X_metals'.
    X_H = composition_species['h1']
    X_He = composition_species['he4'] + composition_species['he3']
    # Metals = sum of everything else (everything that is not H or He)
    X_metals = np.zeros_like(r)
    for sp, X_arr in composition_species.items():
        if sp not in ('h1', 'he3', 'he4'):
            X_metals = X_metals + X_arr
    # Ensure X_H + X_He + X_metals = 1 (renormalize if STELLA mass fractions
    # don't perfectly sum to unity due to truncated species list).
    Xsum = X_H + X_He + X_metals
    # Guard against divide-by-zero
    safe_sum = np.where(Xsum > 0, Xsum, 1.0)
    X_H = X_H / safe_sum
    X_He = X_He / safe_sum
    X_metals = X_metals / safe_sum

    # The composition dict snapshot_analyzer expects has keys 'X_H', 'X_He',
    # 'X_metals'. We keep the per-species mass fractions alongside under their
    # element names for any downstream use.
    composition = {
        'X_H': X_H,
        'X_He': X_He,
        'X_metals': X_metals,
    }
    composition.update(composition_species)  # also keep h1, he4, o16, etc.

    # Mean molecular weight per zone, computed from the FULL composition.
    # μ = 1 / Σ_i (X_i / A_i × (1 + Z_i × ionization_state_i))
    # We don't know per-zone ionization for every species, so use the simple
    # neutral-atom μ for accurate mass→number density:
    #     μ_neutral = 1 / Σ (X_i / A_i)
    # This is fine for v_thermal = sqrt(kT/μm_p) used in turbulent broadening.
    # Atomic weights (approximate, accurate to ~0.5% for our purposes):
    A_ATOMIC = {
        'h1': 1.008, 'he3': 3.016, 'he4': 4.003, 'c12': 12.011, 'n14': 14.007,
        'o16': 15.999, 'ne20': 20.180, 'na23': 22.990, 'mg24': 24.305,
        'si28': 28.086, 's32': 32.065, 'ar36': 39.948, 'ca40': 40.078,
        'ti44': 47.867, 'cr48': 51.996, 'cr60': 51.996, 'fe52': 55.845,
        'fe54': 55.845, 'fe56': 55.845, 'co56': 58.933, 'ni56': 58.693,
    }
    inv_mu = np.zeros_like(r)
    for sp, X_arr in composition_species.items():
        A = A_ATOMIC.get(sp, 1.0)
        inv_mu = inv_mu + X_arr / A
    mu = 1.0 / np.maximum(inv_mu, 1e-30)

    # Mass-weighted ⟨X_H⟩ over zones tentatively above the photosphere
    # (tau_es < 1.0). Used as the scalar H mass fraction passed downstream.
    dr = np.empty_like(r)
    dr[:-1] = np.diff(r)
    dr[-1] = dr[-2]
    dV = 4.0 * np.pi * r * r * dr
    dm = rho * dV
    emit_mask = (tau_es < 1.0) & (X_H > 0)
    if emit_mask.sum() > 0:
        X_H_emit = float(np.average(X_H[emit_mask], weights=dm[emit_mask]))
    else:
        X_H_emit = float(np.average(X_H, weights=dm))
    # Same averaging for X_He — used by production_runner's X_He
    # override message and (potentially) downstream diagnostics.
    # Emit mask here is just (tau_es < 1.0) since X_He > 0 in essentially
    # all zones (vs X_H which can be zero in inner He/O-rich layers).
    he_emit_mask = (tau_es < 1.0)
    if he_emit_mask.sum() > 0:
        X_He_emit = float(np.average(X_He[he_emit_mask], weights=dm[he_emit_mask]))
    else:
        X_He_emit = float(np.average(X_He, weights=dm))

    snap = {
        # HERACLES schema
        'r': r,
        'v': v,
        'rho': rho,
        'n_e': n_e,
        'T': T_gas,
        'L': np.abs(L_signed),
        'tau_es': tau_es,
        'epoch_d': epoch_d,
        'path': path,
        'format': 'stella',
        # Per-zone composition arrays in BOTH conventions so whichever the
        # downstream code uses, it finds the data:
        # 1. Top-level X_H/X_He (some pipelines look here)
        # 2. snap['composition'] dict with X_H/X_He/X_metals (snapshot_analyzer
        #    convention) AND per-species keys (h1, he4, o16, …)
        # 3. snap['comp'] alias to snap['composition'] (some analyzers use 'comp')
        'X_H': X_H,
        'X_He': X_He,
        'X_metals': X_metals,
        'mu': mu,
        'composition': composition,
        'comp': composition,         # alias
        # STELLA-specific extras
        'T_rad': T_rad,
        'L_signed': L_signed,
        'tau_stella': tau_stella,
        'X_H_emit': X_H_emit,
        'X_He_emit': X_He_emit,
    }

    if verbose:
        print(f"[stella_io] {os.path.basename(path)}: "
              f"epoch={epoch_d}d, n_zones={len(r)}")
        print(f"  r:     {r[0]:.2e} to {r[-1]:.2e} cm")
        print(f"  v:     {v[0]/1e5:.1f} to {v[-1]/1e5:.1f} km/s "
              f"(max {v.max()/1e5:.1f})")
        print(f"  T_gas: {T_gas.min():.0f} to {T_gas.max():.0f} K")
        print(f"  ne:    {n_e.min():.2e} to {n_e.max():.2e} cm⁻³")
        print(f"  tau_es total: {tau_es.max():.2f}")
        print(f"  ⟨X_H⟩ (emit zones): {X_H_emit:.4f}")
        print(f"  ⟨X_He⟩ (emit zones): {X_He_emit:.4f}")
        # Velocity non-monotonicity warning (shocks)
        dv = np.diff(v)
        n_rev = int((dv < 0).sum())
        if n_rev > 0:
            print(f"  v(r) has {n_rev}/{len(dv)} reversals (shocks in "
                  f"CSM interaction). MC kernel uses local Doppler broadening, "
                  f"not strict Sobolev, so this is handled but flagged.")
    return snap


# ---------- Photosphere truncation ----------

def truncate_to_photosphere(snap, r_phot=None, tau_es_threshold=2.0/3.0,
                              verbose=True):
    """Truncate a STELLA snapshot to zones outside the photosphere.

    Why this exists
    ---------------
    The MC pipeline (peel_pipeline_abs + sn_mc_voigt_peel) was written for
    HERACLES atmospheres, where the snap file contains only zones at and
    above the photosphere — the inner boundary IS the photosphere by file
    convention. The MC kernel emits cont packets from r[0] (treated as
    opaque BB) and tracks them through r[0] < r ≤ r[-1].

    STELLA snapshots are different: they give the FULL hydrodynamic
    structure from r=0 (deep ejecta, behind the photosphere) out to the
    outer CSM. For a typical SN IIn this includes a dense cool shell (CDS)
    inside R_phot which has enormous τ_es (hundreds). If we hand this to
    the MC unfiltered, cont packets emitted at r[0] random-walk through
    the deep ejecta and almost never escape — the MC is trying to compute
    radiation transport for a region that physically should be hidden
    inside the opaque photosphere.

    The Chugai+2004 SN 1994W model (§5) handles this correctly by treating
    the CDS as an opaque inner boundary (BB at T_phot, R_phot) and running
    MC only on the OUTER CS envelope. This function implements the same
    geometry: keep only zones with r > R_phot, treat the truncated inner
    boundary as the photosphere.

    Parameters
    ----------
    snap : dict
        Loaded STELLA snapshot (output of load_stella_snapshot).
    r_phot : float or None
        Photospheric radius in cm. If None, computed from snap['tau_es']
        (cumulative electron-scattering optical depth from outside).
    tau_es_threshold : float
        Photosphere defined as τ_es from outside = tau_es_threshold (2/3
        is standard for a grey atmosphere with isotropic scattering).
    verbose : bool
        If True, print summary of truncation.

    Returns
    -------
    truncated_snap : dict
        Same keys as snap, but with all per-zone arrays restricted to
        zones with r > r_phot. Also adds:
            'R_phot_inner' : float  - radius of new inner boundary
            'T_phot_inner' : float  - T_gas at the truncated inner boundary
            'L_phot_inner' : float  - luminosity at the truncated inner boundary
            'n_zones_full' : int    - original zone count (before truncation)
            'r_phot_used'  : float  - r_phot value used for truncation
    """
    r = snap['r']
    tau_es = snap['tau_es']  # cumulative from outside; monotone DECREASING in i
    n_zones_full = len(r)

    # Determine R_phot if not given
    if r_phot is None:
        # tau_es decreases with i. Find the largest i where tau_es[i] >= threshold.
        # That zone is the deepest zone the observer can see into; one zone
        # outward is the first "above-photosphere" zone.
        above_phot = np.where(tau_es >= tau_es_threshold)[0]
        if len(above_phot) == 0:
            # Entire structure is already above τ=2/3 — no truncation needed
            i_phot = 0
        else:
            i_phot = int(above_phot[-1])
        r_phot = float(r[i_phot])
    else:
        # Use the given r_phot; find index just inside it
        i_phot = int(np.searchsorted(r, r_phot))
        if i_phot >= len(r):
            raise ValueError(f"r_phot={r_phot:.3e} is outside outer boundary "
                             f"r[-1]={r[-1]:.3e}")

    # Keep zones strictly outside R_phot
    mask = r > r_phot
    n_keep = int(mask.sum())
    if n_keep < 5:
        raise ValueError(f"Truncation at r_phot={r_phot:.3e} cm leaves only "
                         f"{n_keep} zones; need at least 5. Check r_phot or "
                         f"tau_es_threshold (current threshold = {tau_es_threshold}).")

    # T_phot, L_phot at the truncated inner boundary
    # Use the boundary zone (just inside R_phot, last zone with tau >= threshold)
    if i_phot < len(snap['T']):
        T_phot = float(snap['T'][i_phot])
    else:
        T_phot = float(snap['T'][-1])
    if 'L' in snap and i_phot < len(snap['L']):
        L_phot = float(np.abs(snap['L'][i_phot]))
    else:
        # Compute from BB: L = 4π R² σ T⁴
        SIGMA_SB = 5.6704e-5
        L_phot = 4.0 * np.pi * r_phot * r_phot * SIGMA_SB * T_phot**4

    # Build truncated snap
    new_snap = {}
    for key, val in snap.items():
        if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) == n_zones_full:
            new_snap[key] = val[mask].copy()
        elif key in ('composition', 'comp') and isinstance(val, dict):
            new_snap[key] = {sp: arr[mask].copy() for sp, arr in val.items()
                              if isinstance(arr, np.ndarray) and len(arr) == n_zones_full}
        else:
            new_snap[key] = val

    # Re-zero tau_es relative to the new outer boundary
    # tau_es[i] in original = integral from r[i] to r[-1] of n_e σ_T dr
    # After truncation, the new outermost zone has the same tau_es value
    # (which should be ~0 for STELLA which extends to outer wind boundary).
    # We re-reference so the truncated array still represents
    # cumulative-from-outside in the truncated structure (the values from
    # the original tau_es are still meaningful per-zone in this sense).
    # No change needed if outer boundary tau was 0.

    new_snap['R_phot_inner'] = r_phot
    new_snap['T_phot_inner'] = T_phot
    new_snap['L_phot_inner'] = L_phot
    new_snap['n_zones_full'] = n_zones_full
    new_snap['r_phot_used'] = r_phot
    new_snap['tau_es_threshold_used'] = tau_es_threshold

    # ----------------------------------------------------------------
    # Compute T_color_thermalization from the FULL pre-truncation
    # snapshot, used by downstream photoionization solvers as the
    # default source-spectrum temperature.
    #
    # Rationale: in a Thomson-scattering atmosphere, photons emerging
    # at the photosphere (τ_es = 2/3) carry the spectrum of the layer
    # where they last thermalized. The thermalization depth is at
    # τ_es ~ 1/sqrt(3 ε), where ε = κ_abs/κ_total. For typical SN
    # IIn opacities, ε ~ 0.001-0.01, so thermalization at τ_es ~ 10-30.
    # We tabulate T_gas at several fiducial depths so callers can
    # pick one consistent with their ε assumption.
    # ----------------------------------------------------------------
    tau_full = np.asarray(snap['tau_es'], dtype=float)
    T_full = np.asarray(snap['T'], dtype=float)
    T_color_at_tau = {}
    for tau_target in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        idx = int(np.argmin(np.abs(tau_full - tau_target)))
        T_color_at_tau[tau_target] = float(T_full[idx])
    new_snap['T_color_at_tau_es'] = T_color_at_tau
    # Default fiducial T_color: thermalization at τ_es=10 (ε~0.003,
    # appropriate for SN ejecta with iron-group line opacity).
    new_snap['T_color_thermalization'] = T_color_at_tau[10.0]

    if verbose:
        tau_outer = float(new_snap['tau_es'].max()) if 'tau_es' in new_snap else 0.0
        print(f"[stella_truncate] {n_zones_full} → {n_keep} zones")
        print(f"  R_phot = {r_phot:.3e} cm (τ_es = {tau_es_threshold})")
        print(f"  T_phot = {T_phot:.0f} K (gas T at inner boundary)")
        print(f"  L_phot = {L_phot:.3e} erg/s")
        print(f"  Outer-region τ_es (from photosphere) = {tau_outer:.2f}")
        print(f"  Outer zones span r = {new_snap['r'][0]:.3e} to "
              f"{new_snap['r'][-1]:.3e} cm")
        print(f"  T_color at thermalization (τ_es=10, ε~0.003): "
              f"{new_snap['T_color_thermalization']:.0f} K")
        print(f"  T_color sensitivity (T_gas at τ_es = 1, 3, 10, 30, 100, 300):")
        for tau_target in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
            print(f"    τ_es={tau_target:>6.0f}:  T={T_color_at_tau[tau_target]:>6.0f} K")
        v = new_snap.get('v', None)
        if v is not None:
            print(f"  Outer v range: {v.min()/1e5:.0f} to {v.max()/1e5:.0f} km/s")

    return new_snap


# ---------- Format detection helper ----------

def detect_format(path):
    """Return 'stella', 'heracles', or 'unknown' for a snapshot file.

    Detection is filename-based:
      - 'mesa.day*' or 'mesa_day*' → stella
      - 'atmosphere_*_new.dat' → heracles
    Falls back to inspecting the first line of the file.
    """
    name = os.path.basename(path).lower()
    if name.startswith('mesa.day') or name.startswith('mesa_day'):
        return 'stella'
    if name.startswith('atmosphere_') and name.endswith('_new.dat'):
        return 'heracles'
    # Content-based fallback
    try:
        with open(path) as f:
            first = f.readline().strip().lower()
        if 'days post' in first and 'lbol' in first:
            return 'stella'
        if 'time since' in first or 'explosion' in first:
            return 'heracles'
    except Exception:
        pass
    return 'unknown'
