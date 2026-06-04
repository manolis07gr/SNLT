"""
wind_extension.py
==================
Extend a STELLA snapshot's slow CSM outward to capture line/continuum
processing in the unshocked wind that STELLA's grid does not resolve.

Why this exists
---------------
STELLA's outer grid typically cuts off at a few × 10¹⁵ cm — sufficient
for hydrodynamics but truncates the slow CSM well before the
ionized-wind / pre-SN-wind boundary. Real IIn winds extend to
10¹⁶–10¹⁷ cm with a τ_es contribution of order 1-5 in the unshocked
zone. Without that column, Thomson redistribution of line and
continuum photons in the slow CSM is under-represented, leaving the
emerging profile sharper than real IIn observations.

Physical model used here
------------------------
Steady-state pre-SN wind continuation, photoionized to T ~ 10⁴ K by
the SN radiation:

  ρ(r)   = ρ_outer × (r_outer / r)^rho_index           [steady wind, default index=2]
  v(r)   = v_outer                                       [terminal velocity]
  T(r)   = max(T_outer, T_photoionized)                  [SN-illuminated]
  X_H, X_He, X_metals = same as outermost STELLA zone    [homogeneous wind]
  n_e(r) = Saha-ionized at T(r) using local n_H_total    [self-consistent]
  L(r)   = L(r_outer)                                    [no source/sink]

The extension is purely additive — existing STELLA zones are left
intact. Only τ_es is recomputed because the existing zones now see
extra column from the new outer wind.

When NOT to use
---------------
- For HERACLES snapshots (already include the full wind by convention)
- When the snapshot already extends to ≥ 10¹⁶ cm (check r[-1])
- For Type IIP or Ia models where the slow wind is irrelevant

Usage
-----
    from stella_io import load_stella_snapshot, truncate_to_photosphere
    from wind_extension import extend_wind_outward

    snap = load_stella_snapshot(path)
    snap = extend_wind_outward(snap, r_max_factor=20.0, T_photoionized=10000.0)
    snap = truncate_to_photosphere(snap)
    # ... pass to pipeline as usual
"""
from __future__ import annotations
import numpy as np

# CGS constants
KB     = 1.380649e-16
HPL    = 6.62607015e-27
ME     = 9.1093837e-28
MH     = 1.6735575e-24
SIGMAT = 6.65245871e-25
EV     = 1.602176634e-12
CHI_H  = 13.6      # eV, hydrogen ionization energy


def _saha_HII_fraction(T: np.ndarray, n_H_total: np.ndarray,
                        He_correction: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Solve Saha equation self-consistently for the H ionization fraction.

    n_e = n_p (charge neutrality, ignoring He at T < 20 kK)
    n_p² + K(T) × n_p - K(T) × n_H_total × He_correction = 0

    He_correction = n_H_total_effective / n_H_total. If He contributes
    electrons, the available recombination targets effectively increase.
    For T < 20 kK He stays neutral so He_correction = 1.0.

    Returns (X_HII, n_e) per zone.
    """
    T = np.asarray(T, dtype=float)
    n_H_total = np.asarray(n_H_total, dtype=float)
    kT = KB * T
    # K(T) = (2π m_e kT / h²)^(3/2) × exp(-χ_H / kT)
    pref = (2.0 * np.pi * ME * kT / (HPL * HPL))**1.5
    with np.errstate(over='ignore'):
        exparg = np.clip(CHI_H * EV / kT, -700.0, 700.0)
        K = pref * np.exp(-exparg)
    # n_p = (-K + sqrt(K² + 4 K n_H)) / 2  with He_correction
    disc = K * K + 4.0 * K * n_H_total * He_correction
    n_p = 0.5 * (-K + np.sqrt(np.maximum(disc, 0.0)))
    n_p = np.clip(n_p, 0.0, n_H_total)
    with np.errstate(divide='ignore', invalid='ignore'):
        X_HII = np.where(n_H_total > 0, n_p / n_H_total, 0.0)
    X_HII = np.clip(X_HII, 0.0, 1.0)
    n_e = n_p   # ignoring He contribution (cold He)
    return X_HII, n_e


def extend_wind_outward(snap: dict,
                          r_max_factor: float = 20.0,
                          r_max_cm: float | None = None,
                          n_zones_ext: int = 100,
                          T_photoionized: float = 10000.0,
                          rho_index: float = 2.0,
                          density_boost: float = 1.0,
                          spacing: str = 'log',
                          tie_temperature: bool = False,
                          verbose: bool = True) -> dict:
    """Extend the slow CSM of a STELLA snapshot outward.

    Adds `n_zones_ext` zones beyond the existing outer boundary, with
    physics described in the module docstring.

    Parameters
    ----------
    snap : dict
        STELLA snapshot dict (output of stella_io.load_stella_snapshot).
        Must have keys: 'r', 'v', 'rho', 'T', 'n_e', 'X_H', etc.
    r_max_factor : float
        Extend out to r_max = r_max_factor × r_outer. Default 20.
        Ignored if r_max_cm is given.
    r_max_cm : float or None
        Absolute outer radius in cm. Overrides r_max_factor if given.
    n_zones_ext : int
        Number of new zones to add. Default 100. Log-spaced or linear-
        spaced according to `spacing`.
    T_photoionized : float
        Photoionized-wind floor temperature [K]. Default 10000.
        The extension uses T(r) = max(T(r_outer), T_photoionized) per zone.
    rho_index : float
        Density falloff exponent: ρ(r) = ρ_outer × (r_outer/r)^rho_index.
        Default 2 (steady-state wind continuity). Use 1.5 for an Inhomogeneous-
        but-flatter wind; use higher values for steeper falloffs (less typical).
    density_boost : float
        Multiplier on the EXTENSION density (NOT the STELLA zones). Default 1.0.
        Use values > 1 to represent a denser pre-SN wind (e.g. ramped-up mass
        loss in the years before explosion). Values 3-10 are physically defensible
        for bright IIn-like CSM (Ṁ > 0.01 M_sun/yr). Affects τ_es of the wind
        extension linearly.
    spacing : {'log', 'linear'}
        Grid spacing of the extension zones. 'log' (default) puts more
        zones at smaller radii where opacity is highest.
    tie_temperature : bool
        If True, ramp temperature smoothly from T(r_outer) to T_photoionized
        over the first ~10% of the extension to avoid a discontinuity.
        If False (default), just step to T_photoionized at the first
        extension zone.
    verbose : bool
        If True, print τ_es contribution and basic statistics.

    Returns
    -------
    new_snap : dict
        Extended snapshot with N_orig + n_zones_ext zones. Same keys as
        input snap, all arrays grown consistently. τ_es is recomputed.
    """
    if snap.get('format') != 'stella':
        raise ValueError(f"extend_wind_outward expects a STELLA snapshot "
                         f"(got format={snap.get('format')!r})")

    r = np.asarray(snap['r'], dtype=float)
    v = np.asarray(snap['v'], dtype=float)
    rho = np.asarray(snap['rho'], dtype=float)
    T = np.asarray(snap['T'], dtype=float)
    n_e = np.asarray(snap['n_e'], dtype=float)

    n_orig = len(r)
    r_outer = float(r[-1])
    v_outer = float(v[-1])
    rho_outer = float(rho[-1])
    T_outer = float(T[-1])

    # Record original photosphere location (τ_es=2/3 surface of the
    # ORIGINAL grid) so that downstream truncate_to_photosphere can keep
    # the photosphere fixed at the STELLA value instead of letting the
    # added column push it outward through the line-forming CDS.
    tau_es_orig = np.asarray(snap['tau_es'])
    above_orig = np.where(tau_es_orig >= 2.0/3.0)[0]
    if len(above_orig) > 0:
        i_phot_orig = int(above_orig[-1])
        R_phot_orig = float(r[i_phot_orig])
    else:
        i_phot_orig = 0
        R_phot_orig = float(r[0])

    # Composition at outer boundary (used uniformly in extension)
    X_H_outer = float(np.asarray(snap.get('X_H', np.full_like(r, 0.737)))[-1])
    X_He_outer = float(np.asarray(snap.get('X_He', np.full_like(r, 0.249)))[-1])
    X_metals_outer = float(np.asarray(
        snap.get('X_metals', np.full_like(r, 1.0 - X_H_outer - X_He_outer)))[-1])
    mu_outer = float(np.asarray(snap.get('mu', np.full_like(r, 1.3)))[-1])

    # ---- Build the extension grid ----
    if r_max_cm is None:
        r_max = r_max_factor * r_outer
    else:
        r_max = float(r_max_cm)
    if r_max <= r_outer:
        raise ValueError(f"r_max ({r_max:.3e}) must be > r_outer ({r_outer:.3e})")

    if spacing == 'log':
        # Log-spaced from r_outer to r_max, EXCLUDING r_outer (avoid duplicate
        # zone) and INCLUDING r_max
        r_ext = np.logspace(np.log10(r_outer), np.log10(r_max),
                              n_zones_ext + 1)[1:]
    elif spacing == 'linear':
        r_ext = np.linspace(r_outer, r_max, n_zones_ext + 1)[1:]
    else:
        raise ValueError(f"spacing must be 'log' or 'linear'; got {spacing!r}")

    # ---- Extension physics ----
    # Density: power-law continuation, optionally boosted to represent
    # a denser pre-SN wind (e.g. enhanced mass loss before explosion).
    rho_ext = density_boost * rho_outer * (r_outer / r_ext)**rho_index

    # Velocity: terminal (constant)
    v_ext = np.full_like(r_ext, v_outer)

    # Temperature: photoionized floor (with optional smoothing at boundary)
    T_floor = max(T_outer, T_photoionized)
    if tie_temperature and T_outer < T_photoionized:
        # Smooth ramp from T_outer to T_photoionized over the first 10%
        # of the extension's log-radius span.
        log_r_span = np.log(r_ext / r_outer)
        log_r_ramp_end = 0.1 * log_r_span[-1]
        ramp = np.clip(log_r_span / log_r_ramp_end, 0.0, 1.0)
        T_ext = T_outer + ramp * (T_photoionized - T_outer)
    else:
        T_ext = np.full_like(r_ext, T_floor)

    # Ionization & electron density: Saha at T_ext
    n_H_total_ext = X_H_outer * rho_ext / MH
    X_HII_ext, n_e_ext = _saha_HII_fraction(T_ext, n_H_total_ext)

    # ---- Concatenate ----
    r_new = np.concatenate([r, r_ext])
    v_new = np.concatenate([v, v_ext])
    rho_new = np.concatenate([rho, rho_ext])
    T_new = np.concatenate([T, T_ext])
    n_e_new = np.concatenate([n_e, n_e_ext])
    n_orig_keep = n_orig
    n_total = n_orig_keep + n_zones_ext

    # X-arrays: extend with outer-zone values
    X_H_new = np.concatenate([np.asarray(snap.get('X_H', np.full_like(r, X_H_outer))),
                                np.full_like(r_ext, X_H_outer)])
    X_He_new = np.concatenate([np.asarray(snap.get('X_He', np.full_like(r, X_He_outer))),
                                np.full_like(r_ext, X_He_outer)])
    X_metals_new = np.concatenate([np.asarray(snap.get('X_metals', np.full_like(r, X_metals_outer))),
                                     np.full_like(r_ext, X_metals_outer)])
    mu_new = np.concatenate([np.asarray(snap.get('mu', np.full_like(r, mu_outer))),
                              np.full_like(r_ext, mu_outer)])

    # L: extend with last value (no emission/absorption in transparent wind)
    if 'L' in snap:
        L_outer = float(np.asarray(snap['L'])[-1])
        L_new = np.concatenate([np.asarray(snap['L']),
                                  np.full_like(r_ext, L_outer)])
    else:
        L_new = None

    if 'L_signed' in snap:
        L_signed_outer = float(np.asarray(snap['L_signed'])[-1])
        L_signed_new = np.concatenate([np.asarray(snap['L_signed']),
                                          np.full_like(r_ext, L_signed_outer)])
    else:
        L_signed_new = None

    # T_rad: simple constant continuation (could be refined as W(r)^0.25 × T_phot)
    if 'T_rad' in snap:
        T_rad_outer = float(np.asarray(snap['T_rad'])[-1])
        T_rad_new = np.concatenate([np.asarray(snap['T_rad']),
                                       np.full_like(r_ext, T_rad_outer)])
    else:
        T_rad_new = None

    # ---- Recompute τ_es (cumulative from outside) ----
    # τ_es[i] = ∫_{r[i]}^{r_outer_new} σ_T × n_e(r) dr
    dr_new = np.empty_like(r_new)
    dr_new[:-1] = np.diff(r_new)
    dr_new[-1] = dr_new[-2]
    # Integrate from outer inward
    kappa = SIGMAT * n_e_new
    # tau_es_new[i] = sum_{k>=i} kappa[k] * dr[k] (cumulative from outside)
    tau_es_new = np.cumsum((kappa * dr_new)[::-1])[::-1]

    # τ_stella (STELLA opacity-weighted): extend with last value
    if 'tau_stella' in snap:
        tau_stella_outer = float(np.asarray(snap['tau_stella'])[-1])
        tau_stella_new = np.concatenate([np.asarray(snap['tau_stella']),
                                           np.full_like(r_ext, tau_stella_outer)])
    else:
        tau_stella_new = None

    # Composition dict: per-species keys extended with outer values
    composition_new = None
    if 'composition' in snap and isinstance(snap['composition'], dict):
        composition_new = {}
        for key, arr in snap['composition'].items():
            arr = np.asarray(arr)
            if arr.ndim == 1 and len(arr) == n_orig:
                val_outer = float(arr[-1])
                composition_new[key] = np.concatenate(
                    [arr, np.full_like(r_ext, val_outer)])
            else:
                composition_new[key] = arr  # scalar or unrecognized shape

    # ---- Build new snap dict ----
    new_snap = dict(snap)  # shallow copy of metadata
    new_snap['r'] = r_new
    new_snap['v'] = v_new
    new_snap['rho'] = rho_new
    new_snap['T'] = T_new
    new_snap['n_e'] = n_e_new
    new_snap['X_H'] = X_H_new
    new_snap['X_He'] = X_He_new
    new_snap['X_metals'] = X_metals_new
    new_snap['mu'] = mu_new
    new_snap['tau_es'] = tau_es_new
    if L_new is not None:
        new_snap['L'] = L_new
    if L_signed_new is not None:
        new_snap['L_signed'] = L_signed_new
    if T_rad_new is not None:
        new_snap['T_rad'] = T_rad_new
    if tau_stella_new is not None:
        new_snap['tau_stella'] = tau_stella_new
    if composition_new is not None:
        new_snap['composition'] = composition_new
        new_snap['comp'] = composition_new

    # Track that this snapshot was extended (for downstream diagnostics)
    new_snap['wind_extended'] = True
    new_snap['n_zones_orig'] = n_orig
    new_snap['n_zones_ext'] = n_zones_ext
    new_snap['r_outer_orig'] = r_outer
    # Critical: stash the ORIGINAL (pre-extension) R_phot so that downstream
    # truncate_to_photosphere uses it instead of recomputing from the extended
    # tau_es. Without this, the added column would push the τ=2/3 surface
    # outward and hide the line-forming CDS inside the BB inner boundary.
    new_snap['R_phot_pre_extension'] = R_phot_orig
    new_snap['extension_params'] = dict(
        r_max=r_max, r_max_factor=r_max_factor, n_zones_ext=n_zones_ext,
        T_photoionized=T_photoionized, rho_index=rho_index,
        density_boost=density_boost,
        spacing=spacing, tie_temperature=tie_temperature,
    )

    # ---- Diagnostics ----
    if verbose:
        tau_es_ext_contribution = float(np.sum(SIGMAT * n_e_ext * np.diff(
            np.concatenate([[r_outer], r_ext]))))
        print(f"[wind_extension] {n_orig} → {n_total} zones "
              f"(+{n_zones_ext} extension zones)")
        print(f"  r_outer:           {r_outer:.3e} cm  →  "
              f"{r_max:.3e} cm  ({r_max/r_outer:.1f}× extension)")
        print(f"  ρ_outer:           {rho_outer:.3e} g/cm³  →  "
              f"{rho_ext[-1]:.3e} g/cm³  "
              f"(ρ ∝ r^-{rho_index:.1f}, boost={density_boost:.1f})")
        print(f"  v_ext (constant):  {v_outer/1e5:.1f} km/s")
        print(f"  T:                 outer STELLA {T_outer:.0f} K  →  "
              f"extension {T_floor:.0f} K  (photoionized)")
        print(f"  X_HII (extension): {X_HII_ext.min():.3f} – "
              f"{X_HII_ext.max():.3f}")
        print(f"  n_e (extension):   {n_e_ext[0]:.2e} → "
              f"{n_e_ext[-1]:.2e} cm⁻³")
        print(f"  τ_es: STELLA-only {float(snap['tau_es'].max()):.3f}  →  "
              f"extended {float(tau_es_new.max()):.3f}  "
              f"(extension Δτ ≈ {tau_es_ext_contribution:.3f})")
        # R_phot drift warning: if the τ_es=2/3 surface has moved significantly,
        # the truncation that follows will hide more of the inner CDS than
        # before. Estimate the new R_phot location.
        try:
            above_23_old = float(snap['tau_es'].max()) <= 2.0/3.0
            r_phot_old = (r[int(np.argmax(np.asarray(snap['tau_es']) <= 2.0/3.0))]
                          if not above_23_old else r[0])
            r_phot_new = r_new[int(np.argmax(tau_es_new <= 2.0/3.0))]
            drift = (r_phot_new - r_phot_old) / r_phot_old
            print(f"  R_phot (τ_es=2/3): {r_phot_old:.3e} cm  →  "
                  f"{r_phot_new:.3e} cm  ({drift*100:+.1f}%)")
            if abs(drift) > 0.10:
                print(f"  ℹ R_phot drift > 10% if truncate_to_photosphere "
                      f"is called without r_phot override. The extended snap "
                      f"carries R_phot_pre_extension={R_phot_orig:.3e} cm for "
                      f"downstream code (production_runner uses this "
                      f"automatically) to keep the photosphere fixed at the "
                      f"original location and preserve the inner CDS.")
        except (ValueError, IndexError):
            pass

    return new_snap


if __name__ == '__main__':
    # Quick standalone test on a day-40 snapshot
    import sys
    from stella_io import load_stella_snapshot, truncate_to_photosphere

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} mesa.day040 [r_max_factor]")
        sys.exit(1)
    path = sys.argv[1]
    r_max_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

    snap = load_stella_snapshot(path, verbose=True)
    print()
    snap_ext = extend_wind_outward(snap, r_max_factor=r_max_factor,
                                    T_photoionized=10000.0, verbose=True)
    print()
    snap_ext_trunc = truncate_to_photosphere(snap_ext, verbose=True)
    print()
    print(f"After truncation: {len(snap_ext_trunc['r'])} zones kept "
          f"(r: {snap_ext_trunc['r'][0]:.3e} → {snap_ext_trunc['r'][-1]:.3e} cm)")
    print(f"Final τ_es total = {float(snap_ext_trunc['tau_es'].max()):.3f}")
