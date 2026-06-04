"""
measure_jbar.py — Compute J_bar(Hα) per zone from MC line-absorption counts.

For a line of integrated cross-section σ_int and lower-level population n_2,
the rate of line-photon absorption per unit volume is
    n_abs_per_vol = 4π × n_2 × σ_int × J_bar / hν   [photons/s/cm³]
Equivalently in energy:
    E_abs_per_vol = 4π × n_2 × σ_int × J_bar         [erg/s/cm³]

Inverting:
    J_bar = (E_abs_in_zone / dV) / (4π × n_2 × σ_int)
          [erg/s/cm²/sr/Hz at line center, assuming J ≈ const across line]

In the MC, the energy absorbed in zone i per unit time is
    E_abs[i] = (N_ls[i] / n_chunks) × L_per_packet
where N_ls[i] is the summed line-absorption count from n_chunks runs.

Notes
-----
- σ_int is the frequency-integrated line cross-section, in cm² Hz:
    σ_int = (π × e²/m_e c) × f_lu
  For Hα with f_23 = 0.6407, this is 0.01697 cm² Hz.
- The kernel's n_scat_per_zone counts ALL line interactions (the +=1 happens
  before the destroy-or-scatter branch), which is exactly what we want for
  J_bar — each interaction corresponds to one photon being absorbed (then
  possibly re-emitted, but that's a separate event).
- For a split-emit pipeline (cont + vol runs), the total J_bar gets
  contributions from BOTH runs. Sum them.
"""
import numpy as np

# Hα line constants
LAM_HA_CM = 6.56281e-5
NU_HA = 2.998e10 / LAM_HA_CM
F_HA = 0.6407
# Frequency-integrated cross-section (πe²/mc) × f:
SIGMA_INT_HA = np.pi * (4.8032e-10) ** 2 / (9.1094e-28 * 2.998e10) * F_HA


def compute_jbar_from_mc(snap, n_2, stats_continuum, stats_volumetric,
                          L_per_pkt_cont, L_per_pkt_vol, n_chunks,
                          floor_jbar=1e-15, verbose=False):
    """Compute J_bar(Hα) per zone from MC line-absorption counts.

    Parameters
    ----------
    snap : dict with 'r' (cm)
        Used to compute zone volumes.
    n_2 : array (n_zones,)
        Lower-level (n=2) populations [1/cm³] from current iteration.
    stats_continuum, stats_volumetric : dicts from MC kernel runs
        Must contain 'n_scat_per_zone' arrays summed across chunks.
    L_per_pkt_cont, L_per_pkt_vol : float
        Per-packet luminosity weights [erg/s] used in each kernel run.
    n_chunks : int
        Number of MC chunks (for averaging).
    floor_jbar : float
        Floor on J_bar to avoid divide-by-zero in solver. Default 1e-15.
    verbose : bool
        Print J_bar statistics for sanity checking.

    Returns
    -------
    J_bar_Ha : array (n_zones,) in erg/s/cm²/sr/Hz at line center.
    """
    r = np.asarray(snap['r'], dtype=float)
    n_zones = len(r)

    # Zone volumes (spherical shells)
    dr = np.empty_like(r)
    dr[:-1] = np.diff(r)
    dr[-1] = dr[-2]
    dV = 4.0 * np.pi * r * r * dr

    # Sum chunks for each channel; the cont kernel might have been called
    # n_chunks times, with the counter summed in each return.
    ns_cont = np.asarray(stats_continuum.get('n_scat_per_zone',
                                              np.zeros(n_zones)),
                          dtype=float)
    ns_vol = np.asarray(stats_volumetric.get('n_scat_per_zone',
                                              np.zeros(n_zones)),
                         dtype=float)

    # Energy absorbed per zone per unit time [erg/s]
    # (averaged over chunks: each chunk is an independent estimate)
    E_abs_cont = ns_cont * L_per_pkt_cont / max(n_chunks, 1)
    E_abs_vol = ns_vol * L_per_pkt_vol / max(n_chunks, 1)
    E_abs_total = E_abs_cont + E_abs_vol  # erg/s per zone

    # J_bar(line center) = (E_abs/dV) / (4π × n_2 × σ_int)
    denom = 4.0 * np.pi * np.maximum(n_2, 1e-30) * SIGMA_INT_HA * dV
    J_bar = E_abs_total / np.maximum(denom, 1e-30)

    # Floor for numerical stability — solver division-safe value
    J_bar = np.maximum(J_bar, floor_jbar)

    if verbose:
        # Compute zone-by-zone contributions for diagnostics
        pos = J_bar[J_bar > floor_jbar]
        n_pos = pos.size
        if n_pos > 0:
            print(f"  [measure_jbar] J_bar(Hα): {n_pos}/{n_zones} zones with "
                  f"non-trivial pumping. "
                  f"Range: {pos.min():.3e} to {pos.max():.3e}, "
                  f"median: {np.median(pos):.3e} erg/s/cm²/sr/Hz")
            # Realistic outer-wind J_bar ranges from ~1e-10 (cool diffuse CSM)
            # to ~1e-4 (dense wind in thick-line LTE limit). Only warn for
            # values WAY outside this range.
            med = float(np.median(pos))
            if med > 1e-3:
                print(f"  [measure_jbar] WARNING: median J_bar = {med:.2e} "
                      f"is implausibly large (>1e-3 erg/s/cm²/sr/Hz). "
                      f"Check n_2 input and σ_int constants.")
            elif med < 1e-12:
                print(f"  [measure_jbar] NOTE: median J_bar = {med:.2e} "
                      f"is below typical wind values; RT pumping will be "
                      f"negligible for this snapshot.")
        else:
            print(f"  [measure_jbar] J_bar(Hα): zero non-trivial pumping detected.")

    return J_bar
