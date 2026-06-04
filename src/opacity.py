"""opacity.py — Continuum opacity at arbitrary wavelength, per zone.

Per-zone, per-wavelength continuum opacity κ_cont(λ) [cm⁻¹], summing all
channels that matter for the τ_cont = 2/3 surface determination AND for
line-transfer photon propagation when continuum re-absorption is important.

Channels included
-----------------
  κ_es    : Thomson scattering         (n_e × σ_T)
  κ_bf_H  : H bound-free               (Σ_n n_HI(n) × σ_bf(n, λ), n ≥ n_min)
  κ_H-    : H⁻ bound-free              (n_H⁻ × σ_H-,bf(λ))
  κ_ff_H  : H free-free                (electron-proton bremsstrahlung absorption)

NOT included (by design — locked scope is line transfer only for H/He lines)
  - metal bound-free (would dominate UV blanketing below ~3000 Å)
  - He bound-free (corrections at λ < 504 Å only)
  - molecular opacities (irrelevant at T > 3000 K)

Public API
----------
  kappa_cont_total(lam_AA, T, n_e, n_p, n_HI_per_level, X_H) → kappa [cm⁻¹]
       per-zone array; n_HI_per_level is shape (nlev, nzones), in cm⁻³

  kappa_cont_at_wavelength(lam_AA, zone_state) → kappa [cm⁻¹]
       convenience wrapper; zone_state is a dict with the per-zone arrays

  sigma_bf_H(n, lam_AA) → cm² per atom in level n at wavelength λ
       Returns 0 if λ > λ_edge(n) (no absorption below threshold)

  sigma_H_minus(lam_AA) → cm² per H⁻ ion at wavelength λ
       John (1988) polynomial fit, valid 1250–16400 Å

  n_H_minus(T, n_HI_ground, n_e) → cm⁻³
       LTE Saha for H⁻ formation (H⁻ binding 0.754 eV, partition Z_H- = 1)

References
----------
  Mihalas (1978) Stellar Atmospheres, ch. 4
  John, T.L. (1988) A&A 193, 189 (H⁻ continuous opacity)
  Hummer (1988) ApJ 327, 477 (Gaunt factors)
"""
import numpy as np

# ----- physical constants (cgs) -----
KB        = 1.380649e-16
HPL       = 6.62607015e-27
ME        = 9.1093837015e-28
MH        = 1.6735575e-24
C_LIGHT   = 2.99792458e10
EE        = 4.8032068e-10
SIGMA_T   = 6.6524587e-25
EV        = 1.602176634e-12
RYDBERG   = 13.605693122994 * EV    # erg
A_BOHR    = 5.29177e-9              # cm
LAM_RYDBERG_AA = (HPL * C_LIGHT / RYDBERG) * 1e8     # 911.27 Å (ionization edge of n=1)


# ===========================================================================
# Thomson
# ===========================================================================
def kappa_thomson(n_e):
    """κ_es = σ_T × n_e [cm⁻¹]. Wavelength-independent."""
    return SIGMA_T * np.asarray(n_e)


# ===========================================================================
# Hydrogen bound-free
# ===========================================================================
def lambda_edge_H(n):
    """Photoionization threshold wavelength for H level n, in Å.

    λ_edge(n) = n² × λ_Ry = n² × 911.27 Å.
    Photons with λ < λ_edge can ionize from level n.
        n=1 →  911 Å (Lyman edge)
        n=2 → 3646 Å (Balmer edge)
        n=3 → 8204 Å (Paschen edge)
        n=4 → 14584 Å (Brackett edge)
    """
    return (n * n) * LAM_RYDBERG_AA


def sigma_bf_H(n, lam_AA):
    """H bound-free cross-section σ_bf(n, λ) [cm² per atom in level n].

    Kramers formula, no Gaunt factor (g_bf ≈ 1 to 30% accuracy across H).
        σ_bf(n, ν) = (64π⁴/3√3) × e¹⁰ m_e / (c h⁶) × 1/(n⁵ ν³)
        Numerically: 7.91e-18 × n × (ν_n/ν)³  [cm²]
    where ν_n = c / λ_edge(n).

    Returns 0 if λ > λ_edge(n).
    """
    lam_AA = np.atleast_1d(np.asarray(lam_AA, dtype=float))
    lam_edge = lambda_edge_H(n)
    out = np.zeros_like(lam_AA)
    mask = lam_AA < lam_edge
    if mask.any():
        # σ_bf scales as (λ/λ_edge)^3 below threshold
        out[mask] = 7.91e-18 * n * (lam_AA[mask] / lam_edge)**3
    if out.size == 1:
        return float(out[0])
    return out


def kappa_bf_H(lam_AA, n_HI_per_level, n_max=None):
    """Sum H bound-free opacity over all levels with λ_edge > λ.

    Parameters
    ----------
    lam_AA : scalar or (nlam,) array, wavelength in Å
    n_HI_per_level : (nlev, nzones) array, n_HI in level n+1 at zone z
                     (index 0 = ground state n=1, index 1 = n=2, etc.)
    n_max : maximum level to sum to. None = use all levels in n_HI_per_level.

    Returns
    -------
    kappa : (nzones,) or (nlam, nzones) array, cm⁻¹
    """
    n_HI_per_level = np.asarray(n_HI_per_level)
    if n_HI_per_level.ndim == 1:
        n_HI_per_level = n_HI_per_level[:, np.newaxis]
    nlev, nzones = n_HI_per_level.shape
    if n_max is None:
        n_max = nlev

    lam_AA_arr = np.atleast_1d(np.asarray(lam_AA, dtype=float))
    nlam = lam_AA_arr.size
    kappa = np.zeros((nlam, nzones))
    for level_idx in range(min(n_max, nlev)):
        n = level_idx + 1   # principal quantum number
        sigma = sigma_bf_H(n, lam_AA_arr)    # (nlam,)
        # outer product: (nlam,) × (nzones,) → (nlam, nzones)
        kappa += np.outer(sigma, n_HI_per_level[level_idx])

    if nlam == 1:
        return kappa[0]
    return kappa


# ===========================================================================
# H⁻ (negative hydrogen ion) bound-free
# ===========================================================================
def n_H_minus(T, n_HI_ground, n_e):
    """LTE Saha for H⁻ formation: H + e⁻ ⇌ H⁻.

    n_H⁻ / n_HI(n=1) = n_e × (h²/(2π m_e kT))^(3/2) × (1/4) × exp(χ_H- / kT)
    where χ_H- = 0.754 eV is the H⁻ binding energy and 1/4 is the partition-
    function ratio (g_H- = 1, g_HI(n=1) = 2, g_e = 2 → 1·1 / (2·2) = 1/4).

    All inputs are arrays of the same shape (broadcasting OK).
    """
    T = np.asarray(T, dtype=float)
    n_HI_ground = np.asarray(n_HI_ground, dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    chi_Hm = 0.754 * EV
    # thermal de Broglie volume (h²/(2π m_e kT))^(3/2)
    thermal_vol = (HPL * HPL / (2 * np.pi * ME * KB * T)) ** 1.5
    return n_HI_ground * n_e * thermal_vol * 0.25 * np.exp(chi_Hm / (KB * T))


def sigma_H_minus(lam_AA):
    """H⁻ bound-free cross-section [cm² per H⁻ ion].

    John (1988) polynomial fit. Valid 1250 Å < λ < 16400 Å.
    Peak σ ≈ 4×10⁻¹⁷ cm² near 8500 Å, falling off to either side.

    Outside the fit range, returns 0 (caller responsible for ranges).
    """
    lam_AA = np.atleast_1d(np.asarray(lam_AA, dtype=float))
    lam_um = lam_AA * 1e-4    # microns
    out = np.zeros_like(lam_AA)
    # John 1988 cubic fit for λ < 1.6443 μm
    valid = (lam_um > 0.125) & (lam_um < 1.6443)
    if valid.any():
        # σ in units of 1e-18 cm². Approximate fit:
        #   σ(λ) [cm²] = (lam_um - lam_0)² × (a + b×lam_um + c×lam_um² + d×lam_um³) × 1e-18
        # Coefficients from John 1988 (table 1, polynomial form for bound-free)
        lam_0 = 1.6419  # microns; threshold for H⁻ bf
        x = lam_0 - lam_um[valid]
        x = np.maximum(x, 0.0)
        # The form is approximately polynomial in λ; we use the 6-parameter fit:
        sigma_um = (152.519
                    + 49.534 * x
                    - 118.858 * x*x
                    + 92.536 * x**3
                    - 34.194 * x**4
                    + 4.982 * x**5)
        out[valid] = sigma_um * 1e-18 * x**1.5
        out[valid] = np.maximum(out[valid], 0.0)
    if out.size == 1:
        return float(out[0])
    return out


def kappa_H_minus(lam_AA, T, n_HI_ground, n_e):
    """Total H⁻ bound-free opacity [cm⁻¹], using LTE Saha for n_H⁻.

    Note: in NLTE plateau-phase atmospheres n_H⁻ may deviate from LTE Saha
    by factors O(1) due to non-LTE radiation field. We use LTE Saha here as
    a first-cut; Phase 2 NLTE refinement may improve this.
    """
    n_Hm = n_H_minus(T, n_HI_ground, n_e)
    sig = sigma_H_minus(lam_AA)
    sig = np.atleast_1d(sig)
    if sig.size == 1:
        return n_Hm * float(sig[0])
    return np.outer(sig, n_Hm)


# ===========================================================================
# Free-free (Bremsstrahlung absorption)
# ===========================================================================
def kappa_ff_H(lam_AA, T, n_e, n_p, gaunt=1.2):
    """Hydrogenic free-free absorption coefficient [cm⁻¹].

    Standard formula (Rybicki & Lightman eq. 5.18, corrected for stim emis):
       κ_ff = (4 e⁶ / (3 m_e h c)) × (2π / (3 m_e k))^(1/2)
              × Z² × n_e × n_ion × T^(-1/2) × ν^(-3)
              × (1 - exp(-hν/kT)) × g_ff
    For H, Z=1.
    """
    T = np.asarray(T, dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    n_p = np.asarray(n_p, dtype=float)
    lam_AA = np.atleast_1d(np.asarray(lam_AA, dtype=float))
    nu = C_LIGHT / (lam_AA * 1e-8)    # Hz

    # numerical prefactor 3.692e8 in cgs (Rybicki & Lightman)
    prefactor = 3.692e8
    nu_inv3 = 1.0 / nu**3
    # broadcast: nu (nlam,) × T (nzones,)
    stim = 1.0 - np.exp(-HPL * nu[:, None] / (KB * T[None, :]))
    kappa = (prefactor * gaunt
             * n_e[None, :] * n_p[None, :]
             * T[None, :]**(-0.5)
             * nu_inv3[:, None]
             * stim)
    if kappa.shape[0] == 1:
        return kappa[0]
    return kappa


# ===========================================================================
# Total
# ===========================================================================
def kappa_cont_total(lam_AA, T, n_e, n_p, n_HI_per_level,
                      X_H=None, include=('es', 'bf', 'H-', 'ff')):
    """Total continuum opacity κ_cont(λ) per zone [cm⁻¹].

    Parameters
    ----------
    lam_AA : scalar or array, wavelength(s) in Å
    T, n_e, n_p : (nzones,) per-zone temperature [K] and densities [cm⁻³]
    n_HI_per_level : (nlev, nzones), per-level HI populations [cm⁻³]
                     (level 0 = ground state n=1)
    X_H : unused here; reserved for future use (e.g. He ff additions)
    include : which channels to sum. Defaults to all four.

    Returns
    -------
    kappa : (nzones,) or (nlam, nzones), in cm⁻¹

    Notes
    -----
    Robust to missing-data: if n_HI_per_level[0] is zero (X_H=0 case, e.g.
    Ib/Ic ejecta), bf and H⁻ contributions vanish naturally, leaving only
    Thomson — which is correct for hydrogen-free ejecta.
    """
    lam_AA_arr = np.atleast_1d(np.asarray(lam_AA, dtype=float))
    nlam = lam_AA_arr.size
    nzones = np.asarray(T).size
    kappa = np.zeros((nlam, nzones))

    if 'es' in include:
        kappa += kappa_thomson(n_e)[None, :]
    if 'bf' in include:
        kbf = kappa_bf_H(lam_AA_arr, n_HI_per_level)
        if kbf.ndim == 1:
            kbf = kbf[None, :]
        kappa += kbf
    if 'H-' in include:
        # H⁻ only matters when there's neutral H ground-state population
        n_HI_ground = n_HI_per_level[0]
        kHm = kappa_H_minus(lam_AA_arr, T, n_HI_ground, n_e)
        if kHm.ndim == 1:
            kHm = kHm[None, :]
        kappa += kHm
    if 'ff' in include:
        kff = kappa_ff_H(lam_AA_arr, T, n_e, n_p)
        if kff.ndim == 1:
            kff = kff[None, :]
        kappa += kff

    if nlam == 1:
        return kappa[0]
    return kappa


def channel_breakdown(lam_AA, T, n_e, n_p, n_HI_per_level):
    """Return per-channel κ for diagnostics. Same args as kappa_cont_total.
    Returns dict keyed by channel name."""
    return {
        'es':  kappa_thomson(n_e),
        'bf':  np.atleast_1d(kappa_bf_H(lam_AA, n_HI_per_level))[0]
                if np.isscalar(lam_AA) or np.asarray(lam_AA).size == 1
                else kappa_bf_H(lam_AA, n_HI_per_level),
        'H-':  np.atleast_1d(kappa_H_minus(lam_AA, T,
                              n_HI_per_level[0], n_e))[0]
                if np.isscalar(lam_AA) or np.asarray(lam_AA).size == 1
                else kappa_H_minus(lam_AA, T, n_HI_per_level[0], n_e),
        'ff':  np.atleast_1d(kappa_ff_H(lam_AA, T, n_e, n_p))[0]
                if np.isscalar(lam_AA) or np.asarray(lam_AA).size == 1
                else kappa_ff_H(lam_AA, T, n_e, n_p),
    }
