"""
h_populations_nlte.py
=====================
Hydrogen level populations in three modes:

  1. 'lte'   — Saha ionization + Boltzmann distribution among bound levels.
               Populations only depend on (T, n_e). Physically valid in dense
               collisionally-dominated regions.

  2. 'caseb' — Case-B recombination with local Sobolev Lyα escape probability.
               Current default. 2-level-ish with n_3 separately in statistical
               balance with Hα + Hβ decays. Physically valid in thin nebular
               conditions where radiative transitions dominate.

  3. 'nlte'  — Multi-level statistical equilibrium with radiative + collisional
               terms and self-consistent Sobolev escape probabilities. Solves
               the linear rate matrix per zone, iterated until line optical
               depths converge. Works everywhere LTE and Case-B work, and
               handles the intermediate regime where neither is valid (dense
               but radiation-trapped, e.g. the IIn CSM shell).

All three return (n_2, n_3, diag) per zone in cgs with the same API, so you
can A/B compare by switching mode.

Physics references:
  Osterbrock & Ferland 2006 (Case-B rates, collisional cross sections)
  Hummer & Storey 1987 (effective Case-B recombination rates)
  Vriens & Smeets 1980 (H collisional excitation/de-excitation)
  Johnson 1972 (H collisional rates as polynomial fits)
"""
from __future__ import annotations
import numpy as np

# cgs constants
C      = 2.99792458e10
KB     = 1.380649e-16
ME     = 9.1093837e-28
EE     = 4.80320425e-10
HPL    = 6.62607015e-27
MH     = 1.6735e-24
EV     = 1.602176634e-12

CHI_H  = 13.5984       # H ionization potential [eV]

# Excitation energies from n=1, in eV
E_N = {1: 0.0, 2: 10.1988, 3: 12.0875, 4: 12.7485, 5: 13.0545}
# Statistical weights g = 2 n^2
G_N = {1: 2, 2: 8, 3: 18, 4: 32, 5: 50}

# Wavelengths [cm]
LAM = {
    (1, 2): 1215.67e-8,   # Lyα
    (1, 3): 1025.72e-8,   # Lyβ
    (1, 4):  972.54e-8,   # Lyγ
    (1, 5):  949.74e-8,
    (2, 3): 6562.80e-8,   # Hα
    (2, 4): 4861.35e-8,   # Hβ
    (2, 5): 4340.47e-8,   # Hγ
    (3, 4):18751.00e-8,   # Paα
    (3, 5):12821.58e-8,   # Paβ
    (4, 5):40512.00e-8,   # Brα
}

# Einstein A coefficients [s^-1] for upper -> lower
A_EINSTEIN = {
    (2, 1): 4.699e8,   # Lyα
    (3, 1): 5.575e7,   # Lyβ
    (4, 1): 1.278e7,
    (5, 1): 4.125e6,
    (3, 2): 4.410e7,   # Hα
    (4, 2): 8.420e6,   # Hβ
    (5, 2): 2.530e6,
    (4, 3): 8.986e6,   # Paα
    (5, 3): 2.201e6,
    (5, 4): 2.699e6,
}

# Oscillator strengths for lower -> upper
F_OSC = {
    (1, 2): 0.4162,
    (1, 3): 0.0791,
    (1, 4): 0.0290,
    (1, 5): 0.01394,
    (2, 3): 0.6407,
    (2, 4): 0.1193,
    (2, 5): 0.0447,
    (3, 4): 0.8420,
    (3, 5): 0.1506,
    (4, 5): 1.038,
}

# Case-B effective recombination rates to n=2, 3 [cm^3/s]
# (Hummer & Storey 1987 approximations)
def alpha_caseB_n(n: int, T: np.ndarray) -> np.ndarray:
    T4 = T / 1e4
    if n == 2:
        return 4.5e-14 * T4**-0.75
    if n == 3:
        return 1.5e-14 * T4**-0.80
    if n == 4:
        return 7.0e-15 * T4**-0.82
    if n == 5:
        return 3.6e-15 * T4**-0.82
    raise ValueError(f"n={n} not supported")


# ---------------------------------------------------------------------------
# Saha ionization
# ---------------------------------------------------------------------------
def saha_neutral_fraction(T, n_e):
    """Fraction of H atoms that are neutral, from Saha equation."""
    kT = KB * T
    prefac = 2.0 * (2.0 * np.pi * ME * kT / (HPL * HPL))**1.5
    K = prefac * np.exp(-CHI_H * EV / kT)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(n_e > 0, K / n_e, np.inf)
    return 1.0 / (1.0 + ratio)


def photoion_neutral_fraction(T_gas, n_e, r, R_phot, T_rad, L_phot=None,
                                n_H_total=None,
                                f_HI_max=0.5,
                                stromgren_exponent=1.0):
    """Neutral fraction of H from steady-state photoionization-recombination
    balance with outward Lyman-continuum-photon-budget tracking.

    Replaces gas-temperature Saha for the OUTER WIND. Local Saha is wrong
    in IIn winds because:
      (a) gas T is set by mechanical/thermal equilibrium (typ. 12000-15000 K)
      (b) ionizing photons originate at the photosphere (rate Q_H), and as
          they propagate outward they get absorbed by H I in their path
      (c) at some radius (the Strömgren radius R_S), the cumulative
          recombination rate inside that sphere equals Q_H — beyond that,
          there are no ionizing photons left and hydrogen rapidly recombines

    Strömgren transition broadening
    -------------------------------
    The default behavior (stromgren_exponent=1.0) gives a sharp transition
    in n_HI(r) at the Strömgren radius:  f_quench = 1/(1 + ratio_q).

    For stromgren_exponent < 1, the transition is broadened across more
    radii:  f_quench = 1/(1 + ratio_q^p).  This pushes n_HI inward toward
    the inner wind (where ratio_q >> 1), creating absorption at a wider
    range of velocities. Use p=0.3-0.5 to broaden; p=1.0 (default) keeps
    the original sharp Strömgren transition.

    Caveat: broadening creates n_HI in the inner WIND at HIGHER velocities,
    not lower velocities. For atm8-like profiles where the v(r) floor is
    in the outer wind, this broadens the absorption range bluewards. It
    does NOT shift the trough position to smaller |v|.

    Algorithm:
      1. Compute Q_H from L_phot and ionizing fraction of BB at T_rad.
      2. Walking outward from R_phot, accumulate recombination rate
         per zone: R_zone = α_B × n_e × n_p × dV.
      3. Track REMAINING ionizing photon flux Q_remaining = Q_H − Σ R_zone.
      4. If Q_remaining > 0: zone is in ionized regime, use Saha-at-T_gas
         (which gives a small f_HI).
      5. If Q_remaining < 0: zone is beyond Strömgren, hydrogen mostly
         neutral — set f_HI from photoionization rate ≈ 0:
            n_p × n_e × α_B = n_HI × Γ_pi → as Γ_pi → 0, n_HI → n_H_total

    Implementation: compute Saha first as a baseline (fully-ionized limit),
    then inside zones beyond R_S, raise f_HI sharply. The smooth transition
    comes from finite τ_LC attenuation of the leakage.

    Parameters
    ----------
    T_gas : per-zone gas temperature [K]
    n_e   : per-zone electron density [cm⁻³]
    r     : per-zone radius [cm]
    R_phot: photospheric radius [cm]
    T_rad : photospheric (radiation) temperature [K]
    L_phot: photospheric bolometric luminosity [erg/s]
    n_H_total: per-zone total H density [cm⁻³]

    Returns
    -------
    f_HI  : per-zone neutral fraction of hydrogen
    """
    T_gas = np.asarray(T_gas, dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    r = np.asarray(r, dtype=float)

    # Saha baseline (gives the "ionized" answer).
    f_HI_saha = saha_neutral_fraction(T_gas, n_e)

    if L_phot is None or n_H_total is None:
        return f_HI_saha   # not enough info to do photoion correction

    n_H_total = np.asarray(n_H_total, dtype=float)
    n_p_saha = (1.0 - f_HI_saha) * n_H_total

    # Ionizing photon rate from photosphere (BB at T_rad, λ < 912 Å).
    # Approximate fraction of σT⁴ below 912 Å for a BB at T_rad:
    # fit: f_ion ≈ exp(−15.79e4/T_rad) × 5 × (T_rad/1e4)^0.4 (rough; valid ~1e4-3e4 K)
    # More accurate: integrate Planck below 912 Å.
    HPL_loc = 6.626e-27; KB_loc = 1.381e-16; CC_loc = 2.998e10
    SIGMA_SB = 5.6704e-5
    lams_ion = np.linspace(100e-8, 912e-8, 200)
    x_ion = HPL_loc * CC_loc / (lams_ion * KB_loc * T_rad)
    B_ion = (2*HPL_loc*CC_loc**2 / lams_ion**5) / (np.exp(np.minimum(x_ion, 700)) - 1)
    B_ion_int = np.trapezoid(B_ion, lams_ion)        # erg/s/cm²/sr/cm
    L_ion = L_phot * (np.pi * B_ion_int) / (SIGMA_SB * T_rad**4)
    # Mean ionizing photon energy ≈ 1.5 × 13.6 eV ≈ 3.27e-11 erg (fairly weak T dep)
    mean_E_ion = 13.6 * 1.602e-12 * 1.5
    Q_H = L_ion / mean_E_ion

    # α_B at gas T
    alpha_B = 2.59e-13 * (T_gas / 1e4)**(-0.85)

    # Cumulative recombination outward from R_phot
    dr = np.empty_like(r)
    dr[:-1] = np.diff(r); dr[-1] = dr[-2]
    dV = 4.0 * np.pi * r**2 * dr
    R_per_zone = n_e * n_p_saha * alpha_B * dV
    R_per_zone[r <= R_phot] = 0.0
    R_cumul = np.cumsum(R_per_zone)

    # Q_remaining at each zone = Q_H − R_cumul (clipped at 0)
    Q_remaining = np.maximum(Q_H - R_cumul, 0.0)

    # Effective Γ_pi per neutral H atom: Q_remaining absorbed proportional
    # to local n_HI. Approximate: Γ_pi(r) ≈ Q_remaining × σ_LC / (4π r² × Δr_local)
    # — really this is "how many photons per second pass each H atom in the
    # zone", and is set by Q_remaining and n_HI in that zone.
    # 
    # Steady state: n_p × n_e × α_B = n_HI × Γ_pi
    # → n_HI / n_H_total = α_B × n_p × n_e × dV / (Q_remaining × X_zone)
    # where X_zone is the fraction of remaining photons absorbed in this zone.
    # 
    # Simpler: at the Strömgren transition, all incoming photons are absorbed
    # in a thin layer. Use the relation:
    #     n_HI ≈ n_p_saha × n_e × α_B × dV / Γ_pi_local
    # where Γ_pi_local = Q_remaining × σ_LC / V_zone (very approximate, but
    # captures the qualitative behavior).
    # 
    # For zones with Q_remaining > 0: use Saha (ionized).
    # For zones with Q_remaining → 0: f_HI → 1 (Strömgren beyond).
    # Smooth transition with Q_remaining / R_per_zone:
    # Steady-state photoion equilibrium: n_HI / n_p = α_B × n_e / Γ_pi
    # where Γ_pi is the local ionization rate. We don't compute Γ_pi
    # directly — instead we use the photon-budget framework:
    #   ratio_q = Q_remaining / R_per_zone
    # captures how strongly photon-starved the zone is.
    # 
    # Limit ratio_q >> 1 (photon-rich): f_HI → f_HI_saha  (very ionized)
    # Limit ratio_q << 1 (photon-starved): hydrogen recombines toward LTE
    # 
    # Smooth interpolation: f_HI = f_HI_saha + (1 - f_HI_saha) × f_quench
    # where f_quench = (1 + ratio_q)^(-1) → smooth transition at the
    # Strömgren radius. At ratio_q=1 (Strömgren boundary), f_quench=0.5.
    # At ratio_q→0, f_quench→1 (recombines toward neutral but capped at
    # the snapshot's n_e self-consistency: don't return f_HI=1 if n_e>0).
    # 
    # Cap f_HI such that resulting n_p > some floor consistent with the
    # given n_e. The snapshot's n_e is treated as a hard input — so n_p
    # should not exceed n_e (assuming hydrogen is the dominant electron
    # donor; for X_He=0.25 helium contributes additional electrons).
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_q = np.where(R_per_zone > 0, Q_remaining / R_per_zone, np.inf)
    # Strömgren transition with optional broadening exponent p:
    #   p = 1.0 (default): sharp transition at ratio_q = 1
    #   p < 1.0:           broader transition; n_HI extends inward toward
    #                      photoion-rich zones
    p = float(stromgren_exponent)
    if p == 1.0:
        f_quench = 1.0 / (1.0 + ratio_q)
    else:
        # Avoid 0**0 issues: clip very small ratio_q to a small positive value
        ratio_q_safe = np.maximum(ratio_q, 1e-30)
        f_quench = 1.0 / (1.0 + np.power(ratio_q_safe, p))
    # f_HI in [f_HI_saha, 1] interpolated by f_quench
    f_HI = f_HI_saha + (1.0 - f_HI_saha) * f_quench
    # Don't drive f_HI so high that n_p falls below what n_e requires.
    # n_p_max = n_e (if hydrogen alone provides electrons). For X_He
    # contribution, allow up to n_e - 0.5 × n_He × X_he_ion (rough).
    n_p_floor = np.maximum(n_e * 0.5, 0.0)   # conservative: half of n_e
    f_HI_cap_from_ne = 1.0 - n_p_floor / np.maximum(n_H_total, 1e-30)
    f_HI_cap_from_ne = np.clip(f_HI_cap_from_ne, 0.0, 1.0)
    # Residual-ionization floor: even fully Strömgren-shadowed regions have
    # SOME ionization from diffuse UV (recombination radiation, X-rays,
    # time-dependent effects) that prevents f_HI → 1. f_HI_max sets this
    # cap. Default 0.5 — reproduces moderate-trough physics for atm8-like
    # IIn winds. Set higher (closer to 1.0) for "thicker absorber" regimes,
    # lower for "more ionized outer wind" regimes.
    f_HI = np.minimum(f_HI, f_HI_cap_from_ne)
    f_HI = np.minimum(f_HI, f_HI_max)
    f_HI = np.clip(f_HI, 0.0, 1.0)
    return f_HI


# ---------------------------------------------------------------------------
# LTE: Saha + Boltzmann
# ---------------------------------------------------------------------------
def h_populations_lte(rho, T, n_e, X_H=0.737, nlev: int = 5):
    """LTE populations: Saha ionization + Boltzmann among bound levels.

    Returns n_2, n_3 [cm^-3], diag dict.
    """
    T   = np.asarray(T, dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    rho = np.asarray(rho, dtype=float)
    X_H = (np.asarray(X_H, dtype=float) if np.ndim(X_H)
           else np.full_like(rho, X_H))
    n_H_total = X_H * rho / MH
    f_HI = saha_neutral_fraction(T, n_e)
    n_HI = f_HI * n_H_total
    # Boltzmann distribution among n=1..nlev
    kT = KB * T
    weights = []
    for n in range(1, nlev + 1):
        w = G_N[n] * np.exp(-E_N[n] * EV / kT)
        weights.append(w)
    W = np.stack(weights)    # shape (nlev, nzones)
    Z = W.sum(axis=0)
    n_i = n_HI * W / Z       # shape (nlev, nzones)
    n_2 = n_i[1]
    n_3 = n_i[2]
    diag = dict(f_HI=f_HI, n_HI=n_HI, n_p=(1.0 - f_HI) * n_H_total,
                mode='lte', partition=Z)
    return n_2, n_3, diag


# ---------------------------------------------------------------------------
# Collisional rate coefficients [cm^3/s]
# ---------------------------------------------------------------------------
def _collisional_excitation(i: int, j: int, T: np.ndarray) -> np.ndarray:
    """Collisional excitation rate coefficient q_{i->j}(T), i < j.
    Uses effective collision strengths Γ_{ij} approximated from H NLTE tables
    (Vriens & Smeets 1980; Anderson et al 2000). Valid for T ~ 5e3 – 3e4 K.
    """
    # Effective collision strengths (weakly T-dependent; use T4=1 values)
    Gamma = {
        (1, 2): 0.27,    # Lyα
        (1, 3): 0.07,    # Lyβ
        (1, 4): 0.025,
        (1, 5): 0.012,
        (2, 3): 5.0,     # Hα (large optical coll. strength)
        (2, 4): 2.0,     # Hβ
        (2, 5): 0.7,
        (3, 4): 15.0,    # Paα
        (3, 5): 4.0,
        (4, 5): 30.0,
    }.get((i, j), 1.0)
    # Van Regemorter-style formula
    # q_{i->j} = 8.629e-6 * Γ / (g_i * sqrt(T)) * exp(-E_ij/kT)
    E_ij = (E_N[j] - E_N[i]) * EV
    kT = KB * T
    g_i = G_N[i]
    return 8.629e-6 * Gamma / (g_i * np.sqrt(T)) * np.exp(-E_ij / kT)


def _collisional_deexcitation(i: int, j: int, T: np.ndarray) -> np.ndarray:
    """Collisional de-excitation rate coefficient q_{j->i}(T), i < j.
    Detailed balance: q_{j->i} = (g_i/g_j) exp(+E_ij/kT) × q_{i->j}.
    Simplified from van Regemorter: q_{j->i} = 8.629e-6 × Γ / (g_j × √T).
    """
    Gamma = {
        (1, 2): 0.27, (1, 3): 0.07, (1, 4): 0.025, (1, 5): 0.012,
        (2, 3): 5.0,  (2, 4): 2.0,  (2, 5): 0.7,
        (3, 4): 15.0, (3, 5): 4.0,
        (4, 5): 30.0,
    }.get((i, j), 1.0)
    g_j = G_N[j]
    return 8.629e-6 * Gamma / (g_j * np.sqrt(T))


# ---------------------------------------------------------------------------
# NLTE: multi-level statistical equilibrium with Sobolev escape
# ---------------------------------------------------------------------------
def _multizone_beta_lya(n_lower_grid, r_grid, v_grid, T_grid,
                         lam_AA, f_lu, v_turb_cms,
                         resonance_widths: float = 4.0):
    """Non-local escape probability for an optically thick line, integrated
    over zones whose fluid-frame velocity is resonant with each emitting zone.

    For each emitter zone i, sum the line-center optical depth contributions
    from all outward zones j > i whose velocity offset |v_j - v_i| is within
    `resonance_widths` Doppler widths v_D. This captures the dominant
    physics for slow, near-monovelocity winds (where v(r) varies by
    much less than v_D over the whole wind), which is the regime where
    single-zone Sobolev systematically underestimates trapping.

    For fast-ejecta zones (where dv across one zone exceeds many v_D),
    only the local zone contributes — the result reduces smoothly to
    the standard Sobolev β = (1−exp(−τ))/τ.

    Parameters
    ----------
    n_lower_grid : per-zone density of the absorbing level [cm⁻³]
    r_grid       : radial coordinate [cm]
    v_grid       : radial velocity [cm/s]
    T_grid       : gas temperature [K] (sets thermal Doppler width)
    lam_AA       : line wavelength [Å]
    f_lu         : oscillator strength
    v_turb_cms   : microturbulent velocity [cm/s]

    Returns
    -------
    beta_per_zone : escape probability β = (1 − exp(−τ_eff)) / τ_eff
                    where τ_eff = sum of resonant Sobolev contributions.
    """
    nzones = len(r_grid)
    lam_cm = lam_AA * 1e-8
    sigma_int = np.pi * EE * EE / (ME * C) * f_lu        # cm² Hz prefactor

    # Per-zone Doppler width (thermal + turb)
    v_th = np.sqrt(KB * T_grid / MH)                      # cm/s
    v_D  = np.sqrt(v_th * v_th + v_turb_cms * v_turb_cms)  # cm/s

    # local Sobolev τ contribution from each zone
    # τ_local(j) = σ_int × λ × n_l(j) / |dv/dr|(j)
    # but for the multi-zone sum we use a simpler line-integrated form:
    # contribution from zone j seen by emitter i =
    #     σ_int × λ × n_l(j) × dr(j) × φ(Δν_ij) × c
    # where φ is the Voigt/Gaussian profile evaluated at the velocity offset
    # Δv_ij = v_j − v_i in the emitter's frame, and the σ × dr is per-zone
    # column density, multiplied by line-profile peak σ_0 = 1/(√π σ_D)
    dr = np.empty_like(r_grid)
    dr[:-1] = np.diff(r_grid); dr[-1] = dr[-2]

    beta_arr = np.zeros(nzones)
    for i in range(nzones):
        v_i = v_grid[i]
        v_D_i = v_D[i]
        # peak cross section at line center (Gaussian only)
        # σ_0 = σ_int / (√π × ν₀ × v_D_i / c) = σ_int × λ × c / (√π × v_D_i × c)
        # equivalently: σ_0 × n × dr = σ_int × λ × n × dr / (√π × v_D_i × λ / c)
        # = σ_int × c × n × dr / (√π × v_D_i)
        # Actually easier to keep the σ_int × λ form and divide by Doppler velocity:
        tau_eff = 0.0
        for j in range(i, nzones):
            # velocity offset of zone j as seen from zone i
            x = (v_grid[j] - v_i) / v_D_i
            if abs(x) > resonance_widths:
                continue
            # Voigt / Gaussian-only profile peak weight
            phi = np.exp(-x * x) / np.sqrt(np.pi)
            # contribution to τ_emitter from absorber zone j:
            # τ_j = σ_0 × n_l(j) × dr(j)
            # σ_0 = σ_int × λ × c / (v_D_i × λ × c × √π) = σ_int / (√π × v_D_i × λ_in_AA × ...)
            # Cleaner: σ_int has units cm² × Hz. Profile φ(ν) has 1/Hz.
            # Cross section at center: σ(ν₀) = σ_int × φ(0) = σ_int / (√π × σ_D_Hz)
            # σ_D_Hz = ν₀ × v_D / c = (c/λ) × v_D / c = v_D / λ
            # → σ(ν₀) = σ_int × λ / (√π × v_D)
            # → contribution = σ(ν₀) × φ_normalized × n_l × dr
            # where φ_normalized = exp(-x²) / √π is dimensionless, with peak at x=0
            sigma_at_x = sigma_int * lam_cm / (np.sqrt(np.pi) * v_D_i) * phi * np.sqrt(np.pi)
            # The √π × √π cancellation: σ(ν₀) already has 1/√π, and the phi includes √π normalization
            # Clean form: σ_eff = σ_int × λ × exp(-x²) / (√π × v_D_i)
            sigma_at_x = sigma_int * lam_cm * np.exp(-x*x) / (np.sqrt(np.pi) * v_D_i)
            tau_eff += sigma_at_x * max(n_lower_grid[j], 0.0) * dr[j]
        beta_arr[i] = ((1.0 - np.exp(-min(tau_eff, 700.0)))
                       / max(tau_eff, 1e-30))
    return beta_arr


def h_populations_nlte(rho, T, n_e, r, v, X_H=0.737,
                       nlev: int = 3, v_turb_cms: float = 20e5,
                       max_iter: int = 120, tol: float = 1e-3,
                       damping: float = 0.4,
                       multi_zone_lya: bool = True,
                       ionization_mode: str = 'saha',
                       R_phot: float = None,
                       T_phot: float = None,
                       L_phot: float = None,
                       f_HI_max: float = 0.5,
                       stromgren_exponent: float = 1.0,
                       # Phase 3 NLTE-RT coupling: optional radiative pumping
                       # on the Hα (2,3) transition.
                       #
                       # Preferred interface: J_bar_Ha_abs — an ABSOLUTE
                       # per-zone mean intensity at Hα line frequency, in
                       # CGS units erg/s/cm²/sr/Hz. This is the physically
                       # meaningful quantity that the MC directly measures
                       # via line-absorption counting (Phase 1/2 pipeline).
                       #
                       # Legacy interface: J_bar_factor + R_phot + T_phot.
                       # Computes J_bar_Ha = J_bar_factor × W(r) × B_ν(T_phot).
                       # Kept for backwards compat. NOTE: W × B_ν is the
                       # full diluted-photospheric-BB value, which typically
                       # OVERESTIMATES the true wind J_bar by 10–30× because
                       # the photospheric Hα line itself absorbs most of the
                       # photospheric continuum at line center. Use the _abs
                       # interface for physically correct results.
                       # External-ionization override: if provided, an array
                       # of length nzones giving f_HI = n_HI/n_H_total per zone
                       # from an upstream photoionization solver (CLOUDY-style).
                       # Bypasses the Saha / photoion / photoion_decoupled
                       # branches for the bottom-level neutral population.
                       # The NLTE rate equations still iterate level populations
                       # n_2..n_nlev using this n_HI as the ground-state input.
                       f_HI_external=None,
                       J_bar_Ha_abs=None,
                       J_bar_factor=None,
                       # Lyα destruction-probability floor applied to β_(1,2).
                       # The local/multi-zone Sobolev β can reach ~1e-10 in a
                       # slow uniform-velocity wind (every wind zone resonant
                       # with every other), driving b_2 ~ 1e4-1e5 above LTE
                       # and saturating τ_Sob(Hα) at ~1e6. Physically, real
                       # CSM has continuous Lyα destruction via FUV metal-line
                       # opacity (Fe II / Mg II / Si II UV multiplets resonantly
                       # absorbing Lyα photons), 2s 2-photon decay (A=8.2 s⁻¹),
                       # and continuum absorption. Effective ε ranges 1e-3 to
                       # 1e-5 for solar-metallicity CSM. We apply the floor as
                       #     β_eff(Lyα) = max(β_natural, ε)
                       # which both increases the apparent spontaneous-emission
                       # rate A_eff = A × β_eff (so n_2 destruction is faster)
                       # and reduces the indirect 1→2 pumping rate that the
                       # Sobolev approximation encodes implicitly. ε=None
                       # disables the floor (legacy behavior).
                       eps_Lya_destruction: float | None = None,
                       # Phase 2: physically-motivated Lyα destruction via 2s
                       # two-photon decay channel.
                       # When True, the effective n=2 → n=1 spontaneous-emission
                       # rate becomes
                       #     A_eff(2→1) = f_2p · A_Lyα · β_Lyα  +  f_2s · A_2γ
                       # where f_2p = 3/4, f_2s = 1/4 (statistical equilibrium
                       # between 2s and 2p, valid when 2s ↔ 2p l-mixing collision
                       # rate >> A_2γ = 8.23 s⁻¹, i.e. for n_e > ~10⁷ cm⁻³).
                       # This is the first-principles replacement for the
                       # eps_Lya_destruction knob: 2s decays via 2-photon emission
                       # to the continuum, bypassing the resonance trapping that
                       # makes β_Lyα tiny. In deeply-trapped zones (β_Lyα < 10⁻⁸)
                       # the 2γ channel dominates, giving A_eff ≈ A_2γ/4 ≈ 2 s⁻¹
                       # — a physical floor on the destruction rate that needs
                       # no parameter. In optically-thin zones (β_Lyα → 1), the
                       # 3/4 factor is the only effect (since 1/4 of n=2 atoms
                       # are in 2s and can't decay via Lyα even when thin).
                       # Default False = legacy behavior (pure Sobolev β).
                       # When combined with a finite eps_Lya_destruction, the
                       # ε floor is applied to β_natural BEFORE the 2γ term is
                       # added (allows ε to represent additional destruction
                       # channels like metal-line + continuum opacity).
                       two_photon_decay: bool = False,
                       verbose: bool = False):
    """Multi-level NLTE H populations with self-consistent escape probabilities.

    Solves rate equations including:
      - spontaneous emission A_ij with escape probability β_ij
      - collisional excitation/de-excitation
      - Case-B recombination from continuum as source terms
    Iterates until line optical depths (hence β_ij) stop changing.

    Parameters
    ----------
    nlev : number of bound levels to include (3, 4, or 5)
    v_turb_cms : microturbulent velocity [cm/s] for the dv/dr floor
    max_iter : maximum fixed-point iterations. The problem is non-linear
               (β depends on n_i, which depends on β), so we iterate.
    tol : convergence criterion on max fractional β_Ly change between
          consecutive iterations.
    damping : under-relaxation factor in [0, 1]. Population update is
              n_new = damping × n_solved + (1 - damping) × n_old.
              0.4 gives monotonic convergence in ~20 iter on the test IIn
              snapshot. Smaller = more stable, slower.
    multi_zone_lya : if True, compute β_Lyα non-locally by integrating the
              line-center optical depth over outward-resonant zones (i.e.
              zones whose fluid-frame velocity sits within ±v_D of the
              emitting zone). Default True. Only applied to the (1,2)
              transition; other lines use single-zone Sobolev because (a)
              they're not population-controlling for IIn and (b) the
              Sobolev approximation is well-justified for them in fast
              ejecta.

    Returns (n_2, n_3, diag).
    """
    T   = np.asarray(T, dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    rho = np.asarray(rho, dtype=float)
    r   = np.asarray(r, dtype=float)
    v   = np.asarray(v, dtype=float)
    X_H = (np.asarray(X_H, dtype=float) if np.ndim(X_H)
           else np.full_like(r, X_H))
    nzones = len(r)

    # Auto-reduce under-relaxation damping for higher level counts. The
    # default damping=0.4 is calibrated for nlev=3 and was observed to
    # cause divergent limit-cycles for nlev=5 (β_Ly oscillates between
    # 0.4 and 0.67, solver never converges within max_iter). With more
    # levels, the rate matrix couplings are stiffer (more transitions
    # share populations), so smaller damping is needed for monotonic
    # convergence. Empirical scaling: damping / (nlev / 3) gives stable
    # convergence in <30 iter for nlev=5 with test IIn snapshot.
    if nlev > 3:
        damping = min(damping, 0.2)

    # Ionization balance.
    # 'saha'   : standard local-T Saha. Correct in the inner shock-heated
    #            shell where the radiation field is roughly thermal.
    #            Underpredicts neutral fraction in the outer wind because
    #            the radiation field there is the DILUTED photospheric BB,
    #            not a local Planckian. Underpredicts by 10⁴-10⁶× for
    #            typical IIn outer-wind dilution factors.
    # 'photoion': photoionization-recombination equilibrium with diluted
    #            photospheric radiation. Required to reproduce the narrow
    #            blue-side absorption trough seen in CMFGEN spectra (the
    #            kinematic absorption from slow-CSM Hα n=2 absorbers).
    #            Requires R_phot and T_phot.
    n_H_total = X_H * rho / MH
    if f_HI_external is not None:
        # Upstream photoionization solver result: use directly. Skip the
        # Saha / photoion / photoion_decoupled internal logic for the
        # bottom-level (ground-state) population.
        f_HI = np.asarray(f_HI_external, dtype=float)
        if f_HI.shape != n_H_total.shape:
            raise ValueError(
                f"f_HI_external shape {f_HI.shape} does not match "
                f"n_zones={n_H_total.shape}")
        f_HI = np.clip(f_HI, 0.0, 1.0)
    elif ionization_mode == 'photoion':
        if R_phot is None or T_phot is None:
            raise ValueError(
                "ionization_mode='photoion' requires R_phot and T_phot")
        # Use the snapshot's actual L at R_phot if it's been threaded in
        # (kwarg L_phot passed to this function). Otherwise fall back to
        # 4πR²σT⁴ which over-estimates by an order of magnitude in IIn-
        # like atmospheres where the photospheric BB is heavily attenuated.
        # Caller should set L_phot to state.L_phot for the right answer.
        L_phot_use = L_phot if L_phot is not None else (
            4.0 * np.pi * R_phot**2 * 5.6704e-5 * T_phot**4)
        f_HI = photoion_neutral_fraction(
            T, n_e, r, R_phot, T_phot,
            L_phot=L_phot_use, n_H_total=n_H_total,
            f_HI_max=f_HI_max,
            stromgren_exponent=stromgren_exponent)
    elif ionization_mode == 'saha':
        f_HI = saha_neutral_fraction(T, n_e)
    elif ionization_mode == 'photoion_decoupled':
        # Use Saha for the populations the rate-equation solver iterates on
        # (correct for emission). Compute photoion n_HI separately and
        # propagate it via diag['n_HI_absorb'] for use as line absorption
        # opacity along peel-to-observer paths in the MC kernel.
        if R_phot is None or T_phot is None:
            raise ValueError(
                "ionization_mode='photoion_decoupled' requires R_phot and T_phot")
        L_phot_use = L_phot if L_phot is not None else (
            4.0 * np.pi * R_phot**2 * 5.6704e-5 * T_phot**4)
        f_HI = saha_neutral_fraction(T, n_e)
        f_HI_absorb = photoion_neutral_fraction(
            T, n_e, r, R_phot, T_phot,
            L_phot=L_phot_use, n_H_total=n_H_total,
            f_HI_max=f_HI_max,
            stromgren_exponent=stromgren_exponent)
    else:
        raise ValueError(
            f"ionization_mode must be 'saha', 'photoion', or "
            f"'photoion_decoupled'; got {ionization_mode!r}")
    n_HI = f_HI * n_H_total
    n_p  = (1.0 - f_HI) * n_H_total

    # velocity gradient with floor — robust to DUPLICATE radii (STELLA shock
    # zones), which make a raw np.gradient(v,r) NaN and would poison the whole
    # NLTE solve (see velocity_grad).
    from velocity_grad import robust_dvdr
    dv_dr = robust_dvdr(v, r, v_turb_cms=v_turb_cms)

    # initial guess: start from LTE ratios, scaled to n_HI
    kT = KB * T
    weights = np.stack([G_N[n] * np.exp(-E_N[n] * EV / kT)
                        for n in range(1, nlev + 1)])
    Z0 = weights.sum(axis=0)
    n_i = n_HI[None, :] * weights / Z0[None, :]    # shape (nlev, nzones)

    # precompute radiative & collisional rates (functions of T)
    A  = {(u, l): A_EINSTEIN[(u, l)] for (u, l) in A_EINSTEIN if u <= nlev and l <= nlev}
    # Line absorption cross-section prefactors σ_{lu} × λ_{lu}
    sigma_lam = {}
    for l in range(1, nlev):
        for u in range(l + 1, nlev + 1):
            sigma_lam[(l, u)] = (np.pi * EE * EE / (ME * C)
                                 * F_OSC[(l, u)] * LAM[(l, u)])

    # collisional rate coefficients (nlev x nlev matrix per zone)
    C_exc = {}   # from low i to high j
    C_dex = {}   # from high j to low i
    for l in range(1, nlev):
        for u in range(l + 1, nlev + 1):
            C_exc[(l, u)] = _collisional_excitation(l, u, T)
            C_dex[(l, u)] = _collisional_deexcitation(l, u, T)

    # recombination source term (Case-B effective) to each level
    S = np.zeros((nlev, nzones))
    for n in range(2, nlev + 1):
        S[n - 1] = n_e * n_p * alpha_caseB_n(n, T)
    # S[0] = 0 (Case-B: all Lyα-forming recombinations go back to ground,
    # effectively accounted for by the Lyα β_Ly mechanism)

    # Phase 3: optional radiative pumping on Hα (2,3) transition.
    # Two entry points:
    #   (a) J_bar_Ha_abs: per-zone absolute mean intensity [erg/s/cm²/sr/Hz].
    #       Preferred — direct from MC-measured line-absorption counts.
    #   (b) J_bar_factor: dimensionless multiplier of W(r) × B_ν(T_phot).
    #       Legacy/diagnostic; typically over-pumps by 10–30× because the
    #       photospheric Hα line self-absorption is not accounted for.
    # R_23_rad and R_32_rad are then constant per zone through the iteration
    # (they don't depend on n_i). When both are None, the rates are zero
    # and the solver behaves identically to before.
    R_23_rad = np.zeros(nzones)
    R_32_rad = np.zeros(nzones)
    if nlev >= 3:
        J_bar_Ha = None
        if J_bar_Ha_abs is not None:
            J_bar_Ha = np.asarray(J_bar_Ha_abs, dtype=float)
            if J_bar_Ha.shape != (nzones,):
                raise ValueError(
                    f"J_bar_Ha_abs shape {J_bar_Ha.shape} does not match "
                    f"nzones={nzones}")
        elif J_bar_factor is not None:
            if R_phot is None or T_phot is None:
                raise ValueError("J_bar_factor requires R_phot and T_phot")
            f_J_arr = np.asarray(J_bar_factor, dtype=float)
            if f_J_arr.shape != (nzones,):
                raise ValueError(
                    f"J_bar_factor shape {f_J_arr.shape} does not match "
                    f"nzones={nzones}")
            # Dilution factor W(r)
            W_r = np.where(
                r >= R_phot,
                0.5 * (1.0 - np.sqrt(np.maximum(
                    1.0 - (R_phot / np.maximum(r, 1e-30))**2, 0.0))),
                0.5,
            )
            # Planck function at Hα and T_phot
            nu_Ha = C / LAM[(2, 3)]
            x_Ha = HPL * nu_Ha / (KB * max(float(T_phot), 100.0))
            B_Ha_phot = ((2.0 * HPL * nu_Ha**3 / (C * C))
                         / (np.exp(min(x_Ha, 700.0)) - 1.0))
            J_bar_Ha = f_J_arr * W_r * B_Ha_phot

        if J_bar_Ha is not None:
            # Einstein coefficients for Hα
            nu_Ha = C / LAM[(2, 3)]
            A_32 = A_EINSTEIN[(3, 2)]
            B_32_E = A_32 * C * C / (2.0 * HPL * nu_Ha**3)
            B_23_E = (G_N[3] / G_N[2]) * B_32_E
            # Radiative excitation (upward) and stimulated emission (downward)
            R_23_rad = B_23_E * J_bar_Ha
            R_32_rad = B_32_E * J_bar_Ha
            if verbose:
                pos = J_bar_Ha[J_bar_Ha > 0]
                mode_str = ('absolute' if J_bar_Ha_abs is not None
                            else 'factor × W × B_ν')
                print(f"  [nlte-RT] Hα pump enabled ({mode_str}). "
                      f"J_bar_Ha range: "
                      f"{pos.min() if pos.size else 0:.3e} to "
                      f"{J_bar_Ha.max():.3e} erg/s/cm²/sr/Hz")
                if R_phot is not None:
                    wind_mask_d = r > R_phot
                    if wind_mask_d.sum() > 0:
                        print(f"  [nlte-RT] R_23_rad wind median: "
                              f"{np.median(R_23_rad[wind_mask_d]):.3e} s^-1")

    beta_ij_hist = {}    # track previous β for convergence

    for it in range(max_iter):
        # compute Sobolev optical depths and escape probabilities from current n_i
        beta = {}
        for l in range(1, nlev):
            for u in range(l + 1, nlev + 1):
                n_l = n_i[l - 1]
                n_u = n_i[u - 1]
                g_l = G_N[l]; g_u = G_N[u]
                pop_factor = np.maximum(n_l - (g_l / g_u) * n_u, 0.0)
                tau = sigma_lam[(l, u)] * pop_factor / dv_dr
                beta[(l, u)] = ((1.0 - np.exp(-np.minimum(tau, 700.0)))
                                / np.maximum(tau, 1e-30))

        # Replace local Sobolev β_Lyα with non-local multi-zone treatment
        # if requested. Lyα is the only line where this matters because
        # (a) it's the dominant trapping line controlling n=2 populations
        # via stimulated absorption / spontaneous decay, and (b) the IIn
        # wind is slow, so multi-zone resonance is typical.
        if multi_zone_lya:
            n_l = n_i[0]              # n=1 = HI ground state
            beta[(1, 2)] = _multizone_beta_lya(
                n_l, r, v, T, lam_AA=LAM[(1, 2)] * 1e8,
                f_lu=F_OSC[(1, 2)], v_turb_cms=v_turb_cms,
            )

        # Apply Lyα destruction-probability floor (Solution 1).
        # See parameter docstring for the physical motivation. Without this
        # floor, β_Lyα drops to ~3e-10 in slow-wind zones, n_2 climbs to
        # 1e4-1e5 × LTE Boltzmann, and τ_Sob(Hα) saturates at ~1e6 across
        # the wind, producing unphysically deep P-Cygni absorption that
        # doesn't match IIn observations. A floor at 1e-3 reduces the
        # n_2 over-population by 3-4 orders of magnitude (β-floor ratio),
        # bringing τ_Sob(Hα) into the 10-100 regime where Hα forms as a
        # narrow emission core with Thomson-scattered wings.
        if eps_Lya_destruction is not None:
            beta[(1, 2)] = np.maximum(beta[(1, 2)],
                                      float(eps_Lya_destruction))

        # Phase 2: two-photon decay channel from H(2s).
        # See parameter docstring for derivation. The effective β that, when
        # multiplied by A_Lyα = A[(2,1)], reproduces the total n=2 → n=1
        # downward rate (Lyα-escape + 2γ) is
        #     β_eff = 0.75 · β_natural  +  0.25 · (A_2γ / A_Lyα)
        # where A_2γ / A_Lyα = 8.23 / 4.699e8 ≈ 1.75e-8.
        # In the deeply-trapped limit (β_natural → 0), β_eff → 4.4e-9, which
        # gives A_eff → A_2γ/4 = 2.06 s⁻¹ — the "2γ floor". In the optically-
        # thin limit (β_natural → 1), β_eff → 0.75 (the 2s population can't
        # decay via Lyα, only 2p can — so 3/4 of n=2 sees the natural β).
        if two_photon_decay:
            A_2gamma_per_atom = 8.23  # s⁻¹, H(2s → 1s) 2-photon decay rate
            A_Lya = A[(2, 1)]          # = 4.699e8 s⁻¹
            beta_natural = beta[(1, 2)]
            beta_eff = 0.75 * beta_natural + 0.25 * (A_2gamma_per_atom / A_Lya)
            beta[(1, 2)] = beta_eff

        # build rate matrix per zone and solve
        # for each zone, 3-eqn system: Σ (n_j R_{j→i} - n_i R_{i→j}) + S_i = 0 for i≥2
        # plus Σ n_i = n_HI
        new_n_i = np.zeros_like(n_i)
        for z in range(nzones):
            # build nlev x nlev rate matrix M where
            #   M[i,i] = -Σ_{j≠i} R_{i→j}
            #   M[i,j] =  R_{j→i}                     (j≠i)
            # then M · n = -S
            M = np.zeros((nlev, nlev))
            for i in range(1, nlev + 1):
                for j in range(1, nlev + 1):
                    if i == j:
                        continue
                    # rate from level i -> j
                    if i < j:
                        # excitation: collisional + (Hα only) radiative
                        R_ij = C_exc[(i, j)][z] * n_e[z]
                        if i == 2 and j == 3:
                            R_ij += R_23_rad[z]
                    else:
                        # de-excitation: spontaneous with escape prob,
                        # + collisional, + (Hα only) stimulated emission
                        R_ij = (A[(i, j)] * beta[(j, i)][z]
                                + C_dex[(j, i)][z] * n_e[z])
                        if i == 3 and j == 2:
                            R_ij += R_32_rad[z]
                    # M row is for level (i-1), column for origin level (j-1)
                    M[i - 1, i - 1] -= R_ij    # loss from i
                    M[j - 1, i - 1] += R_ij    # gain into j from i
            # replace the first equation with normalization Σ n = n_HI(z)
            M[0, :] = 1.0
            rhs = -S[:, z].copy()
            rhs[0] = n_HI[z]
            try:
                sol = np.linalg.solve(M, rhs)
                sol = np.maximum(sol, 0.0)
            except np.linalg.LinAlgError:
                sol = n_i[:, z]   # keep previous if singular
            new_n_i[:, z] = sol

        # convergence check on β_Ly (most sensitive quantity)
        changed = 0.0
        if (1, 2) in beta and (1, 2) in beta_ij_hist:
            changed = np.max(np.abs(beta[(1, 2)] - beta_ij_hist[(1, 2)])
                             / np.maximum(beta_ij_hist[(1, 2)], 1e-10))
        beta_ij_hist = beta
        # under-relaxed update for stability
        n_i = damping * new_n_i + (1.0 - damping) * n_i

        if verbose:
            print(f"  nlte iter {it:2d}: max β_Ly change = {changed:.3e}")
        if it > 0 and changed < tol:
            if verbose:
                print(f"  nlte converged at iter {it} (tol={tol:.1e})")
            break
    else:
        if verbose:
            print(f"  nlte WARNING: did not converge after {max_iter} iter "
                  f"(last change {changed:.3e}, tol {tol:.1e})")

    n_2 = n_i[1]
    n_3 = n_i[2] if nlev >= 3 else np.zeros_like(n_2)

    # diagnostics
    tau_Ly = sigma_lam[(1, 2)] * np.maximum(
        n_i[0] - (G_N[1] / G_N[2]) * n_i[1], 0) / dv_dr
    diag = dict(
        mode='nlte', f_HI=f_HI, n_HI=n_HI, n_p=n_p,
        n_levels=n_i, tau_Ly=tau_Ly, beta_Ly=beta[(1, 2)],
        tau_Ly_median=float(np.median(tau_Ly[tau_Ly > 0]))
                       if np.any(tau_Ly > 0) else np.nan,
        iterations=it + 1,
    )

    # Phase 2 diagnostic: classify each zone by which channel dominates
    # the n=2 → n=1 destruction. Compare the two contributions:
    #   rate_Lya = 0.75 × A_Lya × β_natural      (Lyα-escape channel)
    #   rate_2γ  = 0.25 × A_2γ                    (2-photon channel)
    # In trapped zones rate_2γ dominates; in thin zones rate_Lya dominates.
    if two_photon_decay:
        A_2gamma_per_atom = 8.23
        A_Lya = A[(2, 1)]
        # Recover the "natural" β from the modified β_eff (inverse of the
        # transformation applied above), to give an honest classification.
        beta_eff = beta[(1, 2)]
        beta_natural_recovered = np.maximum(
            (beta_eff - 0.25 * A_2gamma_per_atom / A_Lya) / 0.75, 0.0)
        rate_Lya = 0.75 * A_Lya * beta_natural_recovered
        rate_2g  = 0.25 * A_2gamma_per_atom
        diag['phase2_two_photon_decay'] = True
        diag['phase2_rate_Lya_per_atom']  = rate_Lya         # s⁻¹, per zone
        diag['phase2_rate_2gamma_per_atom'] = (rate_2g
                                                * np.ones_like(beta_eff))
        diag['phase2_dominant_channel'] = np.where(
            rate_2g > rate_Lya, '2gamma', 'Lya')   # per-zone label
        diag['phase2_A_eff_per_atom'] = rate_Lya + rate_2g    # total rate
        diag['phase2_frac_two_gamma_dominated'] = float(
            np.mean(rate_2g > rate_Lya))
    else:
        diag['phase2_two_photon_decay'] = False

    # Photoion-decoupled mode: also expose an absorption-only n_2 array
    # that scales the Saha-based n_2 by (n_HI_absorb / n_HI_saha) in zones
    # where the photoion correction says hydrogen is more neutral.
    # Used by the MC kernel for line opacity along peel-to-observer paths
    # AND along random-walk-step line opacity tau-marching, but NOT for
    # emissivity (which uses the Saha-based n_2 / n_p / n_e via the
    # caseB α_eff formula). Result: emission comes from the inner ionized
    # shell (correct, Saha) and absorption happens in the outer wind
    # where slow CSM material has elevated n_2 (correct, photoion).
    if ionization_mode == 'photoion_decoupled':
        n_HI_absorb = f_HI_absorb * n_H_total
        # Scale factor: ratio of n_HI_absorb to n_HI_Saha. In zones where
        # photoion says more neutral (outer wind), this is > 1 and elevates
        # n_2 there. Inner zones where Saha and photoion agree, ratio ≈ 1.
        n_HI_safe = np.maximum(n_HI, 1e-30)
        scale = n_HI_absorb / n_HI_safe
        # Don't reduce n_2 below Saha (preserve inner-shell physics).
        scale = np.maximum(scale, 1.0)
        n_2_absorb = n_2 * scale
        diag['n_2_absorb'] = n_2_absorb
        diag['f_HI_absorb'] = f_HI_absorb
        diag['n_HI_absorb'] = n_HI_absorb
        diag['ionization_scale'] = scale

    return n_2, n_3, diag
