"""
photoionize_csm.py
===================
Per-zone photoionization equilibrium for SN snapshots.

What this fixes
---------------
STELLA's grey-diffusion RT does not properly transport ionizing
photons. As a result, its outer-CSM ionization state (X_HII ≈ 0.02-0.7
across the snapshot, dropping to ~0 at large r) does not reflect the
true photoionization equilibrium that should hold in a SN-illuminated
CSM. This module recomputes ionization properly, per zone, given the
snapshot's hydro structure.

Physics
-------
For each zone, solve the photoionization-recombination balance:

    n_HI(r) × Γ_HI(r) = n_HII(r) × n_e(r) × α_B(T_e)            (*)

where:

    Γ_HI(r) = ∫_{ν₀}^∞ (4π J_ν(r) / hν) σ_ν(ν) dν              [s⁻¹ per H atom]

    J_ν(r)  = B_ν(T_source) × W(r) × exp(-τ_ν(r))              [mean intensity]

    W(r)    = ½(1 - √(1 - (R_phot/r)²))                          [geometric dilution]
    W(R_phot) = ½  (radiation emerges into outward hemisphere only)

    τ_ν(r)  = τ_ν₀(r) × (ν₀/ν)³                                 [H bf opacity shape]
    τ_ν₀(r) = ∫_{R_phot}^r σ₀ n_HI(r') dr'                      [column to source]

    σ_ν(ν)  = σ₀ × (ν₀/ν)³,  σ₀ = 6.3e-18 cm²
              [Kramers, accurate to ~10% relative to Verner-Yakovlev]

    α_B(T)  = 2.59e-13 × (T/10⁴)^(-0.83) cm³/s
              [Hummer 1994; case B = on-the-spot, ignores LyC re-emission]

Charge neutrality (H only for now):
    n_e(r) = n_HII(r) = (1 - X_HI(r)) × n_H_total(r)

The quadratic for X_HI per zone (given n_H, T, Γ):
    Let q = n_H × α_B(T) / Γ
    Equation (*) becomes:   X_HII² × q + X_HII - 1 = 0
    Solution:               X_HII = [-1 + √(1 + 4q)] / (2q)
        q → 0    →  X_HII → 1     (well-ionized)
        q → ∞    →  X_HII → 1/√q (Saha-like, mostly neutral)

Iteration is required because τ_ν(r) depends on n_HI which depends
on X_HII. Convergence is fast (5-15 iterations) when the structure
is either fully ionized (transparent) or strongly self-shielding
(sharp I-front); slower when poised near a transition.

Source-spectrum convention
--------------------------
The radiation source is the photosphere of the snapshot. The
emergent spectrum is taken as a diluted blackbody at T_source, where
the default T_source is T_phot from the snapshot (STELLA's gas T at
τ_es = 2/3). This is the strict input-model treatment.

For cool SN photospheres (T_phot ~ 5000 K) the BB Lyman-continuum
flux is exponentially suppressed (exp(-h ν₀/kT) ≈ 10⁻¹³ at T=5300 K),
so this gives nearly-zero photoionization. To capture the harder
emergent spectrum that emerges from a Thomson-scattering photosphere
(where photons can escape from deeper hot layers, giving an effective
color temperature T_color > T_phot), the user can override T_source.
Set T_color_override > T_phot to test the sensitivity of ionization
to spectral hardening.

The L_phot of the source is matched to the snapshot's L_phot by
scaling the BB normalization. If T_source ≠ T_phot, the BB
prescription is "B_ν(T_source) × (L_phot / 4π R² σ T_source⁴)"
which preserves the total bolometric flux at R_phot but redistributes
it according to T_source.

What this module does NOT do (yet)
----------------------------------
- He ionization (set X_HeII = 0; small correction for cool photospheres,
  could matter for T_color > 30000 K).
- Temperature self-consistency: T_e is taken from the snapshot.
  CLOUDY/CMFGEN iterate T_e from heating-cooling balance; we don't.
  Pass T_eq_floor to enforce a minimum T (e.g., 10⁴ K) in photoionized
  zones, since real photoionization equilibrium gives T ~ 10⁴ K.
- Frequency-dependent attenuation by other species (He I/II, metals).
- Non-radial geometry / line transfer / continuum lowering.

Usage
-----
    from photoionize_csm import solve_photoionization_equilibrium
    snap = load_stella_snapshot(...)
    snap = truncate_to_photosphere(snap)  # IMPORTANT: photosphere defines source
    snap_pi = solve_photoionization_equilibrium(snap, verbose=True)
    # snap_pi has updated n_e, X_HII, n_HI, tau_es, optionally T

References
----------
Osterbrock & Ferland, "Astrophysics of Gaseous Nebulae and AGN" (2006), Ch. 2-3.
Hummer 1994, MNRAS 268, 109.
Verner et al. 1996, ApJ 465, 487.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import quad

# CGS constants
KB     = 1.380649e-16
HPL    = 6.62607015e-27
C      = 2.99792458e10
SIGMAT = 6.65245871e-25
MH     = 1.6735575e-24
EV     = 1.602176634e-12
SIGMA_SB = 5.670374e-5

# Hydrogen photoionization
CHI_H_eV = 13.59844     # H I ionization energy (eV)
CHI_H    = CHI_H_eV * EV
NU_H     = CHI_H / HPL                  # Lyman edge frequency [Hz]
SIGMA_H0 = 6.3e-18                       # H bf cross-section at threshold [cm²]
T_LYC    = CHI_H / KB                    # = 1.578e5 K, hν₀/k


# ---------------------------------------------------------------------------
# Photoionization-rate integration
# ---------------------------------------------------------------------------

def _G_function(T_source: float, tau_thresh: float | np.ndarray) -> np.ndarray:
    """Integral over frequency:

      G(T, τ) = ∫_{ν₀}^∞ (B_ν(T) / hν) × σ_ν × exp(-τ × (ν₀/ν)³) dν

    Then Γ_ion(r) = 4π × W(r) × G(T_source, τ_thresh(r)).

    Vectorized over tau_thresh. T_source is scalar.

    Implementation: change variable to x = hν/(kT). The integrand becomes
        (2π × (KB × T)² × x² / (h² c²)) × (σ₀ × (ν₀/ν)³) × exp(-τ × (ν₀/ν)³) / (exp(x) - 1) × (KB T / h)
    Combined: with y = ν₀/ν = T_LYC/(T × x),
        integrand_x = (2 σ₀ KB³ T³ × y³ / h³ c² × x²) × exp(-τ y³) / (exp(x) - 1)
        = const(T) × x² y³ exp(-τ y³) / (exp(x) - 1)

    where const(T) = 2 σ₀ (KB T)³ / (h³ c²).
    Actually let me just keep things in (ν, B_ν, σ_ν) form and
    let scipy.quad handle it. Vectorize over tau by looping.
    """
    tau_arr = np.atleast_1d(tau_thresh).astype(float)
    out = np.zeros_like(tau_arr, dtype=float)

    x_thresh = T_LYC / T_source   # x = hν/kT at threshold

    # If T is very low, x_thresh is huge and the Wien-tail factor exp(-x_thresh)
    # is effectively zero. Avoid wasted quadrature.
    if x_thresh > 700:
        return out * 0.0 if tau_thresh is np.ndarray else 0.0

    x_max = max(x_thresh + 40.0, 80.0)

    for i, tau in enumerate(tau_arr):
        def integrand(x):
            if x > 700:
                return 0.0
            nu = x * KB * T_source / HPL
            # B_ν = 2hν³/c² / (exp(x) - 1)
            B_nu = (2.0 * HPL * nu**3 / (C * C)) / (np.exp(x) - 1.0)
            # ν₀/ν = T_LYC / (T_source × x)
            y = T_LYC / (T_source * x)
            sigma_nu = SIGMA_H0 * y**3
            tau_nu = tau * y**3
            if tau_nu > 50.0:
                return 0.0
            # dν/dx = kT/h
            return (B_nu / (HPL * nu)) * sigma_nu * np.exp(-tau_nu) * (KB * T_source / HPL)

        # Robust integration: split at x = x_thresh + a few to handle the Wien tail
        try:
            val, _ = quad(integrand, x_thresh, x_max, limit=200, epsrel=1e-5)
        except Exception:
            val = 0.0
        out[i] = val

    if np.isscalar(tau_thresh):
        return float(out[0])
    return out


def _alpha_B(T: np.ndarray) -> np.ndarray:
    """Case-B recombination coefficient for H, in cm³/s (Hummer 1994 fit)."""
    T_safe = np.maximum(T, 100.0)
    return 2.59e-13 * (T_safe / 1.0e4)**(-0.83)


# ---------------------------------------------------------------------------
# Shock-bremsstrahlung physics
# ---------------------------------------------------------------------------

def derive_shock_params(snap_full: dict, verbose: bool = False,
                        tau_phot_ref: float = 2.0 / 3.0) -> dict | None:
    """Derive forward-shock parameters from a full (un-truncated) snapshot.

    Uses snapshot hydro (v, ρ, r) to identify the forward-shock location and
    compute the shock dissipation rate and characteristic post-shock
    temperature. The physics is the standard radiative-shock-in-CSM picture
    (Chevalier & Fransson 1994, 2017; Fransson & Björnsson 1998):

        T_shock     = (3/16) × μ m_p k_B⁻¹ × v_s²       [Rankine-Hugoniot]
        L_shock     = 4π R_s² × ½ × ρ_csm × v_s³          [shock dissipation]
        η_rad       = 1 / (1 + t_cool / t_dyn)             [radiative efficiency]
        t_cool      = (3/2) k_B T_shock / (n_e,ps × Λ(T)) [post-shock cooling]
        L_X_brems   = η_rad × L_shock × exp(-hν₀/kT_shock) [ionizing fraction]

    Shock-location heuristic: position of maximum |dv/dr| (the CDS-CSM
    velocity discontinuity). In the radiative-shock limit (high compression),
    the forward-shock front velocity ≈ post-shock CDS velocity, so v_s is
    evaluated at this location.

    Pre-shock density: ρ at the smallest radius outside the shock where v has
    fallen to ~CSM values (v < 2 × v_min). This is the unshocked wind density
    that feeds the forward shock.

    Returns dict with R_s, v_s, ρ_csm, T_shock, L_shock, eta_rad, L_X_brems,
    t_cool, t_dyn, idx_shock; or None if velocity info is unavailable.
    """
    r = np.asarray(snap_full['r'], dtype=float)
    rho = np.asarray(snap_full['rho'], dtype=float)
    v = snap_full.get('v', None)
    if v is None:
        return None
    v = np.asarray(v, dtype=float)
    if np.all(v == 0):
        return None
    if len(r) < 5:
        return None

    # Identify forward shock: max |dv/dr| where v decreases outward
    # (CDS material at high v → CSM at low v as r increases)
    with np.errstate(divide='ignore', invalid='ignore'):
        dr_diff = np.diff(r)
        dr_safe = np.where(dr_diff > 0, dr_diff, np.inf)
        dvdr = np.diff(v) / dr_safe
    idx_shock = int(np.argmin(dvdr))   # most-negative dv/dr (sharpest dropoff)

    R_s = float(r[idx_shock])
    v_s = float(v[idx_shock])
    if v_s <= 0:
        return None

    # Pre-shock CSM density: smallest r outside shock where v ≤ 2 × v_min
    v_outer_min = float(v[idx_shock+1:].min()) if idx_shock+1 < len(v) else float(v_s)
    csm_candidates = np.where((np.arange(len(v)) > idx_shock) &
                                (v <= 2.0 * v_outer_min))[0]
    if len(csm_candidates) > 0:
        idx_csm = int(csm_candidates[0])
    else:
        idx_csm = min(idx_shock + 3, len(r) - 1)
    rho_csm = float(rho[idx_csm])

    # Shock temperature (strong-shock Rankine-Hugoniot, γ=5/3)
    mu = 0.6                      # mean molecular weight (ionized H+He cosmic mix)
    T_shock = (3.0/16.0) * mu * MH * v_s**2 / KB

    # Shock dissipation rate
    L_shock = 4.0 * np.pi * R_s**2 * 0.5 * rho_csm * v_s**3

    # Radiative efficiency from t_cool/t_dyn
    # Post-shock n_e in strong-shock limit (γ=5/3, compression factor 4)
    n_e_ps = 4.0 * rho_csm / MH
    # Cooling function approximation at T~10⁸ K: thermal bremsstrahlung
    # ~ 1.4e-27 × T^0.5 × n_e × n_p / (n_e × n_H), so Λ ≈ 1.4e-27 × T^0.5
    # (per unit n_e × n_H, in erg cm³/s)
    Lambda_T = 1.4e-27 * np.sqrt(T_shock)
    # t_cool = (thermal energy density) / (cooling rate per volume)
    # = (3/2 n k T) / (n_e n_H Λ) ≈ (3/2 kT) / (n_e Λ)  [for ionized gas]
    t_cool = 1.5 * KB * T_shock / max(n_e_ps * Lambda_T, 1e-30)
    t_dyn  = R_s / v_s
    eta_rad = 1.0 / (1.0 + t_cool / t_dyn)

    # Fraction of bremsstrahlung above Lyman edge
    # For thermal brems dL/dE ∝ exp(-E/kT_X), the fraction above hν₀ is
    # exp(-hν₀/kT_X). For T_X >> hν₀/k = 158000 K (the typical IIn shock
    # case), this is ~1.
    alpha_LyC = T_LYC / max(T_shock, 1.0)
    f_above_LyC = np.exp(-alpha_LyC) if alpha_LyC < 700 else 0.0

    L_X_brems = eta_rad * L_shock * f_above_LyC

    # Sanity check: L_X_brems should be of order L_phot or modestly larger
    # (shock dissipates kinetic energy that becomes radiation; in a steady-
    # state IIn this ≈ L_phot). If our derived L_X_brems is >> L_phot, the
    # shock identification (R_s, v_s, ρ_csm) likely failed — typical causes:
    # (1) the "shock" we found is inside the photosphere (still optically
    # thick, energy doesn't escape), (2) the pre-shock density was sampled
    # inside the CDS rather than in true wind, or (3) the snapshot is not
    # a IIn (e.g. stripped-envelope, where there is no forward-shock-into-
    # dense-CSM physics and the formula is misapplied).
    L_phot_snap = float(snap_full.get('L_phot_inner',
                                       snap_full.get('L', np.array([0.0])).max()))
    sanity_ratio = L_X_brems / max(L_phot_snap, 1e-30)
    sanity_flag = sanity_ratio > 100.0  # flag if 100× implausible

    # ---- SMOOTH shock-X-ray escape factor (replaces the old binary R_s≥R_phot
    # gate). The hard X-rays emitted at R_s must cross the overlying gas to reach
    # the photosphere/line-forming CSM; the escaping fraction is exp(−Δτ_es),
    # where Δτ_es is the electron-scattering column BETWEEN the shock and the
    # photosphere (τ_es = tau_phot_ref, default 2/3). The full snapshot's
    # `tau_es` is cumulative from the outer edge inward (small outside, large
    # inside, = tau_phot_ref at the photosphere), so:
    #   shock at/above the photosphere (τ_es ≤ ref) → Δτ = 0      → f = 1
    #   shock buried one optical depth under         → Δτ = 1      → f ≈ 0.37
    #   shock deep in the optically-thick interior   → Δτ ≫ 1      → f → 0
    # Because f is CONTINUOUS in R_s, the X-ray field no longer toggles on/off as
    # the shock front wobbles across R_phot between adjacent snapshots — which was
    # the true source of the metal high-ion (C III↔C IV, [O III]) epoch-to-epoch
    # flicker and the matching H/He ionization jitter at the breakout epochs.
    tau_es_full = snap_full.get('tau_es', None)
    if tau_es_full is not None and idx_shock < len(np.atleast_1d(tau_es_full)):
        tau_sh = float(np.asarray(tau_es_full, float)[idx_shock])
        f_xray_escape = float(np.exp(-max(0.0, tau_sh - float(tau_phot_ref))))
    else:
        # Fallback (no tau_es array): recover the old binary behaviour from the
        # geometric criterion, expressed as f∈{0,1} so downstream code is uniform.
        tau_sh = float('nan')
        R_phot_guess = float(snap_full.get('R_phot_inner', 0.0) or 0.0)
        f_xray_escape = 1.0 if (R_phot_guess <= 0 or R_s >= R_phot_guess) else 0.0

    if verbose:
        kT_keV = KB * T_shock / EV / 1000
        print(f"[derive_shock_params]")
        print(f"  R_s         = {R_s:.3e} cm  (zone {idx_shock} of {len(r)})")
        print(f"  v_s         = {v_s/1e5:.0f} km/s")
        print(f"  ρ_csm       = {rho_csm:.2e} g/cm³  (zone {idx_csm}, r={r[idx_csm]:.2e})")
        print(f"  T_shock     = {T_shock:.2e} K  (kT = {kT_keV:.1f} keV)")
        print(f"  L_shock     = {L_shock:.3e} erg/s")
        print(f"  t_cool      = {t_cool:.2e} s  (Λ ≈ {Lambda_T:.2e} cgs)")
        print(f"  t_dyn       = {t_dyn:.2e} s")
        print(f"  η_rad       = {eta_rad:.3f}")
        print(f"  L_X (>13.6eV) = {L_X_brems:.3e} erg/s  ({f_above_LyC*100:.1f}% of brems)")
        print(f"  f_xray_escape = {f_xray_escape:.3f}  (τ_es@shock = {tau_sh:.2f}, "
              f"ref = {float(tau_phot_ref):.2f}; L_X·f = {L_X_brems*f_xray_escape:.3e} erg/s)")
        if sanity_flag:
            print(f"  ⚠ SANITY WARNING: L_X_brems = {L_X_brems:.2e} is "
                  f"{sanity_ratio:.0f}× L_phot ({L_phot_snap:.2e}).")
            print(f"    Likely the shock identification failed for this snapshot.")
            print(f"    Recommend disabling shock-Xray (--no-shock-xray) or")
            print(f"    inspecting the snapshot's hydro structure manually.")

    return dict(
        R_s=R_s, v_s=v_s, rho_csm=rho_csm,
        T_shock=T_shock, L_shock=L_shock,
        eta_rad=eta_rad, t_cool=t_cool, t_dyn=t_dyn,
        L_X_brems=L_X_brems,
        f_xray_escape=f_xray_escape, tau_es_shock=tau_sh,
        idx_shock=idx_shock, idx_csm=idx_csm,
        sanity_ratio_L_X_over_L_phot=sanity_ratio, sanity_flag=sanity_flag,
    )


def _G_function_brems(T_shock: float, tau_thresh: float | np.ndarray) -> np.ndarray:
    """Per-zone source-integrated photoionization-rate kernel for a thermal
    bremsstrahlung source.

    For a brems source with TOTAL ionizing luminosity L_X_brems at the
    photosphere, the photoionization rate per H atom is:

        Γ_brems(r) = (L_X_brems / (π R_phot²)) × W(r) × G_brems(T_X, τ_LyC(r))

    where

        G_brems(T, τ) = (h / kT) × ∫_{ν₀}^∞ exp(-hν/kT) × σ_ν × exp(-τ_ν) / hν dν

    This normalization absorbs the spectrum integration so that the per-zone
    photoionization rate scales linearly with L_X_brems and the radiation
    geometry, just like the BB-version G_function.

    The h/kT prefactor makes G_brems dimensionally (cm² per erg), so when
    multiplied by L_X / (π R²) we get a photoionization rate per atom [s⁻¹].

    Numerical care: substituting u = x_0/x (so the lower limit x = x_0
    maps to u = 1 and the integrand's small-x peak maps to a well-behaved
    finite endpoint at u = 1) gives a transformed integrand u² exp(-x_0/u)
    that is bounded and smooth on (0, 1). scipy.quad on the original x-space
    integral fails spectacularly when x_0 is tiny because the adaptive
    sampler misses the x ~ x_0 peak; the u-substitution removes this.
    """
    tau_arr = np.atleast_1d(tau_thresh).astype(float)
    out = np.zeros_like(tau_arr, dtype=float)

    x0 = T_LYC / max(T_shock, 1.0)
    if x0 > 700:
        return out * 0.0 if isinstance(tau_thresh, np.ndarray) else 0.0

    # Constant prefactor of the x-space integrand: σ_0 × x_0³ / kT
    # (after factoring out x⁻⁴ exp(-x) × exp(-τ y³) shape)
    kT = KB * T_shock
    prefactor = SIGMA_H0 * x0**3 / kT

    # Use u = x_0 / x substitution. Then:
    #   x = x_0/u,  dx = -(x_0/u²) du
    #   y = x_0/x = u
    #   x⁻⁴ exp(-x) dx → (u⁴/x_0⁴) exp(-x_0/u) × (x_0/u²) du = u²/x_0³ × exp(-x_0/u) du
    # Combined with the prefactor σ_0 × x_0³/kT, the integrand simplifies to:
    #   (σ_0/kT) × u² × exp(-x_0/u) × exp(-τ × u³)   on u ∈ (0, 1]
    n_u = 4096
    u_grid = np.linspace(1e-8, 1.0, n_u)
    for i, tau in enumerate(tau_arr):
        # τ_ν = τ × (ν_0/ν)³ = τ × y³ = τ × u³
        tau_u = tau * u_grid**3
        # Clip extreme values to avoid underflow
        exp_tau = np.where(tau_u > 60.0, 0.0, np.exp(-tau_u))
        # exp(-x_0/u) safe (u > 0)
        with np.errstate(divide='ignore', over='ignore'):
            exp_arg = -x0 / u_grid
            integrand = u_grid**2 * np.exp(np.maximum(exp_arg, -700.0)) * exp_tau
        # Trapezoid (uniform grid in u)
        out[i] = (SIGMA_H0 / kT) * np.trapezoid(integrand, u_grid)

    if np.isscalar(tau_thresh):
        return float(out[0])
    return out


def _G_function_brems_energy(T_shock: float,
                               tau_thresh: float | np.ndarray) -> np.ndarray:
    """Energy-weighted brems kernel for secondary-electron accounting.

    Identical structure to _G_function_brems but with an extra factor of
    (hν - hν₀)/hν₀ = (1/y - 1) = (1/u - 1)  in the integrand under the
    u = x_0/x substitution. The ratio

        ⟨E_pe / χ_H⟩(r) = G_brems_energy(τ) / G_brems(τ)

    gives the rate-weighted mean photoelectron energy per zone, in units of
    the H ionization threshold (13.6 eV). This is the multiplier that enters
    the Shull & van Steenberg 1985 secondary-electron prescription:

        N_sec(r) = φ_ion,H(x_e(r)) × ⟨E_pe/χ_H⟩(r)

    where φ_ion,H is the energy fraction of the photoelectron that goes
    into new H ionizations (shuts off as the gas approaches X_HII → 1
    because secondaries thermalize on free electrons instead of ionizing
    neutrals).

    Why this matters: soft brems photons (just above ν₀) have huge σ_bf
    and dominate primary ionizations near R_phot, but their photoelectrons
    carry only a few eV and make NO secondaries. Hard brems photons (keV)
    have tiny σ but survive deep into the CSM because the soft photons get
    absorbed first — and their photoelectrons carry keV energies and make
    hundreds of secondaries. So the secondary multiplier ramps up with
    τ_LyC, doing its work exactly where the dense CDS has exhausted the
    primary BB+brems channels.

    See _G_function_brems for the numerical-integration rationale (u-substitution
    to handle the x → 0 peak).
    """
    tau_arr = np.atleast_1d(tau_thresh).astype(float)
    out = np.zeros_like(tau_arr, dtype=float)

    x0 = T_LYC / max(T_shock, 1.0)
    if x0 > 700:
        return out * 0.0 if isinstance(tau_thresh, np.ndarray) else 0.0

    kT = KB * T_shock
    n_u = 4096
    u_grid = np.linspace(1e-8, 1.0, n_u)
    # Energy-weight factor: (1/u - 1). Note this is 0 at u=1 and large at u→0.
    energy_weight = (1.0 / u_grid) - 1.0
    for i, tau in enumerate(tau_arr):
        tau_u = tau * u_grid**3
        exp_tau = np.where(tau_u > 60.0, 0.0, np.exp(-tau_u))
        with np.errstate(divide='ignore', over='ignore'):
            exp_arg = -x0 / u_grid
            integrand = (u_grid**2 * np.exp(np.maximum(exp_arg, -700.0))
                         * exp_tau * energy_weight)
        out[i] = (SIGMA_H0 / kT) * np.trapezoid(integrand, u_grid)

    if np.isscalar(tau_thresh):
        return float(out[0])
    return out


def phi_ion_H_shull_vansteenberg(x_e: np.ndarray | float) -> np.ndarray:
    """Shull & van Steenberg 1985 secondary-electron H-ionization fraction.

    Returns the fraction of a fast photoelectron's kinetic energy that goes
    into new H photoionizations, as a function of the local electron
    fraction x_e = n_e / n_H. The fit is:

        φ_ion,H(x_e) = 0.3908 × (1 - x_e^0.4092)^1.7592

    Limits:
        x_e → 0:   φ → 0.39    (mostly neutral, plenty of HI targets)
        x_e = 0.1: φ ≈ 0.24
        x_e = 0.5: φ ≈ 0.10
        x_e → 1:   φ → 0       (no HI to ionize; energy heats electrons)

    Reference: Shull & van Steenberg 1985 ApJ 298, 268.

    The Shull/vSt fits assume a mixed H+He plasma with cosmic abundance.
    For pure H, the He branch is absent; we apply the same fit which is
    conservative (slightly overcounts H ionizations).
    """
    x = np.atleast_1d(np.clip(np.asarray(x_e, dtype=float), 0.0, 1.0))
    # Avoid the 0**0.4 numerical issue at exactly x=0
    x_safe = np.where(x > 1e-12, x, 1e-12)
    phi = 0.3908 * (1.0 - x_safe**0.4092)**1.7592
    phi = np.where(x >= 1.0, 0.0, phi)
    if np.isscalar(x_e):
        return float(phi[0])
    return phi


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_photoionization_equilibrium(
        snap: dict,
        T_source: float | None = None,
        L_normalize_to_snap: bool = True,
        T_eq_floor: float | None = 1.0e4,
        shock_params: dict | None = None,
        include_shock_xray: bool = True,
        max_iter: int = 60,
        tol: float = 1.0e-3,
        damping: float = 0.4,
        n_tau_table: int = 80,
        verbose: bool = True,
) -> dict:
    """Solve per-zone photoionization equilibrium for a SN snapshot.

    Parameters
    ----------
    snap : dict
        Truncated SN snapshot (output of load_snapshot + truncate_to_photosphere).
        Must contain keys: 'r', 'rho', 'T', 'X_H'. The innermost zone r[0] is
        treated as the photospheric source. T_phot is taken from snap['T'][0]
        (or snap['T_phot_inner'] if present) for the default source spectrum.
        L_phot is taken from snap['L_phot_inner'] if present, else computed
        from 4π R² σ T_phot⁴.
    T_source : float or None
        Effective temperature of the emergent radiation spectrum at R_phot.
        Default None → use snap['T_phot_inner'] (= STELLA's T_phot, strict
        input model). Set to a higher value to test sensitivity to harder
        radiation (e.g., color correction or shock contributions).
    L_normalize_to_snap : bool
        If True (default), scale the BB prescription so its bolometric flux
        at R_phot matches snap['L_phot_inner']. This decouples the SED
        SHAPE (set by T_source) from the SED NORMALIZATION (set by L_phot).
        If False, use B_ν(T_source) without renormalization (only consistent
        when T_source = T_phot).
    T_eq_floor : float or None
        If given, raise T_e to max(T_e, T_eq_floor) in zones where X_HII > 0.5.
        Justified because real photoionization-recombination equilibrium gives
        T_e ~ 10⁴ K, but the snapshot may report T < this in zones STELLA
        treats as cool. Default None (use snap T_e unchanged).
    max_iter, tol, damping : iteration controls
        max_iter — outer iterations (typical convergence: 10-30)
        tol     — converged when max ΔX_HII < tol  (default 1e-3)
        damping — under-relaxation (default 0.4; 0.5 if oscillating)
    n_tau_table : int
        Number of τ-grid points for G(τ) tabulation (default 80,
        log-spaced from 1e-3 to 1e5). More → more accurate interp.

    Returns
    -------
    new_snap : dict
        Copy of input snap with these arrays UPDATED:
            'X_HII'   — H ionization fraction per zone (new key)
            'n_HII'   — H II number density (new key)
            'n_HI'    — H I number density (new key)
            'n_e'     — electron density (overwrites STELLA's)
            'tau_es' — recomputed from new n_e
            'T'       — only if T_eq_floor is given
        Original arrays are preserved under <key>_stella for diagnostics:
            'n_e_stella', 'T_stella', 'tau_es_stella'
        Plus diagnostics:
            'X_HII_stella'  — STELLA's implied X_HII = n_e_stella/n_H
            'Gamma_HI'      — final photoionization rate per H atom [s⁻¹]
            'photoionization_params' — dict of solver settings
    """
    r = np.asarray(snap['r'], dtype=float)
    rho = np.asarray(snap['rho'], dtype=float)
    T_gas = np.asarray(snap['T'], dtype=float)
    X_H = np.asarray(snap.get('X_H', np.full_like(r, 0.737)))

    R_phot = float(snap.get('R_phot_inner', r[0]))
    if R_phot <= 0:
        R_phot = float(r[0])
    T_phot = float(snap.get('T_phot_inner', T_gas[0]))
    L_phot = float(snap.get('L_phot_inner',
                              4.0 * np.pi * R_phot * R_phot * SIGMA_SB * T_phot**4))

    # Default source spectrum: T_color at thermalization depth, computed
    # from the full snapshot before truncation (stored as
    # 'T_color_thermalization', default τ_es=10, ε~0.003). Falls back to
    # T_phot only if no thermalization-depth value is available (e.g.
    # non-STELLA snapshots).
    if T_source is None:
        # First-principles: the gas ABOVE the photosphere is illuminated by the
        # EMERGENT photospheric spectrum, whose colour temperature is T_phot.
        # The deep thermalization-layer colour temperature (T_color_thermalization,
        # measured at τ_es≈10) is hotter and Lyman-continuum-rich; using it as the
        # external source over-ionizes the near-photosphere zones by orders of
        # magnitude (it injects LyC photons the emergent field does not contain).
        # The ionizing budget that is physically consistent with the input model
        # is the emergent photospheric continuum + (if it escapes) the shock.
        T_source = T_phot
        if verbose:
            print(f"[photoionize_csm] T_source = emergent T_phot = "
                  f"{T_source:.0f} K")

    # Compute spectral normalization factor: if we use B_ν(T_source) but
    # want to preserve the bolometric L_phot, scale by
    # f_norm = L_phot / (4π R_phot² σ T_source⁴)
    if L_normalize_to_snap:
        L_bb_at_R = 4.0 * np.pi * R_phot * R_phot * SIGMA_SB * T_source**4
        f_norm = L_phot / max(L_bb_at_R, 1e-300)
    else:
        f_norm = 1.0

    n_zones = len(r)
    n_H_total = X_H * rho / MH

    # Dilution factor W(r) for r >= R_phot
    # For r = R_phot exactly, the integration sphere just touches the source
    # and W = 1/2 (outward hemisphere only). For r > R_phot, W < 1/2 monotone.
    arg = 1.0 - (R_phot / np.maximum(r, R_phot))**2
    arg = np.clip(arg, 0.0, 1.0)
    W = 0.5 * (1.0 - np.sqrt(arg))
    W[0] = 0.5  # ensure W=1/2 right at the photosphere

    # Save STELLA values for comparison
    n_e_stella = np.asarray(snap['n_e'], dtype=float)
    tau_es_stella = np.asarray(snap['tau_es'], dtype=float)
    T_stella = T_gas.copy()
    X_HII_stella = np.clip(n_e_stella / np.maximum(n_H_total, 1e-30), 0.0, 1.0)

    # ---- Tabulate G(τ) for fast per-zone lookup ----
    # τ range covers transparent regime (τ≪1) through optically-thick LyC
    # (τ up to 10¹¹). The upper end matters specifically for the shock-brems
    # channel: at threshold-frequency τ_LyC ~ 10⁷-10⁹ (typical CDS columns),
    # the LyC photons are completely extinguished, but HARD X-ray photons
    # with hν/hν₀ > τ_LyC^(1/3) still penetrate (σ ∝ ν⁻³). The integral
    # captures these correctly as long as the τ-table reaches high enough.
    log_tau_lo, log_tau_hi = -3.0, 11.0
    log_tau_table = np.linspace(log_tau_lo, log_tau_hi, n_tau_table)
    tau_table = 10.0**log_tau_table
    G_table = _G_function(T_source, tau_table) * f_norm
    # Also compute G at τ = 0 (fully transparent) for the inner zone
    G_at_zero = _G_function(T_source, 0.0) * f_norm

    # ---- Optionally tabulate G_brems(τ) for shock-X-ray contribution ----
    # Shock X-rays can only ionize the line-forming region if the forward shock
    # sits AT or OUTSIDE the photosphere. A shock buried in the optically-thick
    # interior (R_s < R_phot) has its hard radiation reprocessed/thermalized by
    # the overlying material — that energy is already carried by STELLA's
    # emergent continuum, so adding it here would double-count and spuriously
    # ionize. This single physical criterion auto-includes the genuine IIn
    # forward-shock-into-CSM and auto-excludes interior velocity features
    # (e.g. the inner-ejecta/CDS edge in a plateau IIP).
    # SMOOTH escape: scale the shock X-ray field by f_xray_escape = exp(−Δτ_es)
    # (computed in derive_shock_params from the overlying electron-scattering
    # column) instead of the old binary R_s≥R_phot ON/OFF switch. Continuous in
    # the shock position → no epoch-to-epoch toggling of the ionizing flux.
    f_xray_escape = (float(shock_params.get('f_xray_escape', 1.0))
                     if shock_params is not None else 0.0)
    use_xray = bool(include_shock_xray and shock_params is not None
                    and shock_params.get('L_X_brems', 0.0) > 0.0
                    and f_xray_escape > 1.0e-3)
    if verbose and include_shock_xray and shock_params is not None and f_xray_escape < 0.999:
        print(f"[photoionize_csm] shock X-rays ATTENUATED: f_xray_escape="
              f"{f_xray_escape:.3f} (overlying τ_es column above R_s="
              f"{float(shock_params.get('R_s',0.0)):.2e} cm; smooth gate, not on/off).")
    if use_xray:
        T_shock = float(shock_params['T_shock'])
        L_X_brems = float(shock_params['L_X_brems'])
        # The shock radiates from R_shock; treat it as coincident with R_phot
        # for the geometric dilution factor W(r). The brems-contribution to
        # Γ per atom at zone with τ_LyC = τ:
        #     Γ_brems(r) = (L_X_brems / (π R_phot²)) × W(r) × G_brems(T_shock, τ)
        # G_brems already has the (h/kT)/hν Jacobian baked in; see function docstring.
        G_brems_table = _G_function_brems(T_shock, tau_table)
        G_brems_at_zero = _G_function_brems(T_shock, 0.0)
        # Energy-weighted brems kernel for secondary-electron accounting:
        # ⟨E_pe/χ_H⟩(τ) = G_brems_eng_table(τ) / G_brems_table(τ)
        # This rate-weighted mean photoelectron energy in units of χ_H is
        # then combined with the Shull-van Steenberg φ_ion,H(x_e) factor to
        # give N_sec, the secondary-ionization multiplier per zone.
        G_brems_eng_table = _G_function_brems_energy(T_shock, tau_table)
        G_brems_eng_at_zero = _G_function_brems_energy(T_shock, 0.0)
        # Coefficient that converts G_brems → Γ_brems given W:
        # coef × W × G_brems = Γ_brems. Scaled by the smooth escape fraction so a
        # partially-buried shock contributes a partial (not all-or-nothing) flux.
        xray_coef = f_xray_escape * L_X_brems / (np.pi * R_phot**2)
    else:
        T_shock = 0.0
        L_X_brems = 0.0
        G_brems_table = np.zeros_like(tau_table)
        G_brems_at_zero = 0.0
        G_brems_eng_table = np.zeros_like(tau_table)
        G_brems_eng_at_zero = 0.0
        xray_coef = 0.0

    if verbose:
        x_thresh = T_LYC / T_source
        print(f"[photoionize_csm] solving with source spectrum:")
        print(f"  T_source = {T_source:.0f} K  (T_phot = {T_phot:.0f} K, "
              f"L_phot = {L_phot:.3e} erg/s)")
        print(f"  hν₀/kT = {x_thresh:.2f}  ⇒  Wien suppression exp(-{x_thresh:.1f})"
              f" = {np.exp(-min(x_thresh,700)):.2e}")
        print(f"  L_normalize_to_snap = {L_normalize_to_snap}  "
              f"(f_norm = {f_norm:.3e})")
        print(f"  R_phot = {R_phot:.3e} cm, n_zones = {n_zones}")
        print(f"  G_BB(τ=0) = {G_at_zero:.3e}  →  Γ_BB(R_phot) = "
              f"{4.0*np.pi*0.5*G_at_zero:.3e} s⁻¹")
        if use_xray:
            kT_keV = KB * T_shock / EV / 1000
            Gamma_X_at_R = xray_coef * 0.5 * G_brems_at_zero
            print(f"  + bremsstrahlung shock component:")
            print(f"    T_shock = {T_shock:.2e} K (kT = {kT_keV:.1f} keV)")
            print(f"    L_X_brems = {L_X_brems:.3e} erg/s  "
                  f"({L_X_brems/L_phot:.2f}× L_phot)"
                  if L_phot > 0 else
                  f"    L_X_brems = {L_X_brems:.3e} erg/s  (L_phot=0)")
            print(f"    Γ_brems(R_phot) ≈ {Gamma_X_at_R:.3e} s⁻¹")
            print(f"    Γ_BB/Γ_brems(R_phot) = "
                  f"{(4.0*np.pi*0.5*G_at_zero) / max(Gamma_X_at_R, 1e-30):.2e}")

    # If G(0) is essentially zero, we won't get any ionization regardless.
    Gamma_zero_at_R = 4.0 * np.pi * W[0] * G_at_zero
    if Gamma_zero_at_R < 1e-20:
        if verbose:
            print(f"  ⚠ Γ_ion at the photosphere is {Gamma_zero_at_R:.2e} s⁻¹: "
                  f"the source spectrum has effectively no Lyman-continuum photons. "
                  f"Photoionization will give X_HII ≈ Saha values, matching STELLA.")

    # ---- Outer iteration ----
    # Initial guess: STELLA's X_HII as starting point (since deep zones may
    # remain partially neutral, starting from full ionization can overshoot).
    X_HII = X_HII_stella.copy()
    X_HII = np.where(X_HII < 0.01, 0.1, X_HII)   # avoid zero start

    dr = np.empty_like(r); dr[:-1] = np.diff(r); dr[-1] = dr[-2]

    Gamma_per_zone = np.zeros_like(r)
    Gamma_BB_per_zone = np.zeros_like(r)
    Gamma_X_per_zone = np.zeros_like(r)
    Gamma_X_primary_per_zone = np.zeros_like(r)
    mean_Epe_per_zone = np.zeros_like(r)
    N_sec_per_zone = np.zeros_like(r)
    q_per_zone    = np.zeros_like(r)

    converged = False
    for it in range(max_iter):
        n_HI  = (1.0 - X_HII) * n_H_total
        n_HII = X_HII * n_H_total
        n_e   = n_HII

        # Compute τ_LyC(r) = ∫_{R_phot}^r σ₀ n_HI dr'  (cumulative outward)
        # τ_LyC[0] = 0 (at the source); grows outward as we cross neutral material.
        dtau = SIGMA_H0 * n_HI * dr
        # cumulative from r[0] to r[i]: integrate inclusive of zone width up to i
        tau_LyC = np.concatenate([[0.0], np.cumsum(dtau[:-1])])
        # equivalent: tau_LyC[0] = 0 (at source); tau_LyC[i] = column from r[0] to r[i]

        # G(τ) per zone by interpolation (BB component)
        tau_clip = np.clip(tau_LyC, tau_table[0], tau_table[-1])
        G_vals = np.interp(np.log10(tau_clip), log_tau_table, G_table)
        G_vals = np.where(tau_LyC < tau_table[0], G_at_zero, G_vals)
        G_vals = np.where(tau_LyC > tau_table[-1], 0.0, G_vals)

        Gamma_BB = 4.0 * np.pi * W * G_vals

        # Brems contribution (if shock X-rays are included)
        if use_xray:
            G_brems_vals = np.interp(np.log10(tau_clip), log_tau_table, G_brems_table)
            G_brems_vals = np.where(tau_LyC < tau_table[0], G_brems_at_zero, G_brems_vals)
            G_brems_vals = np.where(tau_LyC > tau_table[-1], 0.0, G_brems_vals)
            Gamma_brems_primary = xray_coef * W * G_brems_vals

            # Secondary-electron multiplier per zone (Shull & van Steenberg 1985)
            # Rate-weighted ⟨E_pe/χ_H⟩ per zone from energy-weighted G kernel
            G_brems_eng_vals = np.interp(np.log10(tau_clip), log_tau_table,
                                            G_brems_eng_table)
            G_brems_eng_vals = np.where(tau_LyC < tau_table[0],
                                          G_brems_eng_at_zero, G_brems_eng_vals)
            G_brems_eng_vals = np.where(tau_LyC > tau_table[-1], 0.0, G_brems_eng_vals)
            # Avoid 0/0 in zones where G_brems is essentially zero (use 
            # np.divide with where= to avoid the RuntimeWarning that np.where
            # produces because the divide happens before the selection).
            mean_Epe_over_chi = np.divide(
                G_brems_eng_vals, G_brems_vals,
                out=np.zeros_like(G_brems_vals),
                where=(G_brems_vals > 1e-30))
            # φ_ion,H depends on the LOCAL ionization fraction
            phi_ion = phi_ion_H_shull_vansteenberg(X_HII)
            # N_sec per zone; multiplier (1 + N_sec) on the primary rate
            N_sec = phi_ion * mean_Epe_over_chi
            sec_multiplier = 1.0 + N_sec
            Gamma_X = Gamma_brems_primary * sec_multiplier
        else:
            Gamma_X = np.zeros_like(Gamma_BB)
            Gamma_brems_primary = np.zeros_like(Gamma_BB)
            mean_Epe_over_chi = np.zeros_like(Gamma_BB)
            N_sec = np.zeros_like(Gamma_BB)
            sec_multiplier = np.ones_like(Gamma_BB)

        Gamma = Gamma_BB + Gamma_X

        # α_B and quadratic for X_HII
        alpha_B = _alpha_B(T_gas)
        with np.errstate(divide='ignore', invalid='ignore'):
            q = np.where(Gamma > 0.0, n_H_total * alpha_B / Gamma, np.inf)
            disc = 1.0 + 4.0 * q
            X_HII_new = np.where(np.isfinite(q),
                                   (-1.0 + np.sqrt(disc)) / (2.0 * q + 1e-300),
                                   0.0)
        X_HII_new = np.clip(X_HII_new, 0.0, 1.0)

        # Damped update
        X_HII_next = damping * X_HII_new + (1.0 - damping) * X_HII
        max_change = float(np.max(np.abs(X_HII_next - X_HII)))
        X_HII = X_HII_next

        Gamma_per_zone = Gamma
        Gamma_BB_per_zone = Gamma_BB
        Gamma_X_per_zone = Gamma_X
        Gamma_X_primary_per_zone = Gamma_brems_primary
        mean_Epe_per_zone = mean_Epe_over_chi
        N_sec_per_zone = N_sec
        q_per_zone = q

        if verbose and (it < 4 or it % 5 == 0):
            ion_fraction = np.mean(X_HII > 0.5)
            print(f"  iter {it:3d}: max ΔX_HII = {max_change:.2e},  "
                  f"⟨X_HII⟩ = {X_HII.mean():.3f},  "
                  f"f(X_HII>0.5) = {ion_fraction:.2f},  "
                  f"range [{X_HII.min():.2e}, {X_HII.max():.3f}]")

        if max_change < tol:
            converged = True
            if verbose:
                print(f"  → converged at iter {it} (Δ={max_change:.2e} < tol={tol})")
            break

    if not converged and verbose:
        print(f"  ⚠ did NOT converge after {max_iter} iters "
              f"(final Δ={max_change:.2e}); proceeding with last iterate")

    # Final state — whole-state per-zone gate.
    #
    # STELLA's radiative transfer already solves the ionization self-consistently
    # for the thermal radiation field it transports; in the optically-thick /
    # collisionally-coupled regime that solution is correct and complete. The
    # photoionization-equilibrium solve here is a NEBULAR correction whose only
    # legitimate role is to ADD the externally-driven ionization (emergent
    # photospheric LyC + escaping shock X-rays) that STELLA's grey/low-resolution
    # opacity treatment misses in genuinely optically-thin, externally-illuminated
    # CSM. It must therefore never REDUCE the ionization below STELLA's value
    # (that is what produced the Lyman-continuum "shadow collapse": one
    # over-ionized skin zeroing every zone behind it).
    #
    # So we adopt the photoeq state ONLY in zones where it exceeds STELLA's
    # ionization; elsewhere we retain STELLA's full, self-consistent state. The
    # swap is whole-state (n_e, n_HII, n_HI together) so the recombination
    # emissivity (∝ n_p n_e) and the NLTE level solve always see a single,
    # internally-consistent ionization — never a mix of STELLA n_e with photoeq
    # n_p or vice versa.
    n_HII_pe = X_HII * n_H_total
    n_HII_stella_implied = X_HII_stella * n_H_total
    # Non-H electron contribution (He/metals) preserved from STELLA; for H-rich
    # ejecta the H term dominates, for stripped-envelope (X_H→0) STELLA's
    # He/metal electrons carry n_e.
    n_e_other = np.maximum(n_e_stella - n_HII_stella_implied, 0.0)
    n_e_pe = n_HII_pe + n_e_other

    adopt_photoeq = X_HII > X_HII_stella          # photoeq only where it ADDS ionization
    n_e   = np.where(adopt_photoeq, n_e_pe,   n_e_stella)
    n_HII = np.where(adopt_photoeq, n_HII_pe, n_HII_stella_implied)
    n_HI  = np.maximum(n_H_total - n_HII, 0.0)
    X_HII = np.where(adopt_photoeq, X_HII, X_HII_stella)

    # No temperature floor. The gas temperature is STELLA's physical output
    # everywhere; the former 1e4 K `T_eq_floor` was a stand-in for the
    # photoionization thermostat that, applied ungated, contaminated cool
    # near-photosphere zones. With the escaping-source + whole-state gate the
    # clamp is unnecessary. (A proper photoionization-heating/cooling thermal
    # balance for the adopted-photoeq zones is a separable, second-order
    # refinement — it shifts α_Hα ∝ T^-0.94 mildly — and is intentionally NOT
    # faked with a floor here.)
    if T_eq_floor is not None and verbose:
        print("[photoionize_csm] NOTE: T_eq_floor is deprecated and ignored; "
              "gas temperature retained from STELLA (no floor).")
    T_new = T_gas.copy()
    n_adopt = int(np.count_nonzero(adopt_photoeq))
    if verbose:
        print(f"[photoionize_csm] whole-state gate: photoeq ionization adopted in "
              f"{n_adopt}/{n_zones} zones (those where it exceeds STELLA); "
              f"STELLA state retained elsewhere; no T floor.")

    # Recompute τ_es (cumulative from outside, monotone decreasing in i)
    tau_es_new = np.cumsum((SIGMAT * n_e * dr)[::-1])[::-1]

    # ---- Build returned snap ----
    new_snap = dict(snap)
    new_snap['n_e']    = n_e
    new_snap['n_HII']  = n_HII
    new_snap['n_HI']   = n_HI
    new_snap['X_HII']  = X_HII
    new_snap['tau_es'] = tau_es_new
    new_snap['T']      = T_new

    # Diagnostics: preserve originals
    new_snap['n_e_stella']      = n_e_stella
    new_snap['T_stella']        = T_stella
    new_snap['tau_es_stella']  = tau_es_stella
    new_snap['X_HII_stella']    = X_HII_stella
    new_snap['Gamma_HI']        = Gamma_per_zone
    new_snap['Gamma_HI_BB']     = Gamma_BB_per_zone
    new_snap['Gamma_HI_brems']  = Gamma_X_per_zone
    new_snap['Gamma_HI_brems_primary'] = Gamma_X_primary_per_zone
    new_snap['mean_Epe_over_chi'] = mean_Epe_per_zone
    new_snap['N_sec_per_zone']  = N_sec_per_zone
    new_snap['q_recomb_over_ion'] = q_per_zone
    new_snap['photoionized']    = True

    new_snap['photoionization_params'] = dict(
        T_source=T_source, T_phot=T_phot, L_phot=L_phot,
        L_normalize_to_snap=L_normalize_to_snap, f_norm=f_norm,
        T_eq_floor=T_eq_floor,
        include_shock_xray=use_xray,
        f_xray_escape=f_xray_escape,
        shock_params=(dict(shock_params) if shock_params is not None else None),
        max_iter=max_iter, tol=tol, damping=damping,
        n_tau_table=n_tau_table,
        converged=converged, iterations_used=it+1, final_residual=max_change,
    )

    if verbose:
        print()
        print(f"[photoionize_csm] RESULT:")
        print(f"  τ_es (STELLA):     {tau_es_stella.max():.3f}  total along LOS")
        print(f"  τ_es (photoionized): {tau_es_new.max():.3f}  total along LOS  "
              f"(ratio {tau_es_new.max()/max(tau_es_stella.max(),1e-30):.2f}×)")
        print(f"  ⟨X_HII⟩ STELLA:    {X_HII_stella.mean():.3f}  "
              f"(range {X_HII_stella.min():.2e} - {X_HII_stella.max():.3f})")
        print(f"  ⟨X_HII⟩ photoeq:    {X_HII.mean():.3f}  "
              f"(range {X_HII.min():.2e} - {X_HII.max():.3f})")
        n_neutral_zones = int(np.sum(X_HII < 0.01))
        n_ionized_zones = int(np.sum(X_HII > 0.99))
        print(f"  zones: {n_ionized_zones}/{n_zones} fully ionized (X>0.99); "
              f"{n_neutral_zones}/{n_zones} neutral (X<0.01)")

    return new_snap


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from stella_io import load_stella_snapshot, truncate_to_photosphere

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} mesa.dayNNN.data "
              f"[T_source_K]")
        sys.exit(1)
    path = sys.argv[1]
    T_src = float(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Loading {path} ...")
    snap = load_stella_snapshot(path, verbose=False)
    snap = truncate_to_photosphere(snap, verbose=False)
    print(f"  truncated: {len(snap['r'])} zones, "
          f"r=[{snap['r'][0]:.2e}, {snap['r'][-1]:.2e}] cm, "
          f"T_phot={snap.get('T_phot_inner', snap['T'][0]):.0f} K, "
          f"L_phot={snap.get('L_phot_inner', 0):.3e} erg/s")
    print()

    new_snap = solve_photoionization_equilibrium(
        snap, T_source=T_src, T_eq_floor=None, verbose=True)
