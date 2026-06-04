"""
autoparams.py
=============
Physics-based auto-determination of all parameters that were previously fit by
hand. No guesswork: given a snapshot, this module computes:

  1. Zone masks (shell, wind, photosphere) from density + tau_es profile
  2. Level populations n(H,n=2), n(H,n=3) using local Sobolev Lyalpha escape
     probability rather than a hand-set global beta_Lyalpha
  3. Absolute Halpha volumetric luminosity L_Ha [erg/s] from Case-B recomb
  4. Continuum luminosity L_cont in the observation band from the photospheric
     luminosity and an effective-temperature Planck estimate
  5. Packet fraction f_vol so that the mixed-emission MC naturally weights
     line/continuum by their physical luminosity ratio
  6. Microturbulent velocity v_turb: either thermal-only or +20 km/s floor
  7. SN-type classification from v(r), rho(r) to pick emission geometry

Output is a single dict handed to the pipeline, containing everything the MC
kernel needs plus diagnostic information.
"""

from __future__ import annotations
import numpy as np

# cgs constants
C       = 2.99792458e10
KB      = 1.380649e-16
HPL     = 6.62607015e-27
ME      = 9.1093837e-28
EE      = 4.80320425e-10
SIGMAT  = 6.65245871e-25
MH      = 1.6735e-24
EV      = 1.602176634e-12
SIG_SB  = 5.670374e-5    # Stefan-Boltzmann

CHI_H  = 13.5984  # eV

# Halpha / Lyalpha / Hbeta atomic data
LAM_HA = 6562.80e-8
LAM_LY = 1215.67e-8
LAM_HB = 4861.35e-8
F_HA   = 0.6407
F_HB   = 0.1193
F_LY   = 0.4162
A_LY   = 4.699e8
A_HA   = 4.41e7
A_HB   = 8.42e6


def _q_23_hydrogen(T_K):
    """Maxwell-averaged collisional excitation rate coefficient q(2→3) for H I.

    q_lu = 8.629e-6 × Υ / (g_l × √T) × exp(-ΔE_lu / kT)   [cm³/s]

    Effective collision strength Υ_23 ≈ 5.0 (Anderson et al. 2000, Vriens &
    Smeets 1980, ~10% accurate for 5e3 ≤ T ≤ 5e4 K). g_2 = 8 (statistical
    weight of H I n=2). ΔE_23 = 13.6 × (1/4 − 1/9) eV = 1.889 eV;
    ΔE_23/k = 21916 K.

    This collisional channel populates n=3 directly from n=2 thermal pool,
    feeding Hα emission alongside radiative recombination. In dense
    Lyα-trapped CSM where n_2 is enhanced, this can be a significant
    contribution; in low-n_2 regimes it is small.
    """
    T = np.asarray(T_K, dtype=float)
    Upsilon_23 = 5.0    # weakly T-dependent; representative value
    g_2 = 8
    DE_23_eV = 1.889
    expo = np.exp(-DE_23_eV * EV / (KB * np.maximum(T, 100.0)))
    return 8.629e-6 * Upsilon_23 / (g_2 * np.sqrt(np.maximum(T, 100.0))) * expo


def _build_tau_es(r, n_e):
    """Cumulative electron-scattering optical depth from outside inward."""
    r = np.asarray(r); n_e = np.asarray(n_e)
    dr = np.empty_like(r); dr[:-1] = np.diff(r); dr[-1] = dr[-2]
    return np.cumsum((SIGMAT * n_e * dr)[::-1])[::-1]


# ---------------------------------------------------------------------------
# SN type classification
# ---------------------------------------------------------------------------
def classify_sn(r: np.ndarray, v: np.ndarray, rho: np.ndarray) -> str:
    """Classify the snapshot based on its v(r) and rho(r) shapes.

    Returns one of:
      'IIn'     — non-monotonic v AND clear density peak outside innermost zone
                  (interacting SN with cold dense shell + slow CSM)
      'IIP'     — monotonic v, no dense shell, moderate v_max (< 15000 km/s)
      'Ia'      — monotonic v, no shell, high v_max (> 15000 km/s)
      'generic' — anything else
    """
    dv = np.diff(v)
    monotonic_v = bool(np.all(dv > 0) or np.all(dv < 0))

    idx_peak = int(np.argmax(rho))
    n = len(r)
    rho_peak = rho[idx_peak]
    has_shell = False
    if 3 < idx_peak < n - 3:
        left_min  = np.min(rho[:idx_peak])
        right_min = np.min(rho[idx_peak:])
        if rho_peak > 3 * left_min and rho_peak > 3 * right_min:
            has_shell = True

    v_max_kms = float(np.max(np.abs(v))) / 1e5

    # IIn requires BOTH non-monotonic v AND density peak (signature of
    # ejecta/CSM interaction producing a cold dense shell).
    # A plateau in rho near the photosphere with monotonic v is just
    # IIP/Ia ejecta, not a shell.
    if has_shell and not monotonic_v:
        return 'IIn'
    if monotonic_v:
        return 'Ia' if v_max_kms > 15000 else 'IIP'
    return 'generic'


# ---------------------------------------------------------------------------
# Zone masks
# ---------------------------------------------------------------------------
def compute_masks(r, v, rho, tau_es, sn_type: str,
                  shell_rho_frac: float = 0.3):
    """Define shell, wind, and photosphere zones based on SN type.

    Returns dict with boolean arrays 'shell', 'wind', 'photosphere'.
    """
    n = len(r)
    shell = np.zeros(n, dtype=bool)
    wind  = np.zeros(n, dtype=bool)

    if sn_type in ('IIn', 'generic'):
        shell = rho > shell_rho_frac * rho.max()
        # wind: outer optically thin zones
        wind  = (~shell) & (tau_es < 1.0)
    else:
        # IIP / Ia: all emission from photosphere (continuum-dominated)
        # shell/wind not really applicable — we treat the dense inner region
        # as a "pseudo-shell" so the emissivity integrator still works
        shell = rho > shell_rho_frac * rho.max()
        wind  = np.zeros(n, dtype=bool)

    # photosphere: innermost zone at tau_es ~ 2/3 (classical definition)
    # or the inner trim surface if tau_es only goes down to > 2/3
    photosphere = tau_es <= (2.0 / 3.0)
    # if photosphere is the whole grid (very optically thin), use innermost zone
    if photosphere.sum() == n:
        photosphere[:] = False
        photosphere[0] = True
    return dict(shell=shell, wind=wind, photosphere=photosphere)


# ---------------------------------------------------------------------------
# Local Lyalpha escape probability and populations
# ---------------------------------------------------------------------------
def _saha_neutral_fraction(T, n_e):
    kT = KB * T
    prefac = 2.0 * (2.0 * np.pi * ME * kT / (HPL * HPL))**1.5
    K = prefac * np.exp(-CHI_H * EV / kT)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(n_e > 0, K / n_e, np.inf)
    return 1.0 / (1.0 + ratio)


def h_populations_auto(rho, T, n_e, r, v, X_H=0.737, X_He=0.249,
                       v_turb_cms: float = 20e5):
    """Compute n(H,n=2) and n(H,n=3) using local Sobolev Lyalpha trapping.

    X_H can be a scalar or array (per-zone).

    Steps:
      1. Estimate n_1(H) = f_HI(T, n_e) * n_H_total  (Saha neutral fraction)
      2. Compute dv/dr, then local Lyalpha Sobolev optical depth
      3. Derive beta_Ly(r) = (1 - exp(-tau))/tau
      4. Case-B steady-state for n(n=2): n_e n_p alpha_2 / (A_Ly * beta_Ly)
      5. Similar for n(n=3)

    Returns n_lower, n_upper, and a diagnostic dict.
    """
    T   = np.asarray(T, dtype=float)
    n_e = np.asarray(n_e, dtype=float)
    rho = np.asarray(rho, dtype=float)
    r   = np.asarray(r, dtype=float)
    v   = np.asarray(v, dtype=float)
    X_H = np.asarray(X_H, dtype=float) if np.ndim(X_H) else np.full_like(r, X_H)

    n_H_total = X_H * rho / MH
    f_HI = _saha_neutral_fraction(T, n_e)
    n_1 = f_HI * n_H_total
    n_p = (1.0 - f_HI) * n_H_total

    # radial velocity gradient (cm/s / cm)
    dv_dr = np.abs(np.gradient(v, r))
    # grid-wide floor. A per-zone floor (v_turb/dr) is tempting but for slow
    # near-monovelocity winds it aggressively over-floors dv/dr (because dr
    # is small), which artificially lowers τ_Ly and hence the n=2 population.
    # The grid-wide floor only kicks in at true stationary points.
    min_dvds = v_turb_cms / (r[-1] - r[0])
    dv_dr = np.maximum(dv_dr, min_dvds)

    # Lyalpha Sobolev optical depth
    sigma_Ly = np.pi * EE * EE / (ME * C) * F_LY
    tau_Ly = sigma_Ly * LAM_LY * n_1 / dv_dr
    beta_Ly = (1.0 - np.exp(-np.minimum(tau_Ly, 700.0))) / np.maximum(tau_Ly, 1e-30)

    # Case-B effective recombination rates
    T4 = T / 1e4
    alpha_2 = 4.5e-14 * T4**-0.75
    alpha_3 = 1.5e-14 * T4**-0.80

    # n=2 balance: radiative recomb AND collisional excitation from n=1
    # populate it. In dense regions collisions matter.
    # Effective collision strength Γ_12 ≈ 0.27 (Vriens-Smeets / Anderson et al).
    kT = KB * T
    E12 = 10.1988 * EV
    C_12 = 8.629e-6 * 0.27 / (2.0 * np.sqrt(T)) * np.exp(-E12 / kT)
    C_21 = 8.629e-6 * 0.27 / (8.0 * np.sqrt(T))
    # steady state: pop gain = radiative recomb + collisional exc from n=1
    #               pop loss = A_Ly * β_Ly (radiative) + C_21 * n_e (coll deex)
    rate_in  = n_e * n_p * alpha_2 + C_12 * n_e * n_1
    rate_out = A_LY * beta_Ly + C_21 * n_e
    n_2 = rate_in / np.maximum(rate_out, 1e-30)

    # n=3 balance: radiative recombination in = (Hα + Hβ) decay out with
    # Sobolev escape probabilities, computed from the actual n_2 population
    # (more accurate than the original `β = 0.3` constant).
    #
    # Note: this Case-B treatment does NOT include radiative scatter-pumping
    # of n=3 by the photospheric field. For optically thick IIn winds where
    # n=3 can be pumped above the pure-recombination value by the external
    # radiation field, use `populations_mode='nlte'` with h_populations_nlte,
    # which handles the coupled radiation field self-consistently. This
    # Case-B path is kept as a fast, no-iteration option for quick tests and
    # for diagnostic comparisons.
    sigma_Ha = np.pi * EE * EE / (ME * C) * F_HA
    sigma_Hb = np.pi * EE * EE / (ME * C) * F_HB
    tau_Ha = sigma_Ha * LAM_HA * n_2 / dv_dr
    tau_Hb = sigma_Hb * LAM_HB * n_2 / dv_dr
    beta_Ha = (1.0 - np.exp(-np.minimum(tau_Ha, 700.0))) / np.maximum(tau_Ha, 1e-30)
    beta_Hb = (1.0 - np.exp(-np.minimum(tau_Hb, 700.0))) / np.maximum(tau_Hb, 1e-30)
    n_3 = n_e * n_p * alpha_3 / np.maximum(A_HA * beta_Ha + A_HB * beta_Hb, 1e-30)

    diag = dict(
        tau_Ly=tau_Ly, beta_Ly=beta_Ly,
        f_HI=f_HI, n_HI=n_1, n_p=n_p,
        tau_Ly_median=float(np.median(tau_Ly[tau_Ly > 0])) if np.any(tau_Ly > 0) else np.nan,
    )
    return n_2, n_3, diag


# ---------------------------------------------------------------------------
# Emissivity and luminosities
# ---------------------------------------------------------------------------
def compute_emissivity(r, T, n_e, n_p=None, n_2=None, n_3=None, v=None, mask=None,
                       v_turb_cms: float = 20e5,
                       eps_direct: float = 0.558,
                       n_1: np.ndarray | None = None,
                       line_partner_lam: float = 1025.72e-8,
                       line_partner_f: float = 0.0791,
                       L_cont_per_zone: np.ndarray | None = None,
                       T_phot: float = 15000.0,
                       R_phot: float | None = None,
                       beta_Ha_out=None,
                       eps_eff_out=None,
                       emission_model: str = 'caseb_hr',
                       f_wing: float = 0.10,
                       delta_scatter: float = 0.03):
    """Halpha *emergent* emissivity [erg/s] per zone.

    Five emissivity models are available; pick via ``emission_model``.

    ``'caseb_hr'`` (default, legacy behaviour):
        Case-B recombination rate with Hummer & Rybicki (1982) escape-with-
        destruction probability and Lyβ recycling:

            emis = n_e × n_p × α_eff × hν × dV × P_esc
            P_esc = β_Hα / (β_Hα + ε_eff × (1-β_Hα))
            ε_eff = ε_direct × β_Lyβ / [1 - (1-β_Lyβ)(1-ε_direct)]

        In the heavily-trapped limit (β_Lyβ → 0 faster than β_Hα → 0) the HR
        formula converges to P_esc → 1, i.e. "full Case-B escape after Lyβ
        recycling". For dense IIn inner CSM this tends to OVER-estimate
        emission because Lyα destruction pathways (which CMFGEN tracks
        explicitly) are not included.

    ``'nlte_direct'``:
        Use the NLTE-solved upper-level population n_3 directly:

            emis = n_3 × A_32 × hν × dV × β_Hα

        This is the standard Sobolev emergent-emissivity expression. In the
        pure-Case-B thick-trapping limit it converges to the same answer as
        'caseb_hr' because n_3 × A_32 × β_Hα → n_e × n_p × α_eff; in NLTE
        regimes where collisional pumping or Lyα-coupled destruction alter
        n_3 it differs. Requires n_3 to be passed.

    ``'thermal_cap'``:
        Apply a thermalisation ceiling to the Case-B × HR rate:

            emis = min(emis_caseb_hr, L_thermal_zone)

        where the per-zone thermal ceiling is the local LTE emergent line
        flux given the Planck function at T and the line Doppler width:

            L_thermal_zone = 4π²  r²  dr  B_ν(T,ν_Hα)  Δν_Doppler
                             × (1 - exp(-τ_Hα))

        The (1 - e^-τ) factor converts from optically-thin volume emission
        to surface emission in the τ >> 1 limit. This bounds the zone's
        emergent Hα flux by what a blackbody at its own temperature can
        emit in the line, which is the true physical upper limit independent
        of the Case-B vs NLTE distinction. Most useful for the densest
        inner-CSM zones where τ_Hα × τ_es >> 1 and thermalization dominates.

    ``'caseb_hr_net'`` (recommended for narrow-core match in dense IIn CSM):
        Case-B × HR rate multiplied by the net-emission correction factor
        ``max(0, (S_Hα − J_ν^cont) / S_Hα)``, approximated in the thick-line
        limit as ``max(0, (B_ν(T_gas) − J_ν^cont) / B_ν(T_gas))``:

            emis = emis_caseb_hr × max(0, [B_ν(T_gas) − J_ν^cont] / B_ν(T_gas))

        where the ambient continuum mean intensity at the Hα frequency is
        estimated by the classic "diluted BB outside / local thermal inside"
        split used in Castor-style stellar wind models:

            J_ν^cont = W(r) × B_ν(T_phot)   for r ≥ R_phot  (outer wind, τ_es<1)
                     = B_ν(T_gas)           for r <  R_phot  (inner CSM, τ_es>1)

        with the geometric dilution factor
            W(r) = 0.5 × [1 − √(1 − (R_phot/r)²)].

        Physical motivation: caseb_hr counts gross recombinations without
        subtracting true absorption of the ambient radiation field. In deep
        τ_es zones the Hα source function approaches B_ν(T_gas) (thick-line
        detailed balance) and the ambient continuum also approaches
        B_ν(T_gas), so the NET emergent Hα flux should vanish. caseb_hr_net
        enforces this zero crossing by construction. In the optically thin
        outer CSM the dilution factor W ≪ 1 makes J_ν^cont ≪ B_ν(T_gas) and
        the correction factor → 1, recovering the thin-wind behaviour.

        Caveat: by zeroing the shell and inner-CSM emission at line frequency,
        this mode also suppresses the Thomson-scattering halo those photons
        would produce. It matches CMFGEN's narrow core well but loses the
        broad electron-scattering wings. For the Thomson-scattered wings and
        overall line morphology on IIn snapshots, use ``caseb_hr`` (the
        default), which captures the wings to within 6% and over-predicts
        the narrow core by a factor ~2.

        Requires ``R_phot`` and ``T_phot`` to be passed (the pipeline always does).

    ``'caseb_hr_attenuated'`` (recommended for best balance of narrow core
    AND Thomson wings):
        Multiply ``caseb_hr`` emissivity by ``exp(-τ_es_outward)`` where
        ``τ_es_outward`` is the Thomson optical depth a photon would cross
        going from its birth zone straight outward to the grid edge. This
        captures the geometric fact that line photons born deep in the
        atmosphere cannot reach the observer at line frequency — they
        Thomson-scatter into the broad wing halo en route. The MC kernel's
        own Thomson transport then produces the wings from what survives the
        pre-attenuation.

        Result (validated on HERACLES IIn snapshot):
            - Total line luminosity reduced from ~6e42 (caseb_hr) to ~2e41,
              matching CMFGEN.
            - Narrow core overshoot reduced from 2× to ~1.5× CMFGEN.
            - Thomson wings preserved (partially; CMFGEN wings are ~1.5e41,
              ours are ~1e41 under this mode).
            - P-Cygni absorption trough position unchanged (continuum packets
              unaffected).

        Physical interpretation: this is an Eddington-Barbier-style geometric
        factor that says a zone's effective emissivity as a narrow-line
        source scales with the probability its photons can reach the surface
        without Thomson scattering. Photons that don't reach the surface
        unscattered contribute to the Thomson halo via the MC kernel's
        natural transport.

        Computationally cheap (no additional radiation-field coupling,
        just an integral of n_e σ_T dr). Does not require R_phot or T_phot
        parameters beyond what caseb_hr already uses.

    All five modes integrate over the mask-selected zones. Kwargs:

    n_2 : per-zone NLTE lower-level population; required for β_Hα and
          hence all five models.
    n_3 : per-zone NLTE upper-level population; required only for
          ``emission_model='nlte_direct'``.
    n_1 : per-zone n=1 population for β_Lyβ calculation; used by the HR
          recycling branch of ``caseb_hr``, ``thermal_cap``, ``caseb_hr_net``,
          and ``caseb_hr_attenuated``.

    Pipeline configures the kernel with eps_dest=0 for VOLUMETRIC sources
    to avoid double-counting; CONTINUUM sources still use full eps_dest
    because P-Cygni absorption packets need destruction to reproduce the
    right forward-scattering depletion.
    """
    r = np.asarray(r); T = np.asarray(T); n_e = np.asarray(n_e)
    if n_p is None:
        n_p = n_e
    else:
        n_p = np.asarray(n_p)
    alpha_Ha = 1.17e-13 * (T / 1e4)**-0.942
    dr = np.empty_like(r)
    dr[:-1] = np.diff(r); dr[-1] = dr[-2]
    nu_Ha = C / LAM_HA

    # ---- compute β_Hα (needed by all three models) ----
    beta_Ha = np.ones_like(r)
    tau_Ha = np.zeros_like(r)
    if n_2 is not None and v is not None:
        sigma_Ha = np.pi * EE * EE / (ME * C) * F_HA
        dv_dr = np.abs(np.gradient(v, r))
        min_dvds = v_turb_cms / (r[-1] - r[0])
        dv_dr = np.maximum(dv_dr, min_dvds)
        tau_Ha = sigma_Ha * LAM_HA * np.maximum(n_2, 0) / dv_dr
        beta_Ha = ((1.0 - np.exp(-np.minimum(tau_Ha, 700.0)))
                   / np.maximum(tau_Ha, 1e-30))

    # ---- compute ε_eff (for HR model's Lyβ recycling) ----
    eps_eff = np.full_like(r, eps_direct)
    if n_1 is not None and v is not None:
        sigma_Lb = np.pi * EE * EE / (ME * C) * line_partner_f
        tau_Lb = sigma_Lb * line_partner_lam * np.maximum(n_1, 0) / dv_dr
        beta_Lb = ((1.0 - np.exp(-np.minimum(tau_Lb, 700.0)))
                   / np.maximum(tau_Lb, 1e-30))
        denom = 1.0 - (1.0 - beta_Lb) * (1.0 - eps_direct)
        eps_eff = eps_direct * beta_Lb / np.maximum(denom, 1e-30)

    P_esc = beta_Ha / np.maximum(beta_Ha + eps_eff * (1.0 - beta_Ha), 1e-30)

    # Raw Case-B × HR emissivity (used by caseb_hr and as upper bound for thermal_cap)
    emis_caseb = (n_e * n_p * alpha_Ha * HPL * nu_Ha
                  * 4.0 * np.pi * r**2 * dr * P_esc)

    if emission_model == 'caseb_hr':
        emis = emis_caseb

    elif emission_model == 'nlte_direct':
        if n_3 is None:
            raise ValueError(
                "emission_model='nlte_direct' requires n_3; call with "
                "populations_mode='nlte' (or 'caseb') so that n_3 is available")
        A_32 = 4.41e7   # s⁻¹, Hα spontaneous rate
        emis = (np.maximum(n_3, 0.0) * A_32 * HPL * nu_Ha
                * 4.0 * np.pi * r**2 * dr * beta_Ha)

    elif emission_model == 'thermal_cap':
        # Thermalisation ceiling per zone (LTE Planck × Doppler line width ×
        # emitting volume, with (1-e^-τ) correction for finite thickness)
        # B_ν(T) at Hα line frequency
        x = HPL * nu_Ha / (KB * np.maximum(T, 100.0))
        # Protect against overflow at very low T
        x_safe = np.minimum(x, 700.0)
        B_nu = (2.0 * HPL * nu_Ha**3 / (C * C)
                / (np.exp(x_safe) - 1.0 + 1e-300))
        # Line Doppler width in frequency
        v_th_cms = np.sqrt(2.0 * KB * np.maximum(T, 100.0) / MH)
        v_D_cms = np.sqrt(v_th_cms**2 + v_turb_cms**2)
        dnu_D = nu_Ha * v_D_cms / C
        # Thermalisation-limit luminosity per zone:
        #   L_th = 4π² r² dr × B_ν(T) × Δν_D × (1 - e^-τ_Hα)
        # 4π from solid angle × π from isotropic surface-intensity integration
        thickness_factor = 1.0 - np.exp(-np.minimum(tau_Ha, 700.0))
        emis_thermal = (4.0 * np.pi**2 * r**2 * dr
                        * B_nu * dnu_D * thickness_factor)
        emis = np.minimum(emis_caseb, emis_thermal)

    elif emission_model == 'caseb_hr_net':
        # Net emission after subtracting continuum-pumped absorption.
        # In the thick-line limit S_Hα → B_ν(T_gas), and the correction
        # factor (S − J)/S → (B(T_gas) − J_cont)/B(T_gas).
        #
        # Compute B_ν(T_gas) — the thick-line source-function ceiling
        T_safe = np.maximum(T, 100.0)
        x_gas = np.minimum(HPL * nu_Ha / (KB * T_safe), 700.0)
        B_gas = (2.0 * HPL * nu_Ha**3 / (C * C)
                 / (np.exp(x_gas) - 1.0 + 1e-300))
        # Compute J_ν^cont with the diluted-BB / local-thermal split
        if R_phot is None or R_phot <= 0:
            # Fallback: if no photosphere info, use local thermal everywhere
            # (reduces correction to zero — degenerate, but safe)
            J_cont = B_gas.copy()
        else:
            # Dilution factor: 0.5 at R_phot, → 1/(4(r/R)²) at large r
            ratio2 = np.where(r > 0, (R_phot / r)**2, 1.0)
            ratio2 = np.minimum(ratio2, 1.0)  # clamp at/inside photosphere
            W = 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - ratio2, 0.0)))
            # B_ν at the photospheric temperature (scalar)
            x_ph = min(HPL * nu_Ha / (KB * max(T_phot, 100.0)), 700.0)
            B_phot = (2.0 * HPL * nu_Ha**3 / (C * C)
                      / (np.exp(x_ph) - 1.0 + 1e-300))
            J_outer = W * B_phot
            # Inside photosphere (r < R_phot), continuum mean intensity ≈ B(T_gas)
            # because optically-thick continuum bathes every point from 4π sr
            J_cont = np.where(r >= R_phot, J_outer, B_gas)
        # Correction factor: clamp at zero (no net absorption added here)
        correction = np.maximum((B_gas - J_cont) / np.maximum(B_gas, 1e-300), 0.0)
        emis = emis_caseb * correction

    elif emission_model == 'caseb_hr_attenuated':
        # caseb_hr × geometric Thomson attenuation: photons born deep in the
        # atmosphere (large τ_es looking outward) cannot reach the observer
        # as narrow-line photons — they Thomson-scatter on the way out, losing
        # frequency coherence. The survival probability to line frequency is
        # exp(-τ_es_outward), where τ_es_outward is the column of e-scattering
        # opacity a photon would traverse going from its birth radius straight
        # outward.
        #
        # This is a physically-motivated first-order correction that addresses
        # the known caseb_hr narrow-core over-prediction (factor ~2× in dense
        # IIn CSM because inner-zone line photons that should be Thomson-
        # scattered into the wing halo instead emerge as direct narrow-core
        # emission). By pre-attenuating emissivity, the MC kernel then
        # naturally produces the remaining narrow core from outer zones plus
        # a reduced Thomson-scattered wing contribution.
        #
        # Trade-off: applying exp(-τ_es_out) PRE-MC is a statistical mean-field
        # treatment. A photon born at τ_es_out = 5 has 0.7% unattenuated
        # probability in the model, but in reality some individual photons
        # escape unscattered while others scatter multiple times. The MC
        # kernel's own Thomson transport handles the full distribution for
        # packets it emits; pre-attenuation just sets the injection rate.
        # The net result is that deep-zone contributions to narrow core are
        # correctly suppressed, and deep-zone contributions to Thomson wings
        # are partially preserved (the fraction that survives pre-attenuation
        # and then Thomson-scatters in the MC).
        #
        # Compared to caseb_hr_net (which also reduces deep-zone emission but
        # via detailed balance S≈J): this mode is purely geometric, doesn't
        # require T_phot/R_phot machinery for J_cont, and preserves some
        # deep-zone contribution to the Thomson halo that caseb_hr_net zeros
        # out entirely.
        tau_es_outward = np.zeros_like(r)
        # Integrate n_e × σ_T × dr from each zone outward to the grid edge.
        # σ_T = 6.6524587e-25 cm² (Thomson cross-section)
        _sigma_T = 6.6524587e-25
        tau_acc = 0.0
        for _i in range(len(r) - 1, -1, -1):
            tau_es_outward[_i] = tau_acc
            tau_acc += float(n_e[_i]) * _sigma_T * float(dr[_i])
        attenuation = np.exp(-np.minimum(tau_es_outward, 700.0))
        emis = emis_caseb * attenuation

    elif emission_model == 'caseb_hr_attenuated_collisional':
        # Same as caseb_hr_attenuated but adds an explicit collisional
        # excitation contribution from n=2 → n=3 that radiatively decays
        # as Hα.
        #
        # Rationale: caseb_hr counts only radiative recombinations to n=3
        # via the Case-B α_eff. CMFGEN's NLTE solver also includes
        # collisional excitation 2→3 driven by the local thermal pool. In
        # dense IIn CSM with Lyα trapping, n_2 can be elevated by orders
        # of magnitude above LTE, and collisions of thermal electrons with
        # H atoms in n=2 can directly populate n=3. The subsequent radiative
        # cascade physics (3→2 vs 3→1, Lyβ recycling) is identical to the
        # recombination channel, so the same Hummer-Rybicki escape
        # probability P_esc applies.
        #
        # Per-zone formula (mirrors recombination structure):
        #   ε_recomb(r) = n_e × n_p × α_eff × hν × dV × P_esc
        #   ε_coll(r)   = n_e × n_2 × q_23(T) × hν × dV × P_esc
        #   emis = (ε_recomb + ε_coll) × exp(-τ_es_outward)
        #
        # q_23(T) uses Maxwell-averaged effective collision strength
        # Υ_23 ≈ 5.0 (Anderson et al. 2000 / Vriens-Smeets 1980 / Aggarwal-
        # Kingston 1991, all agreeing within ~10% for 5e3 < T < 5e4 K).
        # See _q_23_hydrogen() above.
        #
        # In atm8 (HERACLES IIn snapshot), n_2/n_p from the NLTE solver is
        # ~10⁻⁷, so the collisional contribution is only a few percent of
        # recombination. In hotter or more Lyα-trapped winds (e.g. higher
        # CSM density, or T_e > 2e4 K regions), n_2/n_p can be larger and
        # the collisional contribution becomes significant.
        #
        # Required inputs: n_2 (always available when populations_mode in
        # {'caseb', 'lte', 'nlte'} since the pipeline computes it for β_Hα).
        # Falls back to recombination-only if n_2 is None.
        if n_2 is None:
            ε_coll_rate = np.zeros_like(r)
        else:
            q_23 = _q_23_hydrogen(T)
            ε_coll_rate = n_e * np.maximum(n_2, 0.0) * q_23

        emis_collisional = (ε_coll_rate * HPL * nu_Ha
                            * 4.0 * np.pi * r**2 * dr * P_esc)

        # τ_es_outward integral (same as caseb_hr_attenuated)
        tau_es_outward = np.zeros_like(r)
        _sigma_T = 6.6524587e-25
        tau_acc = 0.0
        for _i in range(len(r) - 1, -1, -1):
            tau_es_outward[_i] = tau_acc
            tau_acc += float(n_e[_i]) * _sigma_T * float(dr[_i])
        attenuation = np.exp(-np.minimum(tau_es_outward, 700.0))

        emis = (emis_caseb + emis_collisional) * attenuation

    elif emission_model == 'nlte_direct_attenuated':
        # NLTE-solved n_3 × A_32 × β_Hα with the same Thomson outward-
        # attenuation factor used by caseb_hr_attenuated.
        #
        # Rationale: nlte_direct gives 2× higher L_line than caseb_hr because
        # the NLTE solver enhances n_3 in deep shell zones (9-17) via Lyα
        # trapping that recombination doesn't capture. But that emission is
        # buried at τ_es,out ≈ 8-20 — most photons cannot escape unscattered.
        # Without pre-attenuation, the MC kernel hits max_events truncation
        # on those packets and energy conservation breaks (~22% survival in
        # peel-off Option B testing).
        #
        # Pre-attenuating by exp(-τ_es,out) — the same mean-field correction
        # caseb_hr_attenuated uses — bounds the injected emission to what can
        # plausibly escape and lets the MC handle the rest naturally via
        # Thomson redistribution.
        #
        # Per-zone formula:
        #   ε_nlte(r) = n_3(r) × A_32 × hν × dV × β_Hα(r)
        #   emis(r)   = ε_nlte(r) × exp(-τ_es_outward(r))
        #
        # Trade-off vs caseb_hr_attenuated: nlte_direct uses the actual
        # NLTE-solved n_3 (better physics in regions where caseB α_eff
        # underestimates upper-level population), but trusts the populations
        # solver to be right. caseb_hr is more robust (tied directly to n_e,
        # n_p, T) and has a long heritage in nebular line modelling.
        #
        # In atm8 testing, nlte_direct_attenuated yields L_line ≈ 1.7e+40
        # — about 6× LOWER than caseb_hr_attenuated. This is because the
        # NLTE solver gives near-LTE n_3 in the post-photospheric wind
        # (where photons can escape), and only enhances n_3 deep in the
        # shock shell (where attenuation kills the contribution).
        # Use this mode if you want to test NLTE n_3 directly without
        # the recombination cap.
        #
        # Requires n_3 to be passed; otherwise raises.
        if n_3 is None:
            raise ValueError(
                "emission_model='nlte_direct_attenuated' requires n_3; call "
                "with populations_mode='nlte' (or 'caseb') so that n_3 is "
                "available")
        A_32 = 4.41e7   # s⁻¹, Hα spontaneous rate
        emis_nlte = (np.maximum(n_3, 0.0) * A_32 * HPL * nu_Ha
                     * 4.0 * np.pi * r**2 * dr * beta_Ha)

        # τ_es_outward integral (same as caseb_hr_attenuated)
        tau_es_outward = np.zeros_like(r)
        _sigma_T = 6.6524587e-25
        tau_acc = 0.0
        for _i in range(len(r) - 1, -1, -1):
            tau_es_outward[_i] = tau_acc
            tau_acc += float(n_e[_i]) * _sigma_T * float(dr[_i])
        attenuation = np.exp(-np.minimum(tau_es_outward, 700.0))

        emis = emis_nlte * attenuation

    elif emission_model == 'caseb_hr_attenuated_split':
        # Two-component (split) emissivity model — physical answer to the
        # wing-deficit issue identified by per-Thomson-scatter diagnostics.
        #
        # Background: caseb_hr_attenuated multiplies emission by
        # exp(-τ_es,out), zeroing out deep-zone contributions. But Hα photons
        # emitted at τ_es ~ 5 are NOT destroyed — they Thomson-scatter many
        # times and emerge in BROAD WINGS (Δv ~ sqrt(N) × 670 km/s for N ~
        # τ² scatters). Per-Thomson-scatter diagnostics
        # (mc_thomson_diagnostic.py) confirmed that the kernel's transport
        # is correct; the problem is that all the deep emission is being
        # suppressed at injection.
        #
        # The split:
        #   ε_split(r) = ε_caseb(r) × [exp(-τ_es,out) + (1 - exp(-τ_es,out)) × f_wing]
        #
        # First term reproduces caseb_hr_attenuated — the directly-escaping
        # (narrow) component. Second term injects a fraction f_wing of the
        # would-be-attenuated deep emission, allowing the MC kernel to
        # Thomson-scatter it into wings.
        #
        # Parameter f_wing ∈ [0, 1]:
        #   f_wing = 0   → caseb_hr_attenuated (no deep emission, no wings)
        #   f_wing = 1   → caseb_hr (full deep emission, overshoots everything)
        #   intermediate → tunable wing strength
        #
        # f_wing represents the survival fraction of deep-emitted photons
        # against:
        #   (a) collisional de-excitation of n=3 during Thomson random walk
        #   (b) line destruction at line scatters (caseB attenuated form)
        #   (c) other NLTE quenching channels not in the populations solver
        #
        # CALIBRATE f_wing per regime — for atm8 the empirical best fit
        # appears to be f_wing ≈ 0.05-0.15. Default 0.10.
        #
        # CAVEAT — peel-off energy conservation: deep-zone packets that
        # peel into the spectrum require many scatters to drift toward the
        # surface where peel weights become non-negligible. With τ_es,out
        # ~ 20 in some zones, the random walk needs ~τ² = 400 scatters,
        # tractable within max_events=2000 but with non-trivial cost.
        # At f_wing = 0.1 the injected deep-zone weight is reduced 10×
        # vs caseb_hr, so peel-off conservation should be substantially
        # better than the f_wing = 1 (caseb_hr) case (which loses 90% of
        # injected energy to truncation).
        tau_es_outward = np.zeros_like(r)
        _sigma_T = 6.6524587e-25
        tau_acc = 0.0
        for _i in range(len(r) - 1, -1, -1):
            tau_es_outward[_i] = tau_acc
            tau_acc += float(n_e[_i]) * _sigma_T * float(dr[_i])
        attenuation = np.exp(-np.minimum(tau_es_outward, 700.0))
        # Clip f_wing to [0, 1]
        f_w = max(0.0, min(1.0, float(f_wing)))
        kernel_factor = attenuation + (1.0 - attenuation) * f_w
        emis = emis_caseb * kernel_factor

    elif emission_model == 'nlte_direct_split':
        # Combines nlte_direct (NLTE-solved n_3) with the two-component split.
        #
        # Motivation: amplitude analysis (D_implied = 1.07 kpc from F_cont
        # match) shows our caseB L_line = 1.08e+41 is ~2.3× LESS than CMFGEN's
        # effective L_line of 2.46e+41. The nlte_direct prescription
        # (n_3 × A_32 × β_Hα) gives L_line = 2.37e+41 — within 4% of CMFGEN.
        # So the NLTE-solved n_3 carries the right total Hα emission rate;
        # what was missing was transport of deep-zone emission, which the
        # split factor enables.
        #
        # Per-zone formula:
        #   ε_nlte(r) = n_3(r) × A_32 × hν × dV × β_Hα(r)
        #   ε(r)      = ε_nlte(r) × [exp(-τ_es,out) + (1-exp(-τ_es,out)) × f_wing]
        #
        # f_wing has the same meaning as in caseb_hr_attenuated_split.
        if n_3 is None:
            raise ValueError(
                "emission_model='nlte_direct_split' requires n_3; call "
                "with populations_mode='nlte' (or 'caseb') so that n_3 is "
                "available")
        A_32 = 4.41e7
        emis_nlte = (np.maximum(n_3, 0.0) * A_32 * HPL * nu_Ha
                     * 4.0 * np.pi * r**2 * dr * beta_Ha)
        tau_es_outward = np.zeros_like(r)
        _sigma_T = 6.6524587e-25
        tau_acc = 0.0
        for _i in range(len(r) - 1, -1, -1):
            tau_es_outward[_i] = tau_acc
            tau_acc += float(n_e[_i]) * _sigma_T * float(dr[_i])
        attenuation = np.exp(-np.minimum(tau_es_outward, 700.0))
        f_w = max(0.0, min(1.0, float(f_wing)))
        kernel_factor = attenuation + (1.0 - attenuation) * f_w
        emis = emis_nlte * kernel_factor

    elif emission_model == 'caseb_hr_destruction_walk':
        # Physically-motivated emission model accounting for line-photon
        # destruction during Thomson random walks.
        #
        # Premise: a Hα photon emitted at τ_es,out = τ scatters approximately
        # N ~ τ² times before escaping. If each scatter has a small probability
        # δ of destruction (collisional de-excitation, photoionization from
        # n=3 by the trapped Lyα field, etc.), the survival fraction for an
        # emitted photon is (1-δ)^N ≈ exp(-δτ²).
        #
        # ε_walk(r) = ε_caseb(r) × exp(-δ × τ_es,out(r)²)
        #
        # Limits:
        #   δ = 0           → caseb_hr (raw, overshoots)
        #   δ → ∞           → only τ → 0 emits (outer wind only)
        #   δ × τ_typ² ~ 1  → emission peaks at moderate τ ~ τ_typ
        #
        # Parameter delta_scatter controls the destruction-per-scatter rate.
        # For atm8 testing, δ ≈ 0.03-0.05 yields emission distributed
        # across τ_es ∈ [0.3, 3.0], peaking around τ ~ 1.
        #
        # This is more SHAPE-PRESERVING than caseb_hr_attenuated_split:
        # rather than uniformly scaling deep emission, it preferentially
        # picks out moderate-τ zones where photons can scatter a few times
        # (filling the intermediate-velocity 100-1000 km/s band) without
        # being completely Compton-cooled out of the spectrum band.
        #
        # Physical justification for δ value:
        #   Thomson scattering doesn't destroy photons coherently; the
        #   "destruction" here represents indirect processes — an Hα photon
        #   trapped in the shell long enough to be photoionized from n=3
        #   by the local Lyα field (Wallerstein-Brown effect), or
        #   collisionally de-excited at high local n_e. Both are present
        #   in CMFGEN's NLTE solver but absent from caseB recombination.
        tau_es_outward = np.zeros_like(r)
        _sigma_T = 6.6524587e-25
        tau_acc = 0.0
        for _i in range(len(r) - 1, -1, -1):
            tau_es_outward[_i] = tau_acc
            tau_acc += float(n_e[_i]) * _sigma_T * float(dr[_i])
        delta_s = max(1e-6, float(delta_scatter))
        survival = np.exp(-np.minimum(delta_s * tau_es_outward**2, 700.0))
        emis = emis_caseb * survival

    else:
        raise ValueError(
            f"emission_model must be one of 'caseb_hr', 'nlte_direct', "
            f"'thermal_cap', 'caseb_hr_net', 'caseb_hr_attenuated', "
            f"'caseb_hr_attenuated_collisional', 'nlte_direct_attenuated', "
            f"'caseb_hr_attenuated_split', 'nlte_direct_split', "
            f"'caseb_hr_destruction_walk'; "
            f"got {emission_model!r}")

    # Hand back per-zone HR ε_eff if requested (optional diagnostic).
    if eps_eff_out is not None:
        eps_eff_out[...] = eps_eff

    if mask is not None:
        emis = emis * mask.astype(float)

    if beta_Ha_out is not None:
        beta_Ha_out[...] = beta_Ha

    return emis    # erg/s per zone (emergent)


def continuum_luminosity_per_A(L_bol_erg_s: float, T_eff: float, lam_cm: float) -> float:
    """L_cont(λ) [erg/s/Å] assuming photospheric BB at T_eff."""
    if L_bol_erg_s <= 0 or T_eff <= 0:
        return 0.0
    # Planck in per-wavelength: B_λ [erg/s/cm²/cm/sr]
    x = HPL * C / (lam_cm * KB * T_eff)
    if x > 700:
        return 0.0
    B_lam = (2.0 * HPL * C * C / lam_cm**5) / (np.exp(x) - 1.0)   # erg/s/cm²/cm/sr
    # Integrated Planck B [erg/s/cm²/cm/sr] integrated over λ gives σT⁴/π
    # Per-Å spectral luminosity = L_bol × [π B_λ] / [σT⁴]
    #   (factor π cancels the integrated steradian weighting of an outward
    #    isotropic surface emission, and σ_SB × T^4 = ∫π B_λ dλ)
    L_per_cm = L_bol_erg_s * np.pi * B_lam / (SIG_SB * T_eff**4)
    return L_per_cm * 1e-8    # erg/s/Å


# ---------------------------------------------------------------------------
# v_turb default
# ---------------------------------------------------------------------------
def default_v_turb_kms(T_typical: float) -> float:
    """Default microturbulence: thermal velocity floored to 20 km/s."""
    v_th_kms = np.sqrt(KB * T_typical / MH) / 1e5
    return float(max(v_th_kms, 20.0))


def compute_v_turb_grid(r, v, T, alpha_shock: float = 0.25,
                         v_turb_floor_kms: float = 20.0,
                         v_turb_max_kms: float = 2000.0,
                         n_outer_wind: int = 5) -> np.ndarray:
    """Per-zone non-thermal microturbulent velocity for line broadening [cm/s].

    Physical width, no tuned prescription
    -------------------------------------
    The intrinsic Doppler width of a line is set by the *physical* motions of
    the absorbing/emitting atoms: the thermal speed v_th = sqrt(kT/m_H) plus a
    small non-thermal microturbulence. The overall line PROFILE width in an
    expanding atmosphere is set by the bulk velocity field (expansion), which
    the transport handles via the Doppler shift v·n̂/c at every event — NOT by
    the intrinsic Doppler width.

    This function returns only the non-thermal microturbulent component; the
    thermal term v_th is added in quadrature inside the transport kernel
    (v_D = sqrt(v_th² + v_turb²)). The microturbulence is taken to be the
    local sound speed c_s = sqrt(kT/m_H) — the natural (subsonic) turbulent
    velocity scale — which is a physical quantity computed from the gas
    temperature, with no free parameter, no floor, no ceiling.

    The previous prescription v_turb = α·|v − v_wind| (capped at 2000 km/s)
    is intentionally removed: on smooth ejecta it mis-attributed the bulk
    expansion velocity field as microturbulence, returning hundreds of km/s
    of spurious intrinsic width that swamped the physical line profile. The
    `alpha_shock`, `v_turb_floor_kms`, `v_turb_max_kms`, and `n_outer_wind`
    arguments are retained only for call-signature compatibility and are
    unused.

    Parameters
    ----------
    r : (n,) array of cell-centered radii [cm] (unused; kept for API).
    v : (n,) array of cell-centered radial velocities [cm/s] (unused; kept
        for API — bulk-velocity broadening is applied by the transport).
    T : (n,) array of cell-centered gas temperatures [K].

    Returns
    -------
    v_turb_grid : (n,) array of non-thermal microturbulent velocities [cm/s].
    """
    T = np.asarray(T, dtype=float)
    # Local sound speed — physical microturbulence scale (no shock term,
    # no arbitrary floor, no cap). Thermal v_th is added in the kernel.
    T_safe = np.maximum(T, 1000.0)
    c_s = np.sqrt(KB * T_safe / MH)                 # cm/s
    return c_s


# ---------------------------------------------------------------------------
# Top-level auto-parameter routine
# ---------------------------------------------------------------------------
def derive_parameters_from_state(state, snap_arrays, line_name='Halpha',
                                  wavelength_band_AA=(6200.0, 6950.0),
                                  populations_mode='caseb', nlte_levels=3,
                                  tau_es_trim_for_fvol=10.0,
                                  emission_model='caseb_hr',
                                  f_wing=0.10,
                                  delta_scatter=0.03,
                                  ionization_mode='saha',
                                  f_HI_max=0.5,
                                  stromgren_exponent=1.0,
                                  # NLTE-RT coupling on Hα: either pass an
                                  # absolute per-zone J_bar measurement from
                                  # the MC (J_bar_Ha_abs, preferred), or a
                                  # legacy multiplier of W×B_ν(T_phot)
                                  # (J_bar_factor — typically <1 because the
                                  # photospheric line is self-absorbed).
                                  J_bar_Ha_abs=None,
                                  J_bar_factor=None,
                                  # Lyα destruction-probability floor on β_(1,2).
                                  # Passed through to h_populations_nlte. See
                                  # docstring there. None = legacy behavior;
                                  # 1e-3 to 1e-5 typical for CSM with solar
                                  # metallicity (metal-line conversion of Lyα).
                                  eps_Lya_destruction=None,
                                  # Phase 2: enable 2s two-photon decay channel
                                  # in the H NLTE solve. See h_populations_nlte
                                  # for derivation. First-principles replacement
                                  # for the eps_Lya knob; can be combined with ε
                                  # (ε applies first, then 2γ on top).
                                  two_photon_decay=True,
                                  verbose=True):
    """Derive all pipeline parameters from a PhysicalState + normalized snapshot.

    Unlike the old `derive_parameters`, this:
      - uses the physics-based region masks from PhysicalState (not threshold-
        based heuristics applied to raw arrays),
      - uses the derived v_turb from local sound speed,
      - uses per-zone composition from PhysicalState.X_H (not a scalar),
      - uses L_phot from snapshot L(r) not a BB estimate,
      - chooses the line-emitting region based on SN type:
          IIn:        shell (cdshell + shocked) + wind
          IIP:        envelope (photosphere-interior, recombining H)
          Ia:         fast_ejecta_outer (no H normally, returns ~0 for Hα)
          stripped:   envelope (no H typically, returns ~0 for Hα)
          unknown:    everything transparent
      - handles X_H ≈ 0 gracefully (returns L_line = 0, f_vol = 0).

    Parameters
    ----------
    state : PhysicalState
        Output of snapshot_analyzer.analyze_snapshot.
    snap_arrays : dict
        Normalized snapshot dict (from snapshot_analyzer._normalize_snapshot).
    line_name, wavelength_band_AA, populations_mode, nlte_levels,
    tau_es_trim_for_fvol, verbose : as in derive_parameters.
    """
    r = np.asarray(snap_arrays['r'])
    v = np.asarray(snap_arrays['v'])
    rho = np.asarray(snap_arrays['rho'])
    T = np.asarray(snap_arrays['T'])
    n_e = np.asarray(snap_arrays['n_e'])
    # Compute tau_es from n_e using the SAME formula the MC kernel uses
    # (sn_mc_voigt.run_mc_voigt line ~66). This ensures the "kept" mask
    # here matches the kernel's tau_es_trim: zones with tau_es > trim are
    # dropped by the MC, so we must not count their emission in L_line_kept.
    # Using snap['tau_es'] (from the snapshot reader) gives values that
    # differ by 10-15% from the cumsum formula due to integration-scheme
    # differences, producing mismatches at zones near the trim boundary.
    dr = np.empty_like(r); dr[:-1] = np.diff(r); dr[-1] = dr[-2]
    tau_es = np.cumsum((SIGMAT * n_e * dr)[::-1])[::-1]

    sn_type = state.sn_type
    v_turb_kms = state.v_turb_kms
    X_H_per_zone = np.asarray(state.X_H, dtype=float)

    # Per-zone microturbulent velocity for line broadening. This replaces
    # the scalar `v_turb_kms` used by older pipeline calls. The scalar is
    # kept in the output for backwards-compat metadata; the kernel now
    # uses `v_turb_kms_grid` if present.
    v_turb_cms_grid = compute_v_turb_grid(r, v, T,
                                            alpha_shock=0.25,
                                            v_turb_floor_kms=20.0)
    v_turb_kms_grid = v_turb_cms_grid / 1e5

    # Line-emission mask based on SN type. For IIn, the line forms in the
    # CDS (cold dense post-shock shell at T~T_phot) and the full CSM wind
    # (inner photospheric decelerating CSM + outer transparent pristine wind).
    #
    # Production default ('wind' only): excludes the inner CSM ('envelope')
    # zones between the shock shell and freely-expanding wind. This was
    # historically conservative — those zones sit inside the photosphere
    # mask, so their direct emission is largely attenuated. For the
    # split-emissivity model (caseb_hr_attenuated_split) those zones
    # become important: their τ_es,out ≈ 1-6 means a fraction f_wing of
    # their bare emission Thomson-scatters into the wings.
    #
    # When emission_model='caseb_hr_attenuated_split' is used we extend
    # wind_mask to include the envelope/inner-CSM. For all other models
    # (including production caseb_hr_attenuated) the legacy behavior is
    # preserved.
    #
    # The hot post-shock layer (T >> T_phot) contributes mainly continuum so
    # is excluded from line emission. For IIP, the envelope (recombining H).
    # For Ia/stripped/unknown, return zones that usually have no H — so
    # populations will return ~0 line emission anyway.
    regions = state.regions
    if sn_type == 'IIn':
        shell_mask = regions['cdshell']
        wind_csm = regions.get('wind', np.zeros_like(regions['photosphere']))
        if emission_model == 'caseb_hr_attenuated_split':
            envelope_csm = regions.get('envelope', np.zeros_like(regions['photosphere']))
            wind_mask = wind_csm | envelope_csm
        else:
            wind_mask = wind_csm
    elif sn_type == 'IIP':
        shell_mask = np.zeros_like(regions['photosphere'])
        wind_mask = regions['envelope'] | regions['transparent']
    elif sn_type == 'Ia':
        shell_mask = np.zeros_like(regions['photosphere'])
        wind_mask = regions['fast_ejecta_outer']
    elif sn_type == 'stripped_envelope':
        shell_mask = regions['shocked_ejecta']
        wind_mask = regions['envelope'] | regions['transparent']
    else:
        shell_mask = np.zeros_like(regions['photosphere'])
        wind_mask = regions['transparent']

    # backward-compat aliases expected by the rest of the pipeline
    masks = {
        'photosphere': regions['photosphere'],
        'shell': shell_mask,
        'wind': wind_mask,
    }

    # ---- Level populations ----
    if populations_mode == 'caseb':
        n_2, n_3, popdiag = h_populations_auto(
            rho, T, n_e, r, v, X_H=X_H_per_zone, X_He=0.249,
            v_turb_cms=v_turb_kms * 1e5,
        )
    elif populations_mode == 'lte':
        from h_populations_nlte import h_populations_lte
        n_2, n_3, popdiag = h_populations_lte(
            rho, T, n_e, X_H=X_H_per_zone, nlev=max(nlte_levels, 3))
    elif populations_mode == 'nlte':
        from h_populations_nlte import h_populations_nlte
        # Photoionization-equilibrium override: if the upstream pipeline ran
        # solve_photoionization_equilibrium(), the snap carries 'X_HII' and
        # 'n_HI' per zone. Pass these to the NLTE solver so it uses the
        # photoionization X_HII for the ground-state population instead of
        # Saha. Without this, Saha at the photoionization-equilibrium T and
        # n_e gives a DIFFERENT X_HII (typically higher), defeating the
        # purpose of the upstream photoionization step.
        f_HI_override = None
        if snap_arrays.get('photoionized', False) and 'X_HII' in snap_arrays:
            X_HII_pi = np.asarray(snap_arrays['X_HII'], dtype=float)
            f_HI_override = np.clip(1.0 - X_HII_pi, 0.0, 1.0)
        n_2, n_3, popdiag = h_populations_nlte(
            rho, T, n_e, r, v, X_H=X_H_per_zone,
            nlev=nlte_levels, v_turb_cms=v_turb_kms * 1e5,
            ionization_mode=ionization_mode,
            R_phot=state.R_phot, T_phot=state.T_phot,
            L_phot=state.L_phot,
            f_HI_max=f_HI_max,
            stromgren_exponent=stromgren_exponent,
            f_HI_external=f_HI_override,
            J_bar_Ha_abs=J_bar_Ha_abs,
            J_bar_factor=J_bar_factor,
            eps_Lya_destruction=eps_Lya_destruction,
            two_photon_decay=two_photon_decay,
            verbose=verbose,
        )
    else:
        raise ValueError(f"populations_mode must be 'caseb'|'lte'|'nlte', "
                         f"got {populations_mode!r}")

    # proton density for emissivity. If we used photoion ionization in the
    # NLTE solver, prefer popdiag's f_HI so n_p stays consistent with n_HI.
    n_H_total = X_H_per_zone * rho / MH
    if snap_arrays.get('photoionized', False) and 'X_HII' in snap_arrays:
        # Photoionization-equilibrium override: use the upstream X_HII directly
        f_HI = np.clip(1.0 - np.asarray(snap_arrays['X_HII'], dtype=float), 0.0, 1.0)
    elif ionization_mode == 'photoion' and 'f_HI' in popdiag:
        f_HI = popdiag['f_HI']
    else:
        f_HI = _saha_neutral_fraction(T, n_e)
    n_p = (1.0 - f_HI) * n_H_total
    n_HI_for_eps = popdiag.get('n_HI', f_HI * n_H_total)

    # ---- Use PhysicalState's photospheric properties ----
    R_phot_for_emis = state.R_phot
    T_phot_for_emis = state.T_phot        # gas T at τ=1
    L_phot = state.L_phot

    # Band-integrated continuum for the line/continuum ratio.
    # Use T_phot (gas T at τ=1) for the BB shape:
    #   L_cont(λ) = L_phot × B(λ, T_gas) × π / (σ T_gas⁴)
    # This is a DILUTED BB — L_phot is the actual emergent luminosity
    # (from snapshot L(r)), which may be < the naive 4π R² σ T⁴ (scattering
    # atmosphere has filling factor < 1). The spectral SHAPE is still
    # T_gas-BB because the gas at τ=1 sets the emergent color.
    # Verified against CMFGEN: this gives 2.4% of L_bol in 6200-6950 Å,
    # matching CMFGEN's 2.7% integrated flux in the same band.
    L_cont_per_A_emis = continuum_luminosity_per_A(L_phot, T_phot_for_emis, LAM_HA)
    v_th_cms = np.sqrt(KB * T / 1.6735575e-24)
    v_D_cms = np.sqrt(v_th_cms**2 + (v_turb_kms * 1e5)**2)
    lam_D_AA = LAM_HA * 1e8 * v_D_cms / 2.99792458e10
    L_cont_per_zone = L_cont_per_A_emis * lam_D_AA

    # ---- Per-zone emissivity ----
    # Preallocate ε_eff output array as an optional HR-branching diagnostic.
    # compute_emissivity writes per-zone HR ε_eff here regardless of mode.
    # The pipeline doesn't currently consume this, but it is cheap to compute
    # and useful when debugging emission-model behaviour.
    eps_eff_per_zone = np.zeros_like(r)
    emis_shell = compute_emissivity(r, T, n_e, n_p=n_p, n_2=n_2, n_3=n_3, v=v,
                                     mask=shell_mask, n_1=n_HI_for_eps,
                                     L_cont_per_zone=L_cont_per_zone,
                                     T_phot=T_phot_for_emis,
                                     R_phot=R_phot_for_emis,
                                     eps_eff_out=eps_eff_per_zone,
                                     emission_model=emission_model,
                                     f_wing=f_wing,
                                     delta_scatter=delta_scatter)
    emis_wind = compute_emissivity(r, T, n_e, n_p=n_p, n_2=n_2, n_3=n_3, v=v,
                                    mask=wind_mask, n_1=n_HI_for_eps,
                                    L_cont_per_zone=L_cont_per_zone,
                                    T_phot=T_phot_for_emis,
                                    R_phot=R_phot_for_emis,
                                    eps_eff_out=eps_eff_per_zone,
                                    emission_model=emission_model,
                                    f_wing=f_wing,
                                    delta_scatter=delta_scatter)
    L_shell = float(emis_shell.sum())
    L_wind = float(emis_wind.sum())
    L_line = L_shell + L_wind

    trim_mask = tau_es <= tau_es_trim_for_fvol
    L_shell_kept = float((emis_shell * trim_mask).sum())
    L_wind_kept = float((emis_wind * trim_mask).sum())
    L_line_kept = L_shell_kept + L_wind_kept

    lam_lo_AA, lam_hi_AA = wavelength_band_AA
    L_cont_per_A = continuum_luminosity_per_A(L_phot, T_phot_for_emis, LAM_HA)
    band_AA = lam_hi_AA - lam_lo_AA
    L_cont_band = L_cont_per_A * band_AA

    if L_line_kept + L_cont_band > 0:
        f_vol = L_line_kept / (L_line_kept + L_cont_band)
    else:
        f_vol = 0.0

    emissivity = emis_shell + emis_wind

    out = dict(
        sn_type=sn_type,
        masks=masks,
        tau_es=tau_es,
        n_lower=n_2, n_upper=n_3, n_p=n_p,
        X_H_per_zone=X_H_per_zone,
        eps_eff_per_zone=eps_eff_per_zone,
        R_phot=R_phot_for_emis,
        emissivity=emissivity,
        emissivity_shell=emis_shell, emissivity_wind=emis_wind,
        L_shell=L_shell, L_wind=L_wind, L_line=L_line,
        L_shell_kept=L_shell_kept, L_wind_kept=L_wind_kept, L_line_kept=L_line_kept,
        L_phot=L_phot, T_phot=T_phot_for_emis,
        L_cont_per_A=L_cont_per_A, L_cont_band=L_cont_band,
        f_vol=f_vol,
        v_turb_kms=float(v_turb_kms),
        v_turb_kms_grid=v_turb_kms_grid,   # per-zone microturbulence [km/s]
        wavelength_band_AA=wavelength_band_AA,
        populations_diag=popdiag,
    )
    # Photoion-decoupled mode: also expose n_lower_absorb for the kernel
    # to use in line-absorption opacity (peel-to-observer paths and
    # random-walk-step line tau-marching), while keeping n_lower
    # (Saha-based) for emission-physics.
    if 'n_2_absorb' in popdiag:
        out['n_lower_absorb'] = popdiag['n_2_absorb']

    if verbose:
        print(f"  sn_type              : {sn_type} (confidence {state.sn_type_confidence:.0%})")
        print(f"  shell zones          : {int(shell_mask.sum()):3d}  of {len(r)}")
        print(f"  wind zones           : {int(wind_mask.sum()):3d}  of {len(r)}")
        print(f"  L_phot               : {L_phot:.3e} erg/s (T_phot={T_phot_for_emis:.0f} K)")
        print(f"  L_cont in band       : {L_cont_band:.3e} erg/s")
        print(f"  L_line               : {L_line:.3e} erg/s "
              f"(shell={L_shell:.2e}, wind={L_wind:.2e})")
        print(f"  f_vol                : {f_vol:.3f}")
        print(f"  v_turb (scalar fallback) : {v_turb_kms:.1f} km/s ({state.v_turb_basis})")
        print(f"  v_turb_grid          : "
              f"min={v_turb_kms_grid.min():.1f}, "
              f"max={v_turb_kms_grid.max():.1f}, "
              f"median={float(np.median(v_turb_kms_grid)):.1f} km/s")
        print(f"  populations mode     : {populations_mode}")
        print(f"  emission model       : {emission_model}")

    return out



def derive_parameters(snap_arrays: dict, line_name: str = 'Halpha',
                      wavelength_band_AA=(6200.0, 6950.0),
                      X_H: float = 0.737, X_He: float = 0.249,
                      v_turb_kms: float | None = None,
                      populations_mode: str = 'caseb',
                      nlte_levels: int = 3,
                      tau_es_trim_for_fvol: float = 10.0,
                      verbose: bool = True) -> dict:
    """Given a snapshot (HERACLES or MESA arrays in cgs), return everything
    the pipeline needs.

    populations_mode controls how H(n=2), H(n=3) are computed:
      'caseb' — Case-B recombination with local Sobolev Lyalpha escape
                (default; current behavior)
      'lte'   — Saha ionization + Boltzmann distribution
                (valid in dense collisionally-dominated regimes)
      'nlte'  — multi-level statistical equilibrium with escape probabilities,
                self-consistently iterated
    """
    r   = np.asarray(snap_arrays['r'])
    v   = np.asarray(snap_arrays['v'])
    rho = np.asarray(snap_arrays['rho'])
    T   = np.asarray(snap_arrays['T'])
    n_e = np.asarray(snap_arrays['n_e'])

    # tau_es cumulative from outside inward
    dr = np.empty_like(r); dr[:-1] = np.diff(r); dr[-1] = dr[-2]
    tau_es = np.cumsum((SIGMAT * n_e * dr)[::-1])[::-1]

    sn_type = classify_sn(r, v, rho)
    masks   = compute_masks(r, v, rho, tau_es, sn_type)

    # v_turb default from mean T of shell, or user override
    if v_turb_kms is None:
        T_shell = np.mean(T[masks['shell']]) if masks['shell'].any() else np.mean(T)
        v_turb_kms = default_v_turb_kms(T_shell)

    # level populations: dispatch based on mode
    # X_H: if snap_arrays has 'X_H' array, use it; else use the scalar argument.
    X_H_per_zone = snap_arrays.get('X_H', None)
    if X_H_per_zone is None:
        X_H_per_zone = np.full_like(r, X_H)
    X_H_per_zone = np.asarray(X_H_per_zone, dtype=float)

    if populations_mode == 'caseb':
        n_2, n_3, popdiag = h_populations_auto(
            rho, T, n_e, r, v, X_H=X_H_per_zone, X_He=X_He,
            v_turb_cms=v_turb_kms * 1e5,
        )
    elif populations_mode == 'lte':
        from h_populations_nlte import h_populations_lte
        n_2, n_3, popdiag = h_populations_lte(
            rho, T, n_e, X_H=X_H_per_zone, nlev=max(nlte_levels, 3))
    elif populations_mode == 'nlte':
        from h_populations_nlte import h_populations_nlte
        f_HI_override = None
        if snap_arrays.get('photoionized', False) and 'X_HII' in snap_arrays:
            X_HII_pi = np.asarray(snap_arrays['X_HII'], dtype=float)
            f_HI_override = np.clip(1.0 - X_HII_pi, 0.0, 1.0)
        n_2, n_3, popdiag = h_populations_nlte(
            rho, T, n_e, r, v, X_H=X_H_per_zone,
            nlev=nlte_levels, v_turb_cms=v_turb_kms * 1e5,
            f_HI_external=f_HI_override,
            eps_Lya_destruction=eps_Lya_destruction,
            two_photon_decay=two_photon_decay,
            verbose=verbose,
        )
    else:
        raise ValueError(f"populations_mode must be 'caseb'|'lte'|'nlte', "
                         f"got {populations_mode!r}")

    # proton density for emissivity (not n_e, to avoid overcounting He contribution)
    n_H_total = X_H_per_zone * rho / MH
    f_HI = _saha_neutral_fraction(T, n_e)
    n_p = (1.0 - f_HI) * n_H_total
    n_HI_for_eps = popdiag.get('n_HI', f_HI * n_H_total)

    # Continuum luminosity per zone (within the line-Doppler frequency band):
    # The relevant continuum that gets ABSORBED by the line is the photospheric
    # BB flux at line frequencies, propagating outward. For the net-emissivity
    # calculation, what matters is L_cont (per Doppler width) traversing each
    # zone. For an optically thin (in continuum) wind, L_cont through every
    # zone is approximately L_cont(λ_0) × Δν_Doppler.
    L_phot_for_emis = float(snap_arrays.get('L_phot', 0.0))
    T_phot_for_emis = float(snap_arrays.get('T_phot', 0.0))
    if L_phot_for_emis <= 0 or T_phot_for_emis <= 0:
        idx_in_ = int(np.argmax(masks['photosphere'])) if masks['photosphere'].any() else 0
        T_phot_for_emis = float(T[idx_in_])
        L_phot_for_emis = 4.0 * np.pi * r[idx_in_]**2 * SIG_SB * T_phot_for_emis**4

    L_cont_per_A_emis = continuum_luminosity_per_A(L_phot_for_emis, T_phot_for_emis, LAM_HA)
    # Doppler width in Å:
    v_th_cms = np.sqrt(KB * T / 1.6735575e-24)   # thermal velocity hydrogen
    v_D_cms = np.sqrt(v_th_cms**2 + (v_turb_kms * 1e5)**2)
    lam_D_AA = LAM_HA * 1e8 * v_D_cms / 2.99792458e10   # Å
    L_cont_per_zone = L_cont_per_A_emis * lam_D_AA      # erg/s per zone (in resonance band)

    # absolute emissivity per zone (erg/s): blends gross-recombination (thin
    # regime) with diluted-pump source function (thick regime); see
    # compute_emissivity docstring.
    R_phot_for_emis = float(snap_arrays.get('R_phot', 0.0))
    if R_phot_for_emis <= 0:
        # use the τ_es=1 boundary
        above_tau1 = tau_es <= 1.0
        if np.any(above_tau1):
            R_phot_for_emis = float(r[int(np.argmax(above_tau1))])
        else:
            R_phot_for_emis = float(r[0])

    emis_shell = compute_emissivity(r, T, n_e, n_p=n_p, n_2=n_2, n_3=n_3, v=v,
                                    mask=masks['shell'], n_1=n_HI_for_eps,
                                    L_cont_per_zone=L_cont_per_zone,
                                    T_phot=T_phot_for_emis,
                                    R_phot=R_phot_for_emis,
                                    emission_model=emission_model)
    emis_wind  = compute_emissivity(r, T, n_e, n_p=n_p, n_2=n_2, n_3=n_3, v=v,
                                    mask=masks['wind'],  n_1=n_HI_for_eps,
                                    L_cont_per_zone=L_cont_per_zone,
                                    T_phot=T_phot_for_emis,
                                    R_phot=R_phot_for_emis,
                                    emission_model=emission_model)
    L_shell = float(emis_shell.sum())
    L_wind  = float(emis_wind.sum())
    L_line  = L_shell + L_wind

    # For the f_vol calculation, restrict to zones that will actually survive
    # the MC's tau_es_trim cut — otherwise we allocate too many packets to
    # volumetric mode when most shell emission is from deep-trimmed zones.
    # Default tau_es_trim = 10, configurable via kwarg.
    trim_mask = tau_es <= tau_es_trim_for_fvol
    L_shell_kept = float((emis_shell * trim_mask).sum())
    L_wind_kept  = float((emis_wind * trim_mask).sum())
    L_line_kept  = L_shell_kept + L_wind_kept

    # continuum estimate: need L_phot and T_phot from snap or innermost zone
    L_phot = float(snap_arrays.get('L_phot', 0.0))
    T_phot = float(snap_arrays.get('T_phot', 0.0))
    if L_phot <= 0 or T_phot <= 0:
        # fall back to innermost-zone properties
        idx_in = 0 if masks['photosphere'].any() else 0
        if masks['photosphere'].any():
            idx_in = int(np.argmax(masks['photosphere']))
        T_phot = float(T[idx_in])
        # L_phot from BB of that zone at surface 4πr²
        L_phot = 4.0 * np.pi * r[idx_in]**2 * SIG_SB * T_phot**4

    lam_lo_AA, lam_hi_AA = wavelength_band_AA
    lam0_cm = LAM_HA
    L_cont_per_A = continuum_luminosity_per_A(L_phot, T_phot, lam0_cm)
    band_AA = lam_hi_AA - lam_lo_AA
    L_cont_band = L_cont_per_A * band_AA

    # fraction of packets to emit volumetrically.
    # Use the kept (non-trimmed) L_line so packet allocation matches what
    # the MC kernel will actually transport.
    if L_line_kept + L_cont_band > 0:
        f_vol = L_line_kept / (L_line_kept + L_cont_band)
    else:
        f_vol = 0.0

    emissivity = emis_shell + emis_wind     # only emit in shell+wind, not photosphere

    out = dict(
        sn_type=sn_type,
        masks=masks,
        tau_es=tau_es,
        n_lower=n_2, n_upper=n_3,
        emissivity=emissivity,
        emissivity_shell=emis_shell, emissivity_wind=emis_wind,
        L_shell=L_shell, L_wind=L_wind, L_line=L_line,
        L_shell_kept=L_shell_kept, L_wind_kept=L_wind_kept, L_line_kept=L_line_kept,
        L_phot=L_phot, T_phot=T_phot,
        L_cont_per_A=L_cont_per_A, L_cont_band=L_cont_band,
        f_vol=f_vol,
        v_turb_kms=float(v_turb_kms),
        wavelength_band_AA=wavelength_band_AA,
        populations_diag=popdiag,
    )

    if verbose:
        print(f"  sn_type classified  : {sn_type}")
        print(f"  shell zones         : {int(masks['shell'].sum())}  of {len(r)}")
        print(f"  wind zones          : {int(masks['wind'].sum())}  of {len(r)}")
        print(f"  L_phot              : {L_phot:.3e} erg/s (T_eff={T_phot:.0f} K)")
        print(f"  L_cont in band      : {L_cont_band:.3e} erg/s "
              f"({L_cont_per_A:.2e} erg/s/Å)")
        print(f"  L_line (all zones)  : {L_line:.3e} erg/s "
              f"(shell={L_shell:.2e}, wind={L_wind:.2e})")
        print(f"  L_line (τes<{tau_es_trim_for_fvol:.0f}, used for f_vol): "
              f"{L_line_kept:.3e} erg/s (shell={L_shell_kept:.2e}, "
              f"wind={L_wind_kept:.2e})")
        print(f"  f_vol               : {f_vol:.3f}")
        print(f"  v_turb              : {v_turb_kms:.1f} km/s")
        print(f"  populations mode    : {populations_mode}")
        if 'tau_Ly_median' in popdiag:
            bl = popdiag['tau_Ly_median']
            print(f"  median tau_Ly       : {bl:.2e}  "
                  f"(beta_Ly ~ {1/bl if np.isfinite(bl) and bl > 0 else 0:.2e})")
        if populations_mode == 'nlte' and 'iterations' in popdiag:
            print(f"  NLTE iterations     : {popdiag['iterations']}")
        # diagnostic: Hα self-absorption suppression factor per region
        # (what fraction of naive Case-B emission escapes)
        dv_dr = np.abs(np.gradient(v, r))
        dv_dr = np.maximum(dv_dr, 20e5/(r[-1]-r[0]))
        sigma_Ha = np.pi * EE * EE / (ME * C) * F_HA
        tau_Ha = sigma_Ha * LAM_HA * np.maximum(n_2, 0) / dv_dr
        if masks['shell'].any():
            tau_Ha_shell = np.median(tau_Ha[masks['shell']])
            print(f"  median tau_Hα(shell): {tau_Ha_shell:.2e}  "
                  f"(beta_Hα ~ {(1-np.exp(-min(tau_Ha_shell,700)))/max(tau_Ha_shell,1e-30):.2e})")
        if masks['wind'].any():
            tau_Ha_wind = np.median(tau_Ha[masks['wind']])
            print(f"  median tau_Hα(wind) : {tau_Ha_wind:.2e}  "
                  f"(beta_Hα ~ {(1-np.exp(-min(tau_Ha_wind,700)))/max(tau_Ha_wind,1e-30):.2e})")

    return out
