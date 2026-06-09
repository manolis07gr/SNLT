"""
formal_line_profile.py — Sobolev formal-solution line profiles from a
two-level-atom-with-continuum source function S_L = (1-eps)*J_bar + eps*B.

Motivation
----------
The Monte-Carlo peel pipeline produces the correct *scattering* profile for the
continuum channel, but the volumetric (recombination) channel injects line
photons at the recombination emissivity (a thin shell at the photosphere) and
pure-scatters them. For an optically thick line that thin-shell origin, combined
with occultation of the far hemisphere, yields a spuriously blueshifted emission
peak (~ -2000 km/s for IIP day 50) and an over-strong amplitude (~10x continuum).

The physically correct emergent profile of a scattering-dominated line is the
formal solution with the line source function

    S_L(r) = (1 - eps(r)) * J_bar(r) + eps(r) * B_nu(T(r))           [erg/s/cm2/sr/Hz]

where J_bar is the angle-averaged line-mean-intensity (measured by the RT loop)
and eps is the per-scatter thermal destruction probability. Because J_bar is set
by the diluted radiation field and is extended through the envelope, the emergent
emission peaks AT REST, exactly as observed (SN 1993J) and as produced by
SuperLite for the same input model.

This module computes that emergent profile via the standard Sobolev P-Cygni
formal solution in a homologously expanding atmosphere (v = r/t_exp), including
photospheric occultation of the far hemisphere and (optionally) a single-electron
Thomson-scattering broadening of the emergent line.

The treatment is first-principles: no floors, no caps, no normalization to a
reference. The line STRENGTH follows from S_L and tau_S, and the line SHAPE from
the velocity field and the geometry.
"""

from __future__ import annotations
import numpy as np

# Physical constants (cgs)
C = 2.99792458e10          # cm/s
H = 6.62607015e-27         # erg s
KB = 1.380649e-16          # erg/K
ME = 9.1093837e-28         # g
EE = 4.80320425e-10        # esu
SIGMAT = 6.65245871e-25    # cm^2 (Thomson)
SIGMA_INT = np.pi * EE * EE / (ME * C)   # pi e^2 / m_e c  [cm^2 Hz]


def planck_nu(nu, T):
    """Planck function B_nu(T) [erg/s/cm2/sr/Hz]. Robust to overflow."""
    x = np.minimum(H * nu / (KB * np.maximum(T, 1.0)), 700.0)
    return (2.0 * H * nu**3 / C**2) / np.expm1(x)


def collisional_destruction_eps(n_e, T, A_ul, g_l, g_u, nu0,
                                 collision_strength=5.0):
    """Per-scatter thermal destruction probability from electron collisions.

    eps = C_ul / (A_ul + C_ul),  C_ul = n_e * q_ul,
    q_ul = 8.629e-6 * Upsilon / (g_u * sqrt(T))   [cm^3/s]   (de-excitation)

    This captures the *collisional* destruction only. Radiative branching of the
    upper level into other lines (e.g. n=3 -> Ly-beta) is a separate destruction
    channel that, in an optically thick Lyman environment, is largely recycled
    back into the line; that recycling is handled by the multi-level NLTE solve
    that supplies J_bar, so it is intentionally NOT double-counted here. For Halpha
    the collisional eps is small (~1e-3..1e-2), so the line is scattering-dominated.
    """
    q_ul = 8.629e-6 * collision_strength / (g_u * np.sqrt(np.maximum(T, 1.0)))
    C_ul = n_e * q_ul
    return C_ul / (A_ul + C_ul)


def sobolev_tau(r, v, n_lower, n_upper, f_lu, lam0_cm, g_l, g_u, t_exp):
    """Sobolev optical depth tau_S(r) for homologous expansion (dv/dr = 1/t_exp)."""
    pop = np.maximum(n_lower - (g_l / g_u) * n_upper, 0.0)
    return SIGMA_INT * f_lu * lam0_cm * pop * t_exp


def line_source_function(J_bar, T, nu0, eps):
    """S_L = (1-eps)*J_bar + eps*B_nu(T)  [erg/s/cm2/sr/Hz]."""
    B = planck_nu(nu0, T)
    return (1.0 - eps) * J_bar + eps * B


def formal_pcygni(r, v, tau_S, S_L, R_phot, T_phot, nu0, t_exp,
                  vgrid_kms=None, n_p=500, tau_es_env=0.0,
                  occultation=True):
    """Sobolev P-Cygni formal solution. Returns (vgrid_kms, F_over_Fc).

    For each observed velocity v_obs the line resonates on the plane
    z = -v_obs * t_exp (observer at +z; v_obs>0 = redshift). At impact parameter
    p the resonance point is r_res = sqrt(p^2 + z^2). The emergent flux (in units
    of the photospheric continuum disk) is

        F(v_obs)/F_c = [disk continuum, attenuated by line where in front]
                     + [line emission S_L*(1-exp(-tau_S)) over the resonance
                        surface, excluding the part occulted by the photosphere]

    A flat continuum disk of intensity I_c = B_nu(T_phot) and radius R_phot is
    assumed. Optional uniform envelope Thomson depth tau_es_env attenuates the
    far-side emission (foreground electron scattering) without redistribution
    (a mild correction; full electron redistribution is applied separately).
    """
    Rout = float(r[-1])
    Ic = planck_nu(nu0, T_phot)
    SLi = S_L / Ic                                   # source function in disk units
    if vgrid_kms is None:
        vmax = 1.05 * v[-1] / 1e5
        vgrid_kms = np.linspace(-vmax, vmax, 401)
    vobs = vgrid_kms * 1e5                            # cm/s

    def interp(rr, arr, right=0.0):
        return np.interp(rr, r, arr, left=arr[0], right=right)

    p = np.linspace(0.0, Rout, n_p)
    dp = p[1] - p[0]
    F = np.zeros_like(vobs)
    # continuum disk normalization
    Fc = np.sum(np.where(p < R_phot, 1.0, 0.0) * 2.0 * np.pi * p) * dp
    for i, vo in enumerate(vobs):
        z = -vo * t_exp
        rr = np.sqrt(p * p + z * z)
        inside = (rr >= R_phot) & (rr <= Rout)
        tt = np.where(inside, interp(rr, tau_S), 0.0)
        S = np.where(inside, interp(rr, SLi), 0.0)
        one_minus = 1.0 - np.exp(-tt)
        # disk continuum: present for p<R_phot; absorbed when resonance is in
        # front of the disk (z>0) within the line-forming region.
        disk = np.where(p < R_phot, 1.0, 0.0)
        absorbed = np.where((p < R_phot) & (z > 0) & inside, one_minus, 0.0)
        cont = disk * (1.0 - absorbed)
        # line emission; occult far hemisphere behind the photosphere
        if occultation:
            occ = (p < R_phot) & (z < 0)
        else:
            occ = np.zeros_like(p, dtype=bool)
        emit = np.where(inside & (~occ), S * one_minus, 0.0)
        # foreground envelope Thomson attenuation of the emission (mild)
        if tau_es_env > 0.0:
            emit = emit * np.exp(-tau_es_env)
        F[i] = np.sum((cont + emit) * 2.0 * np.pi * p) * dp
    return vgrid_kms, F / max(Fc, 1e-300)


def electron_scatter_broaden(vgrid_kms, Fnorm, T_e_phot, tau_es,
                              n_kernel=151):
    """Apply a single-scattering Thomson broadening to the line component.

    A fraction (1 - exp(-tau_es)) of photons Thomson-scatter once, acquiring a
    Doppler kick from the thermal electron motion with 1D dispersion
    sigma_e = sqrt(k T_e / m_e) (in velocity). This convolves the *line excess*
    (Fnorm - 1) with a Gaussian of that width and re-adds the continuum. This is
    a first-order correction (full RT does multiple scatters); it broadens the
    wings slightly and does not move the peak.
    """
    if tau_es <= 0.0:
        return Fnorm
    sigma_e_kms = np.sqrt(KB * T_e_phot / ME) / 1e5
    dv = vgrid_kms[1] - vgrid_kms[0]
    half = (n_kernel // 2)
    kv = (np.arange(n_kernel) - half) * dv
    g = np.exp(-0.5 * (kv / sigma_e_kms) ** 2)
    g /= g.sum()
    excess = Fnorm - 1.0
    scattered = np.convolve(excess, g, mode='same')
    fsc = 1.0 - np.exp(-tau_es)
    return 1.0 + (1.0 - fsc) * excess + fsc * scattered


def sobolev_validity(r, v, max_homology_spread=0.5, max_reversal_frac=0.10):
    """Assess whether a Sobolev source-function formal solution is applicable.

    The Sobolev/homologous P-Cygni formal solution assumes a monotonically
    increasing, near-homologous velocity field (v approx proportional to r, so
    a single expansion time t_exp = r/v describes the resonance geometry). This
    holds for SN ejecta (Type IIP-like): std(r/v)/median(r/v) ~ 0.

    It FAILS for a dense, slow, interacting circumstellar medium (Type IIn-like),
    where the velocity field is non-monotonic (shock reversals) and strongly
    non-homologous. In that regime the line is recombination emission reprocessed
    by multiple electron scattering, not a resonance-scattering P-Cygni, and must
    be transported by the Monte-Carlo kernel (local Doppler, arbitrary v-field).

    Returns dict(valid, homology_spread, reversal_frac, reason).
    """
    r = np.asarray(r, float); v = np.asarray(v, float)
    t = r / np.maximum(v, 1e-30)
    homology_spread = float(np.std(t) / max(np.median(t), 1e-30))
    with np.errstate(divide='ignore', invalid='ignore'):
        dvdr = np.gradient(v, r)
    # a non-finite gradient marks a DUPLICATE-radius zone = a STELLA shock front,
    # which IS a velocity reversal — count it as one (not as 'not reversed').
    reversal_frac = float(np.mean((dvdr < 0) | ~np.isfinite(dvdr)))
    valid = (homology_spread <= max_homology_spread) and \
            (reversal_frac <= max_reversal_frac)
    reason = ('homologous (Sobolev valid)' if valid else
              f'non-homologous velocity field '
              f'(std(r/v)/median={homology_spread:.2f}, '
              f'reversals={reversal_frac:.0%}) — dense-CSM/IIn regime')
    return {'valid': valid, 'homology_spread': homology_spread,
            'reversal_frac': reversal_frac, 'reason': reason}


def sobolev_scatter_jbar(r, R_phot, T_phot, nu0, tau_S, eps):
    """Self-consistent line mean intensity for a scattering-dominated line.

    Solving the Sobolev two-level source function S_L = (1-eps) J_bar + eps B
    together with the local mean intensity J_bar = (1-beta) S_L + beta_c W I_c
    (W = geometric dilution, beta = Sobolev escape, beta_c ~ beta the continuum
    coupling) gives, for the radiation field that drives the scattering:

        J_bar = W(r) * I_c                      (eps -> 0, beta_c ~ beta limit)

    i.e. the diluted photospheric continuum that the n=2 -> 3 transition
    resonantly scatters. This is the first-principles scattering source; it is
    NOT the (self-shielded) line-center J_bar one would read off at line center,
    which underestimates the resonant pumping. The recombination contribution is
    recycled through the optically thick Ly-beta / 2-photon channel rather than
    emerging as a net thin-shell source, so for H-alpha it does not dominate.
    """
    Ic = planck_nu(nu0, T_phot)
    x = np.clip((R_phot / np.maximum(r, R_phot)) ** 2, 0.0, 1.0)
    W = 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - x, 0.0)))
    return W * Ic


def halpha_profile_from_state(r, v, n_lower, n_upper, T, n_e, J_bar=None,
                              R_phot=None, T_phot=None, line=None, t_exp=None,
                              eps=None, collision_strength=5.0,
                              tau_es_env=0.0, electron_scatter=True,
                              vgrid_kms=None, jbar_source='scatter'):
    """End-to-end emergent line profile from a state + measured J_bar.

    Parameters
    ----------
    r, v : radii [cm], radial velocities [cm/s]
    n_lower, n_upper : lower/upper level number densities [cm^-3]
    T, n_e : gas temperature [K], electron density [cm^-3]
    J_bar : measured line mean intensity [erg/s/cm2/sr/Hz] (from the RT loop)
    R_phot, T_phot : photospheric radius [cm] and temperature [K]
    line : object with .wavelength_cm (or .lam0_cm), .f (oscillator strength),
           .g_lower, .g_upper, .A_ul (optional)
    t_exp : expansion time [s]; default median(r/v) (homologous)
    eps : per-scatter destruction probability (scalar or array). If None,
          computed from electron collisions.
    Returns: dict with 'velocity_kms', 'F_norm', 'S_L', 'tau_S', 'eps', 'L_line'.
    """
    r = np.asarray(r, float); v = np.asarray(v, float)
    n_lower = np.asarray(n_lower, float); n_upper = np.asarray(n_upper, float)
    T = np.asarray(T, float); n_e = np.asarray(n_e, float)
    lam0 = getattr(line, 'lam0_cm', getattr(line, 'wavelength_cm', None))
    f_lu = getattr(line, 'f_lu', getattr(line, 'f', None))
    g_l = getattr(line, 'g_l', getattr(line, 'g_lower', 8.0))
    g_u = getattr(line, 'g_u', getattr(line, 'g_upper', 18.0))
    nu0 = C / lam0
    A_ul = getattr(line, 'A_ul', (8.0 * np.pi**2 * EE**2 * nu0**2 /
                                  (ME * C**3)) * (g_l / g_u) * f_lu)
    if t_exp is None:
        t_exp = float(np.median(r / np.maximum(v, 1e-30)))
    if eps is None:
        eps = collisional_destruction_eps(n_e, T, A_ul, g_l, g_u, nu0,
                                           collision_strength)
    eps = np.broadcast_to(np.asarray(eps, float), r.shape)
    tau_S = sobolev_tau(r, v, n_lower, n_upper, f_lu, lam0, g_l, g_u, t_exp)
    if jbar_source == 'scatter' or J_bar is None:
        # First-principles Sobolev resonance-scattering source (diluted continuum)
        J_bar = sobolev_scatter_jbar(r, R_phot, T_phot, nu0, tau_S, eps)
    else:
        J_bar = np.asarray(J_bar, float)
    S_L = line_source_function(J_bar, T, nu0, eps)
    vg, Fn = formal_pcygni(r, v, tau_S, S_L, R_phot, T_phot, nu0, t_exp,
                           vgrid_kms=vgrid_kms, tau_es_env=tau_es_env)
    if electron_scatter and tau_es_env > 0.0:
        Fn = electron_scatter_broaden(vg, Fn, T_phot, tau_es_env)
    # escaping line luminosity (Sobolev): integral of 4pi * eta_L * beta * dV,
    # eta_L*beta_escape = (hv/4pi) n_u A_ul beta ; beta=(1-e^-tau)/tau
    beta = np.where(tau_S > 1e-6, (1.0 - np.exp(-np.minimum(tau_S, 700))) /
                    np.maximum(tau_S, 1e-30), 1.0 - 0.5 * tau_S)
    r_edge = np.empty(len(r) + 1)
    r_edge[1:-1] = 0.5 * (r[:-1] + r[1:]); r_edge[0] = r[0]
    r_edge[-1] = r[-1] + (r[-1] - r[-2])
    dV = (4.0 / 3.0) * np.pi * (r_edge[1:]**3 - r_edge[:-1]**3)
    L_line = float(np.sum(H * nu0 * n_upper * A_ul * beta * dV))
    return {'velocity_kms': vg, 'F_norm': Fn, 'S_L': S_L, 'tau_S': tau_S,
            'eps': eps, 'L_line': L_line, 't_exp': t_exp}


# ---------------------------------------------------------------------------
# General multi-line entry points (Hβ, Hγ, He I, He II, ...)
# ---------------------------------------------------------------------------

class _LineAdapter:
    """Lightweight line object accepted by halpha_profile_from_state.

    Built from (lam0_AA, A_ul, g_l, g_u); derives the absorption oscillator
    strength f_lu from the Einstein A via
        f_lu = 1.49919 * (g_u/g_l) * A_ul * lam0_cm**2 .
    """
    __slots__ = ('lam0_cm', 'nu0', 'f_lu', 'g_l', 'g_u', 'A_ul', 'name')

    def __init__(self, lam0_AA, A_ul, g_l, g_u, name=''):
        self.lam0_cm = float(lam0_AA) * 1e-8
        self.nu0 = C / self.lam0_cm
        self.g_l = float(g_l); self.g_u = float(g_u)
        self.A_ul = float(A_ul)
        self.f_lu = 1.4991938 * (self.g_u / self.g_l) * self.A_ul * self.lam0_cm**2
        self.name = name


def line_profile_from_state(*args, **kwargs):
    """General-line alias for halpha_profile_from_state (same signature).

    The underlying solver is line-agnostic: it takes the line's own lower/upper
    populations and atomic data and builds S_L = (1-eps) J_bar + eps B with the
    diluted continuum at *that* line's frequency, so each line's profile and
    luminosity emerge independently — no Hα anchoring.
    """
    return halpha_profile_from_state(*args, **kwargs)


def profile_for_line(line_name, lam0_AA, A_ul, g_l, g_u, r, v, T, n_e,
                     n_lower, n_upper, R_phot, T_phot, vgrid_kms,
                     tau_es_env=0.0, electron_scatter=True,
                     collision_strength=5.0):
    """Convenience wrapper: build the line adapter and return the formal profile.

    Returns the dict from halpha_profile_from_state (velocity_kms, F_norm, S_L,
    tau_S, eps, L_line, t_exp).
    """
    line = _LineAdapter(lam0_AA, A_ul, g_l, g_u, name=line_name)
    return halpha_profile_from_state(
        r, v, n_lower, n_upper, T, n_e, J_bar=None,
        R_phot=R_phot, T_phot=T_phot, line=line, vgrid_kms=vgrid_kms,
        tau_es_env=tau_es_env, electron_scatter=electron_scatter,
        collision_strength=collision_strength, jbar_source='scatter')


# ---------------------------------------------------------------------------
# Per-line recombination luminosity budget (replaces the empirical Hα anchor)
# ---------------------------------------------------------------------------
# Effective case-B recombination coefficients: photons produced per recombination,
# alpha_eff(T) = A * (T/1e4)^b  [cm^3/s], and the recombining ion that feeds the
# line. H values reproduce the standard case-B Balmer decrement at 1e4 K
# (Hα:Hβ:Hγ:Hδ = 2.86:1.00:0.468:0.259; Osterbrock & Ferland 2006).
# He values are PROVISIONAL (recombination only; He I 5876 also has a significant
# COLLISIONAL contribution that is NOT included here) and must be confirmed with
# the He-NLTE state. Each entry: (A_1e4, b, ion_key).
RECOMB_COEFF = {
    'Halpha':     (1.17e-13, -0.942, 'n_p'),
    'Hbeta':      (3.03e-14, -0.874, 'n_p'),
    'Hgamma':     (1.27e-14, -0.886, 'n_p'),
    'Hdelta':     (6.62e-15, -0.890, 'n_p'),
    'Palpha':     (1.36e-14, -0.940, 'n_p'),
    'Pbeta':      (7.00e-15, -0.930, 'n_p'),
    'He_I_5876':  (7.50e-15, -1.000, 'n_HeII'),    # PROVISIONAL, recomb-only
    'He_I_6678':  (2.50e-15, -1.000, 'n_HeII'),    # PROVISIONAL, recomb-only
    'He_I_7065':  (2.80e-15, -1.000, 'n_HeII'),    # PROVISIONAL, recomb-only
    'He_I_10830': (5.0e-14,  -1.000, 'n_HeII'),    # PROVISIONAL, recomb-only
    'He_II_4686': (1.60e-13, -0.900, 'n_HeIII'),   # PROVISIONAL
    'He_II_1640': (6.0e-13,  -0.900, 'n_HeIII'),   # PROVISIONAL
}

# which lines carry an extra collisional channel not captured by recomb-only
COLLISIONAL_CAVEAT = {'He_I_5876', 'He_I_6678', 'He_I_7065', 'He_I_10830'}


def _shell_volumes(r):
    r = np.asarray(r, float)
    edge = np.empty(len(r) + 1)
    edge[1:-1] = 0.5 * (r[:-1] + r[1:]); edge[0] = r[0]
    edge[-1] = r[-1] + (r[-1] - r[-2])
    return (4.0 / 3.0) * np.pi * (edge[1:]**3 - edge[:-1]**3)


def line_recombination_luminosity(line_name, lam0_AA, r, T, n_e, n_ion):
    """First-principles line luminosity from the recombination budget:

        L = integral over volume of  h*nu * alpha_eff(T) * n_e * n_ion  dV

    No anchoring to any other line. Returns (L_erg_s, provisional_bool) or
    (None, _) if the line has no tabulated coefficient.
    """
    if line_name not in RECOMB_COEFF:
        return None, False
    A, b, ion_key = RECOMB_COEFF[line_name]
    T = np.maximum(np.asarray(T, float), 1e3)
    alpha = A * (T / 1e4) ** b
    nu = C / (float(lam0_AA) * 1e-8)
    dV = _shell_volumes(r)
    L = float(np.sum(H * nu * alpha * np.asarray(n_e, float) *
                     np.asarray(n_ion, float) * dV))
    provisional = line_name.startswith('He')
    return L, provisional
