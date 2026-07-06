"""
unified_line_rt.py — Unified nonlocal line radiative-transfer solver (opt-in).

FUTURE_WORK P1 #3 (the deferred nonlocal iterated-J̄ / ALI route). This module is
the foundation of a SWITCH-FREE line-profile treatment that replaces the
regime-gated patchwork (homologous→Sobolev formal / non-homologous→MC emission /
saturated→single-shot β) with ONE nonlocal solver valid across the whole optical-
depth (τ_es ≈ 0 … 1e6) and velocity-field span.

It is DEFAULT-OFF and fully isolated: nothing in the production pipeline imports
or calls it unless the caller explicitly opts in (profile_method='unified'). The
current behaviour is byte-unchanged until that wiring lands (a later phase).

────────────────────────────────────────────────────────────────────────────
PHYSICS (built up in phases; this file currently implements Phase 0 = the ALI
engine + its analytic limits):

  Two-level-atom line source function with a scattering integral:
        S = (1-ε) J̄ + ε B
  where ε is the photon destruction probability (collisional de-excitation /
  (collisional + radiative)), B the thermal (Planck) source, and J̄ the
  profile-weighted mean intensity that couples ALL zones (the nonlocality the
  local Sobolev/EP kernels lack).

  Accelerated Λ-iteration (ALI / Olson-Auer-Buchler):
        J̄ = Λ[S],   Λ = Λ* + (Λ − Λ*)
        S_new = [1 − (1-ε)Λ*]^{-1} { (1-ε)(Λ−Λ*)[S_old] + εB }
  with Λ* the diagonal (local) approximate operator from the formal solution.
  Plain Λ-iteration stalls at high τ (convergence ~ per decade of τ); ALI with
  the diagonal Λ* converges in O(10) iterations even at τ~1e6 — which is exactly
  the PPISN / dense-CSM regime the single-shot kernel cannot reach.

  The formal solution J(τ|S) uses the Feautrier method (second-order, stable) on
  the monotone optical-depth grid, with the diagonal of the Λ operator extracted
  analytically for the ALI acceleration.

  Later phases add: electron-scattering frequency redistribution (the broad
  symmetric IIn/PPISN wings), the velocity-field comoving-frame opacity, and the
  observer-frame emergent profile. This file is numpy-only so it unit-tests
  standalone (validate_unified_rt.py).
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import os
import numpy as np

# ---------------------------------------------------------------------------
# Feautrier formal solution: mean intensity J on an optical-depth grid given a
# source function S, for a 1D slab/ray. Returns (J, lambda_star_diagonal).
#
# Feautrier variable u = (I⁺ + I⁻)/2 satisfies, on the τ grid,
#     d²u/dτ² = u − S
# With the two-stream (μ = 1/√3) closure the angle-averaged J = u. Boundary
# conditions: no incoming radiation at the surface (τ=0) unless `I_surface` is
# given (a photospheric disk illuminating the line-forming layers from below at
# the DEEP boundary). We solve the tridiagonal system; the diagonal of the
# inverse (the response of u_i to a unit local S_i) is the ALI Λ* operator.
# ---------------------------------------------------------------------------
def feautrier_J(tau, S, I_top=0.0, I_bottom=0.0, semi_infinite=False):
    """Angle-averaged mean intensity J(τ) for source S(τ) via Feautrier.

    tau : monotone-increasing optical depth (τ[0]=surface, τ[-1]=deep).
    S   : source function on the same grid.
    I_top      : incident intensity at the surface (τ=0); 0 = vacuum (observer
                 side, no incoming radiation) — the usual outer BC.
    I_bottom   : incident intensity at the deep boundary (τ[-1]); 0 = vacuum,
                 or set to the photospheric continuum B(T_phot) when the line-
                 forming layer is illuminated from below by the photosphere.
    semi_infinite : if True, close the deep boundary with the diffusion limit
                 u=S (an effectively infinite thermal reservoir). Default False
                 = the physical finite shell with an open/illuminated deep BC.

    Returns (J, lstar) where lstar[i] = ∂J_i/∂S_i is the diagonal Λ* operator.

    Standard Feautrier (Mihalas) with first-order-accurate, unconditionally
    stable boundary conditions: at each open boundary the half-flux balances the
    incident intensity, so a finite slab correctly LEAKS (optically-thin → escape,
    S→εB) while a deep grid still THERMALIZES the interior (S→B, surface→√ε·B).
    """
    tau = np.asarray(tau, float)
    S = np.asarray(S, float)
    n = tau.size
    if n < 3:
        # too few points to run Feautrier; J ≈ S (optically thin / trivial)
        return S.copy(), np.ones_like(S)

    dtau = np.diff(tau)
    dtau = np.maximum(dtau, 1e-30)
    dtm = dtau[:-1]            # τ_i − τ_{i-1}
    dtp = dtau[1:]             # τ_{i+1} − τ_i
    dtavg = 0.5 * (dtm + dtp)

    # interior: d²u/dτ² = u − S  →  -a u_{i-1} + b u_i - c u_{i+1} = S_i
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n); rhs = np.zeros(n)
    a[1:-1] = 1.0 / (dtm * dtavg)
    c[1:-1] = 1.0 / (dtp * dtavg)
    b[1:-1] = 1.0 + a[1:-1] + c[1:-1]
    rhs[1:-1] = S[1:-1]

    # Surface boundary (τ=0), 2nd-order accurate, incident I_top (Mihalas):
    #   c_0 = 2/Δ², b_0 = 1 + 2/Δ + 2/Δ², rhs_0 = S_0 + (2/Δ)·I_top
    d0 = dtau[0]
    c[0] = 2.0 / d0 / d0
    b[0] = 1.0 + 2.0 / d0 + 2.0 / d0 / d0
    rhs[0] = S[0] + 2.0 * I_top / d0

    # Deep boundary (τ[-1]):
    if semi_infinite:
        b[-1] = 1.0; a[-1] = 0.0; rhs[-1] = S[-1]          # diffusion closure u=S
    else:
        # 2nd-order open boundary, symmetric to the surface, incident I_bottom
        # (=0 for a detached shell in vacuum; =continuum B for photospheric
        # illumination from below). A finite shell then correctly LEAKS.
        dN = dtau[-1]
        a[-1] = 2.0 / dN / dN
        b[-1] = 1.0 + 2.0 / dN + 2.0 / dN / dN
        rhs[-1] = S[-1] + 2.0 * I_bottom / dN

    # Thomas algorithm (tridiagonal solve), and track the diagonal of the inverse
    # for Λ* via the standard forward/back-substitution response.
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = rhs[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        if abs(m) < 1e-30:                          # sign-preserving pivot floor
            m = 1e-30 if m >= 0 else -1e-30
        cp[i] = c[i] / m
        dp[i] = (rhs[i] + a[i] * dp[i - 1]) / m
    u = np.zeros(n)
    u[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        u[i] = dp[i] + cp[i] * u[i + 1]

    # Λ* diagonal: response of u_i to a unit increase in S_i, holding neighbours.
    # For the tridiagonal (b on diagonal), the local response is ≈ 1/b_i for the
    # interior (the Olson-Kunasz diagonal approximation). This is the standard
    # cheap Λ* that makes ALI converge.
    lstar = 1.0 / np.maximum(b, 1e-300)
    J = u
    return J, lstar


# ---------------------------------------------------------------------------
# Two-level-atom ALI: converge S = (1-ε) J̄ + ε B with the diagonal Λ*.
# ---------------------------------------------------------------------------
def solve_two_level_ali(tau, B, eps, I_top=0.0, I_bottom=0.0,
                        semi_infinite=False, max_iter=4000, tol=1e-6, ng=True):
    """Converge the two-level-atom source function via ALI.

    tau, B, eps : arrays on the radial/optical-depth grid (eps = destruction prob).
    I_top/I_bottom/semi_infinite : boundary conditions (see feautrier_J).
    Returns dict(S, J, n_iter, converged, dS_last).
    """
    tau = np.asarray(tau, float)
    B = np.asarray(B, float)
    eps = np.clip(np.asarray(eps, float), 1e-12, 1.0)

    # ---- grid sanitation (REQUIRED for real STELLA snapshots) ----
    # STELLA piles many zones at near-identical radii (dense shock shells) and
    # zones with χ_line→0 contribute dτ=0, so the cumulative τ grid contains
    # degenerate steps (dτ ≤ 0 or ≪ machine-meaningful). Feautrier coefficients
    # scale as 1/dτ², so a dτ≈1e-30 floor produces ~1e59 entries whose roundoff
    # destroys the tridiagonal elimination → NaN (this — not the physics — is
    # why the solver "failed to converge" on every real snapshot while passing
    # all analytic tests on smooth grids). Solve on a reduced strictly-
    # increasing grid (merge points closer than dτ_min: optically coincident
    # layers, Δτ≲1e-6, are physically identical) and interpolate S back.
    tau_full = tau.copy()
    n_full = tau_full.size
    dtau_min = 1e-6
    keep = np.ones(n_full, bool)
    _last = tau_full[0]
    for _i in range(1, n_full):
        if tau_full[_i] - _last < dtau_min * (1.0 + _last):
            keep[_i] = False
        else:
            _last = tau_full[_i]
    if keep.sum() < 3:
        # whole shell optically coincident / thin: J≈S trivial limit
        S = np.clip(eps * B, 0.0, None)
        return {'S': S, 'J': S.copy(), 'n_iter': 0, 'converged': True,
                'dS_last': 0.0}
    tau = tau_full[keep]
    B_r = B[keep]
    eps_r = eps[keep]

    S = B_r.copy()                                 # LTE start
    hist = []
    converged = False
    dS = np.inf
    for it in range(max_iter):
        J, lstar = feautrier_J(tau, S, I_top=I_top, I_bottom=I_bottom,
                               semi_infinite=semi_infinite)
        # ALI update with the diagonal operator:
        #   S_new = [1 - (1-ε)Λ*]^{-1} [ (1-ε)(J - Λ*·S) + εB ]
        one_eps = 1.0 - eps_r
        num = one_eps * (J - lstar * S) + eps_r * B_r
        den = 1.0 - one_eps * lstar
        S_new = num / np.maximum(den, 1e-300)
        S_new = np.maximum(S_new, 0.0)
        if not np.all(np.isfinite(S_new)):
            # numerical breakdown — do not iterate on garbage; report honestly
            converged = False
            break
        dS = float(np.max(np.abs(S_new - S) / np.maximum(np.abs(S_new), 1e-300)))
        hist.append(S_new.copy())
        S = S_new
        # Ng acceleration every 4 iterations (uses last 3 iterates)
        if ng and len(hist) >= 4 and (it % 4 == 3):
            S_ng = _ng_accelerate(hist[-3], hist[-2], hist[-1])
            if np.all(np.isfinite(S_ng)):
                S = np.maximum(S_ng, 0.0)
                hist[-1] = S.copy()
        if len(hist) > 5:
            hist.pop(0)
        if dS < tol:
            converged = True
            break
    Jf, _ = feautrier_J(tau, S, I_top=I_top, I_bottom=I_bottom,
                        semi_infinite=semi_infinite)
    # map the reduced-grid solution back to the caller's full grid (merged
    # points are optically coincident → same S by construction)
    if keep.sum() != n_full:
        S_full = np.interp(tau_full, tau, S)
        J_full = np.interp(tau_full, tau, Jf)
    else:
        S_full, J_full = S, Jf
    return {'S': S_full, 'J': J_full, 'n_iter': it + 1, 'converged': converged,
            'dS_last': dS}


def _ng_accelerate(s2, s1, s0):
    """Ng (1974) 3-point acceleration of the source-function iteration."""
    with np.errstate(over='ignore', invalid='ignore'):
        d0 = s0 - s1
        d1 = s1 - s2
        q0 = d0
        q1 = d0 - d1
        A = float(np.sum(q1 * q1))
        if not np.isfinite(A) or A <= 1e-300:
            return s0
        b = float(np.sum(q1 * q0)) / A
        b = np.clip(b, -2.0, 2.0)
        return s0 + b * (s1 - s0)


# ===========================================================================
# PHASE 1 — gate-free emergent profile + electron-scattering redistribution
# ===========================================================================
_C_KMS = 2.99792458e5
_KB = 1.380649e-16
_ME = 9.1093837e-28


def emergent_profile(r, v, S_line, chi_line, R_phot, I_cont, vgrid_kms,
                     vth_kms=20.0, occultation=True, n_p=160):
    """General observer-frame ray-traced emergent line profile — valid for ANY
    velocity field v(r) (no Sobolev/homology assumption; this is what makes the
    treatment switch-free).

    r, v        : radius [cm], radial velocity magnitude [cm/s] (expansion, ≥0).
    S_line      : line source function per zone (same units as I_cont) — the
                  NONLOCAL ALI source (not a local Sobolev value).
    chi_line    : line opacity per zone [cm^-1] (line-centre; a local Doppler
                  profile φ of width vth_kms is applied along the ray).
    R_phot      : photospheric disk radius [cm] (continuum I_cont for p<R_phot).
    I_cont      : continuum specific intensity of the photospheric disk.
    vgrid_kms   : observed-velocity grid [km/s] (v_obs>0 = redshift).
    Returns F(v_obs) in units of the continuum-disk flux (F/F_cont), on vgrid_kms.

    Sign convention: observer at +z; a front-side (z>0) expanding shell moves
    toward the observer → blueshift (v_obs<0). Line resonance at v_obs where the
    line-of-sight velocity −v·z/r ≈ v_obs (spread by the thermal width).
    """
    r = np.asarray(r, float); v = np.asarray(v, float)
    S_line = np.asarray(S_line, float); chi_line = np.asarray(chi_line, float)
    Rout = float(r[-1])
    vgrid = np.asarray(vgrid_kms, float)
    nvo = vgrid.size
    F = np.zeros(nvo)

    p = np.linspace(1e-3 * Rout, Rout, n_p)
    dp = p[1] - p[0]
    # continuum disk normalisation (flat disk of radius R_phot, intensity I_cont)
    Fc = np.sum(np.where(p < R_phot, I_cont, 0.0) * 2.0 * np.pi * p) * dp
    if Fc <= 0:
        Fc = 1.0

    def zinterp(rr, arr):
        return np.interp(rr, r, arr, left=arr[0], right=0.0)

    inv_vth = 1.0 / max(vth_kms, 1e-3)
    for j, pj in enumerate(p):
        # line of sight z from back (-zmax) to front (+zmax); observer at +z.
        zmax = np.sqrt(max(Rout * Rout - pj * pj, 0.0))
        if zmax <= 0:
            continue
        nz = 400
        z = np.linspace(-zmax, zmax, nz)
        rr = np.sqrt(pj * pj + z * z)
        vr = zinterp(rr, v) / 1e5                     # km/s radial
        vlos = -vr * (z / np.maximum(rr, 1e-30))      # km/s toward observer (blue<0)
        chi = zinterp(rr, chi_line)
        Sl = zinterp(rr, S_line)
        dz = z[1] - z[0]
        # background intensity entering the BACK of the ray:
        #  the photospheric disk shines only through impact parameters p<R_phot,
        #  from the far side — it is the boundary condition at z=-zmax if p<R_phot
        #  AND the near-side hemisphere doesn't occult it (handled by integrating
        #  through). For p<R_phot the ray starts on the photosphere.
        on_disk = pj < R_phot
        # Occultation: for a ray crossing the photospheric disk (p<R_phot) the
        # far/back hemisphere (z<0) is hidden behind the disk; emission starts at
        # the disk (I = I_cont) and integrates the near hemisphere. Off-disk rays
        # (p≥R_phot) see the whole line of sight with no continuum background.
        Iv = _ray_intensity(z, vlos, chi, Sl, vgrid, inv_vth, dz,
                            I_start=(I_cont if on_disk else 0.0),
                            occult=(occultation and on_disk))
        F += Iv * 2.0 * np.pi * pj * dp
    return F / Fc


def _ray_intensity(z, vlos, chi, Sl, vgrid, inv_vth, dz, I_start, occult):
    """Formal solution I(v_obs) along one impact-parameter ray, all v_obs at once.
    Integrates from back (z<0) to front (z>0, observer). Returns I on vgrid."""
    nvo = vgrid.size
    I = np.full(nvo, I_start, float)
    for k in range(z.size):
        if occult and z[k] < 0:
            continue
        x = (vgrid - vlos[k]) * inv_vth
        dtau = chi[k] * np.exp(-x * x) * dz
        e = np.exp(-dtau)
        I = I * e + Sl[k] * (1.0 - e)
    return I


def escatter_redistribute(vgrid_kms, F, tau_es, T_e=1e4, v_bulk_kms=0.0):
    """Convolve an intrinsic line profile with the Thomson electron-scattering
    redistribution kernel of a scattering envelope of optical depth tau_es.

    Thomson scattering conserves photons but random-walks them in frequency by
    the electron thermal Doppler width per scatter; after ~N≈τ_es(1+τ_es) scatters
    the cumulative shift builds the broad, roughly symmetric wings that DEFINE
    dense-CSM (IIn/PPISN) profiles. Kernel: Gaussian of width
        σ_v = v_th,e · sqrt(N),   v_th,e = sqrt(2 k T_e / m_e)   [km/s]
    Photon number (∫F dv over the line excess) is conserved to machine precision.
    tau_es→0 ⇒ identity.

    v_bulk_kms > 0 (opt-in, --urt-aniso-es): adds the BULK-EXPANSION asymmetry
    (Chugai 2001; Huang & Chevalier 2018) — photons scattering in an expanding
    flow are systematically redshifted, building the observed red-skewed IIn
    wing. Modeled as a one-sided exponential tail (redward, v_obs>0) of scale
        x0 = v_bulk · min(τ_es, 3)          [km/s]
    convolved with the thermal Gaussian; photon-conserving; v_bulk=0 ⇒ exactly
    the symmetric kernel (default path unchanged).
    """
    vgrid = np.asarray(vgrid_kms, float)
    F = np.asarray(F, float)
    if tau_es <= 1e-3 or vgrid.size < 5:
        return F.copy()
    vth_e = np.sqrt(2.0 * _KB * T_e / _ME) / 1e5      # km/s
    N = tau_es * (1.0 + tau_es)
    sig = vth_e * np.sqrt(N)
    dv = float(np.mean(np.diff(vgrid)))
    if dv <= 0 or sig <= 0:
        return F.copy()
    x0 = float(v_bulk_kms) * min(tau_es, 3.0) if v_bulk_kms > 0 else 0.0
    # cap the kernel half-width to fit the grid so the convolution always returns
    # len(F) (np.convolve 'same' otherwise returns the longer kernel length).
    half = int(min(max(4, np.ceil((5 * sig + 8 * x0) / dv)), (vgrid.size - 1) // 2))
    kx = np.arange(-half, half + 1) * dv
    ker = np.exp(-0.5 * (kx / sig) ** 2)
    if x0 > dv * 0.25:
        # red-sided exponential (v_obs>0 = redshift), then combine by convolution
        ker_r = np.where(kx >= 0, np.exp(-kx / x0), 0.0)
        ker_r /= ker_r.sum()
        ker = np.convolve(ker, ker_r, mode='same')
    ker /= ker.sum()
    excess = F - 1.0                                   # scatter only the line photons
    conv = np.convolve(excess, ker, mode='same')
    return 1.0 + conv


# ===========================================================================
# PHASE 2 — top-level coupling: state + per-line NLTE inputs → emergent F_norm.
# Switch-free (no gate); uses the pipeline's NLTE source as the creation term and
# adds nonlocal scattering (ALI) + gate-free ray-trace + electron-scatter wings.
# ===========================================================================
_SIGMA_T = 6.652458732e-25     # Thomson cross-section [cm^2]
_H_PL = 6.62607015e-27
_C_CGS = 2.99792458e10
_SIGMA_INT = 0.02654           # π e²/(m_e c) [cm² Hz]


def _planck_nu(nu, T):
    T = max(float(T), 1.0)
    x = _H_PL * nu / (_KB * T)
    x = np.clip(x, 1e-8, 700.0)
    return 2.0 * _H_PL * nu ** 3 / _C_CGS ** 2 / np.expm1(x)


def _planck_lam(lam0_AA, T):
    """B_λ(T) [erg/s/cm²/Å/sr] — MUST match the pipeline's source-function units
    (mc_multi_line._construct_source_function returns S per Å). Using a per-Hz
    Planck here mismatches S_zone by λ²/c and corrupts the S/continuum balance
    (→ emission where P-Cygni absorption is physical)."""
    T = max(float(T), 1.0)
    lam_cm = lam0_AA * 1e-8
    x = _H_PL * _C_CGS / (lam_cm * _KB * T)
    x = np.clip(x, 1e-8, 700.0)
    return (2.0 * _H_PL * _C_CGS ** 2 / lam_cm ** 5) / np.expm1(x) * 1e-8


def _planck_lam_vec(lam0_AA, T):
    """Per-Å Planck B_λ for an array of temperatures."""
    lam_cm = lam0_AA * 1e-8
    x = np.clip(_H_PL * _C_CGS / (lam_cm * _KB * np.maximum(np.asarray(T, float), 1.0)),
                1e-6, 700.0)
    return (2.0 * _H_PL * _C_CGS ** 2 / lam_cm ** 5) / np.expm1(x) * 1e-8


def _gamma_pi_bb(n, T):
    """Photoionization rate of hydrogen level n in an UNDILUTED blackbody T [s^-1]
    (Kramers hydrogenic cross-section). Multiply by the dilution W(r) for the
    diluted photospheric field."""
    nu_th = 3.288e15 / n ** 2
    nu = np.linspace(nu_th, nu_th * 8.0, 2000)
    sig = 7.906e-18 * n * (nu_th / nu) ** 3
    x = np.clip(_H_PL * nu / (_KB * max(float(T), 1.0)), 1e-6, 700.0)
    Bnu = 2.0 * _H_PL * nu ** 3 / _C_CGS ** 2 / np.expm1(x)
    return float(np.trapezoid(4.0 * np.pi * Bnu / (_H_PL * nu) * sig, nu))


# H-line recombination creation: lambda_rest -> (n_upper, alpha_eff case-B at 1e4 K
# [cm^3/s]). Drives the C_rec term (robust first-principles emissivity) and the
# photoionization-destruction upper level. He lines omit both terms (their trapped
# limit is carried by the He-NLTE S_zone; the scattering limit needs neither).
_H_LINE_REC = {6562.8: (3, 1.17e-13), 4861.3: (4, 3.03e-14), 4340.5: (5, 1.2e-14),
               18751.0: (4, 3.0e-14), 12818.1: (5, 1.4e-14)}

# He I recombination creation (opt-in, --urt-he-rec): lambda_rest ->
# (n_eff, alpha_eff at 1e4 K [cm^3/s]). alpha_eff = case-B effective
# recombination into the line (provisional, Benjamin+ 1999-level accuracy —
# morphology only; the STRENGTH stays the He-NLTE budget). n_eff = hydrogenic
# effective quantum number of the UPPER level from its binding energy, used for
# the photoionization-destruction term (same self-regulating structure that
# validated for H). eta = h nu/(4 pi) * alpha_eff * n_e * n_He+ with
# n_He+ = f_ion * n_e (f_ion ~ 0.1 H-rich, ~1 H-free; passed by the caller).
_HE_LINE_REC = {5875.6: (3.0, 4.4e-14), 6678.2: (3.0, 1.3e-14),
                7065.2: (3.0, 6.0e-15), 10830.3: (1.9, 9.0e-15)}


def zone_trap_weight(r, v, nwin=None):
    """Zone-RESOLVED trapping weight w_i in [0,1] (opt-in, --urt-zone-w).

    Windowed version of homology_trap_weight: for each zone, the homology
    spread of r/v and the |dv|-magnitude-weighted reversal fraction are
    evaluated over a local window, so a STRATIFIED snapshot (homologous inner
    ejecta + quasi-static dense shell — the mid-transition IIn geometry) gets a
    per-zone scatter/trap split instead of one global blend. Magnitude-weighted
    reversals make the measure immune to single-zone velocity noise. Reduces to
    ~the global weight for regime-pure flows; lightly smoothed to avoid window-
    edge jitter.
    """
    r = np.asarray(r, float); v = np.asarray(v, float)
    n = r.size
    if n < 5:
        return np.full(n, homology_trap_weight(r, v))
    if nwin is None:
        nwin = max(3, n // 10)
    w = np.empty(n)
    rv = r / np.maximum(np.abs(v), 1e-30)
    dv = np.diff(v)
    adv = np.abs(dv)
    for i in range(n):
        lo = max(0, i - nwin); hi = min(n, i + nwin + 1)
        seg = rv[lo:hi]
        spread = float(np.std(seg) / np.maximum(np.median(seg), 1e-30))
        dlo = max(0, lo); dhi = min(dv.size, hi - 1)
        if dhi > dlo and adv[dlo:dhi].sum() > 0:
            frev = float(adv[dlo:dhi][dv[dlo:dhi] < 0].sum()
                         / adv[dlo:dhi].sum())
        else:
            frev = 0.0
        w[i] = np.clip(max(spread / 0.5, frev / 0.10), 0.0, 1.0)
    # light smoothing (box of ~nwin/2) against window-edge jitter
    k = max(1, nwin // 2)
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    w = np.convolve(np.pad(w, k, mode='edge'), ker, mode='valid')
    return np.clip(w, 0.0, 1.0)


def homology_trap_weight(r, v):
    """Continuous Sobolev-validity weight w in [0,1] for the SOURCE blend.

    w -> 0 : strongly sheared / homologous flow (photons Doppler-escape the local
             resonance; the local-NLTE trapped-J̄ over-pumps the upper level, so
             the raw S_zone absolute is invalid -> use the scattering source).
    w -> 1 : quasi-static / velocity-reversed dense CSM (resonance trapping is
             real; the NLTE S_zone is the correct local source -> use it).

    Same physics as formal_line_profile.sobolev_validity (homology spread +
    velocity-reversal fraction, thresholds 0.5 / 0.10), but CONTINUOUS — the
    binary gate becomes a smooth blend, removing the regime cliff between epochs.
    """
    r = np.asarray(r, float); v = np.asarray(v, float)
    rv = r / np.maximum(np.abs(v), 1e-30)
    spread = float(np.std(rv) / np.maximum(np.median(rv), 1e-30))
    dv = np.diff(v)
    frev = float(np.mean(dv < 0)) if dv.size else 0.0
    m = max(spread / 0.5, frev / 0.10)
    return float(np.clip(m, 0.0, 1.0))


def unified_line_profile(r, v, T_gas, n_e, line, R_phot, T_phot,
                         vgrid_kms=None, vturb_kms=25.0, n_p=140,
                         use_ali=True, verbose=False, he_f_ion=None):
    """Unified emergent line profile (F/F_cont) for one line — one transport,
    one continuous source, valid across all regimes.

    r, v, T_gas, n_e : per-zone state (cgs; v = radial speed [cm/s]).
    line : dict from mc_multi_line line-input extraction — needs
           'lambda_rest'[Å], 'tau_zone', 'S_zone', 'n_lower_zone', 'A_ul',
           'g_l', 'g_u'.
    R_phot, T_phot : photosphere radius [cm] and temperature [K].
    Returns (lam_grid_AA, F_norm).

    SOURCE (v2, validated against the trusted per-regime references):
      S_i = (1-w)·S_scatter,i + w·S_trap,i   [disk units]
      S_scatter = [(1-ε_eff)β_i W_i + ε_eff B_i/Ic + C_rec,i] / [ε_eff + β_i(1-ε_eff)]
        — the Sobolev-regularized two-level source: β from the zone Sobolev τ,
          W geometric dilution (the formal solver's validated J̄=W·Ic limit),
          thermal creation ε·B(T_gas), first-principles recombination creation
          C_rec=η_rec/χ (H lines), and destruction ε_eff = ε_coll + ε_pi with
          ε_pi = W·Γ_pi(n_up,T_phot)/(W·Γ_pi+A_ul) the photoionization of the
          upper level by the diluted photospheric field (what prevents the
          trapped recombination photons from over-pumping the homologous case).
      S_trap = clip(S_zone/Ic, 0, 150)
        — the local NLTE source, valid where resonance trapping is real
          (quasi-static dense CSM; e.g. A11's S_zone/Ic≈20 == its validated
          emission amplitude).
      w = homology_trap_weight(r, v): the CONTINUOUS Sobolev-validity measure
        (A1 IIP: w≈0.01 → P-Cygni; A11 IIn: w=1.0 → emission), replacing the
        binary gate with a smooth physical blend.
    TRANSPORT: observer-frame ray-trace (any v-field; occultation) →
      electron-scattering redistribution → anti-alias smoothing.

    NOTE use_ali is retained for signature compatibility; the deep-scattering
    ALI engine (solve_two_level_ali) remains for the analytic validation suite,
    but the production source is the closed-form blend above (the static-grid
    Feautrier over-traps sheared flows — Doppler decoupling — so the ALI J̄ is
    not used for the profile).
    """
    r = np.asarray(r, float); v = np.asarray(v, float)
    T_gas = np.maximum(np.asarray(T_gas, float), 1.0)
    n_e = np.maximum(np.asarray(n_e, float), 0.0)
    lam0 = float(line['lambda_rest'])
    nu0 = _C_CGS / (lam0 * 1e-8)
    A_ul = float(line['A_ul']); g_l = float(line['g_l']); g_u = float(line['g_u'])
    tau_zone = np.maximum(np.asarray(line['tau_zone'], float), 0.0)
    S_zone = np.asarray(line['S_zone'], float)          # NLTE source (erg/s/cm²/Hz/sr)

    Ic = _planck_lam(lam0, T_phot)                      # continuum disk (per-Å, matches S_zone)
    if not (Ic > 0):
        Ic = 1.0

    # --- local line opacity χ_line [cm^-1] from the Sobolev τ and the velocity
    #     gradient: a photon crossing the resonance accumulates ≈ τ_zone at line
    #     centre, i.e. χ0·(√π·vth)/|dv/dr| = τ_zone  →  χ0 = τ_zone·|dv/dr|/(√π·vth).
    vth = np.sqrt(2.0 * _KB * T_gas / (4.0 * 1.6726e-24))    # He/H-ish thermal [cm/s]
    vth = np.maximum(vth, vturb_kms * 1e5)                    # turbulent floor
    with np.errstate(divide='ignore', invalid='ignore'):
        dvdr = np.abs(np.gradient(v, r))
    dvdr = np.where(np.isfinite(dvdr) & (dvdr > 0), dvdr, np.nanmedian(dvdr[dvdr > 0]) if np.any(dvdr > 0) else 1e-9)
    chi_line = tau_zone * dvdr / (np.sqrt(np.pi) * vth)
    chi_line = np.where(np.isfinite(chi_line) & (chi_line > 0), chi_line, 0.0)

    # --- v2 source: continuous scatter/trap blend (see docstring) ---
    # collisional destruction ε_coll = q_ul n_e/(q_ul n_e + A_ul)
    q_ul = 8.63e-6 / (g_u * np.sqrt(T_gas))                  # cm³/s (Ω~1)
    Cul = n_e * q_ul
    eps = np.clip(Cul / (Cul + max(A_ul, 1e-30)), 1e-12, 1.0)
    # zone Sobolev escape
    beta = np.where(tau_zone > 1e-6,
                    (1.0 - np.exp(-np.minimum(tau_zone, 700.0)))
                    / np.maximum(tau_zone, 1e-30), 1.0)
    # geometric dilution of the photospheric field
    xdil = np.clip((R_phot / np.maximum(r, R_phot)) ** 2, 0.0, 1.0)
    W = 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - xdil, 0.0)))
    Bn = np.clip(_planck_lam_vec(lam0, T_gas) / Ic, 0.0, 1e6)   # thermal, disk units
    # H-line recombination creation + photoionization destruction
    Cn = np.zeros_like(r)
    eps_pi = 0.0
    for _lamH, (_nup, _aeff) in _H_LINE_REC.items():
        if abs(lam0 - _lamH) < 5.0:
            a_T = _aeff * (np.maximum(T_gas, 100.0) / 1e4) ** -0.9
            eta = _H_PL * nu0 / (4.0 * np.pi) * a_T * n_e * n_e   # n_p ≈ n_e (H-rich)
            dlamD = lam0 * vth / _C_CGS                            # Doppler width [Å]
            Cn = np.where(chi_line > 0,
                          eta / (np.sqrt(np.pi) * np.maximum(dlamD, 1e-10))
                          / np.maximum(chi_line, 1e-30) / Ic, 0.0)
            _G = _gamma_pi_bb(_nup, T_phot)
            eps_pi = W * _G / (W * _G + max(A_ul, 1e-30))
            break
    else:
        # He I recombination creation (opt-in --urt-he-rec): same structure as
        # the H term with n_He+ = f_ion·n_e, plus the SAME photoionization-
        # destruction regulator (hydrogenic n_eff of the upper level) that keeps
        # the homologous scattering limit from over-amplifying.
        if os.environ.get('SNLT_URT_HE_REC'):
            _fi = 0.1 if he_f_ion is None else float(he_f_ion)
            for _lamHe, (_neff, _aeff) in _HE_LINE_REC.items():
                if abs(lam0 - _lamHe) < 5.0:
                    a_T = _aeff * (np.maximum(T_gas, 100.0) / 1e4) ** -1.0
                    eta = _H_PL * nu0 / (4.0 * np.pi) * a_T * n_e * (_fi * n_e)
                    dlamD = lam0 * vth / _C_CGS
                    Cn = np.where(chi_line > 0,
                                  eta / (np.sqrt(np.pi) * np.maximum(dlamD, 1e-10))
                                  / np.maximum(chi_line, 1e-30) / Ic, 0.0)
                    _G = _gamma_pi_bb(_neff, T_phot)
                    eps_pi = W * _G / (W * _G + max(A_ul, 1e-30))
                    break
    eps_eff = np.clip(eps + eps_pi, 1e-12, 1.0)
    S_scatter = ((1.0 - eps_eff) * beta * W + eps_eff * Bn + Cn) \
        / np.maximum(eps_eff + beta * (1.0 - eps_eff), 1e-12)
    S_trap = np.clip(S_zone / Ic, 0.0, 150.0)
    if os.environ.get('SNLT_URT_ZONE_W'):
        # zone-RESOLVED trap weight (opt-in --urt-zone-w): stratified snapshots
        # (homologous ejecta + quasi-static shell) get a per-zone source split.
        w_trap = zone_trap_weight(r, v)
        _w_rep = float(np.median(w_trap))
    else:
        w_trap = homology_trap_weight(r, v)
        _w_rep = w_trap
    S_disk = (1.0 - w_trap) * S_scatter + w_trap * S_trap      # disk units (÷Ic)
    S_disk = np.where(np.isfinite(S_disk) & (S_disk >= 0), S_disk, 0.0)
    if verbose:
        print(f"[urt] λ{lam0:.0f}: w_trap={_w_rep:.2f} eps_pi={np.max(np.atleast_1d(eps_pi)):.1e} "
              f"S/Ic=[{S_disk.min():.3g},{S_disk.max():.3g}]")
    # Debug dump: capture the exact solver inputs for standalone iteration
    # (SNLT_URT_DUMP=<dir>)
    _dumpdir = os.environ.get('SNLT_URT_DUMP')
    if _dumpdir:
        try:
            os.makedirs(_dumpdir, exist_ok=True)
            np.savez(os.path.join(_dumpdir, f'ali_inputs_{lam0:.0f}.npz'),
                     tau_zone=tau_zone, eps=eps, Ic=Ic, S_zone=S_zone,
                     chi_line=chi_line, r=r, v=v, T_gas=T_gas, n_e=n_e,
                     R_phot=R_phot, T_phot=T_phot, w_trap=w_trap,
                     S_scatter=S_scatter, S_trap=S_trap, S_disk=S_disk)
        except Exception:
            pass

    # --- emergent profile (any v-field) ---
    if vgrid_kms is None:
        vmax = 1.25 * float(np.max(np.abs(v))) / 1e5
        vgrid_kms = np.linspace(-vmax, vmax, 351)
    # Doppler width for the ray-trace: the thermal/turbulent vth, but floored at
    # ~1.5× the velocity-grid pixel. A line cannot be resolved below one spectral
    # bin, and — more physically — the intra-zone velocity shear |dv/dr|·dr across
    # a STELLA zone (hundreds–thousands of km/s in SN ejecta) already exceeds the
    # 25 km/s turbulent floor, so a sub-pixel thermal width both under-broadens and
    # ALIASES (the exp(-x²) resonance jumps between grid cells → spiky profile).
    _dv_grid = float(np.median(np.abs(np.diff(np.asarray(vgrid_kms, float)))))
    _vth_prof = max(float(np.median(vth)) / 1e5, 3.0 * _dv_grid)
    F = emergent_profile(r, v, S_disk, chi_line, R_phot, 1.0, vgrid_kms,
                         vth_kms=_vth_prof, occultation=True,
                         n_p=n_p)

    # --- electron-scattering redistribution (τ_es of the overlying envelope) ---
    dr = np.abs(np.diff(r, prepend=r[0]))
    tau_es_r = np.cumsum((n_e * _SIGMA_T * dr)[::-1])[::-1]
    tau_es_line = float(np.average(tau_es_r, weights=np.maximum(chi_line * dr, 1e-30))) \
        if np.any(chi_line > 0) else 0.0
    # bulk-expansion red-skew (opt-in --urt-aniso-es): characteristic flow speed
    # of the line-forming gas drives the one-sided redward tail.
    _vb = 0.0
    if os.environ.get('SNLT_URT_ANISO_ES') and np.any(chi_line > 0):
        _vb = float(np.average(np.abs(v), weights=np.maximum(chi_line * dr, 1e-30))) / 1e5
    F = escatter_redistribute(vgrid_kms, F, tau_es_line, T_e=float(np.median(T_gas)),
                              v_bulk_kms=_vb)

    # final light Gaussian smoothing (σ ≈ 1.5 grid pixels) to remove any residual
    # ray-trace sampling structure below the resolution — photon-conserving on the
    # line excess (the continuum baseline is untouched). Physically, the profile
    # carries no information below ~a resolution element.
    _exc = F - 1.0
    _sig_pix = 2.5
    _hw = int(np.ceil(3.0 * _sig_pix))
    _kx = np.arange(-_hw, _hw + 1)
    _kern = np.exp(-0.5 * (_kx / _sig_pix) ** 2); _kern /= _kern.sum()
    _exc_s = np.convolve(_exc, _kern, mode='same')
    # renormalise to conserve the integrated excess (area = EW proxy)
    _a0 = float(np.trapezoid(_exc, vgrid_kms)); _a1 = float(np.trapezoid(_exc_s, vgrid_kms))
    if _a1 != 0 and np.isfinite(_a0 / _a1):
        _exc_s *= (_a0 / _a1)
    F = 1.0 + _exc_s

    lam_grid = lam0 * (1.0 + np.asarray(vgrid_kms) / _C_KMS)
    return lam_grid, F
