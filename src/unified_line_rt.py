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
        m = m if abs(m) > 1e-300 else 1e-300
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
    S = B.copy()                                   # LTE start
    hist = []
    converged = False
    dS = np.inf
    for it in range(max_iter):
        J, lstar = feautrier_J(tau, S, I_top=I_top, I_bottom=I_bottom,
                               semi_infinite=semi_infinite)
        # ALI update with the diagonal operator:
        #   S_new = [1 - (1-ε)Λ*]^{-1} [ (1-ε)(J - Λ*·S) + εB ]
        one_eps = 1.0 - eps
        num = one_eps * (J - lstar * S) + eps * B
        den = 1.0 - one_eps * lstar
        S_new = num / np.maximum(den, 1e-300)
        S_new = np.maximum(S_new, 0.0)
        dS = float(np.max(np.abs(S_new - S) / np.maximum(np.abs(S_new), 1e-300)))
        hist.append(S_new.copy())
        S = S_new
        # Ng acceleration every 4 iterations (uses last 3 iterates)
        if ng and len(hist) >= 4 and (it % 4 == 3):
            S = _ng_accelerate(hist[-3], hist[-2], hist[-1])
            S = np.maximum(S, 0.0)
            hist[-1] = S.copy()
        if len(hist) > 5:
            hist.pop(0)
        if dS < tol:
            converged = True
            break
    Jf, _ = feautrier_J(tau, S, I_top=I_top, I_bottom=I_bottom,
                        semi_infinite=semi_infinite)
    return {'S': S, 'J': Jf, 'n_iter': it + 1, 'converged': converged,
            'dS_last': dS}


def _ng_accelerate(s2, s1, s0):
    """Ng (1974) 3-point acceleration of the source-function iteration."""
    d0 = s0 - s1
    d1 = s1 - s2
    q0 = d0
    q1 = d0 - d1
    A = float(np.sum(q1 * q1))
    if A <= 1e-300:
        return s0
    b = float(np.sum(q1 * q0)) / A
    b = np.clip(b, -2.0, 2.0)
    return s0 + b * (s1 - s0)
