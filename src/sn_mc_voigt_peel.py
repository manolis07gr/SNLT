"""
sn_mc_voigt.py
==============
Voigt-broadened (Gaussian-profile) Monte Carlo line transport, upgraded
from the pure-Sobolev version in sn_mc_fast.py.

Physics upgrades over sn_mc_fast.py:
  (1) Line absorption uses a finite Gaussian cross section of width
      v_D = sqrt(v_th^2 + v_turb^2), integrated over the packet's path
      rather than applied as a delta at the resonance crossing. This
      is the correct treatment when bulk velocity gradient * mfp is
      comparable to or less than v_th (slow/flat winds).
  (2) Volumetric emission samples the emitted comoving frequency from
      the same Gaussian, not a delta at nu0. This gives the narrow peak
      an intrinsic width set by microturbulence.
  (3) Line re-emission after absorption samples the new comoving
      frequency from the Gaussian, consistent with the absorption profile.

Runtime cost relative to sn_mc_fast.py: ~2x slower per packet because
the inner loop now accumulates tau_line alongside tau_Thomson at every
substep (previously we only checked for resonance crossings).

Use via:
    from sn_mc_voigt import run_mc_voigt
    lam, F, stats = run_mc_voigt(..., v_turb_kms=50.0, ...)

Everything else (readers, population helpers, driver) is unchanged.
"""

from __future__ import annotations
import math
import numpy as np
from numba import njit

# constants (cgs)
C      = 2.99792458e10
KB     = 1.380649e-16
ME     = 9.1093837e-28
SIGMAT = 6.65245871e-25
EE     = 4.80320425e-10
MH     = 1.6735e-24          # H mass (for thermal velocity)


# ---------------------------------------------------------------------------
# low-level helpers (duplicated from sn_mc_fast for self-containment and to
# avoid import-ordering quirks with numba caching)
# ---------------------------------------------------------------------------
@njit(cache=True, inline='always')
def _lin_interp(x, xp, fp):
    n = xp.shape[0]
    if x <= xp[0]: return fp[0]
    if x >= xp[n - 1]: return fp[n - 1]
    lo = 0; hi = n - 1
    while hi - lo > 1:
        mid = (hi + lo) >> 1
        if xp[mid] <= x: lo = mid
        else: hi = mid
    dx = xp[hi] - xp[lo]
    if dx <= 0.0: return fp[lo]
    t = (x - xp[lo]) / dx
    return fp[lo] + t * (fp[hi] - fp[lo])


@njit(cache=True, inline='always')
def _isotropic_dir():
    phi = 2.0 * np.pi * np.random.random()
    mu  = 2.0 * np.random.random() - 1.0
    sm  = np.sqrt(max(0.0, 1.0 - mu * mu))
    return sm * np.cos(phi), sm * np.sin(phi), mu


@njit(cache=True, inline='always')
def _outward_hemisphere_dir(nx, ny, nz):
    phi = 2.0 * np.pi * np.random.random()
    mu  = np.sqrt(np.random.random())
    sm  = np.sqrt(1.0 - mu * mu)
    if abs(nz) < 0.99:
        rx, ry, rz = 0.0, 0.0, 1.0
    else:
        rx, ry, rz = 1.0, 0.0, 0.0
    t1x = ny*rz - nz*ry; t1y = nz*rx - nx*rz; t1z = nx*ry - ny*rx
    t1n = np.sqrt(t1x*t1x + t1y*t1y + t1z*t1z)
    t1x /= t1n; t1y /= t1n; t1z /= t1n
    t2x = ny*t1z - nz*t1y; t2y = nz*t1x - nx*t1z; t2z = nx*t1y - ny*t1x
    cphi = np.cos(phi); sphi = np.sin(phi)
    return (mu*nx + sm*(cphi*t1x + sphi*t2x),
            mu*ny + sm*(cphi*t1y + sphi*t2y),
            mu*nz + sm*(cphi*t1z + sphi*t2z))


@njit(cache=True, inline='always')
def _sphere_exit(px, py, pz, dx, dy, dz, r_outer, r_inner):
    b  = px*dx + py*dy + pz*dz
    r2 = px*px + py*py + pz*pz
    disc_o = b*b - (r2 - r_outer*r_outer)
    s_out = -b + np.sqrt(disc_o) if disc_o >= 0.0 else 1e300
    disc_i = b*b - (r2 - r_inner*r_inner)
    if disc_i < 0.0:
        s_in = 1e300
    else:
        s1 = -b - np.sqrt(disc_i); s2 = -b + np.sqrt(disc_i)
        s_in = s1 if s1 > 1e-20 else (s2 if s2 > 1e-20 else 1e300)
    return (s_out, 0) if s_out < s_in else (s_in, 1)


# ---------------------------------------------------------------------------
# Line profile physics
# ---------------------------------------------------------------------------
@njit(cache=True, inline='always')
def _doppler_width_Hz(nu0, v_D):
    """v_D in cm/s, returns Doppler width in Hz."""
    return nu0 * v_D / C


@njit(cache=True, inline='always')
def _line_cross_section(nu_co, nu0, sigma_D_Hz, sigma_0):
    """Gaussian line cross section [cm^2]. sigma_0 is the frequency-integrated
    absorption coefficient (pi e^2 / m_e c) * f_lu, divided by the profile
    normalization sqrt(2 pi) * sigma_D."""
    x = (nu_co - nu0) / sigma_D_Hz
    if abs(x) > 5.0:   # cheap cutoff — Gaussian is negligible beyond 5 sigma
        return 0.0
    return sigma_0 * np.exp(-0.5 * x * x)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
@njit(cache=True)
def _run_voigt(
    n_packets, step_dr, max_events,
    r_src_inner, r_abs_inner, r_outer, r_vol_inner,
    r_grid, v_grid, dvdr_grid, ne_grid, T_grid, nl_grid, nuu_grid,
    nu0, f_lu, lam0, g_l, g_u,
    v_D_grid_cms,               # per-zone Doppler width sqrt(v_th² + v_turb²) [cm/s]
    lam_lo_src, lam_hi_src,     # source wavelength range [cm]
    lam_edges,
    emissivity_cdf, emit_mode, frac_volumetric,
    eps_dest_grid,              # per-zone line destruction probability
    mu_obs_min,                 # cos(half-angle) of observer cone (1.0 = isotropic 4π
                                #   averaging; 0.95 = forward-cone observer)
    redistribution_mode,        # 0 = semi-PRD (preserve nu_co, add thermal kick),
                                # 1 = CRD (redistribute to Maxwellian around nu0)
    phot_line_boundary,         # 0 = TRANSMIT vol packets through photosphere
                                # 1 = ABSORB vol packets at photosphere (occult)
):
    nbins = lam_edges.shape[0] - 1
    spectrum = np.zeros(nbins)
    n_zones = r_grid.shape[0]
    # Per-zone line-scatter counter (phase 1 diagnostic). Counts every line-
    # absorption event (both destroyed and scattered) by the zone in which
    # it occurs. Used downstream to estimate J_bar(r) from the MC radiation
    # field for NLTE-RT iteration.
    n_scat_per_zone = np.zeros(n_zones, dtype=np.int64)
    g_ratio = g_l / g_u
    # frequency-integrated absorption strength prefactor (before population)
    # sigma_line(nu) = (pi e^2 / m_e c) * f_lu * (1/(sqrt(2pi) sigma_D)) *
    #                 exp(-(nu-nu0)^2/(2 sigma_D^2)) * (n_l - g_ratio n_u)
    # The (n_l - g_ratio n_u) factor is multiplied in at evaluation time.
    sigma_pref = np.pi * EE * EE / (ME * C) * f_lu

    n_line = 0; n_thom = 0; n_esc = 0; n_abs = 0

    for ip in range(n_packets):
        if emit_mode == 0:
            use_vol = False
        elif emit_mode == 1:
            use_vol = True
        else:
            use_vol = (np.random.random() < frac_volumetric)
        packet_weight = 1.0

        if not use_vol:
            # continuum source at inner boundary
            nhx, nhy, nhz = _isotropic_dir()
            r0 = r_src_inner * 1.00001
            px = r0*nhx; py = r0*nhy; pz = r0*nhz
            dx, dy, dz = _outward_hemisphere_dir(nhx, nhy, nhz)
            lam_s = lam_lo_src + (lam_hi_src - lam_lo_src) * np.random.random()
            nu = C / lam_s
        else:
            # volumetric emission — sample radius, isotropic direction,
            # and sample emitted comoving frequency from Gaussian around nu0
            u = np.random.random()
            k_lo = 0; k_hi = emissivity_cdf.shape[0] - 1
            while k_hi - k_lo > 1:
                km = (k_hi + k_lo) >> 1
                if emissivity_cdf[km] < u: k_lo = km
                else: k_hi = km
            r_emit = r_grid[k_hi]
            nhx, nhy, nhz = _isotropic_dir()
            px = r_emit*nhx; py = r_emit*nhy; pz = r_emit*nhz
            dx, dy, dz = _isotropic_dir()
            # thermal velocity at emission site
            T_emit = _lin_interp(r_emit, r_grid, T_grid)
            v_D_emit = _lin_interp(r_emit, r_grid, v_D_grid_cms)
            sigma_D_emit = nu0 * v_D_emit / C
            # sample comoving emitted frequency from Gaussian around nu0
            nu_co_emit = nu0 + sigma_D_emit * np.random.standard_normal()
            # transform to static frame
            vr_ = _lin_interp(r_emit, r_grid, v_grid)
            mu_ = nhx*dx + nhy*dy + nhz*dz
            nu = nu_co_emit / max(1.0 - vr_ * mu_ / C, 1e-30)

        alive = True

        # Inner boundary for sphere-exit geometry:
        # - continuum packets: r_abs_inner (photosphere, which is their source
        #   and a hard reflecting/absorbing boundary)
        # - volumetric packets: r_vol_inner (grid floor at tau_es_trim; the
        #   photosphere is NOT a hard boundary for them — they interact with
        #   photospheric gas via normal Thomson + line absorption stepping,
        #   which is physically correct and produces the proper back-side
        #   emission asymmetry for P-Cygni profiles).
        r_inner_pkt = r_vol_inner if use_vol else r_abs_inner

        for ev in range(max_events):
            if not alive: break
            s_bound, btype = _sphere_exit(px, py, pz, dx, dy, dz,
                                          r_outer, r_inner_pkt)
            tau_th_sample   = -np.log(max(np.random.random(), 1e-300))
            tau_line_sample = -np.log(max(np.random.random(), 1e-300))
            accum_tau_th   = 0.0
            accum_tau_line = 0.0

            s = 0.0
            ds = step_dr
            event = -1
            s_event = 0.0

            while s + ds <= s_bound:
                s_new = s + ds
                # midpoint for step-integrated quantities
                smx = px + 0.5 * (s + s_new) * dx
                smy = py + 0.5 * (s + s_new) * dy
                smz = pz + 0.5 * (s + s_new) * dz
                rm  = np.sqrt(smx*smx + smy*smy + smz*smz)
                vm  = _lin_interp(rm, r_grid, v_grid)
                mum = (smx*dx + smy*dy + smz*dz) / max(rm, 1e-30)
                nu_co_mid = nu * (1.0 - vm * mum / C)

                # (1) line absorption at midpoint
                Tm   = _lin_interp(rm, r_grid, T_grid)
                v_D  = _lin_interp(rm, r_grid, v_D_grid_cms)
                sigma_D = nu0 * v_D / C
                # Gaussian cutoff: only integrate if within 5 widths
                xnorm = (nu_co_mid - nu0) / max(sigma_D, 1e-30)
                if abs(xnorm) < 5.0:
                    n_l  = _lin_interp(rm, r_grid, nl_grid)
                    n_u_ = _lin_interp(rm, r_grid, nuu_grid)
                    pop  = n_l - g_ratio * n_u_
                    if pop > 0.0:
                        phi = np.exp(-0.5 * xnorm * xnorm) / (np.sqrt(2.0 * np.pi) * sigma_D)
                        # cross section in cm^2 at nu_co_mid, times population
                        kappa_line = sigma_pref * phi * pop  # [cm^-1]
                        dtau_line = kappa_line * ds
                        if accum_tau_line + dtau_line > tau_line_sample:
                            # line interaction inside this step
                            over = tau_line_sample - accum_tau_line
                            s_line = s + over / max(kappa_line, 1e-300)
                            event = 0
                            s_event = s_line
                            break
                        accum_tau_line += dtau_line

                # (2) Thomson absorption at midpoint
                ne_mid = _lin_interp(rm, r_grid, ne_grid)
                dtau_th = SIGMAT * ne_mid * ds
                if accum_tau_th + dtau_th > tau_th_sample:
                    over = tau_th_sample - accum_tau_th
                    s_th = s + over / max(SIGMAT * ne_mid, 1e-300)
                    event = 1
                    s_event = s_th
                    break
                accum_tau_th += dtau_th

                s = s_new

            if event < 0:
                # reached boundary
                px += s_bound * dx; py += s_bound * dy; pz += s_bound * dz
                if btype == 0:
                    # OBSERVER BEAMING: only count packets escaping toward
                    # the observer (placed along +z axis at infinity).
                    # mu_esc = dz = cos(angle between escape direction and LOS).
                    # Packet's static-frame frequency `nu` already includes the
                    # correct LOS Doppler shift along its own direction
                    # of propagation, so for packets going in the +z direction
                    # (toward observer) `nu` is the observer-frame frequency.
                    # mu_obs_min = 1.0 falls back to the original 4π-averaged
                    # behavior (count all escaping packets equally).
                    if mu_obs_min < 1.0:
                        if dz < mu_obs_min:
                            # packet escaped toward something other than the
                            # observer — discard.
                            n_esc += 1
                            alive = False
                            break
                        # geometric weight: solid angle the cone subtends
                        # is 2π (1 − μ_min); when packets are uniformly
                        # distributed over 4π, the fraction in the cone is
                        # (1 − μ_min)/2. Scale up by 2/(1−μ_min) to make
                        # the absolute flux match what an observer at large
                        # distance would see (per unit solid angle).
                        beam_weight = packet_weight * 2.0 / (1.0 - mu_obs_min)
                    else:
                        # 4π-averaged: every packet contributes
                        beam_weight = packet_weight
                    lam = C / nu
                    idx = np.searchsorted(lam_edges, lam) - 1
                    if idx >= 0 and idx < nbins:
                        spectrum[idx] += beam_weight
                    n_esc += 1
                    alive = False
                else:
                    # photosphere boundary: distinguish ENTERING vs EXITING
                    # First check whether packet was inside or outside r_abs_inner
                    # BEFORE this step.
                    r_pre = np.sqrt(px*px + py*py + pz*pz)
                    is_inside = (r_pre < r_abs_inner)

                    if is_inside:
                        # Packet was already inside the photosphere disk and
                        # is now CROSSING the boundary going outward. This
                        # represents emission from deep gas (post-shock shell)
                        # finally escaping through the photosphere — it must
                        # be allowed to continue. The Thomson scattering en
                        # route already accounts for in-photosphere diffusion.
                        # Just step past the boundary and continue.
                        px += s_bound * dx; py += s_bound * dy; pz += s_bound * dz
                        # nudge slightly outward to avoid re-triggering the
                        # boundary on the next step
                        nhx_ = px/r_abs_inner; nhy_ = py/r_abs_inner; nhz_ = pz/r_abs_inner
                        px += 1e-6 * r_abs_inner * nhx_
                        py += 1e-6 * r_abs_inner * nhy_
                        pz += 1e-6 * r_abs_inner * nhz_
                        continue

                    # Packet was OUTSIDE r_abs_inner and is now hitting the
                    # photosphere from above (entering the disk).
                    # Behavior depends on packet origin AND the
                    # phot_line_boundary flag:
                    # - continuum-source packets: always REFLECT. The
                    #   photospheric BB source emits outward, so a packet
                    #   coming back from the wind is treated as being
                    #   thermalized and re-emitted back into the outward
                    #   hemisphere. Real BB atmospheres conserve net flux
                    #   this way at the thermalization depth.
                    # - volumetric line packets:
                    #   * phot_line_boundary=0 (TRANSMIT): packet passes
                    #     through the boundary into the interior and
                    #     continues transporting (Thomson scattering may
                    #     redistribute it back out, usually with enough
                    #     frequency smearing to land in broad wings).
                    #   * phot_line_boundary=1 (ABSORB): packet is DESTROYED
                    #     at the boundary. Physically: at Hα line center,
                    #     line opacity χ_line ≈ 2 × χ_es at the photosphere
                    #     — line photons crossing into the photospheric disk
                    #     are predominantly line-absorbed (not Thomson-
                    #     scattered), so their contribution to the emergent
                    #     line profile is zero. This matches CMFGEN's formal-
                    #     solution treatment of photospheric occultation:
                    #     back-hemisphere line emission at impact parameter
                    #     p < R_phot is blocked by the opaque photospheric
                    #     disk, reaching the observer only through p > R_phot
                    #     sight lines. Turning this on shifts the narrow-core
                    #     emission to the red side (CMFGEN-like) at the cost
                    #     of some broad Thomson wing flux.
                    if not use_vol:
                        px += s_bound * dx; py += s_bound * dy; pz += s_bound * dz
                        rr = np.sqrt(px*px + py*py + pz*pz)
                        nhx_ = px/rr; nhy_ = py/rr; nhz_ = pz/rr
                        dx, dy, dz = _outward_hemisphere_dir(nhx_, nhy_, nhz_)
                        px *= 1.0 + 1e-6; py *= 1.0 + 1e-6; pz *= 1.0 + 1e-6
                        continue
                    else:
                        if phot_line_boundary == 1:
                            # ABSORB: destroy packet at photosphere
                            n_abs += 1
                            alive = False
                            break
                        # TRANSMIT: traverse the inner sphere as if it were
                        # vacuum (no emission, no absorption — outside our
                        # model domain). The packet enters at the inner
                        # boundary and exits on the far side along its
                        # current direction. This is the correct treatment
                        # for emission emitted at very deep zones whose
                        # random walk dips through the symmetry center —
                        # without it, the packet ends up below the grid floor
                        # and interpolation gives nonsense, breaking energy
                        # conservation for nlte_direct-style emissivity.
                        #
                        # Chord geometry: from entry point at radius r_inner
                        # along (dx,dy,dz), the chord length through the
                        # inner sphere is 2 × |b'| where b' is the projection
                        # of position-from-center onto direction. Since we
                        # are AT the boundary, position vector p has
                        # |p| = r_inner. Chord exit happens at parameter
                        # s = -2 * (p · d) (because both entry and exit
                        # are on |p + s d| = r_inner).
                        px += s_bound * dx
                        py += s_bound * dy
                        pz += s_bound * dz
                        # Now |p| = r_inner (at entry). Compute chord length.
                        b_proj = px*dx + py*dy + pz*dz
                        s_chord = -2.0 * b_proj   # always positive when
                                                  # entering the inner sphere
                        if s_chord < 0.0:
                            # Numerical edge case — direction is outward at
                            # this point; skip the chord and just nudge out.
                            s_chord = 0.0
                        px += s_chord * dx
                        py += s_chord * dy
                        pz += s_chord * dz
                        # Nudge slightly outward to avoid numerical re-entry
                        rr = np.sqrt(px*px + py*py + pz*pz)
                        if rr > 0.0:
                            px *= (1.0 + 1e-7)
                            py *= (1.0 + 1e-7)
                            pz *= (1.0 + 1e-7)
                        continue
                break

            # move to interaction and process
            px += s_event * dx; py += s_event * dy; pz += s_event * dz
            if event == 0:
                # line absorption: before re-emitting, check if photon is
                # destroyed via a branching channel (e.g. Hα absorption leads
                # to n=3, which can emit Lyβ instead of Hα — 55.8% probability
                # for pure radiative branching). Locally trapped Lyβ can
                # recycle back to Hα, so the *effective* ε depends on τ_Lyβ.
                rxx = np.sqrt(px*px + py*py + pz*pz)
                # Per-zone line-absorption counter (phase 1). Count EVERY
                # line absorption here, whether the packet is subsequently
                # destroyed or coherently scattered. This is the quantity
                # proportional to n_2 × B_lu × J_bar × V_zone needed for
                # the J_bar extraction in the NLTE-RT iteration.
                # Zone index from radius via binary search on r_grid.
                iz = np.searchsorted(r_grid, rxx) - 1
                if iz < 0:
                    iz = 0
                elif iz >= n_zones:
                    iz = n_zones - 1
                n_scat_per_zone[iz] += 1
                eps_here = _lin_interp(rxx, r_grid, eps_dest_grid)
                if np.random.random() < eps_here:
                    # destroyed — photon converted to other channels (Lyβ
                    # escape, 2-photon decay of 2s, collisional de-excitation)
                    n_abs += 1
                    alive = False
                    break
                # Coherent scattering in the FLUID (comoving) frame.
                # This is PRD (partial frequency redistribution) in its
                # simplest form: the comoving-frame frequency is preserved
                # across a scatter, only the direction changes. The classic
                # Sobolev regime for resonance lines in expanding media.
                #
                # Why not CRD (sample new nu_co from Gaussian around nu0):
                # CRD lets line-center photons randomly walk in frequency and
                # eventually diffuse out via wing escape. In optically thick
                # winds, this erases the central self-absorption feature
                # because line-center photons can escape "through" the line
                # by being redistributed to the wings. PRD preserves the
                # in-fluid-frame frequency so line-center photons stay at
                # line center in every fluid-frame, and escape only via
                # Doppler-shift accumulation across velocity gradients —
                # i.e. from the last resonance-crossing surface (Sobolev).
                # This creates the central self-absorption feature that CRD
                # washes out.
                #
                # Algorithm:
                #   1. Compute current fluid-frame frequency at the absorption
                #      point: nu_co = nu_static × (1 − v_r × mu_old / c)
                #      where mu_old is the cosine of the incoming direction.
                #   2. Add a small thermal Doppler shift (electron thermal
                #      motion; much smaller than v_turb for gas atoms).
                #   3. Pick isotropic new direction.
                #   4. Transform back to static frame with the NEW direction:
                #      nu_new = nu_co / (1 − v_r × mu_new / c)
                Tx  = _lin_interp(rxx, r_grid, T_grid)
                vr_ = _lin_interp(rxx, r_grid, v_grid)
                # Step 1: comoving frequency at this point, with OLD direction
                mu_old = (px*dx + py*dy + pz*dz) / max(rxx, 1e-30)
                nu_co = nu * (1.0 - vr_ * mu_old / C)
                # Step 2: thermal Doppler width (atom thermal + microturbulence)
                v_D  = _lin_interp(rxx, r_grid, v_D_grid_cms)
                sigma_D = nu0 * v_D / C
                # Frequency redistribution. Three modes:
                #   redistribution_mode == 0  (semi-PRD, default/legacy):
                #     nu_co_new = nu_co + sigma_D × N(0,1)
                #     Preserves the photon's incoming comoving frequency and
                #     adds a thermal-width kick (atom's thermal motion). Line-
                #     wing photons stay in the wing across scatters — creates
                #     sharp P-Cygni absorption features.
                #   redistribution_mode == 1  (CRD):
                #     nu_co_new = nu0 + sigma_D × N(0,1)
                #     Resamples comoving frequency from a Maxwellian around
                #     line center, independent of incoming frequency. Closer
                #     to CMFGEN's angle-averaged redistribution for Doppler-
                #     core resonance lines. Pulls wing photons back to line
                #     center → more scatters → more angular redistribution →
                #     broader, smoother narrow core. Known risk: can soften
                #     P-Cygni absorption depth in optically thick winds.
                #   redistribution_mode == 2  (AA-PRD / angle-averaged PRD):
                #     nu_co_new = (1-w)×nu_co + w×nu0 + sigma_D × N(0,1)
                #     where w = exp(-½ x²) and x = (nu_co - nu0) / sigma_D.
                #     Physical motivation: for Doppler-core resonance
                #     scattering (Hummer's R_II-A function), line-center
                #     absorptions (|x| ≤ 1) should redistribute approximately
                #     coherently with the thermal pool (CRD-like), while
                #     line-wing absorptions (|x| ≥ 2) preserve frequency
                #     correlation (PRD-like). w = exp(-½x²) smoothly blends
                #     these limits with no free parameter. Intended to
                #     recover most of CRD's narrow-core morphology gain
                #     while preserving most of semi-PRD's P-Cygni trough
                #     depth.
                if redistribution_mode == 1:
                    nu_co_new = nu0 + sigma_D * np.random.standard_normal()
                elif redistribution_mode == 2:
                    x_offset = (nu_co - nu0) / max(sigma_D, 1e-30)
                    w_mix = np.exp(-0.5 * x_offset * x_offset)
                    nu_co_blend = (1.0 - w_mix) * nu_co + w_mix * nu0
                    nu_co_new = nu_co_blend + sigma_D * np.random.standard_normal()
                else:
                    nu_co_new = nu_co + sigma_D * np.random.standard_normal()
                # Step 3: isotropic new direction
                dx, dy, dz = _isotropic_dir()
                # Step 4: transform back to static frame
                mu_ = (px*dx + py*dy + pz*dz) / max(rxx, 1e-30)
                nu = nu_co_new / max(1.0 - vr_ * mu_ / C, 1e-30)
                n_line += 1
            else:
                # Thomson scatter
                rxx = np.sqrt(px*px + py*py + pz*pz)
                T_  = _lin_interp(rxx, r_grid, T_grid)
                vr_ = _lin_interp(rxx, r_grid, v_grid)
                mu_old = (px*dx + py*dy + pz*dz) / max(rxx, 1e-30)
                nu_co = nu * (1.0 - vr_ * mu_old / C)
                dx, dy, dz = _isotropic_dir()
                mu_new = (px*dx + py*dy + pz*dz) / max(rxx, 1e-30)
                sigma = nu_co * np.sqrt(2.0 * KB * T_ / (ME * C * C))
                nu_co_new = nu_co + sigma * np.random.standard_normal()
                nu = nu_co_new / max(1.0 - vr_ * mu_new / C, 1e-30)
                n_thom += 1
        else:
            n_abs += 1

    return spectrum, n_line, n_thom, n_esc, n_abs, n_scat_per_zone


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def run_mc_voigt(
    r, v, rho, T, n_e, n_lower, n_upper, line,
    v_turb_kms=50.0,
    v_turb_kms_grid=None,
    n_packets=100_000, step_dr_frac=3e-3,
    wavelength_range_AA=(6200.0, 6900.0),
    source_padding_AA=1500.0, nbins=250, max_events=2000,
    tau_es_trim=None,
    emit_mode='continuum', emissivity=None, frac_volumetric=0.5,
    eps_dest=None,
    mu_obs_min=0.95,
    deep_continuum_source=False,
    normalize=True,
    line_scatter_redistribution='semi_prd',
    photosphere_line_boundary='transmit',
    seed=42, verbose=True,
):
    """Voigt MC with per-zone line destruction and observer beaming.

    deep_continuum_source : bool, default False
        If True, continuum-mode packets are emitted from the innermost
        kept zone (τ_es ≈ tau_es_trim, the thermalization depth). They
        traverse the full Thomson optical depth on their way out,
        producing the broad electron-scattering wings characteristic of
        IIn supernovae (Hα emission extending to ±5000 km/s with
        F/F_cont ~ 1.3-2.0).
        Cost: 4-5× slower per packet because each packet undergoes
        ~τ_es_trim Thomson scatters.
        If False (default), continuum is emitted from R_phot
        (τ_es=1 surface). Faster; produces narrower line core but the
        morphology of the line center is unchanged.

    mu_obs_min : float in [0, 1]
        cos(half-angle) of the observer cone, with the observer placed at
        +z infinity. Only escaping packets with dz >= mu_obs_min count
        toward the synthesized spectrum, weighted by the inverse cone
        solid-angle fraction so the absolute flux level matches what an
        observer would see. mu_obs_min = 0.95 → ~18° half-cone, gives a
        proper line-of-sight P-Cygni profile with the asymmetric
        blueshifted absorption / redshifted emission characteristic of
        outflowing media. Set mu_obs_min = 1.0 to recover the original
        4π-averaged behavior (no asymmetry from outflow geometry).
        Smaller values (wider cones) give noisier but more LOS-averaged
        profiles; larger values (narrower cones) give crisper P-Cygni
        structure but higher Poisson noise.

    eps_dest : None, scalar, or array of length len(r)
        Destruction probability per line absorption event. For Hα (n=3 upper
        level), pure radiative branching gives ε = A_31/(A_31+A_32) = 0.558
        when Lyβ escapes freely. When Lyβ is locally trapped, absorbed
        Lyβ photons get re-cycled via n=3 and can re-emit as Hα, so the
        effective ε is reduced by (1 - (1-β_Lyβ) × A_32/(A_31+A_32)):

            ε_eff = ε_direct × β_Lyβ / (1 - (1-β_Lyβ) × (1 - ε_direct))

        In the optically thin limit (β_Lyβ=1): ε_eff = 0.558.
        In the thick limit (β_Lyβ→0): ε_eff → 0 (all absorbed photons
        eventually emerge as Hα via Lyβ recycling).
        If eps_dest is None, fall back to ε = 0.558 (thin Lyβ limit).

    line_scatter_redistribution : {'semi_prd', 'crd', 'aa_prd'}, default 'semi_prd'
        Frequency redistribution treatment applied to volumetric line-
        scatter events in the kernel.
        - 'semi_prd' (default, legacy behaviour): preserves the photon's
          incoming comoving frequency at the scatter point and adds a
          thermal-width kick. Line-wing photons stay in the wing across
          scatters, reproducing sharp P-Cygni absorption.
        - 'crd': resamples the post-scatter comoving frequency from a
          Maxwellian around line center, independent of the incoming
          frequency. Closer to CMFGEN's angle-averaged redistribution for
          Doppler-core resonance lines. Redistributes wing photons back to
          line center, driving more scatters and smoother angular
          redistribution. Known risk: softens P-Cygni absorption depth in
          optically thick winds.
        - 'aa_prd': angle-averaged PRD — blends semi-PRD and CRD with a
          weight w = exp(-½ x²) where x = (nu_co - nu0)/sigma_D is the
          photon's fluid-frame frequency offset from line center in
          Doppler widths. Line-center photons (x ≈ 0) redistribute CRD-like
          (w → 1); line-wing photons (|x| ≥ 2) preserve their frequency
          PRD-like (w → 0). Qualitative approximation of Hummer's R_II-A
          redistribution function. No free parameters. Intended as the
          best-of-both-worlds mode for IIn applications.
    """
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    n_e = np.asarray(n_e, dtype=np.float64)
    n_lower = np.asarray(n_lower, dtype=np.float64)
    n_upper = np.asarray(n_upper, dtype=np.float64)

    dr = np.empty_like(r); dr[:-1] = np.diff(r); dr[-1] = dr[-2]
    tau_es = np.cumsum((SIGMAT * n_e * dr)[::-1])[::-1]
    # tau_es_trim cuts inner zones for transport efficiency only.
    if tau_es_trim is not None:
        above = tau_es <= tau_es_trim
        if not np.any(above):
            raise ValueError(f"no zones with tau_es <= {tau_es_trim}")
        i0 = int(np.argmax(above))
    else:
        i0 = 0

    r_t = r[i0:]; v_t = v[i0:]; T_t = T[i0:]; ne_t = n_e[i0:]
    nl_t = n_lower[i0:]; nu_t = n_upper[i0:]
    with np.errstate(divide='ignore', invalid='ignore'):
        dvdr_t = np.gradient(v_t, r_t)
    # robust to DUPLICATE STELLA shock radii: a raw gradient is NaN there (zero
    # spacing) and poisons the Sobolev τ → NaN line flux. Replace non-finite
    # zones with the SIGNED homologous gradient v/r (= 1/t_exp). (velocity_grad)
    dvdr_t = np.where(np.isfinite(dvdr_t), dvdr_t,
                      v_t / np.maximum(np.abs(r_t), 1e-30))

    # destruction probability per line event, per zone
    if eps_dest is None:
        eps_t = np.full_like(r_t, 0.558)   # Hα default (thin Lyβ)
    elif np.ndim(eps_dest) == 0:
        eps_t = np.full_like(r_t, float(eps_dest))
    else:
        eps_t = np.asarray(eps_dest, dtype=np.float64)[i0:]
    eps_t = np.clip(eps_t, 0.0, 1.0)

    # ---- Photospheric disk for occlusion ----
    # The physical photosphere sits at τ_es ≈ 1, NOT at the τ_es_trim
    # boundary used for transport efficiency. The two are different
    # numerical concepts: tau_es_trim drops zones where transport would
    # be hopelessly slow; R_phot is what an external observer sees as
    # "the surface" and what blocks back-side wind from reaching them.
    above_tau1 = tau_es <= 1.0
    if np.any(above_tau1):
        i_phot = int(np.argmax(above_tau1))
        R_phot_disk = float(r[i_phot])
    else:
        R_phot_disk = float(r_t[0])
    # Use max() to ensure r_abs_inner >= grid inner edge (otherwise the
    # occlusion test references a radius outside our grid, which is fine
    # for geometry but inconsistent with where transport happens).
    r_abs_inner = max(R_phot_disk, r_t[0])

    # Continuum source emission depth.
    # See deep_continuum_source kwarg docstring.
    if deep_continuum_source:
        r_src_inner = r_t[0]   # innermost kept zone (tau_es ≈ tau_es_trim)
    else:
        r_src_inner = r_abs_inner   # photosphere surface (τ_es=1)
    # Absorption boundary stays at R_phot (τ_es=1) for the occlusion
    # disk — this controls geometric blockage, NOT where continuum emission
    # originates.
    r_outer     = r_t[-1]
    step_dr     = step_dr_frac * (r_outer - r_abs_inner)

    lam_lo, lam_hi = wavelength_range_AA
    lam_edges = np.linspace(lam_lo, lam_hi, nbins + 1) * 1e-8
    src_lo = (lam_lo - source_padding_AA) * 1e-8
    src_hi = (lam_hi + source_padding_AA) * 1e-8

    mode_map = {'continuum': 0, 'volumetric': 1, 'mixed': 2}
    if emit_mode not in mode_map:
        raise ValueError(f"emit_mode must be one of {list(mode_map.keys())}")
    mode_i = mode_map[emit_mode]

    if mode_i == 0:
        emis_cdf = np.ones(len(r_t), dtype=np.float64)
    else:
        if emissivity is None:
            dr_ = np.empty_like(r_t); dr_[:-1] = np.diff(r_t); dr_[-1] = dr_[-2]
            emis = np.maximum(nu_t, 0.0) * 4.0 * np.pi * r_t**2 * dr_
        else:
            emis = np.asarray(emissivity, dtype=np.float64)[i0:]
            emis = np.maximum(emis, 0.0)
        tot = emis.sum()
        if tot <= 0:
            raise ValueError("total emissivity is zero")
        emis_cdf = np.cumsum(emis) / tot

    np.random.seed(seed)
    # Translate redistribution_mode string to int for the JIT kernel.
    # 0 = semi-PRD, 1 = CRD, 2 = AA-PRD.
    _rdist_map = {'semi_prd': 0, 'crd': 1, 'aa_prd': 2}
    if line_scatter_redistribution not in _rdist_map:
        raise ValueError(
            f"line_scatter_redistribution must be one of "
            f"{list(_rdist_map.keys())}; got {line_scatter_redistribution!r}")
    redistribution_mode_i = _rdist_map[line_scatter_redistribution]

    # Translate photosphere_line_boundary string to int.
    # 0 = TRANSMIT (current default; line packets cross into photosphere and
    #     undergo Thomson scattering inside, may eventually re-emerge, usually
    #     smeared in frequency → contribute to broad Thomson wings).
    # 1 = ABSORB  (CMFGEN-like occultation; line packets crossing the
    #     photospheric boundary from outside are destroyed, modeling line
    #     absorption by the opaque photosphere at Hα wavelength where
    #     χ_line ≈ 2 × χ_es at line center).
    _photbdy_map = {'transmit': 0, 'absorb': 1}
    if photosphere_line_boundary not in _photbdy_map:
        raise ValueError(
            f"photosphere_line_boundary must be one of "
            f"{list(_photbdy_map.keys())}; got {photosphere_line_boundary!r}")
    phot_line_boundary_i = _photbdy_map[photosphere_line_boundary]

    if verbose:
        if v_turb_kms_grid is None:
            vt_desc = f"{v_turb_kms} km/s (scalar)"
        else:
            vt_min = float(np.min(v_turb_kms_grid))
            vt_max = float(np.max(v_turb_kms_grid))
            vt_desc = f"per-zone, min={vt_min:.0f}, max={vt_max:.0f} km/s"
        print(f"[voigt] zones {len(r_t)}/{len(r)}, v_turb={vt_desc}, "
              f"emit={emit_mode}, {n_packets} packets, "
              f"redistrib={line_scatter_redistribution}, "
              f"phot_bdy={photosphere_line_boundary}")

    # Build per-zone Doppler width v_D = sqrt(v_th² + v_turb²) on the
    # trimmed grid the kernel uses. If v_turb_kms_grid is given, use it
    # (after trimming); otherwise fall back to scalar v_turb_kms.
    v_th_t = np.sqrt(KB * T_t / MH)
    if v_turb_kms_grid is not None:
        v_turb_arr = np.asarray(v_turb_kms_grid, dtype=np.float64) * 1e5
        if len(v_turb_arr) != len(r):
            raise ValueError(
                f"v_turb_kms_grid length {len(v_turb_arr)} != n_zones {len(r)}")
        v_turb_t = v_turb_arr[i0:]
    else:
        v_turb_t = np.full_like(r_t, v_turb_kms * 1e5)
    v_D_grid_cms_t = np.sqrt(v_th_t * v_th_t + v_turb_t * v_turb_t)

    import time
    t0 = time.time()
    spectrum, n_line, n_thom, n_esc, n_abs, n_scat_per_zone = _run_voigt(
        n_packets, step_dr, max_events,
        r_src_inner, r_abs_inner, r_outer, float(r_t[0]),
        r_t, v_t, dvdr_t, ne_t, T_t, nl_t, nu_t,
        line.nu0, line.f_lu, line.lam0_cm, float(line.g_l), float(line.g_u),
        v_D_grid_cms_t,
        src_lo, src_hi, lam_edges,
        emis_cdf, mode_i, frac_volumetric,
        eps_t,
        float(mu_obs_min),
        redistribution_mode_i,
        phot_line_boundary_i,
    )
    dt = time.time() - t0

    lam_c = 0.5 * (lam_edges[:-1] + lam_edges[1:])
    dlam  = np.diff(lam_edges)
    f_raw = spectrum / dlam            # photon counts per Å
    if normalize:
        if mode_i == 1:
            F_norm = f_raw / max(f_raw.max(), 1e-300)
        else:
            lam0 = line.lam0_cm
            v_mask = 4000.0 / 3e5
            mask_cont = (lam_c < lam0 * (1 - v_mask)) | (lam_c > lam0 * (1 + v_mask))
            if np.any(mask_cont) and np.any(f_raw[mask_cont] > 0):
                c0 = np.median(f_raw[mask_cont])
            else:
                c0 = np.mean(f_raw[f_raw > 0]) if np.any(f_raw > 0) else 1.0
            F_norm = f_raw / c0
        out_spec = F_norm
    else:
        out_spec = f_raw                # raw photons/Å, caller normalizes
    lam_AA = lam_c * 1e8

    # Map the kept-zone scatter counts back onto the full zone grid for
    # consumers. The MC only transports through the kept zones (after
    # tau_es_trim), so we need to report which radii the counts correspond
    # to. `r_zones_kept_cm` gives the exact radii (in cm) of the kept zones.
    stats = dict(n_line=n_line, n_thomson=n_thom, n_escape=n_esc,
                 n_absorbed=n_abs, wall_time_s=dt, n_zones_kept=len(r_t),
                 n_scat_per_zone=n_scat_per_zone,
                 r_zones_kept_cm=r_t.copy())
    if verbose:
        print(f"[voigt] done in {dt:.2f}s  line={n_line} thom={n_thom} esc={n_esc}")
    return lam_AA, out_spec, stats


# ===========================================================================
# PEELING-OFF / NEXT-EVENT ESTIMATOR (EXPERIMENTAL)
# ===========================================================================
#
# Standard MC variance reduction (Yusef-Zadeh+ 1984, Whitney 2011, Sedona,
# TARDIS): instead of waiting for packets to randomly walk into the observer
# cone and counting them, every emission and every scatter event contributes
# a "peeled" virtual photon to the spectrum at the observer-frame frequency,
# weighted by:
#   - phase function value in observer direction (isotropic = 1/4π)
#   - exp(-tau_to_obs) where tau is integrated along +z from event point
#     to wind exit
#
# Replaces the cone-collection final-escape mechanism. A given packet still
# random-walks through the wind; the spectrum is built from accumulated
# peelings at all its events, not from its final escape direction.
# ===========================================================================

@njit(cache=True)
def _peel_to_observer(px, py, pz, nu_co_at_event, nu0, f_lu, g_l_over_g_u,
                      sigma_pref_, v_D_grid_cms_, r_grid_, v_grid_,
                      T_grid_, ne_grid_, nl_grid_, nuu_grid_,
                      r_outer_, n_steps):
    """Compute the observer-frame frequency and tau-to-observer for a virtual
    photon emerging in +z direction from position (px,py,pz) with comoving
    frequency nu_co_at_event at the event site.

    Returns: (lam_obs_AA, weight_factor) where weight_factor = exp(-tau).
    The (1/4π) isotropic phase factor is NOT applied here — it's the caller's
    responsibility (one factor of 1/(4π) per peel event for isotropic
    emission/scattering).

    Integration: walks along +z from current pz to z = sqrt(r_outer²-rho²)
    where rho = sqrt(px²+py²) is the impact parameter. n_steps is the number
    of equal-step subintervals along this path.
    """
    rho2 = px*px + py*py
    if rho2 > r_outer_ * r_outer_:
        # Already outside wind in projection — just record it
        return 0.0, 0.0
    z_exit = np.sqrt(r_outer_*r_outer_ - rho2)
    if pz >= z_exit:
        # Already past observer-side surface, peel out with no opacity
        # Doppler shift from comoving at this point to observer frame uses
        # gas v at this point projected on +z
        rxx0 = np.sqrt(px*px + py*py + pz*pz)
        if rxx0 > 0:
            vr_here = _lin_interp(rxx0, r_grid_, v_grid_)
            mu_here = pz / rxx0
        else:
            vr_here = 0.0
            mu_here = 0.0
        nu_obs = nu_co_at_event / max(1.0 - vr_here * mu_here / C, 1e-30)
        lam_cm = C / nu_obs
        return lam_cm, 1.0

    # Observer-frame frequency: shift nu_co at current position to static
    # frame for direction +z. The "static frame" here is the observer frame
    # (parallel-rays approximation, observer at +z infinity).
    rxx0 = np.sqrt(px*px + py*py + pz*pz)
    if rxx0 > 0:
        vr_here = _lin_interp(rxx0, r_grid_, v_grid_)
        mu_here = pz / rxx0
    else:
        vr_here = 0.0
        mu_here = 0.0
    nu_obs = nu_co_at_event / max(1.0 - vr_here * mu_here / C, 1e-30)

    # Integrate tau_es + tau_line from (px, py, pz) to (px, py, z_exit)
    # along +z direction.
    #
    # The Thomson term (kappa_es) is smooth and is accumulated at the step
    # midpoint. The LINE term is a near-delta-function in frequency whose
    # peak height scales as 1/v_D, so sampling it at a fixed set of step
    # midpoints systematically MISSES the resonance once the physical
    # Doppler width (~10-30 km/s) is narrower than the step's velocity
    # span — collapsing the line optical depth toward zero and destroying
    # the absorption. To make the result independent of step count and of
    # the (physical) line width, the line optical depth across each step is
    # integrated ANALYTICALLY:
    #
    #   tau_line(step) = ∫ sigma_pref · phi(nu_co) · pop ds
    #                  = sigma_pref · pop · [∫ phi dnu_co] / |dnu_co/ds|
    #
    # With nu_co(s) linear across the step, ∫ phi dnu_co between the step
    # endpoints is exactly 0.5·[erf(x1/√2) − erf(x0/√2)], x = (nu_co−nu0)/σ_D.
    # When a single step straddles the full resonance this returns the exact
    # Sobolev optical depth sigma_pref·pop/|dnu_co/ds| regardless of ds; in
    # the slowly-varying limit (x1→x0) it reduces to the local Gaussian
    # value φ(x)·ds. This is the standard expanding-atmosphere (Sobolev)
    # treatment and carries NO free parameter, floor, or width tuning.
    L_path = z_exit - pz
    if L_path <= 0:
        return (C / nu_obs), 1.0
    ds = L_path / n_steps
    SQRT2 = 1.4142135623730951
    SQRT2PI = 2.5066282746310002
    tau = 0.0
    # Comoving frequency at the ray's starting endpoint (z = pz) equals the
    # event's emitted comoving frequency by construction of nu_obs above.
    nu_co_prev = nu_co_at_event
    for k in range(n_steps):
        z1 = pz + (k + 1) * ds
        r1 = np.sqrt(px*px + py*py + z1*z1)
        if r1 >= r_outer_:
            break
        # endpoint comoving frequency along the +z ray
        vr1 = _lin_interp(r1, r_grid_, v_grid_)
        mu1 = z1 / max(r1, 1e-30)
        nu_co_curr = nu_obs * (1.0 - vr1 * mu1 / C)
        # midpoint quantities (slowly varying): population, n_e, Doppler width
        zm = pz + (k + 0.5) * ds
        rm = np.sqrt(px*px + py*py + zm*zm)
        ne_m = _lin_interp(rm, r_grid_, ne_grid_)
        nl_m = _lin_interp(rm, r_grid_, nl_grid_)
        nu_m = _lin_interp(rm, r_grid_, nuu_grid_)
        v_D_m = _lin_interp(rm, r_grid_, v_D_grid_cms_)
        sigma_D_Hz_m = nu0 * v_D_m / C
        if sigma_D_Hz_m < 1e-30:
            sigma_D_Hz_m = 1e-30
        # dimensionless frequency offsets at the two endpoints (common width)
        x0 = (nu_co_prev - nu0) / sigma_D_Hz_m
        x1 = (nu_co_curr - nu0) / sigma_D_Hz_m
        # analytic line optical depth across the step
        pop_m = nl_m - g_l_over_g_u * nu_m
        if pop_m < 0.0:
            pop_m = 0.0
        dx = x1 - x0
        erf_area = 0.5 * (math.erf(x1 / SQRT2) - math.erf(x0 / SQRT2))
        if abs(dx) < 1e-6:
            # derivative limit: 0.5·d/dx[erf(x/√2)] = exp(-x²/2)/√(2π)
            xm = 0.5 * (x0 + x1)
            ratio = np.exp(-0.5 * xm * xm) / SQRT2PI
        else:
            ratio = erf_area / dx          # = ∫phi ds · sigma_D / ds  (≥ 0)
        if ratio < 0.0:
            ratio = -ratio
        dtau_line = sigma_pref_ * pop_m * ratio * ds / sigma_D_Hz_m
        dtau_es = SIGMAT * ne_m * ds
        tau += dtau_line + dtau_es
        if tau > 30.0:  # save work — exp(-30) is effectively zero
            return (C / nu_obs), 0.0
        nu_co_prev = nu_co_curr
    weight = np.exp(-tau)
    return (C / nu_obs), weight


@njit(cache=True, inline='always')
def _peel_and_bin(px, py, pz, nu_co_at_event, packet_weight,
                  spectrum_, lam_edges_,
                  nu0_, f_lu_, g_l_over_g_u_, sigma_pref_, v_D_grid_cms_,
                  r_grid_, v_grid_, T_grid_, ne_grid_, nl_grid_, nuu_grid_,
                  r_outer_, n_steps_, isotropic_factor):
    """Apply peel-off at this event point: compute lam_obs and weight,
    add to the appropriate spectrum bin. isotropic_factor = 1/(4*pi)
    for isotropic emission/scattering, OR a different phase function.
    """
    lam_obs, w = _peel_to_observer(
        px, py, pz, nu_co_at_event, nu0_, f_lu_, g_l_over_g_u_,
        sigma_pref_, v_D_grid_cms_, r_grid_, v_grid_,
        T_grid_, ne_grid_, nl_grid_, nuu_grid_,
        r_outer_, n_steps_)
    if w <= 0.0:
        return
    nbins_ = lam_edges_.shape[0] - 1
    idx = np.searchsorted(lam_edges_, lam_obs) - 1
    if idx >= 0 and idx < nbins_:
        spectrum_[idx] += packet_weight * w * isotropic_factor


@njit(cache=True)
def _run_voigt_peel(
    n_packets, step_dr, max_events,
    r_src_inner, r_abs_inner, r_outer, r_vol_inner,
    r_grid, v_grid, dvdr_grid, ne_grid, T_grid, nl_grid, nuu_grid,
    nu0, f_lu, lam0, g_l, g_u,
    v_D_grid_cms,
    lam_lo_src, lam_hi_src,
    lam_edges,
    emissivity_cdf, emit_mode, frac_volumetric,
    eps_dest_grid,
    redistribution_mode,
    phot_line_boundary,
    n_steps_peel,
    packet_weight_cont, packet_weight_vol,
    cont_line_emission_int,
):
    """Peel-off variant of _run_voigt. At each emission and scatter event,
    compute virtual peel-off contribution to spectrum. Final escape direction
    is NOT counted (the cone-collection mechanism is replaced).

    cont_line_emission_int: when 1 (default), cont packets contribute peel
    emission events at line-scatter (current behavior). When 0, cont packets
    still line-scatter (PRD redistribution + direction change) but the
    peel-emission event is skipped. This avoids double-counting line
    emission when the vol channel's case-B emissivity already represents
    the unified line source function (S_line × κ_line × dV).

    Returns same tuple shape as _run_voigt for compatibility.
    """
    nbins = lam_edges.shape[0] - 1
    spectrum = np.zeros(nbins)
    n_zones = r_grid.shape[0]
    n_scat_per_zone = np.zeros(n_zones, dtype=np.int64)
    g_ratio = g_l / g_u
    sigma_pref = np.pi * EE * EE / (ME * C) * f_lu
    inv_4pi = 1.0 / (4.0 * np.pi)

    n_line = 0; n_thom = 0; n_esc = 0; n_abs = 0
    n_peel_events = 0

    for ip in range(n_packets):
        if emit_mode == 0:
            use_vol = False
        elif emit_mode == 1:
            use_vol = True
        else:
            use_vol = (np.random.random() < frac_volumetric)
        if use_vol:
            packet_weight = packet_weight_vol
        else:
            packet_weight = packet_weight_cont

        if not use_vol:
            # Continuum source at inner boundary
            nhx, nhy, nhz = _isotropic_dir()
            r0 = r_src_inner * 1.00001
            px = r0*nhx; py = r0*nhy; pz = r0*nhz
            dx, dy, dz = _outward_hemisphere_dir(nhx, nhy, nhz)
            lam_s = lam_lo_src + (lam_hi_src - lam_lo_src) * np.random.random()
            nu = C / lam_s
            # comoving frame nu_co at emission: photospheric source emits
            # isotropically in source rest frame. Use static frame nu directly
            # since photosphere is at v=0 (or we use nu as if comoving).
            vr_src = _lin_interp(r0, r_grid, v_grid)
            mu_emit = nhx*dx + nhy*dy + nhz*dz
            nu_co_emit = nu * (1.0 - vr_src * mu_emit / C)
            # Peel: emission contribution to observer
            _peel_and_bin(px, py, pz, nu_co_emit, packet_weight,
                          spectrum, lam_edges,
                          nu0, f_lu, g_ratio, sigma_pref, v_D_grid_cms,
                          r_grid, v_grid, T_grid, ne_grid, nl_grid, nuu_grid,
                          r_outer, n_steps_peel, inv_4pi)
            n_peel_events += 1
        else:
            u = np.random.random()
            k_lo = 0; k_hi = emissivity_cdf.shape[0] - 1
            while k_hi - k_lo > 1:
                km = (k_hi + k_lo) >> 1
                if emissivity_cdf[km] < u: k_lo = km
                else: k_hi = km
            r_emit = r_grid[k_hi]
            nhx, nhy, nhz = _isotropic_dir()
            px = r_emit*nhx; py = r_emit*nhy; pz = r_emit*nhz
            dx, dy, dz = _isotropic_dir()
            T_emit = _lin_interp(r_emit, r_grid, T_grid)
            v_D_emit = _lin_interp(r_emit, r_grid, v_D_grid_cms)
            sigma_D_emit = nu0 * v_D_emit / C
            nu_co_emit = nu0 + sigma_D_emit * np.random.standard_normal()
            vr_ = _lin_interp(r_emit, r_grid, v_grid)
            mu_ = nhx*dx + nhy*dy + nhz*dz
            nu = nu_co_emit / max(1.0 - vr_ * mu_ / C, 1e-30)
            # Peel: emission contribution to observer
            _peel_and_bin(px, py, pz, nu_co_emit, packet_weight,
                          spectrum, lam_edges,
                          nu0, f_lu, g_ratio, sigma_pref, v_D_grid_cms,
                          r_grid, v_grid, T_grid, ne_grid, nl_grid, nuu_grid,
                          r_outer, n_steps_peel, inv_4pi)
            n_peel_events += 1

        alive = True
        r_inner_pkt = r_vol_inner if use_vol else r_abs_inner

        for ev in range(max_events):
            if not alive: break
            s_bound, btype = _sphere_exit(px, py, pz, dx, dy, dz,
                                          r_outer, r_inner_pkt)
            # Tau-marching to find next interaction (line + Thomson)
            target_tau = -np.log(max(np.random.random(), 1e-300))
            tau_acc = 0.0
            s_acc = 0.0
            n_steps_inner = int(min(s_bound / step_dr + 1, 5000))
            event = -1   # -1 = no event in this stretch, 0 = line, 1 = Thomson
            ds = step_dr
            for step_i in range(n_steps_inner + 1):
                s_now = min(s_acc + ds, s_bound)
                ds_this = s_now - s_acc
                if ds_this <= 0:
                    break
                s_mid = 0.5 * (s_acc + s_now)
                mx = px + dx * s_mid
                my = py + dy * s_mid
                mz = pz + dz * s_mid
                rm = np.sqrt(mx*mx + my*my + mz*mz)
                if rm < r_grid[0]:
                    s_acc = s_now
                    continue
                if rm > r_grid[-1]:
                    break
                ne_m = _lin_interp(rm, r_grid, ne_grid)
                nl_m = _lin_interp(rm, r_grid, nl_grid)
                nu_m_ = _lin_interp(rm, r_grid, nuu_grid)
                v_D_m = _lin_interp(rm, r_grid, v_D_grid_cms)
                sigma_D_Hz_m = nu0 * v_D_m / C
                if sigma_D_Hz_m < 1e-30:
                    sigma_D_Hz_m = 1e-30
                # Line optical depth across the step integrated ANALYTICALLY
                # (Sobolev / error-function form) so that the total line tau
                # along the path — and therefore where and whether the photon
                # scatters — is independent of step size even when the
                # physical Doppler width (~v_th) is far narrower than the
                # step's velocity span. Identical treatment to _peel_to_observer.
                # Comoving frequency at the two step endpoints along the
                # photon's own direction (mu = r̂·d̂).
                ax0 = px + dx * s_acc; ay0 = py + dy * s_acc; az0 = pz + dz * s_acc
                ax1 = px + dx * s_now; ay1 = py + dy * s_now; az1 = pz + dz * s_now
                r0s = np.sqrt(ax0*ax0 + ay0*ay0 + az0*az0)
                r1s = np.sqrt(ax1*ax1 + ay1*ay1 + az1*az1)
                mu0s = (ax0*dx + ay0*dy + az0*dz) / max(r0s, 1e-30)
                mu1s = (ax1*dx + ay1*dy + az1*dz) / max(r1s, 1e-30)
                vr0s = _lin_interp(r0s, r_grid, v_grid)
                vr1s = _lin_interp(r1s, r_grid, v_grid)
                x0 = (nu * (1.0 - vr0s * mu0s / C) - nu0) / sigma_D_Hz_m
                x1 = (nu * (1.0 - vr1s * mu1s / C) - nu0) / sigma_D_Hz_m
                pop_m = nl_m - g_ratio * nu_m_
                if pop_m < 0:
                    pop_m = 0.0
                dxx = x1 - x0
                erf_area = 0.5 * (math.erf(x1 / 1.4142135623730951)
                                  - math.erf(x0 / 1.4142135623730951))
                if abs(dxx) < 1e-6:
                    xmm = 0.5 * (x0 + x1)
                    ratio = np.exp(-0.5 * xmm * xmm) / 2.5066282746310002
                else:
                    ratio = erf_area / dxx
                if ratio < 0.0:
                    ratio = -ratio
                dtau_line = sigma_pref * pop_m * ratio * ds_this / sigma_D_Hz_m
                dtau_es = SIGMAT * ne_m * ds_this
                dtau = dtau_line + dtau_es
                if tau_acc + dtau >= target_tau:
                    frac = (target_tau - tau_acc) / max(dtau, 1e-30)
                    s_event = s_acc + frac * ds_this
                    p_line = dtau_line / max(dtau, 1e-30)
                    if np.random.random() < p_line:
                        event = 0
                    else:
                        event = 1
                    px = px + s_event * dx
                    py = py + s_event * dy
                    pz = pz + s_event * dz
                    break
                tau_acc += dtau
                s_acc = s_now

            if event < 0:
                # No interaction along path — packet hits boundary
                if btype == 0:
                    # Outer boundary escape — IGNORE (peeling already
                    # captured every contribution)
                    n_esc += 1
                    alive = False
                    break
                else:
                    # Inner boundary
                    r_pre = np.sqrt(px*px + py*py + pz*pz)
                    is_inside = (r_pre < r_abs_inner)
                    if is_inside:
                        px += s_bound*dx; py += s_bound*dy; pz += s_bound*dz
                        nhx_ = px/r_abs_inner; nhy_ = py/r_abs_inner; nhz_ = pz/r_abs_inner
                        px += 1e-6 * r_abs_inner * nhx_
                        py += 1e-6 * r_abs_inner * nhy_
                        pz += 1e-6 * r_abs_inner * nhz_
                        continue
                    if not use_vol:
                        # Continuum packet hitting photosphere from outside —
                        # reflect into outward hemisphere, then it'll re-emit
                        # which is not a new emission so no peel here
                        px += s_bound*dx; py += s_bound*dy; pz += s_bound*dz
                        rr = np.sqrt(px*px + py*py + pz*pz)
                        nhx_ = px/rr; nhy_ = py/rr; nhz_ = pz/rr
                        dx, dy, dz = _outward_hemisphere_dir(nhx_, nhy_, nhz_)
                        px *= 1.0 + 1e-6; py *= 1.0 + 1e-6; pz *= 1.0 + 1e-6
                        continue
                    else:
                        if phot_line_boundary == 1:
                            n_abs += 1
                            alive = False
                            break
                        # TRANSMIT: traverse inner sphere as vacuum (chord)
                        # — see equivalent fix in non-peel kernel above for
                        # rationale. Without this, deep-zone vol packets
                        # whose random walks dip through the symmetry center
                        # end up below the grid floor and break energy
                        # conservation.
                        px += s_bound*dx; py += s_bound*dy; pz += s_bound*dz
                        b_proj = px*dx + py*dy + pz*dz
                        s_chord = -2.0 * b_proj
                        if s_chord < 0.0:
                            s_chord = 0.0
                        px += s_chord * dx
                        py += s_chord * dy
                        pz += s_chord * dz
                        rr = np.sqrt(px*px + py*py + pz*pz)
                        if rr > 0.0:
                            px *= (1.0 + 1e-7)
                            py *= (1.0 + 1e-7)
                            pz *= (1.0 + 1e-7)
                        continue

            # Process the interaction
            if event == 0:
                # Line absorption
                rxx = np.sqrt(px*px + py*py + pz*pz)
                iz = np.searchsorted(r_grid, rxx) - 1
                if iz < 0:
                    iz = 0
                elif iz >= n_zones:
                    iz = n_zones - 1
                n_scat_per_zone[iz] += 1
                eps_here = _lin_interp(rxx, r_grid, eps_dest_grid)
                if np.random.random() < eps_here:
                    n_abs += 1
                    alive = False
                    break
                # Coherent line scatter
                Tx = _lin_interp(rxx, r_grid, T_grid)
                vr_ = _lin_interp(rxx, r_grid, v_grid)
                mu_old = (px*dx + py*dy + pz*dz) / max(rxx, 1e-30)
                nu_co = nu * (1.0 - vr_ * mu_old / C)
                v_D  = _lin_interp(rxx, r_grid, v_D_grid_cms)
                sigma_D = nu0 * v_D / C
                if redistribution_mode == 1:
                    nu_co_new = nu0 + sigma_D * np.random.standard_normal()
                elif redistribution_mode == 2:
                    x_offset = (nu_co - nu0) / max(sigma_D, 1e-30)
                    w_mix = np.exp(-0.5 * x_offset * x_offset)
                    nu_co_blend = (1.0 - w_mix) * nu_co + w_mix * nu0
                    nu_co_new = nu_co_blend + sigma_D * np.random.standard_normal()
                else:
                    nu_co_new = nu_co + sigma_D * np.random.standard_normal()
                # PEEL after redistribution: virtual photon emerges in +z
                # with comoving freq nu_co_new.
                # Skip the peel emission when this is a cont packet and
                # cont_line_emission is disabled — avoids double-counting
                # of line emission already represented by vol channel.
                if use_vol or cont_line_emission_int == 1:
                    _peel_and_bin(px, py, pz, nu_co_new, packet_weight,
                                  spectrum, lam_edges,
                                  nu0, f_lu, g_ratio, sigma_pref, v_D_grid_cms,
                                  r_grid, v_grid, T_grid, ne_grid, nl_grid, nuu_grid,
                                  r_outer, n_steps_peel, inv_4pi)
                    n_peel_events += 1
                # Continue random walk with new direction
                dx, dy, dz = _isotropic_dir()
                mu_ = (px*dx + py*dy + pz*dz) / max(rxx, 1e-30)
                nu = nu_co_new / max(1.0 - vr_ * mu_ / C, 1e-30)
                n_line += 1
            else:
                # Thomson scatter
                rxx = np.sqrt(px*px + py*py + pz*pz)
                T_ = _lin_interp(rxx, r_grid, T_grid)
                vr_ = _lin_interp(rxx, r_grid, v_grid)
                mu_old = (px*dx + py*dy + pz*dz) / max(rxx, 1e-30)
                nu_co = nu * (1.0 - vr_ * mu_old / C)
                sigma = nu_co * np.sqrt(2.0 * KB * T_ / (ME * C * C))
                nu_co_new = nu_co + sigma * np.random.standard_normal()
                # PEEL after thermal kick
                _peel_and_bin(px, py, pz, nu_co_new, packet_weight,
                              spectrum, lam_edges,
                              nu0, f_lu, g_ratio, sigma_pref, v_D_grid_cms,
                              r_grid, v_grid, T_grid, ne_grid, nl_grid, nuu_grid,
                              r_outer, n_steps_peel, inv_4pi)
                n_peel_events += 1
                dx, dy, dz = _isotropic_dir()
                mu_new = (px*dx + py*dy + pz*dz) / max(rxx, 1e-30)
                nu = nu_co_new / max(1.0 - vr_ * mu_new / C, 1e-30)
                n_thom += 1

    return spectrum, n_line, n_thom, n_esc, n_abs, n_scat_per_zone


def run_mc_voigt_peel(
    r, v, rho, T, n_e, n_lower, n_upper, line,
    v_turb_kms=50.0,
    v_turb_kms_grid=None,
    n_packets=100_000, step_dr_frac=3e-3,
    wavelength_range_AA=(6200.0, 6900.0),
    source_padding_AA=1500.0, nbins=250, max_events=2000,
    tau_es_trim=None,
    emit_mode='continuum', emissivity=None, frac_volumetric=0.5,
    eps_dest=None,
    line_scatter_redistribution='aa_prd',
    photosphere_line_boundary='transmit',
    n_steps_peel=80,
    packet_weight_cont=1.0, packet_weight_vol=1.0,
    cont_line_emission=True,
    seed=None, verbose=True,
):
    """Peel-off / next-event-estimator Monte Carlo wrapper.

    Drop-in for run_mc_voigt with mu_obs_min effectively = 0 (peel
    captures all directions). The n_steps_peel parameter controls the
    accuracy of tau-to-observer integration (higher = more accurate but
    slower).

    Returns same triple as run_mc_voigt: lam_AA, F (spectrum), stats.
    """
    import time
    t0 = time.time()
    if seed is not None:
        np.random.seed(int(seed))

    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    n_e = np.asarray(n_e, dtype=np.float64)
    n_lower = np.asarray(n_lower, dtype=np.float64)
    n_upper = np.asarray(n_upper, dtype=np.float64)

    # Trim by tau_es_trim if specified
    if tau_es_trim is not None:
        kappa_es = SIGMAT * n_e
        # Build cumulative tau from outside inward
        cum_tau = np.zeros_like(r)
        for i in range(len(r) - 2, -1, -1):
            cum_tau[i] = cum_tau[i+1] + kappa_es[i] * (r[i+1] - r[i])
        i0 = 0
        for i in range(len(r)):
            if cum_tau[i] <= tau_es_trim:
                i0 = i
                break
    else:
        i0 = 0

    r_t = r[i0:]
    v_t = v[i0:]
    T_t = T[i0:]
    ne_t = n_e[i0:]
    nl_t = n_lower[i0:]
    nu_t = n_upper[i0:]
    with np.errstate(divide='ignore', invalid='ignore'):
        dvdr_t = np.gradient(v_t, r_t)
    # robust to DUPLICATE STELLA shock radii: a raw gradient is NaN there (zero
    # spacing) and poisons the Sobolev τ → NaN line flux. Replace non-finite
    # zones with the SIGNED homologous gradient v/r (= 1/t_exp). (velocity_grad)
    dvdr_t = np.where(np.isfinite(dvdr_t), dvdr_t,
                      v_t / np.maximum(np.abs(r_t), 1e-30))

    # Boundaries
    r_src_inner = r_t[0]
    r_abs_inner = r_t[0]
    r_outer = r_t[-1]
    r_vol_inner = r_t[0]

    lam_lo_AA, lam_hi_AA = wavelength_range_AA
    lam_lo_src = (lam_lo_AA - source_padding_AA) * 1e-8
    lam_hi_src = (lam_hi_AA + source_padding_AA) * 1e-8
    lam_edges = np.linspace(lam_lo_AA * 1e-8, lam_hi_AA * 1e-8, nbins + 1)

    # Emissivity CDF
    if emissivity is None:
        emis = np.zeros(len(r_t))
    else:
        emis = np.asarray(emissivity, dtype=np.float64)[i0:]
    weights = np.zeros(len(r_t))
    for k in range(len(r_t)):
        r_lo = r_t[0] if k == 0 else 0.5 * (r_t[k-1] + r_t[k])
        r_hi = r_t[-1] if k == len(r_t)-1 else 0.5 * (r_t[k] + r_t[k+1])
        dV = (4.0/3.0) * np.pi * (r_hi**3 - r_lo**3)
        weights[k] = max(emis[k], 0.0) * dV
    cdf = np.cumsum(weights)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    else:
        cdf = np.linspace(0, 1, len(r_t))

    # Mode dispatch
    mode_map = {'continuum': 0, 'volumetric': 1, 'mixed': 2}
    mode_i = mode_map[emit_mode]

    rdist_map = {'semi_prd': 0, 'crd': 1, 'aa_prd': 2}
    redistribution_mode_i = rdist_map[line_scatter_redistribution]

    pbdy_map = {'transmit': 0, 'absorb': 1}
    phot_line_boundary_i = pbdy_map[photosphere_line_boundary]

    # eps_dest_grid
    if eps_dest is None:
        eps_grid = np.zeros(len(r_t))
    elif np.isscalar(eps_dest):
        eps_grid = np.full(len(r_t), float(eps_dest))
    else:
        eps_grid = np.asarray(eps_dest, dtype=np.float64)[i0:]

    step_dr = step_dr_frac * r_outer

    # Build per-zone Doppler width v_D = sqrt(v_th² + v_turb²) on trimmed grid.
    v_th_t = np.sqrt(KB * T_t / MH)
    if v_turb_kms_grid is not None:
        v_turb_arr = np.asarray(v_turb_kms_grid, dtype=np.float64) * 1e5
        if len(v_turb_arr) != len(r):
            raise ValueError(
                f"v_turb_kms_grid length {len(v_turb_arr)} != n_zones {len(r)}")
        v_turb_t = v_turb_arr[i0:]
    else:
        v_turb_t = np.full_like(r_t, v_turb_kms * 1e5)
    v_D_grid_cms_t = np.sqrt(v_th_t * v_th_t + v_turb_t * v_turb_t)

    if verbose:
        if v_turb_kms_grid is None:
            vt_desc = f"{v_turb_kms} km/s (scalar)"
        else:
            vt_min = float(np.min(v_turb_kms_grid))
            vt_max = float(np.max(v_turb_kms_grid))
            vt_med = float(np.median(v_turb_kms_grid))
            vt_desc = (f"per-zone, min={vt_min:.0f}, "
                       f"med={vt_med:.0f}, max={vt_max:.0f} km/s")
        print(f"[peel] zones {len(r_t)}/{len(r)}, v_turb={vt_desc}, "
              f"emit={emit_mode}, {n_packets} packets, "
              f"redistrib={line_scatter_redistribution}, "
              f"n_steps_peel={n_steps_peel}")

    spectrum, n_line, n_thom, n_esc, n_abs, n_scat_per_zone = _run_voigt_peel(
        n_packets, step_dr, max_events,
        r_src_inner, r_abs_inner, r_outer, r_vol_inner,
        r_t, v_t, dvdr_t, ne_t, T_t, nl_t, nu_t,
        line.nu0, line.f_lu, line.lam0_cm, float(line.g_l), float(line.g_u),
        v_D_grid_cms_t,
        lam_lo_src, lam_hi_src,
        lam_edges,
        cdf, mode_i, frac_volumetric,
        eps_grid,
        redistribution_mode_i,
        phot_line_boundary_i,
        int(n_steps_peel),
        float(packet_weight_cont), float(packet_weight_vol),
        int(1 if cont_line_emission else 0),
    )
    dt = time.time() - t0

    # Output: lam in Angstroms (centers), F = spectrum (raw, un-normalized;
    # caller is responsible for far-wing continuum normalization)
    lam_centers_cm = 0.5 * (lam_edges[1:] + lam_edges[:-1])
    lam_AA = lam_centers_cm * 1e8
    out_spec = spectrum.copy()

    stats = dict(n_line=n_line, n_thomson=n_thom, n_escape=n_esc,
                 n_absorbed=n_abs, wall_time_s=dt, n_zones_kept=len(r_t),
                 n_scat_per_zone=n_scat_per_zone,
                 r_zones_kept_cm=r_t.copy())
    if verbose:
        print(f"[peel] done in {dt:.2f}s  line={n_line} thom={n_thom} esc={n_esc}")
    return lam_AA, out_spec, stats
