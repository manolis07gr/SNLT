"""
metal_ionization.py — Photoionization-equilibrium ionization for C / O / Ne
================================================================================

FUTURE_WORK P2 item 5 (Stage 2). Computes per-zone ion-stage number densities
for carbon, oxygen and neon from PHOTOIONIZATION equilibrium (not Saha), driven
by the same radiation field the H/He pipeline uses:

  • a diluted blackbody at the photospheric temperature T_phot (geometric
    dilution W(r)), and
  • the shock-bremsstrahlung X-ray component (L_X_brems, T_shock).

For each non-top ion stage i the photoionization rate per atom is

    Γ_i(r) = W(r) · [ 4π·G_BB,i  +  (L_X_brems / (π R_phot²))·G_brems,i ]      (s⁻¹)
    G_BB,i    = ∫_{ν_i}^∞ (B_ν(T_phot) / hν) · σ_i(ν) dν
    G_brems,i = ∫_{ν_i}^∞ (φ_brems,ν(T_shock) / hν) · σ_i(ν) dν

with a hydrogenic near-threshold cross-section σ_i(ν) = σ₀,i (ν_i/ν)^{s_i}. This
mirrors the rate structure in photoionize_csm.py (its `_G_function` for BB and
`_G_function_brems` for the shock X-rays) but is SELF-CONTAINED — it reuses only
the radiation-field PARAMETERS, never the validated H solver code, so nothing in
the H/He path is touched.

The metal ion fractions then come from the photoionization-balance ion ladder
(metal_atoms.ion_ladder_fractions): n_{i+1}/n_i = Γ_i / (n_e α_{i+1}(T)). Metals
are TRACE, so this uses the pipeline's electron density n_e and does NOT re-solve
charge neutrality.

v1 APPROXIMATIONS (documented, to refine later):
  • Optically thin to the ionizing radiation (τ_ν ≈ 0). Valid in the metal-rich,
    electron-scattering-thin C-series CSM (τ_es ~ 0.03); for dense H-rich CSM the
    attenuation would need the per-zone τ from photoionize_csm.
  • Hydrogenic power-law cross-sections (PROVISIONAL σ₀, s in metal_atoms).
"""
from __future__ import annotations
import numpy as np
import metal_atoms as ma

H_PL = 6.62607015e-27
C = 2.99792458e10
KB = 1.380649e-16
EV = 1.602176634e-12


def _planck_nu(nu, T):
    x = np.minimum(H_PL * nu / (KB * max(float(T), 1.0)), 700.0)
    return (2.0 * H_PL * nu ** 3 / C ** 2) / np.expm1(x)


def dilution_W(r, R_phot):
    """Geometric dilution W(r)=½[1−√(1−(R_phot/r)²)] (= ½ at R_phot)."""
    r = np.asarray(r, float)
    x = np.clip((float(R_phot) / np.maximum(r, 1e-30)) ** 2, 0.0, 1.0)
    return 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - x, 0.0)))


def _xsec(ion_label, nu):
    """Photoionization cross-section σ(ν) [cm²] for an ion: σ₀ (ν_th/ν)^s above
    threshold, 0 below."""
    d = ma.ION_DATA[ion_label]
    nu_th = d['chi_eV'] * EV / H_PL
    nu = np.asarray(nu, float)
    return np.where(nu >= nu_th, d['sigma0'] * (nu_th / np.maximum(nu, nu_th))
                    ** d['s_xsec'], 0.0)


def _freq_grid(nu_th, nu_max, n=400):
    """Log-spaced frequency grid from threshold to nu_max."""
    return np.logspace(np.log10(nu_th), np.log10(max(nu_max, nu_th * 1.01)), n)


def gamma_unit_rates(ion_label, T_phot, L_X_brems, T_shock, R_phot):
    """Per-unit-W photoionization rate of `ion_label` from the BB + brems field.

    Returns (gamma_unit) such that Γ_i(r) = W(r) · gamma_unit  [s⁻¹]. The full
    rate combines the diluted BB and the shock-bremsstrahlung contributions.
    Integrals are done by trapezoid over a log-frequency grid (shared per ion).
    """
    d = ma.ION_DATA[ion_label]
    nu_th = d['chi_eV'] * EV / H_PL

    # --- diluted BB term:  4π ∫ (B_ν/hν) σ dν ---
    T_phot = max(float(T_phot), 1.0)
    nu_max_bb = nu_th * 60.0 + 60.0 * KB * T_phot / H_PL
    nu_bb = _freq_grid(nu_th, nu_max_bb)
    integ_bb = (_planck_nu(nu_bb, T_phot) / (H_PL * nu_bb)) * _xsec(ion_label, nu_bb)
    G_BB = float(np.trapezoid(integ_bb, nu_bb)) if hasattr(np, 'trapezoid') \
        else float(np.trapz(integ_bb, nu_bb))
    gamma_bb = 4.0 * np.pi * G_BB

    # --- shock-bremsstrahlung term: (L_X/(π R²)) ∫ (φ_ν/hν) σ dν ---
    gamma_brems = 0.0
    if L_X_brems and L_X_brems > 0 and T_shock and T_shock > 0:
        kT = KB * float(T_shock)
        # normalized thermal-brems shape φ_ν = (h/kT) exp(-hν/kT)  (g_ff ≈ 1)
        nu_max_br = nu_th + 60.0 * kT / H_PL
        nu_br = _freq_grid(nu_th, nu_max_br)
        phi = (H_PL / kT) * np.exp(-np.minimum(H_PL * nu_br / kT, 700.0))
        integ_br = (phi / (H_PL * nu_br)) * _xsec(ion_label, nu_br)
        G_br = float(np.trapezoid(integ_br, nu_br)) if hasattr(np, 'trapezoid') \
            else float(np.trapz(integ_br, nu_br))
        gamma_brems = (float(L_X_brems) / (np.pi * float(R_phot) ** 2)) * G_br

    return gamma_bb + gamma_brems


def compute_metal_ionization(T, n_e, r, rho, R_phot, T_phot,
                             X_C=None, X_O=None, X_Ne=None,
                             L_X_brems=0.0, T_shock=0.0, verbose=False):
    """Per-zone metal ion densities from photoionization equilibrium.

    Parameters (all per-zone arrays except the scalars R_phot/T_phot/L_X_brems/
    T_shock): T, n_e, r, rho; mass fractions X_C/X_O/X_Ne (None → element skipped).

    Returns dict {ion_label: n_ion array [cm⁻³]} for every stage of every element
    with a supplied mass fraction (e.g. 'C_III', 'O_I', 'Ne_III', ...), plus
    {'_frac': {ion_label: fractional population}} for diagnostics.
    """
    T = np.asarray(T, float)
    n_e = np.asarray(n_e, float)
    W = dilution_W(r, R_phot)
    nz = T.shape[0]
    out = {}
    frac_all = {}
    elements = [('C', X_C), ('O', X_O), ('Ne', X_Ne)]
    for elem, X in elements:
        if X is None:
            continue
        X = np.asarray(X, float)
        # ionization rate Γ out of each non-top stage
        stages = sorted((k for k, v in ma.ION_DATA.items() if v['elem'] == elem),
                        key=lambda k: ma.ION_DATA[k]['stage'])
        gamma_by_ion = {}
        for ion in stages[:-1]:                      # every stage except the top
            g_unit = gamma_unit_rates(ion, T_phot, L_X_brems, T_shock, R_phot)
            gamma_by_ion[ion] = W * g_unit           # per-zone Γ_i(r)
        fracs = ma.ion_ladder_fractions(elem, gamma_by_ion, T, n_e)
        # --- SELF-SHIELDING for the C³⁺→C⁴⁺ channel (item 3). The simplified
        # trace ladder uses the UNattenuated radiation field; that is the
        # validated status quo for the low stages, but for the new C_V stage it
        # over-ionizes badly: the C³⁺ column itself is optically thick at its own
        # 64.5 eV edge (τ_self ≫ 1 through a dense CSM), so C⁴⁺ exists only in a
        # shielded inner skin. Two-pass: first-pass n(C³⁺) → cumulative
        # τ_self(r) = ∫ σ₀(C_IV) n_C3 dr from the photosphere outward → attenuate
        # ONLY Γ(C_IV) and re-solve. All other channels keep the pre-item-3
        # behaviour exactly. (He⁺ shielding would suppress C⁴⁺ further — this is
        # the conservative upper bound on C_V.)
        if elem == 'C' and 'C_V' in fracs and 'C_IV' in gamma_by_ion:
            n_c3 = fracs['C_IV'] * ma.number_density(elem, X, rho)
            rr = np.asarray(r, float)
            dr = np.diff(rr, prepend=rr[0])
            sigma0 = float(ma.ION_DATA['C_IV']['sigma0'])
            tau_self = np.cumsum(sigma0 * n_c3 * np.maximum(dr, 0.0))
            gamma_by_ion['C_IV'] = gamma_by_ion['C_IV'] * np.exp(
                -np.minimum(tau_self, 200.0))
            fracs = ma.ion_ladder_fractions(elem, gamma_by_ion, T, n_e)
            if verbose:
                print(f"[metal_ion] C_V self-shielding: tau_self(64.5eV) "
                      f"in/out = {tau_self[0]:.2g} / {tau_self[-1]:.2g}")
        n_elem = ma.number_density(elem, X, rho)
        for ion, f in fracs.items():
            out[ion] = f * n_elem
            frac_all[ion] = f
        if verbose:
            mean_frac = {ion: float(np.mean(f)) for ion, f in fracs.items()}
            print(f"[metal_ion] {elem}: mean ion fractions {mean_frac}")
    out['_frac'] = frac_all
    return out
