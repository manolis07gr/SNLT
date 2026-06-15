"""
metal_atoms.py — Atomic data + emissivity formulas for C / O / Ne metal lines
================================================================================

FUTURE_WORK P2 item 5 (foundation module). Provides everything the metal-line
emissivity integrals need, with NO dependency on the H/He pipeline so it can be
unit-tested standalone:

  1. Ionization data per ion stage (potential χ, ground-state photoionization
     cross-section σ₀ + a near-threshold power-law index, total recombination
     coefficient α_rec(T)) — drives the photoionization-equilibrium ion ladder.
  2. The metal line list: per line, its emitting ion, rest wavelength, emission
     MECHANISM ('recomb' permitted | 'cel' collisionally-excited/forbidden), and
     the coefficient(s) that mechanism needs.
  3. Emissivity functions (cgs, per unit volume, erg/s/cm³):
       recombination:  j = hν · α_eff(T) · n_e · n_ion
       collisional:    j = hν · q₁₂(T) · n_e · n_ion · branch,
                       q₁₂ = 8.629e-6 · Ω(T)/(g_l √T) · exp(−ΔE/kT)   [low-density]
  4. A photoionization-equilibrium ion-ladder solver: given a per-stage
     ionization rate Γ_i and n_e, T, returns the fractional populations of each
     ion stage (trace-element approximation — uses the supplied n_e, does NOT
     re-solve charge neutrality, since metals are trace).

⚠ PROVISIONAL ATOMIC DATA. The numeric coefficients below are representative
literature/approximate values (Verner+1996 cross-sections; Badnell RR+DR
recombination; CHIANTI/Osterbrock collision strengths & effective recombination)
chosen to get the physics and scalings right. They MUST be verified against
CHIANTI / Cloudy before any absolute metal-line flux is quoted — exactly the
status of the PROVISIONAL He coefficients in formal_line_profile.RECOMB_COEFF.
The FORMULAS and ionization PHYSICS are the rigorous part; the numbers are
placeholders to refine. Cross-sections use a hydrogenic (ν_th/ν)^s power law
near threshold (same approximation class as the H Kramers σ in photoionize_csm).
"""
from __future__ import annotations
import numpy as np

# Physical constants (cgs)
H_PL = 6.62607015e-27
C = 2.99792458e10
KB = 1.380649e-16
EV = 1.602176634e-12          # erg per eV
MH = 1.6726219e-24            # g (proton mass; atomic masses below in units of MH)

# Atomic masses [g] for number-density conversion n = X·rho/m
M_ATOM = {'C': 12.011 * MH, 'O': 15.999 * MH, 'Ne': 20.180 * MH}

# ---------------------------------------------------------------------------
# 1. Ionization data per ion stage
#    key = ion label (roman numeral = stage; 'C_I' is neutral carbon, etc.)
#    chi_eV   : ionization potential of THIS stage (energy to remove an e⁻)
#    sigma0   : ground-state photoionization cross-section at threshold [cm²]
#    s_xsec   : near-threshold power-law index, σ(ν) ≈ σ₀ (ν_th/ν)^s
#    These drive the ion ladder C_I→C_II→C_III→C_IV, O_I→O_II→O_III, etc.
# ---------------------------------------------------------------------------
ION_DATA = {
    # Carbon
    'C_I':   dict(elem='C', stage=1, chi_eV=11.260, sigma0=1.2e-17, s_xsec=2.0),
    'C_II':  dict(elem='C', stage=2, chi_eV=24.383, sigma0=4.6e-18, s_xsec=2.5),
    'C_III': dict(elem='C', stage=3, chi_eV=47.888, sigma0=1.6e-18, s_xsec=2.5),
    'C_IV':  dict(elem='C', stage=4, chi_eV=64.494, sigma0=7.0e-19, s_xsec=3.0),
    # He-like C: ionizing C³⁺→C⁴⁺ takes 392 eV (hard X-rays only — the shock-brems
    # channel). Added (item 3) as the recombination parent of C IV 5801/12; at
    # photospheric ~2e4 K BB alone, C⁴⁺ is negligible, so the ladder shift to the
    # existing C_IV fraction is small except at strongly shock-irradiated epochs.
    'C_V':   dict(elem='C', stage=5, chi_eV=392.09, sigma0=2.0e-19, s_xsec=3.0),
    # Oxygen
    'O_I':   dict(elem='O', stage=1, chi_eV=13.618, sigma0=2.9e-18, s_xsec=2.0),
    'O_II':  dict(elem='O', stage=2, chi_eV=35.121, sigma0=6.0e-18, s_xsec=2.5),
    'O_III': dict(elem='O', stage=3, chi_eV=54.936, sigma0=2.5e-18, s_xsec=2.5),
    # Neon
    'Ne_I':  dict(elem='Ne', stage=1, chi_eV=21.565, sigma0=5.5e-18, s_xsec=2.0),
    'Ne_II': dict(elem='Ne', stage=2, chi_eV=40.963, sigma0=4.0e-18, s_xsec=2.5),
    'Ne_III':dict(elem='Ne', stage=3, chi_eV=63.423, sigma0=2.0e-18, s_xsec=2.5),
}

# Total recombination coefficient α_rec(T) for ION+ → ION (RR+DR), used in the
# ionization balance n_i Γ_i = n_{i+1} n_e α_{i+1}. (A_1e4 [cm³/s], exponent b;
# α = A (T/1e4)^b). PROVISIONAL ~Badnell. Keyed by the RECOMBINING (upper) ion.
RECOMB_TOTAL = {
    'C_II':  (4.7e-13, -0.65), 'C_III': (2.3e-12, -0.65), 'C_IV': (5.0e-12, -0.70),
    'C_V':   (2.3e-11, -0.70),   # He-like C⁴⁺→C³⁺ (RR+DR, provisional Badnell-form)
    'O_II':  (3.4e-13, -0.65), 'O_III': (2.0e-12, -0.65),
    'Ne_II': (2.4e-13, -0.65), 'Ne_III':(1.8e-12, -0.65),
}


def alpha_recomb_total(ion_upper, T):
    """Total recombination coefficient [cm³/s] for ion_upper → next-lower stage."""
    if ion_upper not in RECOMB_TOTAL:
        return np.zeros_like(np.asarray(T, float))
    A, b = RECOMB_TOTAL[ion_upper]
    return A * (np.maximum(np.asarray(T, float), 100.0) / 1.0e4) ** b


# ---------------------------------------------------------------------------
# 2. Metal line list (full C+O+Ne v1 set)
#    mechanism 'recomb' : permitted, j = hν α_eff n_e n_ion
#       alpha_eff = (A_1e4, b)  effective line recombination coefficient [cm³/s]
#    mechanism 'cel'    : forbidden / intercombination, collisionally excited
#       Omega = (O_1e4, p) effective collision strength Ω(T)=O_1e4 (T/1e4)^p
#       g_l   = lower-(ground)-term statistical weight
#       g_u   = upper-level statistical weight
#       dE_eV = excitation energy of the upper level
#       A_ul  = TOTAL radiative decay rate of the upper level [s⁻¹] (sets n_crit)
#       branch= fraction of upper-level decays emerging in THIS line
#       The critical density n_crit = A_ul / q_ul(T) is computed on the fly, and
#       the emissivity carries the (1 + n_e/n_crit)⁻¹ collisional-de-excitation
#       suppression so it is correct at BOTH low and high density.
# ---------------------------------------------------------------------------
METAL_LINES = {
    # --- Carbon ---
    'C_IV_1549':  dict(ion='C_IV',  lam_AA=1549.05, mech='recomb',
                       alpha_eff=(2.0e-13, -0.80)),          # 2s-2p resonance doublet
    'C_III_1909': dict(ion='C_III', lam_AA=1908.73, mech='cel',   # ] intercombination
                       Omega=(1.10, 0.00), g_l=1, g_u=9, dE_eV=6.50,
                       A_ul=120.0, branch=1.0),               # n_crit ~ 5e9 cm⁻³
    # C III λ4650 (V1 multiplet, 2s3p ³P° → 2s3s ³S) — an OPTICAL RECOMBINATION
    # line: emitted by C²⁺ but POPULATED BY RECOMBINATION OF C³⁺ (recomb_parent=
    # 'C_IV'), and cascade-dominated (the ³P° upper term is spin-forbidden from the
    # ¹S ground, so collisional excitation is negligible — this line genuinely
    # scales with n_e·n(C³⁺), not n(C²⁺)). α_eff = branch · α_tot(C³⁺→C²⁺, T): the
    # multiplet branching times the total recombination rate, which bounds it
    # physically and gives the correct T-dependence. branch≈0.02 is a literature-
    # informed estimate (PPB91-consistent, ~factor-2 uncertain) — drop in the exact
    # Pequignot+1991 V1 effective coefficient here to remove that residual.
    # branch tuned so alpha_eff(10^4 K) = recomb_branch * alpha_tot(C_IV) =
    # 0.048 * 5.0e-12 = 2.4e-13 cm^3/s — the C III V1 (lambda 4647-50-51)
    # effective recombination coefficient in the PPB91 / Davey-Storey-Kisielius
    # range (~1.5x the C II 4267 ORL at 1.6e-13, consistent with PN/WC
    # observations). Was 2.0e-2 (1.0e-13, factor ~2 low). NOTE (measured, GO1):
    # this 2.4x correction is necessary but NOT sufficient to match the observed
    # WC-like lambda 4650 feature (real 1.8-2.2x continuum vs our ~1.001x): with
    # C^3+ abundant (0.86) and the full CSM integrated, the optical ORL is still
    # ~250x too faint. Root cause is FORMATION physics, not the coefficient — the
    # parent C^3+ (hot gas) and the n_e^2 boost (cool dense gas) anti-correlate,
    # and the real feature needs full C III/C IV model-atom recombination +
    # low-T dielectronic data (CMFGEN-class). See FUTURE_WORK.
    'C_III_4647': dict(ion='C_III', lam_AA=4647.42, mech='recomb',
                       recomb_parent='C_IV', recomb_branch=4.8e-2),
    # --- Oxygen ---
    'O_I_6300':   dict(ion='O_I',   lam_AA=6300.30, mech='cel',   # [O I] 1D2 → 3P2
                       Omega=(0.27, 0.10), g_l=9, g_u=5, dE_eV=1.967,
                       A_ul=0.0074, branch=0.76),             # n_crit ~ 1.8e6 cm⁻³
    'O_III_5007': dict(ion='O_III', lam_AA=5006.84, mech='cel',   # [O III] 1D2 → 3P2
                       Omega=(2.17, 0.10), g_l=9, g_u=5, dE_eV=2.513,
                       A_ul=0.0246, branch=0.73),             # n_crit ~ 6.8e5 cm⁻³
    # --- Neon ---
    'Ne_III_3869':dict(ion='Ne_III',lam_AA=3868.76, mech='cel',   # [Ne III] 1D2 → 3P2
                       Omega=(1.40, 0.10), g_l=9, g_u=5, dE_eV=3.204,
                       A_ul=0.234, branch=0.74),              # n_crit ~ 9.7e6 cm⁻³
    # --- Stage-2 optical lines (P2 #7): the diagnostic features real Icn show
    # (SN 2019hgp / 2021csp: C III 5696, C II 4267/6580, O I 7773) that the
    # 6-line v1 set lacked. ALL are recombination-cascade lines: emitted by the
    # ion shown, POPULATED by recombination of `recomb_parent` (one stage up),
    # α_eff = recomb_branch · α_tot(parent,T) — same physically-bounded scheme as
    # C_III_4647. Branch values are literature-informed (PPB91-consistent,
    # ~factor-2): C II 4267 is anchored to the classic α_eff≈1.6e-13 cm³/s
    # (branch=α_eff/α_tot(C_III)≈0.07). Cloudy's light-element model atoms emit
    # ~0 for these ORLs (verified: C III 5696 absent from its line stack), so the
    # recomb channel is authoritative here. NOTE: C IV 5801/12 is DEFERRED — it
    # recombines from C⁴⁺ ('C_V'), which is not in the ion ladder; adding that
    # stage would redistribute the existing C_IV fractions (breaks the
    # backwards-compat gate) and needs its own gated change.
    'C_III_5696': dict(ion='C_III', lam_AA=5695.92, mech='recomb',
                       recomb_parent='C_IV', recomb_branch=8.0e-3),  # 3d¹D→3p¹P°
    'C_II_4267':  dict(ion='C_II',  lam_AA=4267.00, mech='recomb',
                       recomb_parent='C_III', recomb_branch=7.0e-2), # 3d²D→3p²P° (classic ORL)
    'C_II_6580':  dict(ion='C_II',  lam_AA=6580.00, mech='recomb',
                       recomb_parent='C_III', recomb_branch=2.0e-2), # 3p²P°→3s²S (6578+6583)
    'O_I_7774':   dict(ion='O_I',   lam_AA=7773.40, mech='recomb',
                       recomb_parent='O_II', recomb_branch=3.0e-1),  # 3p⁵P→3s⁵S° triplet
    # C IV 5801/12 doublet (3p²P°→3s²S of Li-like C³⁺) — one of the STRONGEST
    # optical features of real Icn (SN 2021csp). Recombination-cascade from
    # He-like C⁴⁺ ('C_V', item 3): at SN temperatures collisional excitation of
    # the 3p level (~40 eV above ground) is negligible, so the line genuinely
    # traces n_e·n(C⁴⁺) — i.e. it lights up only where hard (≥392 eV) shock
    # X-rays maintain C⁴⁺. branch≈0.1 literature-informed, ~factor-2.
    'C_IV_5801':  dict(ion='C_IV',  lam_AA=5805.0, mech='recomb',
                       recomb_parent='C_V', recomb_branch=1.0e-1),
}

# lines that carry an extra channel not captured by the single mechanism above
METAL_PROVISIONAL = set(METAL_LINES.keys())   # ALL are provisional in v1

# Absorption oscillator strengths f_lu for the RESONANCE optical-depth correction.
# The lower level of these transitions is (essentially) the ion ground state, so
# the Sobolev line optical depth τ = σ·f_lu·λ·n_ion·t_exp is significant for the
# permitted resonance line and the emissivity is escape-probability-corrected:
#   • C IV 1549  : true resonance doublet (2s–2p), f≈0.29 → OPTICALLY THICK.
#   • C III] 1909: intercombination (spin-forbidden), f≈5e-7 → essentially thin.
#   • forbidden / recombination lines: f≈0 → thin (no correction).
METAL_F_LU = {
    'C_IV_1549':   0.286,
    'C_III_1909':  5.4e-7,
    'C_III_4647':  0.0,
    'C_III_5696':  0.0,
    'C_II_4267':   0.0,
    'C_II_6580':   0.0,
    'O_I_7774':    0.0,
    'C_IV_5801':   0.0,
    'O_I_6300':    0.0,
    'O_III_5007':  0.0,
    'Ne_III_3869': 0.0,
}


def f_lu(line_name):
    """Absorption oscillator strength for the resonance optical-depth correction
    (0 → treat as optically thin)."""
    return float(METAL_F_LU.get(line_name, 0.0))

# convenience: rest wavelengths (for schema / continuum)
METAL_LINES_AA = {name: d['lam_AA'] for name, d in METAL_LINES.items()}

# pretty labels + plotting colours (by element) for figures/movies
METAL_PRETTY = {
    'C_IV_1549':  'C IV 1549',  'C_III_1909': 'C III] 1909', 'C_III_4647': 'C III 4647',
    'O_I_6300':   '[O I] 6300',  'O_III_5007': '[O III] 5007', 'Ne_III_3869': '[Ne III] 3869',
    'C_III_5696': 'C III 5696', 'C_II_4267': 'C II 4267',
    'C_II_6580':  'C II 6580',  'O_I_7774':  'O I 7774',
    'C_IV_5801':  'C IV 5801/12',
}
METAL_COLOR = {'C': '#8c564b', 'O': '#1f77b4', 'Ne': '#9467bd'}   # brown/blue/purple


def line_color(line_name):
    """Plot colour for a metal line, keyed by element."""
    elem = METAL_LINES.get(line_name, {}).get('ion', '_').split('_')[0]
    return METAL_COLOR.get(elem, '#555555')


# ---------------------------------------------------------------------------
# 3. Emissivity functions  [erg/s/cm³]  (per zone)
# ---------------------------------------------------------------------------
def recomb_emissivity(line_name, T, n_e, n_ion):
    """Recombination-line volume emissivity j = hν · α_eff(T) · n_e · n_ion.

    `n_ion` must be the density of the RECOMBINING (parent) ion — i.e. the ion
    ONE stage ABOVE the emitter — because the line is produced when that ion
    captures an electron and cascades down through the emitting level. The caller
    (metal_lines) supplies the parent density when the line dict carries
    `recomb_parent`; otherwise it passes the line's own ion (correct for the
    resonance-line fallback, where the emitter and the radiating ion coincide).

    Two ways to set α_eff(T):
      • `recomb_branch` (preferred): α_eff = branch · α_tot(parent, T), the
        multiplet branching fraction times the TOTAL (Badnell-form) recombination
        rate of the parent ion. This bounds α_eff ≤ α_tot and carries the real
        recombination temperature dependence; only the branching is uncertain.
      • `alpha_eff = (A, b)`: a direct A·(T/1e4)^b fit (legacy / resonance fallback).
    """
    d = METAL_LINES[line_name]
    T = np.maximum(np.asarray(T, float), 100.0)
    if 'recomb_branch' in d:
        alpha = float(d['recomb_branch']) * alpha_recomb_total(
            d.get('recomb_parent', d['ion']), T)
    else:
        A, b = d['alpha_eff']
        alpha = A * (T / 1.0e4) ** b
    nu = C / (d['lam_AA'] * 1e-8)
    return H_PL * nu * alpha * np.asarray(n_e, float) * np.asarray(n_ion, float)


_COLL_CONST = 8.629e-6     # cgs collisional rate-coefficient prefactor


def critical_density(line_name, T):
    """Critical density n_crit = A_ul / q_ul(T) [cm⁻³] for a CEL line, where
    q_ul = 8.629e-6 · Ω(T) / (g_u √T) is the collisional DE-excitation rate
    coefficient. Above n_crit, collisional de-excitation quenches the line."""
    d = METAL_LINES[line_name]
    O_1e4, p = d['Omega']
    T = np.maximum(np.asarray(T, float), 100.0)
    Omega = O_1e4 * (T / 1.0e4) ** p
    q_ul = _COLL_CONST * Omega / (float(d['g_u']) * np.sqrt(T))
    return float(d['A_ul']) / np.maximum(q_ul, 1e-30)


def cel_emissivity(line_name, T, n_e, n_ion):
    """Collisionally-excited (forbidden/intercombination) line volume emissivity
    [erg/s/cm³], WITH the critical-density (collisional de-excitation) correction:

        q₁₂   = 8.629e-6 · Ω(T) / (g_l √T) · exp(−ΔE/kT)          [excitation]
        j_low = hν · q₁₂ · n_e · n_ion · branch                  [low-density]
        j     = j_low / (1 + n_e / n_crit),   n_crit = A_ul / q_ul

    Limits (correct at BOTH ends):
      • n_e ≪ n_crit : j → j_low                (every excitation radiates)
      • n_e ≫ n_crit : j → hν·A_ul·branch·n_ion·(g_u/g_l)·exp(−ΔE/kT)
                          = the LTE/Boltzmann (thermalised upper level) emissivity
    The n_crit correction matters for the density-sensitive C III] 1909 and
    [O III] 5007 in dense CSM (n_e ≳ 1e6), where the low-density formula would
    over-predict by n_crit/n_e.
    """
    d = METAL_LINES[line_name]
    O_1e4, p = d['Omega']
    T = np.maximum(np.asarray(T, float), 100.0)
    n_e = np.asarray(n_e, float)
    Omega = O_1e4 * (T / 1.0e4) ** p
    q12 = _COLL_CONST * Omega / (float(d['g_l']) * np.sqrt(T)) \
        * np.exp(-float(d['dE_eV']) * EV / (KB * T))
    q_ul = _COLL_CONST * Omega / (float(d['g_u']) * np.sqrt(T))
    n_crit = float(d['A_ul']) / np.maximum(q_ul, 1e-30)
    suppression = 1.0 / (1.0 + n_e / np.maximum(n_crit, 1e-30))
    nu = C / (d['lam_AA'] * 1e-8)
    return (H_PL * nu * q12 * n_e * np.asarray(n_ion, float)
            * float(d['branch']) * suppression)


def line_emissivity(line_name, T, n_e, n_ion):
    """Dispatch to the right mechanism for a metal line. Returns j [erg/s/cm³]."""
    mech = METAL_LINES[line_name]['mech']
    if mech == 'recomb':
        return recomb_emissivity(line_name, T, n_e, n_ion)
    if mech == 'cel':
        return cel_emissivity(line_name, T, n_e, n_ion)
    raise ValueError(f"unknown mechanism {mech!r} for {line_name}")


# ---------------------------------------------------------------------------
# 4. Photoionization-equilibrium ion ladder (trace-element approximation)
# ---------------------------------------------------------------------------
def ion_ladder_fractions(element, gamma_by_ion, T, n_e):
    """Fractional ion populations for one element from photoionization balance.

    For each successive stage i, photoionization balance gives
        n_{i+1}/n_i = Γ_i / (n_e · α_{i+1}(T))
    where Γ_i is the ionization rate OUT of stage i [s⁻¹] (from the radiation
    field; supplied per ion in `gamma_by_ion`) and α_{i+1} is the total
    recombination INTO stage i. The fractions are built from the running product
    of these ratios and normalised to sum to 1, per zone.

    Parameters
    ----------
    element : 'C' | 'O' | 'Ne'
    gamma_by_ion : dict {ion_label: Γ array [s⁻¹]} — ionization rate out of each
                   non-top stage (e.g. {'C_I':…, 'C_II':…, 'C_III':…}).
    T, n_e : per-zone arrays.

    Returns
    -------
    dict {ion_label: fractional population array} for every stage of `element`.
    """
    T = np.asarray(T, float)
    n_e = np.maximum(np.asarray(n_e, float), 1.0)
    stages = sorted((k for k, v in ION_DATA.items() if v['elem'] == element),
                    key=lambda k: ION_DATA[k]['stage'])
    nz = T.shape[0] if T.ndim else 1
    # running product P_i = n_i / n_0 (P_0 = 1)
    prod = [np.ones(nz)]
    for i in range(len(stages) - 1):
        lower = stages[i]
        upper = stages[i + 1]
        gamma = np.asarray(gamma_by_ion.get(lower, np.zeros(nz)), float)
        alpha = alpha_recomb_total(upper, T)
        ratio = gamma / (n_e * np.maximum(alpha, 1e-30))     # n_{i+1}/n_i
        prod.append(prod[-1] * ratio)
    P = np.array(prod)                      # (nstage, nz)
    norm = np.sum(P, axis=0)
    norm = np.where(norm > 0, norm, 1.0)
    return {stages[i]: P[i] / norm for i in range(len(stages))}


def number_density(element, X_elem, rho):
    """Total number density of `element` [cm⁻³] from its mass fraction."""
    return np.asarray(X_elem, float) * np.asarray(rho, float) / M_ATOM[element]
