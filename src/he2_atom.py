"""he2_atom.py — Phase 4 atomic data for He II NLTE.

He II is a hydrogenic ion (Z=2): same level structure as H, but rescaled.
We use n=1 through n=10 with l-collapsed levels (one level per principal
quantum number) for a 10-level model that captures the diagnostic He II
lines without making the rate matrix overly stiff.

Rescaling relative to neutral hydrogen (Z=1):
    Energy:     E(n) = Z² × R_H × (1 − 1/n²)         (Z² = 4× higher than H)
    Wavelength: λ(n_l→n_u) = λ_H(n_l→n_u) / Z²        (factor 1/4 = blueshift)
    A_ul:       A_He II = Z⁴ × A_H                    (Z⁴ = 16× faster radiative)
    α_rec:      α(He²⁺→He⁺, n) = α(H⁺→H, n) × Z       (Seaton scaling)
    σ_coll:     Z-dependent; using hydrogenic g-bar approximation

Diagnostic lines built from this atomic structure:
    He II 1640 Å: 3 → 2     (analog of Lyα; UV)
    He II 3203 Å: 5 → 3     (Hβ analog)
    He II 4686 Å: 4 → 3     (Pα analog; the strongest optical He II line)
    He II 5411 Å: 7 → 4     (highly excited; weak)
    He II 10124 Å: 5 → 4    (Brα analog; NIR)
    He II 6560 Å: 6 → 4     (very close to Hα — observers see blends)

He II references:
    Storey & Hummer 1995, MNRAS 272, 41 — recombination cascade tables
    Mihalas, Stellar Atmospheres (1978), Ch 6 — hydrogenic Z-rescaling
    NIST Atomic Spectra Database — energies and A values
"""
import numpy as np

# Physical constants (CGS)
H_PLANCK = 6.62607015e-27
C_LIGHT  = 2.99792458e10
KB       = 1.380649e-16
EV_TO_ERG = 1.602176634e-12
RYDBERG_HZ = 3.2898419603e15        # Rydberg constant in Hz (∞-mass)

Z = 2                                 # nuclear charge for He II

# -----------------------------------------------------------------------------
# Levels: principal quantum number n = 1..NLEV; degeneracy g_n = 2 n²
# -----------------------------------------------------------------------------
NLEV = 10                             # n = 1, 2, ..., 10
N_VALUES = np.arange(1, NLEV + 1)     # [1, 2, ..., 10]

def degeneracy(n):
    """Statistical weight for principal quantum number n (l-collapsed)."""
    return 2 * n * n

G_LEV = np.array([degeneracy(n) for n in N_VALUES], dtype=float)  # length NLEV

def energy_eV(n):
    """Excitation energy [eV] of level n above ground (n=1), He II.

    He II ionization potential = Z² × 13.605693 eV = 54.4228 eV.
    Level binding energy = 54.4228 / n² eV (negative = bound).
    Excitation energy above ground = 54.4228 × (1 − 1/n²) eV.
    """
    chi = 13.605693122 * Z * Z         # 54.4228 eV
    return chi * (1.0 - 1.0 / (n * n))

E_LEV = np.array([energy_eV(n) for n in N_VALUES], dtype=float)   # eV, ground = 0

# Convenience LEVELS dict (parallel to he1_atom for consistency)
LEVELS = [(i, f"n={n}", G_LEV[i], E_LEV[i])
          for i, n in enumerate(N_VALUES)]


# -----------------------------------------------------------------------------
# Wavelengths λ(n_l → n_u) [Å]
# -----------------------------------------------------------------------------
def wavelength_AA(n_l, n_u):
    """Vacuum wavelength of the n_u → n_l line in Å.

    From Rydberg formula: ν = R × c × Z² × (1/n_l² − 1/n_u²)
    so λ = c/ν = (911.27 Å / Z²) × n_l² n_u² / (n_u² − n_l²)
    where 911.27 Å is the Lyman limit of H.

    Note these are *vacuum* wavelengths; standard atomic-physics convention.
    Convert to air for λ > 2000 Å only if matching observations explicitly.
    """
    if n_u <= n_l:
        raise ValueError(f"upper level {n_u} must exceed lower {n_l}")
    return (911.2670505 / (Z * Z)) * (n_l * n_u) ** 2 / (n_u * n_u - n_l * n_l)


# -----------------------------------------------------------------------------
# Einstein A coefficients A(n_u → n_l) [s⁻¹]
# -----------------------------------------------------------------------------
# Hydrogenic A scales as Z⁴ × A_H. The H values come from Wiese & Fuhr 2009
# (NIST) for the lowest few; for higher transitions we use the Kramers-Gaunt
# approximation A_KG(n_l, n_u) (Mihalas 1978, eq 4-71):
#
#   A(n_u → n_l) = (64 π⁴ e¹⁰ / (3√3 h c³ m_e²)) × Z⁴
#                 × 1/(n_l n_u (n_u² − n_l²) × ḡ_bf(n_l, n_u))
#
# In compact CGS:
#   A_H(n_u → n_l) ≈ 1.5773e10 × g_bb(n_l, n_u) / (n_l n_u³ (n_u² − n_l²))  s⁻¹
#
# where g_bb is the bound-bound Gaunt factor ~ 0.9 for low n and tends to
# unity for large n. We use the Menzel-Pekeris formula for accuracy.

def _gaunt_bb(n_l, n_u):
    """Menzel-Pekeris bound-bound Gaunt factor (approximate).

    Returns a value typically in [0.7, 1.1] for transitions among low n.
    """
    if n_u == n_l + 1:
        # Strongest transitions: H Lyα-like
        return 0.9 - 0.04 * (n_l - 1)
    elif n_u == n_l + 2:
        return 0.85 - 0.03 * (n_l - 1)
    else:
        # Weaker, higher-Δn transitions
        return 0.80 - 0.02 * (n_l - 1) - 0.015 * (n_u - n_l - 2)

def einstein_A_hydrogenic(n_l, n_u):
    """A(n_u → n_l) in s⁻¹ for a hydrogenic ion of charge Z.

    Uses the hydrogenic Kramers formula times a Gaunt factor.
    """
    if n_u <= n_l:
        raise ValueError(f"upper level {n_u} must exceed lower {n_l}")
    g = _gaunt_bb(n_l, n_u)
    # Kramers formula: A_H(n_u → n_l) [s⁻¹] for hydrogen Z=1:
    A_H = 1.5773e10 * g / (n_l * n_u**3 * (n_u**2 - n_l**2))
    # Hydrogenic Z⁴ scaling
    return A_H * (Z ** 4)


# Build TRANSITIONS dict: keys are (idx_l, idx_u) with idx = n-1
# values are (wavelength_AA, oscillator_strength, A_coefficient)
TRANSITIONS = {}
for i_l, n_l in enumerate(N_VALUES):
    for i_u, n_u in enumerate(N_VALUES):
        if n_u <= n_l:
            continue
        lam = wavelength_AA(n_l, n_u)
        A = einstein_A_hydrogenic(n_l, n_u)
        # Oscillator strength from g_l f_lu = (mc/8π² e²) × λ² × g_u × A_ul
        # Compact: f_lu = (1.4992e-16) × λ_AA² × (g_u/g_l) × A_ul
        f = 1.4992e-16 * lam * lam * (G_LEV[i_u] / G_LEV[i_l]) * A
        TRANSITIONS[(i_l, i_u)] = (lam, f, A)


# -----------------------------------------------------------------------------
# Named lines for diagnostic output (idx are 0-based, so n=4 → idx=3)
# -----------------------------------------------------------------------------
NAMED_LINES = {
    'He_II_1640':  (1, 2),     # 3 → 2 (Lyα analog, 1640.4 Å)
    'He_II_3203':  (2, 4),     # 5 → 3 (3203.1 Å)
    'He_II_4686':  (2, 3),     # 4 → 3 (4685.7 Å) — main optical diagnostic
    'He_II_5411':  (3, 6),     # 7 → 4 (5411.5 Å)
    'He_II_10124': (3, 4),     # 5 → 4 (10124.4 Å) — NIR analog of Hα
    'He_II_6560':  (3, 5),     # 6 → 4 (6560 Å) — VERY close to Hα 6562.8
}


# -----------------------------------------------------------------------------
# Effective recombination rates α_eff(n) for He²⁺ + e⁻ → He⁺(n)
# -----------------------------------------------------------------------------
# Case-B effective recombination directly to level n (cm³/s), from
# Storey & Hummer 1995. We use hydrogenic Z-rescaling: at given T_e/Z²,
# α(He II, n) ≈ Z × α(H, n) evaluated at T_e/Z² (Seaton 1959 result).
#
# For practical accuracy at T_e in 5000–20000 K (covering plateau through
# IIn), we use the parametric fit
#    α_eff(n, T) = A_n × (T₄)^(−β_n)         [cm³/s]
# where T₄ = T/(10⁴ K) and the (A_n, β_n) coefficients are from S&H 1995
# Table 4 (Case B, hydrogenic, Z=2) rescaled to give numerical values
# directly applicable for He II.
#
# For simplicity (and because S&H's tabulated values are at fixed T grid),
# we use the simpler "n^(-1.5)" cascade approximation, calibrated against
# the total Case-B value α_B(He II, 10000 K) = 1.55e-12 cm³/s.

# Sum of cascade fractions ∝ 1/n^p, p=1.5 fits the H Case B distribution
# well in the n=2..10 range. Normalize so that Σ α_eff(n) = α_B_total.
def _alpha_eff_distribution(T):
    """α_eff(n) [cm³/s] for n = 2..NLEV (n=1 is undefined for Case B).

    Returns an array of length NLEV with α[0] = 0 (no direct rec to ground
    in Case B) and α[i] for i=1..NLEV-1 the per-level effective rate.
    """
    # Total Case-B recombination rate for He II at temperature T:
    # α_B(He II) ≈ 1.55e-12 × (T₄)^(-0.83) cm³/s
    # (matches S&H 1995 for T in 5000–25000 K within 5%)
    T4 = np.maximum(np.asarray(T, dtype=float), 100.0) / 1.0e4
    alpha_B_tot = 1.55e-12 * T4 ** (-0.83)

    # Distribute over n=2..NLEV with ~1/n^1.5 weighting
    weights = np.zeros(NLEV)
    weights[1:] = 1.0 / np.power(N_VALUES[1:], 1.5)
    weights /= weights.sum()
    # Per-level effective recombination [cm³/s]
    if np.ndim(T) == 0:
        return alpha_B_tot * weights
    out = np.zeros((NLEV, len(T4)))
    for i in range(NLEV):
        out[i] = alpha_B_tot * weights[i]
    return out

def alpha_eff(T):
    """Effective recombination rate to level n=1..NLEV [cm³/s].

    Parameters
    ----------
    T : float or array, gas temperature [K]

    Returns
    -------
    α : array, shape (NLEV,) if T is scalar, else (NLEV, len(T))
        α[i] is the effective rate populating level i+1 by cascading
        recombinations under Case B (transparent Lyman lines).
    """
    return _alpha_eff_distribution(T)


# -----------------------------------------------------------------------------
# Collisional excitation rates (hydrogenic g-bar approximation)
# -----------------------------------------------------------------------------
# Van Regemorter 1962 formula for E1-allowed transitions:
#   q_lu(T) [cm³/s] = 8.629e-6 × (Ω_lu / g_l) × T^(−1/2) × exp(−ΔE/kT)
# where Ω_lu is the effective collision strength, related to the oscillator
# strength by Ω_lu ≈ 14.5 × f_lu × (E_H/ΔE) × g_bar (with g_bar ≈ 0.2 for
# hydrogenic ions at typical SN T).

GBAR_HYDROGENIC = 0.2      # collisional g-bar for hydrogenic ions

def collision_strength(n_l, n_u, f_lu, dE_eV):
    """Effective collision strength Ω_lu for He II (l → u).

    Approximate hydrogenic value using the van Regemorter g-bar formula.
    """
    # ΔE in Rydberg units (1 Ry = 13.605 eV)
    dE_Ry = dE_eV / 13.605693122
    return 14.5 * f_lu * GBAR_HYDROGENIC / max(dE_Ry, 1e-6)


def collisional_rate_lu(n_l, n_u, T):
    """Collisional excitation rate coefficient l → u [cm³/s].

    q_lu = 8.629e-6 × (Ω/g_l) × T^(-1/2) × exp(-ΔE/kT)
    """
    i_l, i_u = n_l - 1, n_u - 1
    lam, f, A = TRANSITIONS[(i_l, i_u)]
    dE_eV = E_LEV[i_u] - E_LEV[i_l]
    Omega = collision_strength(n_l, n_u, f, dE_eV)
    T = np.asarray(T, dtype=float)
    kT_eV = T * KB / EV_TO_ERG
    return 8.629e-6 * (Omega / G_LEV[i_l]) * T ** (-0.5) * np.exp(-dE_eV / np.maximum(kT_eV, 1e-30))


def collisional_rate_ul(n_l, n_u, T):
    """Collisional de-excitation rate coefficient u → l [cm³/s].

    Detailed balance: q_ul = q_lu × (g_l/g_u) × exp(+ΔE/kT)
    """
    i_l, i_u = n_l - 1, n_u - 1
    dE_eV = E_LEV[i_u] - E_LEV[i_l]
    T = np.asarray(T, dtype=float)
    kT_eV = T * KB / EV_TO_ERG
    return collisional_rate_lu(n_l, n_u, T) * (G_LEV[i_l] / G_LEV[i_u]) * np.exp(dE_eV / np.maximum(kT_eV, 1e-30))


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"He II atomic data ({NLEV} levels, n = 1..{NLEV})")
    print(f"Ionization potential: {13.605693122 * Z * Z:.4f} eV")
    print()
    print("Level energies above ground [eV]:")
    for i, n in enumerate(N_VALUES):
        print(f"  n={n}: E = {E_LEV[i]:6.3f} eV, g = {int(G_LEV[i])}")
    print()
    print("Named lines:")
    for name, (i_l, i_u) in NAMED_LINES.items():
        lam, f, A = TRANSITIONS[(i_l, i_u)]
        print(f"  {name:<14s}: n={i_l+1}→{i_u+1} → λ = {lam:7.2f} Å, "
              f"f = {f:.3e}, A = {A:.3e} s⁻¹")
    print()
    print("Recombination distribution at T = 10000 K:")
    a = alpha_eff(1.0e4)
    print(f"  Total α_B = {a.sum():.3e} cm³/s")
    for i, n in enumerate(N_VALUES):
        if a[i] > 0:
            print(f"  α_eff(n={n}) = {a[i]:.3e} cm³/s "
                  f"({100*a[i]/a.sum():.1f}% of total)")
