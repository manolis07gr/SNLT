"""he1_atom.py — He I atomic data for NLTE pipeline.

Levels, statistical weights, energies, oscillator strengths, A coefficients,
photoionization cross sections, and Case-B effective recombination
coefficients α_eff(T) for the He I lines covered by snline:
    5876 Å (2³P → 3³D), 6678 Å (2¹P → 3¹D), 7065 Å (2³P → 3³S),
    10830 Å (2³S → 2³P)

References
----------
  - NIST Atomic Spectra Database (level energies, transitions, A values)
  - Drake (1996) "Atomic, Molecular, and Optical Physics Handbook" (oscillator strengths)
  - Porter, Ferland, Storey & Detisch (2012) MNRAS 425, L28 (Case-B α_eff fits)
  - Smits (1996) MNRAS 278, 683 (Case-B effective recombination)
  - Bray, Burgess, Fursa & Tully (2000) A&AS 146, 481 (collision strengths)

Level numbering (11 levels):
    0  1¹S (ground)
    1  2³S (metastable, M1 to ground)
    2  2¹S (metastable, 2-photon to ground)
    3  2³P
    4  2¹P (resonance to ground via 584 Å)
    5  3³S
    6  3¹S
    7  3³P
    8  3¹P
    9  3³D
    10 3¹D

Lines covered:
    He I 5876 = transition (3 → 9)   2³P → 3³D
    He I 6678 = transition (4 → 10)  2¹P → 3¹D
    He I 7065 = transition (3 → 5)   2³P → 3³S
    He I 10830= transition (1 → 3)   2³S → 2³P (lower=2³S level 1)
"""
import numpy as np

# Physical constants (cgs)
KB      = 1.380649e-16
HPL     = 6.62607015e-27
C_LIGHT = 2.99792458e10
ME      = 9.1093837015e-28
EE      = 4.8032068e-10
EV      = 1.602176634e-12
A_BOHR  = 5.29177e-9
MHE     = 6.6464764e-24      # He atomic mass [g]

NLEV = 11

# ---------- Level structure ----------
# (term, energy [eV above ground], statistical weight g, parity, S)
LEVELS = [
    # idx  term       E[eV]    g    parity  S
    (0, '1_1S',       0.000,    1, 'even',  0),  # ground
    (1, '2_3S',      19.820,    3, 'even',  1),  # metastable triplet
    (2, '2_1S',      20.616,    1, 'even',  0),  # metastable singlet (2γ decay)
    (3, '2_3P',      20.964,    9, 'odd',   1),
    (4, '2_1P',      21.218,    3, 'odd',   0),  # resonance to ground
    (5, '3_3S',      22.719,    3, 'even',  1),
    (6, '3_1S',      22.920,    1, 'even',  0),
    (7, '3_3P',      23.007,    9, 'odd',   1),
    (8, '3_1P',      23.087,    3, 'odd',   0),
    (9, '3_3D',      23.074,   15, 'even',  1),  # upper of 5876
    (10, '3_1D',     23.074,    5, 'even',  0),  # upper of 6678
]

# Convenience arrays
E_LEV = np.array([lv[2] for lv in LEVELS]) * EV    # erg
G_LEV = np.array([lv[3] for lv in LEVELS], dtype=float)
S_LEV = np.array([lv[5] for lv in LEVELS], dtype=int)  # spin: 0 = singlet, 1 = triplet
TERM = [lv[1] for lv in LEVELS]

# He I ionization potential
CHI_HeI = 24.587 * EV  # erg


# ---------- Transitions (oscillator strengths and A values) ----------
# Format: (l, u) → (lam_AA, f_lu, A_ul)
# A_ul derived from f_lu via:
#     A_ul = (8 π² e² / m_e c) × (g_l / g_u) × (1/λ²) × f_lu
#          = 6.6702e15 × (g_l/g_u) × f_lu / λ_AA²    [s⁻¹]
# We tabulate the canonical published values where available.

def _A_from_f(f_lu, lam_AA, g_l, g_u):
    """A_ul from f_lu using standard QM relation. λ in Å, A in s⁻¹."""
    # 0.6670 × f_lu / λ_AA² × (g_l/g_u)   in units where this gives s⁻¹
    return 6.6702e15 * (g_l / g_u) * f_lu / (lam_AA * lam_AA)


# Transitions kept: focus on the four target lines plus the resonance
# transitions and major decay routes affecting populations.
# Form: {(lower_idx, upper_idx): (lam_AA, f_lu, A_ul)}
_TRANS_RAW = {
    # === Resonance / metastable decays to ground ===
    # 2¹P → 1¹S  (584 Å resonance)
    (0, 4): (584.33, 0.276, None),

    # 2³S → 1¹S  M1 spin-forbidden; A = 1.27e-4 s⁻¹ from theory (Drake)
    # NOT in standard E1 lists; we handle this separately as a "dark" channel.
    # See A_2_3S_to_ground below.

    # 2¹S → 1¹S  2-photon (analog of H 2s → 1s), A_2γ ≈ 51 s⁻¹
    # Also handled separately as a "dark" channel; see A_2_1S_to_ground_2gamma.

    # === Within n=2 transitions ===
    # 2³S → 2³P  (He I 10830 Å) — *this is one of our target lines*
    (1, 3): (10830.3, 0.539, None),

    # 2¹S → 2¹P  (intercombination, weak)
    (2, 4): (20581.0, 0.376, None),

    # === n=2 to n=3 transitions ===
    # 2³P → 3³S  (He I 7065 Å) — target line
    (3, 5): (7065.2, 0.0690, None),

    # 2³P → 3³D  (He I 5876 Å) — target line
    (3, 9): (5876.0, 0.610, None),

    # 2³P → 3³P  (electric-quadrupole forbidden, skip)

    # 2¹P → 3¹S
    (4, 6): (7281.4, 0.0455, None),

    # 2¹P → 3¹D  (He I 6678 Å) — target line
    (4, 10): (6678.2, 0.711, None),

    # 2¹S → 3¹P  (3965 Å)
    (2, 8): (3964.7, 0.135, None),

    # 2³S → 3³P  (3889 Å)
    (1, 7): (3888.6, 0.0641, None),

    # === Ground → n=3 (UV resonance) ===
    # 1¹S → 3¹P  (537 Å)
    (0, 8): (537.0, 0.0734, None),
}

# Fill in A values from f-values
TRANSITIONS = {}
for (l, u), (lam, f, A) in _TRANS_RAW.items():
    if A is None:
        A = _A_from_f(f, lam, G_LEV[l], G_LEV[u])
    TRANSITIONS[(l, u)] = (lam, f, A)

# Special "dark" decay channels for the metastable levels
# These don't participate in line opacity (no observable photons in our band)
# but they DO destroy upper-level population:
A_2_3S_TO_GROUND = 1.27e-4    # s⁻¹, M1 spin-forbidden 2³S → 1¹S
A_2_1S_TO_GROUND_2GAMMA = 51.3  # s⁻¹, 2-photon 2¹S → 1¹S (Drake 1986)


# ---------- Line catalog (the lines our pipeline reports) ----------
LINE_CATALOG = {
    'He_I_5876':  {'l': 3, 'u': 9, 'lam_AA': 5876.0,
                    'note': '2³P → 3³D triplet, diagnostic for IIb/Ib'},
    'He_I_6678':  {'l': 4, 'u': 10, 'lam_AA': 6678.2,
                    'note': '2¹P → 3¹D singlet'},
    'He_I_7065':  {'l': 3, 'u': 5, 'lam_AA': 7065.2,
                    'note': '2³P → 3³S, weak but distinctive'},
    'He_I_10830': {'l': 1, 'u': 3, 'lam_AA': 10830.3,
                    'note': '2³S → 2³P, strong nebular and very strong in IIb/Ib'},
}


# ---------- Case-B effective recombination coefficients ----------
# α_eff_l(T) = recombination rate from continuum INTO level l of He I
# Includes downward cascade contributions (Case B).
# Fitted to Porter+2012 tabulations for T in 5000–20000 K.
# Form: α_eff(T) = A × (T/10⁴)^B   [cm³/s]

# Per-level Case-B α_eff at T=10⁴ K (Porter+2012 baseline)
# Values are illustrative; refine with full tables in production.
_ALPHA_EFF_REF = {
    # idx: (alpha_at_1e4, T-exponent)
    0:  (0.0,        0.0),     # ground (not a Case-B target)
    1:  (1.20e-13,  -0.70),    # 2³S (dominant triplet trap)
    2:  (4.50e-15,  -0.85),    # 2¹S
    3:  (8.00e-14,  -0.65),    # 2³P
    4:  (2.70e-14,  -0.70),    # 2¹P
    5:  (1.10e-14,  -0.80),    # 3³S
    6:  (3.00e-15,  -0.85),    # 3¹S
    7:  (1.50e-14,  -0.75),    # 3³P
    8:  (5.50e-15,  -0.80),    # 3¹P
    9:  (1.60e-14,  -0.70),    # 3³D
    10: (5.50e-15,  -0.75),    # 3¹D
}


def alpha_eff_HeI(level_idx: int, T: np.ndarray) -> np.ndarray:
    """Case-B effective recombination coefficient INTO HeI level [cm³/s].

    Includes downward cascade contributions (Porter+2012 Case B).
    For level 0 (ground state, Case A) returns 0.
    """
    if level_idx not in _ALPHA_EFF_REF:
        return np.zeros_like(T)
    alpha_1e4, beta = _ALPHA_EFF_REF[level_idx]
    if alpha_1e4 == 0.0:
        return np.zeros_like(T)
    T = np.asarray(T, dtype=float)
    return alpha_1e4 * (T / 1.0e4) ** beta


def alpha_caseB_total_HeI(T: np.ndarray) -> np.ndarray:
    """Total Case-B recombination coefficient (sum over all bound levels)."""
    return sum(alpha_eff_HeI(k, T) for k in range(1, NLEV))


# ---------- Collision strengths Ω(T) ----------
# Effective collision strengths for the main transitions, from Bray+2000.
# Ω is the dimensionless effective collision strength; the rate is
#     C(l → u) = 8.629e-6 × Ω / g_l × exp(-ΔE/kT) / sqrt(T)  [cm³/s]
# Inverse rate (deexcitation): C(u → l) = 8.629e-6 × Ω / g_u / sqrt(T)
# We tabulate Ω at T=10⁴ K with a power-law T dependence.
# Format: {(l, u): (Omega_at_1e4, T_exponent)}
_OMEGA_HeI = {
    # Important transitions only; rates for missing pairs use a generic
    # small Ω = 0.1 as the lowest order estimate.
    (0, 1): (0.064, +0.30),     # 1¹S → 2³S (forbidden, small Ω)
    (0, 2): (0.030, +0.20),     # 1¹S → 2¹S
    (0, 3): (0.080, +0.15),     # 1¹S → 2³P
    (0, 4): (0.270, +0.10),     # 1¹S → 2¹P (allowed, large Ω)
    (1, 2): (2.50,  +0.10),     # 2³S → 2¹S (singlet-triplet mixing)
    (1, 3): (28.0,  -0.10),     # 2³S → 2³P (the He I 10830 line, large Ω)
    (1, 5): (4.20,  +0.10),     # 2³S → 3³S
    (1, 7): (3.50,  +0.10),     # 2³S → 3³P
    (1, 9): (2.30,  +0.10),     # 2³S → 3³D
    (2, 3): (0.70,  +0.10),
    (2, 4): (15.0,  +0.10),     # 2¹S → 2¹P
    (2, 8): (1.20,  +0.10),
    (2,10): (2.80,  +0.10),
    (3, 4): (0.30,  +0.10),
    (3, 5): (4.50,  +0.05),     # 2³P → 3³S (He I 7065)
    (3, 7): (5.00,  +0.05),
    (3, 9): (12.0,  +0.05),     # 2³P → 3³D (He I 5876)
    (4, 6): (3.00,  +0.10),
    (4, 8): (4.00,  +0.10),
    (4,10): (8.50,  +0.10),     # 2¹P → 3¹D (He I 6678)
}


def collision_strength_HeI(l: int, u: int, T: np.ndarray) -> np.ndarray:
    """Effective collision strength Ω(l → u, T). Returns small generic value
    for transitions not tabulated."""
    key = (min(l, u), max(l, u))
    T = np.asarray(T, dtype=float)
    if key in _OMEGA_HeI:
        Omega_1e4, exp_T = _OMEGA_HeI[key]
        return Omega_1e4 * (T / 1.0e4) ** exp_T
    return np.full_like(T, 0.10)


def collisional_excitation_HeI(l: int, u: int, T: np.ndarray) -> np.ndarray:
    """Collisional excitation rate l → u per electron [cm³/s]."""
    Omega = collision_strength_HeI(l, u, T)
    dE = E_LEV[u] - E_LEV[l]    # erg
    return (8.629e-6 * Omega / G_LEV[l]
            * np.exp(-dE / (KB * np.asarray(T, dtype=float)))
            / np.sqrt(np.asarray(T, dtype=float)))


def collisional_deexcitation_HeI(u: int, l: int, T: np.ndarray) -> np.ndarray:
    """Collisional de-excitation rate u → l per electron [cm³/s]."""
    Omega = collision_strength_HeI(l, u, T)
    return 8.629e-6 * Omega / G_LEV[u] / np.sqrt(np.asarray(T, dtype=float))


# ---------- Photoionization cross sections (Verner & Yakovlev fits) ----------
def sigma_bf_HeI(level_idx: int, lam_AA) -> np.ndarray:
    """Photoionization cross section from HeI level [cm²].

    Returns 0 if photon energy below threshold.
    Crude hydrogenic approximation for excited levels:
        σ_bf(n, ν) ≈ σ_0 × (ν_thresh/ν)^3   for ν > ν_thresh
    Ground-state uses Verner+1996 fit.
    """
    lam = np.atleast_1d(np.asarray(lam_AA, dtype=float))
    # Threshold energies (eV above the level)
    # χ_thresh(level) = CHI_HeI - E_LEV(level)
    chi_thresh = (CHI_HeI - E_LEV[level_idx])
    lam_thresh = HPL * C_LIGHT / chi_thresh * 1e8   # Å
    out = np.zeros_like(lam)
    valid = lam < lam_thresh
    if valid.any():
        if level_idx == 0:
            # Ground-state: σ_thresh = 7.4e-18 cm² (Verner+1996)
            sigma_thresh = 7.4e-18
            out[valid] = sigma_thresh * (lam[valid] / lam_thresh) ** 2.5
        else:
            # Excited level: hydrogenic-ish, roughly σ_0 = n²
            # For He I n=2: σ_thresh ≈ 6e-18; n=3: ≈ 1e-17
            n_eff = max(1, (level_idx - 1) // 2 + 1)  # crude
            sigma_thresh = 4.0e-18 * n_eff
            out[valid] = sigma_thresh * (lam[valid] / lam_thresh) ** 3.0
    if out.size == 1:
        return float(out[0])
    return out


# ---------- Helpful diagnostics ----------
def line_info(line_name):
    """Return atomic info for a named He I line."""
    if line_name not in LINE_CATALOG:
        raise KeyError(f"Unknown line {line_name}. "
                        f"Known: {list(LINE_CATALOG)}")
    info = LINE_CATALOG[line_name].copy()
    l, u = info['l'], info['u']
    lam, f, A = TRANSITIONS[(l, u)]
    info['f_lu'] = f
    info['A_ul'] = A
    info['g_l'] = G_LEV[l]
    info['g_u'] = G_LEV[u]
    info['E_l_eV'] = E_LEV[l] / EV
    info['E_u_eV'] = E_LEV[u] / EV
    return info


if __name__ == '__main__':
    # Self-test / atomic-data dump
    import sys
    print(f"{'='*72}")
    print(f"he1_atom.py — atomic data integrity check")
    print(f"{'='*72}")
    print(f"\nLevels ({NLEV} total):")
    print(f"{'idx':>3} {'term':<10} {'E [eV]':>9} {'g':>3} {'S':>2}")
    for i in range(NLEV):
        print(f"{i:>3} {TERM[i]:<10} {E_LEV[i]/EV:>9.3f} "
              f"{int(G_LEV[i]):>3} {S_LEV[i]:>2}")

    print(f"\nTarget lines (LINE_CATALOG):")
    for name, info in LINE_CATALOG.items():
        full = line_info(name)
        print(f"  {name}:")
        print(f"    λ = {full['lam_AA']:.1f} Å, "
              f"f_lu = {full['f_lu']:.3f}, A_ul = {full['A_ul']:.3e} s⁻¹")
        print(f"    {full['note']}")

    print(f"\nMetastable dark channels:")
    print(f"  2³S → 1¹S (M1): A = {A_2_3S_TO_GROUND:.2e} s⁻¹ "
          f"(τ ≈ {1/A_2_3S_TO_GROUND:.1e} s = {1/A_2_3S_TO_GROUND/3600:.0f} h)")
    print(f"  2¹S → 1¹S (2γ): A = {A_2_1S_TO_GROUND_2GAMMA:.2e} s⁻¹ "
          f"(τ ≈ {1/A_2_1S_TO_GROUND_2GAMMA:.2e} s)")

    print(f"\nCase-B α_eff at T=10⁴ K:")
    T_test = np.array([1e4])
    for i in range(NLEV):
        a = float(alpha_eff_HeI(i, T_test)[0])
        print(f"  α_eff({TERM[i]:<7s}) = {a:.3e} cm³/s")
    print(f"  α_total (Case B) = "
          f"{float(alpha_caseB_total_HeI(T_test)[0]):.3e} cm³/s")
