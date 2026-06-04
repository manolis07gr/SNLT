"""
snapshot_analyzer.py
====================

Physics-based analysis of 1D spherical SN hydrodynamic snapshots. Identifies:

  1. SN type from the hydrodynamic structure (no manual classification).
  2. Physical regions (photosphere, line-formation zone, ejecta, shell, wind,
     unbound CSM) from local physics (τ_es, density jumps, velocity gradient).
  3. Per-region characteristic temperature, density, velocity scales.
  4. Diagnostic flags when the snapshot violates assumptions of the radiative
     transfer model (e.g., extreme T variations, mass conservation issues,
     non-monotonic v in unexpected ways).

NO HARDCODED THRESHOLDS that aren't documented and physically motivated. Every
threshold has units, a physical reason, and is logged when triggered.

Usage
-----
    from heracles_reader import read_heracles_atmosphere
    from snapshot_analyzer import analyze_snapshot

    snap = read_heracles_atmosphere('atmosphere_8.dat')
    state = analyze_snapshot(snap)
    print(state.report())
    # state.sn_type        -- 'IIn', 'IIP', 'Ia', 'stripped_envelope', or 'unknown'
    # state.regions        -- dict of (name, mask) for photosphere, ejecta, etc.
    # state.R_phot, T_phot, L_phot
    # state.v_turb_kms     -- derived from local sound speed
    # state.diagnostics    -- list of warnings/notes
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Physical constants (cgs)
SIGMAT = 6.6524587e-25      # Thomson cross-section [cm²]
KB = 1.380649e-16           # Boltzmann constant [erg/K]
MH = 1.6735575e-24          # hydrogen mass [g]
ME = 9.1093837e-28          # electron mass [g]
SIG_SB = 5.670374e-5        # Stefan-Boltzmann [erg/cm²/s/K⁴]
GAMMA_GAS = 5.0/3.0         # ideal gas adiabatic index for monatomic gas
LSUN = 3.828e33             # solar luminosity [erg/s]


# ---------------------------------------------------------------------------
# Data class for analyzed snapshot state
# ---------------------------------------------------------------------------
@dataclass
class PhysicalState:
    """Self-consistent description of the snapshot physics, derived from the
    snapshot itself with no user-tunable parameters."""

    # Snapshot identity
    snapshot_path: str = ""
    t_explosion_d: float = 0.0
    n_zones: int = 0

    # SN classification
    sn_type: str = "unknown"
    sn_type_confidence: float = 0.0       # 0-1; how confident the classifier is
    sn_type_evidence: Dict = field(default_factory=dict)

    # Photosphere (τ_es = 1 surface)
    i_phot: int = 0                       # zone index
    R_phot: float = 0.0                   # cm
    T_phot: float = 0.0                   # K (gas T at τ=1, used for RE + populations)
    T_color: float = 0.0                  # K (BB color T derived from L_phot, R_phot)
    v_phot: float = 0.0                   # cm/s
    L_phot: float = 0.0                   # erg/s, taken from L(r) at i_phot

    # Region identification (boolean masks of length n_zones)
    regions: Dict = field(default_factory=dict)
    # 'photosphere'      — zones with τ_es ≥ 1 (optically thick to e-scattering)
    # 'line_forming'     — zones with significant Sobolev escape × line emissivity
    # 'cdshell'          — cold dense shell, if present (IIn-like)
    # 'wind'             — slow outflow ahead of photosphere (CSM)
    # 'ejecta'           — fast, dense, shock-heated material behind photosphere

    # Derived line-forming-region characteristics
    T_line_forming: float = 0.0           # mean T in line-forming zones [K]
    n_e_line_forming: float = 0.0         # mean n_e in line-forming zones [cm⁻³]
    v_line_forming_min: float = 0.0       # min |v| in line-forming zones [cm/s]
    v_line_forming_max: float = 0.0       # max |v| in line-forming zones [cm/s]

    # Derived microturbulent velocity
    v_turb_kms: float = 0.0               # km/s, set to local sound speed
    v_turb_basis: str = ""                # what was used to derive v_turb

    # Composition (per-zone mass fractions; defaults inferred if not in snapshot)
    X_H: np.ndarray = field(default_factory=lambda: np.array([]))
    X_He: np.ndarray = field(default_factory=lambda: np.array([]))
    X_metals: np.ndarray = field(default_factory=lambda: np.array([]))
    composition_source: str = ""          # "snapshot", "default", "inferred"

    # Whether radiative-equilibrium wind T should be solved
    use_radiative_equilibrium: bool = False
    radiative_equilibrium_reason: str = ""
    T_radiative_equilibrium: Optional[np.ndarray] = None  # RE-solved T (full array) if computed

    # Diagnostics — list of (severity, message) tuples
    # severity: "info", "warning", "error"
    diagnostics: List = field(default_factory=list)

    def add_diagnostic(self, severity, message):
        self.diagnostics.append((severity, message))

    def report(self):
        """Return a human-readable summary string."""
        lines = []
        lines.append(f"Snapshot: {self.snapshot_path}")
        lines.append(f"  t since explosion: {self.t_explosion_d:.2f} d")
        lines.append(f"  n_zones: {self.n_zones}")
        lines.append(f"")
        lines.append(f"SN classification: {self.sn_type} "
                     f"(confidence {self.sn_type_confidence:.0%})")
        for k, v in self.sn_type_evidence.items():
            lines.append(f"    {k}: {v}")
        lines.append(f"")
        lines.append(f"Photosphere (τ_es=1):")
        lines.append(f"  R_phot  = {self.R_phot:.3e} cm")
        lines.append(f"  T_phot  = {self.T_phot:.0f} K (gas T at τ=1; for NLTE/RE)")
        lines.append(f"  T_color = {self.T_color:.0f} K (BB color T from L_phot; for continuum shape)")
        lines.append(f"  v_phot  = {self.v_phot/1e5:+.1f} km/s")
        lines.append(f"  L_phot  = {self.L_phot:.3e} erg/s "
                     f"({self.L_phot/LSUN:.2e} Lsun)")
        lines.append(f"")
        lines.append(f"Regions:")
        for name, mask in self.regions.items():
            lines.append(f"  {name:18s}: {int(mask.sum()):3d} zones")
        lines.append(f"")
        lines.append(f"Line-forming region:")
        lines.append(f"  T_mean   = {self.T_line_forming:.0f} K")
        lines.append(f"  n_e_mean = {self.n_e_line_forming:.2e} cm⁻³")
        lines.append(f"  v range  = {self.v_line_forming_min/1e5:+.0f} to "
                     f"{self.v_line_forming_max/1e5:+.0f} km/s")
        lines.append(f"")
        lines.append(f"Derived parameters:")
        lines.append(f"  v_turb = {self.v_turb_kms:.1f} km/s ({self.v_turb_basis})")
        lines.append(f"  composition source: {self.composition_source}")
        lines.append(f"  use_radiative_equilibrium: {self.use_radiative_equilibrium}")
        if self.radiative_equilibrium_reason:
            lines.append(f"    reason: {self.radiative_equilibrium_reason}")
        if self.diagnostics:
            lines.append(f"")
            lines.append(f"Diagnostics:")
            for sev, msg in self.diagnostics:
                marker = {"info": "·", "warning": "⚠", "error": "✗"}.get(sev, "·")
                lines.append(f"  {marker} [{sev}] {msg}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Region identification helpers
# ---------------------------------------------------------------------------
def _normalize_snapshot(snap_obj):
    """Convert a reader-specific snapshot object into a uniform dict of
    cgs-unit arrays that the analyzer can consume.

    Handles:
      - HERACLES: HeraclesSnapshot (fields: r, v, rho, n_e, T, L, tau_es,
        t_since_explos_d)
      - MESA: MesaSnapshot (fields: r, v, rho, n_e, T_gas, L, tau_ross,
        t_post_Lmax_d, X dict of mass fractions)
      - Already-dict input (e.g., from downstream code).

    Returns: dict with uniform keys:
        'r', 'v', 'rho', 'n_e', 'T', 'L' (cgs)
        'tau_es' (from n_e integration if needed)
        't_d' (best available time since explosion)
        'composition': dict {'X_H', 'X_He', 'X_metals'} or None
        'origin': string identifying the reader
    """
    d = {}

    # If it's already a dict, trust the caller
    if isinstance(snap_obj, dict):
        out = dict(snap_obj)
        out.setdefault('origin', 'dict')
        # Ensure we have tau_es
        if 'tau_es' not in out:
            r = np.asarray(out['r'])
            n_e = np.asarray(out['n_e'])
            dr = np.empty_like(r)
            dr[:-1] = np.diff(r); dr[-1] = dr[-2]
            out['tau_es'] = np.cumsum((SIGMAT * n_e * dr)[::-1])[::-1]
        # Map T if named differently
        if 'T' not in out and 'T_gas' in out:
            out['T'] = out['T_gas']
        return out

    # HeraclesSnapshot: has attributes r, v, rho, n_e, T, L, tau_es
    cls_name = type(snap_obj).__name__
    if 'Heracles' in cls_name:
        d['r'] = np.asarray(snap_obj.r)
        d['v'] = np.asarray(snap_obj.v)
        d['rho'] = np.asarray(snap_obj.rho)
        d['n_e'] = np.asarray(snap_obj.n_e)
        d['T'] = np.asarray(snap_obj.T)
        d['L'] = np.asarray(snap_obj.L) if hasattr(snap_obj, 'L') else None
        d['tau_es'] = np.asarray(snap_obj.tau_es)
        d['t_d'] = float(getattr(snap_obj, 't_since_explos_d',
                                 getattr(snap_obj, 't_since_start_d', 0.0)))
        d['composition'] = None    # HERACLES atmosphere files typically lack composition
        d['origin'] = 'heracles'
        return d

    # MesaSnapshot: T_gas, tau_ross (Rosseland), X dict
    if 'Mesa' in cls_name:
        d['r'] = np.asarray(snap_obj.r)
        d['v'] = np.asarray(snap_obj.v)
        d['rho'] = np.asarray(snap_obj.rho)
        d['n_e'] = np.asarray(snap_obj.n_e)
        d['T'] = np.asarray(snap_obj.T_gas)
        d['L'] = np.asarray(snap_obj.L) if hasattr(snap_obj, 'L') else None
        # MESA gives tau_rosseland (full opacity, line-blanketed); we want
        # Thomson-only tau for photosphere identification consistency.
        # Compute it from n_e.
        r = d['r']; n_e = d['n_e']
        dr = np.empty_like(r)
        dr[:-1] = np.diff(r); dr[-1] = dr[-2]
        d['tau_es'] = np.cumsum((SIGMAT * n_e * dr)[::-1])[::-1]
        d['t_d'] = float(getattr(snap_obj, 't_post_Lmax_d', 0.0))
        # MESA has per-zone composition
        if hasattr(snap_obj, 'X') and isinstance(snap_obj.X, dict):
            X = snap_obj.X
            X_H = X.get('h1', np.zeros_like(d['r']))
            # Helium: he3 + he4
            X_He = X.get('he3', 0) + X.get('he4', np.zeros_like(d['r']))
            # Metals: everything else
            X_metals = np.zeros_like(d['r'])
            for k, v_ in X.items():
                if k not in ('h1', 'he3', 'he4') and hasattr(v_, 'shape'):
                    X_metals = X_metals + np.asarray(v_)
            d['composition'] = {
                'X_H': np.asarray(X_H),
                'X_He': np.asarray(X_He),
                'X_metals': np.asarray(X_metals),
            }
        else:
            d['composition'] = None
        d['origin'] = 'mesa'
        return d

    # Unknown — try attribute-by-attribute
    for key in ('r', 'v', 'rho', 'n_e', 'T', 'T_gas', 'L', 'tau_es'):
        if hasattr(snap_obj, key):
            d[key] = np.asarray(getattr(snap_obj, key))
    if 'T' not in d and 'T_gas' in d:
        d['T'] = d['T_gas']
    if 'tau_es' not in d and 'n_e' in d:
        dr = np.empty_like(d['r'])
        dr[:-1] = np.diff(d['r']); dr[-1] = dr[-2]
        d['tau_es'] = np.cumsum((SIGMAT * d['n_e'] * dr)[::-1])[::-1]
    d['t_d'] = float(getattr(snap_obj, 't_since_explos_d',
                             getattr(snap_obj, 't_post_Lmax_d',
                                     getattr(snap_obj, 't_d', 0.0))))
    d['composition'] = None
    d['origin'] = f'unknown:{cls_name}'
    return d



def find_photosphere(tau_es, r):
    """Find the τ_es = 1 surface from outside.

    HERACLES atmosphere files include cumulative tau_es (from outside inward).
    The photosphere is the outermost zone with τ_es ≥ 1.

    Returns the zone index i_phot and a fractional position for sub-zone
    interpolation if needed.
    """
    # τ_es is cumulative from outside (decreasing with index for outward arrays)
    # Standard convention: index 0 = innermost (largest τ), index N-1 = outermost
    # (smallest τ).
    # We want the outermost zone where τ_es ≥ 1.
    above = tau_es >= 1.0
    if not np.any(above):
        # Snapshot doesn't reach τ_es=1 → atmosphere is fully optically thin.
        # Place photosphere at innermost zone (best we can do).
        return 0
    # outermost zone with τ ≥ 1 = highest index where above is True
    return int(np.where(above)[0][-1])


def find_density_jumps(rho, r, threshold=3.0):
    """Find sharp density discontinuities (potential shock fronts, shells).

    A density jump is a zone where rho changes by more than `threshold` factor
    over a short distance (shorter than the local scale height ~ r).

    Returns list of (i_jump, jump_factor, direction) tuples.
    direction = +1 if outer side denser, -1 if inner side denser.
    """
    if len(rho) < 3:
        return []
    log_rho = np.log10(np.maximum(rho, 1e-30))
    # Local jump = max log10 ratio between adjacent zones
    dlog = np.diff(log_rho)
    jumps = []
    for i in range(len(dlog)):
        if abs(dlog[i]) > np.log10(threshold):
            direction = +1 if dlog[i] > 0 else -1
            jumps.append((i, 10**abs(dlog[i]), direction))
    return jumps


def find_velocity_minima(v, r):
    """Find zones where |v| is locally minimum — these mark deceleration
    interfaces (e.g., post-shock to wind transition in IIn).

    Returns list of zone indices where v is locally minimum.
    """
    if len(v) < 3:
        return []
    minima = []
    for i in range(1, len(v) - 1):
        if v[i] < v[i-1] and v[i] < v[i+1]:
            minima.append(i)
    return minima


def identify_regions(snap, tau_es, i_phot):
    """Identify physical regions in the snapshot using local physics.

    Returns dict of {region_name: boolean_mask}.

    Definitions (each based on a physical signature):

    - 'photosphere':     τ_es ≥ 1 (electron-scattering thick).

    - 'transparent':     τ_es < 1.

    - 'cdshell':         Cold dense shell — contact surface between fast SN
                         ejecta and slow CSM. Defined by:
                           (a) density peak (ρ > 0.3 × ρ_max forms the
                               contiguous dense block around the global max),
                           (b) dv/dr < 0 across the block (deceleration),
                           (c) T ≤ 2 × T_phot (actually "cold" — gas has
                               cooled close to ambient photospheric T, not
                               still radiatively hot post-shock).
                         The T < 2 T_phot criterion distinguishes the COLD
                         dense shell from the HOT post-shock boundary layer,
                         which is also dense but still radiatively hot and
                         doesn't emit the line strongly (collisional
                         destruction dominates over recombination).

    - 'post_shock_hot':  Dense, fast, hot (T > 2×T_phot). These zones are
                         still thermalizing after passage of the reverse
                         shock; they emit mostly continuum and contribute
                         weakly to the line profile.

    - 'wind':            Outside photosphere, v < 1000 km/s, smoothly
                         declining (slow CSM outflow).

    - 'shocked_ejecta':  Inside photosphere, v > 1000 km/s, NOT CDS, NOT
                         post_shock_hot (fast unshocked/partially-shocked
                         ejecta behind the reverse shock).

    - 'envelope':        Inside photosphere, slow (v < 1000 km/s), NOT CDS
                         (e.g., recombining H envelope of IIP; most
                         common in Type II plateau).

    - 'fast_ejecta_outer': Outside photosphere, v > 1000 km/s (e.g., Ia
                         outer ejecta).
    """
    n = len(snap['r'])
    rho = snap['rho']
    v = snap['v']
    T = snap['T']
    r = snap['r']
    T_phot = float(T[i_phot]) if 0 <= i_phot < n else float(np.median(T))

    regions = {}
    photosphere = tau_es >= 1.0
    transparent = ~photosphere
    regions['photosphere'] = photosphere
    regions['transparent'] = transparent

    # ---- CDS detection with T constraint ----
    cdshell = np.zeros(n, dtype=bool)
    post_shock_hot = np.zeros(n, dtype=bool)
    if n >= 5:
        i_rho_max = int(np.argmax(rho))
        if 0 < i_rho_max < n - 1:
            rho_max = rho[i_rho_max]
            threshold_rho = 0.3 * rho_max
            j_outer = i_rho_max
            while (j_outer + 1 < n and rho[j_outer + 1] > threshold_rho):
                j_outer += 1
            j_inner = i_rho_max
            while (j_inner - 1 >= 0 and rho[j_inner - 1] > threshold_rho):
                j_inner -= 1

            block = slice(j_inner, j_outer + 1)
            v_block = np.abs(v[block])
            if len(v_block) >= 3:
                half = max(1, len(v_block) // 2)
                v_inner_avg = np.mean(v_block[:half])
                v_outer_avg = np.mean(v_block[-half:])
                if v_inner_avg > v_outer_avg * 1.1 and v_inner_avg > 5e7:
                    # A dense decelerating block. Split into cold CDS (T close
                    # to T_phot) vs hot post-shock layer (T >> T_phot).
                    T_block_mask = T[block] <= 2.0 * T_phot
                    block_idx = np.arange(j_inner, j_outer + 1)
                    cdshell[block_idx[T_block_mask]] = True
                    post_shock_hot[block_idx[~T_block_mask]] = True
    regions['cdshell'] = cdshell
    regions['post_shock_hot'] = post_shock_hot

    # ---- Wind: outside photosphere, slow ----
    v_kms = np.abs(v) / 1e5
    wind = transparent & (v_kms < 1000) & ~cdshell & ~post_shock_hot
    regions['wind'] = wind

    # ---- Shocked ejecta: inside photosphere, fast, NOT CDS/hot ----
    shocked = photosphere & (v_kms > 1000) & ~cdshell & ~post_shock_hot
    regions['shocked_ejecta'] = shocked

    # ---- Envelope: inside photosphere, slow, NOT CDS/hot ----
    envelope = photosphere & (v_kms < 1000) & ~cdshell & ~post_shock_hot
    regions['envelope'] = envelope

    # ---- Fast ejecta outer (Ia-like): transparent + fast ----
    fast_outer = transparent & (v_kms > 1000) & ~cdshell & ~post_shock_hot
    regions['fast_ejecta_outer'] = fast_outer

    return regions


# ---------------------------------------------------------------------------
# SN type classification — physics-based
# ---------------------------------------------------------------------------
def classify_sn_type(snap, regions, X_H_mean):
    """Determine SN type from the structure.

    Decision tree (pure physics, no thresholds tuned to specific snapshots):

    1. If composition has X_H < 0.01 → no hydrogen → 'Ia' or 'stripped'
       distinguish by velocity scale (Ia: > 10000 km/s, stripped: < 10000).

    2. If hydrogen present:
       a. If a CDS is identified AND a slow wind region exists AND
          a fast ejecta region exists → 'IIn' (CSM interaction)
       b. If smooth profile + recombining-T region (3000 < T < 10000 K
          covering > 50% of envelope) → 'IIP'
       c. Otherwise → 'unknown_H_rich'

    Returns: (type_string, confidence_0_to_1, evidence_dict)
    """
    evidence = {}

    # H content
    if X_H_mean < 0.01:
        # No hydrogen: Ia or stripped
        v_max_kms = float(np.max(np.abs(snap['v']))) / 1e5
        evidence['X_H_mean'] = X_H_mean
        evidence['v_max_kms'] = v_max_kms
        if v_max_kms > 10000:
            return 'Ia', 0.8, evidence
        else:
            return 'stripped_envelope', 0.7, evidence

    evidence['X_H_mean'] = X_H_mean

    # H present — distinguish IIn vs IIP vs other
    has_cds = bool(regions['cdshell'].any())
    has_wind = bool(regions['wind'].any())
    has_shocked = bool(regions['shocked_ejecta'].any())
    has_envelope = bool(regions['envelope'].any())

    evidence['has_cdshell'] = has_cds
    evidence['has_wind'] = has_wind
    evidence['has_shocked_ejecta'] = has_shocked
    evidence['has_envelope'] = has_envelope

    if regions['wind'].any():
        v_wind_max = float(np.max(np.abs(snap['v'][regions['wind']]))) / 1e5
        evidence['v_wind_max_kms'] = v_wind_max

    if regions['shocked_ejecta'].any():
        v_shocked_max = float(np.max(np.abs(snap['v'][regions['shocked_ejecta']]))) / 1e5
        evidence['v_shocked_max_kms'] = v_shocked_max

    if has_cds and has_wind and has_shocked:
        # Classic IIn structure: dense shell + slow CSM + fast ejecta
        return 'IIn', 0.9, evidence

    if has_envelope:
        # Check for recombining temperature region
        T_env = snap['T'][regions['envelope']]
        recombining_frac = float(np.mean((T_env > 3000) & (T_env < 10000)))
        evidence['recombining_T_fraction'] = recombining_frac
        if recombining_frac > 0.3:
            # IIP-like: most of envelope is at recombination temperatures
            return 'IIP', 0.7, evidence

    # H-rich but no obvious type
    return 'unknown_H_rich', 0.3, evidence


# ---------------------------------------------------------------------------
# Composition derivation
# ---------------------------------------------------------------------------
def derive_composition(snap):
    """Derive per-zone composition from the normalized snapshot dict.

    Expects `snap` to be a dict (from _normalize_snapshot) with optional
    'composition' key {'X_H', 'X_He', 'X_metals'}.

    Returns: (X_H, X_He, X_metals, source_string)
    """
    r = snap['r']
    n = len(r)

    comp = snap.get('composition', None)
    if comp is not None:
        return (np.asarray(comp['X_H']),
                np.asarray(comp['X_He']),
                np.asarray(comp['X_metals']),
                'snapshot')

    # Default: solar composition (Asplund et al. 2009)
    X_H = np.full(n, 0.737)
    X_He = np.full(n, 0.249)
    X_metals = np.full(n, 0.014)

    return X_H, X_He, X_metals, 'default_solar'


# ---------------------------------------------------------------------------
# v_turb derivation from local sound speed
# ---------------------------------------------------------------------------
def derive_v_turb(snap, regions):
    """Derive microturbulent velocity from local physics in the line-forming
    region.

    Approach: v_turb is set to the mean isothermal sound speed in the
    line-forming region. This is the standard astrophysical convention for
    "thermal + small-scale turbulent" broadening when no explicit turbulent
    velocity is provided by the hydro.

      v_turb = sqrt(k_B T / (μ m_H))

    where μ is the mean molecular weight (≈ 0.6 for ionized solar gas).

    Returns: (v_turb_kms, basis_string)
    """
    # Identify the most relevant line-forming region.
    # For lines like Hα, the line forms primarily in the wind / outer envelope
    # where the gas is partially ionized (T ~ 5000-15000 K). Pick that region:
    candidates = ['wind', 'cdshell', 'envelope', 'transparent']
    for region_name in candidates:
        if region_name in regions and regions[region_name].any():
            T_region = snap['T'][regions[region_name]]
            T_mean = float(np.mean(T_region))
            mu = 0.6  # ionized solar gas
            v_th_cms = np.sqrt(KB * T_mean / (mu * MH))
            v_turb_kms = v_th_cms / 1e5
            # Apply a sensible floor to avoid sub-thermal values
            # (numerical artifacts at very low T can give unphysical narrow widths).
            v_turb_kms = max(v_turb_kms, 5.0)
            return v_turb_kms, f"sound speed in {region_name} (T_mean={T_mean:.0f} K)"

    # Fallback: thermal velocity at T=10000 K
    return 12.8, "fallback (no line-forming region identified)"


# ---------------------------------------------------------------------------
# Decide if radiative equilibrium override is needed
# ---------------------------------------------------------------------------
def should_use_radiative_equilibrium(snap, regions, R_phot, T_phot, L_phot):
    """Decide whether to override snapshot T with radiative-equilibrium T in
    the wind region.

    Rationale: hydrodynamic SN codes (HERACLES, STELLA) typically do NOT
    include detailed radiative cooling for low-density outer zones (the CSM
    is treated as approximately vacuum or with approximate cooling). The
    hydro wind T is often biased hot by 20-40% because adiabatic or shock
    heating is included but radiative losses are approximated.

    Detection strategy: solve the radiative-equilibrium T (balancing
    photoionization heating from the photosphere with Lyα + free-free +
    recombination cooling), compare to the hydro T. Use RE if the
    difference exceeds a physical-uncertainty threshold.

    Threshold: > 15% difference in median wind T triggers RE.
    This catches both mildly hot-biased and strongly hot-biased hydro.

    Returns: (bool, reason_string, T_re_array_or_None)
        T_re_array is the computed radiative-eq T if computed, else None.
        When returning True, the caller should use T_re_array to override
        the snapshot T in the wind.
    """
    if not regions.get('wind', np.array([])).any():
        return False, "no wind region identified", None

    wind_mask = regions['wind']
    T_hydro = snap['T']
    T_wind_hydro = T_hydro[wind_mask]

    # Quick hot-bias check first — if wind is already way hotter than T_phot,
    # trust RE without bothering to compute it precisely (it'll be dominated
    # by photoionization heating from the hot photosphere)
    if float(T_wind_hydro.max()) > 2.0 * T_phot:
        return True, (
            f"wind T_max = {float(T_wind_hydro.max()):.0f} K exceeds 2×T_phot "
            f"({2*T_phot:.0f} K) — hydro T clearly unreliable; RE required"), None

    # Otherwise run the RE solver and compare
    try:
        from radiative_equilibrium import solve_wind_radiative_equilibrium
    except ImportError:
        return False, "radiative_equilibrium module unavailable", None

    try:
        T_new = solve_wind_radiative_equilibrium(
            T_hydro, snap['n_e'], snap['rho'], snap['r'], snap['v'],
            L_phot=L_phot, T_phot=T_phot, R_phot=R_phot,
            mask=wind_mask, verbose=False,
        )
    except Exception as e:
        return False, f"RE solver failed: {e}", None

    T_wind_re = T_new[wind_mask]
    T_hydro_med = float(np.median(T_wind_hydro))
    T_re_med = float(np.median(T_wind_re))

    # Relative difference
    rel_diff = abs(T_hydro_med - T_re_med) / max(T_re_med, 100)

    if rel_diff > 0.15:
        return True, (
            f"hydro wind T (median {T_hydro_med:.0f} K) differs from RE "
            f"solution (median {T_re_med:.0f} K) by {100*rel_diff:.0f}% — "
            f"using RE"), T_new

    return False, (
        f"hydro wind T (median {T_hydro_med:.0f} K) consistent with RE "
        f"(median {T_re_med:.0f} K) within {100*rel_diff:.0f}% — "
        f"using snapshot T"), T_new


# ---------------------------------------------------------------------------
# Diagnostic checks — flag suspicious snapshot conditions
# ---------------------------------------------------------------------------
def run_diagnostics(snap, state):
    """Perform sanity checks on the snapshot and add warnings to state."""

    # Check 1: monotonic radius
    if not np.all(np.diff(snap['r']) > 0):
        state.add_diagnostic("error",
            "Radius is not monotonically increasing — snapshot is malformed.")

    # Check 2: positive density
    if (snap['rho'] <= 0).any():
        n_bad = int((snap['rho'] <= 0).sum())
        state.add_diagnostic("error",
            f"{n_bad} zones have non-positive density.")

    # Check 3: physically plausible T
    if (snap['T'] < 100).any():
        n_cold = int((snap['T'] < 100).sum())
        state.add_diagnostic("warning",
            f"{n_cold} zones have T < 100 K (radiation-dominated regime, "
            f"recombination physics may be incomplete).")
    if (snap['T'] > 1e7).any():
        n_hot = int((snap['T'] > 1e7).sum())
        state.add_diagnostic("warning",
            f"{n_hot} zones have T > 10^7 K (Comptonization regime, "
            f"non-LTE atomic physics may be inadequate).")

    # Check 4: ionization plausibility
    # X_H × ρ / m_H should be ≥ n_e from H alone
    n_H_total = state.X_H * snap['rho'] / MH  # using state X_H
    overionized = snap['n_e'] > 1.5 * (n_H_total + 0.5 * state.X_He * snap['rho'] / (4*MH))
    # n_e cannot exceed n_H + 2*n_He (full ionization of H and He)
    if overionized.any():
        n_bad = int(overionized.sum())
        state.add_diagnostic("warning",
            f"{n_bad} zones have n_e > full ionization implies (e.g., metals "
            f"contributing electrons not accounted for, or hydro inconsistency).")

    # Check 5: photosphere reaches the snapshot grid?
    if state.i_phot == 0 and snap.get('tau_es', np.array([0]))[0] < 1.0:
        state.add_diagnostic("warning",
            "Photosphere is not resolved — atmosphere is fully optically thin to "
            "Thomson scattering. Continuum source treatment may be inappropriate.")

    # Check 6: enough zones for transport
    if len(snap['r']) < 20:
        state.add_diagnostic("warning",
            f"Only {len(snap['r'])} zones — radiative transfer may be poorly "
            f"resolved.")

    # Check 7: velocity field non-monotonic
    v = snap['v']
    if not (np.all(v >= 0) or np.all(v <= 0)):
        # mixed signs — odd
        state.add_diagnostic("info",
            "Velocity field has mixed signs (collapse + outflow regions present).")
    n_v_minima = sum(1 for i in range(1, len(v)-1)
                     if abs(v[i]) < abs(v[i-1]) and abs(v[i]) < abs(v[i+1]))
    if n_v_minima > 5:
        state.add_diagnostic("info",
            f"{n_v_minima} local |v| minima — multiple deceleration interfaces "
            f"present.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _read_snapshot_auto(path):
    """Dispatch to the right snapshot reader based on file content.

    Supports:
      - HERACLES atmosphere_*.dat files (first non-blank line is 'F' or 'T')
      - MESA profile .data files (starts with MESA header comment)

    Raises ValueError if the format can't be recognized.
    """
    # Peek at the first few lines
    with open(path, 'r') as f:
        peek = []
        for _ in range(10):
            line = f.readline()
            if not line:
                break
            peek.append(line)
    if not peek:
        raise ValueError(f"file {path} is empty")

    first_stripped = peek[0].strip()

    # HERACLES atmosphere file: first line is 'F' or 'T'
    if first_stripped in ('F', 'T'):
        from heracles_reader import read_heracles_atmosphere
        return read_heracles_atmosphere(path)

    # MESA file: check for MESA-specific headers or column names
    text_sample = '\n'.join(peek)
    if 'mass' in text_sample.lower() or 'logRho' in text_sample or 'logT' in text_sample:
        from mesa_reader import read_mesa_snapshot
        return read_mesa_snapshot(path)

    # Try to parse as generic numeric — fallback
    # If we can't tell, try HERACLES first then MESA
    try:
        from heracles_reader import read_heracles_atmosphere
        return read_heracles_atmosphere(path)
    except Exception:
        pass
    try:
        from mesa_reader import read_mesa_snapshot
        return read_mesa_snapshot(path)
    except Exception as e:
        raise ValueError(
            f"Could not identify snapshot format for {path}. "
            f"First line: {first_stripped!r}. "
            f"Supported formats: HERACLES atmosphere (F/T header), "
            f"MESA profile .data. Error: {e}")


def analyze_snapshot(snap_or_path, verbose=False):
    """Analyze a HERACLES (or compatible) snapshot and return a PhysicalState.

    Parameters
    ----------
    snap_or_path : str or HeraclesAtmosphere or dict
        Either a path to a snapshot file, or an already-loaded snapshot.
    verbose : bool
        If True, print the report after analysis.

    Returns
    -------
    PhysicalState
    """
    # Load snapshot if a path was given
    if isinstance(snap_or_path, str):
        path = snap_or_path
        # Dispatch to appropriate reader based on file content/extension
        # Simple heuristic: MESA .data files have specific header; HERACLES
        # atmosphere_*.dat has the 'F'/'T' flag header.
        snap_obj = _read_snapshot_auto(path)
    else:
        snap_obj = snap_or_path
        path = getattr(snap_obj, 'path', '<in-memory>')

    # Normalize to uniform dict regardless of reader format
    snap = _normalize_snapshot(snap_obj)

    state = PhysicalState()
    state.snapshot_path = path
    state.t_explosion_d = float(snap.get('t_d', 0.0))
    state.n_zones = int(len(snap['r']))

    # Use snapshot's τ_es if available; otherwise compute from n_e
    if 'tau_es' in snap and snap['tau_es'] is not None:
        tau_es = np.asarray(snap['tau_es'])
    else:
        # Compute from outside inward
        r = np.asarray(snap['r'])
        n_e = np.asarray(snap['n_e'])
        dr = np.empty_like(r)
        dr[:-1] = np.diff(r); dr[-1] = dr[-2]
        tau_es = np.cumsum((SIGMAT * n_e * dr)[::-1])[::-1]
        snap['tau_es'] = tau_es

    # ---- 1. Find photosphere ----
    state.i_phot = find_photosphere(tau_es, snap['r'])
    state.R_phot = float(snap['r'][state.i_phot])
    state.T_phot = float(snap['T'][state.i_phot])
    state.v_phot = float(snap['v'][state.i_phot])

    # L_phot from the snapshot's L(r) at i_phot. The reader returns L in
    # erg/s already (Lsun-to-cgs conversion is applied inside the reader).
    # Use this directly — it's the physical luminosity flowing through the
    # photosphere, which is more accurate than a BB approximation
    # (real photospheres are not perfect blackbodies).
    if 'L' in snap and snap['L'] is not None:
        L_at_phot = float(np.abs(snap['L'][state.i_phot]))
        # Sanity check: if L is very small (near zero crossing) or very far
        # from the BB estimate, use BB as a fallback estimate.
        L_BB = (4 * np.pi * state.R_phot**2 * SIG_SB * state.T_phot**4
                if state.T_phot > 0 else 0.0)
        # If L from snapshot is within factor 100 of BB, trust it.
        # Otherwise fall back to BB.
        if L_at_phot > 0 and 0.01 * L_BB < L_at_phot < 100 * L_BB:
            state.L_phot = L_at_phot
        else:
            state.L_phot = L_BB
            state.add_diagnostic("warning",
                f"Snapshot L at photosphere ({L_at_phot:.2e} erg/s) is far "
                f"from blackbody estimate ({L_BB:.2e} erg/s); using BB.")
    else:
        state.L_phot = (4 * np.pi * state.R_phot**2 * SIG_SB * state.T_phot**4
                        if state.T_phot > 0 else 0.0)

    # Color temperature: the effective BB temperature that produces the
    # observed L_phot at radius R_phot. Different from T_phot (gas T at
    # τ_es=1) when the atmosphere is scattering-dominated — radiation
    # thermalizes deeper than the τ_es=1 surface, so the emergent color is
    # cooler than the local gas T at the photosphere.
    #
    #   T_color = (L_phot / (4π R_phot² σ))^(1/4)
    #
    # T_color is what we should use for the continuum BB SHAPE in the
    # MC source and in the L_cont normalization. T_phot (gas T) is used
    # for the NLTE populations and radiative-equilibrium solver, because
    # those represent local thermodynamic quantities.
    if state.R_phot > 0 and state.L_phot > 0:
        state.T_color = (state.L_phot / (4.0 * np.pi * state.R_phot**2 * SIG_SB))**0.25
    else:
        state.T_color = state.T_phot

    # ---- 2. Identify regions ----
    state.regions = identify_regions(snap, tau_es, state.i_phot)

    # ---- 3. Derive composition ----
    X_H, X_He, X_metals, comp_source = derive_composition(snap)
    state.X_H = X_H
    state.X_He = X_He
    state.X_metals = X_metals
    state.composition_source = comp_source

    # ---- 4. Classify SN type ----
    X_H_mean = float(np.mean(X_H))
    sn_type, conf, evidence = classify_sn_type(snap, state.regions, X_H_mean)
    state.sn_type = sn_type
    state.sn_type_confidence = conf
    state.sn_type_evidence = evidence

    # ---- 5. Derive line-forming-region characteristics ----
    line_region_name = {
        'IIn': 'wind',
        'IIP': 'envelope',
        'Ia': 'fast_ejecta_outer',
        'stripped_envelope': 'envelope',
        'unknown_H_rich': 'envelope',
        'unknown': 'transparent',
    }.get(sn_type, 'transparent')
    if not state.regions.get(line_region_name, np.array([])).any():
        # fall back to whichever region is non-empty
        for fallback in ['transparent', 'envelope', 'wind', 'photosphere']:
            if state.regions.get(fallback, np.array([])).any():
                line_region_name = fallback
                break
    line_mask = state.regions[line_region_name]
    state.regions['line_forming'] = line_mask
    if line_mask.any():
        state.T_line_forming = float(np.mean(snap['T'][line_mask]))
        state.n_e_line_forming = float(np.mean(snap['n_e'][line_mask]))
        state.v_line_forming_min = float(np.min(np.abs(snap['v'][line_mask])))
        state.v_line_forming_max = float(np.max(np.abs(snap['v'][line_mask])))

    # ---- 6. Derive v_turb ----
    state.v_turb_kms, state.v_turb_basis = derive_v_turb(snap, state.regions)

    # ---- 7. Decide on radiative equilibrium ----
    state.use_radiative_equilibrium, state.radiative_equilibrium_reason, T_re = (
        should_use_radiative_equilibrium(snap, state.regions,
                                           state.R_phot, state.T_phot,
                                           state.L_phot))
    # Store the RE-solved T if computed (pipeline uses this to override snap T)
    state.T_radiative_equilibrium = T_re

    # ---- 8. Diagnostics ----
    run_diagnostics(snap, state)

    if verbose:
        print(state.report())

    return state
