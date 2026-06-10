"""
metal_cloudy.py — Cloudy (Tier-2) absolute metal-line luminosities
================================================================================

FUTURE_WORK P2 item 5, Tier-2 upgrade. Where metal_nlte (Tier-1, CHIANTI) gives
authoritative *line emissivities* but still relies on OUR photoionization ion
balance and a single-shot Sobolev β for resonance escape, this module hands the
whole problem to **Cloudy** (Ferland et al.): a self-consistent photoionization +
multi-level NLTE + **resonance-line radiative-transfer** solve. That closes the
two approximations Tier-1 leaves open — the ion balance and the resonance-line
escape (C IV 1549, C III] 1909) — and returns proper ABSOLUTE line luminosities.

Division of labour (Tier-2):
  • Cloudy   → absolute line LUMINOSITIES (ionization + NLTE + resonance RT).
  • our MC   → the velocity-resolved PROFILE shape (Cloudy is static, no v-field):
               metal_lines keeps the peel-off transport and just rescales the
               unit-area shape to Cloudy's L_line.

The pipeline NEVER depends on Cloudy: `cloudy_available()` gates everything, and
any failure (not installed, abort, non-convergence, parse error) returns an empty
dict so the caller falls back to Tier-1 (CHIANTI) → provisional. Cloudy is used
opportunistically, per line, when it converges.

Install / config:
  • Cloudy must be compiled (see CLAUDE/FUTURE_WORK). The executable is located via
    $CLOUDY_EXE, else ~/c23.01/source/cloudy.exe, else `cloudy.exe` on PATH.
  • No data-path env is needed for a compiled-in-place Cloudy (it finds its data
    relative to the executable).

H-free handling (C-series, Ibn/Icn): Cloudy abundances are referenced to
hydrogen, but the He-star CSM is H-free. We anchor the density law on HELIUM
(the dominant species) and inject only a trace hydrogen floor (`H_FLOOR_HE`,
n_H/n_He) so Cloudy stays numerically stable; the He/C/O/Ne abundances are then
expressed relative to that floor. For H-rich models (A/B) the physical hydrogen
density exceeds the floor and the floor is inert.
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import tempfile
import numpy as np

# ---------------------------------------------------------------------------
# physical constants (cgs)
_M_U = 1.66053906660e-24        # atomic mass unit [g]
_SIGMA_SB = 5.670374419e-5      # Stefan-Boltzmann [erg cm^-2 s^-1 K^-4]
_A_MASS = {'H': 1.008, 'He': 4.0026, 'C': 12.011, 'O': 15.999, 'Ne': 20.180}

# trace-hydrogen floor for H-free gas: n_H / n_He (number). Small enough that H
# is spectroscopically negligible, large enough to keep Cloudy stable. The He/C/O
# abundances become log10(1/H_FLOOR_HE) ~ +3 dex super-solar — Cloudy handles this.
H_FLOOR_HE = 1.0e-3

# energy-conservation ceiling on a single metal line's luminosity, as a fraction
# of the photospheric bolometric L_phot. No single SN emission line carries more
# than a small fraction of L_bol (the energy is shared across the continuum and
# many lines). A Cloudy run that converges onto the wrong thermal branch can
# report a line at tens-of-percent of L_bol; such values are rejected (→ CHIANTI).
# This is an energy-partitioning bound, not a tuned parameter.
L_MAX_FRAC = 0.1

# ---------------------------------------------------------------------------
# our metal-line name -> list of Cloudy line labels to SUM (doublet / multiplet
# components are resolved individually by Cloudy and summed here). Label format is
# "{elem:<2}{ion:>2} {wavelength}A" exactly (two-char element, two-char ion stage).
CLOUDY_LINES = {
    'C_IV_1549':   ['C  4 1548.19A', 'C  4 1550.77A'],   # resonance doublet
    'C_III_1909':  ['C  3 1906.68A', 'C  3 1908.73A'],   # intercombination pair
    'C_III_4647':  ['C  3 4647.42A'],                    # recombination (best-guess label)
    'O_I_6300':    ['O  1 6300.30A'],                    # [O I]
    'O_III_5007':  ['O  3 5006.84A'],                    # [O III]
    'Ne_III_3869': ['Ne 3 3868.76A'],                    # [Ne III]
}

# all distinct Cloudy labels we request (the line-list file content)
_ALL_LABELS = []
for _lst in CLOUDY_LINES.values():
    for _lab in _lst:
        if _lab not in _ALL_LABELS:
            _ALL_LABELS.append(_lab)

_EXE_CACHE = None


def cloudy_exe():
    """Resolve the Cloudy executable: $CLOUDY_EXE → ~/c23.01/source/cloudy.exe →
    `cloudy.exe`/`cloudy` on PATH. Returns the path or None."""
    global _EXE_CACHE
    if _EXE_CACHE is not None:
        return _EXE_CACHE or None
    cand = []
    env = os.environ.get('CLOUDY_EXE')
    if env:
        cand.append(os.path.expanduser(env))
    cand.append(os.path.expanduser('~/c23.01/source/cloudy.exe'))
    for name in ('cloudy.exe', 'cloudy'):
        p = shutil.which(name)
        if p:
            cand.append(p)
    for p in cand:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            _EXE_CACHE = p
            return p
    _EXE_CACHE = ''
    return None


def cloudy_available():
    """True if a usable Cloudy executable is found."""
    return cloudy_exe() is not None


# ---------------------------------------------------------------------------
def _get(state, *names, default=None):
    for n in names:
        if isinstance(state, dict) and n in state:
            return state[n]
        if hasattr(state, n):
            return getattr(state, n)
    return default


def _comp(state, snap, *keys):
    """Per-zone mass fraction from state.composition / snap.composition / direct
    attribute, trying each key (isotope alias). Returns array or None."""
    for src in (state, snap):
        if src is None:
            continue
        comp = _get(src, 'composition')
        for k in keys:
            if isinstance(comp, dict) and k in comp:
                return np.asarray(comp[k], float)
            v = _get(src, k)
            if v is not None:
                return np.asarray(v, float)
    return None


def _shell_volumes(r):
    r = np.asarray(r, float)
    edge = np.empty(len(r) + 1)
    edge[1:-1] = 0.5 * (r[:-1] + r[1:])
    edge[0] = r[0]
    edge[-1] = r[-1] + (r[-1] - r[-2])
    return (4.0 / 3.0) * np.pi * (edge[1:] ** 3 - edge[:-1] ** 3)


def _scalar(x, default=0.0):
    if x is None:
        return float(default)
    a = np.asarray(x, float)
    return float(np.max(np.abs(a))) if a.size else float(default)


# ---------------------------------------------------------------------------
def build_deck(state, snap=None, include_xray=True, iterate=True,
               n_zone_cap=600):
    """Translate a merged STELLA state into a Cloudy input deck (string), plus the
    line-list file content. Returns (deck_str, linelist_str) or (None, None) if
    the required fields are unavailable.

    Geometry: spherical shell R_phot → R_out, density law from the snapshot.
    Incident field: photospheric blackbody (T_phot, L_phot) + optional shock
    bremsstrahlung (T_shock, L_X_brems). Abundances: He/C/O/Ne anchored on helium
    with a trace-H floor (H-free safe).
    """
    r = _get(state, 'r')
    if r is None and isinstance(snap, dict):
        r = snap.get('r')
    if r is None:
        return None, None
    r = np.asarray(r, float)
    nz = r.size

    def gz(*names):
        v = _get(state, *names)
        if v is None and isinstance(snap, dict):
            v = _get(snap, *names)
        return None if v is None else np.asarray(v, float)

    T = gz('T'); n_e = gz('n_e'); rho = gz('rho')
    if any(x is None for x in (T, n_e, rho)):
        return None, None

    R_phot = float(_get(state, 'R_phot_cm', 'R_phot', default=r[-1]))
    T_phot = float(_get(state, 'T_phot', 'T_color', default=float(T[-1])))
    L_phot = _get(state, 'L_phot', default=None)
    if L_phot is None:
        L_phot = 4.0 * np.pi * R_phot ** 2 * _SIGMA_SB * T_phot ** 4
    L_phot = float(L_phot)
    L_X = _scalar(_get(state, 'L_X_brems', default=0.0))
    T_shock = _scalar(_get(state, 'T_shock', default=0.0))

    # --- composition (mass fractions); fall back to pure-He if metals absent ---
    X_H = _comp(state, snap, 'h1', 'X_H', 'h')
    X_He = _comp(state, snap, 'he4', 'X_He', 'he')
    X_C = _comp(state, snap, 'c12', 'c')
    X_O = _comp(state, snap, 'o16', 'o')
    X_Ne = _comp(state, snap, 'ne20', 'ne')
    if X_He is None:
        X_He = np.full(nz, 1.0)            # assume He-dominated if unknown
    if X_H is None:
        X_H = np.zeros(nz)

    def _al(x):                            # align/broadcast to nz or None
        if x is None:
            return None
        x = np.asarray(x, float)
        return x if x.shape[0] == nz else None
    X_H, X_He = _al(X_H), _al(X_He)
    X_C, X_O, X_Ne = _al(X_C), _al(X_O), _al(X_Ne)
    if X_He is None:
        return None, None

    # number densities per zone
    def ndens(X, elem):
        if X is None:
            return np.zeros(nz)
        return rho * X / (_A_MASS[elem] * _M_U)
    n_He = ndens(X_He, 'He')
    n_C = ndens(X_C, 'C')
    n_O = ndens(X_O, 'O')
    n_Ne = ndens(X_Ne, 'Ne')
    n_H_phys = ndens(X_H, 'H')
    # hydrogen density law: physical H, floored to a trace fraction of He so a
    # H-free model still has a (negligible) reference hydrogen for Cloudy.
    n_H = np.maximum(n_H_phys, H_FLOOR_HE * n_He)
    n_H = np.where(np.isfinite(n_H) & (n_H > 0), n_H, 1e-30)

    # emission-measure weights (n_e^2 dV) → representative (radius-constant) abundances
    dV = _shell_volumes(r)
    w = np.asarray(n_e, float) ** 2 * dV
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        w = dV
    sw = w.sum()

    def abund(n_x):
        num = float(np.sum(n_x * w)) / sw
        den = float(np.sum(n_H * w)) / sw
        if num <= 0 or den <= 0:
            return None
        return np.log10(num / den)

    ab_He = abund(n_He)
    ab_C = abund(n_C)
    ab_O = abund(n_O)
    ab_Ne = abund(n_Ne)

    # --- density law: dlaw table radius (log r [cm], log n_H) ---
    # downsample to <= n_zone_cap points to keep Cloudy's table parser happy and
    # the run fast; keep monotonic increasing radius.
    order = np.argsort(r)
    rr = r[order]; nn = n_H[order]
    if rr.size > n_zone_cap:
        idx = np.unique(np.linspace(0, rr.size - 1, n_zone_cap).astype(int))
        rr = rr[idx]; nn = nn[idx]
    R_in = float(rr[0])
    R_out = float(rr[-1])
    if not (R_out > R_in > 0):
        return None, None

    lines = []
    A = lines.append
    A("title SNLT metal Tier-2")
    A("# === incident continuum ===")
    A(f"blackbody {T_phot:.6e} K")
    A(f"luminosity total {np.log10(max(L_phot, 1e-30)):.5f}")
    if include_xray and L_X > 0 and T_shock > 0:
        A("# shock bremsstrahlung X-rays")
        A(f"brems {T_shock:.6e} K")
        # normalise the second component over a hard band (0.1-100 keV ~ 7.35-7350 Ryd)
        A(f"luminosity {np.log10(max(L_X, 1e-30)):.5f} range 7.35 to 7350 Ryd")
    A("# === geometry ===")
    A(f"radius {np.log10(R_in):.6f} {np.log10(R_out):.6f}")
    A("sphere")
    A("# === density law (log r/cm  log n_H/cm^-3) ===")
    # Cloudy interpolates the table at zone-centre radii that can fall marginally
    # OUTSIDE [R_in, R_out]; pad both ends (flat-extrapolated density) so the
    # interpolation never goes out of range (else `tabval`/depth_table abort).
    lr = np.log10(np.maximum(rr, 1e-30))
    ln = np.log10(np.maximum(nn, 1e-30))
    pad = 0.02 * (lr[-1] - lr[0] + 1e-6)
    lr_tab = np.concatenate(([lr[0] - pad], lr, [lr[-1] + pad]))
    ln_tab = np.concatenate(([ln[0]], ln, [ln[-1]]))   # flat density at the pads
    # Cloudy's readLaw requires STRICTLY increasing radii ("Radii must be in
    # increasing order. Sorry."). STELLA piles many zones at near-identical radii
    # in the dense shock shell; at the %.6f log precision actually written, these
    # collapse to EXACT duplicates (e.g. 135 rows at log r=15.2253 for a 552-zone
    # day-10 deck) and abort the whole Cloudy run — which then silently degrades
    # the resonance line to its CHIANTI fallback and makes C IV flicker epoch to
    # epoch. Enforce strict monotonicity ON THE WRITTEN GRID: work in integer
    # micro-log units (1 unit = 10^-6 in log r, the %.6f resolution), force each
    # entry ≥ previous + 1 unit, convert back. The nudge is ≤ a few ppm in r per
    # collapsed row — physically negligible, and exact (no float drift).
    micro = np.round(lr_tab * 1.0e6).astype(np.int64)
    for i in range(1, micro.size):
        if micro[i] <= micro[i - 1]:
            micro[i] = micro[i - 1] + 1
    lr_tab = micro.astype(float) * 1.0e-6
    A("dlaw table radius")
    for lri, lni in zip(lr_tab, ln_tab):
        A(f"continue {lri:.6f} {lni:.6f}")
    A("end of dlaw")
    A("# === abundances (anchored on He; trace-H floor for H-free) ===")
    A("abundances all -30")                       # zero everything, then add ours
    if ab_He is not None:
        A(f"element abundance helium {ab_He:.4f}")
    if ab_C is not None:
        A(f"element abundance carbon {ab_C:.4f}")
    if ab_O is not None:
        A(f"element abundance oxygen {ab_O:.4f}")
    if ab_Ne is not None:
        A(f"element abundance neon {ab_Ne:.4f}")
    A("# === controls ===")
    A("no molecules")                             # hot ionized CSM; speeds + stabilises
    if iterate:
        # Resonance-line RT needs several iterations. Convergence test on the
        # densest C4 deck (552 zones): the CARBON lines we extract converge to ~1%
        # by iteration 4-5 (C IV 1549: 8.25e40→8.19e40→8.10e40 over iters 3/4/5),
        # while Cloudy's GLOBAL "did not converge" flag is tripped only by an
        # unrelated He-like subordinate line. max 6 lets the lines fully settle;
        # Cloudy stops early when converged, so the fast (sparse) epochs that
        # converge in 2-3 iterations cost nothing extra.
        A("iterate to convergence max 6")
    A("stop temperature 1000 K")
    A('save last line list ".lines" "cloudy.lines" emergent absolute')
    # per-zone LOCAL line emissivity [erg cm^-3 s^-1] vs depth — used to weight
    # the MC profile shape (Item 2: the shape's formation region then matches
    # Cloudy's self-consistent ionization, not our cruder Tier-1 ladder).
    A('save last line emissivity ".emis"')
    for lab in _ALL_LABELS:
        A(lab)
    A("end of line")
    deck = "\n".join(lines) + "\n"
    linelist = "\n".join(_ALL_LABELS) + "\n"
    meta = {'R_in': R_in, 'R_out': R_out}
    return deck, linelist, meta


# ---------------------------------------------------------------------------
def parse_line_list(path):
    """Parse a Cloudy `save line list` file → {cloudy_label: luminosity erg/s}.
    The file is two tab-separated rows: '#lineslist <labels...>' and
    'iteration N <values...>' (last iteration wins). Returns {} on any problem."""
    try:
        with open(path, 'r') as fh:
            rows = [ln.rstrip('\n') for ln in fh if ln.strip()]
    except Exception:
        return {}
    header = None
    data = None
    for ln in rows:
        cells = ln.split('\t')
        if cells and cells[0].lstrip().startswith('#lineslist'):
            header = cells[1:]
        elif cells and cells[0].lstrip().lower().startswith('iteration'):
            data = cells[1:]                       # keep overwriting → last iteration
    if header is None or data is None:
        return {}
    out = {}
    for lab, val in zip(header, data):
        try:
            out[lab.strip()] = float(val)
        except Exception:
            pass
    return out


def parse_emissivity(path, r_in):
    """Parse a Cloudy `save line emissivity` file → {cloudy_label: (radius_cm,
    emiss)} where emiss is the per-zone LOCAL line emissivity [erg cm^-3 s^-1] and
    radius = r_in + depth (depth measured from the illuminated inner face). The
    file is tab-separated: header '#depth <labels...>', then one row per zone.
    Returns {} on any problem."""
    try:
        with open(path, 'r') as fh:
            rows = [ln.rstrip('\n') for ln in fh if ln.strip()]
    except Exception:
        return {}
    if not rows:
        return {}
    header = rows[0].split('\t')
    if not header or not header[0].lstrip().startswith('#'):
        return {}
    labels = [h.strip() for h in header[1:]]
    depth = []
    cols = [[] for _ in labels]
    for ln in rows[1:]:
        cells = ln.split('\t')
        if len(cells) < 1 + len(labels):
            continue
        try:
            depth.append(float(cells[0]))
            for j in range(len(labels)):
                cols[j].append(float(cells[1 + j]))
        except Exception:
            continue
    if len(depth) < 2:
        return {}
    radius = float(r_in) + np.asarray(depth, float)
    out = {}
    for j, lab in enumerate(labels):
        out[lab] = (radius, np.asarray(cols[j], float))
    return out


def run_cloudy(deck, linelist, r_in=None, workdir=None, timeout=1800, keep=False):
    """Run Cloudy on `deck`, returning (lums, emiss, reason) where lums =
    {cloudy_label: L_erg_s}, emiss = {cloudy_label: (radius_cm, emiss)} (empty if
    r_in is None or no .emis), reason a status string. Robust to abort/timeout."""
    exe = cloudy_exe()
    if exe is None:
        return {}, {}, "cloudy executable not found"
    own_tmp = workdir is None
    wd = workdir or tempfile.mkdtemp(prefix="snlt_cloudy_")
    _preserve = False                       # keep the workdir if Cloudy failed
    try:
        with open(os.path.join(wd, "model.in"), "w") as fh:
            fh.write(deck)
        with open(os.path.join(wd, "cloudy.lines"), "w") as fh:
            fh.write(linelist)
        proc = None
        try:
            proc = subprocess.run([exe, "-r", "model"], cwd=wd,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            _preserve = True
            _dump_cloudy_failure(wd, "cloudy timed out", None)
            return {}, {}, "cloudy timed out"
        lpath = os.path.join(wd, ".lines")
        lums = parse_line_list(lpath)
        emiss = {}
        if r_in is not None:
            emiss = parse_emissivity(os.path.join(wd, ".emis"), r_in)
        if not lums:
            # surface the Cloudy stop reason for diagnostics, and PRESERVE the
            # deck + console so the per-epoch crash can be reproduced/inspected.
            reason = "no line-list output"
            outp = os.path.join(wd, "model.out")
            if os.path.isfile(outp):
                try:
                    with open(outp) as fh:
                        txt = fh.read()
                    for key in ("DISASTER", "ABORT", "did not converge",
                                "PROBLEM", "FATAL", "[Stop in", "insanity"):
                        i = txt.find(key)
                        if i >= 0:
                            reason = txt[i:i + 160].splitlines()[0].strip()
                            break
                except Exception:
                    pass
            _preserve = True
            _dump_cloudy_failure(wd, reason, proc)
            return {}, emiss, reason
        return lums, emiss, "ok"
    finally:
        if own_tmp and not keep and not _preserve:
            shutil.rmtree(wd, ignore_errors=True)


def _dump_cloudy_failure(wd, reason, proc):
    """On a Cloudy failure, copy the deck + console to a persistent, discoverable
    location (``$SNLT_CLOUDY_DEBUG`` or ./cloudy_failures/<timestamp-less id>/) and
    print a loud one-line pointer, so an intermittent per-epoch crash (works at
    one epoch, aborts at the next) can be reproduced and root-caused instead of
    silently degrading to the CHIANTI fallback."""
    try:
        base = os.environ.get("SNLT_CLOUDY_DEBUG") or \
            os.path.join(os.getcwd(), "cloudy_failures")
        os.makedirs(base, exist_ok=True)
        # deterministic id from the deck size + a slug of the reason (avoids
        # Python's per-process-randomized str hash); collisions just overwrite.
        mp = os.path.join(wd, "model.in")
        sz = os.path.getsize(mp) if os.path.isfile(mp) else 0
        slug = re.sub(r'[^A-Za-z0-9]+', '_', (reason or 'fail'))[:24].strip('_')
        dst = os.path.join(base, f"fail_{sz}_{slug}")
        os.makedirs(dst, exist_ok=True)
        for fn in ("model.in", "model.out", ".lines", ".emis"):
            src = os.path.join(wd, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst, fn.lstrip(".") or fn))
        if proc is not None:
            tail = "\n".join((proc.stdout or "").splitlines()[-25:])
            errtail = "\n".join((proc.stderr or "").splitlines()[-25:])
            with open(os.path.join(dst, "console.txt"), "w") as fh:
                fh.write(f"reason: {reason}\nreturncode: {proc.returncode}\n"
                         f"--- stdout tail ---\n{tail}\n"
                         f"--- stderr tail ---\n{errtail}\n")
        print(f"[cloudy] FAILURE preserved → {dst}  (reason: {reason})")
    except Exception as _e:
        print(f"[cloudy] (could not preserve failure deck: {_e})")


def metal_line_luminosities(state, snap=None, include_xray=True, iterate=True,
                            verbose=True, keep_workdir=False):
    """Top-level: build a deck from the STELLA state, run Cloudy, and return
    (lums, emiss) where:
      lums  = {our_line_name: L_line erg/s} — the self-consistent absolute, for
              every metal line Cloudy reports a finite positive luminosity for
              (doublet/multiplet components summed);
      emiss = {our_line_name: (radius_cm, emiss erg/cm^3/s)} — Cloudy's per-zone
              LOCAL line emissivity on Cloudy's own radial grid (components summed),
              used to weight the MC profile shape (Item 2).
    Returns ({}, {}) if Cloudy is unavailable or the run fails — the caller then
    uses Tier-1/CHIANTI.
    """
    if not cloudy_available():
        if verbose:
            print("[cloudy] executable not found — Tier-2 unavailable "
                  "(falling back to CHIANTI/provisional).")
        return {}, {}
    # Cloudy is only USED for the C IV 1549 resonance line (metal_lines routes
    # everything else to CHIANTI). C IV needs carbon, so if the gas has negligible
    # carbon (e.g. H-rich A/B models) there is nothing for Cloudy to fix — skip the
    # (slow) run entirely. Big speed-up with no loss.
    _XC = _comp(state, snap, 'c12', 'c')
    if _XC is not None and float(np.nanmax(np.abs(_XC))) < 5.0e-3:
        if verbose:
            print(f"[cloudy] ⟨X_C⟩<5e-3 (negligible carbon) — C IV is ~0, "
                  f"Cloudy skipped (CHIANTI handles the rest).")
        return {}, {}
    deck, linelist, meta = build_deck(state, snap, include_xray=include_xray,
                                      iterate=iterate)
    if deck is None:
        if verbose:
            print("[cloudy] insufficient state fields for a deck — Tier-2 skipped.")
        return {}, {}
    wd = None
    if keep_workdir:
        wd = tempfile.mkdtemp(prefix="snlt_cloudy_keep_")
    lums, emiss_lab, reason = run_cloudy(deck, linelist, r_in=meta.get('R_in'),
                                         workdir=wd, keep=keep_workdir)
    if not lums:
        if verbose:
            print(f"[cloudy] Tier-2 run did not yield lines ({reason}) — "
                  "falling back to CHIANTI/provisional.")
        return {}, {}
    # Energy-conservation ceiling. A single SN line cannot radiate more than a
    # modest fraction of the bolometric luminosity (the energy is spread over the
    # continuum + many lines; observed/theoretical single-line fractions are
    # ≲ few %). Cloudy can occasionally converge onto the WRONG thermal branch
    # (bistability) and report a line at tens-of-percent of L_bol — an artifact
    # that makes the time series flicker. We reject any line above L_MAX_FRAC·L_phot
    # so the caller falls back to the smooth CHIANTI tier for that line/epoch.
    R_phot = float(_get(state, 'R_phot_cm', 'R_phot', default=0.0))
    T_phot = float(_get(state, 'T_phot', 'T_color', default=0.0))
    L_phot = _get(state, 'L_phot', default=None)
    if L_phot is None and R_phot > 0 and T_phot > 0:
        L_phot = 4.0 * np.pi * R_phot ** 2 * _SIGMA_SB * T_phot ** 4
    L_phot = float(L_phot) if (L_phot and np.isfinite(L_phot)) else None
    L_ceiling = (L_MAX_FRAC * L_phot) if L_phot else None
    rejected = []
    out = {}
    out_emiss = {}
    for name, labels in CLOUDY_LINES.items():
        tot = 0.0
        any_hit = False
        rad_ref = None
        em_sum = None
        for lab in labels:
            if lab in lums:
                v = lums[lab]
                if np.isfinite(v) and v > 0:
                    tot += v
                    any_hit = True
            if lab in emiss_lab:                       # sum component emissivities
                rad, em = emiss_lab[lab]
                if rad_ref is None:
                    rad_ref = rad
                    em_sum = np.array(em, float)
                elif em.shape == em_sum.shape:
                    em_sum = em_sum + em
        if any_hit and tot > 0:
            if L_ceiling is not None and tot > L_ceiling:
                rejected.append((name, tot))
                continue                           # unphysical → caller uses CHIANTI
            out[name] = tot
            if rad_ref is not None and em_sum is not None and np.any(em_sum > 0):
                out_emiss[name] = (rad_ref, em_sum)
    if verbose:
        if keep_workdir and wd:
            print(f"[cloudy] workdir kept: {wd}")
        if rejected:
            print(f"[cloudy] REJECTED {len(rejected)} line(s) exceeding the "
                  f"energy-conservation ceiling ({L_MAX_FRAC:.0%}·L_phot="
                  f"{L_ceiling:.2e}) — Cloudy thermal-bistability artifact; "
                  f"falling back to CHIANTI for these:")
            for nm, val in rejected:
                print(f"        {nm:14s} Cloudy L={val:.3e} erg/s  (> ceiling)")
        print("[cloudy] Tier-2 absolute line luminosities (self-consistent "
              "photoionization + NLTE + resonance RT):")
        for name in CLOUDY_LINES:
            if name in out:
                emtag = " +per-zone emiss" if name in out_emiss else ""
                print(f"        {name:14s} L_line={out[name]:.3e} erg/s{emtag}")
            else:
                print(f"        {name:14s} (no Cloudy line / zero)")
    return out, out_emiss


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # tiny self-test: a synthetic He/C/O/Ne CSM shell → Cloudy round-trip.
    print("cloudy_available:", cloudy_available(), "->", cloudy_exe())
    nz = 60
    r = np.linspace(1.0e15, 5.0e15, nz)
    rho = 1.0e-14 * (r[0] / r) ** 2
    T = np.full(nz, 1.2e4)
    n_e = rho / (4.0 * _M_U)               # ~He+ density
    state = {
        'r': r, 'T': T, 'n_e': n_e, 'rho': rho,
        'R_phot_cm': float(r[0]), 'T_phot': 1.5e4,
        'L_phot': 1.0e43, 'L_X_brems': 1.0e41, 'T_shock': 1.0e7,
        'composition': {
            'he4': np.full(nz, 0.94), 'c12': np.full(nz, 0.04),
            'o16': np.full(nz, 0.015), 'ne20': np.full(nz, 0.005),
            'h1': np.zeros(nz),
        },
    }
    deck, ll, meta = build_deck(state)
    print("---- deck ----\n" + (deck or "(none)"))
    print("---- linelist ----\n" + (ll or "(none)"))
    print("---- meta ----", meta)
    res, emiss = metal_line_luminosities(state, keep_workdir=True)
    print("---- L_line ----", res)
    print("---- per-zone emissivity (n_zones per line) ----",
          {k: (v[0].size, float(np.max(v[1]))) for k, v in emiss.items()})
