"""
snline_postproc.py
==================
Shared data-access layer for the SNLT line post-processing / plotting scripts.

Everything that reads pipeline output lives here so the plotting and analysis
scripts (plot_single_run.py, plot_compare_runs.py, analyze_correlations.py)
share one data contract.

WHAT IT READS
-------------
Per-epoch, per-line quantities come from the ``prod_day*_lines.npz`` files
written by phase5_runner. Each npz contains:
    line_names        (N,)  str   - canonical line keys
    lambda_rest       (N,)  float - rest wavelength [AA]
    L_line            (N,)  float - RAW single-shot line luminosity [erg/s]
    L_line_corrected  (N,)  float - corrected luminosity (budget / He-NLTE) [erg/s]
    EW / EW_corrected (N,)  float - equivalent width [AA]   (+ = emission here)
    tau_med           (N,)  float - median line optical depth
    L_cont_band       (N,)  float - continuum-band luminosity [erg/s]
    <line>__lambda, <line>__F_norm_corrected, ...  - per-line profile arrays

We ALWAYS prefer the *_corrected arrays (fall back to raw only if absent),
because those carry the recombination-budget / He-NLTE strengths. The epoch
is parsed from the filename ('day050' -> 50.0 d), since the npz has no epoch
scalar.

TRUST CAVEATS (propagated, not hidden)
--------------------------------------
* Saturated lines (tau >> 1, common in the dense-CSM phase): L is a
  factor-of-few estimate and the profile-integrated EW is unreliable. The
  loader exposes tau so callers can flag/grey these.
* He II lines are ~0 in these models (no He^2+); "strongest He" is therefore
  determined data-drivenly from L, not assumed.
* Late post-interaction epochs (continuum collapse) should be filtered by the
  caller; this module does not invent physics there.
"""

from __future__ import annotations
import os
import re
import glob
import numpy as np
import pandas as pd

C_KMS = 2.99792458e5

# Canonical line groups -------------------------------------------------------
BALMER = ["Halpha", "Hbeta", "Hgamma"]            # fixed, canonical trio
PASCHEN = ["Palpha", "Pbeta"]
HE_I = ["He_I_5876", "He_I_6678", "He_I_7065", "He_I_10830"]
HE_II = ["He_II_1640", "He_II_3203", "He_II_4686", "He_II_10124"]
# P2 #5 metal lines (C/O/Ne). Present only when run with --metal-lines; all
# post-processing reads line_names dynamically, so these are for grouping/colour.
METAL = ["C_IV_1549", "C_III_1909", "C_III_4647",
         "O_I_6300", "O_III_5007", "Ne_III_3869",
         "C_III_5696", "C_II_4267", "C_II_6580", "O_I_7774", "C_IV_5801"]

# Pretty labels for axes / legends
PRETTY = {
    "Halpha": r"H$\alpha$", "Hbeta": r"H$\beta$", "Hgamma": r"H$\gamma$",
    "Palpha": r"Pa$\alpha$", "Pbeta": r"Pa$\beta$",
    "He_I_5876": "He I 5876", "He_I_6678": "He I 6678",
    "He_I_7065": "He I 7065", "He_I_10830": "He I 10830",
    "He_II_1640": "He II 1640", "He_II_3203": "He II 3203",
    "He_II_4686": "He II 4686", "He_II_10124": "He II 10124",
    "C_IV_1549": "C IV 1549", "C_III_1909": "C III] 1909",
    "C_III_4647": "C III 4647", "O_I_6300": "[O I] 6300",
    "O_III_5007": "[O III] 5007", "Ne_III_3869": "[Ne III] 3869",
    "C_III_5696": "C III 5696", "C_II_4267": "C II 4267",
    "C_II_6580": "C II 6580", "O_I_7774": "O I 7774",
    "C_IV_5801": "C IV 5801/12",
}

# A consistent, color-blind-friendly palette keyed by line
LINE_COLOR = {
    "Halpha": "#d62728", "Hbeta": "#1f77b4", "Hgamma": "#2ca02c",
    "Palpha": "#9467bd", "Pbeta": "#8c564b",
    "He_I_5876": "#ff7f0e", "He_I_6678": "#e377c2",
    "He_I_7065": "#17becf", "He_I_10830": "#bcbd22",
    "He_II_4686": "#7f7f7f", "He_II_1640": "#555555",
    "He_II_3203": "#999999", "He_II_10124": "#aaaaaa",
    # metals: brown=C, blue=O, purple=Ne
    "C_IV_1549": "#8c564b", "C_III_1909": "#a0522d", "C_III_4647": "#654321",
    "O_I_6300": "#1f77b4", "O_III_5007": "#2a9df4",
    "Ne_III_3869": "#9467bd",
    "C_III_5696": "#b08968", "C_II_4267": "#7a4419",
    "C_II_6580": "#c8956c", "O_I_7774": "#4a90d9",
    "C_IV_5801": "#5d3a1a",
}


# -----------------------------------------------------------------------------
# Model property registry
# -----------------------------------------------------------------------------
# Prefilled from the model tables provided. M_ej defaults to (M_explosion - 1.4)
# for a ~1.4 Msun compact remnant; EDIT freely or override with --model-table CSV.
# Columns: model, M_csm[Msun], M_ej[Msun], E_SN[1e51 erg], R_prog[Rsun], CSM_comp
_DEFAULT_MODELS = [
    # --- A: RSG, H-rich envelope + H-rich CSM (99em_19-like) ---
    ("A1", 0.0, 16.4, 0.78, 603, "none"),
    ("A4", 0.2, 16.4, 0.78, 603, "H-rich"),
    ("A5", 0.5, 16.4, 0.78, 603, "H-rich"),
    ("A6", 1.0, 16.4, 0.78, 603, "H-rich"),
    ("A7", 2.0, 16.4, 0.78, 603, "H-rich"),
    # --- B: YSG, more stripped but H-rich (12A-like) ---
    ("B1", 0.0, 10.2, 0.28, 525, "none"),
    ("B4", 0.2, 10.2, 0.28, 525, "H-rich"),
    ("B5", 0.5, 10.2, 0.28, 525, "H-rich"),
    ("B6", 1.0, 10.2, 0.28, 525, "H-rich"),
    ("B7", 2.0, 10.2, 0.28, 525, "H-rich"),
    # --- C: stripped He-star, H-free + He-rich CSM (13bvn-like) ---
    ("C1", 0.0, 2.0, 0.80, 7.24, "none"),
    ("C4", 0.2, 2.0, 0.80, 7.24, "He-rich"),
    ("C5", 0.4, 2.0, 0.80, 7.24, "He-rich"),
    ("C6", 1.0, 2.0, 0.80, 7.24, "He-rich"),
    ("C7", 2.0, 2.0, 0.80, 7.24, "He-rich"),
    # --- expanded grid: E_SN variation at fixed family structure ---
    # A: M_ej=16.4, R=603, H-rich.  E_SN=1.0 (A8-11), 1.2 (A12-15)
    ("A8", 0.2, 16.4, 1.0, 603, "H-rich"), ("A9", 0.5, 16.4, 1.0, 603, "H-rich"),
    ("A10", 1.0, 16.4, 1.0, 603, "H-rich"), ("A11", 2.0, 16.4, 1.0, 603, "H-rich"),
    ("A12", 0.2, 16.4, 1.2, 603, "H-rich"), ("A13", 0.5, 16.4, 1.2, 603, "H-rich"),
    ("A14", 1.0, 16.4, 1.2, 603, "H-rich"), ("A15", 2.0, 16.4, 1.2, 603, "H-rich"),
    # B: M_ej=10.2, R=525, H-rich.  E_SN=0.5 (B8-11), 1.0 (B12-15)
    ("B8", 0.2, 10.2, 0.5, 525, "H-rich"), ("B9", 0.5, 10.2, 0.5, 525, "H-rich"),
    ("B10", 1.0, 10.2, 0.5, 525, "H-rich"), ("B11", 2.0, 10.2, 0.5, 525, "H-rich"),
    ("B12", 0.2, 10.2, 1.0, 525, "H-rich"), ("B13", 0.5, 10.2, 1.0, 525, "H-rich"),
    ("B14", 1.0, 10.2, 1.0, 525, "H-rich"), ("B15", 2.0, 10.2, 1.0, 525, "H-rich"),
    # C: M_ej=2.0, R=7.24, He-rich.  E_SN=0.95 (C8-11), 1.5 (C12-15)
    ("C8", 0.2, 2.0, 0.95, 7.24, "He-rich"), ("C9", 0.4, 2.0, 0.95, 7.24, "He-rich"),
    ("C10", 1.0, 2.0, 0.95, 7.24, "He-rich"), ("C11", 2.0, 2.0, 0.95, 7.24, "He-rich"),
    ("C12", 0.2, 2.0, 1.5, 7.24, "He-rich"), ("C13", 0.4, 2.0, 1.5, 7.24, "He-rich"),
    ("C14", 1.0, 2.0, 1.5, 7.24, "He-rich"), ("C15", 2.0, 2.0, 1.5, 7.24, "He-rich"),
]
_MODEL_COLS = ["model", "M_csm", "M_ej", "E_SN", "R_prog", "CSM_comp"]


def model_table(path: str | None = None) -> pd.DataFrame:
    """Return the model-property table as a DataFrame indexed by model id.

    If *path* is given it must be a CSV with at least the columns
    model, M_csm, M_ej, E_SN (R_prog, CSM_comp optional); it OVERRIDES /
    extends the built-in defaults (matching rows replaced by model id).
    """
    df = pd.DataFrame(_DEFAULT_MODELS, columns=_MODEL_COLS)
    if path:
        user = pd.read_csv(path)
        missing = {"model", "M_csm", "M_ej", "E_SN"} - set(user.columns)
        if missing:
            raise ValueError(f"--model-table missing columns: {sorted(missing)}")
        df = df[~df["model"].isin(user["model"])]
        df = pd.concat([df, user], ignore_index=True)
    df = df.drop_duplicates("model", keep="last").set_index("model")
    df["M_csm_over_Mej"] = df["M_csm"] / df["M_ej"]
    return df


# -----------------------------------------------------------------------------
# Per-epoch npz loading
# -----------------------------------------------------------------------------
def parse_epoch_d(path: str) -> float | None:
    """'.../prod_day050_lines.npz' -> 50.0 ; 'day000.8' -> 0.8."""
    m = re.search(r"day[_]?(-?\d+(?:\.\d+)?)", os.path.basename(path))
    return float(m.group(1)) if m else None


def _arr(d, key, fallback=None, n=None):
    if key in d.files:
        return np.asarray(d[key], float)
    if fallback is not None and fallback in d.files:
        return np.asarray(d[fallback], float)
    return np.full(n, np.nan) if n else None


def load_lines_npz(path: str) -> pd.DataFrame:
    """Load one prod_day*_lines.npz into a tidy long DataFrame.

    Columns: epoch_d, line, lambda_rest, L, EW, tau, L_cont_band, peak_F.
    L and EW are the CORRECTED values (raw used only if corrected absent).
    """
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["line_names"]]
    n = len(names)
    lam = _arr(d, "lambda_rest", n=n)
    L = _arr(d, "L_line_corrected", fallback="L_line", n=n)
    EW = _arr(d, "EW_corrected", fallback="EW", n=n)
    tau = _arr(d, "tau_med", n=n)
    Lc = _arr(d, "L_cont_band", n=n)

    # peak F (corrected) per line from the profile arrays, if present
    peakF = np.full(n, np.nan)
    for i, nm in enumerate(names):
        fk = f"{nm}__F_norm_corrected"
        if fk not in d.files:
            fk = f"{nm}__F_norm"
        if fk in d.files:
            F = np.asarray(d[fk], float)
            if F.size:
                peakF[i] = np.nanmax(F)

    ep = parse_epoch_d(path)
    return pd.DataFrame({
        "epoch_d": ep, "line": names, "lambda_rest": lam,
        "L": L, "EW": EW, "tau": tau, "L_cont_band": Lc, "peak_F": peakF,
    })


def load_run(run_dir: str, pattern: str = "prod_day*_lines.npz") -> pd.DataFrame:
    """Load all per-epoch npz in *run_dir* into one tidy long DataFrame.

    Returns columns: epoch_d, line, lambda_rest, L, EW, tau, L_cont_band, peak_F,
    sorted by (epoch_d, line). Raises if no files found.
    """
    files = sorted(glob.glob(os.path.join(run_dir, pattern)))
    if not files:
        # also accept a glob passed directly as run_dir
        files = sorted(glob.glob(run_dir))
    files = [f for f in files if parse_epoch_d(f) is not None]
    if not files:
        raise FileNotFoundError(
            f"No '{pattern}' files with a parseable epoch found in {run_dir!r}")
    frames = [load_lines_npz(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["epoch_d", "line"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Epoch selection
# -----------------------------------------------------------------------------
def select_epochs(df: pd.DataFrame, t0: float | None = None,
                  t1: float | None = None,
                  phases: list[float] | None = None,
                  tol: float = 1e-6) -> pd.DataFrame:
    """Filter a long df by either a [t0, t1] window or an explicit phase list.

    *phases* takes precedence; each requested phase snaps to the nearest
    available epoch (within *tol* days it is treated as exact).
    """
    epochs = np.sort(df["epoch_d"].unique())
    if phases is not None:
        keep = []
        for p in phases:
            j = int(np.argmin(np.abs(epochs - p)))
            keep.append(epochs[j])
        keep = sorted(set(keep))
        return df[df["epoch_d"].isin(keep)].copy()
    lo = -np.inf if t0 is None else t0 - tol
    hi = np.inf if t1 is None else t1 + tol
    return df[(df["epoch_d"] >= lo) & (df["epoch_d"] <= hi)].copy()


def line_series(df: pd.DataFrame, line: str) -> pd.DataFrame:
    """Return per-epoch series (epoch_d, L, EW, tau, peak_F) for one line."""
    s = df[df["line"] == line].sort_values("epoch_d")
    return s[["epoch_d", "L", "EW", "tau", "peak_F", "L_cont_band"]].reset_index(drop=True)


def peak_epoch(df: pd.DataFrame, line: str = "Halpha") -> float:
    """Epoch of maximum corrected L for *line*."""
    s = line_series(df, line).dropna(subset=["L"])
    s = s[s["L"] > 0]
    if s.empty:
        raise ValueError(f"No positive L for {line} to locate a peak.")
    return float(s.loc[s["L"].idxmax(), "epoch_d"])


def efold_triplet(df: pd.DataFrame, line: str = "Halpha"):
    """Return (t_before, t_peak, t_after) epochs.

    Following the requested definition, the spacing is one e-folding of the
    peak time: Delta = t_peak / e. The before/after targets are
    t_peak -/+ Delta, each snapped to the nearest AVAILABLE epoch.
    Degenerate snaps (e.g. t_peak at the series start) collapse gracefully.
    """
    tp = peak_epoch(df, line)
    epochs = np.sort(df[df["line"] == line]["epoch_d"].unique())
    delta = tp / np.e
    def nearest(t):
        return float(epochs[int(np.argmin(np.abs(epochs - t)))])
    tb = nearest(tp - delta)
    ta = nearest(tp + delta)
    return tb, tp, ta


# -----------------------------------------------------------------------------
# Strongest-line selection (data-driven; for He where He II ~ 0)
# -----------------------------------------------------------------------------
def rank_lines_by_median_L(df: pd.DataFrame, candidates: list[str],
                           n: int = 3, t0=None, t1=None) -> list[str]:
    """Rank *candidates* by median positive corrected-L over [t0,t1]; top n."""
    sub = select_epochs(df, t0=t0, t1=t1)
    med = {}
    for ln in candidates:
        v = sub.loc[sub["line"] == ln, "L"].to_numpy()
        v = v[np.isfinite(v) & (v > 0)]
        med[ln] = np.median(v) if v.size else 0.0
    ranked = sorted(candidates, key=lambda k: med[k], reverse=True)
    return [r for r in ranked if med[r] > 0][:n] or ranked[:n]


def strong_he_lines(df, n=3, prefer_optical=True, t0=None, t1=None):
    """Pick the n strongest He lines by median L (data-driven).

    He II lines are ~0 in these models, so this naturally returns He I lines.
    If prefer_optical, the IR He I 10830 is allowed but optical lines win ties.
    """
    cands = HE_I + HE_II
    ranked = rank_lines_by_median_L(df, cands, n=len(cands), t0=t0, t1=t1)
    if prefer_optical:
        opt = [l for l in ranked if "10830" not in l]
        ir = [l for l in ranked if "10830" in l]
        ranked = opt + ir
    return ranked[:n]


# -----------------------------------------------------------------------------
# Ratio helpers
# -----------------------------------------------------------------------------
def ratio_series(df: pd.DataFrame, num: str, den: str) -> pd.DataFrame:
    """Per-epoch luminosity ratio num/den on the common epoch grid."""
    a = line_series(df, num).rename(columns={"L": "La"})[["epoch_d", "La"]]
    b = line_series(df, den).rename(columns={"L": "Lb"})[["epoch_d", "Lb"]]
    m = pd.merge(a, b, on="epoch_d", how="inner")
    m["ratio"] = m["La"] / m["Lb"].replace(0, np.nan)
    return m[["epoch_d", "ratio"]]


def saturated_mask(df_line: pd.DataFrame, tau_thresh: float = 1.0):
    """Boolean per-epoch mask where the line is optically thick (tau>thresh)."""
    return df_line["tau"].to_numpy() > tau_thresh


# -----------------------------------------------------------------------------
# EW sign convention
# -----------------------------------------------------------------------------
# The npz stores EW under the physical definition  EW = int(1 - F/F_cont) dlambda,
# so EMISSION is NEGATIVE and ABSORPTION is POSITIVE. production_runner's headline
# Halpha EW uses the opposite (observer) convention, emission > 0. To avoid that
# mismatch in figures, callers pass an ew_sign and use ew_display().
EW_SIGN_DEFAULT = "emission_positive"   # match production headline + intuition


def ew_display(ew, sign: str = EW_SIGN_DEFAULT):
    """Map the stored (emission-negative) EW to the requested display sign.

    sign='emission_positive' -> negate (emission > 0; matches production headline)
    sign='physical'          -> leave as stored (emission < 0; EW=int(1-F/Fc)dl)
    """
    ew = np.asarray(ew, float)
    if sign == "physical":
        return ew
    if sign == "emission_positive":
        return -ew
    raise ValueError(f"unknown ew_sign {sign!r}")


def ew_axis_label(sign: str = EW_SIGN_DEFAULT) -> str:
    if sign == "physical":
        return r"EW [Å]   (physical: emission $<$ 0)"
    return r"EW [Å]   (emission $>$ 0)"
