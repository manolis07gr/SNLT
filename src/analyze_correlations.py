#!/usr/bin/env python3
"""
analyze_correlations.py  --  Deliverable (3)
============================================
Scan a set of model runs, build one feature row per model (at its peak epoch
by default), and look for structure between the PHYSICAL parameters of the CSM
-interaction model and the LINE properties it produces.

It does three things:
  1. Builds a master table (one row per model) and writes it to CSV:
        physical : M_csm, M_ej, M_csm/M_ej, E_SN, R_prog, f_Hrich, f_Herich
        lines    : L_Halpha, L_Hbeta, L_Hgamma, EW_Halpha,
                   Halpha/Hbeta, Halpha/Hgamma, total Balmer L,
                   L_He1 (strongest He), He1/He2 ratio, total He L
  2. Correlations: Pearson AND Spearman between every physical parameter and
     every line property; prints the strongest associations and saves a
     heatmap. H-line correlations are computed only over models that actually
     have hydrogen (L_Halpha above a floor), so the H-free C-series does not
     swamp them with zeros.
  3. PCA (via SVD, no sklearn): standardise the line-property block, find the
     principal components, report explained variance and loadings, and then
     correlate PC1/PC2 with the physical parameters AND fit a least-squares
     model of PC1 on the physical parameters -- i.e. "which combination of
     physical parameters best reproduces the line behaviour".

USAGE
-----
  python analyze_correlations.py --runs A1:runs/A1 A4:runs/A4 A5:runs/A5 \
         A6:runs/A6 A7:runs/A7 B4:runs/B4 ... --outdir analysis/

  # or point it at a parent folder of model dirs:
  python analyze_correlations.py --runs-glob 'runs/*' --outdir analysis/

Dependencies: numpy, pandas, matplotlib only.
"""
from __future__ import annotations
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import snline_postproc as sp

H_FLOOR = 1e30   # below this L_Halpha is treated as "no hydrogen"

PHYS = ["M_csm", "M_ej", "M_csm_over_Mej", "E_SN", "R_prog", "f_Hrich", "f_Herich"]
LINEPROPS = ["L_Halpha", "L_Hbeta", "L_Hgamma", "EW_Halpha",
             "Halpha_Hbeta", "Halpha_Hgamma", "L_Balmer_tot",
             "L_He1", "He1_He2", "L_He_tot"]


# --------------------------------------------------------------------------- #
def _discover(args):
    runs = []
    if args.runs:
        for it in args.runs:
            if ":" in it and not os.path.isdir(it.split(":", 1)[0]):
                mid, path = it.split(":", 1)
            else:
                mid, path = os.path.basename(os.path.normpath(it)), it
            runs.append((mid, path))
    if args.runs_glob:
        for path in sorted(glob.glob(args.runs_glob)):
            if glob.glob(os.path.join(path, "prod_day*_lines.npz")):
                runs.append((os.path.basename(os.path.normpath(path)), path))
    # de-dup by id
    seen, out = set(), []
    for mid, path in runs:
        if mid not in seen:
            out.append((mid, path)); seen.add(mid)
    return out


def _at_peak(df, anchor):
    tp = sp.peak_epoch(df, anchor)
    s = df[df["epoch_d"] == tp].set_index("line")
    return tp, s


def build_master(runs, mt, epoch_mode="peak"):
    rows = []
    for mid, path in runs:
        try:
            df = sp.load_run(path)
        except FileNotFoundError as e:
            print(f"[analyze] {mid}: {e}; skipping"); continue
        if mid not in mt.index:
            print(f"[analyze] {mid}: not in model table; skipping"); continue

        # anchor: Halpha if it has H, else strongest He
        try:
            ha = sp.line_series(df, "Halpha")["L"].to_numpy()
            ha = ha[np.isfinite(ha)]
            has_H = (ha.max() if ha.size else 0.0) > H_FLOOR
        except Exception:
            has_H = False
        he_lines = sp.strong_he_lines(df, n=2)
        anchor = "Halpha" if has_H else (he_lines[0] if he_lines else "Halpha")

        try:
            tp, s = _at_peak(df, anchor)
        except ValueError:
            print(f"[analyze] {mid}: no peak; skipping"); continue

        def L(ln):
            return float(s.loc[ln, "L"]) if ln in s.index else np.nan
        def EW(ln):
            return float(s.loc[ln, "EW"]) if ln in s.index else np.nan

        LHa, LHb, LHg = L("Halpha"), L("Hbeta"), L("Hgamma")
        he1 = he_lines[0] if len(he_lines) >= 1 else None
        he2 = he_lines[1] if len(he_lines) >= 2 else None
        LHe1 = L(he1) if he1 else np.nan
        LHe2 = L(he2) if he2 else np.nan
        he_all = sp.strong_he_lines(df, n=4)
        LHe_tot = np.nansum([L(x) for x in he_all])

        prow = mt.loc[mid]
        rows.append({
            "model": mid, "peak_epoch_d": tp, "anchor": anchor, "has_H": has_H,
            "M_csm": prow.M_csm, "M_ej": prow.M_ej,
            "M_csm_over_Mej": prow.M_csm_over_Mej, "E_SN": prow.E_SN,
            "R_prog": prow.R_prog,
            "f_Hrich": 1.0 if str(prow.CSM_comp).lower().startswith("h") else 0.0,
            "f_Herich": 1.0 if str(prow.CSM_comp).lower().startswith("he") else 0.0,
            "L_Halpha": LHa, "L_Hbeta": LHb, "L_Hgamma": LHg, "EW_Halpha": EW("Halpha"),
            "Halpha_Hbeta": LHa / LHb if (LHb and LHb > 0) else np.nan,
            "Halpha_Hgamma": LHa / LHg if (LHg and LHg > 0) else np.nan,
            "L_Balmer_tot": np.nansum([LHa, LHb, LHg]),
            "L_He1": LHe1, "He1_He2": LHe1 / LHe2 if (LHe2 and LHe2 > 0) else np.nan,
            "L_He_tot": LHe_tot, "he1_line": he1 or "", "he2_line": he2 or "",
        })
    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------------------- #
def correlations(master, outdir):
    H = master[master["has_H"]]
    out_lines = []
    # build correlation matrix phys x lineprops (Pearson & Spearman)
    def corr_block(dfb, method):
        rows = {}
        for lp in LINEPROPS:
            if lp not in dfb:
                continue
            # H-only props use H subset, He props use full
            base = H if lp.startswith(("L_Halpha", "L_Hbeta", "L_Hgamma",
                                       "EW_Halpha", "Halpha_", "L_Balmer")) else master
            y = base[lp].astype(float)
            r = {}
            for ph in PHYS:
                x = base[ph].astype(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
                    r[ph] = np.nan; continue
                if method == "spearman":
                    xv = pd.Series(x[ok]).rank().to_numpy()
                    yv = pd.Series(y[ok]).rank().to_numpy()
                else:
                    xv, yv = x[ok].to_numpy(), y[ok].to_numpy()
                r[ph] = float(np.corrcoef(xv, yv)[0, 1])
            rows[lp] = r
        return pd.DataFrame(rows).T  # rows=lineprops, cols=phys

    pear = corr_block(master, "pearson")
    spear = corr_block(master, "spearman")
    pear.to_csv(os.path.join(outdir, "corr_pearson.csv"))
    spear.to_csv(os.path.join(outdir, "corr_spearman.csv"))

    # heatmap (Pearson)
    fig, axx = plt.subplots(figsize=(9, 7))
    im = axx.imshow(pear.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axx.set_xticks(range(len(pear.columns))); axx.set_xticklabels(pear.columns, rotation=45, ha="right")
    axx.set_yticks(range(len(pear.index))); axx.set_yticklabels(pear.index)
    for i in range(pear.shape[0]):
        for j in range(pear.shape[1]):
            v = pear.to_numpy()[i, j]
            if np.isfinite(v):
                axx.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="k" if abs(v) < 0.6 else "w")
    fig.colorbar(im, label="Pearson r")
    axx.set_title("Line properties vs physical parameters (Pearson)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "correlation_heatmap.png"), dpi=200)
    plt.close(fig)

    # rank strongest associations
    stacked = pear.stack().dropna()
    stacked = stacked[stacked.abs() < 0.99999]  # drop trivial self
    top = stacked.reindex(stacked.abs().sort_values(ascending=False).index)
    out_lines.append("STRONGEST PEARSON CORRELATIONS (|r|):")
    for (lp, ph), v in top.head(12).items():
        out_lines.append(f"   {lp:16s} vs {ph:16s}  r = {v:+.3f}")
    return pear, spear, "\n".join(out_lines)


# --------------------------------------------------------------------------- #
def pca_block(master, outdir):
    feats = [c for c in LINEPROPS if c in master]
    X = master[feats].astype(float)
    # log-transform luminosities/EW magnitude (spans many decades), keep ratios linear
    Xt = X.copy()
    for c in feats:
        if c.startswith("L_") or c == "EW_Halpha":
            Xt[c] = np.log10(np.clip(X[c].abs(), 1e-30, None))
    Xt = Xt.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if Xt.shape[0] < 3:
        return "PCA skipped: <3 complete rows."
    mu, sd = Xt.mean(), Xt.std(ddof=0).replace(0, 1)
    Z = (Xt - mu) / sd
    U, S, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    evr = (S ** 2) / np.sum(S ** 2)
    loadings = pd.DataFrame(Vt.T, index=feats,
                            columns=[f"PC{i+1}" for i in range(len(S))])
    scores = pd.DataFrame(U * S, index=Xt.index,
                          columns=[f"PC{i+1}" for i in range(len(S))])

    # correlate PC1/PC2 with physical params, and regress PC1 ~ phys
    lines = ["", "PCA ON LINE-PROPERTY BLOCK (standardised; L,EW in log10):",
             f"   explained variance: " +
             ", ".join(f"PC{i+1}={evr[i]*100:.1f}%" for i in range(min(4, len(evr))))]
    lines.append("\n   PC1 loadings (which line properties define it):")
    for f, v in loadings["PC1"].sort_values(key=abs, ascending=False).items():
        lines.append(f"      {f:16s} {v:+.3f}")

    phys_ok = [p for p in PHYS if master.loc[Xt.index, p].std() > 0]
    lines.append("\n   PC1 correlation with physical parameters:")
    pc1 = scores["PC1"]
    corr_phys = {}
    for p in phys_ok:
        x = master.loc[Xt.index, p].astype(float)
        ok = np.isfinite(x) & np.isfinite(pc1)
        if ok.sum() >= 3 and x[ok].std() > 0:
            corr_phys[p] = float(np.corrcoef(x[ok], pc1[ok])[0, 1])
    for p, v in sorted(corr_phys.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"      {p:16s} r(PC1) = {v:+.3f}")

    # least-squares PC1 ~ standardized phys  -> best linear combination
    Xp = master.loc[Xt.index, phys_ok].astype(float)
    Xp = (Xp - Xp.mean()) / Xp.std(ddof=0).replace(0, 1)
    Xp = Xp.dropna(axis=1, how="any")
    if Xp.shape[1] >= 1 and Xp.shape[0] >= Xp.shape[1] + 1:
        A = np.column_stack([np.ones(len(Xp)), Xp.to_numpy()])
        beta, *_ = np.linalg.lstsq(A, pc1.to_numpy(), rcond=None)
        pred = A @ beta
        ss_res = np.sum((pc1.to_numpy() - pred) ** 2)
        ss_tot = np.sum((pc1.to_numpy() - pc1.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        lines.append(f"\n   Best linear predictor  PC1 ~ physical params  (R²={r2:.3f}):")
        for name, b in zip(["intercept"] + list(Xp.columns), beta):
            lines.append(f"      {name:16s} {b:+.3f}")
        terms = sorted(zip(Xp.columns, beta[1:]), key=lambda kv: -abs(kv[1]))
        dom = ", ".join(f"{n}" for n, _ in terms[:2])
        lines.append(f"   -> line behaviour is driven mainly by: {dom}")

    loadings.to_csv(os.path.join(outdir, "pca_loadings.csv"))
    scores.to_csv(os.path.join(outdir, "pca_scores.csv"))

    # scree + PC1-PC2 scatter
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(range(1, len(evr) + 1), evr * 100, color="#4575b4")
    ax[0].set_xlabel("PC"); ax[0].set_ylabel("explained variance [%]"); ax[0].set_title("scree")
    if scores.shape[1] >= 2:
        ax[1].scatter(scores["PC1"], scores["PC2"], s=60, edgecolor="k")
        for m in scores.index:
            ax[1].annotate(m, (scores.loc[m, "PC1"], scores.loc[m, "PC2"]),
                           fontsize=8, xytext=(4, 3), textcoords="offset points")
        ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].set_title("model scores")
        ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "pca_summary.png"), dpi=200)
    plt.close(fig)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", help="MODELID:path entries (or bare paths)")
    p.add_argument("--runs-glob", help="glob of parent dirs, e.g. 'runs/*'")
    p.add_argument("--model-table", default=None)
    p.add_argument("--outdir", default="analysis")
    args = p.parse_args(argv)

    runs = _discover(args)
    if not runs:
        raise SystemExit("No runs found (use --runs or --runs-glob).")
    os.makedirs(args.outdir, exist_ok=True)
    mt = sp.model_table(args.model_table)

    master = build_master(runs, mt)
    if master.empty:
        raise SystemExit("Master table empty.")
    master.to_csv(os.path.join(args.outdir, "master_table.csv"))

    _, _, corr_report = correlations(master, args.outdir)
    pca_report = pca_block(master, args.outdir)

    report = ("=" * 70 + "\n SNLT line / parameter-space analysis\n" + "=" * 70 +
              f"\n models: {list(master.index)}\n"
              f" (H-line stats use the {int(master['has_H'].sum())} H-bearing models)\n\n"
              + corr_report + "\n" + pca_report + "\n")
    with open(os.path.join(args.outdir, "correlations_report.txt"), "w") as f:
        f.write(report)
    print(report)
    print(f"[analyze] wrote master_table.csv, corr_*.csv, correlation_heatmap.png, "
          f"pca_loadings.csv, pca_scores.csv, pca_summary.png, correlations_report.txt "
          f"to {args.outdir}/")


if __name__ == "__main__":
    main()
