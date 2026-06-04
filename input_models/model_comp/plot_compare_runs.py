#!/usr/bin/env python3
"""
plot_compare_runs.py  --  Deliverable (2)
=========================================
Publication-quality 4-panel figure comparing several model runs against a
physical control parameter (M_CSM/M_ej, or E_SN).

For each selected model only THREE epochs are shown:
   * the PEAK-luminosity epoch (peak of L(Halpha) for Balmer, of the strongest
     He line for He)  -> drawn as a BOLD, filled marker;
   * one epoch ~one e-folding BEFORE peak (target t_peak - t_peak/e), and
   * one epoch ~one e-folding AFTER  peak (target t_peak + t_peak/e),
     -> drawn dimmer / more transparent.
The three epochs of a model are joined by a dotted line.

Panels (x-axis = M_CSM/M_ej, or E_SN with --xaxis esn)
   (a) top-left   : L(Halpha), L(Hbeta), L(Hgamma)         (He: 3 strongest)
   (b) top-right  : EW of the same lines
   (c) bottom-left: Halpha/Hbeta ratio                     (He: strongest/2nd)
   (d) bottom-right (chosen): TOTAL Balmer luminosity
                    L(Halpha)+L(Hbeta)+L(Hgamma)           (He: sum of the set)
                    -- the integrated H (or He) line budget, the cleanest
                    single scalar for how much the CSM reprocesses into lines.

USAGE
-----
  python plot_compare_runs.py --runs A1:runs/A1 A4:runs/A4 A7:runs/A7 \
         --species balmer --xaxis ratio --out cmp_balmer_ratio.png

  # emit all four figures (balmer/he x ratio/esn) at once:
  python plot_compare_runs.py --runs A4:runs/A4 A5:runs/A5 A7:runs/A7 --all \
         --outdir figs/

Each --runs entry is 'MODELID:path' (path = dir of prod_day*_lines.npz). If you
omit 'MODELID:', the directory name is used as the id and must match the model
table so its M_CSM/M_ej and E_SN can be looked up.
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import snline_postproc as sp

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "figure.constrained_layout.use": True,
})

XLABELS = {"ratio": r"M$_{\rm CSM}$ / M$_{\rm ej}$",
           "esn": r"E$_{\rm SN}$ [10$^{51}$ erg]"}
XCOL = {"ratio": "M_csm_over_Mej", "esn": "E_SN"}


def _parse_runs(items):
    out = []
    for it in items:
        if ":" in it and not os.path.isdir(it.split(":", 1)[0]):
            mid, path = it.split(":", 1)
        else:
            mid, path = os.path.basename(os.path.normpath(it)), it
        out.append((mid, path))
    return out


def _model_triplet(df, lines, anchor):
    """Return dict line-> {epoch_kind: (epoch, L, EW)} plus ratio/total per kind.

    Epoch kinds: 'before', 'peak', 'after' from the e-folding triplet on anchor.
    """
    tb, tp, ta = sp.efold_triplet(df, anchor)
    kinds = {"before": tb, "peak": tp, "after": ta}
    res = {"epochs": kinds, "lines": {}, "total": {}, "ratio": {}}
    for ln in lines:
        s = sp.line_series(df, ln).set_index("epoch_d")
        res["lines"][ln] = {}
        for k, t in kinds.items():
            if t in s.index:
                res["lines"][ln][k] = (float(s.loc[t, "L"]), float(s.loc[t, "EW"]),
                                        float(s.loc[t, "tau"]))
            else:
                res["lines"][ln][k] = (np.nan, np.nan, np.nan)
    for k in kinds:
        tot = np.nansum([res["lines"][ln][k][0] for ln in lines])
        res["total"][k] = tot
        a = res["lines"][lines[0]][k][0]
        b = res["lines"][lines[1]][k][0] if len(lines) > 1 else np.nan
        res["ratio"][k] = a / b if (b and np.isfinite(b) and b > 0) else np.nan
    return res


_KIND_STYLE = {"peak":  dict(ms=11, alpha=1.0, mew=1.0, zorder=5, fill=True),
               "before": dict(ms=7, alpha=0.40, mew=0.6, zorder=3, fill=True),
               "after":  dict(ms=7, alpha=0.40, mew=0.6, zorder=3, fill=True)}


def _scatter_triplet(ax, x, vals_by_kind, color, marker="o"):
    """Plot the 3 kinds at x (dotted-connected), peak bold, others dim."""
    order = ["before", "peak", "after"]
    ys = [vals_by_kind[k] for k in order]
    xs = [x, x, x]
    ax.plot(xs, ys, ":", color=color, lw=1.0, alpha=0.7, zorder=2)
    for k, y in zip(order, ys):
        if not np.isfinite(y):
            continue
        st = _KIND_STYLE[k]
        ax.plot(x, y, marker=marker, color=color, ms=st["ms"], alpha=st["alpha"],
                mfc=color if st["fill"] else "white", mec="k", mew=st["mew"],
                zorder=st["zorder"], ls="none")


def _jitter(xvals):
    """Add a small symmetric jitter to tied x so models don't overlap."""
    xvals = np.asarray(xvals, float)
    out = xvals.copy()
    rng = (np.nanmax(xvals) - np.nanmin(xvals)) or 1.0
    seen = {}
    groups = {}
    for i, x in enumerate(xvals):
        groups.setdefault(round(x, 9), []).append(i)
    for key, idxs in groups.items():
        if len(idxs) == 1:
            continue
        span = 0.02 * rng
        offs = np.linspace(-span, span, len(idxs))
        for j, i in enumerate(idxs):
            out[i] = xvals[i] + offs[j]
    return out


def make_figure(models, species, xaxis, model_table, outpath, he_n=3,
                ew_sign="emission_positive"):
    mt = model_table
    # per-model data
    recs = []
    for mid, path in models:
        df = sp.load_run(path)
        if species == "balmer":
            lines = [l for l in sp.BALMER if l in df["line"].unique()]
            anchor = "Halpha"
        else:
            lines = sp.strong_he_lines(df, n=he_n)
            anchor = lines[0]
        if len(lines) < 2:
            print(f"[compare] {mid}: <2 usable lines, skipping")
            continue
        if mid not in mt.index:
            print(f"[compare] {mid}: not in model table, skipping")
            continue
        x = float(mt.loc[mid, XCOL[xaxis]])
        recs.append(dict(mid=mid, x=x, lines=lines,
                         trip=_model_triplet(df, lines, anchor)))
    if not recs:
        raise SystemExit("No plottable models.")

    # consistent line set / colours from the first model
    line_set = recs[0]["lines"]
    xj = _jitter([r["x"] for r in recs])
    for r, xx in zip(recs, xj):
        r["xj"] = xx

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) luminosity, (b) EW, per line
    for r in recs:
        for ln in r["lines"]:
            c = sp.LINE_COLOR.get(ln)
            Lk = {k: r["trip"]["lines"][ln][k][0] for k in ("before", "peak", "after")}
            Ek = {k: sp.ew_display(r["trip"]["lines"][ln][k][1], ew_sign)
                  for k in ("before", "peak", "after")}
            _scatter_triplet(ax[0, 0], r["xj"], Lk, c)
            _scatter_triplet(ax[0, 1], r["xj"], Ek, c)
        # (c) ratio, (d) total -- one colour per model-quantity
        Rk = r["trip"]["ratio"]
        Tk = r["trip"]["total"]
        _scatter_triplet(ax[1, 0], r["xj"], Rk, "#333333")
        _scatter_triplet(ax[1, 1], r["xj"], Tk, "#b2182b")
        # annotate model id at its peak marker in EVERY panel
        ref = r["lines"][0]
        ref_L = r["trip"]["lines"][ref]["peak"][0]
        ref_EW = sp.ew_display(r["trip"]["lines"][ref]["peak"][1], ew_sign)

        def _label(a, y):
            if y is not None and np.isfinite(y):
                a.annotate(r["mid"], (r["xj"], y), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=8, color="0.25",
                           fontweight="bold", clip_on=False)
        _label(ax[0, 0], ref_L)
        _label(ax[0, 1], ref_EW)
        _label(ax[1, 0], Rk["peak"])
        _label(ax[1, 1], Tk["peak"])

    ax[0, 0].set_yscale("log"); ax[0, 0].set_ylabel(r"L$_{\rm line}$ [erg s$^{-1}$]")
    ax[0, 0].set_title("(a) line luminosity")
    ax[0, 1].set_ylabel(sp.ew_axis_label(ew_sign)); ax[0, 1].set_title("(b) equivalent width")
    num, den = line_set[0], line_set[1]
    ax[1, 0].set_ylabel(f"{sp.PRETTY.get(num,num)}/{sp.PRETTY.get(den,den)}")
    ax[1, 0].set_title("(c) line ratio")
    if species == "balmer":
        ax[1, 0].axhline(2.86, ls=":", color="0.5", lw=1.0)
        ax[1, 0].annotate("case-B", (ax[1, 0].get_xlim()[0], 2.86), fontsize=7,
                          color="0.5", va="bottom")
    ax[1, 1].set_yscale("log")
    tot_lbl = "Balmer" if species == "balmer" else "He"
    ax[1, 1].set_ylabel(rf"$\Sigma$ L({tot_lbl}) [erg s$^{{-1}}$]")
    ax[1, 1].set_title(f"(d) total {tot_lbl} luminosity")
    for a in ax.flat:
        a.set_xlabel(XLABELS[xaxis])

    # legends: lines (colour) + epoch-kind (marker style)
    line_handles = [Line2D([0], [0], marker="o", ls="", color=sp.LINE_COLOR.get(l),
                           mec="k", label=sp.PRETTY.get(l, l)) for l in line_set]
    kind_handles = [
        Line2D([0], [0], marker="o", ls="", color="0.3", ms=11, mec="k",
               label="peak L (bold)"),
        Line2D([0], [0], marker="o", ls="", color="0.3", ms=7, alpha=0.4, mec="k",
               label="∓ 1 e-folding (dim)"),
        Line2D([0], [0], ls=":", color="0.3", label="same model"),
    ]
    ax[0, 0].legend(handles=line_handles, frameon=False, title="lines")
    ax[0, 1].legend(handles=kind_handles, frameon=False)

    fig.suptitle(f"Cross-model comparison — {tot_lbl} lines vs "
                 f"{'M_CSM/M_ej' if xaxis=='ratio' else 'E_SN'}   "
                 f"(peak ∓ e-folding)", fontsize=14)
    fig.savefig(outpath)
    print(f"[plot_compare_runs] wrote {outpath}  "
          f"(models={[r['mid'] for r in recs]}, lines={line_set})")
    return outpath


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="MODELID:path entries (or bare paths)")
    p.add_argument("--species", choices=["balmer", "he"], default="balmer")
    p.add_argument("--xaxis", choices=["ratio", "esn"], default="ratio")
    p.add_argument("--he-n", type=int, default=3)
    p.add_argument("--ew-sign", choices=["emission_positive", "physical"],
                   default="emission_positive",
                   help="EW display sign (default emission>0, matches production).")
    p.add_argument("--model-table", default=None)
    p.add_argument("--all", action="store_true",
                   help="emit all 4 figures: {balmer,he} x {ratio,esn}")
    p.add_argument("--out", default=None)
    p.add_argument("--outdir", default=".")
    args = p.parse_args(argv)

    models = _parse_runs(args.runs)
    mt = sp.model_table(args.model_table)
    os.makedirs(args.outdir, exist_ok=True)

    if args.all:
        outs = []
        for sptype in ("balmer", "he"):
            for xax in ("ratio", "esn"):
                op = os.path.join(args.outdir, f"compare_{sptype}_{xax}.png")
                outs.append(make_figure(models, sptype, xax, mt, op, he_n=args.he_n, ew_sign=args.ew_sign))
        return outs
    op = args.out or os.path.join(args.outdir,
                                  f"compare_{args.species}_{args.xaxis}.png")
    return make_figure(models, args.species, args.xaxis, mt, op, he_n=args.he_n, ew_sign=args.ew_sign)


if __name__ == "__main__":
    main()
