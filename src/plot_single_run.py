#!/usr/bin/env python3
"""
plot_single_run.py  --  Deliverable (1)
=======================================
Publication-quality 4-panel figure showing the time evolution of the hydrogen
(Balmer) lines -- or the strongest helium lines -- for ONE model run.

Panels
------
  (a) top-left   : L_line(t) for the three lines              [log y]
  (b) top-right  : EW(t) for the three lines
  (c) bottom-left: line ratios vs t  (Halpha/Hbeta, Halpha/Hgamma;
                   for He: strongest/2nd, strongest/3rd) with case-B refs
  (d) bottom-right (chosen): decrement-vs-luminosity TRACK -- the
                   Halpha/Hbeta ratio plotted against L(Halpha), points
                   colour-coded by epoch. This exposes how the Balmer
                   decrement changes as the line brightens and fades, a
                   direct optical-depth / density-evolution diagnostic that a
                   plain ratio-vs-time panel does not show.

Optically-thick (tau>1) epochs are drawn with open markers, because their
absolute L is a factor-of-few estimate and their profile-integrated EW is not
reliable (single-shot kernel) -- the figure shows them but flags them.

USAGE
-----
  # whole available range, Balmer:
  python plot_single_run.py --run path/to/A4 --species balmer --out A4_balmer.png

  # a phase window 0-120 d, helium:
  python plot_single_run.py --run path/to/A7 --species he --t0 0 --t1 120

  # an explicit phase list:
  python plot_single_run.py --run path/to/A5 --species balmer --phases 0,1,5,10,20

'--run' is a directory containing prod_day*_lines.npz (or a glob).
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

CASE_B = {"Halpha/Hbeta": 2.86, "Halpha/Hgamma": 6.11}  # case-B, 1e4 K, ~thin


def _model_id(run: str) -> str:
    base = os.path.basename(os.path.normpath(run)).replace("*", "").replace(".npz", "")
    return base or "run"


def _phase_args(args):
    if args.phases:
        return None, None, [float(x) for x in args.phases.split(",")]
    return args.t0, args.t1, None


def _plot_L_or_EW(ax, df, lines, field, ylabel, logy=False):
    for ln in lines:
        s = sp.line_series(df, ln)
        if s.empty:
            continue
        y = s[field].to_numpy()
        thick = sp.saturated_mask(s)
        c = sp.LINE_COLOR.get(ln, None)
        ax.plot(s["epoch_d"], y, "-", color=c, lw=1.6, zorder=2,
                label=sp.PRETTY.get(ln, ln))
        # filled = thin (trustworthy), open = thick (factor-of-few)
        ax.scatter(s["epoch_d"][~thick], y[~thick], s=34, color=c,
                   edgecolor="k", linewidth=0.4, zorder=3)
        ax.scatter(s["epoch_d"][thick], y[thick], s=34, facecolor="white",
                   edgecolor=c, linewidth=1.1, zorder=3)
    ax.set_xlabel("epoch [d post-Lbol-max]")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False)


def _plot_ratios(ax, df, ratios, refs=None):
    for (num, den), color in zip(ratios, ["#222222", "#7b3294"]):
        r = sp.ratio_series(df, num, den)
        lbl = f"{sp.PRETTY.get(num,num)}/{sp.PRETTY.get(den,den)}"
        ax.plot(r["epoch_d"], r["ratio"], "o-", color=color, lw=1.6, ms=5, label=lbl)
        if refs and lbl_key(num, den) in (refs or {}):
            ax.axhline(refs[lbl_key(num, den)], ls=":", color=color, alpha=0.7,
                       lw=1.2, label=f"case-B {lbl}")
    ax.set_xlabel("epoch [d post-Lbol-max]")
    ax.set_ylabel("line luminosity ratio")
    ax.legend(frameon=False)


def lbl_key(num, den):
    return f"{num}/{den}"


def _plot_track(ax, df, num, den, anchor):
    """Bottom-right: ratio(num/den) vs L(anchor), colour-coded by epoch."""
    r = sp.ratio_series(df, num, den)
    a = sp.line_series(df, anchor)[["epoch_d", "L"]].rename(columns={"L": "Lanchor"})
    m = r.merge(a, on="epoch_d").dropna()
    m = m[m["Lanchor"] > 0].sort_values("epoch_d")
    if m.empty:
        ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
        return
    ax.plot(m["Lanchor"], m["ratio"], "-", color="0.6", lw=1.0, zorder=1)
    sctr = ax.scatter(m["Lanchor"], m["ratio"], c=m["epoch_d"], cmap="viridis",
                      s=55, edgecolor="k", linewidth=0.4, zorder=3)
    cb = ax.figure.colorbar(sctr, ax=ax, pad=0.01)
    cb.set_label("epoch [d]")
    ax.set_xscale("log")
    ax.set_xlabel(f"L({sp.PRETTY.get(anchor,anchor)}) [erg s$^{{-1}}$]")
    ax.set_ylabel(f"{sp.PRETTY.get(num,num)}/{sp.PRETTY.get(den,den)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="dir with prod_day*_lines.npz (or a glob)")
    p.add_argument("--species", choices=["balmer", "he"], default="balmer")
    p.add_argument("--t0", type=float, default=None, help="start epoch [d]")
    p.add_argument("--t1", type=float, default=None, help="end epoch [d]")
    p.add_argument("--phases", default=None, help="explicit list e.g. '0,1,5,10,20'")
    p.add_argument("--he-n", type=int, default=3, help="how many strongest He lines")
    p.add_argument("--model-table", default=None, help="optional model-property CSV")
    p.add_argument("--out", default=None, help="output PNG (auto if omitted)")
    args = p.parse_args(argv)

    df_all = sp.load_run(args.run)
    t0, t1, phases = _phase_args(args)
    df = sp.select_epochs(df_all, t0=t0, t1=t1, phases=phases)
    if df.empty:
        raise SystemExit("No epochs selected -- check --t0/--t1/--phases.")

    mid = _model_id(args.run)

    if args.species == "balmer":
        lines = [l for l in sp.BALMER if l in df["line"].unique()]
        ratios = [("Halpha", "Hbeta"), ("Halpha", "Hgamma")]
        track = ("Halpha", "Hbeta", "Halpha")
        refs = CASE_B
        sp_label = "Balmer (H)"
    else:
        lines = sp.strong_he_lines(df, n=args.he_n, t0=t0, t1=t1)
        if len(lines) < 2:
            raise SystemExit("Fewer than 2 He lines with positive L; cannot plot ratios.")
        ratios = [(lines[0], lines[1])]
        if len(lines) >= 3:
            ratios.append((lines[0], lines[2]))
        track = (lines[0], lines[1], lines[0])
        refs = None
        sp_label = "strongest He"

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.5))
    _plot_L_or_EW(ax[0, 0], df, lines, "L", r"L$_{\rm line}$ [erg s$^{-1}$]", logy=True)
    ax[0, 0].set_title("(a) line luminosity")
    _plot_L_or_EW(ax[0, 1], df, lines, "EW", "EW [Å]  (+ emission)")
    ax[0, 1].set_title("(b) equivalent width")
    _plot_ratios(ax[1, 0], df, ratios, refs=refs)
    ax[1, 0].set_title("(c) line ratios")
    _plot_track(ax[1, 1], df, *track)
    ax[1, 1].set_title("(d) decrement vs luminosity (colour = time)")

    # global caveat legend (filled=thin, open=thick)
    handles = [Line2D([0], [0], marker="o", color="0.3", ls="", mfc="0.3",
                      mec="k", label="optically thin (τ<1): reliable"),
               Line2D([0], [0], marker="o", color="0.3", ls="", mfc="white",
                      mec="0.3", label="optically thick (τ>1): factor-of-few")]
    ax[0, 0].legend(handles=ax[0, 0].get_legend().legend_handles + handles,
                    frameon=False, fontsize=8)

    mt = sp.model_table(args.model_table)
    sub = ""
    if mid in mt.index:
        row = mt.loc[mid]
        sub = (f"   M$_{{\\rm CSM}}$={row.M_csm} M$_\\odot$, "
               f"M$_{{\\rm CSM}}$/M$_{{\\rm ej}}$={row.M_csm_over_Mej:.3f}, "
               f"E$_{{\\rm SN}}$={row.E_SN}×10$^{{51}}$ erg, {row.CSM_comp} CSM")
    fig.suptitle(f"{mid} — {sp_label} line evolution{sub}", fontsize=14)

    out = args.out or f"{mid}_{args.species}_evolution.png"
    fig.savefig(out)
    print(f"[plot_single_run] wrote {out}  ({df['epoch_d'].nunique()} epochs, "
          f"lines={lines})")
    return out


if __name__ == "__main__":
    main()
