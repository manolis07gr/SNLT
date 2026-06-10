#!/usr/bin/env python3
"""grand_comparison.py — cross-model, cross-epoch spectral-feature ranking.

Ranks EVERY spectral feature (H, He, and the C/O/Ne metal lines) by corrected
line luminosity at every epoch of every model, so the output makes explicit
**which features dominate the spectrum in each epoch for each model** — the
"grand comparison" across the A/B/C grid.

Outputs (written to --outdir, default the SNLT root):
  • grand_ranking.csv       — full long table: model, epoch_d, rank, line,
                              species, L_line, EW, tau, frac_epoch (line's
                              share of the epoch's total line luminosity)
  • grand_metal_metrics.csv — per model-epoch metal diagnostics: total metal L,
                              dominant metal line + its L, metal/(all-line) frac,
                              C/O/Ne sub-totals
  • grand_comparison.png    — per-model panel: log L vs epoch for the strongest
                              features (which dominates, when), species-coloured
  • grand_dominant_map.png  — model × epoch map of the SINGLE dominant feature
                              (cell labelled + coloured by species)

Reads the same per-epoch prod_day*_lines.npz as the rest of the pipeline (all
lines, incl. metals, read dynamically). Run from the SNLT root:
    python src/grand_comparison.py
    python src/grand_comparison.py --models A1 A4 C4 --top 6
"""
from __future__ import annotations
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snline_postproc as sp   # data layer: load_run, PRETTY, LINE_COLOR, METAL

# representative epochs for the dominant-feature map columns (nearest available
# epoch is used per model); chosen to span breakout → interaction → decline.
_MAP_EPOCHS = [1, 3, 5, 10, 20, 30, 50, 80, 100]
_SPECIES_COLOR = {'H': '#d62728', 'He': '#ff7f0e', 'metal': '#8c564b'}


def _species(line: str) -> str:
    if line.startswith(('C_', 'O_', 'Ne_')) and not line.startswith('He_'):
        return 'metal'
    if line.startswith('He_'):
        return 'He'
    return 'H'


def _short(line: str) -> str:
    """Compact label for the dominant-feature map cells."""
    return (sp.PRETTY.get(line, line)
            .replace('] ', '').replace(' ', '').replace('$', '')
            .replace('\\alpha', 'a').replace('\\beta', 'b').replace('\\gamma', 'g')
            .replace('[', ''))


def discover_models(root: str, only: list[str] | None) -> list[tuple[str, str]]:
    out = []
    for d in sorted(glob.glob(os.path.join(root, 'input_models', '*'))):
        name = os.path.basename(d)
        if only and name not in only:
            continue
        if glob.glob(os.path.join(d, 'prod_day*_lines.npz')):
            out.append((name, d))
    return out


def build_ranking(models: list[tuple[str, str]]) -> pd.DataFrame:
    """Long ranked table over all (model, epoch, line)."""
    rows = []
    for name, d in models:
        try:
            df = sp.load_run(d)
        except Exception as e:
            print(f"[grand] {name}: load failed ({e}) — skipped")
            continue
        for ep, g in df.groupby('epoch_d'):
            g = g[np.isfinite(g['L']) & (g['L'] > 0.0)]
            if g.empty:
                continue
            tot = float(g['L'].sum())
            g = g.sort_values('L', ascending=False).reset_index(drop=True)
            for rank, r in g.iterrows():
                rows.append(dict(
                    model=name, epoch_d=float(ep), rank=int(rank) + 1,
                    line=r['line'], species=_species(r['line']),
                    L_line=float(r['L']), EW=float(r['EW']), tau=float(r['tau']),
                    frac_epoch=float(r['L']) / tot if tot > 0 else np.nan))
    return pd.DataFrame(rows)


def build_metal_metrics(rank_df: pd.DataFrame) -> pd.DataFrame:
    """Per model-epoch metal-line diagnostics."""
    rows = []
    for (model, ep), g in rank_df.groupby(['model', 'epoch_d']):
        tot_all = float(g['L_line'].sum())
        m = g[g['species'] == 'metal']
        tot_metal = float(m['L_line'].sum())
        dom = m.sort_values('L_line', ascending=False).head(1)
        dom_line = str(dom['line'].iloc[0]) if not dom.empty else ''
        dom_L = float(dom['L_line'].iloc[0]) if not dom.empty else np.nan

        def _elem(pref):
            return float(m[m['line'].str.startswith(pref)]['L_line'].sum())
        rows.append(dict(
            model=model, epoch_d=float(ep),
            L_metal_total=tot_metal,
            metal_frac_of_all=(tot_metal / tot_all if tot_all > 0 else np.nan),
            dominant_metal=dom_line, dominant_metal_L=dom_L,
            L_C=_elem('C_'), L_O=_elem('O_'), L_Ne=_elem('Ne_'),
            overall_rank_of_dominant_metal=(
                int(g[g['line'] == dom_line]['rank'].iloc[0]) if dom_line else -1)))
    return pd.DataFrame(rows).sort_values(['model', 'epoch_d'])


def plot_per_model(rank_df, models, top, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [m for m, _ in models]
    nm = len(names)
    ncols = min(3, nm)
    nrows = (nm + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows),
                             squeeze=False)
    mt = sp.model_table()
    for k, model in enumerate(names):
        ax = axes[k // ncols][k % ncols]
        g = rank_df[rank_df['model'] == model]
        if g.empty:
            ax.set_visible(False)
            continue
        # the strongest `top` lines for THIS model, by peak luminosity
        peak = g.groupby('line')['L_line'].max().sort_values(ascending=False)
        for line in peak.head(top).index:
            s = g[g['line'] == line].sort_values('epoch_d')
            ax.plot(s['epoch_d'], s['L_line'], '-o', ms=3, lw=1.4,
                    color=sp.LINE_COLOR.get(line, '0.5'),
                    label=sp.PRETTY.get(line, line))
        ax.set_yscale('log')
        ax.set_xlabel('t [d post-Lbol-max]', fontsize=8)
        ax.set_ylabel(r'L$_{line}$ [erg/s]', fontsize=8)
        sub = mt[mt['model'] == model] if 'model' in mt.columns else None
        extra = ''
        if sub is not None and not sub.empty:
            r = sub.iloc[0]
            extra = f"  (M_csm={r.get('M_csm','?')}, {r.get('CSM_comp','?')})"
        ax.set_title(f"{model}{extra}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=6, ncol=2, loc='best')
    for j in range(nm, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Grand comparison — strongest features per model "
                 f"(top {top} by peak L_line)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[grand] wrote {out_png}")


def plot_dominant_map(rank_df, models, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [m for m, _ in models]
    epochs = _MAP_EPOCHS
    fig, ax = plt.subplots(figsize=(1.15 * len(epochs) + 2.5, 0.5 * len(names) + 1.5))
    for yi, model in enumerate(names):
        g = rank_df[rank_df['model'] == model]
        avail = np.array(sorted(g['epoch_d'].unique())) if not g.empty else np.array([])
        for xi, ep in enumerate(epochs):
            if avail.size == 0:
                continue
            ej = float(avail[np.argmin(np.abs(avail - ep))])
            if abs(ej - ep) > max(0.35 * ep, 3.0):     # no nearby epoch → blank
                continue
            top1 = g[(g['epoch_d'] == ej) & (g['rank'] == 1)]
            if top1.empty:
                continue
            line = str(top1['line'].iloc[0]); spc = _species(line)
            ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1,
                                       color=_SPECIES_COLOR[spc], alpha=0.30))
            ax.text(xi, yi, _short(line), ha='center', va='center', fontsize=6.5)
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([f"{e}d" for e in epochs], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(-0.5, len(epochs) - 0.5); ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlabel('epoch (nearest available)', fontsize=9)
    ax.set_title("Dominant spectral feature  (strongest line per model × epoch)",
                 fontsize=12, fontweight='bold')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4)
               for c in _SPECIES_COLOR.values()]
    ax.legend(handles, list(_SPECIES_COLOR.keys()), title='species',
              fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[grand] wrote {out_png}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Grand cross-model feature ranking")
    p.add_argument('--root', default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    p.add_argument('--outdir', default=None)
    p.add_argument('--models', nargs='*', default=None,
                   help="restrict to these model names (default: all found)")
    p.add_argument('--top', type=int, default=6,
                   help="lines per panel in grand_comparison.png (default 6)")
    args = p.parse_args(argv)
    outdir = args.outdir or args.root

    models = discover_models(args.root, args.models)
    if not models:
        print("[grand] no model dirs with prod_day*_lines.npz found.")
        return
    print(f"[grand] {len(models)} model(s): {', '.join(m for m, _ in models)}")

    rank_df = build_ranking(models)
    if rank_df.empty:
        print("[grand] no rankable lines found.")
        return
    metal_df = build_metal_metrics(rank_df)

    rank_csv = os.path.join(outdir, 'grand_ranking.csv')
    metal_csv = os.path.join(outdir, 'grand_metal_metrics.csv')
    rank_df.sort_values(['model', 'epoch_d', 'rank']).to_csv(rank_csv, index=False)
    metal_df.to_csv(metal_csv, index=False)
    print(f"[grand] wrote {rank_csv}  ({len(rank_df)} rows)")
    print(f"[grand] wrote {metal_csv}  ({len(metal_df)} model-epochs)")

    plot_per_model(rank_df, models, args.top,
                   os.path.join(outdir, 'grand_comparison.png'))
    plot_dominant_map(rank_df, models,
                      os.path.join(outdir, 'grand_dominant_map.png'))

    # console summary: the single dominant feature per model at a few epochs
    print("\n=== dominant feature (rank 1) at representative epochs ===")
    for model, _ in models:
        g = rank_df[(rank_df['model'] == model) & (rank_df['rank'] == 1)]
        if g.empty:
            continue
        cells = []
        avail = np.array(sorted(g['epoch_d'].unique()))
        for ep in (5, 20, 50, 100):
            ej = avail[np.argmin(np.abs(avail - ep))]
            if abs(ej - ep) <= max(0.35 * ep, 3.0):
                ln = g[g['epoch_d'] == ej]['line'].iloc[0]
                cells.append(f"{ep}d:{_short(ln)}")
        print(f"  {model:4s}  " + "  ".join(cells))


if __name__ == '__main__':
    main(sys.argv[1:])
