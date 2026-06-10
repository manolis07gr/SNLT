#!/usr/bin/env python3
"""make_phase5_movie.py — multi-line (Phase 5) evolution movie.

Animates all H + He line profiles across a STELLA post-Lbol-max time series,
reading the per-epoch ``prod_day*_lines.npz`` files written by phase5_runner.

Design (addresses the two axis/label requirements):
  * x-axis per panel: a per-LINE range that encloses that line's full velocity
    departure across the WHOLE series (padded), so a broad early P-Cygni wing
    and a narrow late core are both shown without clipping, and the velocity
    scale of a given line stays stable frame-to-frame (no 13-panel jitter).
  * y-axis per panel: re-scaled EVERY frame to the current epoch's profile, so
    the whole shape (trough + peak) fills the panel at every epoch, with a floor
    so a ~flat line shows as flat rather than zoomed-in noise.
  * every frame is stamped with the supernova phase in days (figure suptitle),
    and each panel title carries that line's current L_line and EW.

CLI (called by production_runner --batch, or standalone):
    python make_phase5_movie.py 'prod_*_lines.npz' --out movie.mp4 --fps 3
"""
import sys
import os
import re
import glob
import argparse
import numpy as np

C_KMS = 299792.458

# Pretty species labels for the figure title.
_SPECIES_LABEL = {'all': 'H + He + metal', 'metal': 'metal (C/O/Ne)',
                  'he': 'He', 'h': 'H'}


def _load_lightcurve(path):
    """Load a STELLA mesa.lbol bolometric lightcurve → (t_days, log10_Lbol) or
    None. Format: header row then columns [time(d), L_ubvri, L_bol(log10), ...];
    the time origin is ~Lbol-max (peak near t=0), matching the snapshot epochs
    (day*_post_Lbol_max), so no offset is needed."""
    try:
        d = np.genfromtxt(path, skip_header=1)
        if d.ndim != 2 or d.shape[1] < 3:
            return None
        t = np.asarray(d[:, 0], float)
        logL = np.asarray(d[:, 2], float)
        m = np.isfinite(t) & np.isfinite(logL)
        if m.sum() < 2:
            return None
        order = np.argsort(t[m])
        return t[m][order], logL[m][order]
    except Exception:
        return None


def _find_lightcurve(files, explicit):
    """Resolve the mesa.lbol path: explicit --lightcurve, else next to the npzs."""
    if explicit:
        return _load_lightcurve(explicit)
    if files:
        cand = os.path.join(os.path.dirname(files[0]) or '.', 'mesa.lbol')
        if os.path.isfile(cand):
            return _load_lightcurve(cand)
    return None

# Preferred panel order (H first, then He I, then He II); any extra lines in the
# file but not listed here are appended in file order.
_PREFERRED = [
    'Halpha', 'Hbeta', 'Hgamma', 'Palpha', 'Pbeta',
    'He_I_5876', 'He_I_6678', 'He_I_7065', 'He_I_10830',
    'He_II_1640', 'He_II_3203', 'He_II_4686', 'He_II_10124',
]


def _epoch_from_name(path):
    m = re.search(r'day[_]?(\d+(?:\.\d+)?)', path)
    return float(m.group(1)) if m else 0.0


def _auto_vlim(dv, F, thresh=0.02, pad=0.10, min_hw=700.0):
    dv = np.asarray(dv, float); F = np.asarray(F, float)
    dev = np.abs(F - 1.0) > thresh
    if dev.any():
        lo, hi = float(dv[dev].min()), float(dv[dev].max())
    else:
        lo, hi = -min_hw, min_hw
    c = 0.5 * (lo + hi); hw = max(0.5 * (hi - lo), min_hw)
    p = pad * 2 * hw
    return c - hw - p, c + hw + p


def _auto_flim(F, pad=0.10, floor_lo=0.92, floor_hi=1.08):
    F = np.asarray(F, float)
    ymin = min(float(np.nanmin(F)), floor_lo)
    ymax = max(float(np.nanmax(F)), floor_hi)
    p = pad * (ymax - ymin + 1e-6)
    return max(0.0, ymin - p), ymax + p


def _load_frame(path):
    """Return (epoch_d, {line: dict(dv,F,L,EW,tau)}) from one npz."""
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d['line_names']]
    lam_rest = {n: float(v) for n, v in zip(names, np.asarray(d['lambda_rest'], float))}
    # prefer corrected strengths/profiles when present
    Lc = d['L_line_corrected'] if 'L_line_corrected' in d.files else d['L_line']
    EWc = d['EW_corrected'] if 'EW_corrected' in d.files else d['EW']
    tau = d['tau_med'] if 'tau_med' in d.files else np.full(len(names), np.nan)
    Lc = np.asarray(Lc, float); EWc = np.asarray(EWc, float); tau = np.asarray(tau, float)
    out = {}
    for i, n in enumerate(names):
        lam = d.get(f'{n}__lambda')
        Fk = f'{n}__F_norm_corrected'
        if Fk not in d.files:
            Fk = f'{n}__F_norm'
        F = d.get(Fk)
        if lam is None or F is None:
            continue
        lam = np.asarray(lam, float); F = np.asarray(F, float)
        dv = C_KMS * (lam - lam_rest[n]) / lam_rest[n]
        out[n] = dict(dv=dv, F=F, L=Lc[i], EW=EWc[i], tau=tau[i])
    return _epoch_from_name(path), out


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 5 multi-line evolution movie")
    p.add_argument('pattern', help="glob for per-epoch npz, e.g. 'prod_*_lines.npz'")
    p.add_argument('--out', default='batch_lines_evolution.mp4')
    p.add_argument('--fps', type=int, default=3)
    p.add_argument('--species', default='all',
                   choices=['all', 'metal', 'he', 'h'],
                   help="which lines to animate: 'all' (default), 'metal' "
                        "(C/O/Ne — P2 #5), 'he', or 'h'.")
    p.add_argument('--grid', action='store_true',
                   help="instead of a movie, save a STATIC grid PNG tiling the "
                        "selected lines (rows) across all epochs (columns). "
                        "--out should be a .png.")
    p.add_argument('--lightcurve', default=None,
                   help="path to a STELLA mesa.lbol bolometric lightcurve "
                        "(time[d], _, log10 L_bol). Default: auto-detect mesa.lbol "
                        "next to the npz files. Adds a lightcurve panel to the "
                        "movie with a moving dot at the current epoch.")
    args = p.parse_args(argv)

    files = sorted(glob.glob(args.pattern), key=_epoch_from_name)
    if len(files) < 1:
        print(f"[phase5-movie] no files match {args.pattern}")
        return
    frames = [_load_frame(f) for f in files]
    epochs = [e for e, _ in frames]

    # panel line order
    present = list(frames[0][1].keys())

    # optional species filter (P2 #5: metal-only movie, etc.)
    def _is_metal(n):
        return n.startswith(('C_', 'O_', 'Ne_')) and not n.startswith('He_')

    def _is_he(n):
        return n.startswith('He_')

    def _is_h(n):
        return n.startswith(('Halpha', 'Hbeta', 'Hgamma', 'Hdelta',
                             'Palpha', 'Pbeta', 'Pgamma'))
    if args.species != 'all':
        sel = {'metal': _is_metal, 'he': _is_he, 'h': _is_h}[args.species]
        present = [n for n in present if sel(n)]
        if not present:
            print(f"[phase5-movie] no '{args.species}' lines found in files")
            return

    order = [n for n in _PREFERRED if n in present] + \
            [n for n in present if n not in _PREFERRED]
    nlines = len(order)
    if nlines == 0:
        print("[phase5-movie] no line profiles found in files")
        return

    # per-LINE global x-range across all epochs (encloses full departure)
    xlim = {}
    for n in order:
        lo, hi = +1e9, -1e9
        for _, fr in frames:
            if n in fr:
                a, b = _auto_vlim(fr[n]['dv'], fr[n]['F'])
                lo, hi = min(lo, a), max(hi, b)
        xlim[n] = (lo, hi) if lo < hi else (-2000, 2000)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ---- static GRID mode: tile lines (rows) × epochs (columns) into one PNG ----
    if args.grid:
        nep = len(frames)
        fig, axes = plt.subplots(nlines, nep,
                                 figsize=(max(1.9 * nep, 4.0), max(1.5 * nlines, 3.0)),
                                 squeeze=False)
        for ri, n in enumerate(order):
            for ci, (ep, fr) in enumerate(frames):
                ax = axes[ri][ci]
                ax.axhline(1.0, color='gray', ls=':', lw=0.4)
                ax.axvline(0.0, color='gray', ls=':', lw=0.4)
                if n in fr:
                    ax.plot(fr[n]['dv'], fr[n]['F'], lw=0.7, color='C3'
                            if _is_metal(n) else ('C0' if _is_he(n) else 'C2'))
                    ax.set_xlim(*xlim[n])
                if ri == 0:
                    ax.set_title(f"{ep:g}d", fontsize=7)
                if ci == 0:
                    ax.set_ylabel(n.replace('_', ' '), fontsize=6.5)
                ax.tick_params(labelsize=5)
                if ri < nlines - 1:
                    ax.set_xticklabels([])
        fig.suptitle(f"{args.species}-line evolution grid  "
                     f"(rows = lines, columns = epoch [days]; F/F_cont vs Δv)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        out_png = args.out
        if not out_png.lower().endswith('.png'):
            out_png = out_png.rsplit('.', 1)[0] + '_grid.png'
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"[phase5-movie] Saved grid {out_png}  "
              f"({nlines} lines × {nep} epochs)")
        return

    from matplotlib.animation import FuncAnimation, PillowWriter

    # load the bolometric lightcurve (for the reference panel + moving dot)
    lc = _find_lightcurve(files, args.lightcurve)

    ncols = 5
    # reserve ONE extra cell for the lightcurve panel (if a lightcurve is found)
    n_cells = nlines + (1 if lc is not None else 0)
    nrows = (n_cells + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    sup = fig.suptitle('', fontsize=17, y=0.998, fontweight='bold')
    species_label = _SPECIES_LABEL.get(args.species, args.species)

    lines2d, titles = {}, {}
    for k, n in enumerate(order):
        ax = axes_flat[k]
        (ln,) = ax.plot([], [], 'C0-', lw=1.6)
        lines2d[n] = ln
        ax.axhline(1, color='gray', ls=':', lw=0.5)
        ax.axvline(0, color='gray', ls=':', lw=0.5)
        ax.set_xlim(*xlim[n])
        ax.set_xlabel('Δv [km/s]', fontsize=8)
        ax.set_ylabel('F / F_cont', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        titles[n] = ax.set_title(n, fontsize=9)

    # ---- lightcurve reference panel + moving epoch marker ----
    lc_dot = lc_vline = lc_txt = None
    if lc is not None:
        lc_ax = axes_flat[nlines]
        t_lc, logL_lc = lc
        lc_ax.plot(t_lc, logL_lc, '-', color='0.4', lw=1.4)
        lc_ax.set_xlabel('t [d post-Lbol-max]', fontsize=8)
        lc_ax.set_ylabel(r'log$_{10}$ L$_{bol}$ [erg/s]', fontsize=8)
        lc_ax.set_title('Bolometric lightcurve', fontsize=9)
        lc_ax.tick_params(labelsize=7)
        lc_ax.grid(alpha=0.3)
        # keep the x-range to the epochs actually shown (+ a little)
        emin, emax = min(epochs), max(epochs)
        lc_ax.set_xlim(min(emin, float(t_lc[0])) - 2,
                       max(emax, emin + 1) + 5)
        (lc_dot,) = lc_ax.plot([], [], 'o', color='crimson', ms=11, zorder=5)
        lc_vline = lc_ax.axvline(epochs[0], color='crimson', ls='--', lw=1.0,
                                 alpha=0.7)
        lc_txt = lc_ax.text(0.04, 0.06, '', transform=lc_ax.transAxes,
                            fontsize=11, fontweight='bold', color='crimson',
                            ha='left', va='bottom')
        first_empty = nlines + 1
    else:
        first_empty = nlines
    for j in range(first_empty, len(axes_flat)):
        axes_flat[j].set_visible(False)

    def update(idx):
        ep, fr = frames[idx]
        sup.set_text(f"{species_label} line evolution      "
                     f"t = {ep:.1f} d post-Lbol-max      "
                     f"(frame {idx+1}/{len(frames)})")
        arts = [sup]
        for n in order:
            ln = lines2d[n]
            ax = ln.axes
            if n in fr:
                dv, F = fr[n]['dv'], fr[n]['F']
                ln.set_data(dv, F)
                lo, hi = xlim[n]
                m = (dv >= lo) & (dv <= hi)
                ax.set_ylim(*_auto_flim(F[m] if m.any() else F))
                L, EW, tau = fr[n]['L'], fr[n]['EW'], fr[n]['tau']
                titles[n].set_text(
                    f"{n}   τ={tau:.1e}\nL={L:.2e}  EW={EW:+.1f} Å")
            else:
                ln.set_data([], [])
            arts += [ln, titles[n]]
        # advance the lightcurve marker to the current epoch
        if lc_dot is not None:
            t_lc, logL_lc = lc
            yv = float(np.interp(ep, t_lc, logL_lc))
            lc_dot.set_data([ep], [yv])
            lc_vline.set_xdata([ep, ep])
            lc_txt.set_text(f"t = {ep:.1f} d\nlog L = {yv:.2f}")
            arts += [lc_dot, lc_vline, lc_txt]
        return arts

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // max(args.fps, 1), blit=False)
    try:
        anim.save(args.out, fps=args.fps, writer='ffmpeg', dpi=110,
                  extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"[phase5-movie] Saved {args.out}")
    except Exception as e:
        gif = args.out.replace('.mp4', '.gif')
        print(f"[phase5-movie] ffmpeg unavailable ({e}); saving GIF: {gif}")
        anim.save(gif, fps=args.fps, writer=PillowWriter(fps=args.fps))
        print(f"[phase5-movie] Saved {gif}")
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1:])
