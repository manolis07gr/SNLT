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
import re
import glob
import argparse
import numpy as np

C_KMS = 299792.458

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
    args = p.parse_args(argv)

    files = sorted(glob.glob(args.pattern), key=_epoch_from_name)
    if len(files) < 1:
        print(f"[phase5-movie] no files match {args.pattern}")
        return
    frames = [_load_frame(f) for f in files]
    epochs = [e for e, _ in frames]

    # panel line order
    present = list(frames[0][1].keys())
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
    from matplotlib.animation import FuncAnimation, PillowWriter

    ncols = 5
    nrows = (nlines + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    sup = fig.suptitle('', fontsize=15, y=0.995)

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
    for j in range(nlines, len(axes_flat)):
        axes_flat[j].set_visible(False)

    def update(idx):
        ep, fr = frames[idx]
        sup.set_text(f"Phase 5 multi-line H + He evolution    "
                     f"t = {ep:.1f} d post-Lbol-max    (frame {idx+1}/{len(frames)})")
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
