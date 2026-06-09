#!/usr/bin/env python
"""
check_backtest.py — consistency report for the SNLT back-test grid.

Loads the per-epoch `prod_day*_lines.npz` written by run_backtest.sh for each
model (A1, A4, B4, C1, C4), extracts the key per-line diagnostics, writes a flat
CSV (backtest/backtest_metrics.csv), and runs a set of PHYSICALLY-MOTIVATED
cross-model consistency checks — so a regression in any of the recent changes
(continuum renorm, adaptive window, saturated RT, composition-general budget,
metal lines, Cloudy Tier-2) shows up as a flagged anomaly rather than a silent
drift. No pipeline run needed — pure numpy on the npz outputs.

Run from the SNLT root after run_backtest.sh:
    python backtest/check_backtest.py
"""
from __future__ import annotations
import os
import glob
import re
import numpy as np

C_KMS = 2.99792458e5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ['A1', 'A4', 'B4', 'C1', 'C4']
H_RICH = {'A1', 'A4', 'B4'}            # expect Hα; H-free C-series expects Hα≈0
# the sparse back-test grid (must match run_backtest.sh --epochs). The model dirs
# may ALSO contain stale prod_day*.npz from earlier full runs — only these epochs
# are part of the back-test, so we ignore everything else.
BT_EPOCHS = [0.1, 1, 3, 5, 10, 20, 30, 40, 50, 80, 100]
HHE_LINES = ['Halpha', 'He_I_5876', 'He_I_10830', 'He_II_4686']
METAL_LINES = ['C_IV_1549', 'C_III_1909', 'O_III_5007']
KEY_LINES = HHE_LINES + METAL_LINES


def _is_bt_epoch(ep):
    return any(abs(ep - e) < 1e-4 for e in BT_EPOCHS)


def _epoch_of(path):
    m = re.search(r'day(\d+(?:\.\d+)?)', os.path.basename(path))
    return float(m.group(1)) if m else 9999.0


def _line_metrics(d, name):
    """Return dict(L, EW, peakF, peakv, cont) for `name` in npz `d`, or None."""
    names = list(d['line_names'])
    if name not in names:
        return None
    i = names.index(name)
    L = float(np.asarray(d['L_line_corrected'])[i]) if 'L_line_corrected' in d \
        else float(np.asarray(d['L_line'])[i])
    EW = float(np.asarray(d['EW_corrected'])[i]) if 'EW_corrected' in d \
        else float(np.asarray(d['EW'])[i])
    lk, fk = name + '__lambda', name + '__F_norm_corrected'
    peakF = peakv = cont = cont_edge = float('nan')
    if lk in d.files and fk in d.files:
        lam = np.asarray(d[lk], float); Fn = np.asarray(d[fk], float)
        lam0 = float(np.asarray(d['lambda_rest'])[i])
        finite = np.isfinite(Fn) & np.isfinite(lam)
        # need a usable (non-all-NaN) profile — cont-suppressed UV lines (e.g.
        # C IV 1549 at a cold photosphere) store an all-NaN F_norm_corrected.
        if lam0 > 0 and lam.size > 4 and finite.sum() >= 3:
            dv = (lam / lam0 - 1.0) * C_KMS
            j = int(np.nanargmax(np.where(finite, Fn, -np.inf)))
            peakF = float(Fn[j]); peakv = float(dv[j])
            win = max(abs(dv.min()), abs(dv.max()))
            fw = finite & (np.abs(dv) > 0.82 * win)
            fw_edge = finite & (np.abs(dv) > 0.95 * win)   # the very edge
            if fw.sum() >= 3:
                cont = float(np.nanmedian(Fn[fw]))
            cont_edge = (float(np.nanmedian(Fn[fw_edge]))
                         if fw_edge.sum() >= 3 else cont)
    return dict(L=L, EW=EW, peakF=peakF, peakv=peakv, cont=cont,
                cont_edge=cont_edge)


def main():
    rows = []
    anomalies = []
    found_any = False
    for model in MODELS:
        d_dir = os.path.join(ROOT, 'input_models', model)
        npzs = sorted(glob.glob(os.path.join(d_dir, 'prod_day*_lines.npz')),
                      key=_epoch_of)
        if not npzs:
            anomalies.append(f"[{model}] NO prod_day*_lines.npz found "
                             f"(run_backtest.sh not run for this model?)")
            continue
        found_any = True
        for npz in npzs:
            ep = _epoch_of(npz)
            if not _is_bt_epoch(ep):
                continue                    # stale npz from an earlier full run
            try:
                d = np.load(npz, allow_pickle=True)
            except Exception as e:
                anomalies.append(f"[{model} day{ep}] npz load failed: {e}")
                continue
            for ln in KEY_LINES:
                m = _line_metrics(d, ln)
                if m is None:
                    continue
                rows.append((model, ep, ln, m['L'], m['EW'], m['peakF'],
                             m['peakv'], m['cont']))
                # ---- per-line consistency checks ----
                tag = f"[{model} day{ep:g} {ln}]"
                if not np.isfinite(m['L']) or m['L'] < 0:
                    anomalies.append(f"{tag} L_line non-finite/negative: {m['L']}")
                # continuum renorm: far-wing F/F_cont must be ~1.0 — ONLY for H/He
                # lines (metals are line-centre-normalized). BUT only flag when the
                # far-wing is a FLAT continuum that's off 1.0 (a genuine renorm
                # issue). A broad/thick line (He I 10830, IIn Hα) fills the window,
                # so its "far-wing" is a SLOPED line wing, not continuum — the
                # renorm correctly declines to normalize against it, and that is
                # not an error. Distinguish via the slope between 0.82·win and the
                # very edge (0.95·win): flat (|Δ|<0.1) → real continuum.
                _flat = (np.isfinite(m.get('cont_edge', np.nan)) and
                         abs(m['cont'] - m['cont_edge']) < 0.10)
                if (ln in HHE_LINES and np.isfinite(m['cont'])
                        and not (0.80 <= m['cont'] <= 1.20) and _flat):
                    anomalies.append(f"{tag} continuum off 1.0 "
                                     f"(flat far-wing F/F_cont={m['cont']:.2f}) "
                                     f"— renorm regression?")
                # H presence: H-rich models must have real Hα; C-series ~0
                if ln == 'Halpha':
                    if model in H_RICH and m['L'] < 1e35:
                        anomalies.append(f"{tag} H-rich model but Hα L={m['L']:.1e} "
                                         f"(too weak)")
                    if model not in H_RICH and m['L'] > 1e36:
                        anomalies.append(f"{tag} H-free model but Hα L={m['L']:.1e} "
                                         f"(should be ~0)")
    # ---- write CSV ----
    out_csv = os.path.join(ROOT, 'backtest', 'backtest_metrics.csv')
    with open(out_csv, 'w') as f:
        f.write("model,epoch_d,line,L_line,EW,peakF,peak_v_kms,cont_level\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:g},{r[2]},{r[3]:.4e},{r[4]:.3f},"
                    f"{r[5]:.3f},{r[6]:.1f},{r[7]:.4f}\n")

    if not found_any:
        print("No back-test outputs found. Run:  bash backtest/run_backtest.sh")
        return

    # ---- metal-line time-series spike check: line luminosities evolve smoothly;
    #      a single-epoch up-then-down jump of > 2.5 dex is a numerical artifact
    #      (e.g. Cloudy thermal-bistability spike). Flag per (model, metal line). ----
    by_ml = {}
    for r in rows:
        if r[2] in METAL_LINES and np.isfinite(r[3]) and r[3] > 0:
            by_ml.setdefault((r[0], r[2]), []).append((r[1], r[3]))
    for (model, ln), pts in by_ml.items():
        pts.sort()
        for k in range(1, len(pts) - 1):
            lprev, lcur, lnext = pts[k - 1][1], pts[k][1], pts[k + 1][1]
            up = np.log10(lcur / lprev); down = np.log10(lcur / lnext)
            if up > 2.5 and down > 2.5:     # spike: >2.5 dex above BOTH neighbours
                anomalies.append(f"[{model} day{pts[k][0]:g} {ln}] L_line SPIKE "
                                 f"{lcur:.1e} (>2.5 dex above day{pts[k-1][0]:g}="
                                 f"{lprev:.1e} and day{pts[k+1][0]:g}={lnext:.1e}) "
                                 f"— Cloudy bistability / tier flicker?")

    # ---- blueshift-vs-epoch trend (physical expectation: peak velocity becomes
    #      LESS blueshifted as the photosphere recedes to lower velocity) ----
    print("=" * 70)
    print("BLUESHIFT vs EPOCH — He I 5876 peak velocity (km/s) per model")
    print("  (physical: |peak v| should DECREASE with epoch as v_phot recedes)")
    print("=" * 70)
    for model in MODELS:
        pts = sorted([(r[1], r[6]) for r in rows
                      if r[0] == model and r[2] == 'He_I_5876'
                      and np.isfinite(r[6])])
        if not pts:
            continue
        s = "  ".join(f"d{e:g}:{v:+.0f}" for e, v in pts)
        print(f"  {model}: {s}")
        # check monotone-ish recession (allow noise): early more blue than late
        if len(pts) >= 3:
            early = np.mean([v for e, v in pts if e <= 10])
            late = np.mean([v for e, v in pts if e >= 50])
            if np.isfinite(early) and np.isfinite(late) and late < early - 1500:
                anomalies.append(f"[{model}] He I 5876 peak velocity MORE blueshifted "
                                 f"at late epochs ({late:+.0f}) than early ({early:+.0f}) "
                                 f"— unexpected (photosphere should recede).")

    # ---- summary ----
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(rows)} line-measurements across "
          f"{len({(r[0], r[1]) for r in rows})} model-epochs")
    print(f"CSV: {out_csv}")
    print("=" * 70)
    if anomalies:
        print(f"\n⚠  {len(anomalies)} ANOMALIES FLAGGED:")
        for a in anomalies:
            print("   " + a)
    else:
        print("\n✓ No anomalies — continuum at 1.0, H/He composition consistent, "
              "blueshift recedes with epoch across all models.")


if __name__ == "__main__":
    main()
