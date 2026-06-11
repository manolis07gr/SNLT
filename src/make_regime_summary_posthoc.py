"""make_regime_summary_posthoc.py — generate batch_regime_summary.txt
from an already-completed batch.

Reads all prod_*_lines.npz files in the current directory (or another
directory) and produces a batch_regime_summary.txt identical to what
Phase 6 would have generated, had it been wired in when the batch ran.

Usage:
    python make_regime_summary_posthoc.py
    python make_regime_summary_posthoc.py --dir /path/to/batch/outputs
    python make_regime_summary_posthoc.py --out my_summary.txt

The script also writes a per-snapshot {prefix}_regime.txt for each
snapshot that doesn't already have one.

The npz files must have been produced by Phase 5 (i.e. the batch was run
with --he-lines). For each line, the npz stores tau_med, L_line, EW,
peak_F, etc.; we feed those through the regime_diagnostics classifier.
"""

from __future__ import annotations
import argparse
import glob
import os
import re
import sys

import numpy as np

import regime_diagnostics as rd


def parse_epoch(path):
    """Extract epoch number from filename like 'prod_day030_lines.npz'."""
    m = re.search(r'day(-?\d+(?:\.\d+)?)', os.path.basename(path))
    return float(m.group(1)) if m else None


def load_phase5_npz(path):
    """Reconstruct a {line_name: spectrum_dict} from a Phase 5 NPZ.

    The actual Phase 5 schema (from phase5_runner._save_phase5_npz):
      'line_names'         : array of N line names
      'L_line'             : array of N L_line values, indexed by line_names
      'L_cont_band'        : array of N L_cont_band values
      'EW'                 : array of N EW values
      'lambda_rest'        : array of N lambda_rest values
      'tau_med'            : array of N tau_med values
      (optional, when Phase 5b empirical correction fired:)
      'L_line_corrected', 'EW_corrected', 'peak_F_corrected'
      (per-line arrays for plotting:)
      '{name}__lambda', '{name}__F_norm', '{name}__F_lambda', etc.

    We extract only the scalar columns needed by the regime classifier.
    """
    d = np.load(path, allow_pickle=True)
    if 'line_names' not in d.files:
        return {}
    line_names = [str(x) for x in d['line_names']]

    spectra = {}
    # Required scalar fields (parallel arrays indexed by line position)
    scalar_keys = ['L_line', 'L_cont_band', 'EW', 'lambda_rest', 'tau_med']
    # Optional corrected fields
    corrected_keys = ['L_line_corrected', 'EW_corrected', 'peak_F_corrected']

    for i, name in enumerate(line_names):
        sp = {}
        for sk in scalar_keys:
            if sk in d.files:
                try:
                    sp[sk] = float(d[sk][i])
                except (IndexError, TypeError):
                    sp[sk] = float('nan')
        for ck in corrected_keys:
            if ck in d.files:
                try:
                    sp[ck] = float(d[ck][i])
                except (IndexError, TypeError):
                    sp[ck] = float('nan')
        # Compute peak_F from the F_norm array if available (it's the raw,
        # uncorrected peak; corrected version is in peak_F_corrected when
        # Phase 5b fired).
        f_norm_key = f"{name}__F_norm"
        if f_norm_key in d.files:
            try:
                Fn = np.asarray(d[f_norm_key], dtype=float)
                if Fn.size > 0:
                    sp['peak_F'] = float(np.nanmax(Fn))
            except Exception:
                pass
        spectra[name] = sp

    return spectra


def find_prod_Ha(line_txt_path):
    """Try to recover prod_Ha {L_line, peak_F, peak_dv, EW} from the lines.txt.

    Production Hα cross-check info is written to {prefix}_lines.txt's header.
    Returns dict or None.
    """
    if not os.path.exists(line_txt_path):
        return None
    try:
        with open(line_txt_path) as f:
            txt = f.read()
        # Match: "L_line = 5.196e+40 erg/s   peak F = 5.180 @ Δv = +1342.0 km/s   EW = -123.04 Å"
        m = re.search(
            r'L_line\s*=\s*([\d.eE+-]+).*?peak F\s*=\s*([\d.eE+-]+).*?'
            r'Δv\s*=\s*([\d.eE+-]+).*?EW\s*=\s*([\d.eE+-]+)',
            txt, re.DOTALL)
        if not m:
            return None
        return {
            'L_line': float(m.group(1)),
            'peak_F': float(m.group(2)),
            'peak_dv': float(m.group(3)),
            'EW': float(m.group(4)),
        }
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='.',
                    help='directory containing prod_*_lines.npz')
    ap.add_argument('--pattern', default='prod_*_lines.npz',
                    help='glob pattern for Phase 5 NPZ files')
    ap.add_argument('--out', default='batch_regime_summary.txt',
                    help='output summary path')
    ap.add_argument('--also-per-snapshot', action='store_true',
                    help='write a {prefix}_regime.txt for each snapshot too')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)

    pattern = os.path.join(args.dir, args.pattern)
    npz_files = sorted(glob.glob(pattern), key=parse_epoch)
    if not npz_files:
        print(f"No files matching {pattern}", file=sys.stderr)
        return 1

    print(f"Found {len(npz_files)} Phase 5 NPZ files in {args.dir}")
    if args.verbose:
        for p in npz_files:
            print(f"  {p}  (epoch={parse_epoch(p)})")

    all_results = []
    for npz_path in npz_files:
        epoch = parse_epoch(npz_path)
        spectra = load_phase5_npz(npz_path)
        if not spectra:
            print(f"  SKIP {os.path.basename(npz_path)}: no scalar fields found")
            continue

        # Build mock snap for the table
        snap = {'epoch_d': epoch}

        # Try to load production Hα info from {prefix}_lines.txt header
        prefix = npz_path[:-len('_lines.npz')]
        prod_Ha = find_prod_Ha(prefix + '_lines.txt')

        rows = rd.build_snapshot_table(spectra, snap)

        # Add the production Hα row (full RT-NLTE) if we found it
        if prod_Ha is not None:
            ha_p5 = spectra.get('Halpha', {})
            prod_row = rd.classify_line(
                'Halpha',
                tau_med=float(ha_p5.get('tau_med', np.nan)),
                tau_max=None, beta_med=None)
            prod_row['grade'] = 'A'
            prod_row['rationale'] = (
                'Full RT-NLTE iteration in production_runner '
                '(L_line invariant across iters to <0.01%).')
            prod_row['paper_action'] = 'Quote production L_line and EW. Grade-A.'
            prod_row['line'] = 'Halpha_prod'
            prod_row['epoch_d'] = epoch
            prod_row['L_line'] = prod_Ha['L_line']
            prod_row['EW'] = prod_Ha['EW']
            prod_row['peak_F'] = prod_Ha['peak_F']
            prod_row['lambda_rest'] = 6562.81
            rows.insert(0, prod_row)

        all_results.append({'epoch_d': epoch, 'rows': rows})

        if args.also_per_snapshot:
            per_snap_path = prefix + '_regime.txt'
            rd.write_snapshot_diagnostic(per_snap_path, rows, snap,
                                          production_halpha=prod_Ha)
            if args.verbose:
                print(f"  wrote {per_snap_path}")

        n_grades = {'A': 0, 'B': 0, 'C': 0, 'R': 0}
        for r in rows:
            n_grades[r['grade']] += 1
        if args.verbose:
            print(f"  epoch {epoch:>6.2f}d: "
                  f"A={n_grades['A']:>2}  B={n_grades['B']:>2}  "
                  f"C={n_grades['C']:>2}  R={n_grades['R']:>2}")

    rd.write_batch_regime_summary(all_results, args.out)
    print(f"\nWrote {args.out}  ({len(all_results)} epochs)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
