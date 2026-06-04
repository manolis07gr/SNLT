# snline — Production Runbook

STELLA → photoeq → H/He NLTE → Monte-Carlo line-profile pipeline for
interacting Type II supernovae. This document is the per-model operating
procedure. Follow it for every new STELLA model series.

---

## 0. What this pipeline does

Ingests a STELLA hydrodynamic snapshot series and produces, per epoch:
- RT-NLTE-iterated Hα line profile (the primary, highest-trust product)
- 13-line H + He profile panel (5 Balmer/Paschen H lines + 8 He I/II lines)
- Per-zone ionization, hydro, and emissivity diagnostics
- A per-line, per-epoch **trust grade** (A / B / C / R) so you know which
  numbers to quote directly, which to quote with a caveat, and which to use
  for line shape only.

It is a **self-diagnosing screening tool**. It runs end-to-end without
intervention and tells you, via the regime grades, where to be careful.
It does NOT certify its own output blindly — you must read the regime grid
for each model before quoting any line (see Step 4).

---

## 1. One-time setup for a new directory

Copy ALL files listed in `FILE_MANIFEST` (see the manifest section at the
bottom, or the separate FILE_MANIFEST.txt) into the new directory. There are
two groups:

- **Staging files** — the modules developed/maintained in this effort.
- **External modules** — six files that live only in your existing working
  directory and were never in the staging set. **These are mandatory runtime
  dependencies.** Copy them from your current `LINE MODELING` directory:
  `snapshot_analyzer.py`, `measure_jbar.py`, `wind_extension.py`,
  `photoionize_csm.py`, `sn_mc_voigt_peel.py`, `lines.py`.

If any external module is missing, `production_runner.py` will fail at import
with `ModuleNotFoundError`. That is the first thing to check if startup fails.

Environment: Python 3.13, numpy 2.x, scipy, matplotlib. No compiled
extensions. Put the STELLA snapshot files (`mesa.dayXXX_post_Lbol_max.data`)
in the same directory or pass paths explicitly.

---

## 2. Smoke-test ONE snapshot first (≈2 min)

Never launch a multi-hour batch without smoke-testing a single mid-plateau
snapshot. This catches format/column/epoch-parsing surprises in a new model
in 2 minutes instead of after a 3-hour batch.

```bash
python production_runner.py mesa.day080_post_Lbol_max.data --format stella \
    --he1-nlte --he2-nlte --he2-x-heiii-fraction 0.20 \
    --he-lines --he-lines-n-packets 200000 \
    --n-per 200000 --n-chunks 2 \
    --iter-n 100000 \
    --out-prefix smoke_day80 \
    2>&1 | tee smoke_day80.log
```

(Pick whatever epoch is a representative mid-series snapshot for the new
model; day 80 is just the example from the validated series.)

### Five-line verification — all of these MUST appear in the log:

```
[state-fix] T_phot: ... → ...                      (photosphere T override fired)
[STELLA] state.X_H overridden with per-zone ...    (H composition plumbed)
[STELLA] state.X_He overridden with per-zone ...   (He composition plumbed)
[phase5] prod_Ha cross-check: L_line = ...          (production Hα recorded)
[phase5b] ... R = ... = 0.0X                         (empirical correction fired)
[regime] Saved smoke_day80_regime.txt ...           (grading ran)
```

Grep them in one shot:

```bash
grep -E "state-fix|state.X_H |state.X_He|prod_Ha cross-check|R = |Saved .*regime" smoke_day80.log
```

If any are missing, STOP and diagnose before batching. Common causes:
- Missing external module → ModuleNotFoundError at top.
- Different STELLA column layout → check `COMPOSITION_COLS` and the column
  index constants at the top of `stella_io.py`.
- Epoch not parsed → check the filename matches `dayXXX_post` or the header
  `days post ... Lbol` line.

Also sanity-check the smoke-test PNG/regime file: is T_phot physical
(thousands of K, not 10000 K hardcoded)? Is L_line in a plausible range
(10^39–10^41 erg/s for Hα)? Does the regime file list grades for all 14 rows?

---

## 3. Run the production batch (≈2.5–3 hr for ~30 snapshots)

```bash
python production_runner.py --batch --batch-format stella \
    --he1-nlte --he2-nlte --he2-x-heiii-fraction 0.20 \
    --he-lines --he-lines-n-packets 200000 \
    --n-per 200000 --n-chunks 2 \
    --iter-n 100000 \
    2>&1 | tee batch_final.log
```

Packet counts (tuned for paper-quality, low MC noise):
- `--he-lines-n-packets 200000` : 13-line panel packets per line
- `--n-per 200000 --n-chunks 2` : 400k total for production Hα
- `--iter-n 100000`             : RT-NLTE iteration MC step

To go faster for a quick look (noisier panels), drop these to 50000 /
50000×2 / 50000 respectively (~50 min). Use full counts for anything that
goes in a paper.

Outputs per epoch: `prod_dayXXX.{png,txt}`, `prod_dayXXX_lines.{png,txt,npz}`,
`prod_dayXXX_hydro.png`, `prod_dayXXX_ionization.png`,
`prod_dayXXX_regime.txt`, `prod_dayXXX_convergence_audit.txt`.
Batch-level: `batch_summary.txt`, `batch_regime_summary.txt`,
`batch_metrics.csv`, `batch_grid.png`, `batch_lines_evolution.mp4`.

---

## 4. READ THE REGIME GRID before quoting anything (per model!)

**This step is non-negotiable and is what makes the output trustworthy.**
The grades are model-specific — they depend on the CSM density and velocity
structure of THIS model. Grades from a previous model do NOT carry over.

Open `batch_regime_summary.txt`. For each line you want to use:

| Grade | Meaning | What you may quote |
|-------|---------|--------------------|
| **A** | optically thin OR full RT-NLTE | L_line and EW directly |
| **B** | optically thick + Hα-anchored empirical R | L_line as "empirical estimate" |
| **C** | saturated, no per-line iteration | shape only (v_peak, FWHM); L uncertain ×2–5 |
| **R** | atomic physics outside scope | shape only (e.g. He II 1640 Lyα-cascade) |

Production Hα (`Halpha_prod` row) should be grade A at essentially all
epochs — it is the headline product. If it is NOT grade A somewhere, check
that epoch's `_convergence_audit.txt` (look for L_line variation >1%).

---

## 5. Standing limitations (model-independent — put in paper methods)

These apply to EVERY model and belong in the methods/caveats section:

- **He II 1640 is grade-R always** (Lyα-cascade & continuum-recombination
  physics not implemented). Use for shape only.
- **He I 10830 is grade-C in any optically-thick plateau** (metastable
  2³S→2³P triplet pumping not per-line iterated). Shape only until the
  optional "Phase 5b-rigorous" upgrade.
- **Late nebular epochs are unphysical without ⁵⁶Ni.** Wherever the
  continuum collapses you will see peak F in the hundreds–thousands in
  `batch_summary.txt`. Those epochs are OUT OF SCOPE. Truncate paper
  coverage before the continuum-collapse epoch.
- **Fine profile structure is resolution-limited where the velocity field
  has many reversals in a thin (few-tens-of-zones) atmosphere.** Quote
  luminosity; describe shape qualitatively at such epochs. (See the day-20
  lesson in the validated series: `diagnose_day20_peaks.py`,
  `diagnose_shock_emergence.py`.)
- **Continuum opacity is Thomson + analytic H bound-free, no line
  blanketing.** Matters more in the UV than the optical.
- **Sobolev assumes monotonic v; kernel uses local Doppler** to tolerate the
  ~10% of zones with reversals. Documented approximation.

---

## 6. Optional diagnostics (already included)

- `make_regime_summary_posthoc.py` — regenerate `batch_regime_summary.txt`
  from existing `prod_*_lines.npz` if you ran a batch before the Phase-6 hook
  existed, or want to re-grade without re-running:
  `python make_regime_summary_posthoc.py --also-per-snapshot --verbose`
- `diagnose_shock_emergence.py` — quantify when/if a fast shock shell emerges
  above the photosphere across early epochs (ties hydro to line emissivity):
  `python diagnose_shock_emergence.py mesa.day0*.data mesa.day0[1-4]0*.data --v-thresh 2000`
- `diagnose_day20_peaks.py` — relate multi-peak profile structure to the
  velocity field (τ vs projected velocity) at a transitional epoch.

---

## 7. Key flags reference

| Flag | Meaning |
|------|---------|
| `--batch --batch-format stella` | process all STELLA snapshots in dir |
| `--he1-nlte` / `--he2-nlte` | enable He I / He II NLTE solvers |
| `--he2-x-heiii-fraction 0.20` | He III initial guess (NLTE re-derives) |
| `--he-lines` | produce 13-line Phase-5 panel + empirical R |
| `--he-lines-n-packets N` | packets per He/H line in the panel |
| `--n-per N --n-chunks K` | production Hα = N×K total packets |
| `--iter-n N` | packets per RT-NLTE iteration step |
| `--out-prefix STR` | output filename prefix (single-snapshot mode) |
| `SNLINE_R_MODE=ew` (env) | legacy EW-based R-correction instead of L-based |
| `SNLINE_DEBUG_DAY80=1` (env) | verbose per-zone day-80 debug dump |

---

## 8. Troubleshooting quick table

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError` at startup | external module not copied | copy the 6 external modules (Step 1) |
| T_phot stuck at 10000 K | state-fix override not firing | confirm latest `production_runner.py`; check `[state-fix]` in log |
| He lines all flat / missing | `--he-lines` or `--he1/2-nlte` omitted | add the flags |
| Empirical R not applied | `[phase5b]` line absent | confirm `--he-lines`; check phase5_runner version |
| Peak F in the hundreds | nebular continuum collapse | that epoch is out of scope (no ⁵⁶Ni) |
| Profile noisy / spiky | too few packets | raise `--he-lines-n-packets`, `--n-per`, `--iter-n` |
| L_line varies >1% across iters | under-converged at high τ | raise `--iter-n`; inspect `_convergence_audit.txt` |
| Multi-peak profile at thin epoch | reversal-rich under-resolved v field | quote L; describe shape qualitatively |

---

## FILE MANIFEST (summary — see FILE_MANIFEST.txt for the authoritative list)

**Staging — production core (16):** production_runner.py, snline_autoparams.py,
peel_pipeline_abs.py, stella_io.py, phase5_runner.py, mc_multi_line.py,
phase5_continuum.py, regime_diagnostics.py, make_phase5_movie.py,
snline_he1_integration.py, snline_he2_integration.py, nlte_he1.py, nlte_he2.py,
he1_atom.py, he2_atom.py, h_populations_nlte.py

**Staging — Phase-1 core (2):** opacity.py, photosphere_v2.py

**External — copy from your existing working dir (6, MANDATORY):**
snapshot_analyzer.py, measure_jbar.py, wind_extension.py, photoionize_csm.py,
sn_mc_voigt_peel.py, lines.py

**Staging — diagnostics/utilities (3, optional):**
make_regime_summary_posthoc.py, diagnose_shock_emergence.py,
diagnose_day20_peaks.py

**Staging — tests (9, optional but recommended for validation):**
test_joint_clean.py, test_phase1.py, test_phase1_synthetic_IIP.py,
test_phase2.py, test_phase3.py, test_phase3_integration.py,
test_phase4_he2.py, test_phase4_v2.py, test_solution1_eps_lya.py
