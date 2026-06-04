# snline — Run Guide & CLI Reference

STELLA → photoeq → H/He NLTE → Monte-Carlo line-profile pipeline for
interacting Type II supernovae.

This README is the quick-start and complete flag reference. For the full
per-model operating procedure (smoke test, verification checklist, standing
limitations) see **RUNBOOK.md**. For the complete list of files to copy into
a new directory see **FILE_MANIFEST.txt**.

---

## Requirements

- Python 3.13, numpy 2.x, scipy, matplotlib (no compiled extensions)
- All 24 required modules present (see FILE_MANIFEST.txt). Verify with:
  ```bash
  python -c "import production_runner" && echo "imports OK"
  ```
  If this raises `ModuleNotFoundError`, the named module is missing — almost
  always one of the six EXTERNAL modules (`snapshot_analyzer.py`,
  `measure_jbar.py`, `wind_extension.py`, `photoionize_csm.py`,
  `sn_mc_voigt_peel.py`, `lines.py`) that must be copied from your existing
  working directory.
- STELLA snapshots named `mesa.day*_post_Lbol_max.data` (or `mesa_day*`) in
  the working directory.

---

## How to run

One script, two modes.

```bash
# Single snapshot — pass a file path
python production_runner.py <snapshot.data> [flags]

# Batch — pass --batch (no path); auto-globs mesa.day*_post_Lbol_max.data
python production_runner.py --batch [flags]
```

---

## The two commands you actually want

### Most physics-complete single snapshot (smoke test / one epoch)

```bash
python production_runner.py mesa.day080_post_Lbol_max.data --format stella \
    --he1-nlte --he2-nlte --he2-x-heiii-fraction 0.20 \
    --two-photon-decay \
    --he-lines --he-lines-n-packets 200000 \
    --n-per 200000 --n-chunks 2 \
    --iter-n 100000 --max-iter 8 --tol 0.02 \
    --out-prefix prod_day80
```

### Most physics-complete batch (the production run)

```bash
python production_runner.py --batch --batch-format stella \
    --he1-nlte --he2-nlte --he2-x-heiii-fraction 0.20 \
    --two-photon-decay \
    --he-lines --he-lines-n-packets 200000 \
    --n-per 200000 --n-chunks 2 \
    --iter-n 100000 --max-iter 8 --tol 0.02 \
    2>&1 | tee batch_final.log
```

Runtime: ≈2.5–3 hr for ~30 snapshots at these packet counts.

**What makes these "physics-complete":** the four switches that turn physics
ON (all default OFF) are `--he1-nlte`, `--he2-nlte`, `--he-lines`, and
`--two-photon-decay`. The two interaction terms — per-zone photoionization
and shock X-ray — are ON by default; you pass `--no-photoionize` /
`--no-shock-xray` only to turn them OFF, so for an interacting SN do **not**
pass them. `--he2-x-heiii-fraction 0.20` is the validated value. He I
two-photon decay is ON by default (disable only via
`--no-he1-two-photon-decay`). `--max-iter 8 --tol 0.02` tightens convergence
versus the defaults (6 / 0.03); harmless where it already converges in 6.

Do **not** add `--extend-wind` unless your STELLA grid is spatially
truncated and you need CSM beyond the outer boundary — it changes the
outer-boundary physics and was not used in the validated series.

---

## CLI reference (all flags)

### Mode / input

| Flag | Default | Meaning |
|---|---|---|
| `snap` (positional) | — | Snapshot file for single mode |
| `--batch` | off | Process all snapshots in current dir |
| `--format` | `auto` | Single-snapshot format: `auto`/`stella`/`heracles` |
| `--batch-format` | `auto` | Which family to batch: `stella`/`heracles` |
| `--skip-epochs` | "" | Comma-separated epoch values to skip in batch |
| `--out-prefix` | auto | Output filename prefix (single mode) |
| `--ref` | none | CMFGEN reference file for residual panel (single mode) |

### RT-NLTE iteration (the H Hα solve)

| Flag | Default | Meaning |
|---|---|---|
| `--iter-n` | 50000 | MC packets per iteration step. Raise to 100000 for paper. |
| `--max-iter` | 6 | Max RT iterations |
| `--tol` | 0.03 | Population convergence tolerance (relative) |
| `--damping` | 0.3 | Under-relaxation for J_bar updates (stability) |
| `--no-iter` | off | Disable iteration (legacy single-shot mode) |

### Production MC (the final Hα profile)

| Flag | Default | Meaning |
|---|---|---|
| `--n-per` | 100000 | Packets per chunk |
| `--n-chunks` | 2 | Chunks → total = n-per × n-chunks. Use 200000×2 for paper. |
| `--nbins` | 1200 | Wavelength bins across the band |
| `--smooth-kms` | 25.0 | Gaussian σ for displayed F_norm (cosmetic) |
| `--band-lo` / `--band-hi` | 6200 / 6950 | Output wavelength band [Å] |
| `--source-padding` | 1500 | Source band padding [Å] |
| `--calibration` | `auto` | F_norm baseline mode |
| `--line-redistribution` | `aa_prd` | Scattering redistribution (angle-averaged PRD) |

### Photosphere

| Flag | Default | Meaning |
|---|---|---|
| `--photosphere-mode` | `es` | Photosphere definition (`es` = electron-scattering τ=2/3) |
| `--photosphere-lam-ref` | 6562.8 | Reference λ for τ_cont surface [Å] |

### Photoionization & shock (interaction physics)

| Flag | Default | Meaning |
|---|---|---|
| `--no-photoionize` | (on) | Disable per-zone photoionization equilibrium. Leave ON for interacting SNe. |
| `--photoionize-T-source` | auto | Override photoionizing source T [K] (auto = thermalization-layer T) |
| `--photoionize-T-eq-floor` | 10000 | Min gas T in photoionized zones [K] |
| `--no-shock-xray` | (on) | Disable shock-bremsstrahlung X-ray photoionization. Leave ON. |

### H NLTE detail (Phase 2)

| Flag | Default | Meaning |
|---|---|---|
| `--two-photon-decay` | off | Enable H 2s→1s two-photon decay. Turn ON for full physics. |
| `--eps-lya-destruction` | none | Lyα destruction-probability floor |

### He I NLTE (Phase 3)

| Flag | Default | Meaning |
|---|---|---|
| `--he1-nlte` | off | Enable He I NLTE (11 levels). Required for He I lines. |
| `--he1-ionization-mode` | `follow_H` | How to split He I/He II per zone |
| `--he1-eps-resonance` | none | ε destruction floor on He I 584 Å resonance |
| `--no-he1-two-photon-decay` | (on) | Disable He I 2γ decay (default keeps it ON) |

### He II NLTE (Phase 4)

| Flag | Default | Meaning |
|---|---|---|
| `--he2-nlte` | off | Enable He II NLTE (10 levels). Required for He II lines. |
| `--he2-x-heiii-fraction` | none | Self-consistent X_HeIII override; 0.20 is the validated value |
| `--he2-x-heiii-mode` | `saha_local` | How X_HeIII is set per zone (if fraction not given) |
| `--he2-x-heiii-scalar` | none | Uniform X_HeIII override |

### Multi-line panel (Phase 5)

| Flag | Default | Meaning |
|---|---|---|
| `--he-lines` | off | Produce the 13-line H+He panel + empirical R-correction. |
| `--he-lines-n-packets` | 50000 | Packets per line. Raise to 200000 for paper-clean panels. |
| `--he-lines-calibration` | `theoretical_ew` | Phase-5 peel calibration mode |
| `--he-lines-reference-mc` | off | Use pure-Sobolev reference MC instead |

### Wind extension (optional; only if STELLA grid is truncated)

| Flag | Default | Meaning |
|---|---|---|
| `--extend-wind` | off | Append a photoionized pre-SN wind beyond the outer boundary |
| `--wind-r-max-factor` | 20.0 | Extend to r_max = factor × r_outer |
| `--wind-n-zones` | 100 | Zones added |
| `--wind-T-photoionized` | 10000 | Floor T of the added wind [K] |
| `--wind-rho-index` | 2.0 | Density falloff ρ ∝ r^(−index) |
| `--wind-density-boost` | 1.0 | Density multiplier on the extension |

### Movie

| Flag | Default | Meaning |
|---|---|---|
| `--phase5-movie-out` | auto | Filename for the batch multi-line movie |
| `--phase5-movie-fps` | 3 | Frames per second |

---

## Environment variables

| Variable | Effect |
|---|---|
| `SNLINE_R_MODE=ew` | Use legacy EW-based empirical R-correction instead of the default L-based one |
| `SNLINE_DEBUG_DAY80=1` | Verbose per-zone day-80 debug dump |

---

## Output files

**Per snapshot** (prefix `prod_dayXXX` by default):

| File | Contents |
|---|---|
| `prod_dayXXX.png` / `.txt` | Production RT-NLTE Hα profile (6-panel figure + metrics) |
| `prod_dayXXX_lines.png` / `.txt` / `.npz` | 13-line H+He panel, table, and raw arrays |
| `prod_dayXXX_hydro.png` | Density/velocity/T/composition/τ/emissivity structure |
| `prod_dayXXX_ionization.png` | Ionization equilibrium + shock X-ray diagnostics |
| `prod_dayXXX_regime.txt` | Per-line trust grades (A/B/C/R) + action recommendations |
| `prod_dayXXX_convergence_audit.txt` | RT-NLTE iteration history; L_line variation |

**Batch-level:**

| File | Contents |
|---|---|
| `batch_summary.txt` | Per-epoch L_line, global & core peak F, trough, runtime |
| `batch_regime_summary.txt` | Cross-epoch grade grid (lines × epochs) — READ THIS before quoting |
| `batch_metrics.csv` | Machine-readable metrics for all epochs and lines |
| `batch_grid.png` | Hα montage across all epochs |
| `batch_lines_evolution.mp4` | 13-line evolution movie |

---

## Reading the output: trust grades

Before quoting any line, open `batch_regime_summary.txt` and check its grade
**for this model** (grades are model-specific and do not carry over):

| Grade | Meaning | Quote |
|---|---|---|
| **A** | optically thin OR full RT-NLTE | L_line and EW directly |
| **B** | optically thick + Hα-anchored empirical R | L_line as "empirical estimate" |
| **C** | saturated, no per-line iteration | shape only (v_peak, FWHM); L uncertain ×2–5 |
| **R** | atomic physics outside scope | shape only (e.g. He II 1640 Lyα-cascade) |

Standing model-independent limitations (always disclose): He II 1640 is
grade-R always; He I 10830 is grade-C in any optically-thick plateau; late
nebular epochs are unphysical without ⁵⁶Ni (peak F in the hundreds = continuum
collapse, out of scope); fine profile structure is resolution-limited at thin,
reversal-rich epochs. Continuum opacity is Thomson + analytic H bound-free
(no line blanketing). See RUNBOOK.md §5 for the full list.

---

## Diagnostics (optional, included)

```bash
# Re-grade an existing batch from saved *_lines.npz (no re-run)
python make_regime_summary_posthoc.py --also-per-snapshot --verbose

# Quantify when a fast shock shell emerges above the photosphere
python diagnose_shock_emergence.py mesa.day0[12345]0_post_Lbol_max.data --v-thresh 2000

# Relate multi-peak profile structure to the velocity field at a given epoch
python diagnose_day20_peaks.py mesa.day020_post_Lbol_max.data --npz prod_day020_lines.npz
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` at startup | external module not copied | copy the 6 EXTERNAL modules (FILE_MANIFEST.txt) |
| T_phot stuck at 10000 K | state-fix override not firing | confirm latest `production_runner.py`; check `[state-fix]` in log |
| He lines flat / missing | `--he-lines` / `--he1-nlte` / `--he2-nlte` omitted | add the flags |
| Empirical R not applied | `[phase5b]` line absent in log | confirm `--he-lines` |
| Peak F in the hundreds | nebular continuum collapse (no ⁵⁶Ni) | that epoch is out of scope |
| Profile noisy / spiky | too few packets | raise `--he-lines-n-packets`, `--n-per`, `--iter-n` |
| L_line varies >3% across iters | MC noise in J_bar at high τ | raise `--iter-n` (not `--max-iter`); see `_convergence_audit.txt` |
| Multi-peak profile at thin epoch | reversal-rich under-resolved v field | quote L; describe shape qualitatively |
