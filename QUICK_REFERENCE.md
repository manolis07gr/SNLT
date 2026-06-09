# SNLT pipeline — Quick Reference

Everything below is run from inside a model's directory (the one holding that
model's `mesa.day*_post_Lbol_max.data` snapshots), with the venv active.

---

## 1. The three commands you actually need

### (a) Batch — full time series for one model
Processes every `mesa.day*_post_Lbol_max.data` snapshot in the current directory.
Works identically for **every regime** (IIP, IIn, stripped/Ibn) — the Sobolev
gate picks the profile method per epoch automatically. **Do not** add
`--line-profile-method-lock`.

```bash
python production_runner.py --batch \
       --line-profile-method formal \
       --he-lines --he1-nlte --he2-nlte
```
Writes: `prod_day*_lines.npz/.txt/.png`, `prod_day*_regime.txt`,
`batch_grid.png`, `batch_movie.mp4`, `batch_lines_evolution.mp4`,
`batch_summary.txt`, `batch_regime_summary.txt`, `batch_metrics.csv`.

### (b) Single snapshot — one epoch
Same flags, one file instead of `--batch`:

```bash
python production_runner.py mesa.day080_post_Lbol_max.data \
       --line-profile-method formal \
       --he-lines --he1-nlte --he2-nlte
```

### (c) Multi-panel line-evolution movie (standalone, no re-run)
Builds the H+He per-line evolution movie straight from existing `*_lines.npz`:

```bash
python make_phase5_movie.py 'prod_*_lines.npz' --out batch_lines_evolution.mp4 --fps 3
```

**Reading the batch log:** `[formal] SKIPPED …` means that epoch was found
non-homologous (dense-CSM / IIn) and the MC emission-line profile was kept; no
SKIPPED line means it was homologous (IIP-like) and the formal P-Cygni solution
was used. Quote `L_corr` (not `L_raw`) from the `*_lines.txt` files, and quote
`L_line` rather than peak-F at the interaction-brightening / continuum-collapse
epochs.

### (d) Optional physics flags (opt-in; defaults reproduce prior results)

| flag | item | what it does |
|------|------|--------------|
| `--saturated-rt` | P1 #3 | For optically-thick He lines (τ_med ≥ 1): drops the empirical Hα-anchored `R_flat` and keeps the bare single-shot β escape luminosity (= first-principles escape-probability value), and applies multiple-electron-scattering to the profile SHAPE (`He-NLTE(thick,EP-esc)` mode). Strengths only; the gate still owns the shape. |
| `--he-budget` | P1 #4 | Composition-general continuum-collapse guard: floors a Wien-collapsed `L_cont_band` to the energy-conserving color temperature, plus an energy-conservation check and a first-principles He decrement (He I 10830 reference). **AUTO-enabled when ⟨X_H⟩ < 1e-3** (H-free), so C-series runs get it without the flag. |
| `--metal-lines` | P2 #5 | Phase 5c: add C/O/Ne lines (C IV 1549, C III] 1909, C III 4647, [O I] 6300, [O III] 5007, [Ne III] 3869). First-principles emissivity integrals on photoionization-equilibrium ion densities, transported through the He MC peel-off kernel for realistic profiles; merged into npz/regime/plots/movies dynamically; writes `{prefix}_metal_lines.png`. Emissivities use **CHIANTI NLTE (Tier-1)** when ChiantiPy + `$XUVTOP` are present, else provisional. Most useful for the stripped/Icn C-series. |
| `--metal-cloudy` | P2 #5 Tier-2 | Override the metal ABSOLUTE luminosities with **Cloudy** (self-consistent photoionization + NLTE + **resonance-line RT**), fixing the C IV 1549 / C III] 1909 resonance absolutes and the ion balance. One Cloudy run per snapshot; the MC keeps the velocity SHAPE. Per-line graceful fallback Cloudy→CHIANTI→provisional. **Implies `--metal-lines`.** Requires Cloudy compiled and locatable via `$CLOUDY_EXE` or `~/c23.01/source/cloudy.exe`. |
| `--keep-shared-snapshots` | P0 #1 | Disables the default skip of late-epoch snapshots that are byte-identical to a sibling model dir (the shared-placeholder artifact). |

Without these flags, every output is byte-identical to the pre-P1 pipeline
(except H-free models, where `--he-budget` auto-engages by design). Validate the
new physics analytically with `python validate_p1_physics.py` (no pipeline run).

```bash
# Example: full IIn series with the saturated-line RT and the budget guard
python production_runner.py --batch \
       --line-profile-method formal \
       --he-lines --he1-nlte --he2-nlte \
       --saturated-rt --he-budget

# Example: stripped/Icn C-series snapshot with metal lines + Cloudy Tier-2
#   (set XUVTOP for CHIANTI Tier-1; CLOUDY_EXE if Cloudy is not at the default path)
XUVTOP=~/Documents/SNLT/chianti CLOUDY_EXE=~/c23.01/source/cloudy.exe \
python production_runner.py mesa.day010_post_Lbol_max.data --format stella \
       --he-lines --he1-nlte --he2-nlte \
       --metal-lines --metal-cloudy \
       --out-prefix c1_day010
```

---

## 2. Post-processing & plotting

All three scripts import `snline_postproc.py` (keep it alongside them) and read
the per-epoch `prod_day*_lines.npz` files. Model properties (M_csm, M_ej, E_SN,
R_prog, CSM composition) are built in for A/B/C series; override or extend with
`--model-table your_models.csv` (columns: `model,M_csm,M_ej,E_SN,R_prog,CSM_comp`).

### (1) Single-run 4-panel evolution figure
L(t), EW(t), line ratios, and a decrement-vs-luminosity track. Choose Balmer or
the strongest He lines; set a phase window **or** an explicit phase list.

```bash
# full range, Balmer:
python plot_single_run.py --run path/to/A4 --species balmer --out A4_balmer.png

# phase window 0–120 d, helium:
python plot_single_run.py --run path/to/A7 --species he --t0 0 --t1 120

# explicit phases:
python plot_single_run.py --run path/to/A5 --species balmer --phases 0,1,5,10,20
```

### (2) Cross-run comparison (peak ∓ one e-folding) vs M_csm/M_ej or E_SN

```bash
# one figure:
python plot_compare_runs.py --runs A1:runs/A1 A4:runs/A4 A7:runs/A7 \
       --species balmer --xaxis ratio --out cmp_balmer_ratio.png

# all four at once (balmer/he × ratio/E_SN):
python plot_compare_runs.py --runs A4:runs/A4 A5:runs/A5 A6:runs/A6 A7:runs/A7 \
       --all --outdir figs/
```
Each `--runs` entry is `MODELID:path`. Peak epoch = bold filled marker; the two
e-folding epochs = dim/transparent; the three are joined by a dotted line.

### (3) Parameter-space correlation + PCA analysis

```bash
python analyze_correlations.py \
       --runs A1:runs/A1 A4:runs/A4 A5:runs/A5 A6:runs/A6 A7:runs/A7 \
              B4:runs/B4 B5:runs/B5 B6:runs/B6 B7:runs/B7 \
       --outdir analysis/

# or point at a parent folder of model dirs:
python analyze_correlations.py --runs-glob 'runs/*' --outdir analysis/
```
Writes `master_table.csv`, `corr_pearson.csv`, `corr_spearman.csv`,
`correlation_heatmap.png`, `pca_loadings.csv`, `pca_scores.csv`,
`pca_summary.png`, and `correlations_report.txt`.

---

## 3. Filtering the untrustworthy late epochs

The **shared-late-snapshot loader bug is now fixed** (P0 #1): `--batch`
auto-skips byte-identical sibling-directory placeholders at run time, so the
late rows are no longer duplicated across models (override with
`--keep-shared-snapshots`). The unphysical `L_cont_band` collapse at cold/compact
late epochs is addressed by `--he-budget` (auto for H-free; floors the continuum
energy-consistently).

You may still want to truncate the post-continuum-collapse tail for plotting on
*physics* grounds (the late nebular phase is not paper-ready): use `--t1` to cut
at the collapse epoch, identified in `batch_metrics.csv` as the point where
`L_cont_band` falls by orders of magnitude and `tau_es_photoeq` drops below ~0.3.
The plotting scripts honour `--t1`, so e.g. `--t1 120` keeps only the trustworthy
interaction phase.
