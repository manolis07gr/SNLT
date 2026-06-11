# SNLT — Getting Started (new-user setup guide)

SNLT post-processes STELLA supernova snapshots into physical H / He / metal
**line luminosities and profiles** across the IIP / IIn / Ib / Ic / Ibn / Icn
regimes. This guide walks a brand-new checkout from zero to a working run.

> **Time budget:** ~30–60 min, most of it compiling Cloudy. You need a working
> C++ compiler, Python 3.11+, and ~3 GB of disk for Cloudy + the CHIANTI database.

---

## 0. What you need (overview)

| Component | Why | Notes |
|-----------|-----|-------|
| **Python 3.11+** (3.13 tested) | runs the pipeline | a dedicated venv is strongly recommended |
| **numpy, scipy, numba, matplotlib, pandas** | core numerics + JIT + figures | `pip install` |
| **ChiantiPy + CHIANTI atomic database** | Tier-1 metal-line NLTE emissivities | needs the `$XUVTOP` data dir |
| **Cloudy C23.01** (compiled) | Tier-2 metal absolutes (resonance-line RT) | compile from source; set `$CLOUDY_EXE` |
| **ffmpeg** | the evolution movies (`.mp4`) | optional; falls back to `.gif` |

The pipeline degrades gracefully: without ChiantiPy/Cloudy, metal lines fall back
to provisional atomic data (and a warning); without ffmpeg, movies become GIFs.
**H and He lines work with just the core Python stack.**

---

## 1. Python environment

```bash
cd ~/Documents/SNLT          # your checkout
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy numba matplotlib pandas ChiantiPy
```

Verify:
```bash
python -c "import numpy, scipy, numba, matplotlib, pandas, ChiantiPy; print('core deps OK')"
```
If `numba` fails to import, make sure you are on Python ≤3.13 (numba lags the
newest Python by a few months).

---

## 2. CHIANTI atomic database (for `$XUVTOP`)

ChiantiPy needs the CHIANTI **database** (separate from the Python package).

1. Download the latest CHIANTI database tarball from
   <https://www.chiantidatabase.org/chianti_download.html> (the
   "CHIANTI_v10.x_database.tar.gz" file).
2. Unpack it somewhere stable, e.g. `~/Documents/SNLT/chianti`:
   ```bash
   mkdir -p ~/Documents/SNLT/chianti
   tar xzf CHIANTI_v10.1_database.tar.gz -C ~/Documents/SNLT/chianti
   ```
3. Point `$XUVTOP` at it (the dir that contains `VERSION`, `masterlist/`, …):
   ```bash
   export XUVTOP=~/Documents/SNLT/chianti
   ```
Verify:
```bash
python -c "import os,ChiantiPy.core as ch; print('XUVTOP=',os.environ['XUVTOP']); ch.ion('c_3', temperature=1e4)"
```

---

## 3. Cloudy C23.01 (for `$CLOUDY_EXE`)

Cloudy supplies the self-consistent resonance-line RT for the metal **absolutes**
(C IV 1549 etc.). Build it from source:

```bash
cd ~
curl -LO https://data.nublado.org/cloudy_releases/c23/c23.01.tar.gz
tar xzf c23.01.tar.gz
cd c23.01/source
make                       # ~5-10 min
```
- **macOS:** if `make` fails with `Undefined symbols ... _main` (a wrong default
  compiler, e.g. the MESA SDK g++), force the system clang:
  ```bash
  make CXX=/usr/bin/clang++
  ```
- The build produces `c23.01/source/cloudy.exe`. Point `$CLOUDY_EXE` at it:
  ```bash
  export CLOUDY_EXE=~/c23.01/source/cloudy.exe
  ```
Verify:
```bash
echo "test" | "$CLOUDY_EXE"     # prints a Cloudy banner and exits cleanly
```
> SNLT also auto-discovers Cloudy at `~/c23.01/source/cloudy.exe` if `$CLOUDY_EXE`
> is unset.

---

## 4. Environment variables (put these in your shell profile)

```bash
# ~/.zshrc or ~/.bashrc
export XUVTOP=~/Documents/SNLT/chianti
export CLOUDY_EXE=~/c23.01/source/cloudy.exe
```
Re-`source` your profile (or open a new shell) so every run inherits them.

---

## 5. Repository layout (IMPORTANT — how model dirs find the code)

```
SNLT/
├── src/                      # ALL pipeline code lives here
│   ├── production_runner.py  # main driver
│   ├── photoionize_csm.py, phase5_runner.py, metal_*.py, ...
│   └── grand_comparison.py, analyze_correlations.py, ...   # analysis
└── input_models/
    ├── C4/                   # one directory PER model
    │   ├── mesa.day*_post_Lbol_max.data   # STELLA snapshots (input)
    │   ├── mesa.lbol                       # bolometric lightcurve (input)
    │   └── *.py -> ../../src/*.py          # SYMLINKS to the code
    └── ...
```

**Each model directory must symlink the code modules from `src/`.** You run the
pipeline *from inside* a model dir, and Python imports the modules sitting next to
`production_runner.py` (which are symlinks pointing back to the single source of
truth in `src/`). This guarantees every model runs the exact same, latest code.

### Setting up a NEW model directory
If you add a model (drop its `mesa.day*` snapshots + `mesa.lbol` into a new dir),
create the symlinks by copying an existing model's set:
```bash
cd ~/Documents/SNLT
NEWDIR=input_models/MYMODEL
for f in input_models/C4/*.py; do
    ln -sfn "../../src/$(basename "$f")" "$NEWDIR/$(basename "$f")"
done
```
> **Do NOT `cp` modules into a model dir** — a real copy goes stale the moment
> `src/` is edited. Always symlink. (You can audit with:
> `find input_models/MYMODEL -maxdepth 1 -name '*.py' -type l | wc -l` — it should
> match the count in a working dir, currently 38.)

---

## 6. Input data

Each model dir needs:
- **`mesa.day<NNN>_post_Lbol_max.data`** — STELLA snapshots (1D spherical state),
  one per epoch. The epoch label is days **post-L_bol-max** (negatives allowed,
  e.g. `day-005`).
- **`mesa.lbol`** — the bolometric lightcurve: header row, then columns
  `time[d]  L_ubvri  log10(L_bol)  ...` (time origin ≈ L_bol-max). Used for the
  lightcurve panel in the movies.

---

## 7. Run a model

From inside a model directory:
```bash
cd ~/Documents/SNLT/input_models/C4
python production_runner.py --batch --batch-format stella \
    --line-profile-method formal --he-lines --he1-nlte --he2-nlte \
    --saturated-rt --he-budget --metal-lines --metal-cloudy
```

Flag cheat-sheet:
| flag | meaning |
|------|---------|
| `--batch --batch-format stella` | process every `mesa.day*` snapshot in this dir |
| `--he-lines --he1-nlte --he2-nlte` | He I / He II NLTE line solvers |
| `--saturated-rt` | escape-probability RT for optically-thick lines |
| `--he-budget` | composition-general continuum-collapse guard (auto for H-free) |
| `--metal-lines` | compute C/O/Ne metal lines (Phase 5c) |
| `--metal-cloudy` | use Cloudy for the metal *absolutes* (needs `$CLOUDY_EXE`) |
| `--epochs 3,5,10` | (optional) only these epochs |

**Outputs** (per epoch + per model, written into the model dir):
- `prod_day<NNN>_lines.npz/.txt/.png` — per-line L, EW, τ, profiles
- `prod_day<NNN>_metal_lines.png` — the 6-panel metal figure
- `batch_lines_evolution.mp4`, `batch_metal_evolution.mp4`, `batch_he_evolution.mp4`
  — evolution movies (phase title + bolometric-lightcurve panel)
- `batch_metal_grid.png`, `batch_he_grid.png` — static evolution grids

> Single snapshot instead of a batch:
> `python production_runner.py path/to/mesa.day010_post_Lbol_max.data --he-lines ...`

---

## 8. Cross-model analysis (after running several models)

From the repo root:
```bash
# ranking of which features dominate each epoch/model + metal metrics
python src/grand_comparison.py --outdir analysis/

# master table + Pearson/Spearman correlations + PCA vs physical params
python src/analyze_correlations.py --runs A1:input_models/A1 A4:input_models/A4 ... --outdir analysis/

# line budget vs M_csm / E_SN, per family
python src/plot_compare_runs.py --runs A1:input_models/A1 ... --species balmer --all --outdir analysis/

# single-model 4-panel figure
python src/plot_single_run.py --run input_models/C4 --species he --out C4_he.png
```
Model physical properties live in `src/snline_postproc.py` (`_DEFAULT_MODELS`);
add your model there (or pass `--model-table my.csv`) so correlations can use it.

---

## 9. Quick install smoke-test

```bash
cd ~/Documents/SNLT
python - <<'EOF'
import os, importlib, sys
sys.path.insert(0, 'src')
for m in ['numpy','scipy','numba','matplotlib','pandas','ChiantiPy']:
    importlib.import_module(m)
print('python deps OK')
print('XUVTOP   =', os.environ.get('XUVTOP'),    '(exists:', os.path.isdir(os.environ.get('XUVTOP','')), ')')
exe = os.environ.get('CLOUDY_EXE', os.path.expanduser('~/c23.01/source/cloudy.exe'))
print('CLOUDY_EXE=', exe, '(exists:', os.path.isfile(exe), ')')
import production_runner   # imports the whole pipeline
print('pipeline imports OK')
EOF
```
All four "OK"/"exists: True" → you're ready. Then run one model (§7).

---

## 10. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: numba` | wrong/empty venv, or Python too new — recreate venv on Python 3.11–3.13 |
| `No module named 'ChiantiPy'` | `pip install ChiantiPy` **and** set `$XUVTOP` to the CHIANTI **database** dir |
| metals all say `[prov]`/`PROVISIONAL` | ChiantiPy or Cloudy not found — check `$XUVTOP` / `$CLOUDY_EXE` (metals still run, just on fallback data) |
| `[cloudy] ... did not yield lines` repeatedly | Cloudy not built or `$CLOUDY_EXE` wrong; verify §3. Single-epoch crashes preserve the deck under `./cloudy_failures/` |
| `ModuleNotFoundError` for a pipeline module when running in a model dir | that dir is missing a symlink — re-run the symlink loop in §5 |
| a model runs *old* code after you edit `src/` | the dir has a **real copy**, not a symlink — `ls -l` it; replace with a symlink (§5) |
| movies are `.gif` not `.mp4` | install `ffmpeg` (e.g. `brew install ffmpeg`) |
| `Truncation ... leaves only N zones` (no-CSM controls) | expected at edge epochs for M_csm=0 models; that epoch is skipped, the batch continues |

For the physics, design decisions, trust ceilings, and validated results, see
`CLAUDE.md` (handover), `QUICK_REFERENCE.md`, and `FUTURE_WORK.md`.
