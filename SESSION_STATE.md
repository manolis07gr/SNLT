# SNLT — Session handoff / resume state (live)

Snapshot of in-flight work so a new session (or post-summarization continuation)
picks up with zero context loss. Durable companions: `CLAUDE.md` (physics
handover), `FUTURE_WORK.md` (staged plan, esp. **P2 #7**), `GETTING_STARTED.md`.

## TL;DR — where we are
- Branch **`p0-p1-physics`** (== `main`), all code committed + pushed.
- **Two production grids DONE & analyzed:** the original 15 models, then the
  expanded **39-model grid + GO1** (blind test). Outputs in `analysis/` and
  `analysis_full/`.
- **GO1 blind test PASSED:** from line profiles alone the pipeline called it a
  Type **Icn** (H-free, C III]/C IV carbon-dominated, He-bearing), **massive CSM
  (predicted ≳2 M⊙; truth M_csm=5.0)**, **energetic (predicted elevated; truth
  E=2.0)**, fast dense CSM (truth v_csm=3000). Strong end-to-end validation.
- **Now executing the optical line-list upgrade** (FUTURE_WORK P2 #7), gated,
  backwards-compat-enforced. **Stage 1 (narrow-CSM profile) integrated; its
  backwards-compat regression is RUNNING.**

## Validated physics shipped this session (do not regress)
Smooth shock-X-ray escape gate (binary→`exp(-τ)`, killed the metal breakout
flicker); Cloudy robustness (dlaw strict-monotonic radii fix → 0 readLaw across
both grids; timeout 1800 s; iterate max 6, carbon lines converge ~1%); C III λ4650
ORL parent-ion + branching×total fix; boxy-width validation (#2); negative-epoch
regex (GO1 pre-max); registry expanded to 39 models. Key commits: `a86e1a9`,
`449d3fc`, `b4b4cc1`, `74999f8`, `45259a9`, `b5cdf2b`, `0ddb3fe`.

## What is RUNNING / SCHEDULED right now
- **Detached:** `/tmp/reg_run.sh` — Stage-1 backwards-compat regression (C4/A1/A4,
  off+on single-snapshot runs → `/tmp/reg/`, marker `/tmp/reg/COMPLETE`). Survives
  anything (OS-level).
- **Scheduled wakeup** (this session): evaluates the regression via
  `/tmp/reg_compare.py`, commits Stage 1 if clean, then drives Stages 2→4. The
  wakeup prompt is fully self-contained.

## The line-list plan (FUTURE_WORK P2 #7) — current position
1. **Stage 1 — narrow-CSM P-Cygni** ✅ built (`csm_narrow_profile.py`, unit-tested)
   + integrated into `metal_lines` behind **`--narrow-csm` (DEFAULT OFF =
   byte-identical)**. ⏳ **regression in flight** (acceptance: default-off matches
   baseline `/tmp/regress_baseline/baseline.pkl`; narrow-on keeps every `L_corr`
   identical, changes only metal profiles, H/He untouched).
2. **Stage 2 — optical C lines** (C III 5696, C IV 5801/12, C II 4267/6580…),
   chosen by mining Cloudy's full `save line list`; add to `metal_atoms.METAL_LINES`.
3. **Stage 3 — O/Ne/Mg lines.**
4. **Stage 4 — `synthetic_spectrum.py`**: continuum + Σ line profiles on a
   3500–9500 Å grid (narrow-CSM on) → overplot vs the real Icn spectra in
   `obs_comparison/` (SN 2019hgp, 2021csp — TNS public data, already downloaded).
**Hard gate after every stage:** existing 19-line `L_corr` unchanged, H/He untouched.

## Environment (essential)
- **Prod venv (numba+ChiantiPy):** `/Users/manoschatzopoulos/Downloads/LINE MODELING/path/to/venv/bin/python`
- numpy-only test venv: `/Users/manoschatzopoulos/Downloads/claude_snrt/.venv/bin/python`
- `export XUVTOP=~/Documents/SNLT/chianti` · `export CLOUDY_EXE=~/c23.01/source/cloudy.exe`
- **Run a model** (from a model dir): `python production_runner.py --batch --batch-format stella --line-profile-method formal --he-lines --he1-nlte --he2-nlte --saturated-rt --he-budget --metal-lines --metal-cloudy`
- **Model dirs symlink `src/` — NEVER `cp` a module into a model dir (goes stale).**
  New module → symlink into all dirs: `for d in input_models/*/; do ln -sfn ../../src/NEW.py "$d"; done`

## How to RESUME in a fresh conversation
1. Read `CLAUDE.md`, `FUTURE_WORK.md` (P2 #7), and this file.
2. `git -C ~/Documents/SNLT log --oneline -15` to see the latest state.
3. Check the line-list progress: is `/tmp/reg/COMPLETE` present? did Stage 1
   commit? `git log` will show "Stage 1 ... narrow-CSM integrated" if it passed.
4. If the autonomous wakeup chain didn't carry over, re-establish it: run
   `/tmp/reg_compare.py`, then continue Stage 2→4 per P2 #7.
5. The full grids are done — **do NOT re-run all 40 models** unless explicitly
   regenerating the final paper dataset; validation uses A1/A4/C4 at a few epochs.
