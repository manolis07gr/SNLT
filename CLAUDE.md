# SNLT pipeline — Handover

A self-contained summary of what the pipeline does, the physics it implements,
the file inventory, and the validated results — written so a future session can
pick up without re-deriving the history.

---

## 1. What this is

A post-processing pipeline that takes STELLA supernova snapshots
(`mesa.day*_post_Lbol_max.data`, 1D spherical) and computes **physical Hα — and
all-line Balmer / Paschen / He I / He II — luminosities and profiles**, from
first principles (no artificial floors or normalizations), across the full set
of CSM-interaction regimes:

- **Type IIP** (fast homologous H-rich ejecta, no CSM),
- **Type IIn** (slow dense H-rich CSM interaction),
- **Stripped / Ibn-like** (H-free, He-rich CSM interaction).

It was validated against the SuperLite Model A1 IIP Hα (rest-peak amplitude ≈ 1.3).

## 2. The physics, in brief

- **STELLA ingest + photosphere truncation** (`stella_io.py`): loads the
  snapshot, derives shock parameters, truncates to the photosphere, applies
  per-zone photoionization equilibrium (CLOUDY/CMFGEN-style) including the
  shock-bremsstrahlung X-ray component.
- **Ionization / excitation**: per-zone H NLTE (`h_populations_nlte.py`) and He
  I / He II NLTE solvers (ionization + excitation incl. collisional channels)
  driven by the STELLA state + photoionization.
- **Profile transport** (`mc_multi_line.py`, `formal_line_profile.py`): a Monte
  Carlo peel-off transport for the emergent profile, plus a Sobolev
  source-function P-Cygni ("formal") solver for homologous ejecta.
- **The Sobolev-validity GATE** (the central design decision): per snapshot it
  measures the homology of the velocity field
  (`sobolev_validity(r, v)` → `std(r/v)/median`, reversal fraction). If
  homologous (IIP-like) it uses the **formal** P-Cygni solution; if
  non-homologous (dense-CSM / IIn-like) the formal solution is unphysical
  (it produces an absorption trough), so it keeps the **MC** emission-line
  profile. **The gate is always authoritative for the profile shape** — nothing
  (no CLI flag) can force `formal` onto a dense-CSM snapshot.
- **Line strengths** (`phase5_runner.py`): H lines use a **per-line
  recombination budget** on the validated production-Hα scale — the budget
  *ratio* L(line)/L(Hα) cancels the absolute escape problem and reproduces the
  case-B decrement (Hα/Hβ ≈ 2.9, Hα/Hγ ≈ 2.1×2.9). He lines take their strength
  directly from the He-NLTE solvers (first-principles), with a single-shot
  escape correction (~factor-2) for the optically-thick ones. There is **no
  empirical Hα anchor** in this path.
- **Profile display (`phase5_runner.py`)** — two corrections so the F/F_cont
  panels are interpretable in the dense-CSM (high-τ_es) regime:
  - **Adaptive velocity window**: the per-line window widens to the
    emission-measure-weighted (n_e²·dV) v95 of the line-forming gas (×1.6,
    floored at the requested ±win_kms, capped ±25000), with n_pix scaled to keep
    resolution. A fixed ±5000 clipped broad/fast-photosphere lines and starved
    the continuum baseline. Slow/homologous models (A1/B1) keep ±5000 unchanged.
  - **Emergent-continuum renorm** (`_renormalize_to_emergent_continuum`): at high
    τ_es the directly-escaping continuum is well below the un-attenuated BB, so
    BB-normalized He lines sat at F<1 in line-free regions while the
    emergent-normalized H/production lines sat at 1.0. Each H/He line is now
    divided by its clean far-wing (emergent) continuum → all read 1.0
    consistently (observer convention; L_line untouched, EW recomputed). Metals
    are already line-centre-normalized and skipped.

## 3. Known trust ceiling (important)

- Production Hα and the **corrected** luminosities (`L_corr`) are the
  trustworthy outputs; factor-of-two for optically-thick lines.
- **Saturated lines** (τ≫1, common in the interaction phase): profile *shapes*
  and profile-integrated *EWs* are not reliable (single-shot kernel). Use the
  production Hα for the Hα shape; for a thick line's EW prefer
  L_corr / L_cont_band over the profile-integrated value. **Update (P1 #3):**
  `--saturated-rt` drops the empirical Hα anchor for thick He lines (keeps the
  first-principles single-shot β escape luminosity) and applies multiple
  electron scattering to the shape; the residual ~factor-2 (nonlocal ALI) is
  still an explicit uncertainty. See `line_rt_escape.py` + FUTURE_WORK P1 #3.
- **He II lines are ~0** in all models computed so far (no He²⁺ source): correct.
- **Early-epoch lines are strongly BLUE-PEAKED — this is physical, not a bug.**
  When the photosphere is fast (e.g. C1 day 10: v_phot ≈ 6700 km/s) the
  recombination emission (∝ n_e²) forms in a thin layer at R_phot and the
  receding hemisphere is occulted by the photospheric disk, so the emergent line
  peaks near −v_phot with an electron-scattering red wing (the MC computes this
  correctly). The deeper formation-layer emission Thomson-diffuses out and
  emerges from the *same* photosphere, also blue-peaked — so it does NOT
  symmetrise the line. (A `--spread-emission` blend that symmetrised the profile
  was prototyped and **removed** as physically incorrect: the diffuse
  photospheric emission distribution ∝ |v_los| is peaked at −v_phot, not flat.)
  The blueshift recedes with epoch as the photosphere slows — a back-test
  consistency check (`backtest/`).
- At the interaction-brightening / continuum-collapse epochs, peak-F is inflated
  by the collapsing continuum — **quote L_line, not peak-F**.
- **Late post-interaction / nebular epochs are not paper-ready** (continuum
  collapse; formal solver mis-applied; τ_es → nebular). Truncate there.
  **Update (P1 #4):** the unphysical `L_cont_band` Wien-collapse is now floored
  energy-consistently by `--he-budget` (auto for H-free); see
  `continuum_compgen.py`. The nebular RT itself is still out of scope.
- **Shared-late-snapshot bug** — **FIXED (P0 #1).** The `--batch` loader now
  content-hashes each STELLA snapshot against the same-named file in sibling
  model dirs and skips byte-identical shared placeholders (warns per epoch);
  override with `--keep-shared-snapshots`. No-op outside the model-grid layout.

## 4. File inventory

**Core pipeline (do not lose):**
- `production_runner.py` — main driver (`--batch` / single snapshot). Gate is
  authoritative; `--line-profile-method-lock` is a no-op for shape (kept for
  CLI compatibility, must not be used to force formal).
- `phase5_runner.py` — multi-line strengths (recombination budget + He-NLTE),
  writes `*_lines.npz/.txt/.png` with consistent corrected (`L_corr`/`EW_corr`)
  values.
- `formal_line_profile.py` — Sobolev formal solver + `sobolev_validity` gate +
  recombination-budget coefficients.
- `mc_multi_line.py` — MC peel-off transport.
- `line_rt_escape.py` — **(P1 #3)** saturated-line RT: escape-probability
  luminosity identity, continuum-pumped EP source (diagnostic), and the
  `thomson_multiscatter` shape redistribution. Opt-in via `--saturated-rt`.
- `continuum_compgen.py` — **(P1 #4)** composition-general continuum-collapse
  guard (energy-conserving `L_cont_band` floor), energy-conservation check,
  H-free composition switch, first-principles He decrement. `--he-budget` /
  auto when ⟨X_H⟩ < 1e-3.
- `regime_diagnostics.py` — per-line grade/regime + paper-action text (now
  reports the `EP-esc` method under `--saturated-rt`; grade 'N' when the line's
  element is absent, ⟨X_elem⟩ < 1e-3).
- **Metal lines (P2 #5)** — opt-in Phase 5c for C/O/Ne (Icn / late Ibn):
  - `metal_lines.py` — Stage-3 driver: per-line emissivity → resonance β →
    adaptive velocity window → MC peel-off profile (shape) → EW guard. Tiered
    strength: Cloudy → CHIANTI → provisional; `save_metal_png`. `--metal-lines`.
    Resonance lines (C IV 1549) get a photon-conserving Sobolev **P-Cygni**
    overlay (`pcygni_absorption_overlay`: blue trough + re-emitted continuum, net
    EW = thermal/Cloudy). With `--metal-cloudy`, the MC shape is weighted by
    Cloudy's per-zone emissivity (`shape_source='cloudy'`).
  - `metal_atoms.py` — provisional atomic data + emissivity (recomb / CEL with
    n_crit), `f_lu` resonance oscillator strengths, line colors/pretty names.
  - `metal_ionization.py` — photoionization-equilibrium ion ladder (diluted-BB +
    shock-brems Γ rates), self-contained.
  - `metal_nlte.py` — **Tier-1**: authoritative CHIANTI NLTE emissivities via
    ChiantiPy (`ion.emiss()`); needs ChiantiPy + `$XUVTOP`. Falls back silently.
  - **Cloudy robustness (time-series consistency):** Cloudy is used for the
    ABSOLUTE only on genuine **resonance lines** (f_lu>1e-4 → C IV 1549), where
    its resonance-line RT is the unique contribution; intercombination/forbidden/
    recombination lines (C III] 1909, [O III], [Ne III], …) take the **stable
    CHIANTI** NLTE absolute, so they don't flicker on Cloudy's per-epoch thermal-
    bistability variability. An **energy-conservation ceiling** (`L_MAX_FRAC=0.1`
    of L_phot, in `metal_cloudy` + `metal_lines`) rejects any line that converges
    onto the wrong branch and reports tens-of-% of L_bol → falls back to CHIANTI.
    **Cloudy intermittency — root-caused + FIXED.** Cloudy was crashing on a
    single epoch's deck (`[Stop in readLaw … Radii must be in increasing order]`)
    while succeeding on its neighbour, so the resonance line (C IV 1549) fell back
    to the CHIANTI single-β value — a ~5-dex UNDERESTIMATE for a β≈1e-7 thick
    resonance line — making C IV flicker (e.g. C4 day5 Cloudy 5.7e40 vs day10
    fallback 8.6e34, both C³⁺≈0.9). Cause: STELLA piles many zones at near-equal
    radii in the dense shock shell; at the `%.6f` log precision the deck writes,
    they collapse to **exact-duplicate dlaw rows** (135 of them in a 552-zone
    day-10 deck) and Cloudy's `readLaw` rejects non-strictly-increasing radii.
    Fix in `build_deck`: enforce strict monotonicity ON THE WRITTEN GRID via
    integer micro-log units (force each row ≥ prev + 10⁻⁶ in log r; a ≤few-ppm
    nudge). Verified on the preserved deck: readLaw abort gone, day-10 C IV →
    8.3e40 (matches day5). Also surfaced that these dense decks are slow
    (~250 s / 3 iterations, resonance-RT-limited, NOT dlaw-size-limited), so
    `run_cloudy` timeout was raised 240→480 s so they complete instead of timing
    out into the CHIANTI fallback. `run_cloudy` also now PRESERVES any failing
    deck to `./cloudy_failures/` (or `$SNLT_CLOUDY_DEBUG`). **Convergence —
    resolved.** A 12-iteration test on the densest C4 deck showed the CARBON lines
    we extract converge to ~1% by iteration 4-5 (C IV 1549: 8.25e40→8.19e40→
    8.10e40 over iters 3/4/5; C III 1909 stable to <2% from iter 2); the deck's
    global "did not converge" is tripped only by an unrelated *He-like
    subordinate* line, not the metal lines. So `iterate to convergence max` was
    raised 3→6 (the lines fully settle; Cloudy stops early when converged, so
    sparse fast epochs cost nothing) and the timeout 480→720 s. The thick
    resonance absolute is now converged to ~1%, not a factor.
  - `metal_cloudy.py` — **Tier-2** (`--metal-cloudy`): override metal ABSOLUTES
    with Cloudy (self-consistent photoionization + NLTE + resonance-line RT),
    fixing C IV 1549 / C III] 1909. Builds a deck from the STELLA state
    (BB + shock-brems field, spherical `dlaw`, He-anchored abundances + trace-H
    floor for the H-free C-series), runs Cloudy, parses `save line list`
    (absolutes) + `save line emissivity` (per-zone, → MC shape weighting). MC
    keeps the velocity SHAPE. Locate via `$CLOUDY_EXE` or
    `~/c23.01/source/cloudy.exe`; any failure → CHIANTI/provisional per line.
  - `validate_metals.py` — analytic harness (n_crit limits, ladder norm).
- `photoionize_csm.py`, `stella_io.py`, `h_populations_nlte.py`,
  `phase5_continuum.py`, `snapshot_analyzer.py`, `opacity.py` (He free-free,
  P1 #4 root-fix) — supporting physics.
- `make_phase5_movie.py` — standalone multi-line evolution movie from `*_lines.npz`
  (`--species all/metal/he/h` filter).
- `fix_he_strengths.py` — post-processor to re-derive He strengths from existing
  npz without an MC re-run.

**Post-processing / analysis (this session):**
- `snline_postproc.py` — shared data layer: npz loader, model-property registry
  (A/B/C series prefilled), epoch/line selection, e-folding triplet, data-driven
  strongest-He selection.
- `plot_single_run.py` — single-run 4-panel evolution figure (Balmer or He).
- `plot_compare_runs.py` — cross-run comparison, peak ∓ e-folding, vs M_csm/M_ej
  or E_SN.
- `analyze_correlations.py` — master table + Pearson/Spearman + PCA.
- `validate_p1_physics.py` — standalone analytic-limit harness for P1 #3 + #4
  (no pipeline/snapshot needed); `validate_p1_continuum.py` holds the #4 checks.
- `backtest/run_backtest.sh` + `backtest/check_backtest.py` — sparse-grid
  consistency harness: runs A1/A4/B4/C1/C4 at days 0.1/1/3/5/10/20/30/40/50/80/100
  (via the `--epochs` keep-only batch filter) with the full current physics, then
  checks continuum=1.0, H-rich vs H-free composition, and the blueshift-recedes-
  with-epoch trend across all regimes; writes `backtest_metrics.csv`.
- `QUICK_REFERENCE.md`, `FUTURE_WORK.md`, this file.

**Data contract:** per-epoch `prod_day*_lines.npz` holds `line_names`,
`lambda_rest`, `L_line`, `L_line_corrected`, `EW`, `EW_corrected`, `tau_med`,
`L_cont_band` (all length-13, aligned to `line_names`) plus per-line profile
arrays `<line>__lambda`, `<line>__F_norm_corrected`. The 13 lines are
He_II_1640/3203/4686/10124, He_I_5876/6678/7065/10830, Halpha, Hbeta, Hgamma,
Palpha, Pbeta. Epoch is parsed from the filename. `batch_metrics.csv` carries
the Hα-production line + per-epoch structural/ionization diagnostics (R_phot,
T_phot, τ_es, X_HII, He densities, shock) — but **not** the per-line L/EW (those
are in the npz).

## 5. Model grid run so far

Three progenitor families spanning (CSM mass, progenitor compactness, CSM
composition):

- **A** — RSG (603 R⊙), H-rich envelope + H-rich CSM, E_SN=0.78. A1 (no CSM,
  IIP control) + A4/A5/A6/A7 (M_csm = 0.2/0.5/1/2 M⊙).
- **B** — YSG (525 R⊙), more stripped but H-rich, lower E_SN=0.28. B1 + B4–B7
  (same CSM ladder).
- **C** — stripped He-star (7.24 R⊙), **H-free / He-rich** CSM, E_SN=0.80.
  C1 + C4–C7.

## 6. Validated results / physical findings

- **A1 / B1 / C1 controls** behave correctly: A1/B1 give textbook IIP/II Hα
  P-Cygni at all epochs (gate → formal); C1 (no H) gives Hα ≈ 0 — a clean null
  confirming the pipeline does not manufacture hydrogen.
- **Within each H-rich family (A, B):** interaction-phase duration and Hα
  luminosity both scale monotonically with M_csm; the IIn→II transition (CSM
  consumed, continuum collapse, photosphere recedes into homologous ejecta)
  moves later with more CSM — A: ~day 130→150 across A4→A7; B (less energetic,
  more compact) transitions ~10–20 d earlier at matched CSM mass.
- **Across families:** at fixed CSM mass the more compact / lower-energy B gives
  a somewhat weaker, earlier-peaking signature than A — a progenitor-structure
  effect at fixed wind.
- **C-series (H-free):** clean He recombination wave (He²⁺ → He⁺ → He⁰ as the
  CSM cools), shock-X-ray-dominated ionization (~10× the A-series), He lines the
  only emitters. Qualitatively coherent in the interaction phase; absolute He
  fluxes await the He-regime continuum/RT validation (Future Work P1).
- **Ionization physics is self-consistent** everywhere: H and He recombination
  waves track temperature; He²⁺ ≈ 0 (no hot source); optically-thick He I lines
  flagged factor-2.

## 7. The one big methodological fix in the history

The `--line-profile-method-lock` flag originally **bypassed the gate and forced
`formal` on every epoch**. Harmless for homologous IIP models, but it forced the
broken P-Cygni absorption onto the entire IIn series ("looked like another
IIP"). Resolution: the Sobolev gate is now **always** authoritative for the
profile shape; the lock only signals strength-policy consistency and can never
force `formal`. **Never reintroduce a flag that overrides the gate.**
