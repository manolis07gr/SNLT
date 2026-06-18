# SNLT pipeline — Future Work (prioritized)

Ordered by priority. Each item lists the gap, why it matters, and a concrete
first step. **Status tags:** ✅ DONE · ◑ PARTIAL · ☐ OPEN.

---

## P0 — Data-plumbing fixes (cheap, blocking for publication)

**0. Binary shock-X-ray escape gate flickered the metal high-ion lines.  ✅ DONE.**
~~The full-grid C4 run revealed C IV 1549 oscillating at the shock-breakout
epochs: day4 1.3e35 → day5 5.7e40 → day6 1.8e35 → day7 6.0e40. The line tracks the
carbon ionization, which flickered: C³⁺ = 0.0002 (d4) / 0.96 (d5) / 0.00 (d6) /
0.89 (d7). Root cause was the **binary** interior-shock gate in
`photoionize_csm.py` (`shock_escapes = R_s >= R_phot`): around breakout the shock
sits *at* the photosphere, so `R_s` crosses `R_phot` between snapshots and the
`>=` toggled the ENTIRE shock X-ray field on/off.~~ *Fixed:* `derive_shock_params`
now computes a SMOOTH transmission `f_xray_escape = exp(−max(0, τ_es@shock −
τ_phot_ref))` from the overlying electron-scattering column (the full snapshot's
cumulative `tau_es`, photosphere ref = the run's `tau_es_phot`), stored in
`photoionization_params`. The H/He field (`solve_photoionization_equilibrium`
scales `xray_coef`) and the metals (`phase5_runner._build_merged_state` scales
`merged.L_X_brems`, which `metal_lines`→`metal_ionization` read) both multiply by
it instead of the old all-or-nothing gate; falls back to the binary flag for
pre-existing snapshots. **Validated on C4 day4-7:** f_xray_escape now
0.74/1.00/0.98/1.00 (was 0/1/0/1), C³⁺ now 0.997/0.963/0.987/0.893 (was
~0/0.96/0/0.89), C IV now 2.16e41/5.74e40/1.02e41/5.98e40 — smooth, no 5-dex
collapse, and the already-escaping epochs (day5/7, f=1) are bit-unchanged.
Physically correct (X-ray escape is exponential in overlying τ, not a step) and
regime-general (feeds H/He and metals identically across A/B/C).

**1. Shared late-epoch snapshot bug.  ✅ DONE.**
~~The batch loader substitutes a single common file for late epochs when a
model-specific snapshot is missing, producing byte-identical day-150/160 (A),
day-120–160 (C), day-160 (B) rows across *all* models — including no-CSM
controls.~~ *Fixed* in `production_runner.py` (`detect_shared_snapshots`): in
`--batch` STELLA mode the loader content-hashes each snapshot against the
same-named file in sibling model directories and SKIPS any byte-identical copy
(a non-model-local shared placeholder), emitting a per-epoch warning. Override
with `--keep-shared-snapshots`. No-op outside the model-grid layout. The skip
coincides with the physics-motivated continuum-collapse truncation.

**2. Regime-grade should know about composition.  ◑ PARTIAL.**
`Halpha_prod` is still graded "A" even in H-free models. The composition switch
needed for the fix now exists (`continuum_compgen.is_h_free`, ⟨X_H⟩ < 1e-3, and
`mean_X_H`); the remaining work is to wire it into `regime_diagnostics` to gate
the H-line grades on a minimum H abundance / `L_line` floor (and He-line grades
likewise) and emit "no element" instead of a trust grade.

---

## P1 — Physics for the regimes we already touch

**3. Saturated-line radiative transfer (IIn / Ibn interaction phase).  ✅ DONE
(local EP; nonlocal ALI deferred).**
~~The single-shot kernel cannot represent saturated transport (τ≫1)...~~
*Resolved* behind the opt-in `--saturated-rt` flag (`line_rt_escape.py`):
  - **Luminosity:** a numeric check proved the single-shot β luminosity is
    *already* the first-principles escape-probability luminosity for the He-NLTE
    populations (`escape_probability_luminosity()`; identity verified to 1e-5).
    So the fix DROPS the empirical Hα-anchored `R_flat` for thick He lines and
    keeps the bare β value — removing the empirical anchor (the stated goal).
  - **Shape:** `thomson_multiscatter()` applies the multiple-electron-scattering
    redistribution (photon-conserving; broadens wings, suppresses peak).
  - The mode label is `He-NLTE(thick,EP-esc)`; `regime_diagnostics` reports it
    honestly (no longer claims an Hα anchor).
*Deliberately deferred:* the **nonlocal iterated-J̄ / ALI** solver. A local
closed-form EP cannot beat single-shot β without the continuum-pumped source,
which over-pumps recombination lines by ~10³× — so the residual ~factor-2
(nonlocal escape suppression) is still an explicit uncertainty, not papered over.
`ep_source_function()` is retained as a diagnostic / ALI foundation.

**4. Continuum / energy budget without hydrogen.  ✅ DONE (guard + diagnostics;
upstream opacity deferred).**
~~The recombination-budget and continuum normalization were built for H-rich
gas... `L_cont_band` collapses unphysically at cold compact late epochs.~~
*Resolved* behind `--he-budget` (AUTO-enabled when ⟨X_H⟩ < 1e-3) in
`continuum_compgen.py`:
  - **Composition-general continuum-collapse guard:** detects the Wien collapse
    (`4πR²σT_phot⁴ ≪ L_phot`) and floors each line's `L_cont_band` to the
    energy-conserving color temperature (`4πR²σT_floor⁴ = L_phot`), per line via
    the Planck ratio. Fixes `L_corr/L_cont_band` EW estimates. Profile shapes
    untouched. (Validated on C1: correctly *no-op* at warm day100, floors at the
    cold late epochs.)
  - **Energy-conservation check** (Σ L_line ≤ L_phot) and a **first-principles
    He decrement** referenced to He I 10830 (no external anchor — per the chosen
    budget policy).
*Deliberately deferred:* the upstream root-fix — He bound-free/free-free
continuum opacity in `opacity.py` / `photosphere_v2.py` so the H-free photosphere
is placed correctly in the first place (touches the most-validated module). The
guard corrects the *symptom* (collapsed `L_cont_band`) energy-consistently; the
opacity fix would remove the *cause*.

---

## P2 — New emitting species

**5. Metal lines for H- and He-poor CSM interaction. — DONE (Tier-1 + Tier-2).**
Add the strongest **carbon, oxygen, and neon** lines (C IV 1549, C III] 1909,
C III 4647, [O I] 6300, [O III] 5007, [Ne III] 3869) expected from stripped,
metal-enriched CSM interaction (Icn / late Ibn). Implemented as opt-in Phase 5c:
- `--metal-lines` — first-principles per-zone emissivity integrals
  (`metal_atoms.py`: recombination + collisional CEL with the n_crit correction)
  on photoionization-equilibrium ion densities (`metal_ionization.py`,
  shock-X-ray-aware), transported through the **same MC peel-off kernel** as the
  He lines (`metal_lines.mc_metal_profile` → velocity field + multiple electron
  scattering) for realistic profiles, with an adaptive emissivity-weighted
  velocity window and a continuum-suppression EW guard. Merged into the npz /
  regime / plots / movies dynamically (the schema is data-driven). A dedicated
  `{prefix}_metal_lines.png` multi-panel and a `--species metal` movie filter
  are added.
- **Tier-1 (`metal_nlte.py`, CHIANTI/ChiantiPy):** replaces the provisional
  collisional emissivities with authoritative multi-level NLTE values from the
  CHIANTI database (`ion.emiss()`), per line, when ChiantiPy + `$XUVTOP` are
  present. Falls back to provisional otherwise.
- **Tier-2 (`metal_cloudy.py`, Cloudy, `--metal-cloudy`):** overrides the metal
  ABSOLUTE luminosities with **Cloudy** — a self-consistent photoionization +
  multi-level NLTE + **resonance-line RT** solve — fixing the two approximations
  Tier-1 leaves open (the ion balance and the resonance-line escape for
  C IV 1549 / C III] 1909). One Cloudy run per snapshot translates the STELLA
  state into a deck (photospheric BB + shock-bremsstrahlung incident field,
  spherical `dlaw` density profile, He-anchored abundances with a trace-H floor
  for the H-free C-series); the emergent `save line list` luminosities replace
  L_line per line. The MC keeps the velocity-resolved SHAPE (Cloudy is static).
  Strength tiering is per-line and graceful: Cloudy → CHIANTI → provisional;
  any Cloudy failure (not installed, abort, non-convergence) silently falls back.
  Requires Cloudy compiled and locatable via `$CLOUDY_EXE` or
  `~/c23.01/source/cloudy.exe`. Each line carries a `data_source` tag
  (`Cloudy-Tier2` / `CHIANTI-NLTE` / `provisional`) and the pre-override
  `L_emiss` for audit.

Two profile-physics refinements are also in (both validated on C1 day010):
- **Item 1 — resonance-line P-Cygni.** Resonance metal lines (C IV 1549; f_lu>1e-4)
  now get a photon-conserving Sobolev P-Cygni profile, not emission-only:
  `metal_lines.pcygni_absorption_overlay` overlays a blue ABSORPTION trough
  exp(−τ(Δv<0)) on the continuum and RE-EMITS the absorbed continuum
  (∫=L_abs·P_emiss) so the net EW equals the THERMAL emission (= Cloudy's net),
  with no double-counting and no spurious net absorption. Gated by a real
  continuum (skipped when cont-suppressed) and the radial Sobolev τ. Forbidden /
  intercombination lines (f_lu≈0) stay emission-only. Tagged `pcygni` per line +
  "P-Cyg" in the PNG. (Residual: no stellar-disk occultation of the receding
  hemisphere — a second-order SEI refinement.)
- **Item 2 — Cloudy per-zone emissivity → MC shape.** `--metal-cloudy` now also
  parses Cloudy's `save line emissivity` (per-zone local emissivity) and uses it
  (interpolated onto our grid) to weight the MC profile, so the line's formation
  region is consistent with Cloudy's self-consistent ionization rather than the
  cruder Tier-1 ladder. Tagged `shape_source='cloudy'`.

*C III λ4650 recombination ORL — physics corrected (one residual: the branching
normalization).* Cloudy's default C III model atom emits ~0 for this optical
recombination line (ORLs need dedicated effective-recombination data not in the
light-element model atoms), so it stays on the metal_atoms recombination channel.
That channel was fixed to be physically correct: (a) the line now scales with the
RECOMBINING PARENT ion n(C³⁺)='C_IV' (it is emitted by C²⁺ but populated by C³⁺
recombination; the ³P° upper level is spin-forbidden from the ¹S ground so
collisional excitation is negligible) — previously it wrongly used n(C²⁺); (b) the
effective coefficient is now `branch · α_tot(C³⁺→C²⁺, T)` (Badnell-form total ×
multiplet branching), which bounds it physically and carries the real
recombination T-dependence. *Residual:* the multiplet branching is a literature-
informed estimate (≈0.02, PPB91-consistent, ~factor-2); drop the exact
Pequignot+1991 V1 effective coefficient into `METAL_LINES['C_III_4647']` to remove
it. A per-line **boxy-width validation** (profile HWHM vs the emission-weighted
shell velocity; stored in the npz + printed) now flags any artificially
narrow/broad metal profile — all C4 lines pass (ratio ~0.6-0.8, boxy-shoulder).

*Other remaining:* time-dependent (non-equilibrium) ionization and CSM
clumping/filling-factor are not modelled (Cloudy is steady-state, smooth); lines
are transported independently (no blending); spherical CSM (no torus/disk
asymmetry or dust blueshift). Absolute He-regime continuum normalization (P1)
bounds the EW reliability for thick UV lines.

**5b. Cloudy intermittency → C IV resonance-line flicker — ROOT-CAUSED + FIXED.**
The back-test surfaced an apparent C IV 1549 "spike" at C4 day5 (5.7e40 between
~1e35 neighbours). Re-running with per-epoch Cloudy tracing showed it is NOT a
Cloudy thermal-bistability spike (an earlier photon-budget guard built on that
mis-diagnosis was reverted). The real cause: **Cloudy was crashing** on a single
epoch's deck while succeeding on its neighbour — day5 (C³⁺=0.96) got the Cloudy
resonance-RT absolute (5.7e40, EW≈−10 Å), but day10 (C³⁺=0.88, comparable)
crashed and fell back to the **CHIANTI single-β** value (8.6e34), a ~5-dex
UNDERESTIMATE for a β≈1e-7 thick resonance line — so C IV flipped between its
strong physical value and the weak fallback purely on whether Cloudy ran.
The preserved failing deck (`run_cloudy` now saves it to `./cloudy_failures/` or
`$SNLT_CLOUDY_DEBUG`) gave the exact abort: `[Stop in readLaw … Radii must be in
increasing order]`. STELLA piles many zones at near-equal radii in the dense
shock shell; at the `%.6f` log precision the deck writes, 135 of them collapsed
to **exact-duplicate dlaw rows**, which Cloudy's `readLaw` rejects (it needs
STRICTLY increasing radii). *Fixes applied:*
- `build_deck` now enforces strict monotonicity ON THE WRITTEN GRID using integer
  micro-log units (each row ≥ prev + 10⁻⁶ in log r; a ≤few-ppm physically-null
  nudge). **Verified on the preserved deck**: the readLaw abort is gone and the
  day-10 C IV absolute → 8.3e40, matching day5 — the flicker is removed.
- `run_cloudy` timeout 240→480 s: the dense decks take ~250 s / 3 iterations
  (resonance-RT-limited; NOT dlaw-table-size-limited — a 200-row deck was just as
  slow), so 240 s was clipping them into the same CHIANTI fallback.

*Convergence — RESOLVED.* A 12-iteration test on the densest C4 deck (the same
552-zone day-10 deck) showed the CARBON lines we extract converge to ~1% by
iteration 4-5 (C IV 1549: 9.17e40 / 7.91e40 / 8.25e40 / 8.19e40 / 8.10e40 over
iters 1-5 → successive Δ +4.3% / -0.8% / -1.1%; C III 1909 stable to <2% from
iteration 2). Cloudy's GLOBAL "did not converge" message is `because He-like
subord[inate]` — an unrelated He-like ion line, NOT the metals — so the lines we
actually read off are converged even though the global flag trips. `iterate to
convergence max` raised 3→6 (the deck naturally converges at iter 5; Cloudy stops
early when converged, so sparse fast epochs are unaffected), timeout 480→720 s.
The thick resonance absolute is now converged to ~1%, not a factor.

*Remaining nicety (optional):* a proper resonance escape (Sobolev/EP, as
`line_rt_escape` does for thick He lines) as a Cloudy-failure-robust fallback, so
that even if an epoch's Cloudy run is ever lost the resonance line doesn't
collapse 5 dex to the single-β CHIANTI value.

**6. Forbidden / nebular-phase lines.**
The late, optically-thin (τ_es ≲ 0.3) epochs are nebular, where neither the
photospheric MC nor the homologous P-Cygni applies. *Fix:* a thin-gas nebular
emission mode (collisionally excited + recombination, no photosphere) so the
post-interaction tail becomes physical rather than discarded.

---

**7. Optical-spectrum fidelity: expanded line list + narrow-CSM profile +
synthetic-spectrum assembly.  ◑ IN PROGRESS (Stage 1 done).** Benchmarking C4-C7
against the real Icn prototypes (SN 2019hgp/2021csp, TNS public spectra in
`obs_comparison/`) showed three concrete gaps: (i) the optical carbon forest
(C III 5696, C IV 5801/12, C II 4267/6580/7236) plus O/Ne/Mg lines beyond our
6-line metal set; (ii) the early narrow/intermediate-width P-Cygni from the
UNSHOCKED CSM (we produce only the broad interaction shell); (iii) blends + a
comparable assembled spectrum. **Chosen approach (not a full RT synthesis
engine):** keep the validated per-line physics, expand the diagnostically-
important lines (chosen DATA-DRIVENLY by mining Cloudy's full `save line list`
per composition/epoch), add the narrow-CSM component, and ASSEMBLE all line
profiles onto a wavelength grid over the computed continuum → a synthetic
spectrum overplottable on observations. Full opacity-expansion synthesis (needed
only for the IIn Fe II forest) stays a separate future project.
*Staged plan (each gated, backwards-compat enforced — existing IIP/IIn/Ibn/Icn
line results must be unchanged when new features are OFF):*
- **Stage 1 — narrow-CSM P-Cygni profile.  ✅ INTEGRATED + VALIDATED (commit 9171343).**
  `--narrow-csm` (default OFF = byte-identical). Narrow core at the outer-zone
  wind velocity, resonance blue trough, area-conserving (L unchanged), physical
  skip when no slow wind exists (no-CSM / post-sweep). 3-round regression gate
  passed (C4/A1/A4).
  `csm_narrow_profile.py`: additive narrow component (flat-top emission at
  v_wind; resonance lines get a sub-continuum blue trough via exp(-τ) continuum
  attenuation; forbidden lines pure emission). Off-state = exactly zero. Unit
  tests pass. *Integration into `metal_lines` (add the narrow component to the
  broad MC profile, amplitude from the unshocked-CSM emission measure) is gated
  on the running grid finishing — editing live modules mid-run would desync it.*
- **Stage 2 — optical C lines** (C III 5696, C IV 5801/12, C II 4267/6580/7236),
  atomic data Cloudy/CHIANTI-anchored, added to `metal_atoms.METAL_LINES`.
- **Stage 3 — O/Ne/Mg lines** (O II, [O II], O I 7774, Mg II 4481, [Ne III] have)
  per the regime, data-driven from the Cloudy line-list mining.
- **Stage 4 — `synthetic_spectrum.py`** assembly: continuum + Σ line profiles on
  a λ grid → overplot vs `obs_comparison/` real spectra.
*Validation gates (post-grid):* after EACH stage re-run a IIP (A1), IIn (A4), Ibn
(C... He-dominated) and Icn (C4) control and diff the EXISTING 13+6 lines'
L/EW/profiles — must be unchanged to numerical precision; only NEW lines/the
narrow component may differ. Backwards compatibility is the acceptance criterion.

## P3 — Geometry & viewing angle

**7. 2D / asymmetric snapshots.**
All current input is 1D spherical STELLA. Real interaction (disk/torus CSM,
aspherical ejecta) imprints viewing-angle dependence on line profiles and
polarization. *Fix:* ingest 2D (axisymmetric) snapshots and run the MC peel-off
along a user-specified inclination; output profile vs viewing angle. Large
effort — the MC transport is already 3D-capable in principle, but the
input/geometry handling, the gate's homology test, and the continuum need
generalizing.

---

## P4 — Validation & usability

**8. Observational benchmarking.** Compare the H-rich IIn grid (A/B series)
against well-observed SNe IIn Hα luminosity/decrement evolution; compare the
C-series against observed Ibn He I evolution, to calibrate the factor-of-few
saturated-line uncertainties.

**9. Uncertainty propagation.** Carry the MC Poisson + iteration-convergence
uncertainties through to the per-line L/EW so the plotting scripts can draw
error bars instead of bare points.

**10. Auto epoch-range selection.** Have the batch detect and tag the
interaction / transition / nebular phase boundaries (from `L_cont_band`,
`tau_es`, `R_phot` trends) so the post-processing can truncate automatically
rather than by hand.

---

## P2 #7 — Optical line-list upgrade + Icn synthetic spectra — ✅ COMPLETE (2026-06-13)

All five tightening items done, gated, committed; final 40-model production grid
(re)run with the full physics and analyzed.

- **Item 1** absolute-continuum synthetic spectra (B_λ(T_phot) shape) — c00f2a7.
- **Item 2** right-model comparison (GO1/C7/C11): continuum slopes match real Icn.
- **Item 3** C_V ladder stage + C IV 5801/12 with **self-shielded** C³⁺→C⁴⁺
  ionization (τ_self 0→2200 through the CSM; C⁴⁺ 0.80→0.0026; existing 23 lines
  shifted ≤1.0%) — f90c8f8.
- **Item 4** emission-measure narrow fraction (emergent f_n: C IV 1549 0.65
  in-wind / C III] 0.05 shell; L conserved exactly) + relative wind gate
  (embedded-CSM aware) — 41ae1d5.
- **Item 5** T_phot of the Icn comparators (GO1/C7/C11) = 20.3–22.7 kK, squarely
  in the real 15–25 kK range → **no He-opacity root-fix needed** for the Icn
  comparison (the cool-photosphere worry was a low-CSM C4 artifact). + R=120
  instrumental convolution (92c5af3).
- **Final grid:** 40/40 models, production complete, 0 readLaw (dlaw fix holds),
  analysis in `analysis_final2/`. The 4 new optical C lines + C IV 5801 are in
  all 1101 model-epoch rankings; **C II 4267 is the #1 dominant feature in 20
  late C-series epochs**, O I 7774 in 3 — the optical carbon forest the v1 set
  lacked now drives the late Icn spectra. PCA stable: PC1 80.9%, structure-
  dominated (R_prog −0.98, f_Herich +0.94).

**Remaining honest gaps (next work):**
- **C III 4650 contrast gap:** the synthetic 4647 bump is ~0.9× continuum vs the
  real ~1.5–2×. Prime cause: the C III 4647 recomb **branch (2e-2) is factor-2
  low** — drop in the exact Pequignot+1991 V1 effective coefficient. (The
  5696/5876 complex already matches at 1.59× vs real 1.5–2×.)
- **Narrow-component absolute** (f_n redistribution) is shape-validated but its
  luminosity is **not yet validated against observed narrow-line fluxes**.
- **C_V self-shielding** is a conservative upper bound (He⁺ shielding would lower
  C⁴⁺ further) — refine if C IV 5801 becomes a quantitative diagnostic.
- Fe-group line blanketing (blue, IIn) remains out of scope.

---

## P2 #8 — Optical C III/C IV WC-like feature absolutes (the real λ4650 gap)

**Status: ✅ RESOLVED for C III 4650 (2026-06-18). C IV 5801 remains weak.**

**Fix:** the existing Cloudy request used a PHANTOM label `C 3 4647.42A` (=0 in
Cloudy — no transition there); the real V1-multiplet emission is at `4650.25A`.
Fixed the label + routed the optical ORL ABSOLUTES through Cloudy's full C III/
C IV model atom (`_CLOUDY_ORL` in metal_lines; energy ceiling + provisional
fallback retained). **Result (GO1):** C III 4650 L 5.5e38→5.43e40 (×100), EW
−0.06→−5.54 Å (day3) / −3.19 Å (day5, stable); the synthetic λ4650 F/F_cont
contrast went ~1.0→**1.55–1.66** vs the real **1.8–2.2** — gap closed from ~300×
to within ~20%. No regression (only C III 4647 + C IV 5801 changed; ΔL/L=0 for
all else). **C IV 5801** stays weak even from Cloudy's atom (L 1.9e36, EW ~0) —
its Li-like recombination needs the C⁴⁺ (C_V) parent that is rare at these
conditions; left on provisional, noted as a minor residual. Below is the original
analysis.

---

**(original) Status: ROOT-CAUSED.** The single highest-leverage remaining gap for the
Icn synthetic-spectrum match.

**The gap (measured, GO1 day3/5):** the observed WC-like optical carbon features
are strong — real EW ≈ 19–83 Å at λ4650, 17–30 Å at λ5696, 11–39 Å at λ5801
(SN 2019hgp / 2021csp) — but our synthetic optical C III/C IV recombination lines
are ~100–600× too faint (C III 4647 EW ≈ 0.14 Å; C III 5696, C IV 5801 ≈ 0).

**Root cause (NOT a coefficient):** with C³⁺ abundant (0.86) and the full CSM
integrated, the deficit is intrinsic to treating these as simple ORLs. The parent
ion (C³⁺ for λ4650, needs hot/ionized gas) and the n_e² emission-measure boost
(needs cool/dense gas) anti-correlate, so a pure recombination emissivity
n_e·n(C³⁺)·α_eff stays faint. The PPB91 effective-coefficient correction
(branch 0.02→0.048, α_eff 2.4e-13; commit 7eda25f) is *correct* but closes only
2.4× of the ~300× gap. The narrow-CSM component inherits the same deficit for the
ORLs (validated in `validate_narrow_flux.py`); only the resonance lines
(C IV 1549, C III] 1909) — which take Cloudy's resonance-line-RT absolute — are on
a trustworthy optical-adjacent scale.

**Proper fixes (pick one):**
1. **Extend Cloudy emissivity-weighting to the optical ORLs.** Tier-2 already
   parses Cloudy `save line emissivity` for the resonance lines (MC shape) and
   `save line list` for resonance absolutes. Add C III 4647/5696 + C IV 5801 to
   the Cloudy line list so their ABSOLUTES come from Cloudy's full C III/C IV
   model atom (which carries the low-T dielectronic + cascade physics our
   provisional α_eff lacks) instead of metal_atoms. Lowest-effort, reuses the
   existing Cloudy plumbing; gated like the resonance lines (energy ceiling +
   CHIANTI fallback).
2. **CMFGEN-class C III/C IV model-atom recombination + LTDR data** dropped into
   metal_atoms (highest fidelity, most work).

Until then: quote the resonance-line carbon diagnostics (C IV 1549, C III] 1909)
and the He/H lines as quantitative; treat the optical C III/C IV ORL *absolutes*
as lower limits (shapes/fractions are right). Items 1 (coeff) + 2 (narrow-flux
validation) of the P2 #7 tightening are DONE; this is the surviving open item.
