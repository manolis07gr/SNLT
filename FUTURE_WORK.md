# SNLT pipeline — Future Work (prioritized)

Ordered by priority. Each item lists the gap, why it matters, and a concrete
first step. **Status tags:** ✅ DONE · ◑ PARTIAL · ☐ OPEN.

---

## P0 — Data-plumbing fixes (cheap, blocking for publication)

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

*Remaining:* the C III 4647 recombination line has no default Cloudy line-list
label matched yet (stays Tier-1/provisional); time-dependent (non-equilibrium)
ionization and CSM clumping/filling-factor are not modelled (Cloudy is
steady-state, smooth); lines are transported independently (no blending). Absolute
He-regime continuum normalization (P1) bounds the EW reliability for thick UV lines.

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
