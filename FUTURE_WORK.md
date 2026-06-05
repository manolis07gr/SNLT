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

**5. Metal lines for H- and He-poor CSM interaction.**
Add the strongest **carbon, oxygen, and neon** lines (e.g. C III/IV, O III,
[O I], [Ne III]) expected from stripped, metal-enriched CSM interaction (Icn /
late Ibn). *Fix:* extend the NLTE/escape machinery with C/O/Ne ions and their
recombination + collisional channels; add them to the `line_names` schema so
all downstream post-processing (npz, plotting, correlation analysis) picks them
up automatically.

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
