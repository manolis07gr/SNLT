# SNLT pipeline — Future Work (prioritized)

Ordered by priority. Each item lists the gap, why it matters, and a concrete
first step.

---

## P0 — Data-plumbing fixes (cheap, blocking for publication)

**1. Shared late-epoch snapshot bug.**
The batch loader substitutes a single common file for late epochs when a
model-specific snapshot is missing, producing byte-identical day-150/160 (A),
day-120–160 (C), day-160 (B) rows across *all* models — including no-CSM
controls. *Fix:* make the snapshot resolver skip an epoch that has no
model-local file instead of falling back to a shared path; emit a warning
listing skipped epochs. Until fixed, truncate every series at its
continuum-collapse epoch (already the physics-motivated cut).

**2. Regime-grade should know about composition.**
`Halpha_prod` is graded "A" even in H-free models where the line is numerical
noise. *Fix:* gate the H-line grades on a minimum H abundance / minimum
`L_line` floor, and the He-line grades likewise; emit "no element" instead of a
trust grade.

---

## P1 — Physics for the regimes we already touch

**3. Saturated-line radiative transfer (IIn / Ibn interaction phase).**
The single-shot kernel cannot represent saturated transport (τ≫1), so in the
dense-CSM phase the line *shapes* and profile-integrated EWs are factor-of-few
and the absolute luminosities are factor-of-two. This is the dominant physics
limitation for the interaction epochs we most care about. *Fix:* a proper
multiple-electron-scattering + per-line escape-probability RT module (iterated
J_bar per line, not just Hα), validated against the production Hα. This would
upgrade most B/C grades from B/C to A and make He EWs quotable.

**4. Continuum / energy budget without hydrogen.**
The recombination-budget and continuum normalization were built and validated
for H-rich gas. For the H-free He-rich (C-series) interaction the He lines come
from the He-NLTE solver (correct machinery), but the overall budget and
`L_cont_band` are unvalidated, and `L_cont_band` collapses unphysically at cold
compact late epochs. *Fix:* a composition-general continuum treatment and a
He-anchored budget analogous to the H case-B decrement.

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
