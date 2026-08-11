# Toward a STELLA PPISN model for iPTF14hls — burial scan of res_day3.0 + proposed setup

## 1. Burial scan of res_day3.0 (why it cannot make 14hls spectra)

Scanned all 245 diagnosed epochs (of 305) for the embedded-interaction condition
(τ_es above the shock ≫ 1):

- **235/245 epochs: shock EXPOSED** — R_shock/R_phot = 0.999–1.030 (median 1.002),
  with τ_es at the photosphere ≤ 0.65 (median ~1e−7), so the column *above* the
  shock is transparent at essentially every luminous epoch, including both LC
  peaks. Exposed interaction → narrow/intermediate recombination Balmer lines
  (IIn morphology) — which is exactly what the pipeline produces and exactly
  what Arcavi et al. (2017) exclude for iPTF14hls.
- **10 nominally buried epochs** are all pre-max (day −225 … +5). Only the
  pre-pulse compact phases (day −225: τ_es = 870; day −175: τ_es = 36) are
  optically buried; the day −75…+5 group has R_s/R_phot ≈ 0.22–0.42 but
  τ_es(phot) ≈ 0.66 — geometrically interior yet optically translucent.

**Verdict:** in res_day3.0 the collisions always happen at (or optically above)
the photosphere during the luminous phases. No epoch range exists where the
pipeline would produce the 14hls signature; this is a property of the
progenitor/pulse structure, not of the radiative transfer.

## 2. What iPTF14hls requires (targets to engineer)

From Arcavi et al. 2017 (Nature 551, 210):
T1. Broad Balmer P Cygni at all times; Hα absorption 8,000 → 6,000 km/s over
    600 d (25% decline); Fe II constant ≈ 4,000 km/s.
T2. Spectral evolution ~10× slower than a normal IIP (day-600 spectrum ≈ a
    normal day-60 spectrum).
T3. ≥ 5 LC undulations over ~600 d around M ≈ −18 (L ~ 1–3×10⁴² erg/s);
    total radiated ≳ 2×10⁵⁰ erg.
T4. NO narrow/intermediate emission lines, no strong blue continuum, no
    X-rays/radio → any interaction must be optically buried at all epochs.

Design translation:
- Fast, massive, H-rich **outer** envelope in homologous expansion → carries the
  photosphere and the stable broad P Cygni for 600 d (T1, T2).
- The undulating power source must sit **below** τ_es ≳ 10 of that envelope at
  every undulation epoch (T3, T4): energy thermalizes and diffuses out as
  continuum, smoothed and delayed, with no exposed shock lines.
- Sustained photospheric temperature (interaction heating) is what slows the
  apparent spectral evolution (T2) — the photosphere neither cools nor recedes
  on the normal timescale.

## 3. Proposed initial STELLA setup

**Progenitor.** M_ZAMS ≈ 105–120 M⊙ at Z ≈ 10⁻³ (low-Z to retain hydrogen);
end state: He core ≈ 45–55 M⊙ (lower PPISN range → pulses cluster close to
core collapse) inside a retained H envelope of **M_env ≈ 20–30 M⊙**,
R ≈ 700–1000 R⊙. (H retention is the known objection to PPISN for 14hls —
low-Z + no rotation-enhanced loss is the lever.)

**Pulse history — the critical tuning.** Pulses in the FINAL months–years
before collapse, so the ejected shells are still **compact** at explosion
(R_shell = v_shell·Δt ~ 10¹³–10¹⁵ cm, i.e. inside/just above the envelope
rather than detached at 10¹⁶⁺ cm as in res_day3.0):

| shell | mass | v_shell | ejected pre-CC | R at collapse |
|---|---|---|---|---|
| S3 (inner) | ~3 M⊙ | 250 km/s | 0.5 yr | ~4×10¹³ cm |
| S2 | ~5 M⊙ | 400 km/s | 1.2 yr | ~1.5×10¹⁴ cm |
| S1 (outer) | ~8 M⊙ | 600 km/s | 2.5 yr | ~5×10¹⁴ cm |

**Final explosion.** E_SN ≈ 5–8×10⁵¹ erg into the remaining ~30–40 M⊙
(envelope + core leftover); M_Ni ≈ 0.05–0.1 M⊙ (subdominant — undulations and
plateau power come from the buried collisions).

**Why this geometry works.** The outer ~10–20 M⊙ of the final ejecta reaches
6,000–9,000 km/s and overtakes/sweeps the compact shells within days, so ALL
collisions occur beneath the (still optically thick) swept envelope. Overlying
column check: τ_es ≈ κM/(4πR²) with κ=0.34, M_above=15 M⊙, R=2×10¹⁵ cm
(8,000 km/s × day 300) gives **τ ≈ 200** — comfortably buried through day ~600
(τ→~10 near day 600 as required). Collision timing maps onto undulations via
t_i ≈ R_i/(v_ej − v_shell): the S3/S2/S1 ladder above gives energy-injection
episodes at roughly days ~5–15, ~30–60, ~120–250 (diffusion through the
envelope broadens and delays each into a smooth LC bump — several bumps over
hundreds of days, T3). Add a 4th, slower shell (~1,000 km/s, ~4 yr pre-CC) if a
late (day ~400–500) undulation is desired.

**Two admissible variants** (both satisfy the burial criterion):
- (a) *SN-ejecta ↔ shell collisions* (the setup above) — cleanest control over
  undulation timing via shell radii.
- (b) *Shell ↔ shell collisions pre-CC + final SN* — pulses tuned so successive
  shells collide with each other under the envelope before collapse (undulations
  can even precede the SN); harder to tune in STELLA, but reproduces
  pre-discovery variability (14hls's 1954 eruption analogue).

## 4. Verification targets for the STELLA run (pipeline-checkable)

1. v_phot ≈ 8,000 km/s at day 30–100, declining ≤ 25% by day 600.
2. τ_es above the outermost collision > 10 through day ≥ 500 (this scan's
   criterion — re-run the burial scan on the new snapshot series first).
3. L ≈ 1–3×10⁴² erg/s with ≥ 3 discernible undulations over ~600 d.
4. Pipeline output: Sobolev gate/unified w→scatter at all luminous epochs
   (homologous outer envelope) → stable broad Hα P Cygni; NO narrow emission
   components; v2.1 `--urt-zone-w` handles any stratified embedded-shock
   epochs; `--urt-aniso-es` irrelevant here (no exposed τ_es shell).

If the STELLA run meets (1)–(3), the existing pipeline — unchanged — should
produce the 14hls-like spectroscopic sequence, and failure modes will be
informative (e.g., if shells are too massive/slow the photosphere stalls in
slow material and lines narrow → detectable immediately in the movies).

*(2026-08-05; companion to the res_day3.0 delivery. Burial-scan code inline in
the session log; batch_metrics.csv columns used: R_phot, shock_R_s,
tau_es_stella.)*
