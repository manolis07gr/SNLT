# iPTF14hls PPISN modelling — status, results, and open problem

Written 2026-08-30. Companion files: `iPTF14hls_model_design.md` (design),
`~/Desktop/ppisn_14hls/stella_prep/` (all code + outputs),
`~/Documents/snlt_head_backup/` (durable copies).

---

## 1. What is finished and trustworthy

**MESA PPISN progenitors** (`~/Desktop/ppisn_14hls`, `~/Desktop/ppisn_14hls_m125`)

| | M_ZAMS = 110 | M_ZAMS = 125 |
|---|---|---|
| He core | 43.5 M⊙ | 52.8 M⊙ |
| CO core | 37.7 M⊙ | 49.1 M⊙ |
| Mass retained | 74.5 M⊙ | 123.8 M⊙ |
| Pulses | #1 weak (no ejection), #2 ejected **26.6 M⊙** (E≈2.5e50 erg) | cycling, no ejection yet |
| Pulse→collapse delay | ~10 days | — |

Deliverables in `~/Documents/snlt_head_backup/mesa_progenitor/`:
`profile209.data` (pre-SN star, 47.5 M⊙ remnant), `prerelax_prof001.data`
(the 26.6 M⊙ ejected shell with its velocity field), `he_dep.mod`, photos.

MESA fixes required to get here (all documented in the inlists):
`max_number_retries` 500 → 100000; the mass-relax init settings re-firing at
stage-2 load (removed); `x_ctrl(6)` max_v_for_relax 20 → 25 km/s;
`op_split_burn` for the Si-burning phase.

**STELLA pulse-ejection light curve** — computed, verified, delivered.
Breakout flash log L = 44.0 at day 0.4, then a plateau at log L ≈ 42.2 → 42.0
through day 100 (i.e. 1–2 × 10^42 erg/s, the iPTF14hls luminosity), bulk
velocity ~2340 km/s. Files: `stella_prep/res_pulse/`, figure `pulse_LC.png`.
Independently cross-checked against a purpose-written gray flux-limited
diffusion code: agreement ≤ 0.22 dex at every epoch.

**blcode collision hydrodynamics** — installed, patched, validated; produced
the prompt-collision dynamics and the ρ/T/v movie (`blcode_collision_movie.mp4`,
`run2/blcode_run2_movie.mp4`). Code fixes: `mass_cut` and `grid_type` runtime
parameters added; parser leaked unit 27 on optional-parameter miss;
`read_abar_from_profile` was only parsed when `ncomps>0`, leaving `abar`
uninitialised → Inf pressures (the fatal one).

**Two genuine STELLA code-level fixes** (both required for any interaction run):
1. `strad/hcdfnrad.f` rejected any trial state containing a zone below **30 K**
   (`BADSTE` → `ifail=13` → timestep collapse). Interaction ejecta cools past
   that within a day, so every configuration died at the identical moment.
   Now clamped at the validity limit instead of rejected.
2. Handoff gridding: index-space resampling produced a resolution cliff
   (3 giant inner cells, ~247 cells at the minimum-separation floor) and
   silently lost 3 M⊙. Replaced with a log-uniform-in-radius regrid.

With these, STELLA integrates a 400-day interaction model end to end
(`run2/res_selfconsistent/`, 1400 steps, clean stop) — the first time this
build has run a collision configuration at all.

**56Ni physics** — verified against analytic decay: injecting 0.05 M⊙ gives a
tail at log L = 41.07 at day 200 versus 41.1 predicted.

**Quantitative scalings established** (independent of the open problem below):
- LC duration is set by the shock-crossing time of the shell, so the
  pulse→SN delay controls it. Delays of 10 d and 120 d both give a shell swept
  in < 14 days; a 600-day interaction needs the shock to cross ~5e16 cm, i.e.
  a shell **years** old at explosion.
- Ejecta transition velocity from the MESA fallback cut (4.5 M⊙ H-rich above a
  43 M⊙ cut, E_kin = 1.6e51) is **7147 km/s** — matching iPTF14hls's observed
  ~7000 km/s photospheric velocity without tuning.
- Energetics: iPTF14hls radiated ~1e50 erg over 600 d. A 4.5 M⊙ ejecta hitting
  a 26.6 M⊙ shell thermalises nearly all of its 1.6e51 erg — 10× too much.
  Matching 14hls requires M_CSM/M_ej ~ 0.2–0.3 (~1 M⊙), or ρ_CSM ~ 1e-18 g/cc
  at 3e16 cm (~0.2 M⊙), i.e. a *weak* pulse or dense wind, not a massive shell.

**Observed comparison data**: iPTF14hls pseudo-bolometric light curve rebuilt
from the public photometry (1904 points, 1280 days; blackbody fits to g/r/i;
T_BB ≈ 5000–6000 K, matching published values) →
`stella_prep/iptf14hls_pseudobol.csv`.

---

## 2. The open problem: interaction-LC amplitude

Every interaction model radiates ~10× more energy than it contains.
The most recent, cleanest configuration (`run5/an15_fixed`, analytic homologous
ejecta + 0.2 M⊙ CSM at 6e15–4.5e16 cm) has, verified in the input file:

- thermal energy 3.00e50 erg (target 3.00e50, exact)
- kinetic energy 1.60e51 erg
- outer-boundary blackbody 3.3e38 erg/s (negligible)

yet radiates 2.95e51 erg in the first 44 days at a flat log L ≈ 44.9, *before*
the ejecta reaches the CSM. That is ~10× the thermal budget and ~1.5× the total
(thermal + kinetic).

Bugs found and fixed along the way (each produced believable-looking light
curves, which is why they survived so long):
- temperature interpolation inflating energy through T⁴ (E_rad reached 3.5e53,
  170× E_SN);
- the energy normalisation being dominated by the wind's volume rather than the
  ejecta, so it set the *wind* temperature;
- my own safety clamp re-heating cold CSM zones to `FLOOR(3)` = 3000 K every
  step — an unlimited energy source in an extended wind;
- the grid extending to 1.8e17 cm with a 1000 K wind, so 4πR²σT⁴ at the outer
  boundary alone gave 3e45 erg/s;
- `eve`'s hardcoded 2000 K model-construction floor (patched to 30 K).

**Prime remaining suspect**, identified but not yet resolved: the initial model
contains a velocity discontinuity at the ejecta/wind interface (zone 140,
r = 4.8e14 cm: **27813 → 100 km/s** across one zone). The fast, low-density
ejecta tail shocks against the wind from t = 0, and artificial viscosity
converts that kinetic energy to radiation continuously. This would explain a
sustained ~1e45 that is insensitive to CSM mass and appears long before the
intended collision. It is also physical in origin (the tail does shock), but
the efficiency in the model is too high.

**Suggested next steps** (in order):
1. Taper the ejecta velocity smoothly into the wind (or truncate the ejecta
   density profile at a lower maximum velocity, e.g. 3 v_t instead of 4 v_t),
   then re-audit radiated energy versus budget over the first 40 days.
2. Reduce the handoff thermal energy: E_th = 15% of E_SN at day 2 is too high;
   adiabatic degradation from a 5e13 cm progenitor to 1e15 cm gives ~5e49 erg.
3. Only once radiated ≤ available over the pre-collision phase, re-run the
   CSM-mass and pulse-delay grid — the two-parameter map that answers what
   iPTF14hls requires.

---

## 3. Reproduction

Model builders (all in `stella_prep/`):
`build_analytic_sn.py <shell_age_yr> <out> [M_shell_scale] [t0_days]` —
analytic homologous ejecta + MESA shell + wind, energy-audited (recommended);
`build_stella_csm.py` — blcode-ejecta + shell variant;
`blcode_to_stella.py <frame> <out>` — blcode frame → STELLA (log-uniform grid);
`make_blcode_input*.py` — MESA remnant + aged shell → blcode profile;
`swd_to_stella.py` — STELLA state → STELLA restart (leapfrog).

STELLA run: copy `<model>.hyd/.abn` to `stella_new/modmake/mesa.{hyd,abn}`,
`rm -f strad/run/stop`, then `./rn` with `MESASDK_ROOT=/Applications/mesasdk`.
Light curve from `res/mesa.tt` (col 0 = days, col 6 = M_bol;
L = 10^((4.74−M_bol)/2.5) × 3.828e33). Non-stock `mesa.dat` settings and all
source patches are listed in the project memory entry.

To harvest a wedged run without losing its light curve:
`touch strad/run/stop` (clean stop, writes `mesa.tt`), then remove the file
before the next launch.
