"""regime_diagnostics.py — Phase 6 line-formation-regime audit.

For each line and each snapshot, classify which radiative-transfer regime
the line is in. Used to (a) defensively annotate paper figures with
trust-level labels and (b) flag epochs where the empirical or single-shot
approximations break down and rigorous iteration is needed.

The classifier uses three diagnostics:

  τ_med           median Sobolev optical depth across emitting zones
  τ_max           maximum Sobolev optical depth
  β_med           median Sobolev escape probability

and emits a regime label and a paper-grade trust label per line per epoch.

Regimes
-------
Optically thin
    τ_max < 1.0
    All emission escapes; line strength = ∫ j_line dV. No β suppression,
    no self-absorption. The MC kernel is exact in this regime.

Intermediate
    1 < τ_med < 100
    Sobolev β meaningful, kernel correctly handles partial escape.
    Source function approaches LTE × dilution factor. Single-shot β is
    accurate to ~10% for the integrated line.

Optically thick
    100 < τ_med < 10⁴
    β << 1 everywhere. Line photons trapped, source function controlled
    by net radiative balance. Single-shot β over-estimates L by factor
    ~5-20×; J_bar iteration required for definitive values.

Saturated
    τ_med > 10⁴
    Line is fully blanketed across the entire emitting region. Profile
    shape (P-Cygni geometry) is correct but absolute scale is dominated
    by re-scattering effects the single-shot kernel does not capture.
    Per-line J_bar iteration (Phase 5b-rigorous) is essential.

Trust-grade labels (paper-defensibility)
---------------------------------------
A   τ regime well within single-shot kernel validity (thin / intermediate).
    The line is paper-quality at single-shot β. No caveat needed.

B   Optically thick. The Hα-anchored empirical correction reduces the
    systematic to ~factor-of-2 for most non-Hα lines. Quote L_line as
    "Hα-anchored estimate" and cite the correction factor R.

C   Saturated or near-saturated. The empirical correction is approximate
    only; per-line J_bar iteration would change the absolute value by
    factors of 2-5. Quote L_line with explicit warning that it requires
    iteration; treat profile shape (peak v, FWHM, blue absorption velocity)
    as the primary published quantity rather than absolute L.

R   "Reference for shape only" — pathological regime (extreme τ + very
    low X_HII or very hot continuum source where source-function
    approximations fundamentally fail). Don't quote any absolute number;
    show profile shape and note "requires full CMFGEN-style RT for
    quantitative comparison".

API
---
classify_line(name, tau_med, tau_max, beta_med) -> dict
    Returns {'regime', 'grade', 'rationale', 'paper_action'} for a
    single line.

build_snapshot_table(state, snap, spectra) -> list of dicts
    For all 13 lines in a Phase 5 output, build the full regime table.

write_snapshot_diagnostic(filepath, table, snap, ...)
    Save a per-epoch human-readable regime audit.

write_batch_regime_summary(all_results, filepath)
    Cross-epoch regime evolution table — when does each line transition
    between regimes? Used in paper to defend the temporal coverage
    claims.

The module is import-once; all rules below can be edited if better
calibration is found.
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np

# P0 #2: a line whose element is essentially absent (mean mass fraction below
# this floor) is numerical noise, not a physical line — graded 'N' (no element)
# rather than given a trust grade. Matches continuum_compgen.X_H_FREE_THRESH.
ELEM_FLOOR = 1.0e-3


def _line_element(name: str) -> Optional[str]:
    """Return the emitting element ('H','He','C','O','Ne') for a line key, else
    None. 'Halpha_prod' → 'H'; metal lines like 'C_III_1909' → 'C'."""
    if name.startswith('He_'):
        return 'He'
    if name.startswith('C_'):
        return 'C'
    if name.startswith('Ne_'):
        return 'Ne'
    if name.startswith('O_'):
        return 'O'
    if name.startswith(('Halpha', 'Hbeta', 'Hgamma', 'Hdelta',
                        'Palpha', 'Pbeta', 'Pgamma')):
        return 'H'
    return None


# ----------------------------------------------------------------------
# Line-specific paper rules
# ----------------------------------------------------------------------
# Each line has its own atomic-physics quirks (metastability,
# resonance trapping, recombination cascading) that change which
# regime requires iteration. These are encoded as per-line overrides
# on top of the generic τ-regime classifier.
#
# Schema:
#   'needs_iter_above_tau':
#       Below this τ_med, single-shot β is paper-quality. Above it,
#       J_bar iteration changes the answer by >2× and the line gets
#       at best a grade-B label.
#   'caveat':
#       Always-on warning for this line, regardless of τ.
#       (e.g. He II 1640 needs Lyα-cascade physics we don't model.)
#   'reference_only':
#       True if this line should always be reported shape-only.

LINE_RULES = {
    # ---- Hydrogen ----
    'Halpha': {
        'needs_iter_above_tau': 1e4,    # we ARE iterating (RT-NLTE)
        'caveat': None,
        'reference_only': False,
        'iteration_status': 'full',     # RT-NLTE in production runner
    },
    'Hbeta': {
        'needs_iter_above_tau': 1e3,
        'caveat': 'Single-shot β; empirical R-scaled.',
        'reference_only': False,
        'iteration_status': 'empirical',
    },
    'Hgamma': {
        'needs_iter_above_tau': 1e3,
        'caveat': 'Single-shot β; empirical R-scaled.',
        'reference_only': False,
        'iteration_status': 'empirical',
    },
    'Palpha': {
        'needs_iter_above_tau': 1e3,
        'caveat': 'Paschen series; same iteration status as Hβ.',
        'reference_only': False,
        'iteration_status': 'empirical',
    },
    'Pbeta': {
        'needs_iter_above_tau': 1e3,
        'caveat': 'Paschen series; same iteration status as Hβ.',
        'reference_only': False,
        'iteration_status': 'empirical',
    },
    # ---- He I ----
    'He_I_5876': {
        'needs_iter_above_tau': 1e3,
        'caveat': None,
        'reference_only': False,
        'iteration_status': 'empirical',
    },
    'He_I_6678': {
        'needs_iter_above_tau': 1e2,
        'caveat': 'Marginal line; quoting requires τ < 1 epoch flag.',
        'reference_only': False,
        'iteration_status': 'empirical',
    },
    'He_I_7065': {
        'needs_iter_above_tau': 1e3,
        'caveat': None,
        'reference_only': False,
        'iteration_status': 'empirical',
    },
    'He_I_10830': {
        # Metastable 2³S lower level — J_bar pumping at 2³S→2³P is
        # crucial. Single-shot β over- or under-estimates by factors
        # of 2-5 depending on pumping regime.
        'needs_iter_above_tau': 1e2,
        'caveat': ('METASTABLE TRIPLET: 2³S → 2³P pumping not iterated. '
                   'Quote shape; absolute L requires Phase 5b-rigorous.'),
        'reference_only': False,
        'iteration_status': 'empirical_with_caveat',
    },
    # ---- He II ----
    'He_II_1640': {
        # Resonance line: Lyα cascade physics not modeled.
        'needs_iter_above_tau': 10.0,
        'caveat': ('RESONANCE LINE: Lyα-cascade and continuum recombination '
                   'physics not fully modeled; absolute L systematically '
                   'over-predicted. Use for shape comparison only.'),
        'reference_only': True,
        'iteration_status': 'shape_only',
    },
    'He_II_3203': {
        'needs_iter_above_tau': 1e2,
        'caveat': None,
        'reference_only': False,
        'iteration_status': 'empirical',
    },
    'He_II_4686': {
        'needs_iter_above_tau': 1e2,
        'caveat': None,
        'reference_only': False,
        'iteration_status': 'empirical',
    },
    'He_II_10124': {
        'needs_iter_above_tau': 1e2,
        'caveat': None,
        'reference_only': False,
        'iteration_status': 'empirical',
    },
}


# ----------------------------------------------------------------------
# Regime classifier
# ----------------------------------------------------------------------
def _generic_regime(tau_med: float, tau_max: float) -> str:
    """Classify by τ alone (no line-specific rules)."""
    if not np.isfinite(tau_max) or tau_max < 1.0:
        return 'optically_thin'
    if not np.isfinite(tau_med) or tau_med < 10.0:
        return 'intermediate'
    if tau_med < 1e4:
        return 'optically_thick'
    return 'saturated'


def classify_line(name: str,
                   tau_med: float,
                   tau_max: Optional[float] = None,
                   beta_med: Optional[float] = None,
                   strength_mode: Optional[str] = None,
                   x_elem: Optional[float] = None,
                   L_line: Optional[float] = None) -> dict:
    """Classify one line into regime + paper-grade trust level.

    Parameters
    ----------
    name : str
        Line key (e.g. 'Halpha', 'He_II_1640'). Must be in LINE_RULES.
    tau_med, tau_max : float
        Sobolev optical depth statistics. tau_max defaults to 10*tau_med
        if not given (a typical zone-to-zone spread).
    beta_med : float, optional
        Median Sobolev escape; informational only, not used for grading.

    Returns
    -------
    dict
        regime         : str
        grade          : 'A' | 'B' | 'C' | 'R'
        rationale      : str
        paper_action   : str (one-line recommendation for paper figures)
    """
    if tau_max is None:
        tau_max = 10.0 * abs(tau_med) if np.isfinite(tau_med) else np.inf

    # P0 #2: no-element gate. If the line's element is essentially absent
    # (mean mass fraction < ELEM_FLOOR), the "line" is numerical noise — emit
    # grade 'N' (no element) instead of a misleading τ-based trust grade.
    # x_elem is None when composition is unknown → no downgrade (safe default).
    element = _line_element(name)
    if (element in ('H', 'He') and x_elem is not None
            and np.isfinite(x_elem) and x_elem < ELEM_FLOOR):
        elem_name = 'hydrogen' if element == 'H' else 'helium'
        return {
            'regime': 'no_element',
            'grade': 'N',
            'rationale': (f'{elem_name.capitalize()} essentially absent '
                          f'(⟨X_{element}⟩ = {x_elem:.1e} < {ELEM_FLOOR:.0e}); '
                          f'L_line is numerical noise, not a physical line.'),
            'paper_action': (f'Do not quote — no {elem_name} in the ejecta. '
                             f'(Correct null for this composition.)'),
            'tau_med': float(tau_med),
            'tau_max': float(tau_max),
            'beta_med': (float(beta_med) if beta_med is not None
                         and np.isfinite(beta_med) else None),
            'iteration_status': 'no_element',
            'strength_mode': strength_mode,
        }

    # P2 #5: metal lines (C/O/Ne). strength_mode = 'metal-{mech}-{src}', where
    # src is 'chianti' (authoritative NLTE emissivity from CHIANTI) or 'prov'
    # (provisional metal_atoms data). Both are grade B, but the caveat differs:
    # for CHIANTI lines the atomic data is solid and the residual systematic is
    # the photoionization ion-balance; for provisional lines the atomic numbers
    # themselves are placeholders.
    sm_parts = (strength_mode.split('-') if isinstance(strength_mode, str)
                else [])
    if (element in ('C', 'O', 'Ne')
            or (sm_parts and sm_parts[0] == 'metal')):
        mech = sm_parts[1] if len(sm_parts) > 1 else 'emissivity'
        src = sm_parts[2] if len(sm_parts) > 2 else 'prov'
        if src == 'chianti':
            rationale = ('CHIANTI NLTE emissivity (authoritative atomic data: '
                         'multi-level statistical equilibrium, cascades, n_crit). '
                         'Absolute L scales with the photoionization ion-balance '
                         '(shock-ID sensitive) and the STELLA abundance — the '
                         'residual systematic is the IONIZATION, not the line.')
            action = ('Quote L_line (CHIANTI-NLTE line physics). Note the ion '
                      'balance as the systematic; a Cloudy ionization would close it.')
        else:
            rationale = ('First-principles ' + mech + ' emissivity on '
                         'photoionization-equilibrium ion densities, but the '
                         'C/O/Ne atomic data is PROVISIONAL (recombination line / '
                         'ChiantiPy unavailable). CEL lines carry the n_crit '
                         'correction.')
            action = ('Quote as PROVISIONAL: use the profile shape and relative '
                      'trends; do NOT quote absolute L until the atomic data is '
                      'verified (install ChiantiPy + CHIANTI for the CEL lines).')
        return {
            'regime': 'metal_' + mech + ('_chianti' if src == 'chianti' else ''),
            'grade': 'B',
            'rationale': rationale,
            'paper_action': action,
            'tau_med': float(tau_med),
            'tau_max': float(tau_max),
            'beta_med': (float(beta_med) if beta_med is not None
                         and np.isfinite(beta_med) else None),
            'iteration_status': ('metal_chianti' if src == 'chianti'
                                 else 'metal_provisional'),
            'strength_mode': strength_mode,
        }

    regime = _generic_regime(tau_med, tau_max)
    rule = LINE_RULES.get(name, {})
    iter_status = rule.get('iteration_status', 'empirical')
    needs_iter = rule.get('needs_iter_above_tau', 1e3)
    reference_only = rule.get('reference_only', False)
    caveat = rule.get('caveat', None)

    # Apply grade logic
    if reference_only:
        grade = 'R'
        rationale = 'Atomic physics outside current implementation scope.'
        action = ('Plot profile shape and FWHM/velocity. Do not quote '
                  'absolute L_line as a definitive value.')
    elif regime == 'optically_thin':
        grade = 'A'
        rationale = 'τ < 1 everywhere; single-shot kernel exact.'
        action = 'Quote L_line and EW directly. Grade-A.'
    elif regime == 'intermediate' or tau_med < needs_iter:
        if iter_status == 'full':
            grade = 'A'
            rationale = ('Intermediate τ + full RT-NLTE iteration available '
                         '(production Hα).')
            action = 'Quote L_line and EW directly. Grade-A.'
        else:
            grade = 'B'
            rationale = ('Intermediate τ; single-shot β + empirical R '
                         'is paper-quality.')
            action = 'Quote L_line as "Hα-anchored empirical estimate".'
    elif regime == 'optically_thick':
        if iter_status == 'full':
            grade = 'A'
            rationale = 'Optically thick but with full RT-NLTE iteration.'
            action = 'Quote L_line and EW directly. Grade-A.'
        else:
            grade = 'B'
            rationale = ('Optically thick; empirical Hα-anchored R reduces '
                         'systematic to factor-of-2.')
            action = ('Quote L_line as estimate; note "per-line J_bar '
                      'iteration would refine to ~factor of 2".')
    else:    # saturated
        if iter_status == 'full':
            grade = 'B'
            rationale = ('Saturated but with RT-NLTE iteration; absolute L '
                         'still has ~2x systematic from kernel/iteration '
                         'interaction.')
            action = ('Quote L_line with caveat; use peak F and v_peak as '
                      'primary observables.')
        else:
            grade = 'C'
            rationale = ('Saturated AND no per-line iteration — absolute '
                         'L uncertain to factors of 2-5.')
            action = ('Plot profile; note absolute scale uncertainty in '
                      'caption. Use shape diagnostics (FWHM, v_peak) as '
                      'primary observable.')

    # P1 #3 (--saturated-rt): if the thick-He empirical Hα anchor was DROPPED in
    # favour of the first-principles single-shot β escape luminosity + Thomson
    # multiple-scattering shape, the rationale/action must not claim an
    # "Hα-anchored empirical R". The grade letter is unchanged (the residual
    # ~factor-2 is now the absent nonlocal J̄/ALI iteration, not an anchor). The
    # mode string is set by phase5_runner._apply_recombination_budget.
    if strength_mode == 'He-NLTE(thick,EP-esc)':
        rationale = ('Optically-thick He: bare single-shot β escape luminosity '
                     '(= first-principles escape-probability value; empirical Hα '
                     'anchor REMOVED via --saturated-rt). Profile carries '
                     'multiple-electron-scattering (Thomson MC). Residual ~factor-2 '
                     'reflects the absent nonlocal J̄ (ALI) iteration, NOT an anchor.')
        action = ('Quote L_line as first-principles β-escape (no empirical anchor); '
                  'use the Thomson-broadened profile for shape diagnostics.')

    if caveat:
        rationale = (caveat + ' ' + rationale).strip()

    return {
        'regime': regime,
        'grade': grade,
        'rationale': rationale,
        'paper_action': action,
        'tau_med': float(tau_med),
        'tau_max': float(tau_max),
        'beta_med': (float(beta_med) if beta_med is not None
                     and np.isfinite(beta_med) else None),
        'iteration_status': iter_status,
        'strength_mode': strength_mode,
    }


# ----------------------------------------------------------------------
# Snapshot-level table
# ----------------------------------------------------------------------
def build_snapshot_table(spectra: dict,
                          snap: Optional[dict] = None) -> list:
    """Build a list of regime-classification rows for one snapshot.

    Parameters
    ----------
    spectra : dict
        Phase 5 output, line_name -> per-line dict with at least 'tau_med'
        and ideally 'tau_max', 'beta_med'.
    snap : dict, optional
        Snapshot dict; if given, epoch_d is extracted for the row metadata.

    Returns
    -------
    rows : list[dict]
        One entry per line. Each entry includes the classify_line() output
        plus 'line', 'epoch_d', 'L_line', 'EW' for downstream tables.
    """
    rows = []
    epoch_d = (snap.get('epoch_d') if isinstance(snap, dict) else None)
    # P0 #2 / P2 #5: mean composition for the no-element gate (None if absent)
    x_h = x_he = None
    x_C = x_O = x_Ne = None
    try:
        import continuum_compgen as _cg
        x_h = _cg.mean_X_H(snap)
        x_he = _cg.mean_X_He(snap)
    except Exception:
        pass
    comp = snap.get('composition') if isinstance(snap, dict) else None
    if isinstance(comp, dict):
        def _mean_comp(key):
            a = comp.get(key)
            if a is None:
                return None
            a = np.asarray(a, float)
            return float(np.mean(a)) if a.size else None
        x_C = _mean_comp('c12'); x_O = _mean_comp('o16'); x_Ne = _mean_comp('ne20')
    _x_by_elem = {'H': x_h, 'He': x_he, 'C': x_C, 'O': x_O, 'Ne': x_Ne}
    for name, sp in spectra.items():
        if name.startswith('_'):    # skip metadata keys
            continue
        if not isinstance(sp, dict):
            continue
        tau_med = float(sp.get('tau_med', np.nan))
        tau_max = sp.get('tau_max', None)
        if tau_max is not None:
            tau_max = float(tau_max)
        beta_med = sp.get('beta_med', None)
        if beta_med is not None:
            beta_med = float(beta_med)
        strength_mode = sp.get('strength_mode', None)
        _elem = _line_element(name)
        x_elem = _x_by_elem.get(_elem)
        L_for_gate = sp.get('L_line_corrected', sp.get('L_line', None))
        cls = classify_line(name, tau_med, tau_max, beta_med,
                            strength_mode=strength_mode,
                            x_elem=x_elem, L_line=L_for_gate)
        cls['line'] = name
        cls['epoch_d'] = epoch_d
        cls['lambda_rest'] = float(sp.get('lambda_rest', np.nan))
        # Prefer corrected values if available (post-Phase-5b)
        cls['L_line'] = float(sp.get('L_line_corrected', sp.get('L_line', np.nan)))
        cls['EW'] = float(sp.get('EW_corrected', sp.get('EW', np.nan)))
        cls['peak_F'] = float(sp.get('peak_F_corrected', sp.get('peak_F', np.nan)))
        rows.append(cls)
    return rows


def write_snapshot_diagnostic(filepath: str,
                                rows: list,
                                snap: Optional[dict] = None,
                                production_halpha: Optional[dict] = None):
    """Save a human-readable per-snapshot regime diagnostic.

    Format: a header with snap-level metadata + a fixed-width table of
    line-by-line regime classifications and paper-grade labels.
    """
    epoch_d = (snap.get('epoch_d') if isinstance(snap, dict) else '?')
    lines = []
    lines.append('=' * 96)
    lines.append(f' regime_diagnostics — per-snapshot line classification')
    lines.append('=' * 96)
    if epoch_d != '?':
        lines.append(f'epoch:    {epoch_d:.2f} d')
    if snap is not None and isinstance(snap, dict):
        lines.append(f'T_phot:   {snap.get("T_phot_inner", "?")} K')
        lines.append(f'R_phot:   {snap.get("R_phot_inner", "?")} cm')
    if production_halpha is not None:
        L = production_halpha.get('L_line', None)
        if L is not None:
            lines.append(f'Hα (production RT-NLTE): L_line = {L:.3e} erg/s')
    lines.append('')
    lines.append(f'{"line":<14} {"grade":<6} {"regime":<18} '
                  f'{"τ_med":<10} {"τ_max":<10} {"β_med":<10} {"L_line[erg/s]":<14}')
    lines.append('-' * 96)
    for r in rows:
        bm = r.get('beta_med')
        bm_s = f'{bm:.2e}' if bm is not None else '   --   '
        lines.append(
            f'{r["line"]:<14} {r["grade"]:<6} {r["regime"]:<18} '
            f'{r["tau_med"]:.2e}  {r["tau_max"]:.2e}  '
            f'{bm_s:<10} {r["L_line"]:.3e}')
    lines.append('')
    lines.append('Per-line action recommendations:')
    lines.append('-' * 96)
    for r in rows:
        lines.append(f'  {r["line"]:<14} ({r["grade"]}): {r["paper_action"]}')
        if r.get('rationale'):
            lines.append(f'    └─ {r["rationale"]}')
    lines.append('')
    lines.append('Grade legend:')
    lines.append('  A = Paper-quality. Quote L and EW directly.')
    lines.append('  B = Paper-quality with caveat (single-shot β; empirical Hα '
                 'anchor, OR --saturated-rt first-principles β-escape).')
    lines.append('  C = Shape only; absolute L uncertain to factors 2-5.')
    lines.append('  R = Reference shape only; atomic physics outside scope.')
    lines.append('  N = No element (⟨X⟩ < 1e-3); line is numerical noise — '
                 'do not quote (correct null for this composition).')
    if any(r.get('strength_mode') == 'He-NLTE(thick,EP-esc)' for r in rows):
        lines.append('')
        lines.append('Note: thick He lines marked EP-esc used --saturated-rt '
                     '(P1 #3): empirical Hα anchor REMOVED; L_line is the bare '
                     'single-shot β escape (first-principles), profile carries '
                     'Thomson multiple-scattering.')
    lines.append('=' * 96)
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')


# ----------------------------------------------------------------------
# Batch-level summary
# ----------------------------------------------------------------------
def write_batch_regime_summary(all_results: list,
                                 filepath: str):
    """Build a cross-epoch regime evolution summary.

    Parameters
    ----------
    all_results : list of dicts
        Each dict has 'epoch_d' and 'rows' (list of per-line classifications).
    filepath : str
        Output path for the .txt summary.

    Writes a grid: rows=lines, columns=epochs, cells=grade letter.
    Plus a per-line "best/worst epoch" diagnostic.
    """
    # Collect epoch and line index
    epochs = sorted([r.get('epoch_d', 0.0) for r in all_results
                      if r.get('epoch_d') is not None])
    all_lines = []
    for r in all_results:
        for row in r.get('rows', []):
            if row['line'] not in all_lines:
                all_lines.append(row['line'])

    # Bail gracefully if there's nothing to summarize
    if not epochs or not all_lines:
        with open(filepath, 'w') as f:
            f.write('=' * 100 + '\n')
            f.write(' regime_diagnostics — BATCH-LEVEL CROSS-EPOCH REGIME SUMMARY\n')
            f.write('=' * 100 + '\n\n')
            f.write(f'No usable per-snapshot regime data found '
                    f'(epochs={len(epochs)}, lines={len(all_lines)}).\n')
            f.write('Check that Phase 5 ran and that the NPZ schema matches '
                    'what regime_diagnostics expects.\n')
        return

    # Build line→{epoch: grade} map
    grade_grid = {ln: {} for ln in all_lines}
    tau_grid = {ln: {} for ln in all_lines}
    L_grid = {ln: {} for ln in all_lines}
    for snap_res in all_results:
        ep = snap_res.get('epoch_d')
        if ep is None:
            continue
        for row in snap_res.get('rows', []):
            grade_grid[row['line']][ep] = row['grade']
            tau_grid[row['line']][ep] = row['tau_med']
            L_grid[row['line']][ep] = row['L_line']

    # Write
    lines = []
    lines.append('=' * 100)
    lines.append(' regime_diagnostics — BATCH-LEVEL CROSS-EPOCH REGIME SUMMARY')
    lines.append('=' * 100)
    lines.append('')
    lines.append(f'Epochs processed: {len(epochs)} ({min(epochs):.2f}d - {max(epochs):.2f}d)')
    lines.append('')
    # Grade grid
    lines.append('Paper-grade evolution (rows = line, columns = epoch_d):')
    lines.append('')
    header = '  ' + ' '.join(f'{e:>6.1f}' for e in epochs)
    lines.append(f'{"line":<14}{header}')
    lines.append('-' * (14 + 7 * len(epochs) + 2))
    for ln in all_lines:
        row_str = f'{ln:<14}'
        for e in epochs:
            row_str += f'  {grade_grid[ln].get(e, "?"):>5}'
        lines.append(row_str)
    lines.append('')
    # Per-line trust summary
    lines.append('Per-line trust summary across batch:')
    lines.append('-' * 100)
    for ln in all_lines:
        grades = list(grade_grid[ln].values())
        if not grades:
            continue
        n_A = grades.count('A')
        n_B = grades.count('B')
        n_C = grades.count('C')
        n_R = grades.count('R')
        n_tot = len(grades)
        # Best/worst epoch
        tau_history = [(ep, tau_grid[ln].get(ep, np.nan)) for ep in epochs
                        if ep in tau_grid[ln]]
        if tau_history:
            best_ep = min(tau_history, key=lambda x: x[1] if np.isfinite(x[1]) else np.inf)
            worst_ep = max(tau_history, key=lambda x: x[1] if np.isfinite(x[1]) else 0.0)
            summary_extra = (
                f'  best τ at epoch {best_ep[0]:.2f}d (τ={best_ep[1]:.2e}); '
                f'worst at epoch {worst_ep[0]:.2f}d (τ={worst_ep[1]:.2e})')
        else:
            summary_extra = ''
        lines.append(
            f'  {ln:<14}  A={n_A:>2d} B={n_B:>2d} C={n_C:>2d} R={n_R:>2d}  '
            f'(of {n_tot}){summary_extra}')
    lines.append('')
    # Paper-recommendation overview
    lines.append('Paper-recommendation overview:')
    lines.append('-' * 100)
    lines.append('  A-graded across all epochs:')
    for ln in all_lines:
        g = list(grade_grid[ln].values())
        if g and all(x == 'A' for x in g):
            lines.append(f'    ✓ {ln}')
    lines.append('  Mixed B/C/R (paper-worthy with explicit caveat):')
    for ln in all_lines:
        g = list(grade_grid[ln].values())
        if g and any(x != 'A' for x in g):
            most_common = max(set(g), key=g.count)
            lines.append(f'    • {ln}   (dominant grade: {most_common})')
    lines.append('')
    lines.append('=' * 100)
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')


# ----------------------------------------------------------------------
# Per-zone diagnostic (optional, slower)
# ----------------------------------------------------------------------
def compute_per_zone_tau_beta(state, line_info: dict) -> dict:
    """Re-compute per-zone Sobolev τ and β for one line.

    Used for the deeper per-zone diagnostic plot when we want to see
    the τ distribution within a snapshot, not just the median.

    Parameters
    ----------
    state : PhysicalState
        Must have .r, .v, .h_levels (or .he1_levels, .he2_levels).
    line_info : dict
        Must have 'sigma_lambda' (πe²/mc × f × λ in cgs), 'lower_idx',
        'upper_idx', 'g_lower', 'g_upper', 'species' ('HI', 'HeI', 'HeII').

    Returns
    -------
    dict with 'tau_zone', 'beta_zone' arrays.
    """
    r = np.asarray(state.r)
    v = np.asarray(state.v)
    from velocity_grad import robust_dvdr
    dv_dr = robust_dvdr(v, r)
    dv_dr = np.maximum(dv_dr, 1e-30)

    species = line_info['species']
    if species == 'HI':
        levels = state.h_levels
    elif species == 'HeI':
        levels = state.he1_levels
    elif species == 'HeII':
        levels = state.he2_levels
    else:
        raise ValueError(f"Unknown species: {species}")

    n_lo = levels[line_info['lower_idx']]
    n_up = levels[line_info['upper_idx']]
    g_lo = line_info['g_lower']
    g_up = line_info['g_upper']

    n_diff = np.maximum(n_lo - (g_lo / g_up) * n_up, 0.0)
    tau_zone = line_info['sigma_lambda'] * n_diff / dv_dr

    with np.errstate(over='ignore', invalid='ignore'):
        beta_zone = np.where(tau_zone > 1e-6,
                              (1.0 - np.exp(-tau_zone)) / tau_zone,
                              1.0 - 0.5 * tau_zone)
    beta_zone = np.clip(beta_zone, 0.0, 1.0)

    return {
        'tau_zone': tau_zone,
        'beta_zone': beta_zone,
        'tau_med': float(np.median(tau_zone)),
        'tau_max': float(tau_zone.max()),
        'tau_min': float(tau_zone.min()),
        'beta_med': float(np.median(beta_zone)),
    }
