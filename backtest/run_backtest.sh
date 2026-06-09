#!/bin/bash
# ============================================================================
# SNLT back-test runner
# ----------------------------------------------------------------------------
# Runs the model grid A1, A4, B4, C1, C4 over a SPARSE epoch set
#   (days 0.1, 1, 3, 5, 10, 20, 30, 40, 50, 80, 100)
# with the full current physics (continuum renorm + adaptive window + saturated
# RT + composition-general budget + C/O/Ne metal lines + Cloudy Tier-2), so the
# recent changes can be checked for consistency across all SN regimes
# (IIP / IIn / Ib / Ic / Ibn / Icn). Only the listed epochs are run (--epochs),
# so it is ~11 snapshots/model instead of the full 30-34.
#
# Usage:   bash backtest/run_backtest.sh           # all 5 models
#          bash backtest/run_backtest.sh C1 C4     # subset
#
# Then:    python backtest/check_backtest.py       # consistency report
# ============================================================================
set -u

SNLT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="${SNLT_PY:-python}"                       # override with SNLT_PY=/path/to/python
export XUVTOP="${XUVTOP:-$SNLT_ROOT/chianti}"    # CHIANTI (Tier-1)
export CLOUDY_EXE="${CLOUDY_EXE:-$HOME/c23.01/source/cloudy.exe}"  # Cloudy (Tier-2)

EPOCHS="0.1,1,3,5,10,20,30,40,50,80,100"
# Same flags for every model so the SAME code paths are exercised everywhere.
# The Sobolev gate + composition switches adapt per model automatically.
# NOTE: no --out-prefix in batch — each epoch auto-names prod_dayXXX_lines.npz
# (a fixed prefix would clobber every epoch onto one file).
FLAGS="--batch --batch-format stella --epochs $EPOCHS \
       --line-profile-method formal \
       --he-lines --he1-nlte --he2-nlte \
       --saturated-rt --he-budget \
       --metal-lines --metal-cloudy"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(A1 A4 B4 C1 C4)
fi

# The models are fully independent (separate dirs, separate Cloudy temp dirs), so
# run them in PARALLEL by default — ~5× wall-clock speed-up, no change to physics
# or packet counts. Set SNLT_SERIAL=1 to force the old sequential mode.
SERIAL="${SNLT_SERIAL:-0}"

echo "SNLT back-test  |  root=$SNLT_ROOT"
echo "  XUVTOP=$XUVTOP"
echo "  CLOUDY_EXE=$CLOUDY_EXE  ($([ -x "$CLOUDY_EXE" ] && echo found || echo MISSING))"
echo "  epochs=$EPOCHS"
echo "  models=${MODELS[*]}   mode=$([ "$SERIAL" = 1 ] && echo serial || echo PARALLEL)"
echo

run_one() {
    local M="$1"; local DIR="$SNLT_ROOT/input_models/$M"
    if [ ! -d "$DIR" ]; then echo "[skip] $M: no dir $DIR"; return; fi
    ( cd "$DIR" && $PYBIN production_runner.py $FLAGS > "backtest_$M.log" 2>&1 )
    echo " [$M] done  ($(date '+%H:%M:%S'))"
}

if [ "$SERIAL" = 1 ]; then
    for M in "${MODELS[@]}"; do echo " MODEL $M ($(date '+%H:%M:%S'))"; run_one "$M"; done
else
    pids=()
    for M in "${MODELS[@]}"; do
        echo " launching $M ($(date '+%H:%M:%S'))  → input_models/$M/backtest_$M.log"
        run_one "$M" &
        pids+=($!)
    done
    echo " ${#pids[@]} models running in parallel; waiting…"
    wait "${pids[@]}"
fi

echo
echo "All models done. Now run:  $PYBIN backtest/check_backtest.py"
