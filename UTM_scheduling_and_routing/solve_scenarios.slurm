#!/bin/bash
#SBATCH --account=dtiwari2-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:15:00
#SBATCH --job-name=solve-scenarios
#SBATCH --output=slurm_logs/slurm-solve-%x-%a-%j.out
#SBATCH --error=slurm_logs/slurm-solve-%x-%a-%j.err
#SBATCH --nodelist=ccc0362
#SBATCH --array=0-99%20

# For final timing benchmarks, uncomment this. It prevents interference
# from other jobs sharing the same node, but may increase queue time.
##SBATCH --exclusive

# If this feature is available in IllinoisComputes, uncomment to keep
# all benchmark jobs on the AMD EPYC 7713 node class.
##SBATCH --constraint=AE7713

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

SCENARIO_ROOT="${SCENARIO_ROOT:-$SLURM_SUBMIT_DIR/scenarios}"
PYTHON_BIN="/projects/illinois/eng/mechse/salapaka/dhananjay/.venv/bin/python"
# PYTHON_BIN="${PYTHON_BIN:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
METHODS=("cbf" "slsqp" "seq")
TIME_LIMIT="${TIME_LIMIT:-3600}"

# Prevent numerical libraries from using cores beyond --cpus-per-task.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mapfile -t SCENARIOS < <(
    find "$SCENARIO_ROOT" -type f -name scenario_data.pkl -print \
        | sed 's#/scenario_data.pkl$##' \
        | sort
)

N_SCENARIOS=${#SCENARIOS[@]}

if [[ "$N_SCENARIOS" -eq 0 ]]; then
    echo "ERROR: No scenarios found in $SCENARIO_ROOT"
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
fi

if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N_SCENARIOS" ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} exceeds ${N_SCENARIOS} scenarios. Exiting."
    exit 0
fi

SCENARIO_PATH="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"
SCENARIO="${SCENARIO_PATH#$SCENARIO_ROOT/}"

if [[ ! -f "$SCENARIO_PATH/scenario_data.pkl" ]]; then
    echo "ERROR: Missing scenario_data.pkl in $SCENARIO_PATH"
    exit 1
fi

# Rotate order: cbf→slsqp→seq, then slsqp→seq→cbf, then seq→cbf→slsqp.
# This keeps all methods in the same job, while avoiding systematic order bias.
OFFSET=$((SLURM_ARRAY_TASK_ID % ${#METHODS[@]}))
METHOD_ORDER=()
for ((i = 0; i < ${#METHODS[@]}; i++)); do
    METHOD_ORDER+=("${METHODS[$(((OFFSET + i) % ${#METHODS[@]}))]}")
done

echo "=========================================="
echo "Array task:       $SLURM_ARRAY_TASK_ID"
echo "Scenario:         $SCENARIO"
echo "Scenario path:    $SCENARIO_PATH"
echo "CPU allocation:   $SLURM_CPUS_PER_TASK"
echo "Method order:     ${METHOD_ORDER[*]}"
echo "Per-method limit: $TIME_LIMIT seconds"
echo "=========================================="

for METHOD in "${METHOD_ORDER[@]}"; do
    echo "Starting $METHOD at $(date --iso-8601=seconds)"

    "$PYTHON_BIN" main.py solve \
        --scenario-path "$SCENARIO_PATH" \
        --methods "$METHOD" \
        --time-limit "$TIME_LIMIT"

    echo "Completed $METHOD at $(date --iso-8601=seconds)"
done

echo "Completed all methods for $SCENARIO"