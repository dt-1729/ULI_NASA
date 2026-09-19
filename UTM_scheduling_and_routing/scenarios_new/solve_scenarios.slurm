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
#SBATCH --nodelist=ccc0363
#SBATCH --array=0-159%50

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

SCENARIO_ROOT="${SCENARIO_ROOT:-$SLURM_SUBMIT_DIR/scenarios_new}"
PYTHON_BIN="${PYTHON_BIN:-/projects/illinois/eng/mechse/salapaka/dhananjay/.venv/bin/python}"
METHODS=("cbf")
TIME_LIMIT="${TIME_LIMIT:-12000}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

if [[ ! -d "$SCENARIO_ROOT" ]]; then
    echo "ERROR: Scenario root not found: $SCENARIO_ROOT"
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
fi

# Finds scenarios in the new hierarchy:
# scenarios/nwp10_na3/scenario_001_seed.../scenario_data.pkl
mapfile -t SCENARIOS < <(
    find "$SCENARIO_ROOT" -type f -name "scenario_data.pkl" -printf '%h\n' | sort
)

N_SCENARIOS=${#SCENARIOS[@]}
if [[ "$N_SCENARIOS" -eq 0 ]]; then
    echo "ERROR: No scenario_data.pkl files found under $SCENARIO_ROOT"
    exit 1
fi

if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N_SCENARIOS" ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} exceeds available scenario count (${N_SCENARIOS}). Exiting."
    exit 0
fi

SCENARIO_PATH="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"
SCENARIO="${SCENARIO_PATH#"$SCENARIO_ROOT"/}"

# Rotate solver order to avoid systematic order bias.
OFFSET=$((SLURM_ARRAY_TASK_ID % ${#METHODS[@]}))
METHOD_ORDER=()
for ((i = 0; i < ${#METHODS[@]}; i++)); do
    METHOD_ORDER+=("${METHODS[$(((OFFSET + i) % ${#METHODS[@]}))]}")
done

echo "=========================================="
echo "Array task:       $SLURM_ARRAY_TASK_ID / $((N_SCENARIOS - 1))"
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