#!/bin/bash
#SBATCH --account=dtiwari2-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=solve-scenarios
#SBATCH --output=slurm_logs/slurm-solve-%x-%a-%j.out
#SBATCH --error=slurm_logs/slurm-solve-%x-%a-%j.err
#SBATCH --array=0-29

set -euo pipefail

# Use SLURM_SUBMIT_DIR (the directory from which sbatch was invoked)
cd "$SLURM_SUBMIT_DIR"

# Create slurm_logs directory
mkdir -p slurm_logs

# Configuration
SCENARIO_ROOT="${SCENARIO_ROOT:-$SLURM_SUBMIT_DIR/cc_scenarios}"
PYTHON_BIN="${PYTHON_BIN:-python}"
METHODS=("cbf" "slsqp" "gurobi")
TIME_LIMIT="${TIME_LIMIT:-3600}"

# List all scenario directories
SCENARIOS=($(ls -d "$SCENARIO_ROOT"/scenario_* 2>/dev/null | xargs -n1 basename | sort))

if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
    echo "ERROR: No scenarios found in $SCENARIO_ROOT"
    exit 1
fi

# Calculate which scenario and method this array task should run
N_METHODS=${#METHODS[@]}
N_SCENARIOS=${#SCENARIOS[@]}
TOTAL_TASKS=$((N_SCENARIOS * N_METHODS))

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_TASKS ]]; then
    echo "Array task ID $SLURM_ARRAY_TASK_ID exceeds total tasks ($TOTAL_TASKS). Exiting."
    exit 0
fi

SCENARIO_IDX=$((SLURM_ARRAY_TASK_ID / N_METHODS))
METHOD_IDX=$((SLURM_ARRAY_TASK_ID % N_METHODS))

SCENARIO="${SCENARIOS[$SCENARIO_IDX]}"
METHOD="${METHODS[$METHOD_IDX]}"
SCENARIO_PATH="$SCENARIO_ROOT/$SCENARIO"

echo "=========================================="
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Scenario Index: $SCENARIO_IDX / $N_SCENARIOS"
echo "Method Index: $METHOD_IDX / $N_METHODS"
echo "Scenario: $SCENARIO"
echo "Method: $METHOD"
echo "Scenario Path: $SCENARIO_PATH"
echo "Time Limit: $TIME_LIMIT seconds"
echo "=========================================="

if [[ ! -d "$SCENARIO_PATH" ]]; then
    echo "ERROR: Scenario directory not found: $SCENARIO_PATH"
    exit 1
fi

"$PYTHON_BIN" main_cbf.py solve \
    --method "$METHOD" \
    --scenario-path "$SCENARIO_PATH" \
    --time-limit "$TIME_LIMIT"

echo "Completed solve for $SCENARIO with $METHOD method"
