#!/bin/bash
# Submit after the solve array with:
#   sbatch --dependency=afterok:<solve_job_id> compare_scenarios.slurm
#SBATCH --account=dtiwari2-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=compare-scenarios
#SBATCH --output=slurm_logs/slurm-compare-%x-%j.out
#SBATCH --error=slurm_logs/slurm-compare-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs

SCENARIO_ROOT="${SCENARIO_ROOT:-$SLURM_SUBMIT_DIR/scenarios}"
PYTHON_BIN="${PYTHON_BIN:-$SLURM_SUBMIT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found or not executable: $PYTHON_BIN"
    exit 1
fi

mapfile -t SCENARIOS < <(find "$SCENARIO_ROOT" -type f -name scenario_data.pkl -print \
    | sed 's#/scenario_data.pkl$##' | sort)

if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
    echo "ERROR: No scenarios found under $SCENARIO_ROOT"
    exit 1
fi

expected=$(( ${#SCENARIOS[@]} * 3 ))
actual=$(find "$SCENARIO_ROOT" -type f \( \
    -name solution_cbf.pkl -o -name solution_slsqp.pkl -o -name solution_seq.pkl \
    \) | wc -l | tr -d ' ')

if [[ "$actual" -ne "$expected" ]]; then
    echo "ERROR: Expected $expected solution files, found $actual"
    exit 1
fi

"$PYTHON_BIN" main.py compare \
    --scenario-root "$SCENARIO_ROOT" \
    --methods cbf slsqp seq