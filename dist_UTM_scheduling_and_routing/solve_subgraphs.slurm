#!/bin/bash
#SBATCH --account=dtiwari2-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:15:00
#SBATCH --job-name=solve-subgraphs
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
# One array task per subgraph; the task id is the subgraph id. Any node in the partition may run
# a task. submit_lower_level.sh sets --array from the scenario's subgraph count.
#SBATCH --array=0-9

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Required: SCENARIO_FOLDER (folder containing coarse/solution_coarse_cbf.pkl).
: "${SCENARIO_FOLDER:?Set SCENARIO_FOLDER, e.g. scenarios/multi_nwp60_na6_p3_seed1}"
PYTHON_BIN="${PYTHON_BIN:-/projects/illinois/eng/mechse/salapaka/dhananjay/.venv/bin/python}"
TIME_LIMIT="${TIME_LIMIT:-3600}"
# Extra lower_main.py flags (e.g. "--n_beta 15 --tf 10")
EXTRA_ARGS="${EXTRA_ARGS:-}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
fi

N_SUBGRAPHS=$("$PYTHON_BIN" lower_main.py --scenario_folder "$SCENARIO_FOLDER" --n_subgraphs)
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$N_SUBGRAPHS" ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} exceeds ${N_SUBGRAPHS} subgraphs. Exiting."
    exit 0
fi

echo "=========================================="
echo "Scenario:         $SCENARIO_FOLDER"
echo "Subgraph:         $SLURM_ARRAY_TASK_ID of $N_SUBGRAPHS"
echo "Node:             $(hostname)"
echo "CPU allocation:   $SLURM_CPUS_PER_TASK"
echo "Time limit:       $TIME_LIMIT seconds"
echo "=========================================="

# shellcheck disable=SC2086
"$PYTHON_BIN" lower_main.py \
    --scenario_folder "$SCENARIO_FOLDER" \
    --subgraphs "$SLURM_ARRAY_TASK_ID" \
    --time_limit "$TIME_LIMIT" \
    $EXTRA_ARGS

echo "Completed subgraph $SLURM_ARRAY_TASK_ID at $(date --iso-8601=seconds)"
