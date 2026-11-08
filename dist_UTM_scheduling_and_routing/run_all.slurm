#!/bin/bash
#SBATCH --account=dtiwari2-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=run-all
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err

# Whole pipeline in one job: higher level, then the lower level.
#   BACKEND=local (default): all subgraphs are solved one after another inside this job.
#   BACKEND=slurm: after the higher level, submit_lower_level.sh submits a job array (one task per
#                  subgraph) plus a combine job; this job then ends.
# Usage: sbatch [--export=ALL,BACKEND=slurm,RUN_ARGS="--n_beta 15"] run_all.slurm scenarios/<folder>
#   (the SCENARIO_FOLDER env variable still works as a fallback; the CLI argument takes precedence)
# Without a scenario folder a scenario is generated from N_WAYPOINTS, N_AGENTS, N_PARTS, SEED, NETWORK_TYPE.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

PYTHON_BIN="${PYTHON_BIN:-/projects/illinois/eng/mechse/salapaka/dhananjay/.venv/bin/python}"
BACKEND="${BACKEND:-local}"
RUN_ARGS="${RUN_ARGS:-}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHON_BIN

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
fi

SCENARIO_FOLDER="${1:-${SCENARIO_FOLDER:-}}"

# Trim whitespace and check if SCENARIO_FOLDER is set and non-empty
SCENARIO_FOLDER=$(echo "$SCENARIO_FOLDER" | xargs)

SCENARIO_ARGS=()
if [[ -n "$SCENARIO_FOLDER" ]]; then
    if [[ ! -d "$SCENARIO_FOLDER" ]]; then
        echo "ERROR: Scenario folder not found: $SCENARIO_FOLDER" >&2
        exit 1
    fi
    SCENARIO_ARGS=(--scenario_folder "$SCENARIO_FOLDER")
else
    SCENARIO_ARGS=(
        --n_waypoints "${N_WAYPOINTS:-60}" --n_agents "${N_AGENTS:-6}"
        --n_parts "${N_PARTS:-3}" --seed "${SEED:-1}" --network_type "${NETWORK_TYPE:-multi}"
    )
fi

echo "SCENARIO_ARGS: ${SCENARIO_ARGS[@]}"

# shellcheck disable=SC2086
"$PYTHON_BIN" run_all.py "${SCENARIO_ARGS[@]}" --lower_backend "$BACKEND" $RUN_ARGS
