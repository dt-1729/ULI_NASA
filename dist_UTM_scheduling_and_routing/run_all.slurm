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
# Usage: sbatch --export=ALL,SCENARIO_FOLDER=scenarios/<folder>[,BACKEND=slurm][,RUN_ARGS="--n_beta 15"] run_all.slurm
# Without SCENARIO_FOLDER a scenario is generated from N_WAYPOINTS, N_AGENTS, N_PARTS, SEED, NETWORK_TYPE.

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

SCENARIO_ARGS=()
if [[ -n "${SCENARIO_FOLDER:-}" ]]; then
    SCENARIO_ARGS=(--scenario_folder "$SCENARIO_FOLDER")
else
    SCENARIO_ARGS=(
        --n_waypoints "${N_WAYPOINTS:-60}" --n_agents "${N_AGENTS:-6}"
        --n_parts "${N_PARTS:-3}" --seed "${SEED:-1}" --network_type "${NETWORK_TYPE:-multi}"
    )
fi

# shellcheck disable=SC2086
"$PYTHON_BIN" run_all.py "${SCENARIO_ARGS[@]}" --lower_backend "$BACKEND" $RUN_ARGS
