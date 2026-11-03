#!/bin/bash
# Submit the lower-level solve for one scenario: a job array (one task per subgraph) followed by a
# combine job that runs once all array tasks have finished (whether or not they succeeded; the
# combine step reports subgraphs that have no saved solution).
#
# Usage: ./submit_lower_level.sh scenarios/<scenario_folder> [extra lower_main.py flags]
# Env:   PYTHON_BIN, TIME_LIMIT are forwarded to the jobs.
# The coarse solution must already exist: python3 higher_main.py --mode solve --scenario_folder <folder>
set -euo pipefail

SCENARIO_FOLDER="${1:?Usage: $0 <scenario_folder> [extra lower_main.py flags]}"
shift || true
EXTRA_ARGS="$*"

cd "$(dirname "$0")"
mkdir -p slurm_logs

PYTHON_BIN="${PYTHON_BIN:-/projects/illinois/eng/mechse/salapaka/dhananjay/.venv/bin/python}"

if [[ ! -f "$SCENARIO_FOLDER/coarse/solution_coarse_cbf.pkl" ]]; then
    echo "ERROR: $SCENARIO_FOLDER/coarse/solution_coarse_cbf.pkl not found. Solve the higher level first."
    exit 1
fi

N_SUBGRAPHS=$("$PYTHON_BIN" lower_main.py --scenario_folder "$SCENARIO_FOLDER" --n_subgraphs)
echo "Scenario $SCENARIO_FOLDER has $N_SUBGRAPHS subgraphs"

EXPORTS="ALL,SCENARIO_FOLDER=$SCENARIO_FOLDER,PYTHON_BIN=$PYTHON_BIN,EXTRA_ARGS=$EXTRA_ARGS"
[[ -n "${TIME_LIMIT:-}" ]] && EXPORTS="$EXPORTS,TIME_LIMIT=$TIME_LIMIT"

ARRAY_ID=$(sbatch --parsable --export="$EXPORTS" --array="0-$((N_SUBGRAPHS - 1))" solve_subgraphs.slurm)
COMBINE_ID=$(sbatch --parsable --export="$EXPORTS" --dependency="afterany:$ARRAY_ID" combine_subgraphs.slurm)

echo "Submitted subgraph array job $ARRAY_ID and combine job $COMBINE_ID"
echo "Logs: slurm_logs/   Results: $SCENARIO_FOLDER/{subgraphs,combined}/"
