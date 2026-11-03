#!/bin/bash
#SBATCH --account=dtiwari2-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=combine-subgraphs
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

: "${SCENARIO_FOLDER:?Set SCENARIO_FOLDER, e.g. scenarios/multi_nwp60_na6_p3_seed1}"
PYTHON_BIN="${PYTHON_BIN:-/projects/illinois/eng/mechse/salapaka/dhananjay/.venv/bin/python}"

export MPLBACKEND=Agg

"$PYTHON_BIN" lower_main.py --scenario_folder "$SCENARIO_FOLDER" --combine_only
