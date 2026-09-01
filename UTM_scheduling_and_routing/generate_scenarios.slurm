#!/bin/bash
#SBATCH --account=dtiwari2-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --job-name=generate-scenarios
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err

set -euo pipefail

# Use SLURM_SUBMIT_DIR (the directory from which sbatch was invoked)
cd "$SLURM_SUBMIT_DIR"

# Create slurm_logs directory
mkdir -p slurm_logs

# Override these with sbatch --export or by editing the submission command.
PYTHON_BIN="${PYTHON_BIN:-python}"
SCENARIO_ROOT="${SCENARIO_ROOT:-$SLURM_SUBMIT_DIR/cc_scenarios_small}"
N_SCENARIOS="${N_SCENARIOS:-10}"
MIN_WAYPOINTS="${MIN_WAYPOINTS:-3}"
MAX_WAYPOINTS="${MAX_WAYPOINTS:-10}"
MIN_AGENTS="${MIN_AGENTS:-2}"
MAX_AGENTS="${MAX_AGENTS:-5}"
SEED="${SEED:-123}"

mkdir -p "$SCENARIO_ROOT"

"$PYTHON_BIN" gen_scenarios.py \
    --output-root "$SCENARIO_ROOT" \
    --n-scenarios "$N_SCENARIOS" \
    --min-waypoints "$MIN_WAYPOINTS" \
    --max-waypoints "$MAX_WAYPOINTS" \
    --min-agents "$MIN_AGENTS" \
    --max-agents "$MAX_AGENTS" \
    --seed "$SEED"

echo "Scenario generation completed: $SCENARIO_ROOT"