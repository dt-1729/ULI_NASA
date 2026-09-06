#!/bin/bash
#SBATCH --account=dtiwari2-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --job-name=generate-scenarios
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err

set -euo pipefail

# Use SLURM_SUBMIT_DIR (the directory from which sbatch was invoked)
cd "$SLURM_SUBMIT_DIR"

# Create slurm_logs directory
mkdir -p slurm_logs

# Override these with sbatch --export or by editing the submission command.
PYTHON_BIN="${PYTHON_BIN:-$SLURM_SUBMIT_DIR/.venv/bin/python}"
SCENARIO_ROOT="${SCENARIO_ROOT:-$SLURM_SUBMIT_DIR/scenarios}"
N_SCENARIOS="${N_SCENARIOS:-10}"
SEED="${SEED:-123}"

# Each entry is "number_of_waypoints:number_of_agents".  Keeping waypoint
# and agent counts fixed per invocation creates the repository's grouped layout.
SCENARIO_SPECS="${SCENARIO_SPECS:-8:3 12:5 15:7 20:9 25:13 25:18 30:18 35:21 40:25 40:29}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found or not executable: $PYTHON_BIN"
    exit 1
fi

mkdir -p "$SCENARIO_ROOT"

for spec in $SCENARIO_SPECS; do
    IFS=: read -r n_waypoints n_agents <<< "$spec"
    group_root="$SCENARIO_ROOT/local_scenarios_${n_waypoints}_${n_agents}"

    echo "Generating $N_SCENARIOS scenarios in $group_root"
    mkdir -p "$group_root"

    "$PYTHON_BIN" gen_scenarios.py \
        --output-root "$group_root" \
        --n-scenarios "$N_SCENARIOS" \
        --min-waypoints "$n_waypoints" \
        --max-waypoints "$n_waypoints" \
        --min-agents "$n_agents" \
        --max-agents "$n_agents" \
        --seed "$SEED"
done

echo "Scenario generation completed: $SCENARIO_ROOT"