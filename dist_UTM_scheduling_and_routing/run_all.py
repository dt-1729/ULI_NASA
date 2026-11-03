"""Run the whole hierarchical pipeline in one command: scenario -> higher level -> lower level.

Local:    python3 run_all.py --scenario_folder scenarios/<folder>
          python3 run_all.py --n_waypoints 60 --n_agents 6 --n_parts 3 --seed 1   (generates the scenario)
Cluster:  sbatch --export=ALL,SCENARIO_FOLDER=scenarios/<folder> run_all.slurm
          (or --lower_backend slurm to solve the higher level here and submit the subgraph job array)

All optimizer/annealing flags of higher_main.py apply to both levels.
"""
import subprocess
import sys
import time
from pathlib import Path

from higher_main import _make_parser, solve_scenario
from lower_main import solve_lower_level
from scenario_generator import create_scenario, load_scenario, save_scenario


def _make_run_parser():
    parser = _make_parser()
    parser.description = "Solve the higher level, then every subgraph, then combine"
    for flag, kind, default in (
        ("--n_waypoints", int, 60),
        ("--n_agents", int, 6),
        ("--n_parts", int, 3),
        ("--seed", int, 1),
    ):
        parser.add_argument(flag, type=kind, default=default, help="scenario generation")
    parser.add_argument(
        "--network_type", default="multi", choices=["grid", "ring", "multi", "random"]
    )
    parser.add_argument("--scenario_root", default="scenarios")
    parser.add_argument(
        "--lower_backend", choices=("local", "slurm"), default="local",
        help="local: solve subgraphs here; slurm: submit a job array via submit_lower_level.sh",
    )
    parser.add_argument(
        "--lower_args", default="",
        help="extra flags for the SLURM lower-level jobs, e.g. '--n_beta 15'",
    )
    return parser


def main():
    args = _make_run_parser().parse_args()
    if args.n_beta < 1:
        raise ValueError("--n_beta must be at least one")
    if args.time_limit <= 0:
        raise ValueError("--time_limit must be positive")

    if args.scenario_folder:
        folder = Path(args.scenario_folder)
        scenario = load_scenario(folder)
    else:
        scenario = create_scenario(
            args.n_waypoints, args.n_agents, args.n_parts, args.seed, args.network_type
        )
        folder = save_scenario(scenario, Path(args.scenario_root))
        print(f"Generated scenario: {folder}")

    started = time.time()
    coarse_path = Path(args.output) if args.output else folder / "coarse" / "solution_coarse_cbf.pkl"
    print("=== Higher level ===")
    solve_scenario(
        scenario,
        coarse_path,
        boundary_fraction=args.boundary_fraction,
        cbf_mode=args.cbf_mode,
        time_limit=args.time_limit,
        anneal_print=args.anneal_print,
        args=args,
    )

    print("=== Lower level ===")
    if args.lower_backend == "slurm":
        script = Path(__file__).with_name("submit_lower_level.sh")
        command = ["bash", str(script), str(folder), *args.lower_args.split()]
        subprocess.run(command, check=True)
        print("Subgraph jobs submitted; the combine job writes combined/ when they finish.")
    else:
        solve_lower_level(coarse_path, folder, args=args)
    print(f"Pipeline finished in {time.time() - started:.1f} s (outputs in {folder})")


if __name__ == "__main__":
    sys.exit(main())
