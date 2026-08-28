from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import MIRS
import mep_opt
import problem_generator
import utils
import visualize


# ============================================================
# Configuration
# ============================================================

N_WAYPOINTS = 5
N_AGENTS = 2
SEED = 120

COST_MODE = "sum"          # Options: "sum", "slowest"
LM = 1                     # Used for slowest-agent cost
OPTIMIZER_NAME = "MEP_CBF"

RESULTS_ROOT = Path("Final_results") / "mep_cbf_solutions"

# Supported modes:
#   "optimize"       -> run optimization and save everything
#   "load"           -> load one existing solution and regenerate plots
#   "load_all"       -> regenerate plots for every matching pickle file
RUN_MODE = "optimize"

# Used when RUN_MODE == "load".
# Set this to None to automatically select the most recent solution.
EXISTING_SOLUTION_FILE: Path | None = None

# Used when RUN_MODE == "load_all".
EXISTING_FILE_PATTERN = "**/solution.pkl"

SHOW_PLOTS = True
PLOT_DPI = 300

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Examples
    --------
    Run a new optimization:
        python run_mep.py optimize

    Load the latest saved solution:
        python run_mep.py load

    Load a particular saved solution:
        python run_mep.py load --solution-file path/to/solution.pkl

    Regenerate plots for all saved solutions:
        python run_mep.py load-all
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run a new MEP optimization or regenerate plots from "
            "previously saved solution data."
        )
    )

    parser.add_argument(
        "run_mode",
        nargs="?",
        choices=("optimize", "load", "load-all"),
        help=(
            "Execution mode: 'optimize' runs a new optimization, "
            "'load' loads one existing solution, and 'load-all' "
            "regenerates plots for all saved solutions."
        ),
    )

    parser.add_argument(
        "--solution-file",
        type=Path,
        default=None,
        help=(
            "Path to a particular solution pickle file. "
            "Used only with the 'load' mode. If omitted, the most "
            "recent solution is loaded."
        ),
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help=(
            "Root directory containing saved results. "
            f"Default: {RESULTS_ROOT}"
        ),
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default=EXISTING_FILE_PATTERN,
        help=(
            "Glob pattern used to locate solution files. "
            f"Default: {EXISTING_FILE_PATTERN!r}"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=PLOT_DPI,
        help=f"Resolution used for saved plots. Default: {PLOT_DPI}.",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively after generating them.",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not replace plot files that already exist.",
    )

    parser.add_argument(
        "--n-waypoints",
        type=int,
        default=5,
        help="Number of waypoints. Default: 5.",
    )

    parser.add_argument(
        "--n-agents",
        type=int,
        default=2,
        help="Number of agents. Default: 2.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=120,
        help="Random seed used for problem generation. Default: 120.",
    )

    parser.add_argument(
        "--cost-mode",
        type=str,
        choices=("sum", "slowest"),
        default="sum",
        help="Objective cost mode. Default: sum.",
    )

    parser.add_argument(
        "--lm",
        type=float,
        default=1.0,
        help=(
            "Weight used by the slowest-agent objective. "
            "Default: 1.0."
        ),
    )

    args = parser.parse_args()

    # When no positional mode is supplied, ask the user interactively.
    if args.run_mode is None:
        args.run_mode = prompt_for_run_mode()

    if args.solution_file is not None and args.run_mode != "load":
        parser.error(
            "--solution-file can only be used when run_mode is 'load'."
        )

    if args.n_waypoints <= 0:
        parser.error("--n-waypoints must be greater than zero.")

    if args.n_agents <= 0:
        parser.error("--n-agents must be greater than zero.")

    if args.n_agents > args.n_waypoints:
        parser.error(
            "--n-agents cannot be greater than --n-waypoints."
        )

    if args.lm < 0:
        parser.error("--lm must be nonnegative.")

    if args.dpi <= 0:
        parser.error("--dpi must be a positive integer.")

    return args


def prompt_for_run_mode() -> str:
    """
    Ask the user to select a run mode interactively.

    This is used when the script is launched without a positional mode:

        python run_mep.py
    """
    modes = {
        "1": "optimize",
        "2": "load",
        "3": "load-all",
    }

    print("\nSelect a run mode:")
    print("  1. optimize  - Run and save a new optimization")
    print("  2. load      - Load one saved solution and regenerate plots")
    print("  3. load-all  - Regenerate plots for all saved solutions")

    while True:
        selection = input("\nEnter 1, 2, or 3: ").strip().lower()

        if selection in modes:
            return modes[selection]

        # Also allow the user to type the mode name.
        normalized_selection = selection.replace("_", "-")

        if normalized_selection in modes.values():
            return normalized_selection

        print(
            "Invalid selection. Enter 1, 2, 3, "
            "'optimize', 'load', or 'load-all'."
        )


# ============================================================
# File-management utilities
# ============================================================

def create_run_directory(
    results_root: Path,
    optimizer_name: str,
    n_waypoints: int,
    n_agents: int,
    seed: int,
) -> Path:
    """
    Create a unique directory for one optimization run.

    Example:
        Final_results/mep_cbf_solutions/
            MEP_CBF_nwp5_na2_seed120_2026_07_18__18_30_45/
    """
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    run_name = (
        f"{optimizer_name}"
        f"_nwp{n_waypoints}"
        f"_na{n_agents}"
        f"_seed{seed}"
        f"_{timestamp}"
    )

    run_directory = results_root / run_name
    run_directory.mkdir(parents=True, exist_ok=False)

    return run_directory


def save_pickle(data: Any, file_path: Path) -> None:
    """Save arbitrary Python data as a pickle file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Solution data saved to:\n  {file_path.resolve()}")


def load_pickle(file_path: Path) -> Any:
    """Load data from a pickle file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Solution file does not exist: {file_path}")

    with file_path.open("rb") as file:
        data = pickle.load(file)

    print(f"Solution data loaded from:\n  {file_path.resolve()}")
    return data


def get_solution_files(
    results_root: Path,
    pattern: str = "**/solution.pkl",
) -> list[Path]:
    """Return matching solution files ordered from oldest to newest."""
    files = [path for path in results_root.glob(pattern) if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime)


def get_latest_solution_file(
    results_root: Path,
    pattern: str = "**/solution.pkl",
) -> Path:
    """Return the most recently modified solution file."""
    files = get_solution_files(results_root, pattern)

    if not files:
        raise FileNotFoundError(
            f"No solution files matching '{pattern}' were found under "
            f"{results_root.resolve()}."
        )

    return files[-1]


def save_run_summary(
    solution_data: dict[str, Any],
    file_path: Path,
) -> None:
    """
    Save a lightweight, human-readable summary.

    Large arrays and optimizer histories remain in solution.pkl.
    """
    summary = {
        "name": solution_data.get("name"),
        "n_waypoints": int(solution_data["wp_xy"].shape[0]),
        "n_agents": int(solution_data["n_agents"]),
        "seed": int(solution_data["seed"]),
        "cost_mode": solution_data["cost_mode"],
        "cbf_mode": str(solution_data["cbf_mode"]),
        "runtime_seconds": float(solution_data["runtime"]),
        "final_cost": float(np.asarray(solution_data["cost"])[-1]),
        "offset_energy": float(solution_data["offset_energy"]),
        "filter_wp_thresh": float(solution_data["filter_wp_thresh"]),
        "prune_mode": bool(solution_data["prune_mode"]),
        "created_at": solution_data.get("created_at"),
        "lm": float(solution_data["lm"])
    }

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"Run summary saved to:\n  {file_path.resolve()}")


# ============================================================
# Plot-management utilities
# ============================================================

def validate_solution_data_for_plotting(
    solution_data: dict[str, Any],
) -> None:
    """Check that the loaded solution contains the required plotting fields."""
    required_keys = {
        "wp_xy",
        "mask",
        "dist_mat",
        "sd_mat",
        "agent_routes",
        "agent_schedules",
        "agent_colors",
        "T_array",
        "association_mat",
        "processing_time",
        "cat",
    }

    missing_keys = required_keys.difference(solution_data)

    if missing_keys:
        raise KeyError(
            "The solution file is missing fields required for plotting: "
            + ", ".join(sorted(missing_keys))
        )


def save_solution_plots(
    solution_data: dict[str, Any],
    output_directory: Path,
    *,
    show_plots: bool = False,
    dpi: int = 300,
    overwrite: bool = True,
) -> dict[str, Path]:
    """
    Generate network and schedule plots from a solution-data dictionary.

    This function works for:
      1. newly generated solutions;
      2. solutions loaded from an existing pickle file.
    """
    validate_solution_data_for_plotting(solution_data)

    output_directory.mkdir(parents=True, exist_ok=True)

    network_path = output_directory / "network.png"
    schedules_path = output_directory / "schedules.png"

    if not overwrite:
        existing_files = [
            path for path in (network_path, schedules_path) if path.exists()
        ]
        if existing_files:
            raise FileExistsError(
                "The following plot files already exist:\n"
                + "\n".join(str(path) for path in existing_files)
            )

    visualize.plotNetwork(
        figuresize=(20, 14),
        wp_xy=solution_data["wp_xy"],
        mask=solution_data["mask"],
        dist_mat=solution_data["dist_mat"],
        sd_mat=solution_data["sd_mat"],
        routes=solution_data["agent_routes"],
        schedules=solution_data["agent_schedules"],
        agent_colors=solution_data["agent_colors"],
        showEdgeLength=False,
        save_path=network_path,
        dpi=dpi,
        show_plot=show_plots,
    )

    visualize.plot_waypoint_agent_schedules(
        routes=solution_data["agent_routes"],
        schedules=solution_data["agent_schedules"],
        schedule_matrix=solution_data["T_array"][-1],
        association_matrix=solution_data["association_mat"],
        process_T=solution_data["processing_time"],
        tolArray=solution_data["cat"],
        agent_colors=solution_data["agent_colors"],
        figuresize=(30, 24),
        bar_thickness=0.3,
        marker_size=20,
        index_size=40,
        x_tick_size=36,
        y_tick_size=36,
        x_label_size=48,
        y_label_size=48,
        colorbar_tick_size=28,
        colorbar_label_size=30,
        safe_gap_multiplier=2.0,
        save_path=schedules_path,
        dpi=300,
        show_plot=show_plots,
    )

    print(
        "Plots saved to:\n"
        f"  {network_path.resolve()}\n"
        f"  {schedules_path.resolve()}"
    )

    return {
        "network": network_path,
        "schedules": schedules_path,
    }


# ============================================================
# Optimization utilities
# ============================================================

def create_mirs_environment(
    n_waypoints: int,
    n_agents: int,
    seed: int,
    cost_mode: str,
    lm: float,
) -> tuple[MIRS.MIRS, Any, np.ndarray, Any]:
    """Create the network parameters and MIRS environment."""
    net_params, tolerance_array = problem_generator.get_network_params(
        n_waypoints,
        tol_range=[3.0, 5.0],
        name="random",
    )

    cbf_mode = utils.get_cbf_mode(
        "rect",
        tolerance_array,
    )

    mirs = MIRS.MIRS(
        n_waypoints,
        n_agents,
        tolerance_array,
        net_params,
        seed=seed,
        offset_energy=1,
        stagewiseCostCoeffs=np.array([0.1, 10.0, 0.1]),
        selfHop=0,
        cost_mode=cost_mode,
        lm=lm,
        ca_cbf=cbf_mode,
        filter_wp_thresh=1e-10,
        prune_mode=False,
        printFlag=False,
    )

    return mirs, net_params, tolerance_array, cbf_mode


def create_agent_colors(n_agents: int) -> dict[int, Any]:
    """Assign one color to each agent."""
    colormap = plt.get_cmap("tab20")

    return {
        agent_id: colormap(agent_id / max(n_agents, 1))
        for agent_id in range(n_agents)
    }


def run_mep_optimization(
    mirs: MIRS.MIRS,
) -> tuple[mep_opt.MIRSOptimizer, dict[str, Any]]:
    """Run the MEP optimizer and return its raw outputs."""
    initial_schedule = mirs.sched_mat.copy()
    initial_speed = mirs.speed_vec.copy()
    active_waypoints = range(mirs.n_waypoints)

    mirs.prune_mode = True
    mirs.filter_wp_thresh = 0.01

    optimizer_config, anneal_config = utils.set_mep_opt_config("cbf")

    optimizer = mep_opt.MIRSOptimizer(
        mirs,
        optimizer_config,
        anneal_config,
    )

    start_time = time.perf_counter()

    (
        schedule_array,
        speed_array,
        cost_array,
        final_probability,
        chi_array,
        compute_time_array,
    ) = optimizer.anneal(
        initial_schedule,
        initial_speed,
        active_waypoints=active_waypoints,
        annealPrint=True,
    )

    runtime = time.perf_counter() - start_time

    routes, schedules = optimizer.mirs.solution_table(
        final_probability,
        schedule_array[-1],
        speed_array[-1],
        optimizer.b_arr[-1],
    )

    # Kept as speed_array[-2] because that was used in the original code.
    # Change to speed_array[-1] if the final speed is intended.
    reach_matrix, association_matrix = (
        optimizer.mirs.calc_agent_reach_mat_v1(
            schedule_array[-1],
            speed_array[-2],
            beta=optimizer.b_arr[-1],
        )
    )
    print(f"run_mep_optimization: cost_array:\n{cost_array}")

    outputs = {
        "T_array": schedule_array,
        "V_array": speed_array,
        "cost": cost_array,
        "runtime": runtime,
        "final_Pb": final_probability,
        "chi_arr": chi_array,
        "t_compute_array": compute_time_array,
        "agent_routes": routes,
        "agent_schedules": schedules,
        "reach_mat": reach_matrix,
        "association_mat": association_matrix,
        "optim_config": optimizer_config,
        "anneal_config": anneal_config,
    }

    return optimizer, outputs


def build_solution_data(
    optimizer: mep_opt.MIRSOptimizer,
    optimization_outputs: dict[str, Any],
    net_params: Any,
    cbf_mode: Any,
    agent_colors: dict[int, Any],
    seed: int,
    cost_mode: str,
) -> dict[str, Any]:
    """Construct the complete serializable solution dictionary."""
    mirs = optimizer.mirs

    return {
        "name": OPTIMIZER_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),

        "n_agents": mirs.n_agents,
        "n_waypoints": mirs.n_waypoints,

        "wp_xy": mirs.wp_locations,
        "wp_params": net_params,
        "mask": mirs.mask,
        "dist_mat": mirs.dist_mat,
        "sd_mat": mirs.sd_mat,

        "T_array": optimization_outputs["T_array"],
        "V_array": optimization_outputs["V_array"],
        "cost": optimization_outputs["cost"],
        "runtime": optimization_outputs["runtime"],
        "final_Pb": optimization_outputs["final_Pb"],
        "chi_arr": optimization_outputs["chi_arr"],
        "t_compute_array": optimization_outputs["t_compute_array"],

        "agent_routes": optimization_outputs["agent_routes"],
        "agent_schedules": optimization_outputs["agent_schedules"],
        "reach_mat": optimization_outputs["reach_mat"],
        "association_mat": optimization_outputs["association_mat"],
        "agent_colors": agent_colors,

        "b_arr": optimizer.b_arr,
        "processing_time": mirs.process_T,
        "cat": mirs.tolArray,

        "optim_config": optimization_outputs["optim_config"],
        "anneal_config": optimization_outputs["anneal_config"],

        "cost_mode": cost_mode,
        "cbf_mode": cbf_mode,
        "seed": seed,
        "offset_energy": mirs.offset_energy,
        "stagewise_cost_coeffs": mirs.stagewiseCostCoeffs,
        "prune_mode": mirs.prune_mode,
        "filter_wp_thresh": mirs.filter_wp_thresh,
        "inf": mirs.INF,
        "speed_lim_mat": mirs.speed_lim_mat,
        "start_times": mirs.start_times,
        "t_up": mirs.T_upper_bound,
        "agent_weights": mirs.agent_weights,
        "wp_weights": mirs.wp_weights,
        "lm": mirs.lm,
        "self_hop": mirs.selfHop,
        "run_config": {
            "n_waypoints": mirs.n_waypoints,
            "n_agents": mirs.n_agents,
            "seed": seed,
            "cost_mode": cost_mode,
            "lm": mirs.lm,
            "cbf_mode": cbf_mode,
        },
    }


# ============================================================
# Main workflows
# ============================================================

def optimize_and_save(
    results_root: Path,
    n_waypoints: int,
    n_agents: int,
    seed: int,
    cost_mode: str,
    lm: float,
    *,
    show_plots: bool = False,
    dpi: int = 300,
    overwrite_plots: bool = True,
) -> Path:
    """Run a new optimization and save its data and plots."""
    results_root.mkdir(parents=True, exist_ok=True)

    mirs, net_params, tolerance_array, cbf_mode = create_mirs_environment(
        n_waypoints=n_waypoints,
        n_agents=n_agents,
        seed=seed,
        cost_mode=cost_mode,
        lm=lm,
    )

    agent_colors = create_agent_colors(mirs.n_agents)

    optimizer, optimization_outputs = run_mep_optimization(mirs)

    solution_data = build_solution_data(
        optimizer=optimizer,
        optimization_outputs=optimization_outputs,
        net_params=net_params,
        cbf_mode=cbf_mode,
        agent_colors=agent_colors,
        seed=seed,
        cost_mode=cost_mode,
    )

    run_directory = create_run_directory(
        results_root=results_root,
        optimizer_name=OPTIMIZER_NAME,
        n_waypoints=mirs.n_waypoints,
        n_agents=mirs.n_agents,
        seed=seed,
    )

    solution_file = run_directory / "solution.pkl"
    summary_file = run_directory / "summary.json"

    save_pickle(solution_data, solution_file)
    save_run_summary(solution_data, summary_file)

    save_solution_plots(
        solution_data,
        output_directory=run_directory,
        show_plots=show_plots,
        dpi=dpi,
        overwrite=overwrite_plots
    )

    print(f"Run directory:\n {run_directory.resolve()}")

    return solution_file


def regenerate_plots_from_file(
    solution_file: Path,
    *,
    output_directory: Path | None = None,
    show_plots: bool = False,
    dpi: int = 300,
    overwrite: bool = True,
) -> dict[str, Path]:
    solution_file = solution_file.expanduser().resolve()
    solution_data = load_pickle(solution_file)

    if output_directory is None:
        output_directory = solution_file.parent

    return save_solution_plots(
        solution_data,
        output_directory=output_directory,
        show_plots=show_plots,
        dpi=dpi,
        overwrite=overwrite,
    )

def regenerate_all_existing_plots(
    results_root: Path,
    pattern: str = "**/solution.pkl",
    *,
    dpi: int = 300,
    overwrite: bool = True,
) -> None:
    solution_files = get_solution_files(results_root, pattern)

    if not solution_files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} were found under "
            f"{results_root.resolve()}."
        )

    failures = []

    for index, solution_file in enumerate(solution_files, start=1):
        print(
            f"\n[{index}/{len(solution_files)}] Processing:\n"
            f"  {solution_file.resolve()}"
        )

        try:
            regenerate_plots_from_file(
                solution_file=solution_file,
                show_plots=False,
                dpi=dpi,
                overwrite=overwrite,
            )
        except Exception as error:
            failures.append((solution_file, error))
            print(f"Failed: {error}")

    successful_count = len(solution_files) - len(failures)

    print(
        f"\nCompleted plot regeneration for "
        f"{successful_count}/{len(solution_files)} solution files."
    )

    if failures:
        print("\nFailed files:")
        for solution_file, error in failures:
            print(f"  {solution_file}: {error}")


def main() -> None:
    args = parse_arguments()

    print(f"\nSelected run mode: {args.run_mode}")

    if args.run_mode == "optimize":
        optimize_and_save(
            results_root=args.results_root,
            n_waypoints=args.n_waypoints,
            n_agents=args.n_agents,
            seed=args.seed,
            cost_mode=args.cost_mode,
            lm=args.lm,
            show_plots=args.show_plots,
            dpi=args.dpi,
            overwrite_plots=not args.no_overwrite,
        )

    elif args.run_mode == "load":
        solution_file = args.solution_file

        if solution_file is None:
            solution_file = get_latest_solution_file(
                results_root=args.results_root,
                pattern=args.pattern,
            )

            print(
                "No solution file was specified. "
                "Using the most recent solution:\n"
                f"  {solution_file.resolve()}"
            )

        regenerate_plots_from_file(
            solution_file=solution_file,
            show_plots=args.show_plots,
            dpi=args.dpi,
            overwrite=not args.no_overwrite,
        )

    elif args.run_mode == "load-all":
        regenerate_all_existing_plots(
            results_root=args.results_root,
            pattern=args.pattern,
            dpi=args.dpi,
            overwrite=not args.no_overwrite,
        )

    else:
        # This should never occur because argparse validates the choices.
        raise RuntimeError(f"Unexpected run mode: {args.run_mode}")


if __name__ == "__main__":
    main()

