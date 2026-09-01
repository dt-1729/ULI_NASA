from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import MIRS
import gb_opt
import mep_opt
import utils
import visualize


DEFAULT_SCENARIO_ROOT = Path("local_scenarios")
SUPPORTED_METHODS = ("cbf", "slsqp", "cbf_static", "slsqp_static", "gurobi")


def load_scenario_data(scenario_dir: Path) -> Dict[str, Any]:
    data_file = scenario_dir / "scenario_data.pkl"
    if not data_file.exists():
        raise FileNotFoundError(f"Scenario data file not found: {data_file}")

    with open(data_file, "rb") as f:
        return pickle.load(f)


def list_scenario_dirs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Scenario root does not exist: {root_dir}")

    scenario_dirs = sorted(p for p in root_dir.iterdir() if p.is_dir() and "scenario_" in p.name)
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario directories found under {root_dir}")
    return scenario_dirs


def reconstruct_mirs_from_scenario(scenario_data: Dict[str, Any]) -> MIRS.MIRS:
    kwargs = dict(scenario_data["mirs_constructor_kwargs"])
    return MIRS.MIRS(**kwargs)


def output_filename_for_method(method: str) -> str:
    method = method.lower()
    mapping = {
        "cbf": "solution_cbf.pkl",
        "slsqp": "solution_slsqp.pkl",
        "cbf_static": "solution_cbf_static.pkl",
        "slsqp_static": "solution_slsqp_static.pkl",
        "gurobi": "solution_gurobi.pkl",
    }
    if method not in mapping:
        raise ValueError(f"Unsupported method: {method}")
    return mapping[method]


def _base_solution_data(
    method: str,
    scenario_data: Dict[str, Any],
    scenario_dir: Path,
    mirs: MIRS.MIRS,
    cost: float,
    runtime: float,
    agent_routes: List[List[int]],
    agent_schedules: List[List[float]],
    association_mat: np.ndarray,
    T_array: np.ndarray | None = None,
    V_array: np.ndarray | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    if "mirs_summary" in scenario_data:
        summary = scenario_data["mirs_summary"]
    else:
        summary = {}

    payload = {
        "name": {
            "cbf": "MEP_CBF",
            "slsqp": "MEP_SLSQP",
            "cbf_static": "MEP_CBF_STATIC",
            "slsqp_static": "MEP_SLSQP_STATIC",
            "gurobi": "GUROBI",
        }[method],
        "n_agents": mirs.n_agents,
        "wp_xy": summary.get("wp_locations", mirs.wp_locations),
        "wp_params": scenario_data.get("network_params", scenario_data["mirs_constructor_kwargs"]["wp_params"]),
        "mask": summary.get("mask", mirs.mask),
        "dist_mat": summary.get("dist_mat", mirs.dist_mat),
        "sd_mat": summary.get("sd_mat", mirs.sd_mat),
        "cost": cost,
        "runtime": runtime,
        "agent_routes": agent_routes,
        "agent_schedules": agent_schedules,
        "association_mat": association_mat,
        "processing_time": summary.get("process_T", mirs.process_T),
        "cat": mirs.tolArray,
        "cost_mode": scenario_data.get("cost_mode", "sum"),
        "seed": scenario_data.get("seed"),
        "offset_energy": scenario_data["mirs_constructor_kwargs"]["offset_energy"],
        "stagewise_cost_coeffs": scenario_data["mirs_constructor_kwargs"]["stagewiseCostCoeffs"],
        "prune_mode": scenario_data["mirs_constructor_kwargs"]["prune_mode"],
        "filter_wp_thresh": scenario_data["mirs_constructor_kwargs"]["filter_wp_thresh"],
        "inf": mirs.INF,
        "speed_lim_mat": summary.get("speed_lim_mat", mirs.speed_lim_mat),
        "start_times": summary.get("start_times", mirs.start_times),
        "t_up": summary.get("T_upper_bound", mirs.T_upper_bound),
        "agent_weights": summary.get("agent_weights", mirs.agent_weights),
        "wp_weights": summary.get("wp_weights", mirs.wp_weights),
        "lm": scenario_data["mirs_constructor_kwargs"]["lm"],
        "self_hop": scenario_data["mirs_constructor_kwargs"]["selfHop"],
        "scenario_id": scenario_data.get("scenario_id"),
        "scenario_dir": str(scenario_dir),
    }

    if T_array is not None:
        payload["T_array"] = T_array
    if V_array is not None:
        payload["V_array"] = V_array
    if method in {"cbf", "slsqp", "cbf_static", "slsqp_static"}:
        payload["cbf_mode"] = scenario_data["mirs_constructor_kwargs"]["ca_cbf"]

    payload.update(extra)
    return payload


def solve_cbf_scenario(
    scenario_dir: Path,
    scenario_data: Dict[str, Any],
    anneal_print: bool = False,
    time_limit: float = 3600.0,
    method_name: str = "cbf",
) -> Dict[str, Any]:
    mirs = reconstruct_mirs_from_scenario(scenario_data)

    initial_conditions = scenario_data["initial_conditions"]
    T0 = np.array(initial_conditions["T0"], dtype=float)
    V0 = np.array(initial_conditions["V0"], dtype=float)
    active_waypoints = list(initial_conditions["active_waypoints"])

    optimizer_specs = scenario_data["optimizer_specs"]["cbf_mep"]
    optim_config = optimizer_specs["config"]
    anneal_config = optimizer_specs["anneal_config"]

    t0 = time.time()
    optimizer = mep_opt.MIRSOptimizer(mirs, optim_config, anneal_config)
    T_array, V_array, F_vals, Pb_a, chi_array, t_compute_array = optimizer.anneal(
        T0,
        V0,
        active_waypoints=active_waypoints,
        annealPrint=anneal_print,
    )
    runtime = time.time() - t0

    agent_routes, agent_schedules = optimizer.mirs.solution_table(
        Pb_a,
        T_array[-1],
        V_array[-1],
        optimizer.b_arr[-1],
    )

    reach_mat, association_mat = optimizer.mirs.calc_agent_reach_mat_v1(
        T_array[-1],
        V_array[-2],
        beta=optimizer.b_arr[-1],
    )

    final_cost = F_vals[-1] if isinstance(F_vals, np.ndarray) and F_vals.size > 0 else F_vals
    solution_data = _base_solution_data(
        method=method_name,
        scenario_data=scenario_data,
        scenario_dir=scenario_dir,
        mirs=mirs,
        cost=final_cost,
        runtime=runtime,
        agent_routes=agent_routes,
        agent_schedules=agent_schedules,
        association_mat=association_mat,
        T_array=T_array,
        V_array=V_array,
        final_Pb=Pb_a,
        chi_arr=chi_array,
        b_arr=optimizer.b_arr,
        optim_config=optim_config,
        anneal_config=anneal_config,
        compute_time_per_beta=t_compute_array,
        reach_mat=reach_mat,
    )

    output_path = scenario_dir / output_filename_for_method(method_name)
    with open(output_path, "wb") as f:
        pickle.dump(solution_data, f)

    return solution_data


def solve_slsqp_scenario(
    scenario_dir: Path,
    scenario_data: Dict[str, Any],
    anneal_print: bool = False,
    time_limit: float = 3600.0,
    method_name: str = "slsqp",
) -> Dict[str, Any]:
    mirs = reconstruct_mirs_from_scenario(scenario_data)

    initial_conditions = scenario_data["initial_conditions"]
    T0 = np.array(initial_conditions["T0"], dtype=float)
    V0 = np.array(initial_conditions["V0"], dtype=float)
    active_waypoints = list(initial_conditions["active_waypoints"])

    optimizer_specs = scenario_data["optimizer_specs"]["slsqp_mep"]
    optim_config = optimizer_specs["config"]
    anneal_config = optimizer_specs["anneal_config"]

    t0 = time.time()
    optimizer = mep_opt.MIRSOptimizer(mirs, optim_config, anneal_config)
    T_array, V_array, F_vals, Pb_a, chi_array, t_compute_array = optimizer.anneal(
        T0,
        V0,
        active_waypoints=active_waypoints,
        annealPrint=anneal_print,
    )
    runtime = time.time() - t0

    agent_routes, agent_schedules = optimizer.mirs.solution_table(
        Pb_a,
        T_array[-1],
        V_array[-1],
        optimizer.b_arr[-1],
    )

    reach_mat, association_mat = optimizer.mirs.calc_agent_reach_mat_v1(
        T_array[-1],
        V_array[-2],
        beta=optimizer.b_arr[-1],
    )

    final_cost = F_vals[-1] if isinstance(F_vals, np.ndarray) and F_vals.size > 0 else F_vals
    solution_data = _base_solution_data(
        method=method_name,
        scenario_data=scenario_data,
        scenario_dir=scenario_dir,
        mirs=mirs,
        cost=final_cost,
        runtime=runtime,
        agent_routes=agent_routes,
        agent_schedules=agent_schedules,
        association_mat=association_mat,
        T_array=T_array,
        V_array=V_array,
        final_Pb=Pb_a,
        chi_arr=chi_array,
        b_arr=optimizer.b_arr,
        optim_config=optim_config,
        anneal_config=anneal_config,
        compute_time_per_beta=t_compute_array,
        reach_mat=reach_mat,
    )

    output_path = scenario_dir / output_filename_for_method(method_name)
    with open(output_path, "wb") as f:
        pickle.dump(solution_data, f)

    return solution_data


def solve_gurobi_scenario(
    scenario_dir: Path,
    scenario_data: Dict[str, Any],
    time_limit: float = 3600.0,
) -> Dict[str, Any]:
    mirs = reconstruct_mirs_from_scenario(scenario_data)

    t0 = time.time()
    optimizer = gb_opt.MIRSGurobiOptimizer(mirs, "gb_mirs_model")
    optimizer.model.setParam("OutputFlag", 0)
    optimizer.optimize(time_limit=float(time_limit), mip_gap=0.05, stagnation_limit=20000)
    runtime = time.time() - t0

    if optimizer.model.SolCount == 0:
        raise RuntimeError(f"No feasible Gurobi solution found for scenario {scenario_dir}")

    agent_routes, agent_schedules, agent_speeds, T_mat_gb, assoc_mat_gb = optimizer.extract_routes_and_schedules()
    final_cost = optimizer.model.ObjVal

    solution_data = _base_solution_data(
        method="gurobi",
        scenario_data=scenario_data,
        scenario_dir=scenario_dir,
        mirs=mirs,
        cost=final_cost,
        runtime=runtime,
        agent_routes=agent_routes,
        agent_schedules=agent_schedules,
        association_mat=assoc_mat_gb,
        T_array=T_mat_gb,
        V_array=None,
        T_mat=T_mat_gb,
        speeds=agent_speeds,
        agent_speeds=agent_speeds,
    )

    output_path = scenario_dir / output_filename_for_method("gurobi")
    with open(output_path, "wb") as f:
        pickle.dump(solution_data, f)

    return solution_data


def solve_scenario_by_method(
    scenario_dir: Path,
    scenario_data: Dict[str, Any],
    method: str,
    anneal_print: bool = False,
    time_limit: float = 3600.0,
) -> Dict[str, Any]:
    method = method.lower()
    if method in {"cbf", "cbf_static"}:
        original_cbf = scenario_data["mirs_constructor_kwargs"].get("ca_cbf")
        try:
            if method == "cbf_static":
                scenario_data["mirs_constructor_kwargs"]["ca_cbf"] = utils.get_cbf_mode("lin_static", np.asarray(scenario_data["tol_array"]))
            return solve_cbf_scenario(scenario_dir, scenario_data, anneal_print=anneal_print, time_limit=time_limit, method_name=method)
        finally:
            if original_cbf is not None:
                scenario_data["mirs_constructor_kwargs"]["ca_cbf"] = original_cbf
            else:
                scenario_data["mirs_constructor_kwargs"].pop("ca_cbf", None)
    if method in {"slsqp", "slsqp_static"}:
        original_cbf = scenario_data["mirs_constructor_kwargs"].get("ca_cbf")
        try:
            if method == "slsqp_static":
                scenario_data["mirs_constructor_kwargs"]["ca_cbf"] = utils.get_cbf_mode("lin_static", np.asarray(scenario_data["tol_array"]))
            return solve_slsqp_scenario(scenario_dir, scenario_data, anneal_print=anneal_print, time_limit=time_limit, method_name=method)
        finally:
            if original_cbf is not None:
                scenario_data["mirs_constructor_kwargs"]["ca_cbf"] = original_cbf
            else:
                scenario_data["mirs_constructor_kwargs"].pop("ca_cbf", None)
    if method == "gurobi":
        return solve_gurobi_scenario(scenario_dir, scenario_data, time_limit=time_limit)
    raise ValueError(f"Unsupported method '{method}'. Supported methods: {SUPPORTED_METHODS}")


def solve_all_scenarios(root_dir: Path, method: str, anneal_print: bool = False, time_limit: float = 3600.0) -> List[Path]:
    scenario_dirs = list_scenario_dirs(root_dir)
    solved_paths: List[Path] = []

    print(f"[{method.upper()}] Starting solve for {len(scenario_dirs)} scenarios in {root_dir}")
    for idx, scenario_dir in enumerate(scenario_dirs, start=1):
        print(f"[{method.upper()}] Scenario {idx}/{len(scenario_dirs)}: {scenario_dir.name}")
        scenario_data = load_scenario_data(scenario_dir)
        result = solve_scenario_by_method(
            scenario_dir,
            scenario_data,
            method,
            anneal_print=anneal_print,
            time_limit=time_limit,
        )
        print(
            f"[{method.upper()}] Finished {scenario_dir.name} | "
            f"cost={result.get('cost', 'N/A')} | runtime={result.get('runtime', 'N/A')}s"
        )
        solved_paths.append(scenario_dir)

    print(f"[{method.upper()}] Completed all solves for {len(solved_paths)} scenarios")
    return solved_paths


def plot_scenario_solution(scenario_dir: Path, method: str, solution_data: Dict[str, Any]) -> None:
    wp_locs = np.asarray(solution_data["wp_xy"])
    mask = np.asarray(solution_data["mask"])
    dist_mat = np.asarray(solution_data["dist_mat"])
    sd_mat = np.asarray(solution_data["sd_mat"])
    agent_routes = solution_data["agent_routes"]
    agent_schedules = solution_data["agent_schedules"]
    agent_colors = {}
    cmap = plt.get_cmap("tab20")
    for i in range(len(agent_routes)):
        agent_colors[i] = cmap(i / max(1, len(agent_routes)))

    if method in {"cbf", "slsqp", "cbf_static", "slsqp_static"}:
        T_schedule = solution_data["T_array"][-1]
    elif method == "gurobi":
        T_schedule = solution_data.get("T_mat", solution_data.get("T_array"))
        if T_schedule is None:
            T_schedule = np.array(solution_data["agent_schedules"])[:, :, 0]
    else:
        raise ValueError(f"Unsupported method '{method}'")

    assoc_mat = solution_data["association_mat"]
    process_T = solution_data.get("processing_time", np.zeros_like(np.asarray(solution_data["dist_mat"])))
    tol_array = solution_data.get("cat", np.ones(wp_locs.shape[0]))

    network_path = scenario_dir / f"network_plot_{method}.png"
    visualize.plotNetwork(
        figuresize=(20, 14),
        wp_xy=wp_locs,
        mask=mask,
        dist_mat=dist_mat,
        sd_mat=sd_mat,
        routes=agent_routes,
        schedules=agent_schedules,
        agent_colors=agent_colors,
        showEdgeLength=False,
        save_path=str(network_path),
        show_plot=False,
    )

    schedule_path = scenario_dir / f"schedule_plot_{method}.png"
    visualize.plot_waypoint_agent_schedules(
        agent_routes,
        agent_schedules,
        T_schedule,
        assoc_mat,
        process_T,
        tol_array,
        agent_colors,
        figuresize=(24, 24),
        bar_thickness=0.05,
        marker_size=8,
        save_path=str(schedule_path),
        show_plot=False,
    )

    print(f"Saved plots to: {network_path} and {schedule_path}")


def plot_all_scenarios(root_dir: Path, method: str) -> List[Path]:
    scenario_dirs = list_scenario_dirs(root_dir)
    plotted_dirs: List[Path] = []

    for scenario_dir in scenario_dirs:
        solution_path = scenario_dir / output_filename_for_method(method)
        if not solution_path.exists():
            print(f"Skipping {scenario_dir}: no {solution_path.name} found.")
            continue

        with open(solution_path, "rb") as f:
            solution_data = pickle.load(f)

        plot_scenario_solution(scenario_dir, method, solution_data)
        plotted_dirs.append(scenario_dir)

    return plotted_dirs


def _scenario_problem_size(scenario_data: Dict[str, Any]) -> float:
    n_agents = int(scenario_data.get("n_agents", scenario_data["mirs_constructor_kwargs"]["n_agents"]))
    n_waypoints = int(scenario_data.get("n_waypoints", scenario_data["mirs_constructor_kwargs"]["n_waypoints"]))
    return float(n_agents * (n_waypoints ** 3) + n_agents * n_waypoints)


def compare_methods_across_scenarios(
    root_dir: Path,
    methods: List[str] | None = None,
    save_path: Path | None = None,
) -> Path:
    if methods is None:
        methods = list(SUPPORTED_METHODS)
    methods = [m.lower() for m in methods]
    invalid_methods = [m for m in methods if m not in SUPPORTED_METHODS]
    if invalid_methods:
        raise ValueError(f"Unsupported methods requested: {invalid_methods}. Supported methods: {SUPPORTED_METHODS}")

    scenario_dirs = list_scenario_dirs(root_dir)
    scenario_records: List[tuple[Path, Dict[str, Any], Dict[str, Dict[str, Any]]]] = []

    for scenario_dir in scenario_dirs:
        scenario_data = load_scenario_data(scenario_dir)
        method_solutions: Dict[str, Dict[str, Any]] = {}
        for method in methods:
            solution_path = scenario_dir / output_filename_for_method(method)
            if not solution_path.exists():
                break
            with open(solution_path, "rb") as f:
                method_solutions[method] = pickle.load(f)
        else:
            scenario_records.append((scenario_dir, scenario_data, method_solutions))

    if not scenario_records:
        raise FileNotFoundError(
            f"No complete method solution files found under {root_dir} for methods: {methods}."
        )

    sorted_records = sorted(scenario_records, key=lambda item: _scenario_problem_size(item[1]))
    x_values = np.array([_scenario_problem_size(scenario_data) for _, scenario_data, _ in sorted_records], dtype=float)

    method_labels = {
        "cbf": "CBF",
        "slsqp": "SLSQP",
        "cbf_static": "CBF Static",
        "slsqp_static": "SLSQP Static",
        "gurobi": "Gurobi",
    }
    method_colors = {
        "cbf": "tab:blue",
        "slsqp": "tab:orange",
        "cbf_static": "tab:cyan",
        "slsqp_static": "tab:brown",
        "gurobi": "tab:green",
    }

    plotting_values = {method: {"cost": [], "runtime": []} for method in methods}
    for _, scenario_data, method_solutions in sorted_records:
        for method in methods:
            solution = method_solutions[method]
            plotting_values[method]["cost"].append(float(solution.get("cost", np.nan)))
            plotting_values[method]["runtime"].append(float(solution.get("runtime", np.nan)))

    if save_path is None:
        save_path = root_dir / "method_comparison_cost_runtime.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for method in methods:
        costs = np.asarray(plotting_values[method]["cost"], dtype=float)
        runtimes = np.asarray(plotting_values[method]["runtime"], dtype=float)

        axes[0].plot(x_values, costs, marker="o", linewidth=2, label=method_labels[method], color=method_colors[method])
        axes[1].plot(x_values, runtimes, marker="s", linewidth=2, label=method_labels[method], color=method_colors[method])

    axes[0].set_xscale("log")
    axes[0].set_xlabel("N*M^3 + N*M")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("Cost comparison across scenarios")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_xlabel("N*M^3 + N*M")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_title("Runtime comparison across scenarios")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison plot to: {save_path}")
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve or plot MIRS scenarios using the CBF, SLSQP, or Gurobi optimizer workflow from main.ipynb."
        )
    )
    parser.add_argument(
        "mode",
        choices=("solve", "plot", "compare"),
        default="solve",
        help="Execution mode. 'solve' computes a solution; 'plot' renders saved solution plots; 'compare' plots cost/runtime across all solved scenarios and methods.",
    )
    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default="cbf",
        help="Optimizer to run: cbf, slsqp, cbf_static, slsqp_static, or gurobi.",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=DEFAULT_SCENARIO_ROOT,
        help=f"Folder containing scenario subfolders. Default: {DEFAULT_SCENARIO_ROOT}",
    )
    parser.add_argument(
        "--scenario-path",
        type=Path,
        default=None,
        help="Optional exact scenario directory to process instead of all scenarios in --scenario-root.",
    )
    parser.add_argument(
        "--anneal-print",
        action="store_true",
        help="Enable detailed annealing printout for the CBF/SLSQP optimizer during solve mode.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=None,
        help="Methods to include in compare mode. Defaults to all methods when omitted. Options: cbf slsqp cbf_static slsqp_static gurobi.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=3600.0,
        help="Maximum wall-clock time per optimization in seconds. Default: 3600.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method = args.method.lower()

    if args.mode == "solve":
        if args.scenario_path is not None:
            scenario_dir = Path(args.scenario_path)
            print(f"[{method.upper()}] Solving single scenario: {scenario_dir}")
            scenario_data = load_scenario_data(scenario_dir)
            solved = solve_scenario_by_method(
                scenario_dir,
                scenario_data,
                method,
                anneal_print=args.anneal_print,
                time_limit=args.time_limit,
            )
            print(f"[{method.upper()}] Solved scenario: {scenario_dir}")
            print(f"[{method.upper()}] Final cost: {solved['cost']}")
            print(f"[{method.upper()}] Saved solution to: {scenario_dir / output_filename_for_method(method)}")
            return

        print(f"[{method.upper()}] Solve mode with time limit {args.time_limit}s")
        solved_dirs = solve_all_scenarios(
            args.scenario_root,
            method,
            anneal_print=args.anneal_print,
            time_limit=args.time_limit,
        )
        print(f"[{method.upper()}] Solved {len(solved_dirs)} scenarios under {args.scenario_root}")
        for d in solved_dirs:
            print(f"- {d.name}: {output_filename_for_method(method)}")
        return

    if args.mode == "compare":
        methods = args.methods if args.methods is not None else list(SUPPORTED_METHODS)
        comparison_path = compare_methods_across_scenarios(
            args.scenario_root,
            methods=methods,
            save_path=args.scenario_root / "method_comparison_cost_runtime.png",
        )
        print(f"Saved cross-method comparison plot for {methods}: {comparison_path}")
        return

    if args.scenario_path is not None:
        scenario_dir = Path(args.scenario_path)
        solution_path = scenario_dir / output_filename_for_method(method)
        if not solution_path.exists():
            raise FileNotFoundError(f"No {solution_path.name} found for scenario: {scenario_dir}")

        with open(solution_path, "rb") as f:
            solution_data = pickle.load(f)

        plot_scenario_solution(scenario_dir, method, solution_data)
        print(f"Plotted scenario with {method}: {scenario_dir}")
        return

    plotted_dirs = plot_all_scenarios(args.scenario_root, method)
    print(f"Plotted {len(plotted_dirs)} scenarios under {args.scenario_root} with {method}")
    for d in plotted_dirs:
        print(f"- {d.name}: network_plot_{method}.png and schedule_plot_{method}.png")


if __name__ == "__main__":
    main()
