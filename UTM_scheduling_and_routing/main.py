from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import MIRS
import cbs_search
import gb_opt
import mep_opt
import seq_opt
import utils
import visualize


DEFAULT_SCENARIO_ROOT = Path("local_scenarios")
SUPPORTED_METHODS = ("cbf", "slsqp", "cbf_static", "slsqp_static", "gurobi", "cbs", "seq")


def load_scenario_data(scenario_dir: Path) -> Dict[str, Any]:
    data_file = scenario_dir / "scenario_data.pkl"
    if not data_file.exists():
        raise FileNotFoundError(f"Scenario data file not found: {data_file}")

    with open(data_file, "rb") as f:
        return pickle.load(f)


def list_scenario_dirs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Scenario root does not exist: {root_dir}")

    scenario_dirs = sorted(
        data_file.parent
        for data_file in root_dir.rglob("scenario_data.pkl")
        if data_file.is_file()
    )
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
        "cbs": "solution_cbs.pkl",
        "seq": "solution_seq.pkl",
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
            "cbs": "CBS",
                "seq": "SEQUENTIAL_SHORTEST_PATH",
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
    optimizer_config_source: str = "scenario",
) -> Dict[str, Any]:
    mirs = reconstruct_mirs_from_scenario(scenario_data)

    initial_conditions = scenario_data["initial_conditions"]
    T0 = np.array(initial_conditions["T0"], dtype=float)
    V0 = np.array(initial_conditions["V0"], dtype=float)
    active_waypoints = list(initial_conditions["active_waypoints"])

    key = "cbf_static_mep" if method_name == "cbf_static" else "cbf_mep"
    optimizer_specs = None
    if optimizer_config_source == "scenario":
        optimizer_specs = scenario_data["optimizer_specs"].get(key)
    if optimizer_specs is None:
        config_name = "cbf_static" if method_name == "cbf_static" else "cbf"
        optim_config, anneal_config = utils.set_mep_opt_config(config_name)
        optimizer_specs = {"config": optim_config, "anneal_config": anneal_config}
    else:
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
    optimizer_config_source: str = "scenario",
) -> Dict[str, Any]:
    mirs = reconstruct_mirs_from_scenario(scenario_data)

    initial_conditions = scenario_data["initial_conditions"]
    T0 = np.array(initial_conditions["T0"], dtype=float)
    V0 = np.array(initial_conditions["V0"], dtype=float)
    active_waypoints = list(initial_conditions["active_waypoints"])

    key = "slsqp_static_mep" if method_name == "slsqp_static" else "slsqp_mep"
    optimizer_specs = None
    if optimizer_config_source == "scenario":
        optimizer_specs = scenario_data["optimizer_specs"].get(key)
    if optimizer_specs is None:
        config_name = "slsqp_static" if method_name == "slsqp_static" else "slsqp"
        optim_config, anneal_config = utils.set_mep_opt_config(config_name)
        optimizer_specs = {"config": optim_config, "anneal_config": anneal_config}
    else:
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


def solve_seq_scenario(
    scenario_dir: Path,
    scenario_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Solve one scenario with sequential shortest-path routing and scheduling."""
    mirs = reconstruct_mirs_from_scenario(scenario_data)
    t0 = time.time()
    routes, arrival_schedules = seq_opt.shortest_path_baseline(scenario_data)
    runtime = time.time() - t0

    n_agents = int(scenario_data["n_agents"])
    n_waypoints = int(scenario_data["n_waypoints"])
    schedule_matrix = np.full((n_agents, n_waypoints), np.nan, dtype=float)
    association_mat = np.zeros((n_agents, n_waypoints), dtype=float)
    for agent_index, (route, schedule) in enumerate(zip(routes, arrival_schedules)):
        for waypoint, arrival_time in zip(route, schedule):
            schedule_matrix[agent_index, waypoint] = arrival_time
            association_mat[agent_index, waypoint] = 1.0

    # The network plot uses one extra initial timestamp to label each edge.
    plot_schedules = [[schedule[0], *schedule] for schedule in arrival_schedules]
    cost_schedule = schedule_matrix.copy()
    cost_schedule[~np.isfinite(cost_schedule)] = mirs.T_upper_bound

    optimizer_specs = scenario_data.get("optimizer_specs", {}).get("cbf_mep")
    if optimizer_specs is None:
        _, anneal_config = utils.set_mep_opt_config("cbf")
    else:
        anneal_config = optimizer_specs["anneal_config"]
    terminal_beta = 10.0 ** float(anneal_config["log_bmax"])
    seq_speed_vec = mirs.speed_lim_mat[:, 1]
    seq_cost, _ = mirs.transportCost_v1(
        cost_schedule,
        seq_speed_vec,
        terminal_beta,
        returnGrad=False,
    )

    solution_data = _base_solution_data(
        method="seq",
        scenario_data=scenario_data,
        scenario_dir=scenario_dir,
        mirs=mirs,
        cost=float(seq_cost),
        runtime=runtime,
        agent_routes=routes,
        agent_schedules=plot_schedules,
        association_mat=association_mat,
        schedule_matrix=schedule_matrix,
        arrival_schedules=arrival_schedules,
    )

    output_path = scenario_dir / output_filename_for_method("seq")
    with open(output_path, "wb") as f:
        pickle.dump(solution_data, f)
    return solution_data


def solve_scenario_by_method(
    scenario_dir: Path,
    scenario_data: Dict[str, Any],
    method: str,
    anneal_print: bool = False,
    time_limit: float = 3600.0,
    cbs_time_step: float = 1.0,
    cbs_horizon: float | None = None,
    optimizer_config_source: str = "scenario",
) -> Dict[str, Any]:
    method = method.lower()
    if method == "cbs":
        return cbs_search.solve_cbs_scenario(
            scenario_dir,
            scenario_data,
            time_step=cbs_time_step,
            horizon=cbs_horizon,
        )
    if method in {"cbf", "cbf_static"}:
        original_cbf = scenario_data["mirs_constructor_kwargs"].get("ca_cbf")
        try:
            if method == "cbf_static":
                scenario_data["mirs_constructor_kwargs"]["ca_cbf"] = utils.get_cbf_mode("lin_static", np.asarray(scenario_data["tol_array"]))
            return solve_cbf_scenario(scenario_dir, scenario_data, anneal_print=anneal_print, time_limit=time_limit, method_name=method, optimizer_config_source=optimizer_config_source)
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
            return solve_slsqp_scenario(scenario_dir, scenario_data, anneal_print=anneal_print, time_limit=time_limit, method_name=method, optimizer_config_source=optimizer_config_source)
        finally:
            if original_cbf is not None:
                scenario_data["mirs_constructor_kwargs"]["ca_cbf"] = original_cbf
            else:
                scenario_data["mirs_constructor_kwargs"].pop("ca_cbf", None)
    if method == "gurobi":
        return solve_gurobi_scenario(scenario_dir, scenario_data, time_limit=time_limit)
    if method == "seq":
        return solve_seq_scenario(scenario_dir, scenario_data)
    raise ValueError(f"Unsupported method '{method}'. Supported methods: {SUPPORTED_METHODS}")


def solve_all_scenarios(
    root_dir: Path,
    method: str,
    anneal_print: bool = False,
    time_limit: float = 3600.0,
    cbs_time_step: float = 1.0,
    cbs_horizon: float | None = None,
    optimizer_config_source: str = "scenario",
) -> List[Path]:
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
            cbs_time_step=cbs_time_step,
            cbs_horizon=cbs_horizon,
            optimizer_config_source=optimizer_config_source,
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
    elif method == "cbs":
        T_schedule = solution_data["schedule_matrix"]
    elif method == "seq":
        T_schedule = solution_data["schedule_matrix"]
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


def _solution_arrival_times(solution: Dict[str, Any]) -> List[np.ndarray]:
    routes = solution.get("agent_routes", [])
    schedules = solution.get("agent_schedules", [])
    arrival_times: List[np.ndarray] = []

    for route, schedule in zip(routes, schedules):
        values = np.asarray(schedule, dtype=float).reshape(-1)
        if values.size >= len(route) + 1:
            values = values[1 : len(route) + 1]
        else:
            values = values[: len(route)]
        arrival_times.append(values)

    return arrival_times


def _solution_comparison_metrics(
    scenario_data: Dict[str, Any], solution: Dict[str, Any]
) -> Dict[str, float]:
    routes = solution.get("agent_routes", [])
    arrival_times = _solution_arrival_times(solution)
    summary = scenario_data.get("mirs_summary", {})

    start_times = np.asarray(
        solution.get("start_times", summary.get("start_times", np.zeros(len(routes)))),
        dtype=float,
    ).reshape(-1)
    if start_times.size < len(routes):
        start_times = np.pad(start_times, (0, len(routes) - start_times.size))

    travel_times = []
    for agent_index, times in enumerate(arrival_times):
        if times.size == 0:
            travel_times.append(np.nan)
        else:
            travel_times.append(float(times[-1] - start_times[agent_index]))

    finite_travel_times = np.asarray(travel_times, dtype=float)
    makespan = float(np.nanmax(finite_travel_times)) if finite_travel_times.size else np.nan
    sum_travel_time = float(np.nansum(finite_travel_times))

    tolerances = np.asarray(
        solution.get("cat", summary.get("tol_array", np.ones(scenario_data["n_waypoints"]))),
        dtype=float,
    ).reshape(-1)
    conflict_count = 0
    conflict_violation = 0.0
    for first_agent in range(len(routes)):
        first_arrivals = {
            int(node): [] for node in routes[first_agent]
        }
        for node, arrival_time in zip(routes[first_agent], arrival_times[first_agent]):
            first_arrivals[int(node)].append(float(arrival_time))

        for second_agent in range(first_agent + 1, len(routes)):
            second_arrivals = {
                int(node): [] for node in routes[second_agent]
            }
            for node, arrival_time in zip(routes[second_agent], arrival_times[second_agent]):
                second_arrivals[int(node)].append(float(arrival_time))

            for node in first_arrivals.keys() & second_arrivals.keys():
                if node >= tolerances.size:
                    continue
                threshold = 0.9 * tolerances[node]
                gaps = [
                    abs(first_time - second_time)
                    for first_time in first_arrivals[node]
                    for second_time in second_arrivals[node]
                ]
                conflicting_gaps = [gap for gap in gaps if gap < threshold]
                if conflicting_gaps:
                    conflict_count += 1
                    gap = min(conflicting_gaps)
                    tolerance = tolerances[node]
                    if tolerance > 0:
                        conflict_violation += abs(tolerance - gap) / tolerance

    return {
        "makespan": makespan,
        "sum_time": sum_travel_time,
        "runtime": float(solution.get("runtime", np.nan)),
        "conflicts": float(conflict_count),
        "conflict_violation": float(conflict_violation),
    }


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

    grouped_records: Dict[tuple[str, int, int], List[tuple[Path, Dict[str, Any], Dict[str, Dict[str, Any]]]]] = {}
    for record in scenario_records:
        scenario_data = record[1]
        key = (
            str(scenario_data.get("network_type", "unknown")),
            int(scenario_data["n_waypoints"]),
            int(scenario_data["n_agents"]),
        )
        grouped_records.setdefault(key, []).append(record)
    group_keys = sorted(grouped_records)
    x_values = np.arange(len(group_keys), dtype=float)

    method_labels = {
        "cbf": "CBF",
        "slsqp": "SLSQP",
        "cbf_static": "CBF Static",
        "slsqp_static": "SLSQP Static",
        "gurobi": "Gurobi",
        "cbs": "CBS",
        "seq": "Sequential shortest path",
    }
    method_colors = {
        "cbf": "tab:blue",
        "slsqp": "tab:orange",
        "cbf_static": "tab:cyan",
        "slsqp_static": "tab:brown",
        "gurobi": "tab:green",
        "cbs": "tab:red",
        "seq": "tab:purple",
    }

    metric_names = ("makespan", "sum_time", "runtime", "conflict_violation")
    plotting_values = {
        method: {
            metric: {"mean": [], "std": [], "count": []} for metric in metric_names
        }
        for method in methods
    }
    for group_key in group_keys:
        records = grouped_records[group_key]
        for method in methods:
            metrics_by_name = [
                _solution_comparison_metrics(data, solutions[method])
                for _, data, solutions in records
            ]
            for metric in metric_names:
                values = np.asarray([metrics[metric] for metrics in metrics_by_name], dtype=float)
                finite_values = values[np.isfinite(values)]
                plotting_values[method][metric]["mean"].append(
                    float(np.mean(finite_values)) if finite_values.size else np.nan
                )
                plotting_values[method][metric]["std"].append(
                    float(np.std(finite_values)) if finite_values.size else np.nan
                )
                plotting_values[method][metric]["count"].append(int(finite_values.size))

    if save_path is None:
        save_path = root_dir / "method_comparison_metrics.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = save_path.with_name(f"{save_path.stem}_summary.json")
    summary = {
        "groups": [
            {
                "network_type": network_type,
                "n_waypoints": n_waypoints,
                "n_agents": n_agents,
                "n_scenarios": len(grouped_records[(network_type, n_waypoints, n_agents)]),
            }
            for network_type, n_waypoints, n_agents in group_keys
        ],
        "methods": plotting_values,
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    metric_labels = {
        "makespan": "Makespan (slowest agent time)",
        "sum_time": "Sum of individual travel times",
        "runtime": "Solution runtime (s)",
        "conflict_violation": "Conflict violation score",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), squeeze=False)
    axes_by_metric = dict(zip(metric_names, axes.flat))

    for method in methods:
        for metric in metric_names:
            means = np.asarray(plotting_values[method][metric]["mean"], dtype=float)
            stds = np.asarray(plotting_values[method][metric]["std"], dtype=float)
            axes_by_metric[metric].errorbar(
                x_values,
                means,
                yerr=stds,
                marker="o",
                linewidth=1.5,
                capsize=4,
                label=method_labels[method],
                color=method_colors[method],
            )

    for metric, axis in axes_by_metric.items():
        axis.set_xticks(x_values)
        axis.set_xticklabels([f"{network}\nN={n_agents}, M={n_waypoints}" for network, n_waypoints, n_agents in group_keys])
        axis.set_xlabel("Scenario group (error bars show standard deviation)")
        axis.set_ylabel(metric_labels[metric])
        axis.set_title(f"Mean {metric_labels[metric]} across seeds")
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()

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
        help="Execution mode. 'solve' computes a solution; 'plot' renders saved solution plots; 'compare' plots four solution metrics across scenarios and methods.",
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
        default=None,
        metavar="METHOD",
        help="Methods to run, plot, or compare. Use 'all' to select every supported method.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=3600.0,
        help="Maximum wall-clock time per optimization in seconds. Default: 3600.",
    )
    parser.add_argument(
        "--cbs-time-step",
        type=float,
        default=1.0,
        help="Time resolution for CBS vertex conflicts. Default: 1.0.",
    )
    parser.add_argument(
        "--cbs-horizon",
        type=float,
        default=None,
        help="Optional time horizon for CBS searches.",
    )
    parser.add_argument(
        "--optimizer-config-source",
        choices=("scenario", "utils"),
        default="scenario",
        help=(
            "Source for CBF/SLSQP optimizer specs: 'scenario' uses the saved "
            "scenario_data.pkl values; 'utils' rebuilds them from "
            "utils.set_mep_opt_config(). Default: scenario."
        ),
    )
    args = parser.parse_args()
    if args.methods is not None:
        args.methods = [method.lower() for method in args.methods]
        invalid_methods = [
            method for method in args.methods
            if method not in SUPPORTED_METHODS and method != "all"
        ]
        if invalid_methods:
            parser.error(
                f"unsupported method(s): {invalid_methods}; "
                f"choose from {', '.join(SUPPORTED_METHODS)} or all"
            )
        if "all" in args.methods and args.methods != ["all"]:
            parser.error("'all' must be used by itself with --methods")
    return args


def main() -> None:
    args = parse_args()
    selected_methods = (
        list(SUPPORTED_METHODS)
        if args.methods == ["all"]
        else args.methods
    )
    method = selected_methods[0] if selected_methods else "cbf"

    if args.mode == "solve":
        solve_methods = selected_methods or [method]
        if args.scenario_path is not None:
            scenario_dir = Path(args.scenario_path)
            for solve_method in solve_methods:
                print(f"[{solve_method.upper()}] Solving single scenario: {scenario_dir}")
                scenario_data = load_scenario_data(scenario_dir)
                solved = solve_scenario_by_method(
                    scenario_dir,
                    scenario_data,
                    solve_method,
                    anneal_print=args.anneal_print,
                    time_limit=args.time_limit,
                    cbs_time_step=args.cbs_time_step,
                    cbs_horizon=args.cbs_horizon,
                    optimizer_config_source=args.optimizer_config_source,
                )
                print(f"[{solve_method.upper()}] Solved scenario: {scenario_dir}")
                print(f"[{solve_method.upper()}] Final cost: {solved['cost']}")
                print(
                    f"[{solve_method.upper()}] Saved solution to: "
                    f"{scenario_dir / output_filename_for_method(solve_method)}"
                )
            return

        for solve_method in solve_methods:
            print(f"[{solve_method.upper()}] Solve mode with time limit {args.time_limit}s")
            solved_dirs = solve_all_scenarios(
                args.scenario_root,
                solve_method,
                anneal_print=args.anneal_print,
                time_limit=args.time_limit,
                cbs_time_step=args.cbs_time_step,
                cbs_horizon=args.cbs_horizon,
                optimizer_config_source=args.optimizer_config_source,
            )
            print(
                f"[{solve_method.upper()}] Solved {len(solved_dirs)} scenarios "
                f"under {args.scenario_root}"
            )
            for d in solved_dirs:
                print(f"- {d.name}: {output_filename_for_method(solve_method)}")
        return

    if args.mode == "compare":
        methods = selected_methods or list(SUPPORTED_METHODS)
        comparison_path = compare_methods_across_scenarios(
            args.scenario_root,
            methods=methods,
            save_path=args.scenario_root / "method_comparison_metrics.png",
        )
        print(f"Saved cross-method comparison plot for {methods}: {comparison_path}")
        return

    plot_methods = selected_methods or [method]

    if args.scenario_path is not None:
        scenario_dir = Path(args.scenario_path)
        for plot_method in plot_methods:
            solution_path = scenario_dir / output_filename_for_method(plot_method)
            if not solution_path.exists():
                raise FileNotFoundError(f"No {solution_path.name} found for scenario: {scenario_dir}")

            with open(solution_path, "rb") as f:
                solution_data = pickle.load(f)

            plot_scenario_solution(scenario_dir, plot_method, solution_data)
            print(f"Plotted scenario with {plot_method}: {scenario_dir}")
        return

    for plot_method in plot_methods:
        plotted_dirs = plot_all_scenarios(args.scenario_root, plot_method)
        print(f"Plotted {len(plotted_dirs)} scenarios under {args.scenario_root} with {plot_method}")
    for d in plotted_dirs:
        print(f"- {d.name}: network_plot_{method}.png and schedule_plot_{method}.png")


if __name__ == "__main__":
    main()
