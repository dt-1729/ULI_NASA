from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import MIRS
import cbs_search
import comparison
import gb_opt
import mep_opt
import seq_opt
import utils
import visualize
from scenario_io import (
    METHOD_LABELS,
    SUPPORTED_METHODS,
    list_scenario_dirs,
    load_scenario_data,
    method_display_name as _method_display_name,
    output_filename_for_method,
)

DEFAULT_SCENARIO_ROOT = Path("local_scenarios")


def reconstruct_mirs_from_scenario(scenario_data: Dict[str, Any]) -> MIRS.MIRS:
    kwargs = dict(scenario_data["mirs_constructor_kwargs"])
    mirs = MIRS.MIRS(**kwargs)
    utils.apply_dist_mat_overrides(mirs, scenario_data.get("dist_mat_overrides"))
    return mirs


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
        "name": METHOD_LABELS[method],
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
    if method in {"cbf", "mep_cbf_pruned", "slsqp", "cbf_static", "slsqp_static"}:
        payload["cbf_mode"] = scenario_data["mirs_constructor_kwargs"]["ca_cbf"]

    payload.update(extra)
    return payload


def get_common_initial_conditions(scenario_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Initial (T0, V0) shared by every optimizer-based method for a scenario.

    Built with the static (lin_static) barrier so that all agent pairs start
    separated by at least the waypoint tolerance, which the static methods need
    for a feasible first QP step.
    """
    kwargs = dict(scenario_data["mirs_constructor_kwargs"])
    kwargs["ca_cbf"] = utils.get_cbf_mode("lin_static", np.asarray(scenario_data["tol_array"]))
    static_mirs = MIRS.MIRS(**kwargs)
    return (
        np.array(static_mirs.sched_mat, dtype=float),
        np.array(static_mirs.speed_vec, dtype=float),
    )


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
    T0, V0 = get_common_initial_conditions(scenario_data)
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
        active_waypoints=None,
        time_limit=time_limit,
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
    T0, V0 = get_common_initial_conditions(scenario_data)
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
        time_limit=time_limit,
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
    prune_mode: bool | None = None,
) -> Dict[str, Any]:
    method = method.lower()
    if method == "cbf":
        prune_mode = False
    elif method == "mep_cbf_pruned":
        prune_mode = True
    if prune_mode is not None:
        scenario_data["mirs_constructor_kwargs"]["prune_mode"] = prune_mode
    if method == "cbs":
        return cbs_search.solve_cbs_scenario(
            scenario_dir,
            scenario_data,
            time_step=cbs_time_step,
            horizon=cbs_horizon,
        )
    if method in {"cbf", "mep_cbf_pruned", "cbf_static"}:
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
    prune_mode: bool | None = None,
) -> List[Path]:
    scenario_dirs = list_scenario_dirs(root_dir)
    solved_paths: List[Path] = []

    print(f"[{_method_display_name(method)}] Starting solve for {len(scenario_dirs)} scenarios in {root_dir}")
    for idx, scenario_dir in enumerate(scenario_dirs, start=1):
        print(f"[{_method_display_name(method)}] Scenario {idx}/{len(scenario_dirs)}: {scenario_dir.name}")
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
            prune_mode=prune_mode,
        )
        print(
            f"[{_method_display_name(method)}] Finished {scenario_dir.name} | "
            f"cost={result.get('cost', 'N/A')} | runtime={result.get('runtime', 'N/A')}s"
        )
        solved_paths.append(scenario_dir)

    print(f"[{_method_display_name(method)}] Completed all solves for {len(solved_paths)} scenarios")
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

    if method in {"cbf", "mep_cbf_pruned", "slsqp", "cbf_static", "slsqp_static"}:
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
        figuresize=(14, 12),
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

    agent_schedule_path = scenario_dir / f"agent_schedule_plot_{method}.png"
    visualize.plot_agent_waypoint_schedules(
        agent_routes,
        agent_schedules,
        T_schedule,
        assoc_mat,
        process_T,
        tol_array,
        agent_colors,
        figuresize=(28, 12),
        marker_size=18,
        index_size=52,
        x_tick_size=44,
        y_tick_size=44,
        x_label_size=56,
        y_label_size=56,
        save_path=str(agent_schedule_path),
        show_plot=False,
    )

    agent_route_schedule_path = scenario_dir / f"agent_route_schedule_plot_{method}.png"
    visualize.plot_agent_route_schedule_table(
        agent_routes,
        T_schedule,
        agent_colors=agent_colors,
        save_path=str(agent_route_schedule_path),
        show_plot=False,
    )

    print(
        f"Saved plots to: {network_path}, {schedule_path}, {agent_schedule_path}, "
        f"and {agent_route_schedule_path}"
    )

    if method in {"cbf", "mep_cbf_pruned"} and "b_arr" in solution_data and "chi_arr" in solution_data:
        beta_assoc_path = scenario_dir / f"waypoint_association_vs_beta_{method}.png"
        visualize.plot_agent_waypoint_associations_vs_beta(
            b_arr=solution_data["b_arr"],
            chi_arr=solution_data["chi_arr"],
            save_path=str(beta_assoc_path),
            show_plot=False,
        )
        print(f"Saved beta-sweep association plot to: {beta_assoc_path}")

    if "b_arr" in solution_data and "T_array" in solution_data and "V_array" in solution_data:
        # Clean up legacy single-file plot if present
        legacy_path_prob_file = scenario_dir / f"path_probability_vs_beta_{method}.png"
        if legacy_path_prob_file.exists():
            legacy_path_prob_file.unlink()

        path_prob_dir = scenario_dir / f"path_probability_vs_beta_{method}"
        visualize.plot_path_probabilities_vs_beta(
            solution_data=solution_data,
            num_agents=len(solution_data["agent_routes"]),
            num_paths=6,
            save_dir=str(path_prob_dir),
            show_plot=False,
        )
        print(f"Saved beta-sweep path probability plots to folder: {path_prob_dir}")

        # gradient of F_beta at the terminal beta_max w.r.t. every existing edge length l_ij
        scenario_data = load_scenario_data(scenario_dir)
        mirs = reconstruct_mirs_from_scenario(scenario_data)
        beta_max = solution_data["b_arr"][-1]
        grad_mat = mirs.edgeGradientMatrix_l(
            solution_data["T_array"][-1],
            solution_data["V_array"][-1],
            beta_max,
        )
        edge_grad_path = scenario_dir / f"edge_gradient_heatmap_{method}.png"
        visualize.plot_edge_gradient_heatmap(
            figuresize=(20, 14),
            wp_xy=wp_locs,
            mask=mask,
            grad_mat=grad_mat,
            save_path=str(edge_grad_path),
            show_plot=False,
        )
        print(f"Saved edge gradient heat-map to: {edge_grad_path}")


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


def animation_schedules(
    routes: List[List[int]],
    fallback_schedules: List[List[float]],
    schedule_matrix: Any,
) -> List[List[float]]:
    """Convert a method's waypoint-time matrix into route-ordered schedules."""
    matrix = np.asarray(schedule_matrix)
    if matrix.ndim != 2 or matrix.shape[0] < len(routes):
        return fallback_schedules

    schedules: List[List[float]] = []
    for agent_id, route in enumerate(routes):
        try:
            arrival_times = [float(matrix[agent_id, int(waypoint)]) for waypoint in route]
        except (IndexError, TypeError, ValueError):
            return fallback_schedules
        schedules.append([0.0, *arrival_times])
    return schedules


def animate_scenario_solution(
    scenario_dir: Path,
    method: str,
    solution_data: Dict[str, Any],
    fps: int = 10,
    max_frames: int = 240,
) -> Path:
    wp_locs = np.asarray(solution_data["wp_xy"])
    mask = np.asarray(solution_data["mask"])
    agent_routes = solution_data["agent_routes"]
    fallback_schedules = solution_data["agent_schedules"]

    if method in {"cbf", "mep_cbf_pruned", "slsqp", "cbf_static", "slsqp_static"}:
        method_schedule = solution_data["T_array"][-1]
    elif method == "gurobi":
        method_schedule = solution_data.get("T_mat", solution_data.get("T_array"))
    elif method in {"cbs", "seq"}:
        method_schedule = solution_data["schedule_matrix"]
    else:
        raise ValueError(f"Unsupported method '{method}'")

    schedules = animation_schedules(
        agent_routes,
        fallback_schedules,
        method_schedule,
    )
    cmap = plt.get_cmap("tab20")
    agent_colors = {
        agent_id: cmap(agent_id / max(1, len(agent_routes)))
        for agent_id in range(len(agent_routes))
    }
    animation_path = scenario_dir / f"agent_animation_{method}.gif"
    visualize.animate_agent_routes(
        wp_xy=wp_locs,
        mask=mask,
        routes=agent_routes,
        schedules=schedules,
        agent_colors=agent_colors,
        save_path=animation_path,
        fps=fps,
        max_frames=max_frames,
    )
    return animation_path


def animate_all_scenarios(
    root_dir: Path,
    method: str,
    fps: int = 10,
    max_frames: int = 240,
) -> List[Path]:
    animation_paths: List[Path] = []
    for scenario_dir in list_scenario_dirs(root_dir):
        solution_path = scenario_dir / output_filename_for_method(method)
        if not solution_path.exists():
            print(f"Skipping {scenario_dir}: no {solution_path.name} found.")
            continue

        with open(solution_path, "rb") as f:
            solution_data = pickle.load(f)
        animation_path = animate_scenario_solution(
            scenario_dir,
            method,
            solution_data,
            fps=fps,
            max_frames=max_frames,
        )
        animation_paths.append(animation_path)
        print(f"Saved animation: {animation_path}")
    return animation_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve or plot MIRS scenarios using MEP-CBF, MEP-SLSQP, or Gurobi."
        )
    )
    parser.add_argument(
        "mode",
        choices=("solve", "plot", "compare", "anim"),
        default="solve",
        help="Execution mode. 'solve' computes a solution; 'plot' renders saved solution plots; 'anim' creates route animations; 'compare' saves comparison plots.",
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
        help="Enable detailed annealing printout for the MEP-CBF/MEP-SLSQP optimizer during solve mode.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Methods to run, plot, or compare. Use 'all' to select every supported method.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=comparison.COMPARISON_METRICS,
        default=None,
        metavar="METRIC",
        help="Metrics to include as subplot rows in compare mode. Defaults to all metrics.",
    )
    parser.add_argument(
        "--metric-figsize",
        nargs=2,
        type=float,
        default=(16.0, 3.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Width and height in inches for each metric subplot. Default: 16 3.",
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
        "--animation-fps",
        type=int,
        default=10,
        help="Frames per second for anim mode. Default: 10.",
    )
    parser.add_argument(
        "--animation-max-frames",
        type=int,
        default=240,
        help="Maximum frames generated per animation. Default: 240.",
    )
    parser.add_argument(
        "--optimizer-config-source",
        choices=("scenario", "utils"),
        default="scenario",
        help=(
            "Source for MEP-CBF/MEP-SLSQP optimizer specs: 'scenario' uses the saved "
            "scenario_data.pkl values; 'utils' rebuilds them from "
            "utils.set_mep_opt_config(). Default: scenario."
        ),
    )
    parser.add_argument(
        "--prune-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override saved prune_mode for methods without a fixed policy "
            "(--prune-mode / --no-prune-mode). MEP-CBF is always unpruned; "
            "mep_cbf_pruned is always pruned."
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
                print(f"[{_method_display_name(solve_method)}] Solving single scenario: {scenario_dir}")
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
                    prune_mode=args.prune_mode,
                )
                print(f"[{_method_display_name(solve_method)}] Solved scenario: {scenario_dir}")
                print(f"[{_method_display_name(solve_method)}] Final cost: {solved['cost']}")
                print(
                    f"[{_method_display_name(solve_method)}] Saved solution to: "
                    f"{scenario_dir / output_filename_for_method(solve_method)}"
                )
            return

        for solve_method in solve_methods:
            print(f"[{_method_display_name(solve_method)}] Solve mode with time limit {args.time_limit}s")
            solved_dirs = solve_all_scenarios(
                args.scenario_root,
                solve_method,
                anneal_print=args.anneal_print,
                time_limit=args.time_limit,
                cbs_time_step=args.cbs_time_step,
                cbs_horizon=args.cbs_horizon,
                optimizer_config_source=args.optimizer_config_source,
                prune_mode=args.prune_mode,
            )
            print(
                f"[{_method_display_name(solve_method)}] Solved {len(solved_dirs)} scenarios "
                f"under {args.scenario_root}"
            )
            for d in solved_dirs:
                print(f"- {d.name}: {output_filename_for_method(solve_method)}")
        return

    if args.mode == "compare":
        methods = selected_methods or list(SUPPORTED_METHODS)
        comparison_path = comparison.compare_methods_across_scenarios(
            args.scenario_root,
            methods=methods,
            save_path=args.scenario_root / "comparison_metric_plots" / "method_comparison_metrics.png",
            plot_metrics=args.metrics,
            metric_figsize=tuple(args.metric_figsize),
        )
        method_names = [_method_display_name(selected_method) for selected_method in methods]
        print(f"Saved cross-method comparison plot for {method_names}: {comparison_path}")
        return

    if args.mode == "anim":
        animation_methods = selected_methods or [method]
        if args.scenario_path is not None:
            scenario_dir = Path(args.scenario_path)
            for animation_method in animation_methods:
                solution_path = scenario_dir / output_filename_for_method(animation_method)
                if not solution_path.exists():
                    raise FileNotFoundError(
                        f"No {solution_path.name} found for scenario: {scenario_dir}"
                    )
                with open(solution_path, "rb") as f:
                    solution_data = pickle.load(f)
                animation_path = animate_scenario_solution(
                    scenario_dir,
                    animation_method,
                    solution_data,
                    fps=args.animation_fps,
                    max_frames=args.animation_max_frames,
                )
                print(f"Animated scenario with {_method_display_name(animation_method)}: {animation_path}")
            return

        for animation_method in animation_methods:
            animation_paths = animate_all_scenarios(
                args.scenario_root,
                animation_method,
                fps=args.animation_fps,
                max_frames=args.animation_max_frames,
            )
            print(
                f"Animated {len(animation_paths)} scenarios under "
                f"{args.scenario_root} with {_method_display_name(animation_method)}"
            )
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
            print(f"Plotted scenario with {_method_display_name(plot_method)}: {scenario_dir}")
        return

    for plot_method in plot_methods:
        plotted_dirs = plot_all_scenarios(args.scenario_root, plot_method)
        print(f"Plotted {len(plotted_dirs)} scenarios under {args.scenario_root} with {_method_display_name(plot_method)}")
    for d in plotted_dirs:
        print(f"- {d.name}: network_plot_{method}.png and schedule_plot_{method}.png")


if __name__ == "__main__":
    main()
