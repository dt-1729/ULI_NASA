from __future__ import annotations

import json
import pickle
import tempfile
import unittest
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, ScalarFormatter
import numpy as np

from scenario_io import (
    METHOD_LABELS,
    SUPPORTED_METHODS,
    list_scenario_dirs,
    load_scenario_data,
    method_display_name as _method_display_name,
    output_filename_for_method,
)


COMPARISON_METRICS = (
    "makespan",
    "pct_cost_reduction",
    "normalized_mean_cost_performance",
    "runtime",
    "conflict_rate",
    "mean_conflict_deficit",
    "temporal_constraint_violation",
)
BETA_RUNTIME_METHODS = ("cbf", "mep_cbf_pruned", "slsqp")


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


def _temporal_constraint_violation(
    scenario_data: Dict[str, Any],
    solution: Dict[str, Any],
    routes: List[List[int]],
    arrival_times: List[np.ndarray],
) -> float:
    """Return the largest agent-summed normalized edge travel-time violation."""
    summary = scenario_data.get("mirs_summary", {})
    dist_mat_data = solution.get("dist_mat")
    if dist_mat_data is None:
        dist_mat_data = summary.get("dist_mat")
    speed_lim_data = solution.get("speed_lim_mat")
    if speed_lim_data is None:
        speed_lim_data = summary.get("speed_lim_mat")
    if dist_mat_data is None:
        raise KeyError("Scenario and solution data do not contain dist_mat")
    if speed_lim_data is None:
        raise KeyError("Scenario and solution data do not contain speed_lim_mat")

    dist_mat = np.asarray(dist_mat_data, dtype=float)
    speed_lim_mat = np.asarray(speed_lim_data, dtype=float)
    if dist_mat.ndim != 2 or dist_mat.shape[0] != dist_mat.shape[1]:
        raise ValueError("Distance matrix must be square")
    if speed_lim_mat.shape != (len(routes), 2):
        raise ValueError(
            "Speed limits must contain [min_speed, max_speed] for each agent"
        )
    if len(arrival_times) != len(routes):
        raise ValueError("Each agent route must have a corresponding schedule")

    agent_violations = []
    for agent_index, (route, times) in enumerate(zip(routes, arrival_times)):
        if len(times) != len(route):
            raise ValueError(
                f"Agent {agent_index} has {len(route)} waypoints but "
                f"{len(times)} arrival times"
            )
        min_speed, max_speed = speed_lim_mat[agent_index]
        if (
            not np.isfinite(min_speed)
            or not np.isfinite(max_speed)
            or min_speed <= 0
            or max_speed < min_speed
        ):
            raise ValueError(f"Agent {agent_index} has invalid speed limits")

        total_violation = 0.0
        for edge_index, (source, destination) in enumerate(zip(route, route[1:])):
            source = int(source)
            destination = int(destination)
            if source == destination:
                continue
            if (
                source < 0
                or destination < 0
                or source >= dist_mat.shape[0]
                or destination >= dist_mat.shape[1]
            ):
                raise ValueError(
                    f"Agent {agent_index} route has an out-of-range edge "
                    f"{source} -> {destination}"
                )

            distance = float(dist_mat[source, destination])
            if not np.isfinite(distance) or distance <= 0:
                raise ValueError(
                    f"Agent {agent_index} route has an invalid edge "
                    f"{source} -> {destination}"
                )
            transition_time = float(times[edge_index + 1] - times[edge_index])
            if not np.isfinite(transition_time):
                raise ValueError(
                    f"Agent {agent_index} has a non-finite transition time "
                    f"on edge {source} -> {destination}"
                )

            min_transition_time = distance / max_speed
            max_transition_time = distance / min_speed
            allowed_time_width = max_transition_time - min_transition_time
            if allowed_time_width <= 0:
                raise ValueError(
                    f"Agent {agent_index} has no nonzero allowed time range "
                    f"on edge {source} -> {destination}"
                )

            lower_violation = min(0.0, transition_time - min_transition_time)
            upper_violation = max(0.0, transition_time - max_transition_time)
            total_violation += (
                float(np.hypot(lower_violation, upper_violation))
                / allowed_time_width
            )
        agent_violations.append(total_violation)

    return float(max(agent_violations, default=0.0))


def _scenario_edge_count(scenario_data: Dict[str, Any]) -> int:
    summary = scenario_data.get("mirs_summary", {})
    mask = summary.get("mask", scenario_data.get("mask"))
    if mask is None:
        raise KeyError("Scenario data does not contain a graph mask")

    adjacency = np.asarray(mask)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Scenario graph mask must be a square matrix")
    return int(np.count_nonzero(np.triu(adjacency != 0, k=1)))


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
    comparable_pair_count = 0
    conflict_deficits = []
    arrivals_by_waypoint = defaultdict(list)
    for route, agent_arrivals in zip(routes, arrival_times):
        for node, arrival_time in zip(route, agent_arrivals):
            if np.isfinite(arrival_time):
                arrivals_by_waypoint[int(node)].append(float(arrival_time))

    for node, arrivals in arrivals_by_waypoint.items():
        if node >= tolerances.size or tolerances[node] <= 0:
            continue

        sorted_arrivals = sorted(arrivals)
        threshold = 0.9 * tolerances[node]
        for first_arrival, second_arrival in zip(
            sorted_arrivals[:-1], sorted_arrivals[1:]
        ):
            gap = second_arrival - first_arrival
            comparable_pair_count += 1
            if gap < threshold:
                conflict_count += 1
                conflict_deficits.append((threshold - gap) / threshold)

    conflict_rate = (
        float(conflict_count / comparable_pair_count)
        if comparable_pair_count
        else np.nan
    )
    conflict_free = float(conflict_count == 0) if comparable_pair_count else np.nan
    mean_conflict_deficit = (
        float(np.sum(conflict_deficits) / comparable_pair_count)
        if comparable_pair_count
        else 0.0
    )
    max_conflict_deficit = (
        float(np.max(conflict_deficits)) if conflict_deficits else 0.0
    )

    return {
        "makespan": makespan,
        "sum_time": sum_travel_time,
        "runtime": float(solution.get("runtime", np.nan)),
        "conflicts": float(conflict_count),
        "comparable_pairs": float(comparable_pair_count),
        "conflict_rate": conflict_rate,
        "conflict_free": conflict_free,
        "mean_conflict_deficit": mean_conflict_deficit,
        "max_conflict_deficit": max_conflict_deficit,
        "temporal_constraint_violation": _temporal_constraint_violation(
            scenario_data, solution, routes, arrival_times
        ),
    }


def _beta_runtime_summary(
    grouped_records: Dict[tuple[str, int, int], List[tuple[Path, Dict[str, Any], Dict[str, Dict[str, Any]]]]],
    group_keys: List[tuple[str, int, int]],
    methods: List[str],
) -> Dict[str, Any] | None:
    """Aggregate elapsed time at common first, middle and last scheduled betas."""
    beta_methods = [method for method in BETA_RUNTIME_METHODS if method in methods]
    reference_betas = None
    valid_records = []
    missing_count = 0
    incompatible_count = 0
    for group_index, group_key in enumerate(group_keys):
        for scenario_dir, _, solutions in grouped_records[group_key]:
            for method in beta_methods:
                if method not in solutions:
                    continue
                solution = solutions[method]
                if "b_arr" not in solution or "compute_time_per_beta" not in solution:
                    missing_count += 1
                    continue
                betas = np.asarray(solution["b_arr"], dtype=float)
                times = np.asarray(solution["compute_time_per_beta"], dtype=float)
                if (
                    betas.ndim != 1
                    or betas.size < 3
                    or not np.all(np.isfinite(betas))
                    or np.any(betas <= 0)
                    or np.any(np.diff(betas) <= 0)
                    or times.ndim != 1
                    or times.size > betas.size
                    or not np.all(np.isfinite(times))
                    or np.any(times < 0)
                ):
                    raise ValueError(f"Invalid beta schedule or timing data in {scenario_dir} ({method})")
                indices = (0, (len(betas) - 1) // 2, len(betas) - 1)
                selected_betas = betas[list(indices)]
                if reference_betas is None:
                    reference_betas = selected_betas
                elif not np.allclose(selected_betas, reference_betas, rtol=1e-8, atol=0):
                    incompatible_count += 1
                    continue
                valid_records.append((group_index, method, indices, times))

    if missing_count:
        warnings.warn(f"{missing_count} solution(s) have no per-beta timing data and were excluded.")
    if incompatible_count:
        warnings.warn(
            f"{incompatible_count} solution(s) have incompatible first/middle/last beta values "
            "and were excluded from the beta-runtime comparison."
        )
    if reference_betas is None:
        if beta_methods:
            warnings.warn("No per-beta timing data available; beta-runtime comparison plot was not created.")
        return None

    samples = {
        method: [[[] for _ in group_keys] for _ in range(3)]
        for method in beta_methods
    }
    for group_index, method, indices, times in valid_records:
        for beta_index, schedule_index in enumerate(indices):
            if schedule_index < len(times):
                samples[method][beta_index][group_index].append(float(times[schedule_index]))

    for method in beta_methods:
        if not any(values for per_group in samples[method] for values in per_group):
            warnings.warn(f"No selected beta-stage timings available for {_method_display_name(method)}.")
    if not any(
        values
        for method in beta_methods
        for per_group in samples[method]
        for values in per_group
    ):
        warnings.warn("No selected beta-stage timings available; beta-runtime comparison plot was not created.")
        return None

    return {
        "beta_values": reference_betas.tolist(),
        "timing": "per-beta elapsed wall time in seconds, including beta-stage setup; convergence is not verified",
        "methods": {
            method: [
                {
                    "mean": [float(np.mean(values)) if values else None for values in per_group],
                    "count": [len(values) for values in per_group],
                }
                for per_group in samples[method]
            ]
            for method in beta_methods
        },
    }


def _plot_beta_runtime_comparison(
    summary: Dict[str, Any],
    x_values: np.ndarray,
    output_path: Path,
    method_colors: Dict[str, str],
    method_markers: Dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for beta_index, (axis, beta) in enumerate(zip(axes, summary["beta_values"])):
        for method, beta_data in summary["methods"].items():
            axis.plot(
                x_values,
                beta_data[beta_index]["mean"],
                marker=method_markers[method],
                color=method_colors[method],
                linestyle="--" if method == "slsqp" else "-",
                label=METHOD_LABELS[method],
            )
        axis.set_xscale("log")
        axis.set_title(rf"$\beta = {beta:.3g}$")
        axis.set_xlabel("Problem Size")
        axis.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Mean per-beta elapsed time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def compare_methods_across_scenarios(
    root_dir: Path,
    methods: List[str] | None = None,
    save_path: Path | None = None,
    plot_metrics: List[str] | None = None,
    metric_figsize: tuple[float, float] = (16.0, 3.0),
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
                continue
            with open(solution_path, "rb") as f:
                method_solutions[method] = pickle.load(f)
        if method_solutions:
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
    mean_problem_sizes = {
        group_key: float(
            np.mean(
                [
                    int(scenario_data["n_agents"])
                    * _scenario_edge_count(scenario_data)
                    * int(scenario_data["n_waypoints"])
                    for _, scenario_data, _ in grouped_records[group_key]
                ]
            )
        )
        for group_key in group_keys
    }
    group_keys.sort(key=mean_problem_sizes.__getitem__)
    x_values = np.asarray(
        [mean_problem_sizes[group_key] for group_key in group_keys], dtype=float
    )
    method_labels = METHOD_LABELS
    method_colors = {
        "cbf": "tab:orange",
        "mep_cbf_pruned": "tab:olive",
        "slsqp": "tab:blue",
        "cbf_static": "tab:cyan",
        "slsqp_static": "tab:brown",
        "gurobi": "tab:green",
        "cbs": "tab:red",
        "seq": "tab:purple",
    }
    method_markers = {
        "cbf": "o",
        "mep_cbf_pruned": "h",
        "slsqp": "s",
        "cbf_static": "^",
        "slsqp_static": "D",
        "gurobi": "v",
        "cbs": "P",
        "seq": "X",
    }

    metric_names = COMPARISON_METRICS
    summary_metric_names = metric_names + (
        "conflicts",
        "comparable_pairs",
        "conflict_free",
        "max_conflict_deficit",
    )
    plotting_values = {
        method: {
            metric: {"mean": [], "count": []}
            for metric in summary_metric_names
        }
        for method in methods
    }
    for group_key in group_keys:
        records = grouped_records[group_key]
        for method in methods:
            metrics_by_name = [
                _solution_comparison_metrics(data, solutions[method])
                for _, data, solutions in records
                if method in solutions
            ]
            for metric in summary_metric_names:
                baseline_method = (
                    "cbf"
                    if metric in {"runtime", "normalized_mean_cost_performance"}
                    else "seq"
                )
                if metric in {
                    "runtime",
                    "pct_cost_reduction",
                    "normalized_mean_cost_performance",
                } and baseline_method in methods:
                    ratio_values = []
                    for _, data, solutions in records:
                        if method not in solutions or baseline_method not in solutions:
                            continue
                        source_metric = (
                            "sum_time"
                            if metric
                            in {"pct_cost_reduction", "normalized_mean_cost_performance"}
                            else metric
                        )
                        method_value = _solution_comparison_metrics(
                            data, solutions[method]
                        )[source_metric]
                        baseline_value = _solution_comparison_metrics(
                            data, solutions[baseline_method]
                        )[source_metric]
                        if (
                            np.isfinite(method_value)
                            and np.isfinite(baseline_value)
                            and baseline_value != 0
                        ):
                            if metric == "pct_cost_reduction":
                                ratio_values.append(
                                    100.0
                                    * (baseline_value - method_value)
                                    / baseline_value
                                )
                            elif metric == "normalized_mean_cost_performance":
                                ratio_values.append(
                                    100.0 * method_value / baseline_value
                                )
                            else:
                                ratio_values.append(method_value / baseline_value)
                    values = np.asarray(ratio_values, dtype=float)
                    solved_count = int(values.size)
                elif metric in {
                    "runtime",
                    "pct_cost_reduction",
                    "normalized_mean_cost_performance",
                }:
                    values = np.asarray([], dtype=float)
                    solved_count = 0
                else:
                    values = np.asarray([metrics[metric] for metrics in metrics_by_name], dtype=float)
                    solved_count = sum(
                        np.isfinite(metrics["runtime"])
                        for metrics in metrics_by_name
                    )
                finite_values = values[np.isfinite(values)]
                plotting_values[method][metric]["mean"].append(
                    float(np.mean(finite_values))
                    if finite_values.size
                    else np.nan
                )
                plotting_values[method][metric]["count"].append(int(solved_count))

    output_dir = root_dir / "comparison_metric_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    beta_runtime_summary = _beta_runtime_summary(grouped_records, group_keys, methods)
    summary_path = output_dir / "method_comparison_metrics_summary.json"
    summary = {
        "groups": [
            {
                "network_type": network_type,
                "n_waypoints": n_waypoints,
                "n_agents": n_agents,
                "n_scenarios": len(grouped_records[(network_type, n_waypoints, n_agents)]),
                "mean_n_agents_edges_waypoints": mean_problem_sizes[
                    (network_type, n_waypoints, n_agents)
                ],
            }
            for network_type, n_waypoints, n_agents in group_keys
        ],
        "methods": plotting_values,
        "beta_runtime": beta_runtime_summary,
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    metric_labels = {
        "makespan": "Makespan",
        "pct_cost_reduction": "% Saved vs SEQ-SPS",
        "normalized_mean_cost_performance": "Relative Cost (MEP-CBF=100)",
        "runtime": "Relative Runtime (MEP-CBF=1)",
        "conflict_rate": "Conflict rate",
        "mean_conflict_deficit": "Conflict deficit",
        "temporal_constraint_violation": "Temporal constraint violation",
    }
    plot_metrics = metric_names if plot_metrics is None else plot_metrics
    invalid_metrics = [metric for metric in plot_metrics if metric not in metric_names]
    if invalid_metrics:
        raise ValueError(
            f"Unsupported comparison metrics: {invalid_metrics}. "
            f"Supported metrics: {metric_names}"
        )

    plt.rcParams["font.family"] = "Times New Roman"
    n_columns = min(4, len(methods))
    legend_rows = int(np.ceil(len(methods) / n_columns))
    legend_height = 0.6 * legend_rows + 0.5
    plot_path = save_path or output_dir / "method_comparison_metrics.png"

    def save_metric_figure(horizontal: bool, figure_path: Path) -> None:
        if horizontal:
            nrows, ncols = 1, len(plot_metrics)
            figure_width = metric_figsize[0] * len(plot_metrics)
            plot_height = metric_figsize[1]
        else:
            nrows, ncols = len(plot_metrics), 1
            figure_width = metric_figsize[0]
            plot_height = metric_figsize[1] * len(plot_metrics)

        figure_height = plot_height + legend_height
        fig, axes_grid = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figure_width, figure_height),
            sharex=not horizontal,
            squeeze=False,
        )
        axes = axes_grid.ravel()
        for metric, axis in zip(plot_metrics, axes):
            axis.set_xscale("log")
            for method in methods:
                means = np.asarray(plotting_values[method][metric]["mean"], dtype=float)
                axis.plot(
                    x_values,
                    means,
                    marker=method_markers[method],
                    linestyle={"slsqp": "--", "seq": "--"}.get(method, "-"),
                    linewidth=4.0,
                    markersize=18,
                    label=method_labels[method],
                    color=method_colors[method],
                )

            axis.tick_params(axis="both", which="major", labelsize=40)
            axis.tick_params(axis="both", which="minor", labelsize=40)
            axis.text(
                0.01,
                0.98,
                f"Mean {metric_labels[metric]}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=36,
                fontname="Times New Roman",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
            )
            metric_values = np.concatenate(
                [
                    np.asarray(plotting_values[method][metric][statistic], dtype=float)
                    for method in methods
                    for statistic in ("mean",)
                ]
            )
            finite_values = metric_values[np.isfinite(metric_values)]

            if metric in {"conflict_rate", "mean_conflict_deficit"}:
                axis.set_ylim(-0.1, 1)
            if metric == "makespan":
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((0, 0))
                axis.yaxis.set_major_formatter(formatter)
                axis.yaxis.get_offset_text().set_fontsize(40)
                axis.yaxis.get_offset_text().set_fontname("Times New Roman")
            else:
                axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.grid(True, linestyle="--", alpha=0.4)

            axis.xaxis.set_major_locator(LogLocator(base=10.0))
            axis.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

        if horizontal:
            fig.supxlabel("Problem Size", fontsize=40, fontname="Times New Roman")
            layout_rect = (0, 0.1, 1, 1 - legend_height / figure_height)
        else:
            axes[-1].set_xlabel(
                "Problem Size",
                fontsize=40,
                fontname="Times New Roman",
            )
            layout_rect = (0, 0, 1, 1 - legend_height / figure_height)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=n_columns,
            fontsize=40,
            frameon=False,
        )
        fig.tight_layout(rect=layout_rect)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=400, bbox_inches="tight")
        plt.close(fig)

    save_metric_figure(horizontal=False, figure_path=plot_path)
    horizontal_plot_path = plot_path.with_name(
        f"{plot_path.stem}_horizontal{plot_path.suffix}"
    )
    save_metric_figure(horizontal=True, figure_path=horizontal_plot_path)

    if beta_runtime_summary is not None:
        beta_plot_path = output_dir / "beta_runtime_comparison.png"
        _plot_beta_runtime_comparison(
            beta_runtime_summary, x_values, beta_plot_path, method_colors, method_markers
        )
        print(f"Saved per-beta elapsed-time comparison plot: {beta_plot_path}")

    return plot_path


class BetaRuntimeTests(unittest.TestCase):
    def test_summary_uses_elapsed_times_and_excludes_unreached_betas(self):
        keys = [("grid", 3, 1), ("grid", 4, 1)]
        betas = np.array([1e-6, 1e-5, 1e-3, 1e-1, 1.0])
        record = lambda times: {"b_arr": betas, "compute_time_per_beta": times}
        groups = {
            keys[0]: [
                (Path("a"), {}, {"cbf": record([1, 2, 3, 4, 5]), "slsqp": record([8, 7, 6])}),
                (Path("b"), {}, {"cbf": record([3, 4, 5, 6, 7]), "mep_cbf_pruned": record([2, 1, 4, 1, 6])}),
            ],
            keys[1]: [(Path("c"), {}, {"cbf": record([10, 9, 8, 7, 6])})],
        }
        result = _beta_runtime_summary(groups, keys, list(BETA_RUNTIME_METHODS))
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["beta_values"], [1e-6, 1e-3, 1.0])
        self.assertEqual(result["methods"]["cbf"][0], {"mean": [2.0, 10.0], "count": [2, 1]})
        self.assertEqual(result["methods"]["cbf"][1], {"mean": [4.0, 8.0], "count": [2, 1]})
        self.assertEqual(result["methods"]["slsqp"][2], {"mean": [None, None], "count": [0, 0]})
        self.assertEqual(result["methods"]["mep_cbf_pruned"][2], {"mean": [6.0, None], "count": [1, 0]})

    def test_missing_and_incompatible_schedules_are_reported(self):
        key = ("grid", 3, 1)
        groups = {
            key: [
                (Path("a"), {}, {"cbf": {"b_arr": [1, 2, 3], "compute_time_per_beta": [1, 2, 3]}}),
                (Path("b"), {}, {"slsqp": {"b_arr": [1, 2, 4], "compute_time_per_beta": [4, 5, 6]}}),
                (Path("c"), {}, {"mep_cbf_pruned": {}}),
            ]
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _beta_runtime_summary(groups, [key], list(BETA_RUNTIME_METHODS))
        messages = [str(warning.message) for warning in caught]
        self.assertTrue(any("no per-beta timing data" in message for message in messages))
        self.assertTrue(any("incompatible" in message for message in messages))
        self.assertEqual(result["methods"]["slsqp"][0]["count"], [0])
        self.assertEqual(result["methods"]["mep_cbf_pruned"][0]["count"], [0])

    def test_beta_plot_is_rendered(self):
        summary = {
            "beta_values": [1e-6, 1e-3, 1.0],
            "methods": {
                method: [{"mean": [1.0], "count": [1]} for _ in range(3)]
                for method in BETA_RUNTIME_METHODS
            },
        }
        colors = {"cbf": "tab:orange", "mep_cbf_pruned": "tab:olive", "slsqp": "tab:blue"}
        markers = {"cbf": "o", "mep_cbf_pruned": "h", "slsqp": "s"}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "beta.png"
            _plot_beta_runtime_comparison(summary, np.array([3.0]), path, colors, markers)
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_compare_writes_beta_summary_and_three_panel_plot(self):
        from main import main as main_cli
        from visualize import report_comparison_tables

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenario = root / "scenario"
            scenario.mkdir()
            dist = np.array([[0.0, 2.0], [2.0, 0.0]])
            data = {
                "n_agents": 1,
                "n_waypoints": 2,
                "network_type": "grid",
                "mirs_summary": {
                    "mask": np.array([[0, 1], [1, 0]]),
                    "dist_mat": dist,
                    "speed_lim_mat": np.array([[1.0, 2.0]]),
                    "tol_array": np.ones(2),
                    "start_times": np.zeros(1),
                },
            }
            solution = {
                "agent_routes": [[0, 1]],
                "agent_schedules": [[0, 0, 1]],
                "runtime": 1.0,
                "b_arr": [1e-6, 1e-3, 1.0],
                "compute_time_per_beta": [1.0, 2.0, 3.0],
            }
            with (scenario / "scenario_data.pkl").open("wb") as file:
                pickle.dump(data, file)
            for method in BETA_RUNTIME_METHODS:
                with (scenario / output_filename_for_method(method)).open("wb") as file:
                    pickle.dump(solution, file)

            saved_plots = []

            def save_figure(fig, path, **kwargs):
                saved_plots.append((
                    Path(path).name,
                    [axis.get_title() for axis in fig.axes],
                    [[line.get_label() for line in axis.lines] for axis in fig.axes],
                ))

            argv = [
                "main.py", "compare", "--scenario-root", str(root),
                "--methods", *BETA_RUNTIME_METHODS,
                "--metrics", "temporal_constraint_violation",
                "--metric-figsize", "8", "3",
            ]
            with patch("matplotlib.figure.Figure.savefig", save_figure), patch("sys.argv", argv):
                main_cli()
            summary_path = root / "comparison_metric_plots" / "method_comparison_metrics_summary.json"
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["beta_runtime"]["methods"]["cbf"][1]["mean"], [2.0])
            beta_plots = [(titles, labels) for name, titles, labels in saved_plots if name == "beta_runtime_comparison.png"]
            self.assertEqual(len(beta_plots), 1)
            self.assertEqual(len(beta_plots[0][0]), 3)
            self.assertIn("0.001", beta_plots[0][0][1])
            for labels in beta_plots[0][1]:
                self.assertEqual(labels, ["MEP-CBF", "MEP-CBF-P", "MEP-SLSQP"])

            table = report_comparison_tables(root, minimum_solved=1)
            self.assertIn("MEP-CBF-P", table)
            self.assertIn("temporal_constraint_violation", table)

if __name__ == "__main__":
    unittest.main()
