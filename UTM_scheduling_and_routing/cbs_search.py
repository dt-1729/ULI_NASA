from __future__ import annotations

import argparse
import heapq
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


Vertex = Tuple[int, int]
Constraint = Tuple[int, int, float, float]
ForbiddenWindow = Tuple[int, float, float]


def load_scenario(path: str | Path) -> Dict:
    scenario_path = Path(path)
    data_file = scenario_path / "scenario_data.pkl"
    if not data_file.exists():
        raise FileNotFoundError(f"Scenario data file not found: {data_file}")

    with open(data_file, "rb") as f:
        return pickle.load(f)


def build_adjacency(
    scenario_data: Dict, agent_index: int = 0
) -> Tuple[List[List[Tuple[int, float]]], np.ndarray, np.ndarray, np.ndarray]:
    summary = scenario_data.get("mirs_summary", {})
    dist_mat = np.asarray(
        summary.get(
            "dist_mat", np.zeros((scenario_data["n_waypoints"], scenario_data["n_waypoints"]))
        )
    )
    mask = np.asarray(summary.get("mask", np.ones_like(dist_mat, dtype=bool)))
    speed_lim_mat = np.asarray(
        summary.get("speed_lim_mat", np.zeros((scenario_data["n_agents"], 2)))
    )

    if speed_lim_mat.size == 0:
        speed_lim_mat = np.ones((scenario_data["n_agents"], 2))

    n_nodes = dist_mat.shape[0]
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]

    if agent_index < speed_lim_mat.shape[0]:
        max_speed = float(np.clip(np.max(speed_lim_mat[agent_index, :]), 1e-6, None))
    else:
        max_speed = float(np.clip(np.max(speed_lim_mat), 1e-6, None))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if not bool(mask[i, j] or mask[j, i]):
                continue
            edge_length = float(dist_mat[i, j])
            if not np.isfinite(edge_length) or edge_length <= 0:
                continue
            travel_time = max(edge_length / max_speed, 1e-6)
            adjacency[i].append((j, travel_time))
            adjacency[j].append((i, travel_time))

    return adjacency, dist_mat, mask, speed_lim_mat


def _agent_data(scenario_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
    summary = scenario_data.get("mirs_summary", {})
    if "sd_mat" in summary:
        sd_mat = np.asarray(summary["sd_mat"])
    elif "mirs_constructor_kwargs" in scenario_data and "sd_mat" in scenario_data["mirs_constructor_kwargs"]:
        sd_mat = np.asarray(scenario_data["mirs_constructor_kwargs"]["sd_mat"])
    else:
        raise KeyError("Scenario does not contain agent start/destination data under 'sd_mat'.")

    if "start_times" in summary:
        start_times = np.asarray(summary["start_times"], dtype=float)
    elif "initial_conditions" in scenario_data and "T0" in scenario_data["initial_conditions"]:
        start_times = np.asarray(scenario_data["initial_conditions"]["T0"], dtype=float)[:, 0]
    else:
        start_times = np.zeros(scenario_data["n_agents"], dtype=float)
    return sd_mat, start_times


def _default_horizon(
    scenario_data: Dict, dist_mat: np.ndarray, speed_lim_mat: np.ndarray, start_time: float
) -> float:
    """Return a horizon that covers masked-graph routes for every agent."""
    _, start_times = _agent_data(scenario_data)
    route_arrival_bounds: List[float] = []

    for agent_index in range(int(scenario_data["n_agents"])):
        sd_mat, _ = _agent_data(scenario_data)
        start_node = int(sd_mat[agent_index, 0])
        goal_node = int(sd_mat[agent_index, 1])
        adjacency, _, _, _ = build_adjacency(scenario_data, agent_index)

        distances = [float("inf")] * len(adjacency)
        distances[start_node] = 0.0
        queue: List[Tuple[float, int]] = [(0.0, start_node)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances[node]:
                continue
            if node == goal_node:
                break
            for neighbor, travel_time in adjacency[node]:
                next_distance = distance + travel_time
                if next_distance < distances[neighbor]:
                    distances[neighbor] = next_distance
                    heapq.heappush(queue, (next_distance, neighbor))

        if np.isfinite(distances[goal_node]):
            route_arrival_bounds.append(float(start_times[agent_index]) + distances[goal_node])

    if not route_arrival_bounds:
        raise RuntimeError("No masked-graph route exists for any agent in the scenario.")

    return max(route_arrival_bounds) + 100.0


def search_single_agent_on_time_expanded_graph(
    scenario_data: Dict,
    agent_index: int,
    time_step: float = 1.0,
    horizon: float | None = None,
    forbidden_nodes: Iterable[Vertex] | None = None,
    forbidden_windows: Iterable[ForbiddenWindow] | None = None,
) -> Tuple[List[int], List[float], float]:
    if agent_index < 0 or agent_index >= scenario_data["n_agents"]:
        raise ValueError(f"agent_index out of range: {agent_index} for {scenario_data['n_agents']} agents")
    if time_step <= 0:
        raise ValueError(f"time_step must be positive, got {time_step}")

    sd_mat, start_times = _agent_data(scenario_data)
    start_node = int(sd_mat[agent_index, 0])
    goal_node = int(sd_mat[agent_index, 1])
    start_time = float(start_times[agent_index])

    adjacency, dist_mat, _, speed_lim_mat = build_adjacency(scenario_data, agent_index)
    if horizon is None:
        horizon = _default_horizon(scenario_data, dist_mat, speed_lim_mat, start_time)

    forbidden = set(forbidden_nodes or ())
    windows = tuple(forbidden_windows or ())
    start_bucket = int(np.floor(start_time / time_step))
    start_state = (start_node, start_bucket)
    if start_state in forbidden:
        raise RuntimeError(f"Agent {agent_index} is constrained at its start node and time.")
    if any(
        node == start_node and abs(start_time - center) < radius
        for node, center, radius in windows
    ):
        raise RuntimeError(f"Agent {agent_index} is constrained at its start node and time.")

    pq: List[Tuple[float, int, int]] = []
    best_cost: Dict[Vertex, float] = {start_state: start_time}
    prev: Dict[Vertex, Vertex | None] = {start_state: None}
    heapq.heappush(pq, (start_time, start_node, start_bucket))

    while pq:
        arrival_time, node, bucket = heapq.heappop(pq)
        state = (node, bucket)
        if arrival_time > best_cost.get(state, float("inf")):
            continue

        if node == goal_node:
            route: List[int] = []
            schedule: List[float] = []
            cursor: Vertex | None = state
            while cursor is not None:
                node_idx, _ = cursor
                route.append(node_idx)
                schedule.append(best_cost[cursor])
                cursor = prev[cursor]
            route.reverse()
            schedule.reverse()
            return route, schedule, arrival_time - start_time

        transitions = list(adjacency[node])
        transitions.append((node, time_step))
        for neighbor, travel_time in transitions:
            next_time = arrival_time + travel_time
            if next_time > horizon:
                continue
            next_bucket = int(np.floor(next_time / time_step))
            next_state = (neighbor, next_bucket)
            if next_state in forbidden:
                continue
            if any(
                node == neighbor and abs(next_time - center) < radius
                for node, center, radius in windows
            ):
                continue
            if next_state in best_cost and next_time >= best_cost[next_state]:
                continue
            best_cost[next_state] = next_time
            prev[next_state] = state
            heapq.heappush(pq, (next_time, neighbor, next_bucket))

    raise RuntimeError(f"No feasible route found for agent {agent_index} from {start_node} to {goal_node}.")


@dataclass(frozen=True)
class _CBSNode:
    constraints: frozenset[Constraint]
    routes: Tuple[Tuple[int, ...], ...]
    schedules: Tuple[Tuple[float, ...], ...]
    cost: float


def _first_vertex_conflict(
    routes: Tuple[Tuple[int, ...], ...],
    schedules: Tuple[Tuple[float, ...], ...],
    tolerances: np.ndarray,
) -> Tuple[int, int, int, float, float, float] | None:
    occupied: Dict[int, List[Tuple[int, float, bool]]] = {}
    for agent_index, (route, schedule) in enumerate(zip(routes, schedules)):
        for path_index, (node, arrival_time) in enumerate(zip(route, schedule)):
            node = int(node)
            arrival_time = float(arrival_time)
            if node >= tolerances.size or tolerances[node] <= 0:
                continue
            radius = 0.9 * float(tolerances[node])
            for other_agent, other_time, other_is_initial in occupied.get(node, []):
                if other_agent == agent_index:
                    continue
                # Two agents already placed at the same start node are not
                # schedulable by routing; generated scenarios space these starts.
                if path_index == 0 and other_is_initial:
                    continue
                if abs(arrival_time - other_time) < radius:
                    return other_agent, agent_index, node, other_time, arrival_time, radius
            occupied.setdefault(node, []).append((agent_index, arrival_time, path_index == 0))
    return None


def search_multi_agent_cbs(
    scenario_data: Dict,
    time_step: float = 1.0,
    horizon: float | None = None,
) -> Tuple[List[List[int]], List[List[float]], float]:
    """Plan all agents with vertex-conflict CBS at the requested time resolution."""
    if time_step <= 0:
        raise ValueError(f"time_step must be positive, got {time_step}")

    n_agents = int(scenario_data["n_agents"])
    summary = scenario_data.get("mirs_summary", {})
    tolerances = np.asarray(
        summary.get("tol_array", scenario_data.get("tol_array", np.ones(scenario_data["n_waypoints"]))),
        dtype=float,
    ).reshape(-1)
    initial_routes: List[Tuple[int, ...]] = []
    initial_schedules: List[Tuple[float, ...]] = []
    total_cost = 0.0
    for agent_index in range(n_agents):
        route, schedule, cost = search_single_agent_on_time_expanded_graph(
            scenario_data, agent_index, time_step=time_step, horizon=horizon
        )
        initial_routes.append(tuple(route))
        initial_schedules.append(tuple(schedule))
        total_cost += cost

    root = _CBSNode(frozenset(), tuple(initial_routes), tuple(initial_schedules), total_cost)
    open_nodes: List[Tuple[float, int, _CBSNode]] = [(root.cost, 0, root)]
    serial = 1
    _, start_times = _agent_data(scenario_data)

    while open_nodes:
        _, _, node = heapq.heappop(open_nodes)
        conflict = _first_vertex_conflict(node.routes, node.schedules, tolerances)
        if conflict is None:
            return (
                [list(route) for route in node.routes],
                [list(schedule) for schedule in node.schedules],
                node.cost,
            )

        (
            first_agent,
            second_agent,
            conflict_node,
            first_arrival,
            second_arrival,
            conflict_radius,
        ) = conflict
        for constrained_agent, forbidden_center in (
            (first_agent, second_arrival),
            (second_agent, first_arrival),
        ):
            constraint = (constrained_agent, conflict_node, forbidden_center, conflict_radius)
            if constraint in node.constraints:
                continue
            child_constraints = node.constraints | {constraint}
            child_routes = list(node.routes)
            child_schedules = list(node.schedules)
            try:
                route, schedule, _ = search_single_agent_on_time_expanded_graph(
                    scenario_data,
                    constrained_agent,
                    time_step=time_step,
                    horizon=horizon,
                    forbidden_windows=(
                        (node_id, center, radius)
                        for agent_id, node_id, center, radius in child_constraints
                        if agent_id == constrained_agent
                    ),
                )
            except RuntimeError:
                continue

            child_routes[constrained_agent] = tuple(route)
            child_schedules[constrained_agent] = tuple(schedule)
            child_cost = sum(
                schedule[-1] - start_times[agent_id]
                for agent_id, schedule in enumerate(child_schedules)
            )
            child = _CBSNode(
                frozenset(child_constraints),
                tuple(child_routes),
                tuple(child_schedules),
                child_cost,
            )
            heapq.heappush(open_nodes, (child.cost, serial, child))
            serial += 1

    raise RuntimeError("No collision-free multi-agent route exists under the current horizon and time step.")


def _plot_schedule(route: List[int], schedule: List[float]) -> List[float]:
    return [schedule[0], *schedule]


def _waypoint_schedule_data(
    scenario_data: Dict, routes: List[List[int]], schedules: List[List[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    n_agents = int(scenario_data["n_agents"])
    n_waypoints = int(scenario_data["n_waypoints"])
    schedule_matrix = np.full((n_agents, n_waypoints), np.nan, dtype=float)
    association_matrix = np.zeros((n_agents, n_waypoints), dtype=float)

    for agent_index, (route, schedule) in enumerate(zip(routes, schedules)):
        for node, arrival_time in zip(route, schedule):
            association_matrix[agent_index, node] = 1.0
            if np.isnan(schedule_matrix[agent_index, node]):
                schedule_matrix[agent_index, node] = arrival_time

    return schedule_matrix, association_matrix


def build_cbs_solution_data(
    scenario_data: Dict,
    routes: List[List[int]],
    schedules: List[List[float]],
    cost: float,
    runtime: float,
    time_step: float,
    scenario_dir: Path | None = None,
) -> Dict[str, Any]:
    summary = scenario_data.get("mirs_summary", {})
    schedule_matrix, association_matrix = _waypoint_schedule_data(
        scenario_data, routes, schedules
    )
    n_waypoints = int(scenario_data["n_waypoints"])
    return {
        "name": "CBS",
        "method": "cbs",
        "n_agents": int(scenario_data["n_agents"]),
        "wp_xy": summary.get("wp_locations", scenario_data.get("wp_locations")),
        "wp_params": scenario_data.get("network_params"),
        "mask": summary.get("mask"),
        "dist_mat": summary.get("dist_mat"),
        "sd_mat": summary.get("sd_mat"),
        "cost": float(cost),
        "runtime": float(runtime),
        "agent_routes": routes,
        "agent_schedules": [_plot_schedule(route, schedule) for route, schedule in zip(routes, schedules)],
        "schedule_matrix": schedule_matrix,
        "association_mat": association_matrix,
        "processing_time": summary.get("process_T", np.zeros((n_waypoints, n_waypoints))),
        "cat": scenario_data.get("tol_array", np.ones(n_waypoints)),
        "cost_mode": scenario_data.get("cost_mode", "sum"),
        "seed": scenario_data.get("seed"),
        "speed_lim_mat": summary.get("speed_lim_mat"),
        "start_times": summary.get("start_times"),
        "scenario_id": scenario_data.get("scenario_id"),
        "scenario_dir": str(scenario_dir) if scenario_dir is not None else None,
        "time_step": float(time_step),
    }


def solve_cbs_scenario(
    scenario_dir: Path,
    scenario_data: Dict,
    time_step: float = 1.0,
    horizon: float | None = None,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    routes, schedules, cost = search_multi_agent_cbs(
        scenario_data, time_step=time_step, horizon=horizon
    )
    runtime = time.perf_counter() - start_time
    solution_data = build_cbs_solution_data(
        scenario_data,
        routes,
        schedules,
        cost=cost,
        runtime=runtime,
        time_step=time_step,
        scenario_dir=scenario_dir,
    )
    output_path = scenario_dir / "solution_cbs.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(solution_data, f)
    return solution_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Conflict-based multi-agent search for a saved scenario.")
    parser.add_argument("--scenario-path", type=str, required=True, help="Path to a scenario folder containing scenario_data.pkl")
    parser.add_argument("--agent-index", type=int, default=None, help="Plan only this agent instead of running CBS")
    parser.add_argument("--time-step", type=float, default=1.0, help="Time resolution used for vertex conflicts")
    parser.add_argument("--horizon", type=float, default=None, help="Optional search horizon in time units")
    args = parser.parse_args()

    scenario_data = load_scenario(args.scenario_path)
    if args.agent_index is None:
        solution_data = solve_cbs_scenario(
            Path(args.scenario_path),
            scenario_data,
            time_step=args.time_step,
            horizon=args.horizon,
        )
        print(f"routes={solution_data['agent_routes']}")
        print(f"schedules={solution_data['agent_schedules']}")
        print(f"total_cost={solution_data['cost']}")
        print(f"runtime={solution_data['runtime']}")
        print(f"saved_solution={Path(args.scenario_path) / 'solution_cbs.pkl'}")
    else:
        route, schedule, total_time = search_single_agent_on_time_expanded_graph(
            scenario_data,
            agent_index=args.agent_index,
            time_step=args.time_step,
            horizon=args.horizon,
        )
        print(f"agent_index={args.agent_index}")
        print(f"route={route}")
        print(f"schedule={schedule}")
        print(f"arrival_time={total_time}")


if __name__ == "__main__":
    main()