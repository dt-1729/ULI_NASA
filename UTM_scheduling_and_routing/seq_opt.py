from __future__ import annotations

import argparse
import heapq
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _scenario_arrays(scenario_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return the graph mask, edge distances, and agent start/destination pairs."""
	summary = scenario_data.get("mirs_summary", {})
	try:
		mask = np.asarray(summary["mask"])
		dist_mat = np.asarray(summary["dist_mat"], dtype=float)
		sd_mat = np.asarray(summary["sd_mat"], dtype=int)
	except KeyError as exc:
		raise KeyError(f"Scenario is missing required field: {exc.args[0]}") from exc

	if mask.shape != dist_mat.shape or mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
		raise ValueError("Scenario mask and dist_mat must be square matrices of the same shape")
	if sd_mat.ndim != 2 or sd_mat.shape[1] != 2:
		raise ValueError("Scenario sd_mat must have shape (n_agents, 2)")
	return mask, dist_mat, sd_mat


def shortest_path_routes(scenario_data: Dict[str, Any]) -> List[List[int]]:
	"""Find each agent's shortest waypoint route on the scenario graph.

	Edge costs are the Euclidean distances in ``mirs_summary['dist_mat']`` and
	only edges enabled by ``mirs_summary['mask']`` are traversable.
	"""
	mask, dist_mat, sd_mat = _scenario_arrays(scenario_data)
	n_waypoints = dist_mat.shape[0]
	adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n_waypoints)]

	for source in range(n_waypoints):
		for target in range(source + 1, n_waypoints):
			if not bool(mask[source, target] or mask[target, source]):
				continue
			edge_distance = float(dist_mat[source, target])
			if np.isfinite(edge_distance) and edge_distance > 0:
				adjacency[source].append((target, edge_distance))
				adjacency[target].append((source, edge_distance))

	routes: List[List[int]] = []
	for agent_index, (start, destination) in enumerate(sd_mat):
		start = int(start)
		destination = int(destination)
		if not 0 <= start < n_waypoints or not 0 <= destination < n_waypoints:
			raise ValueError(
				f"Agent {agent_index} has waypoint outside graph: {start}, {destination}"
			)

		distances = [float("inf")] * n_waypoints
		previous: List[int | None] = [None] * n_waypoints
		distances[start] = 0.0
		queue: List[Tuple[float, int]] = [(0.0, start)]

		while queue:
			distance, node = heapq.heappop(queue)
			if distance > distances[node]:
				continue
			if node == destination:
				break
			for neighbor, edge_distance in adjacency[node]:
				next_distance = distance + edge_distance
				if next_distance < distances[neighbor]:
					distances[neighbor] = next_distance
					previous[neighbor] = node
					heapq.heappush(queue, (next_distance, neighbor))

		if not np.isfinite(distances[destination]):
			raise ValueError(
				f"No graph route exists for agent {agent_index}: {start} -> {destination}"
			)

		route: List[int] = []
		node: int | None = destination
		while node is not None:
			route.append(node)
			node = previous[node]
		routes.append(list(reversed(route)))

	return routes


def _scenario_schedule_data(
	scenario_data: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return agent start times, speed limits, and waypoint tolerances."""
	summary = scenario_data.get("mirs_summary", {})
	n_agents = int(scenario_data["n_agents"])
	try:
		start_times = np.asarray(summary["start_times"], dtype=float).reshape(-1)
		speed_lim_mat = np.asarray(summary["speed_lim_mat"], dtype=float)
	except KeyError as exc:
		raise KeyError(f"Scenario is missing required field: {exc.args[0]}") from exc

	tolerances = np.asarray(
		summary.get("tol_array", scenario_data.get("tol_array")), dtype=float
	).reshape(-1)
	if tolerances.size == 0:
		tolerances = np.zeros(int(scenario_data["n_waypoints"]), dtype=float)
	if start_times.size != n_agents or speed_lim_mat.shape != (n_agents, 2):
		raise ValueError("Scenario start_times and speed_lim_mat do not match n_agents")
	if tolerances.size != int(scenario_data["n_waypoints"]):
		raise ValueError("Scenario tol_array does not match n_waypoints")
	return start_times, speed_lim_mat, tolerances


def _agent_priority_order(scenario_data: Dict[str, Any]) -> List[int]:
	"""Return agents in descending weight order, or their original order."""
	summary = scenario_data.get("mirs_summary", {})
	weights = summary.get("agent_weights", scenario_data.get("agent_weights"))
	if weights is None:
		return list(range(int(scenario_data["n_agents"])))
	weights = np.asarray(weights, dtype=float).reshape(-1)
	if weights.size != int(scenario_data["n_agents"]):
		raise ValueError("Scenario agent_weights does not match n_agents")
	return np.argsort(-weights, kind="stable").astype(int).tolist()


def assign_prioritized_schedules(
	scenario_data: Dict[str, Any], routes: List[List[int]] | None = None
) -> List[List[float]]:
	"""Assign conflict-free arrival schedules to routes in priority order.

	Each segment starts at the previous route waypoint and uses a speed in the
	agent's configured ``[min_speed, max_speed]`` interval. A later agent is
	delayed at a conflicting waypoint to at least the earlier agent's arrival
	time plus that waypoint's tolerance.
"""
	if routes is None:
		routes = shortest_path_routes(scenario_data)
	if len(routes) != int(scenario_data["n_agents"]):
		raise ValueError("The number of routes must match n_agents")

	_, dist_mat, _ = _scenario_arrays(scenario_data)
	start_times, speed_lim_mat, tolerances = _scenario_schedule_data(scenario_data)
	priority_order = _agent_priority_order(scenario_data)
	completed_schedules: Dict[int, List[float]] = {}

	for agent_index in priority_order:
		route = [int(node) for node in routes[agent_index]]
		if not route:
			raise ValueError(f"Agent {agent_index} has an empty route")
		min_speed, max_speed = speed_lim_mat[agent_index]
		if min_speed <= 0 or max_speed < min_speed:
			raise ValueError(f"Agent {agent_index} has invalid speed limits: {speed_lim_mat[agent_index]}")

		schedule = [float(start_times[agent_index])]
		for source, destination in zip(route, route[1:]):
			edge_distance = float(dist_mat[source, destination])
			if not np.isfinite(edge_distance) or edge_distance <= 0:
				raise ValueError(
					f"Agent {agent_index} route contains invalid edge: {source} -> {destination}"
				)

			nominal_arrival = schedule[-1] + edge_distance / max_speed
			earliest_allowed_arrival = nominal_arrival
			for previous_index, previous_schedule in completed_schedules.items():
				previous_route = routes[previous_index]
				for previous_position, previous_node in enumerate(previous_route):
					if previous_node == destination:
						earliest_allowed_arrival = max(
							earliest_allowed_arrival,
							previous_schedule[previous_position] + max(float(tolerances[destination]), 0.0),
						)

			travel_time = earliest_allowed_arrival - schedule[-1]
			minimum_allowed_travel_time = edge_distance / min_speed
			if travel_time > minimum_allowed_travel_time + 1e-9:
				raise ValueError(
					f"Agent {agent_index} cannot avoid a conflict at waypoint {destination} "
					f"within its speed limits on segment {source} -> {destination}"
				)
			schedule.append(float(earliest_allowed_arrival))

		completed_schedules[agent_index] = schedule

	return [completed_schedules[agent_index] for agent_index in range(len(routes))]


def shortest_path_baseline(
	scenario_data: Dict[str, Any],
) -> Tuple[List[List[int]], List[List[float]]]:
	"""Return shortest routes and their prioritized, speed-feasible schedules."""
	routes = shortest_path_routes(scenario_data)
	return routes, assign_prioritized_schedules(scenario_data, routes)


def load_scenario(scenario_path: str | Path) -> Dict[str, Any]:
	"""Load scenario data from a scenario directory or scenario_data.pkl file."""
	path = Path(scenario_path)
	data_file = path / "scenario_data.pkl" if path.is_dir() else path
	if not data_file.exists():
		raise FileNotFoundError(f"Scenario data file not found: {data_file}")
	with data_file.open("rb") as scenario_file:
		return pickle.load(scenario_file)


def run_scenario(
	scenario_path: str | Path, output_path: str | Path | None = None
) -> Dict[str, Any]:
	"""Run the baseline for one scenario and save its routes and schedules."""
	path = Path(scenario_path)
	scenario_data = load_scenario(path)
	start_time = time.perf_counter()
	routes, schedules = shortest_path_baseline(scenario_data)
	runtime = time.perf_counter() - start_time

	solution_data = {
		"name": "SEQUENTIAL_SHORTEST_PATH",
		"method": "sequential_shortest_path",
		"scenario_id": scenario_data.get("scenario_id"),
		"n_agents": int(scenario_data["n_agents"]),
		"agent_routes": routes,
		"agent_schedules": schedules,
		"start_times": scenario_data["mirs_summary"]["start_times"],
		"speed_lim_mat": scenario_data["mirs_summary"]["speed_lim_mat"],
		"agent_weights": scenario_data["mirs_summary"].get("agent_weights"),
		"runtime": runtime,
	}

	if output_path is None:
		output_file = path / "solution_seq.pkl" if path.is_dir() else path.parent / "solution_seq.pkl"
	else:
		output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	with output_file.open("wb") as solution_file:
		pickle.dump(solution_data, solution_file)
	solution_data["output_path"] = str(output_file)
	return solution_data


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run the sequential shortest-path routing and scheduling baseline."
	)
	parser.add_argument(
		"--scenario-path",
		type=Path,
		required=True,
		help="Scenario directory or path to scenario_data.pkl.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional output pickle path. Defaults to solution_seq.pkl beside the scenario.",
	)
	args = parser.parse_args()

	solution_data = run_scenario(args.scenario_path, args.output)
	print(f"scenario={solution_data['scenario_id']}")
	print(f"routes={solution_data['agent_routes']}")
	print(f"schedules={solution_data['agent_schedules']}")
	print(f"runtime={solution_data['runtime']:.6f}")
	print(f"saved_solution={solution_data['output_path']}")


if __name__ == "__main__":
	main()
