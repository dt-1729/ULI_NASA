from __future__ import annotations

import argparse
import heapq
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_scenario(path: str | Path) -> Dict:
    scenario_path = Path(path)
    data_file = scenario_path / "scenario_data.pkl"
    if not data_file.exists():
        raise FileNotFoundError(f"Scenario data file not found: {data_file}")

    with open(data_file, "rb") as f:
        return pickle.load(f)


def build_adjacency(scenario_data: Dict, agent_index: int = 0) -> Tuple[List[List[Tuple[int, float]]], np.ndarray, np.ndarray, np.ndarray]:
    summary = scenario_data.get("mirs_summary", {})
    dist_mat = np.asarray(summary.get("dist_mat", np.zeros((scenario_data["n_waypoints"], scenario_data["n_waypoints"]))))
    mask = np.asarray(summary.get("mask", np.ones_like(dist_mat, dtype=bool)))
    speed_lim_mat = np.asarray(summary.get("speed_lim_mat", np.zeros((scenario_data["n_agents"], 2))))

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
            connected = bool(mask[i, j] or mask[j, i])
            if not connected:
                continue
            edge_length = float(dist_mat[i, j])
            if not np.isfinite(edge_length) or edge_length <= 0:
                continue
            travel_time = edge_length / max_speed
            if travel_time <= 0:
                travel_time = 1e-6
            adjacency[i].append((j, travel_time))
            adjacency[j].append((i, travel_time))

    return adjacency, dist_mat, mask, speed_lim_mat


def search_single_agent_on_time_expanded_graph(
    scenario_data: Dict,
    agent_index: int,
    time_step: float = 1.0,
    horizon: float | None = None,
) -> Tuple[List[int], List[float], float]:
    if agent_index < 0 or agent_index >= scenario_data["n_agents"]:
        raise ValueError(f"agent_index out of range: {agent_index} for {scenario_data['n_agents']} agents")

    summary = scenario_data.get("mirs_summary", {})
    if "sd_mat" in summary:
        sd_mat = np.asarray(summary["sd_mat"])
    elif "mirs_constructor_kwargs" in scenario_data and "sd_mat" in scenario_data["mirs_constructor_kwargs"]:
        sd_mat = np.asarray(scenario_data["mirs_constructor_kwargs"]["sd_mat"])
    else:
        raise KeyError("Scenario does not contain agent start/destination data under 'sd_mat'.")

    if "start_times" in summary:
        start_times = np.asarray(summary["start_times"])
    elif "initial_conditions" in scenario_data and "T0" in scenario_data["initial_conditions"]:
        start_times = np.asarray(scenario_data["initial_conditions"]["T0"])[:, 0]
    else:
        start_times = np.zeros(scenario_data["n_agents"], dtype=float)

    start_node = int(sd_mat[agent_index, 0])
    goal_node = int(sd_mat[agent_index, 1])
    start_time = float(start_times[agent_index])

    adjacency, dist_mat, _, speed_lim_mat = build_adjacency(scenario_data, agent_index)
    if horizon is None:
        speed_values = np.asarray(summary.get("speed_lim_mat", speed_lim_mat))
        if speed_values.size == 0:
            speed_values = np.ones((scenario_data["n_agents"], 2), dtype=float)
        max_speed = max(float(np.min(speed_values[:, 1])), 1e-6)
        horizon = float(np.max(dist_mat)) / max_speed + start_time + 100.0

    pq: List[Tuple[float, int, int]] = []
    best_cost: Dict[Tuple[int, int], float] = {(start_node, 0): start_time}
    prev: Dict[Tuple[int, int], Tuple[int, int] | None] = {(start_node, 0): None}
    heapq.heappush(pq, (start_time, start_node, 0))

    while pq:
        arrival_time, node, bucket = heapq.heappop(pq)
        state = (node, bucket)

        if arrival_time > best_cost.get(state, float("inf")):
            continue

        if node == goal_node:
            route: List[int] = []
            schedule: List[float] = []
            cursor = state
            while cursor is not None:
                node_idx, _ = cursor
                route.append(node_idx)
                schedule.append(best_cost[cursor])
                cursor = prev[cursor]
            route.reverse()
            schedule.reverse()
            return route, schedule, arrival_time - start_time

        for neighbor, travel_time in adjacency[node]:
            next_time = arrival_time + travel_time
            if next_time > horizon:
                continue

            next_bucket = int(np.floor(next_time / max(time_step, 1e-6)))
            next_state = (neighbor, next_bucket)
            if next_state in best_cost and next_time >= best_cost[next_state]:
                continue

            best_cost[next_state] = next_time
            prev[next_state] = state
            heapq.heappush(pq, (next_time, neighbor, next_bucket))

    raise RuntimeError(f"No feasible route found for agent {agent_index} from {start_node} to {goal_node}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-expanded single-agent search for a saved scenario.")
    parser.add_argument("--scenario-path", type=str, required=True, help="Path to a scenario folder containing scenario_data.pkl")
    parser.add_argument("--agent-index", type=int, default=0, help="Index of the agent to plan")
    parser.add_argument("--time-step", type=float, default=1.0, help="Discrete time step used in the expanded graph")
    parser.add_argument("--horizon", type=float, default=None, help="Optional search horizon in time units")
    args = parser.parse_args()

    scenario_data = load_scenario(args.scenario_path)
    path, schedule, total_time = search_single_agent_on_time_expanded_graph(
        scenario_data,
        agent_index=args.agent_index,
        time_step=args.time_step,
        horizon=args.horizon,
    )

    print(f"agent_index={args.agent_index}")
    print(f"route={path}")
    print(f"schedule={schedule}")
    print(f"arrival_delay={total_time}")


if __name__ == "__main__":
    main()
