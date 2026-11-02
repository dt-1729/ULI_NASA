import pickle
from pathlib import Path

import numpy as np

from cli_args import generator_args
import problem_generator
from problem_generator import init_waypoints, init_agent_params
from dist_MIRS import generate_partition

SCENARIO_ROOT = Path("scenarios")
INF = 1e8


def partition_labels(n_waypoints, n_parts):
    # generate_multigraph assigns nodes to n_graphs contiguous blocks (sizes differ by at most one)
    sizes = [n_waypoints // n_parts + (g < n_waypoints % n_parts) for g in range(n_parts)]
    return np.repeat(np.arange(n_parts), sizes)


def create_scenario(n_waypoints, n_agents, n_parts, seed=1, network_type="multi", tol_range=(3.0, 5.0), inter_edges_per_pair=2, max_edge_length=None):
    net_params, tol_array = problem_generator.get_network_params(n_waypoints, list(tol_range), network_type)
    if network_type == "random":
        net_params["max_edge_length"] = max_edge_length
    if network_type == "multi":
        net_params["n_graphs"] = n_parts
        # keep inter-subgraph edges few: about inter_edges_per_pair for each pair of subgraphs
        n_cross_pairs = (n_waypoints**2 - sum(sz**2 for sz in np.bincount(partition_labels(n_waypoints, n_parts)))) // 2
        n_part_pairs = n_parts * (n_parts - 1) / 2
        net_params["inter_graph_connectivity"] = inter_edges_per_pair * n_part_pairs / n_cross_pairs
    wp_locations, mask, dist_mat, _, _ = init_waypoints(net_params, seed, INF)
    sd_mat = init_agent_params(n_agents, wp_locations, seed, tolArray=tol_array)[1]

    if network_type == "multi":
        partition = partition_labels(n_waypoints, n_parts)
    else:
        partition = generate_partition(wp_locations, n_parts, seed)

    return dict(wp_locations=wp_locations, mask=mask, dist_mat=dist_mat, sd_mat=sd_mat,
                partition=partition, n_agents=n_agents, n_parts=n_parts, seed=seed,
                network_type=network_type, net_params=net_params, tolArray=tol_array)


def save_scenario(scenario, root=SCENARIO_ROOT):
    s = scenario
    folder = root / f"{s['network_type']}_nwp{len(s['wp_locations'])}_na{s['n_agents']}_p{s['n_parts']}_seed{s['seed']}"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "scenario.pkl", "wb") as f:
        pickle.dump(s, f)
    return folder


def load_scenario(folder):
    with open(Path(folder) / "scenario.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    a = generator_args()
    scenario = create_scenario(a.n_waypoints, a.n_agents, a.n_parts, a.seed, a.network_type,
                               (a.tol_min, a.tol_max), a.inter_edges_per_pair, a.max_edge_length)
    print(save_scenario(scenario, Path(a.scenario_root)))
