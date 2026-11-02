import argparse


def add_problem_args(parser):
    parser.add_argument("--n_waypoints", type=int, default=40)
    parser.add_argument("--n_agents", type=int, default=8)
    parser.add_argument("--n_parts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--network_type", default="multi", choices=["grid", "ring", "multi", "random"])
    parser.add_argument("--inter_edges_per_pair", type=int, default=2,
                        help="multi only: approx. edges between each pair of subgraphs")
    parser.add_argument("--max_edge_length", type=float, default=None,
                        help="random only: no long-range edge is longer than this")
    parser.add_argument("--tol_min", type=float, default=3.0)
    parser.add_argument("--tol_max", type=float, default=5.0)
    parser.add_argument("--scenario_root", default="scenarios")


def add_scenario_arg(parser):
    parser.add_argument("--scenario_folder", required=True, help="folder containing scenario.pkl")


def generator_args():
    parser = argparse.ArgumentParser(description="Generate a partitioned MIRS scenario")
    add_problem_args(parser)
    return parser.parse_args()


def test_args():
    parser = argparse.ArgumentParser(description="Build and plot original and coarse graphs")
    add_scenario_arg(parser)
    parser.add_argument("--boundary_fraction", type=float, default=0.2,
                        help="fraction of edges between each pair of subgraphs used as boundary (min 1)")
    return parser.parse_args()
