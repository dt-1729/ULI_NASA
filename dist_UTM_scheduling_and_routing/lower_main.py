import pickle
import time
from pathlib import Path

import numpy as np

import visualization

from dist_MIRS import heir_MIRS
from higher_main import _make_parser, build_configs
from scenario_generator import load_scenario


def _load_higher(solution_path):
    with Path(solution_path).open("rb") as file:
        higher = pickle.load(file)
    missing = [
        k for k in ("scenario", "boundary_fraction", "routes", "route_schedules")
        if k not in higher
    ]
    if missing:
        raise ValueError(f"Saved solution is missing required data: {', '.join(missing)}")
    return higher


def combine_saved_subgraphs(solution_path, output_root):
    """Combine the per-subgraph pickles written by independent (e.g. cluster) solves.

    A subgraph without a pickle is treated as having no routable agents; combine_solutions
    raises if such a subgraph actually has visits to route.
    """
    higher = _load_higher(solution_path)
    scenario = higher["scenario"]
    graph = heir_MIRS(scenario, higher["boundary_fraction"])
    output_root = Path(output_root)
    results = {}
    for subgraph in sorted(int(p) for p in np.unique(graph.partition)):
        path = (
            output_root / "subgraphs" / f"subgraph_{subgraph}"
            / f"solution_subgraph_{subgraph}.pkl"
        )
        if path.exists():
            with path.open("rb") as file:
                results[subgraph] = pickle.load(file)
        else:
            print(f"Subgraph {subgraph}: no saved solution at {path}")
            results[subgraph] = None
    return _write_combined(
        graph, higher, results, scenario, solution_path, output_root, runtime=None
    )


def _write_combined(graph, higher, results, scenario, solution_path, output_root, runtime):
    combined_dir = Path(output_root) / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    full_routes, full_schedules, overlap, paths = graph.visualize_combined(
        higher["routes"], higher["route_schedules"], results,
        scenario["tolArray"], combined_dir,
    )
    combined = {
        "routes": full_routes,
        "route_schedules": full_schedules,
        "max_overlap": overlap,
        "plots": [str(p) for p in paths],
    }
    print(f"Combined solution: max exit/entry overlap={overlap:.3g}")
    for agent, (route, schedule) in enumerate(zip(full_routes, full_schedules)):
        print(f"  Agent {agent}: route={route}, schedule={np.round(schedule, 3)}")
    print(f"Saved combined plots: {', '.join(str(p) for p in paths)}")
    combined_path = combined_dir / "solution_combined.pkl"
    with combined_path.open("wb") as file:
        pickle.dump(
            {
                "higher_solution": str(solution_path),
                "subgraph_solutions": {k: v for k, v in results.items() if v is not None},
                "combined_solution": combined,
                "runtime": runtime,
            },
            file,
        )
    print(f"Saved combined solution: {combined_path}")
    return combined


def solve_lower_level(solution_path, output_root, *, args, subgraphs=None):
    """Solve every subgraph (or the selected ones) from a saved higher-level solution.

    Results go under output_root: subgraphs/subgraph_<id>/ per subgraph and combined/.
    The combined solution is only produced when all subgraphs are solved in this call.
    """
    higher = _load_higher(solution_path)
    scenario = higher["scenario"]
    optim_config, anneal_config, _ = build_configs(args, scenario["tolArray"])
    graph = heir_MIRS(scenario, higher["boundary_fraction"])
    started = time.time()
    results = graph.solve_subgraphs(
        higher["routes"],
        higher["route_schedules"],
        scenario["tolArray"],
        subgraphs=subgraphs,
        time_limit=args.time_limit,
        anneal_print=args.anneal_print,
        optim_config=optim_config,
        anneal_config=anneal_config,
        stagewiseCostCoeffs=args.stagewise_cost_coeffs,
        selfHop=args.self_hop,
        cost_mode=args.cost_mode,
        lm=args.lm,
        filter_wp_thresh=args.filter_wp_thresh,
        prune_mode=args.prune_mode,
    )
    runtime = time.time() - started

    saved = {}
    for subgraph, result in results.items():
        if result is None:
            print(f"Subgraph {subgraph}: no routable agents, skipped")
            continue
        saved[subgraph] = {k: v for k, v in result.items() if k != "mirs"}
        print(
            f"Subgraph {subgraph}: agents={[int(a) for a in result['agent_ids']]}, "
            f"final cost={result['cost_history'][-1]:.6g}, runtime={result['runtime']:.1f} s"
        )
        for agent, route, schedule in zip(
            result["agent_ids"], result["routes"], result["route_schedules"]
        ):
            print(f"  Agent {int(agent)}: route={route}, schedule={np.round(schedule, 3)}")

    output_root = Path(output_root)
    for subgraph, result in results.items():
        if result is None:
            continue
        folder = output_root / "subgraphs" / f"subgraph_{subgraph}"
        paths = visualization.plot_subgraph_solution(
            result, scenario["wp_locations"], scenario["tolArray"], folder
        )
        saved[subgraph]["plots"] = [str(p) for p in paths]
        with (folder / f"solution_subgraph_{subgraph}.pkl").open("wb") as file:
            pickle.dump(saved[subgraph], file)
        print(f"Saved subgraph {subgraph} solution and plots in {folder}")

    if subgraphs is None:
        _write_combined(
            graph, higher, results, scenario, solution_path, output_root, runtime
        )
    else:
        print("Skipping combined solution: not all subgraphs were solved")
    return saved


def main():
    parser = _make_parser()
    parser.description = "Solve all subgraphs from a saved higher-level MIRS solution"
    parser.add_argument(
        "--subgraphs", type=int, nargs="+", help="Subgraph ids to solve (default: all)"
    )
    parser.add_argument(
        "--combine_only", action="store_true",
        help="Skip solving; combine saved subgraph solutions (used after a cluster job array)",
    )
    parser.add_argument(
        "--n_subgraphs", action="store_true",
        help="Print the number of subgraphs of the scenario and exit",
    )
    args = parser.parse_args()

    folder = Path(args.scenario_folder) if args.scenario_folder else None
    if args.n_subgraphs:
        if folder is None:
            raise ValueError("--scenario_folder is required with --n_subgraphs")
        print(len(np.unique(load_scenario(folder)["partition"])))
        return
    solution = args.solution or (folder / "coarse" / "solution_coarse_cbf.pkl" if folder else None)
    if not solution:
        raise ValueError("--solution or --scenario_folder is required")
    solution = Path(solution)
    if args.output:
        output_root = Path(args.output)  # output directory
    elif folder:
        output_root = folder
    else:
        output_root = solution.parent.parent if solution.parent.name == "coarse" else solution.parent
    if args.combine_only:
        combine_saved_subgraphs(solution, output_root)
        return
    solve_lower_level(solution, output_root, args=args, subgraphs=args.subgraphs)


if __name__ == "__main__":
    main()
