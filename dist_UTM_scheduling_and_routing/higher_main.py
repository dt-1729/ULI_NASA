import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from MIRS_coarse import MIRS_coarse
from dist_MIRS import heir_MIRS
from higher_mep_opt import MIRSOptimizer
from scenario_generator import load_scenario
import visualization


def build_configs(args, tol_array):
    optim_config = {
        "name": "cbf_clf",
        "dt_init": args.dt_init,
        "dt_min": args.dt_min,
        "dt_max": args.dt_max,
        "Tf": args.tf,
        "gamma": args.gamma,
        "alpha_s": args.alpha_s,
        "alpha_c": args.alpha_c,
        "alpha_r": args.alpha_r,
        "alpha_q": args.alpha_q,
        "p": args.slack_penalty,
        "stop_tol": args.stop_tol,
        "stop_tol_weight": np.asarray(args.stop_tol_weights, dtype=float),
        "verbose": int(args.verbose),
        "QP_solver_name": args.qp_solver,
        "qp_solver_opts": {
            "verbose": args.solver_verbose,
            "max_iter": args.solver_max_iter,
            "eps_abs": args.solver_tolerance,
            "eps_rel": args.solver_tolerance,
        },
    }
    anneal_config = {
        "log_bmin": args.log_bmin,
        "log_bmax": args.log_bmax,
        "nb": args.n_beta,
        "ra": args.ra,
        "rb": args.rb,
        "rw": args.rw,
        "rh": args.rh,
        "ra_rate": args.ra_rate,
        "rb_rate": args.rb_rate,
        "rw_rate": args.rw_rate,
        "rh_rate": args.rh_rate,
    }
    if args.cbf_mode == "rect":
        ca_cbf = {
            "mode": "rect",
            "width": 1e-3 * tol_array,
            "height": 1e-4 * tol_array,
            "gamma": args.rect_gamma,
            "width_correction_fac": args.width_correction,
            "height_correction_fac": args.height_correction,
        }
    elif args.cbf_mode == "ellipse":
        ca_cbf = {
            "mode": "ellipse",
            "major_axis": np.ones_like(tol_array),
            "minor_axis": tol_array.copy(),
            "degree": args.ellipse_degree,
        }
    else:
        ca_cbf = {"mode": "lin_static"}
    return optim_config, anneal_config, ca_cbf


def solve_scenario(
    scenario,
    output_path,
    *,
    boundary_fraction=0.2,
    cbf_mode="rect",
    time_limit=3600.0,
    anneal_print=False,
    args,
):
    if args.cbf_mode != cbf_mode:
        raise ValueError("args.cbf_mode and cbf_mode must match")

    optim_config, anneal_config, ca_cbf = build_configs(args, scenario["tolArray"])
    mirs = MIRS_coarse(
        scenario,
        boundary_fraction=boundary_fraction,
        offset_energy=True,
        stagewiseCostCoeffs=args.stagewise_cost_coeffs,
        selfHop=args.self_hop,
        cost_mode=args.cost_mode,
        lm=args.lm,
        ca_cbf=ca_cbf,
        filter_wp_thresh=args.filter_wp_thresh,
        prune_mode=args.prune_mode,
        T_upper_bound=args.schedule_upper_bound,
    )
    optimizer = MIRSOptimizer(mirs, optim_config, anneal_config)

    started = time.time()
    T_array, V_array, F_values, path_probabilities, reach_array, compute_times = (
        optimizer.anneal(
            mirs.sched_mat,
            mirs.speed_vec,
            active_waypoints=None,
            time_limit=time_limit,
            annealPrint=anneal_print,
        )
    )
    runtime = time.time() - started
    final_schedule = T_array[-1]
    final_speed = V_array[-1]
    final_beta = optimizer.b_arr[len(F_values) - 1]
    routes, route_schedules = mirs.solution_table(
        path_probabilities,
        final_schedule,
        final_speed,
        final_beta,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    network_plot_path = output_path.parent / "network_plot_coarse_cbf.png"
    schedule_plot_path = output_path.parent / "schedule_plot_coarse_cbf.png"
    agent_schedule_plot_path = (
        output_path.parent / "agent_schedule_plot_coarse_cbf.png"
    )
    visualization.plot_solution_network(
        mirs.coarse_graph,
        routes,
        route_schedules,
        save_path=network_plot_path,
    )
    visualization.plot_solution_waypoint_schedules(
        routes,
        route_schedules,
        mirs.sd_mat,
        scenario["tolArray"],
        save_path=schedule_plot_path,
    )
    visualization.plot_solution_schedules(
        routes,
        route_schedules,
        mirs.sd_mat,
        save_path=agent_schedule_plot_path,
    )
    subgraph_visits = mirs.coarse_graph.subgraph_agent_visits(routes, route_schedules)
    subgraph_plot_paths = visualization.plot_subgraph_visits(
        subgraph_visits, output_path.parent / "subgraph_plots"
    )
    reach_mat, association_mat = mirs.calc_agent_reach_mat_v1(
        final_schedule, final_speed, final_beta
    )

    solution = {
        "scenario": scenario,
        "boundary_fraction": boundary_fraction,
        "cbf_mode": cbf_mode,
        "optimizer_config": optim_config,
        "anneal_config": anneal_config,
        "beta_array": optimizer.b_arr[: len(F_values)],
        "schedule_history": T_array,
        "speed_history": V_array,
        "cost_history": F_values,
        "reach_history": reach_array,
        "compute_times": compute_times,
        "final_schedule": final_schedule,
        "final_speed": final_speed,
        "final_cost": float(F_values[-1]),
        "final_beta": float(final_beta),
        "path_probabilities": path_probabilities,
        "reachability": reach_mat,
        "association": association_mat,
        "routes": routes,
        "route_schedules": route_schedules,
        "runtime": runtime,
        "network_plot": str(network_plot_path),
        "schedule_plot": str(schedule_plot_path),
        "agent_schedule_plot": str(agent_schedule_plot_path),
        "subgraph_visits": subgraph_visits,
        "subgraph_plots": [str(path) for path in subgraph_plot_paths],
    }
    with output_path.open("wb") as file:
        pickle.dump(solution, file)

    print(f"Coarse waypoints: {len(mirs.coarse_graph.coarse_nodes)}")
    print(f"Final beta: {final_beta:.6g}")
    print(f"Final cost: {F_values[-1]:.6g}")
    print(f"Runtime: {runtime:.2f} s")
    print(f"Saved solution: {output_path}")
    print(
        f"Saved plots: {network_plot_path}, {schedule_plot_path}, "
        f"{agent_schedule_plot_path}"
    )
    for i, route in enumerate(routes):
        print(f"Agent {i}: route={route}, schedule={np.round(route_schedules[i], 3)}")
    return solution


def visualize_saved_solution(solution_path, output_dir=None):
    """Regenerate coarse network and both schedule views from a saved solution."""
    solution_path = Path(solution_path)
    with solution_path.open("rb") as file:
        solution = pickle.load(file)

    required = ("scenario", "boundary_fraction", "routes", "route_schedules")
    missing = [key for key in required if key not in solution]
    if missing:
        raise ValueError(
            f"Saved solution is missing required visualization data: {', '.join(missing)}"
        )

    scenario = solution["scenario"]
    coarse_graph = heir_MIRS(scenario, solution["boundary_fraction"])
    output_dir = Path(output_dir) if output_dir else solution_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    network_plot_path = output_dir / "network_plot_coarse_cbf.png"
    schedule_plot_path = output_dir / "schedule_plot_coarse_cbf.png"
    agent_schedule_plot_path = output_dir / "agent_schedule_plot_coarse_cbf.png"

    visualization.plot_solution_network(
        coarse_graph,
        solution["routes"],
        solution["route_schedules"],
        save_path=network_plot_path,
    )
    visualization.plot_solution_waypoint_schedules(
        solution["routes"],
        solution["route_schedules"],
        np.asarray(scenario["sd_mat"]),
        scenario["tolArray"],
        save_path=schedule_plot_path,
    )
    visualization.plot_solution_schedules(
        solution["routes"],
        solution["route_schedules"],
        np.asarray(scenario["sd_mat"]),
        save_path=agent_schedule_plot_path,
    )
    subgraph_plot_paths = visualization.plot_subgraph_visits(
        coarse_graph.subgraph_agent_visits(
            solution["routes"], solution["route_schedules"]
        ),
        output_dir / "subgraph_plots",
    )
    print(
        f"Saved plots: {network_plot_path}, {schedule_plot_path}, "
        f"{agent_schedule_plot_path}"
    )
    print(f"Saved {len(subgraph_plot_paths)} subgraph plots in {output_dir / 'subgraph_plots'}")
    return network_plot_path, schedule_plot_path, agent_schedule_plot_path


def _make_parser():
    parser = argparse.ArgumentParser(
        description="Solve or visualize a hierarchical coarse MIRS CBF-CLF solution"
    )
    parser.add_argument(
        "--mode",
        choices=("solve", "visualize"),
        default="solve",
        help="Solve a scenario or regenerate plots from a saved solution",
    )
    parser.add_argument("--scenario_folder", help="Folder containing scenario.pkl")
    parser.add_argument(
        "--solution",
        help="Saved solution pickle (required for visualize; defaults to the solve output)",
    )
    parser.add_argument(
        "--plot_dir",
        help="Directory for regenerated plots (default: saved solution directory)",
    )
    parser.add_argument("--output", help="Solution pickle path (default: scenario folder)")
    parser.add_argument("--boundary_fraction", type=float, default=0.2)
    parser.add_argument("--cbf_mode", choices=("rect", "ellipse", "lin_static"), default="rect")
    parser.add_argument("--time_limit", type=float, default=3600.0)
    parser.add_argument("--anneal_print", action="store_true")
    parser.add_argument("--n_beta", type=int, default=15)
    parser.add_argument("--log_bmin", type=float, default=-6.0)
    parser.add_argument("--log_bmax", type=float, default=0.0)
    parser.add_argument("--dt_init", type=float, default=0.01)
    parser.add_argument("--dt_min", type=float, default=1e-6)
    parser.add_argument("--dt_max", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha_s", type=float, default=100.0)
    parser.add_argument("--alpha_c", type=float, default=10.0)
    parser.add_argument("--alpha_r", type=float, default=100.0)
    parser.add_argument("--alpha_q", type=float, default=100.0)
    parser.add_argument("--slack_penalty", type=float, default=1.0)
    parser.add_argument("--stop_tol", type=float, default=5e-6)
    parser.add_argument(
        "--stop_tol_weights", type=float, nargs=4, default=(0.0, 0.0, 1.0, 0.0)
    )
    parser.add_argument("--qp_solver", default="OSQP")
    parser.add_argument("--solver_max_iter", type=int, default=100000)
    parser.add_argument("--solver_tolerance", type=float, default=1e-3)
    parser.add_argument("--solver_verbose", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rect_gamma", type=float, default=1.0)
    parser.add_argument("--width_correction", type=float, default=0.5)
    parser.add_argument("--height_correction", type=float, default=0.5)
    parser.add_argument("--ellipse_degree", type=int, default=2)
    parser.add_argument("--ra", type=float, default=0.01)
    parser.add_argument("--rb", type=float, default=0.001)
    parser.add_argument("--rw", type=float, default=0.001)
    parser.add_argument("--rh", type=float, default=0.01)
    parser.add_argument("--ra_rate", type=float, default=2.0)
    parser.add_argument("--rb_rate", type=float, default=2.0)
    parser.add_argument("--rw_rate", type=float, default=2.0)
    parser.add_argument("--rh_rate", type=float, default=2.0)
    parser.add_argument(
        "--stagewise_cost_coeffs", type=float, nargs=3, default=(0.1, 10.0, 0.1)
    )
    parser.add_argument("--self_hop", action="store_true")
    parser.add_argument("--cost_mode", choices=("sum", "slowest"), default="sum")
    parser.add_argument("--lm", type=float, default=1.0)
    parser.add_argument("--filter_wp_thresh", type=float, default=0.0)
    parser.add_argument("--prune_mode", action="store_true")
    parser.add_argument("--schedule_upper_bound", type=float, default=10000.0)
    return parser


def main():
    args = _make_parser().parse_args()
    if args.mode == "visualize":
        if not args.solution:
            raise ValueError("--solution is required when --mode=visualize")
        visualize_saved_solution(args.solution, args.plot_dir)
        return

    if not args.scenario_folder:
        raise ValueError("--scenario_folder is required when --mode=solve")
    if args.n_beta < 1:
        raise ValueError("--n_beta must be at least one")
    if args.time_limit <= 0:
        raise ValueError("--time_limit must be positive")
    scenario_folder = Path(args.scenario_folder)
    scenario = load_scenario(scenario_folder)
    output_path = (
        Path(args.output)
        if args.output
        else scenario_folder / "coarse" / "solution_coarse_cbf.pkl"
    )
    solve_scenario(
        scenario,
        output_path,
        boundary_fraction=args.boundary_fraction,
        cbf_mode=args.cbf_mode,
        time_limit=args.time_limit,
        anneal_print=args.anneal_print,
        args=args,
    )

if __name__ == "__main__":
    main()
