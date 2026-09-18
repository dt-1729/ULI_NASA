from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import MIRS
import gb_opt
import mep_opt
import problem_generator
import utils
import visualize


DEFAULT_SCENARIO_ROOT = Path("local_scenarios")


def _to_python_primitive(value: Any) -> Any:
    """Convert NumPy / array-like values into JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _to_python_primitive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_python_primitive(v) for v in value]
    if isinstance(value, tuple):
        return [_to_python_primitive(v) for v in value]
    return value


def _scenario_folder_name(n_waypoints: int, n_agents: int, seed: int, index: int) -> str:
    """Name one seed realization within a fixed problem-size group."""
    return f"scenario_{index:03d}_seed{seed}"


def _problem_size_folder(n_waypoints: int, n_agents: int) -> Path:
    """Return the stable directory hierarchy for one (nwp, na) problem size."""
    return Path(f"nwp{n_waypoints}_na{n_agents}")


def _parse_problem_size(value: str) -> Tuple[int, int]:
    """Parse one command-line problem size written as NWP:NA (for example, 10:3)."""
    try:
        nwp_text, na_text = value.split(":", maxsplit=1)
        n_waypoints, n_agents = int(nwp_text), int(na_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid problem size '{value}'. Use NWP:NA, for example 10:3."
        ) from exc
    if n_waypoints <= 0 or n_agents <= 0 or n_agents > n_waypoints:
        raise argparse.ArgumentTypeError(
            f"Invalid problem size '{value}': require 0 < NA <= NWP."
        )
    return n_waypoints, n_agents


def build_mirs_instance(
    n_waypoints: int,
    n_agents: int,
    seed: int,
    tol_range: Tuple[float, float] = (3.0, 5.0),
    network_type: str = "grid",
    cost_mode: str = "sum",
    lm: float = 1,
    cbf_mode_name: str = "rect",
    offset_energy: int = 1,
    stagewise_cost_coeffs: np.ndarray | None = None,
    self_hop: int = 0,
    filter_wp_thresh: float = 1e-3,
    prune_mode: bool = False,
    print_flag: bool = False,
    T_upper_bound: float = 10000,
) -> Tuple[MIRS.MIRS, Dict[str, Any], Dict[str, Any]]:
    """
    Build one MIRS instance and the associated optimizer metadata.
    Returns: (mirs, constructor_kwargs, initial_conditions)
    """
    if stagewise_cost_coeffs is None:
        stagewise_cost_coeffs = np.array([0.1, 10.0, 0.1], dtype=float)

    net_params, tol_array = problem_generator.get_network_params(
        n_waypoints,
        tol_range=list(tol_range),
        name=network_type,
    )
    cbf_mode = utils.get_cbf_mode(cbf_mode_name, tol_array)

    mirs = MIRS.MIRS(
        n_waypoints=n_waypoints,
        n_agents=n_agents,
        tolArray=tol_array,
        wp_params=net_params,
        seed=seed,
        offset_energy=offset_energy,
        stagewiseCostCoeffs=stagewise_cost_coeffs,
        selfHop=self_hop,
        cost_mode=cost_mode,
        lm=lm,
        ca_cbf=cbf_mode,
        filter_wp_thresh=filter_wp_thresh,
        prune_mode=prune_mode,
        printFlag=print_flag,
        T_upper_bound=T_upper_bound,
    )

    constructor_kwargs = {
        "n_waypoints": n_waypoints,
        "n_agents": n_agents,
        "tolArray": tol_array,
        "wp_params": net_params,
        "seed": seed,
        "offset_energy": offset_energy,
        "stagewiseCostCoeffs": stagewise_cost_coeffs,
        "selfHop": self_hop,
        "cost_mode": cost_mode,
        "lm": lm,
        "ca_cbf": cbf_mode,
        "filter_wp_thresh": filter_wp_thresh,
        "prune_mode": prune_mode,
        "printFlag": print_flag,
        "T_upper_bound": T_upper_bound,
    }

    initial_conditions = {
        "T0": mirs.sched_mat.copy(),
        "V0": mirs.speed_vec.copy(),
        "active_waypoints": list(range(mirs.n_waypoints)),
    }

    return mirs, constructor_kwargs, initial_conditions


def build_optimizer_payload() -> Dict[str, Any]:
    """Return the optimizer configuration used by the project."""
    cbf_config, anneal_config = utils.set_mep_opt_config("cbf")
    slsqp_config, _ = utils.set_mep_opt_config("slsqp")
    cbf_static_config, _ = utils.set_mep_opt_config("cbf_static")
    slsqp_static_config, _ = utils.set_mep_opt_config("slsqp_static")

    return {
        "cbf_mep": {
            "config": cbf_config,
            "anneal_config": anneal_config,
            "entry_point": "mep_opt.MIRSOptimizer",
            "method_name": "CBF_CLF_at_beta",
            "note": "Uses the CBF-CLF fixed-beta loop from mep_opt.py on the default rect CBF mode.",
        },
        "slsqp_mep": {
            "config": slsqp_config,
            "anneal_config": anneal_config,
            "entry_point": "mep_opt.MIRSOptimizer",
            "method_name": "slsqp_at_beta",
            "note": "Uses the SLSQP fixed-beta optimizer from mep_opt.py on the default rect CBF mode.",
        },
        "cbf_static_mep": {
            "config": cbf_static_config,
            "anneal_config": anneal_config,
            "entry_point": "mep_opt.MIRSOptimizer",
            "method_name": "CBF_CLF_at_beta",
            "note": "Uses the CBF-CLF fixed-beta loop from mep_opt.py with a temporary static barrier mode override.",
        },
        "slsqp_static_mep": {
            "config": slsqp_static_config,
            "anneal_config": anneal_config,
            "entry_point": "mep_opt.MIRSOptimizer",
            "method_name": "slsqp_at_beta",
            "note": "Uses the SLSQP fixed-beta optimizer from mep_opt.py with a temporary static barrier mode override.",
        },
        "gurobi": {
            "model_name": "gb_mirs_model",
            "entry_point": "gb_opt.MIRSGurobiOptimizer",
            "note": "Uses the Gurobi-based optimizer from gb_opt.py.",
        },
    }


def generate_single_scenario(
    root_dir: Path,
    scenario_index: int,
    n_waypoints: int,
    n_agents: int,
    seed: int,
    tol_range: Tuple[float, float],
    network_type: str,
    cost_mode: str,
    lm: float,
    cbf_mode_name: str,
    offset_energy: int,
    self_hop: int,
    filter_wp_thresh: float,
    prune_mode: bool,
    print_flag: bool,
    T_upper_bound: float,
) -> Dict[str, Any]:
    mirs, constructor_kwargs, initial_conditions = build_mirs_instance(
        n_waypoints=n_waypoints,
        n_agents=n_agents,
        seed=seed,
        tol_range=tol_range,
        network_type=network_type,
        cost_mode=cost_mode,
        lm=lm,
        cbf_mode_name=cbf_mode_name,
        offset_energy=offset_energy,
        stagewise_cost_coeffs=np.array([0.1, 10.0, 0.1], dtype=float),
        self_hop=self_hop,
        filter_wp_thresh=filter_wp_thresh,
        prune_mode=prune_mode,
        print_flag=print_flag,
        T_upper_bound=T_upper_bound,
    )

    scenario_dir = root_dir / _problem_size_folder(n_waypoints, n_agents) / _scenario_folder_name(
        n_waypoints=n_waypoints,
        n_agents=n_agents,
        seed=seed,
        index=scenario_index,
    )
    scenario_dir.mkdir(parents=True, exist_ok=True)

    scenario_data = {
        "scenario_id": scenario_dir.name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_waypoints": n_waypoints,
        "n_agents": n_agents,
        "seed": seed,
        "network_type": network_type,
        "tol_range": list(tol_range),
        "cost_mode": cost_mode,
        "lm": lm,
        "cbf_mode_name": cbf_mode_name,
        "offset_energy": offset_energy,
        "self_hop": self_hop,
        "filter_wp_thresh": filter_wp_thresh,
        "prune_mode": prune_mode,
        "T_upper_bound": T_upper_bound,
        "mirs_constructor_kwargs": constructor_kwargs,
        "initial_conditions": initial_conditions,
        "optimizer_specs": build_optimizer_payload(),
        "network_params": constructor_kwargs["wp_params"],
        "tol_array": mirs.tolArray,
        "mirs_summary": {
            "wp_locations": mirs.wp_locations,
            "mask": mirs.mask,
            "dist_mat": mirs.dist_mat,
            "sd_mat": mirs.sd_mat,
            "speed_lim_mat": mirs.speed_lim_mat,
            "process_T": mirs.process_T,
            "start_times": mirs.start_times,
            "T_upper_bound": mirs.T_upper_bound,
            "agent_weights": mirs.agent_weights,
            "wp_weights": mirs.wp_weights,
        },
    }

    data_file = scenario_dir / "scenario_data.pkl"
    with open(data_file, "wb") as f:
        pickle.dump(scenario_data, f)

    summary_file = scenario_dir / "scenario_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(_to_python_primitive({
            "scenario_id": scenario_data["scenario_id"],
            "n_waypoints": n_waypoints,
            "n_agents": n_agents,
            "seed": seed,
            "network_type": network_type,
            "tol_range": list(tol_range),
            "cost_mode": cost_mode,
            "lm": lm,
            "cbf_mode_name": cbf_mode_name,
            "data_file": str(data_file.name),
        }), f, indent=2)

    network_plot_path = scenario_dir / "network_plot_generated.png"
    visualize.plotNetwork(
        figuresize=(10, 8),
        wp_xy=np.asarray(mirs.wp_locations),
        mask=np.asarray(mirs.mask),
        dist_mat=np.asarray(mirs.dist_mat),
        sd_mat=np.asarray(mirs.sd_mat),
        routes=[],
        schedules=[],
        agent_colors={},
        showEdgeLength=False,
        save_path=str(network_plot_path),
        show_plot=False,
    )

    return scenario_data


def generate_scenarios(
    output_root: Path,
    problem_sizes: List[Tuple[int, int]],
    seeds_per_size: int,
    seed: int,
    tol_range: Tuple[float, float],
    network_type: str,
    cost_mode: str,
    lm: float,
    cbf_mode_name: str,
    offset_energy: int,
    self_hop: int,
    filter_wp_thresh: float,
    prune_mode: bool,
    print_flag: bool,
    T_upper_bound: float = 10000,
) -> List[Path]:
    """Generate independent seed realizations for each requested problem size.

    Scenario directories are grouped as ``nwp<N>_na<A>/scenario_<i>_seed<S>``.
    Thus every (N, A) pair receives exactly ``seeds_per_size`` scenarios,
    irrespective of the selected network type.
    """
    if not problem_sizes:
        raise ValueError("At least one problem size is required.")
    if seeds_per_size <= 0:
        raise ValueError("seeds_per_size must be positive.")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    scenario_paths: List[Path] = []

    for n_waypoints, n_agents in problem_sizes:
        for scenario_index in range(1, seeds_per_size + 1):
            scenario_seed = int(rng.integers(1, 10_000_000))
            generate_single_scenario(
                root_dir=output_root,
                scenario_index=scenario_index,
                n_waypoints=n_waypoints,
                n_agents=n_agents,
                seed=scenario_seed,
                tol_range=tol_range,
                network_type=network_type,
                cost_mode=cost_mode,
                lm=lm,
                cbf_mode_name=cbf_mode_name,
                offset_energy=offset_energy,
                self_hop=self_hop,
                filter_wp_thresh=filter_wp_thresh,
                prune_mode=prune_mode,
                print_flag=print_flag,
                T_upper_bound=T_upper_bound,
            )
            scenario_paths.append(
                output_root
                / _problem_size_folder(n_waypoints, n_agents)
                / _scenario_folder_name(n_waypoints, n_agents, scenario_seed, scenario_index)
            )

    manifest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root_dir": str(output_root),
        "seeds_per_size": seeds_per_size,
        "n_problem_sizes": len(problem_sizes),
        "n_scenarios": len(scenario_paths),
        "scenarios": [
            {
                "folder": scenario_dir.name,
                "data_file": "scenario_data.pkl",
                "n_waypoints": int(scenario_dir.parent.name.split("_")[0].removeprefix("nwp")),
                "n_agents": int(scenario_dir.parent.name.split("_")[1].removeprefix("na")),
                "seed": int(scenario_dir.name.split("_seed")[1]),
                "network_type": network_type,
            }
            for scenario_dir in scenario_paths
        ],
    }

    with open(output_root / "scenario_manifest.json", "w", encoding="utf-8") as f:
        json.dump(_to_python_primitive(manifest), f, indent=2)

    return scenario_paths


def load_scenario_data(scenario_dir: Path | str) -> Dict[str, Any]:
    scenario_path = Path(scenario_dir)
    data_file = scenario_path / "scenario_data.pkl"
    if not data_file.exists():
        raise FileNotFoundError(f"Scenario data file not found: {data_file}")

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate scenario folders for MIRS benchmark runs. "
            "Each scenario stores constructor data and optimizer metadata so later "
            "solvers can be launched independently."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SCENARIO_ROOT,
        help=f"Folder where all scenario directories will be written. Default: {DEFAULT_SCENARIO_ROOT}",
    )
    parser.add_argument(
        "--problem-sizes",
        type=_parse_problem_size,
        nargs="+",
        required=True,
        metavar="NWP:NA",
        help="Problem sizes to generate, e.g. --problem-sizes 5:2 10:3 15:5.",
    )
    parser.add_argument(
        "--seeds-per-size",
        type=int,
        default=10,
        help="Independent seeded scenarios generated for each NWP:NA pair. Default: 10.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base seed for scenario generation.")
    parser.add_argument("--tol-range", type=float, nargs=2, default=[5.0, 5.0], help="Tolerance range as min max.")
    parser.add_argument("--network-type", choices=["grid", "ring", "random", "multigraph", "multi"], default="grid",
                        help="Problem network type.")
    parser.add_argument("--cost-mode", choices=["sum", "slowest"], default="sum", help="Cost mode for the MIRS objective.")
    parser.add_argument("--lm", type=float, default=1.0, help="Slowest-agent weighting parameter.")
    parser.add_argument("--cbf-mode-name", choices=["rect", "el", "lin_static"], default="rect", help="CBF mode name.")
    parser.add_argument("--offset-energy", type=int, default=1, help="MIRS offset_energy value.")
    parser.add_argument("--self-hop", type=int, default=0, help="MIRS selfHop value.")
    parser.add_argument("--filter-wp-thresh", type=float, default=1e-4, help="Waypoint pruning threshold.")
    parser.add_argument("--prune-mode", action="store_true", help="Enable waypoint pruning mode.")
    parser.add_argument("--print-flag", action="store_true", help="Enable verbose problem initialization output.")
    parser.add_argument("--t-upper-bound", type=float, default=2000.0, help="Upper bound for waypoint schedule times.")

    args = parser.parse_args()

    if args.seeds_per_size <= 0:
        parser.error("--seeds-per-size must be > 0")
    if len(set(args.problem_sizes)) != len(args.problem_sizes):
        parser.error("--problem-sizes must not contain duplicate NWP:NA pairs")

    return args


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root

    generated_dirs = generate_scenarios(
        output_root=output_root,
        problem_sizes=args.problem_sizes,
        seeds_per_size=args.seeds_per_size,
        seed=args.seed,
        tol_range=(float(args.tol_range[0]), float(args.tol_range[1])),
        network_type=args.network_type,
        cost_mode=args.cost_mode,
        lm=args.lm,
        cbf_mode_name=args.cbf_mode_name,
        offset_energy=args.offset_energy,
        self_hop=args.self_hop,
        filter_wp_thresh=args.filter_wp_thresh,
        prune_mode=args.prune_mode,
        print_flag=args.print_flag,
        T_upper_bound=args.t_upper_bound,
    )

    print(
        f"Generated {len(generated_dirs)} scenarios across "
        f"{len(args.problem_sizes)} problem sizes in {output_root}"
    )
    for d in generated_dirs:
        print(f"- {d.name}: scenario_data.pkl")


if __name__ == "__main__":
    main()
