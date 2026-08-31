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


DEFAULT_SCENARIO_ROOT = Path("generated_scenarios")


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
    return f"scenario_{index:03d}_nwp{n_waypoints}_na{n_agents}_seed{seed}"


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
    filter_wp_thresh: float = 1e-10,
    prune_mode: bool = False,
    print_flag: bool = False,
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

    return {
        "cbf_mep": {
            "config": cbf_config,
            "anneal_config": anneal_config,
            "entry_point": "mep_opt.MIRSOptimizer",
            "method_name": "CBF_CLF_at_beta",
            "note": "Uses the CBF-CLF fixed-beta loop from mep_opt.py.",
        },
        "slsqp_mep": {
            "config": slsqp_config,
            "anneal_config": anneal_config,
            "entry_point": "mep_opt.MIRSOptimizer",
            "method_name": "slsqp_at_beta",
            "note": "Uses the SLSQP fixed-beta optimizer from mep_opt.py.",
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
    )

    scenario_dir = root_dir / _scenario_folder_name(
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
    n_scenarios: int,
    min_waypoints: int,
    max_waypoints: int,
    min_agents: int,
    max_agents: int,
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
) -> List[Path]:
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive.")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    scenario_paths: List[Path] = []

    for i in range(n_scenarios):
        n_waypoints = int(rng.integers(min_waypoints, max_waypoints + 1))
        n_agents = int(rng.integers(min_agents, min(max_agents, n_waypoints) + 1))
        scenario_seed = int(rng.integers(1, 10_000_000))

        generate_single_scenario(
            root_dir=output_root,
            scenario_index=i + 1,
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
        )

        scenario_dir = output_root / _scenario_folder_name(
            n_waypoints=n_waypoints,
            n_agents=n_agents,
            seed=scenario_seed,
            index=i + 1,
        )
        scenario_paths.append(scenario_dir)

    manifest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root_dir": str(output_root),
        "n_scenarios": n_scenarios,
        "scenarios": [
            {
                "folder": scenario_dir.name,
                "data_file": "scenario_data.pkl",
                "n_waypoints": int((scenario_dir.name.split("_nwp")[1].split("_na")[0])),
                "n_agents": int((scenario_dir.name.split("_na")[1].split("_seed")[0])),
                "seed": int(scenario_dir.name.split("_seed")[1]),
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
    parser.add_argument("--n-scenarios", type=int, default=5, help="Number of scenarios to generate.")
    parser.add_argument("--min-waypoints", type=int, default=5, help="Minimum waypoint count per scenario.")
    parser.add_argument("--max-waypoints", type=int, default=15, help="Maximum waypoint count per scenario.")
    parser.add_argument("--min-agents", type=int, default=2, help="Minimum agent count per scenario.")
    parser.add_argument("--max-agents", type=int, default=5, help="Maximum agent count per scenario.")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for scenario generation.")
    parser.add_argument("--tol-range", type=float, nargs=2, default=[5.0, 5.0], help="Tolerance range as min max.")
    parser.add_argument("--network-type", choices=["grid", "ring", "random", "multigraph", "multi"], default="grid",
                        help="Problem network type.")
    parser.add_argument("--cost-mode", choices=["sum", "slowest"], default="sum", help="Cost mode for the MIRS objective.")
    parser.add_argument("--lm", type=float, default=1.0, help="Slowest-agent weighting parameter.")
    parser.add_argument("--cbf-mode-name", choices=["rect", "el", "lin"], default="rect", help="CBF mode name.")
    parser.add_argument("--offset-energy", type=int, default=1, help="MIRS offset_energy value.")
    parser.add_argument("--self-hop", type=int, default=0, help="MIRS selfHop value.")
    parser.add_argument("--filter-wp-thresh", type=float, default=1e-10, help="Waypoint pruning threshold.")
    parser.add_argument("--prune-mode", action="store_true", help="Enable waypoint pruning mode.")
    parser.add_argument("--print-flag", action="store_true", help="Enable verbose problem initialization output.")

    args = parser.parse_args()

    if args.min_waypoints <= 0:
        parser.error("--min-waypoints must be > 0")
    if args.max_waypoints < args.min_waypoints:
        parser.error("--max-waypoints must be >= --min-waypoints")
    if args.min_agents <= 0:
        parser.error("--min-agents must be > 0")
    if args.max_agents < args.min_agents:
        parser.error("--max-agents must be >= --min-agents")
    if args.min_agents > args.max_waypoints:
        parser.error("--min-agents cannot exceed --max-waypoints")
    if args.max_agents > args.max_waypoints:
        args.max_agents = min(args.max_agents, args.max_waypoints)

    return args


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root

    generated_dirs = generate_scenarios(
        output_root=output_root,
        n_scenarios=args.n_scenarios,
        min_waypoints=args.min_waypoints,
        max_waypoints=args.max_waypoints,
        min_agents=args.min_agents,
        max_agents=args.max_agents,
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
    )

    print(f"Generated {len(generated_dirs)} scenarios in {output_root}")
    for d in generated_dirs:
        print(f"- {d.name}: scenario_data.pkl")


if __name__ == "__main__":
    main()
