"""
Generate perturbed copies of an existing scenario for sensitivity analysis.

Given a scenario folder (containing scenario_data.pkl), this script produces one or
more new scenario folders that are identical to the source scenario except for a
single edge length l_ij = dist_mat[i, j] (and its symmetric counterpart dist_mat[j, i]),
which is overridden to a new value. All other scenario parameters (waypoint layout,
mask, agent start/destination pairs, speed limits, cost coefficients, optimizer
configs, etc.) are left untouched, so the perturbed scenarios can be solved and
plotted with the exact same main.py workflow as any other scenario.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import utils

DEFAULT_OUTPUT_ROOT = Path("perturbed_scenarios")


def load_scenario_data(scenario_dir: Path) -> Dict[str, Any]:
    data_file = Path(scenario_dir) / "scenario_data.pkl"
    if not data_file.exists():
        raise FileNotFoundError(f"Scenario data file not found: {data_file}")
    with open(data_file, "rb") as f:
        return pickle.load(f)


def _to_python_primitive(value: Any) -> Any:
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
    if isinstance(value, (list, tuple)):
        return [_to_python_primitive(v) for v in value]
    return value


def resolve_new_length(
    original_length: float,
    new_length: float | None = None,
    delta: float | None = None,
    delta_pct: float | None = None,
) -> float:
    given = [v is not None for v in (new_length, delta, delta_pct)]
    if sum(given) != 1:
        raise ValueError("Exactly one of new_length, delta, or delta_pct must be provided.")
    if new_length is not None:
        return float(new_length)
    if delta is not None:
        return float(original_length + delta)
    return float(original_length * (1.0 + delta_pct))


def make_perturbed_scenario_data(
    scenario_data: Dict[str, Any],
    i: int,
    j: int,
    new_length: float,
) -> Dict[str, Any]:
    mask = np.asarray(scenario_data["mirs_summary"]["mask"])
    if i == j:
        raise ValueError("Edge (i, j) must connect two distinct waypoints.")
    if mask[i, j] == 0:
        raise ValueError(f"Edge ({i}, {j}) does not exist in this scenario's network mask.")

    dist_mat = np.asarray(scenario_data["mirs_summary"]["dist_mat"])
    original_length = float(dist_mat[i, j])

    perturbed = copy.deepcopy(scenario_data)

    # dist_mat_overrides is applied on top of the deterministic (wp_params, seed) reconstruction
    overrides = dict(perturbed.get("dist_mat_overrides") or {})
    overrides[(i, j)] = new_length
    overrides[(j, i)] = new_length
    perturbed["dist_mat_overrides"] = overrides

    perturbed_dist_mat = np.array(dist_mat, copy=True)
    perturbed_dist_mat[i, j] = new_length
    perturbed_dist_mat[j, i] = new_length
    perturbed["mirs_summary"]["dist_mat"] = perturbed_dist_mat

    perturbed["perturbation"] = {
        "source_scenario_dir": scenario_data.get("scenario_dir", scenario_data.get("scenario_id")),
        "source_scenario_id": scenario_data.get("scenario_id"),
        "edge": (i, j),
        "original_length": original_length,
        "new_length": new_length,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return perturbed


def _perturbed_folder_name(base_scenario_id: str, i: int, j: int, new_length: float) -> str:
    tag = f"{new_length:.3f}".replace(".", "p").replace("-", "neg")
    return f"{base_scenario_id}_l{i}_{j}_{tag}"


def generate_perturbed_scenario(
    scenario_dir: Path,
    output_root: Path,
    i: int,
    j: int,
    new_length: float | None = None,
    delta: float | None = None,
    delta_pct: float | None = None,
) -> Path:
    scenario_data = load_scenario_data(scenario_dir)
    original_length = float(np.asarray(scenario_data["mirs_summary"]["dist_mat"])[i, j])
    resolved_length = resolve_new_length(original_length, new_length, delta, delta_pct)

    perturbed_data = make_perturbed_scenario_data(scenario_data, i, j, resolved_length)

    base_scenario_id = scenario_data.get("scenario_id", Path(scenario_dir).name)
    folder_name = _perturbed_folder_name(base_scenario_id, i, j, resolved_length)
    perturbed_dir = Path(output_root) / folder_name
    perturbed_dir.mkdir(parents=True, exist_ok=True)
    perturbed_data["scenario_id"] = folder_name

    with open(perturbed_dir / "scenario_data.pkl", "wb") as f:
        pickle.dump(perturbed_data, f)

    with open(perturbed_dir / "perturbation_summary.json", "w", encoding="utf-8") as f:
        json.dump(_to_python_primitive(perturbed_data["perturbation"]), f, indent=2)

    return perturbed_dir


def generate_edge_length_sweep(
    scenario_dir: Path,
    output_root: Path,
    i: int,
    j: int,
    delta_pcts: List[float],
) -> List[Path]:
    scenario_data = load_scenario_data(scenario_dir)
    original_length = float(np.asarray(scenario_data["mirs_summary"]["dist_mat"])[i, j])

    perturbed_dirs = []
    for delta_pct in delta_pcts:
        perturbed_dirs.append(
            generate_perturbed_scenario(scenario_dir, output_root, i, j, delta_pct=delta_pct)
        )

    manifest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_scenario_dir": str(scenario_dir),
        "edge": [i, j],
        "original_length": original_length,
        "delta_pcts": delta_pcts,
        "scenarios": [str(d) for d in perturbed_dirs],
    }
    with open(Path(output_root) / f"sensitivity_manifest_l{i}_{j}.json", "w", encoding="utf-8") as f:
        json.dump(_to_python_primitive(manifest), f, indent=2)

    return perturbed_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or more perturbed scenarios (single edge length change) for sensitivity analysis."
    )
    parser.add_argument("--scenario-dir", type=Path, required=True, help="Source scenario folder (contains scenario_data.pkl).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help=f"Folder to write perturbed scenarios. Default: {DEFAULT_OUTPUT_ROOT}")
    parser.add_argument("--edge", type=int, nargs=2, required=True, metavar=("I", "J"), help="Waypoint indices (i, j) of the edge to perturb.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--new-lengths", type=float, nargs="+", help="One or more absolute new lengths for l_ij.")
    group.add_argument("--deltas", type=float, nargs="+", help="One or more additive changes to apply to l_ij.")
    group.add_argument("--delta-pcts", type=float, nargs="+", help="One or more fractional changes to apply to l_ij (e.g. -0.2 for -20%%).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    i, j = args.edge

    if args.new_lengths is not None:
        values, kind = args.new_lengths, "new_length"
    elif args.deltas is not None:
        values, kind = args.deltas, "delta"
    else:
        values, kind = args.delta_pcts, "delta_pct"

    perturbed_dirs = []
    for value in values:
        perturbed_dirs.append(
            generate_perturbed_scenario(args.scenario_dir, args.output_root, i, j, **{kind: value})
        )

    for d in perturbed_dirs:
        print(f"Wrote perturbed scenario: {d}")


if __name__ == "__main__":
    main()
