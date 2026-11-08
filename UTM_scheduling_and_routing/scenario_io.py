import pickle
from pathlib import Path
from typing import Any, Dict, List


SUPPORTED_METHODS = (
    "cbf",
    "mep_cbf_pruned",
    "slsqp",
    "cbf_static",
    "slsqp_static",
    "gurobi",
    "cbs",
    "seq",
)
METHOD_LABELS = {
    "cbf": "MEP-CBF",
    "mep_cbf_pruned": "MEP-CBF-P",
    "slsqp": "MEP-SLSQP",
    "cbf_static": "MEP-CBF Static",
    "slsqp_static": "MEP-SLSQP Static",
    "gurobi": "Gurobi",
    "cbs": "CBS",
    "seq": "SEQ-SPS",
}


def method_display_name(method: str) -> str:
    return METHOD_LABELS.get(method, method.upper())


def load_scenario_data(scenario_dir: Path) -> Dict[str, Any]:
    data_file = scenario_dir / "scenario_data.pkl"
    if not data_file.exists():
        raise FileNotFoundError(f"Scenario data file not found: {data_file}")

    with open(data_file, "rb") as f:
        return pickle.load(f)


def list_scenario_dirs(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Scenario root does not exist: {root_dir}")

    scenario_dirs = sorted(
        data_file.parent
        for data_file in root_dir.rglob("scenario_data.pkl")
        if data_file.is_file()
    )
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario directories found under {root_dir}")
    return scenario_dirs


def output_filename_for_method(method: str) -> str:
    method = method.lower()
    mapping = {
        "cbf": "solution_cbf.pkl",
        "mep_cbf_pruned": "solution_mep_cbf_pruned.pkl",
        "slsqp": "solution_slsqp.pkl",
        "cbf_static": "solution_cbf_static.pkl",
        "slsqp_static": "solution_slsqp_static.pkl",
        "gurobi": "solution_gurobi.pkl",
        "cbs": "solution_cbs.pkl",
        "seq": "solution_seq.pkl",
    }
    if method not in mapping:
        raise ValueError(f"Unsupported method: {method}")
    return mapping[method]
