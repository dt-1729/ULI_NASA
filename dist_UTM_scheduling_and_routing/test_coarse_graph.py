import numpy as np
import matplotlib.pyplot as plt

from cli_args import test_args
from scenario_generator import load_scenario
from dist_MIRS import heir_MIRS
from visualization import plot_original, plot_coarse

args = test_args()
folder = args.scenario_folder
h = heir_MIRS(load_scenario(folder), args.boundary_fraction)
adj = np.triu(h.mask > 0, 1)
cross = adj & (h.partition[:, None] != h.partition[None, :])
print(f"edges: {adj.sum()} total, {cross.sum()} inter-subgraph; "
      f"boundary nodes: {len(h.boundary_nodes())}/{h.n_waypoints}; "
      f"coarse graph: {len(h.coarse_nodes)} nodes "
      f"({h.coarse_node_counts()[0]} boundary + {h.coarse_node_counts()[1]} entry/exit-only), "
      f"{int(np.triu(h.coarse_mask, 1).sum())} edges")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
plot_original(axes[0], h)
plot_coarse(axes[1], h)
fig.tight_layout()
fig.savefig(f"{folder}/graphs.png", dpi=150)
print(f"Saved {folder}/graphs.png")
