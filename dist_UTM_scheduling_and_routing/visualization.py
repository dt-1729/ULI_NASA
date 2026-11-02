import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_original(ax, h):
    loc, mask, partition = h.wp_locations, h.mask > 0, h.partition
    cmap = plt.get_cmap("tab10")
    is_bnd = np.zeros(len(loc), bool)
    is_bnd[h.boundary_nodes()] = True
    selected = {(min(i, j), max(i, j)) for i, j in h.selected_edges}
    for i, j in zip(*np.where(np.triu(mask, 1))):
        if partition[i] == partition[j]:
            style = dict(color="lightgray", lw=1)
        elif (i, j) in selected:
            style = dict(color="red", lw=2)
        else:
            style = dict(color="red", lw=1, ls=":", alpha=0.5)
        ax.plot(*zip(loc[i], loc[j]), zorder=1, **style)
    ax.scatter(*loc.T, c=[cmap(p % 10) for p in partition], s=60, zorder=2)
    ax.scatter(*loc[is_bnd].T, s=160, facecolors="none", edgecolors="k", lw=1.5, zorder=3)
    for i, p in enumerate(loc):
        ax.annotate(str(i), p, fontsize=7, ha="center", va="center", zorder=4)
    ax.plot([], [], color="red", label="selected inter-partition edge")
    ax.plot([], [], color="red", ls=":", alpha=0.5, label="unselected inter-partition edge")
    ax.scatter([], [], s=160, facecolors="none", edgecolors="k", label="boundary node")
    ax.set_title(f"Original graph: {len(loc)} nodes, {int(np.triu(mask, 1).sum())} edges, "
                 f"{len(np.unique(partition))} subgraphs (color = partition)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_aspect("equal")


def plot_coarse(ax, h):
    loc, mask = h.coarse_wp_locations, h.coarse_mask > 0
    cmap = plt.get_cmap("tab10")
    for a, b in zip(*np.where(np.triu(mask, 1))):
        ax.plot(*zip(loc[a], loc[b]), color="gray", lw=1, zorder=1)
    colors = [cmap(h.partition[f] % 10) for f in h.coarse_nodes]
    ax.scatter(*loc.T, c=colors, s=80, zorder=2)
    agent_nodes = set(h.coarse_sd_mat.ravel().tolist())
    ax.scatter(*loc[sorted(agent_nodes)].T, marker="*", s=200, c="gold", edgecolors="k", zorder=3,
               label="agent entry/exit")
    for c, f in enumerate(h.coarse_nodes):
        ax.annotate(str(f), loc[c], fontsize=7, ha="center", va="center", zorder=4)
    ax.set_title(f"Coarse graph: {len(h.coarse_nodes)} nodes, {int(np.triu(mask, 1).sum())} edges")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_aspect("equal")


def plot_solution_network(h, routes, route_schedules, save_path, dpi=200):
    """Plot agent routes and traversal times over the coarse graph."""
    locations = h.coarse_wp_locations
    coarse_mask = h.coarse_mask > 0
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_aspect("equal", adjustable="datalim")
    cmap = plt.get_cmap("tab20")

    for i, j in zip(*np.where(np.triu(coarse_mask, 1))):
        ax.plot(
            *zip(locations[i], locations[j]),
            color="lightgray",
            linewidth=1.5,
            zorder=1,
        )
        midpoint = (locations[i] + locations[j]) / 2
        ax.annotate(
            f"{h.coarse_dist[i, j]:.0f}",
            midpoint,
            fontsize=7,
            color="dimgray",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
            zorder=2,
        )

    for node, (x, y) in enumerate(locations):
        fine_node = int(h.coarse_nodes[node])
        partition = int(h.partition[fine_node])
        ax.scatter(
            x,
            y,
            s=110,
            color=cmap(partition % 20),
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
        ax.annotate(
            str(fine_node),
            (x, y),
            fontsize=8,
            ha="center",
            va="center",
            zorder=5,
        )

    traversals = defaultdict(list)
    starts = h.sd_mat[:, 0]
    for agent_id, (route, schedule) in enumerate(zip(routes, route_schedules)):
        path = [int(starts[agent_id]), *map(int, route)]
        times = np.asarray(schedule, dtype=float)
        if len(times) < len(path) + 1:
            raise ValueError(
                f"Agent {agent_id} schedule has {len(times)} entries; "
                f"route requires at least {len(path) + 1}."
            )
        for index, (source, destination) in enumerate(zip(path, path[1:])):
            if source == destination:
                continue
            source_coarse = h.fine_to_coarse[source]
            destination_coarse = h.fine_to_coarse[destination]
            traversals[(source_coarse, destination_coarse)].append(
                (agent_id, times[index + 1], times[index + 2])
            )

    for (source, destination), edge_traversals in traversals.items():
        start = locations[source]
        end = locations[destination]
        direction = end - start
        length = np.linalg.norm(direction)
        if length <= 1e-12:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        count = len(edge_traversals)
        for order, (agent_id, departure, arrival) in enumerate(edge_traversals):
            color = cmap(agent_id % 20)
            offset = (order - (count - 1) / 2) * 0.035
            arrow_start = start + normal * offset
            arrow_end = end + normal * offset
            ax.annotate(
                "",
                xy=arrow_end,
                xytext=arrow_start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 2.5,
                    "shrinkA": 10,
                    "shrinkB": 10,
                },
                zorder=3,
            )
            midpoint = (arrow_start + arrow_end) / 2
            ax.annotate(
                f"A{agent_id}: {departure:.1f}-{arrival:.1f}",
                midpoint,
                fontsize=7,
                color=color,
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
                zorder=6,
            )

    ax.set_title("Coarse network routes and traversal schedules")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_solution_schedules(routes, route_schedules, sd_mat, save_path, dpi=200):
    """Plot each agent's scheduled visits in chronological order."""
    n_agents = len(routes)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.65 * n_agents + 2)))
    cmap = plt.get_cmap("tab20")

    for agent_id, (route, schedule) in enumerate(zip(routes, route_schedules)):
        waypoints = [int(sd_mat[agent_id, 0]), *map(int, route)]
        times = np.asarray(schedule, dtype=float)[1 : len(waypoints) + 1]
        if len(times) != len(waypoints):
            raise ValueError(
                f"Agent {agent_id} schedule does not align with its route."
            )
        color = cmap(agent_id % 20)
        row = np.full(len(times), agent_id, dtype=float)
        ax.plot(times, row, color=color, alpha=0.55, linewidth=1.5, zorder=1)
        ax.scatter(times, row, color=color, edgecolors="black", s=45, zorder=2)
        for time, waypoint in zip(times, waypoints):
            ax.annotate(
                str(waypoint),
                (time, agent_id),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )

    ax.set_yticks(np.arange(n_agents))
    ax.set_yticklabels([f"Agent {i}" for i in range(n_agents)])
    ax.set_xlabel("Schedule time")
    ax.set_ylabel("Agent")
    ax.set_title("Agent waypoint schedules (waypoint IDs shown at visits)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_ylim(-0.75, max(n_agents - 0.25, 0.75))
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
