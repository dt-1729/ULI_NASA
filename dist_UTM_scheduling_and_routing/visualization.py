import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


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
    n_boundary, n_entry_exit, n_total = h.coarse_node_counts()
    ax.set_title(
        f"Coarse graph: {n_total} nodes ({n_boundary} boundary + {n_entry_exit} entry/exit-only), "
        f"{int(np.triu(mask, 1).sum())} edges"
    )
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


def plot_solution_schedules(
    routes, route_schedules, sd_mat, save_path, dpi=200, agent_labels=None
):
    """Plot each agent's scheduled visits in chronological order."""
    visits_by_agent = _scheduled_agent_visits(routes, route_schedules, sd_mat)
    n_agents = len(visits_by_agent)
    labels = list(range(n_agents)) if agent_labels is None else [int(a) for a in agent_labels]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.65 * n_agents + 2)))
    cmap = plt.get_cmap("tab20")

    for agent_id, (waypoints, times) in enumerate(visits_by_agent):
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
    ax.set_yticklabels([f"Agent {i}" for i in labels])
    ax.set_xlabel("Schedule time")
    ax.set_ylabel("Agent")
    ax.set_title("Agent waypoint schedules (waypoint IDs shown at visits)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_ylim(-0.75, max(n_agents - 0.25, 0.75))
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _scheduled_agent_visits(routes, route_schedules, sd_mat):
    if len(routes) != len(route_schedules):
        raise ValueError(
            "routes and route_schedules must contain the same number of agents."
        )

    starts = np.asarray(sd_mat)
    if starts.ndim != 2 or starts.shape[1] < 1 or starts.shape[0] < len(routes):
        raise ValueError("sd_mat must contain a start waypoint for every agent.")

    visits_by_agent = []
    for agent_id, (route, schedule) in enumerate(zip(routes, route_schedules)):
        waypoints = [int(starts[agent_id, 0]), *map(int, route)]
        schedule_times = np.asarray(schedule, dtype=float)
        if schedule_times.ndim != 1:
            raise ValueError(f"Agent {agent_id} schedule must be one-dimensional.")
        times = schedule_times[1 : len(waypoints) + 1]
        if len(times) != len(waypoints):
            raise ValueError(
                f"Agent {agent_id} schedule does not align with its route."
            )
        if not np.all(np.isfinite(times)):
            raise ValueError(
                f"Agent {agent_id} schedule contains non-finite visit times."
            )
        visits_by_agent.append((waypoints, times))
    return visits_by_agent


def plot_solution_waypoint_schedules(
    routes, route_schedules, sd_mat, tol_array, save_path, dpi=200, agent_labels=None
):
    """Plot arrivals by waypoint, coloring gaps by inter-agent safety margin."""
    visits_by_agent = _scheduled_agent_visits(routes, route_schedules, sd_mat)
    tolerances = np.asarray(tol_array, dtype=float)
    if tolerances.ndim != 1 or not np.all(np.isfinite(tolerances)):
        raise ValueError("tol_array must be a one-dimensional array of finite values.")
    if np.any(tolerances < 0):
        raise ValueError("tol_array cannot contain negative tolerances.")

    visits_by_waypoint = defaultdict(list)
    for agent_id, (waypoints, times) in enumerate(visits_by_agent):
        for waypoint, time in zip(waypoints, times):
            if waypoint < 0 or waypoint >= len(tolerances):
                raise ValueError(
                    f"Waypoint {waypoint} is outside tol_array of length {len(tolerances)}."
                )
            visits_by_waypoint[waypoint].append((float(time), agent_id))

    waypoint_ids = sorted(visits_by_waypoint)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * len(waypoint_ids) + 2)))
    agent_cmap = plt.get_cmap("tab20")
    safety_cmap = plt.get_cmap("RdYlGn")

    for row, waypoint in enumerate(waypoint_ids):
        visits = sorted(visits_by_waypoint[waypoint])
        times = np.asarray([time for time, _ in visits], dtype=float)
        tolerance = float(tolerances[waypoint])
        for (start, _), (end, _) in zip(visits, visits[1:]):
            gap = end - start
            safety = (
                np.clip(gap / (2.0 * tolerance), 0.0, 1.0)
                if tolerance > 0
                else float(gap > 0)
            )
            ax.barh(
                row,
                width=gap,
                left=start,
                height=0.55,
                color=safety_cmap(safety),
                alpha=0.55,
                linewidth=0,
                zorder=1,
            )

        label_offset = (
            max(0.1, 0.012 * float(np.ptp(times))) if len(times) > 1 else 0.1
        )
        for time, agent_id in visits:
            ax.vlines(
                time,
                row - 0.275,
                row + 0.275,
                color="white",
                linewidth=2,
                zorder=3,
            )
            ax.scatter(
                time,
                row,
                color=agent_cmap(agent_id % 20),
                edgecolors="black",
                linewidths=0.5,
                s=30,
                zorder=4,
            )
            ax.text(
                time - label_offset,
                row,
                str(agent_id if agent_labels is None else int(agent_labels[agent_id])),
                ha="right",
                va="center",
                fontsize=8,
                zorder=5,
            )

    ax.set_yticks(np.arange(len(waypoint_ids)))
    ax.set_yticklabels([str(waypoint) for waypoint in waypoint_ids])
    ax.set_xlabel("Schedule time")
    ax.set_ylabel("Waypoint")
    ax.set_title("Waypoint arrival schedules (agent IDs shown at visits)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_ylim(-0.75, max(len(waypoint_ids) - 0.25, 0.75))
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_subgraph_visits(visits, save_dir, dpi=200, prefix="subgraph_visits"):
    """Save one timeline per subgraph from heir_MIRS.subgraph_agent_visits output.

    Each row is an agent; each bar spans a visit from entry to exit time and is
    labeled with its entry and exit waypoints.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20")
    paths = []
    for subgraph, subgraph_visits in sorted(visits.items()):
        agents = sorted({v["agent"] for v in subgraph_visits})
        rows = {agent: i for i, agent in enumerate(agents)}
        fig, ax = plt.subplots(figsize=(12, max(3, 0.65 * len(agents) + 2)))
        for v in subgraph_visits:
            row = rows[v["agent"]]
            color = cmap(v["agent"] % 20)
            width = v["exit_time"] - v["entry_time"]
            ax.barh(row, width, left=v["entry_time"], height=0.5, color=color,
                    alpha=0.55, edgecolor="black", linewidth=0.8, zorder=2)
            ax.vlines([v["entry_time"], v["exit_time"]], row - 0.25, row + 0.25,
                      color="white", linewidth=2, zorder=3)
            ax.annotate(f"in {v['entry_node']}", (v["entry_time"], row), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=8, color=color)
            ax.annotate(f"out {v['exit_node']}", (v["exit_time"], row), xytext=(0, -18),
                        textcoords="offset points", ha="center", fontsize=8, color=color)
        ax.set_yticks(np.arange(len(agents)))
        ax.set_yticklabels([f"Agent {a}" for a in agents])
        ax.set_ylim(-0.75, max(len(agents) - 0.25, 0.75))
        ax.set_xlabel("Schedule time")
        ax.set_ylabel("Agent")
        ax.set_title(f"Subgraph {subgraph}: agent entry/exit (waypoint IDs and times)")
        if not agents:
            ax.text(0.5, 0.5, "No agents pass through this subgraph",
                    transform=ax.transAxes, ha="center", va="center")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()
        path = save_dir / f"{prefix}_{subgraph}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_subgraph_solution(result, wp_locations, tol_array, save_dir, dpi=200):
    """Plot one solved subgraph (network, agent schedules, waypoint schedules) in save_dir.

    result is one entry of heir_MIRS.solve_subgraphs; ids in the plots are global.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mirs = result["mirs"]
    nodes = np.asarray(result["node_ids"], dtype=int)
    agent_ids = [int(a) for a in result["agent_ids"]]
    # fine routes start with the entry node, which the plot helpers add themselves
    routes = [list(route[1:]) for route in result["routes"]]
    schedules = result["route_schedules"]
    global_sd = nodes[mirs.sd_mat]
    locations = np.asarray(wp_locations, dtype=float)[nodes]

    plot_solution_schedules(
        routes, schedules, global_sd, save_dir / "agent_schedule_plot.png", dpi, agent_ids
    )
    plot_solution_waypoint_schedules(
        routes, schedules, global_sd, tol_array,
        save_dir / "schedule_plot.png", dpi, agent_ids,
    )

    local_of = {int(g): i for i, g in enumerate(nodes)}
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_aspect("equal", adjustable="datalim")
    cmap = plt.get_cmap("tab20")
    for i, j in zip(*np.where(np.triu(np.asarray(mirs.mask) > 0, 1))):
        ax.plot(*zip(locations[i], locations[j]), color="lightgray", linewidth=1.2, zorder=1)
    ax.scatter(*locations.T, s=90, color="white", edgecolors="black", zorder=3)
    for local, (x, y) in enumerate(locations):
        ax.annotate(str(nodes[local]), (x, y), fontsize=7, ha="center", va="center", zorder=5)

    traversals = defaultdict(list)
    for k, (agent, route, schedule) in enumerate(zip(agent_ids, routes, schedules)):
        path = [int(global_sd[k, 0]), *map(int, route)]
        times = np.asarray(schedule, dtype=float)
        for n, (a, b) in enumerate(zip(path, path[1:])):
            if a != b:
                traversals[(local_of[a], local_of[b])].append((agent, times[n + 1], times[n + 2]))
    for (a, b), items in traversals.items():
        direction = locations[b] - locations[a]
        length = np.linalg.norm(direction)
        if length <= 1e-12:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        for order, (agent, t0, t1) in enumerate(items):
            offset = (order - (len(items) - 1) / 2) * 0.02 * length
            start, end = locations[a] + normal * offset, locations[b] + normal * offset
            color = cmap(agent % 20)
            ax.annotate(
                "", xy=end, xytext=start, zorder=4,
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 2.2,
                            "shrinkA": 8, "shrinkB": 8},
            )
            ax.annotate(
                f"A{agent}: {t0:.1f}-{t1:.1f}", (start + end) / 2, fontsize=7, color=color,
                ha="center", va="bottom", zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
            )
    ax.set_title("Subgraph routes and traversal schedules")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(save_dir / "network_plot.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [save_dir / n for n in ("network_plot.png", "schedule_plot.png", "agent_schedule_plot.png")]


def plot_full_solution_network(h, routes, route_schedules, save_path, dpi=200):
    """Plot the combined routes and traversal times over the entire fine graph."""
    locations = np.asarray(h.wp_locations, dtype=float)
    mask = np.asarray(h.mask) > 0
    partition = np.asarray(h.partition)
    part_cmap = plt.get_cmap("tab10")
    agent_cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal", adjustable="datalim")

    for i, j in zip(*np.where(np.triu(mask, 1))):
        crossing = partition[i] != partition[j]
        ax.plot(
            *zip(locations[i], locations[j]),
            color="red" if crossing else "lightgray",
            linewidth=0.8 if crossing else 1.0,
            alpha=0.35 if crossing else 1.0,
            zorder=1,
        )
    ax.scatter(
        *locations.T, s=70, c=[part_cmap(int(p) % 10) for p in partition],
        edgecolors="black", linewidths=0.6, zorder=3,
    )
    for node, (x, y) in enumerate(locations):
        ax.annotate(str(node), (x, y), fontsize=6, ha="center", va="center", zorder=5)

    traversals = defaultdict(list)
    for agent, (route, schedule) in enumerate(zip(routes, route_schedules)):
        path = [int(h.sd_mat[agent, 0]), *map(int, route)]
        times = np.asarray(schedule, dtype=float)
        for n, (a, b) in enumerate(zip(path, path[1:])):
            if a != b:
                traversals[(a, b)].append((agent, times[n + 1], times[n + 2]))
    for (a, b), items in traversals.items():
        direction = locations[b] - locations[a]
        length = np.linalg.norm(direction)
        if length <= 1e-12:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        for order, (agent, t0, t1) in enumerate(items):
            offset = (order - (len(items) - 1) / 2) * 0.02 * length
            start, end = locations[a] + normal * offset, locations[b] + normal * offset
            color = agent_cmap(agent % 20)
            ax.annotate(
                "", xy=end, xytext=start, zorder=4,
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 2.0,
                            "shrinkA": 6, "shrinkB": 6},
            )
            ax.annotate(
                f"A{agent}: {t0:.0f}-{t1:.0f}", (start + end) / 2, fontsize=6, color=color,
                ha="center", va="bottom", zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
            )
    for agent in range(len(routes)):
        ax.plot([], [], color=agent_cmap(agent % 20), linewidth=2.5, label=f"Agent {agent}")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Combined solution on the entire graph (node color = subgraph)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
