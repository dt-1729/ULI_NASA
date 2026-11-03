import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import shortest_path


class heir_MIRS:
    """Hierarchical wrapper: builds a coarse graph from graph parameters and a fixed partition.

    partition: int array of length n_waypoints, partition[i] = subgraph id of node i.
    Between each pair of connected subgraphs only the shortest `boundary_fraction` of the
    connecting edges (at least one) are selected; their endpoints are the boundary nodes.
    Coarse nodes: agent entry/exit nodes and boundary nodes.
    Coarse edges: direct edges are kept; unconnected nodes in the same subgraph are joined
    by an edge whose length is the shortest-path distance within that subgraph.
    """

    def __init__(self, graph, boundary_fraction=0.2):
        """graph: dict with wp_locations, mask, dist_mat, partition, sd_mat."""
        self.wp_locations = graph["wp_locations"]
        self.mask = graph["mask"]
        self.dist_mat = graph["dist_mat"]
        self.partition = np.asarray(graph["partition"])
        self.sd_mat = graph["sd_mat"]
        self.n_waypoints = len(self.wp_locations)
        self.boundary_fraction = boundary_fraction
        assert len(self.partition) == self.n_waypoints
        self.selected_edges = self.select_boundary_edges()
        self.build_coarse_graph()

    def select_boundary_edges(self):
        """Per pair of connected subgraphs, keep the shortest fraction of the edges joining them."""
        mask, dist, part = self.mask > 0, self.dist_mat, self.partition
        pair_edges = {}
        for i, j in zip(*np.where(np.triu(mask, 1) & (part[:, None] != part[None, :]))):
            pair_edges.setdefault((min(part[i], part[j]), max(part[i], part[j])), []).append((i, j))
        selected = []
        for edges in pair_edges.values():
            edges.sort(key=lambda e: dist[e])
            selected += edges[:max(1, int(np.ceil(self.boundary_fraction * len(edges))))]
        return selected

    def boundary_nodes(self):
        return np.unique(np.array(self.selected_edges, dtype=int).ravel())

    def coarse_node_counts(self):
        """(boundary, entry/exit only, total) node counts of the coarse graph.

        Entry/exit-only nodes are agent start/destination nodes that are not boundary nodes.
        """
        boundary = set(int(n) for n in self.boundary_nodes())
        entry_exit = set(int(n) for n in np.asarray(self.sd_mat).ravel())
        return len(boundary), len(entry_exit - boundary), len(boundary | entry_exit)

    def build_coarse_graph(self):
        mask = self.mask > 0
        dist = self.dist_mat
        part = self.partition
        selected = {(min(i, j), max(i, j)) for i, j in self.selected_edges}
        
        self.coarse_nodes = np.unique(np.concatenate([self.sd_mat.ravel(), self.boundary_nodes()]))
        print(f"boundary nodes: {self.boundary_nodes()}")
        n = len(self.coarse_nodes)
        self.fine_to_coarse = {int(f): c for c, f in enumerate(self.coarse_nodes)}

        # shortest paths restricted to edges inside a subgraph
        same = part[:, None] == part[None, :]
        w = np.where(mask & same, dist, 0.0)
        sp = shortest_path(w, directed=False)

        self.coarse_mask = np.zeros((n, n))
        self.coarse_dist = np.zeros((n, n))
        for a, i in enumerate(self.coarse_nodes):
            for b, j in enumerate(self.coarse_nodes):
                if a == b:
                    continue
                if mask[i, j]:
                    if part[i] != part[j] and (min(i, j), max(i, j)) not in selected:
                        continue  # unselected inter-subgraph edge
                    length = dist[i, j]
                elif part[i] == part[j] and np.isfinite(sp[i, j]):
                    length = sp[i, j]
                else:
                    continue
                self.coarse_mask[a, b] = 1
                self.coarse_dist[a, b] = length

        self.coarse_wp_locations = self.wp_locations[self.coarse_nodes]
        self.coarse_sd_mat = np.vectorize(self.fine_to_coarse.get)(self.sd_mat)

        # shared nodes: boundary nodes, common to the stage graph of every agent
        self.shared_nodes = np.array([self.fine_to_coarse[int(f)] for f in self.boundary_nodes()])

    def agent_graph(self, k):
        """Start links and stage graph for agent k.

        Stage nodes = shared nodes + the agent's destination (last). If the destination is itself a
        shared node, its shared copy is disconnected so the node is represented only once.
        """
        s, d = self.coarse_sd_mat[k]
        nodes = np.append(self.shared_nodes, d)
        dist_int = self.coarse_dist[np.ix_(nodes, nodes)]
        mask_int = self.coarse_mask[np.ix_(nodes, nodes)].copy()
        mask_start = self.coarse_mask[s, nodes].copy()
        dup = np.where(self.shared_nodes == d)[0]
        mask_int[dup, :] = 0
        mask_int[:, dup] = 0
        mask_start[dup] = 0
        return dict(dist_start=self.coarse_dist[s, nodes], mask_start=mask_start,
                    dist_int=dist_int, mask_int=mask_int)

    def subgraph_agent_visits(self, routes, route_schedules):
        """Per-subgraph agent traffic extracted from a solution.

        routes[k] lists the fine waypoints agent k visits after its start node, and
        route_schedules[k][i + 1] is the time at the i-th node of [start, *route]
        (the convention used by the solution plots).

        Returns {subgraph_id: [visit, ...]} with one visit per maximal run of consecutive
        waypoints inside that subgraph; an agent re-entering a subgraph gets a new visit.
        Each visit is a dict: agent, entry_node, exit_node, entry_time, exit_time.
        """
        visits = {int(p): [] for p in np.unique(self.partition)}
        for agent, (route, schedule) in enumerate(zip(routes, route_schedules)):
            path = [int(self.sd_mat[agent, 0]), *map(int, route)]
            times = np.asarray(schedule, dtype=float)
            if len(times) < len(path) + 1:
                raise ValueError(
                    f"Agent {agent} schedule has {len(times)} entries; "
                    f"route requires at least {len(path) + 1}."
                )
            run_start = 0
            for k in range(1, len(path) + 1):
                if k < len(path) and self.partition[path[k]] == self.partition[path[run_start]]:
                    continue
                visits[int(self.partition[path[run_start]])].append(dict(
                    agent=agent,
                    entry_node=path[run_start],
                    exit_node=path[k - 1],
                    entry_time=float(times[run_start + 1]),
                    exit_time=float(times[k]),
                ))
                run_start = k
        for subgraph_visits in visits.values():
            subgraph_visits.sort(key=lambda v: (v["entry_time"], v["agent"]))
        return visits

    def fine_problem(self, subgraph, visits, tolArray, speed_lim_mat=None, agent_weights=None):
        """Input dict for MIRS_fine from one subgraph and its visits (subgraph_agent_visits output).

        Each visit becomes one fine-level agent: it enters at the visit's entry node/time and must
        reach its exit node by the visit's exit time, which is that agent's schedule upper bound.
        Visits with entry_node == exit_node have nothing to route and are skipped.
        Nodes are renumbered locally; node_ids and agent_ids map back to global ids.
        """
        nodes = np.flatnonzero(self.partition == subgraph)
        local = {int(g): i for i, g in enumerate(nodes)}
        kept = [v for v in visits[int(subgraph)] if v["entry_node"] != v["exit_node"]]
        agent_ids = np.array([v["agent"] for v in kept], dtype=int)

        problem = dict(
            node_ids=nodes,
            agent_ids=agent_ids,
            wp_locations=self.wp_locations[nodes],
            mask=(np.asarray(self.mask)[np.ix_(nodes, nodes)] > 0).astype(int),
            dist_mat=np.asarray(self.dist_mat)[np.ix_(nodes, nodes)],
            tolArray=np.asarray(tolArray, dtype=float)[nodes],
            sd_mat=np.array([[local[v["entry_node"]], local[v["exit_node"]]] for v in kept],
                            dtype=int).reshape(-1, 2),
            start_times=np.array([v["entry_time"] for v in kept], dtype=float),
            T_upper_bound=np.array([v["exit_time"] for v in kept], dtype=float),
        )
        if speed_lim_mat is not None:
            problem["speed_lim_mat"] = np.asarray(speed_lim_mat, dtype=float)[agent_ids]
        if agent_weights is not None:
            problem["agent_weights"] = np.asarray(agent_weights, dtype=float)[agent_ids]
        return problem


    def solve_subgraphs(self, routes, route_schedules, tolArray, subgraphs=None,
                        speed_lim_mat=None, agent_weights=None, time_limit=3600.0,
                        anneal_print=False, optim_config=None, anneal_config=None,
                        boundary_conditions=None, **mirs_kwargs):
        """Solve selected (default: all) subgraphs from the higher-level solution.

        Each subgraph is a MIRS_fine problem built from the agents' entry/exit nodes and times
        (fine_problem), solved with the CBF-CLF annealing optimizer using the rect barrier and the
        default optimizer/annealing settings unless optim_config/anneal_config override them.
        mirs_kwargs are passed to MIRS_fine (e.g. stagewiseCostCoeffs, selfHop, cost_mode).

        Returns {subgraph_id: result}. A subgraph with no routable agent has result None.
        A result dict holds the MIRS_fine instance, the annealing histories, and for every
        fine agent: agent_ids (global), routes (global waypoint ids) and route_schedules
        ([entry time, schedule at each route node]).
        """
        from MIRS_fine import MIRS_fine
        from lower_mep_opt import (
            MIRSOptimizer_fine, default_anneal_config, default_optim_config, default_rect_cbf,
        )

        visits = self.subgraph_agent_visits(routes, route_schedules)
        selected = sorted(visits) if subgraphs is None else [int(s) for s in np.atleast_1d(subgraphs)]
        unknown = [s for s in selected if s not in visits]
        if unknown:
            raise ValueError(f"Unknown subgraph ids: {unknown}")

        results = {}
        for subgraph in selected:
            problem = self.fine_problem(subgraph, visits, tolArray, speed_lim_mat, agent_weights)
            if len(problem["agent_ids"]) == 0:
                results[subgraph] = None
                continue

            optim = {**default_optim_config(), **(optim_config or {})}
            anneal = {**default_anneal_config(), **(anneal_config or {})}
            mirs = MIRS_fine(problem, ca_cbf=default_rect_cbf(problem["tolArray"]), **mirs_kwargs)
            optimizer = MIRSOptimizer_fine(mirs, optim, anneal)

            started = time.time()
            T_array, V_array, F_values, path_probabilities, reach_array, compute_times = (
                optimizer.anneal(mirs.sched_mat, mirs.speed_vec, active_waypoints=None,
                                 time_limit=time_limit, annealPrint=anneal_print)
            )
            runtime = time.time() - started
            final_beta = optimizer.b_arr[len(F_values) - 1]
            local_routes, local_schedules = mirs.solution_table(
                path_probabilities, T_array[-1], V_array[-1], final_beta
            )
            results[subgraph] = dict(
                mirs=mirs,
                node_ids=problem["node_ids"],
                agent_ids=problem["agent_ids"],
                routes=[problem["node_ids"][np.asarray(r, dtype=int)].tolist() for r in local_routes],
                route_schedules=[np.asarray(s, dtype=float) for s in local_schedules],
                local_routes=local_routes,
                schedule_history=T_array,
                speed_history=V_array,
                cost_history=F_values,
                reach_history=reach_array,
                compute_times=compute_times,
                path_probabilities=path_probabilities,
                final_schedule=T_array[-1],
                final_speed=V_array[-1],
                final_beta=float(final_beta),
                runtime=runtime,
            )
        return results

    def combine_solutions(self, routes, route_schedules, lower_results):
        """Merge the higher-level solution with the subgraph solutions into full-graph routes.

        Each visit of an agent to a subgraph is replaced by its fine route and schedule. Visits with
        entry == exit were not solved at the fine level and keep their single higher-level node.
        Returns (full_routes, full_schedules, max_overlap) in the same convention as the inputs
        (full_schedules[k] = [start time, time at start, time at route[0], ...]). max_overlap is the
        largest amount by which a fine exit time passes the next visit's entry time (0 if none).
        """
        visits = self.subgraph_agent_visits(routes, route_schedules)
        fine_segment = {}
        for subgraph, items in visits.items():
            kept = [v for v in items if v["entry_node"] != v["exit_node"]]
            if not kept:
                continue
            result = lower_results.get(subgraph)
            if result is None or len(result["routes"]) != len(kept):
                raise ValueError(f"Subgraph {subgraph} has no matching lower-level solution")
            for visit, route, schedule in zip(kept, result["routes"], result["route_schedules"]):
                fine_segment[id(visit)] = (
                    [int(n) for n in route], np.asarray(schedule, dtype=float)[1:]
                )

        by_agent = {}
        for items in visits.values():
            for visit in items:
                by_agent.setdefault(visit["agent"], []).append(visit)

        full_routes, full_schedules, max_overlap = [], [], 0.0
        for agent in range(len(routes)):
            nodes, times = [], []
            for visit in sorted(by_agent[agent], key=lambda v: v["entry_time"]):
                seg_nodes, seg_times = fine_segment.get(
                    id(visit), ([visit["entry_node"]], np.array([visit["entry_time"]]))
                )
                if times:
                    max_overlap = max(max_overlap, times[-1] - float(seg_times[0]))
                nodes += seg_nodes
                times += [float(t) for t in seg_times]
            full_routes.append(nodes[1:])
            full_schedules.append(np.array([times[0], *times]))
        return full_routes, full_schedules, float(max_overlap)

    def visualize_combined(self, routes, route_schedules, lower_results, tolArray, save_dir):
        """Combine the higher and lower level solutions and plot them on the entire graph."""
        import visualization

        full_routes, full_schedules, overlap = self.combine_solutions(
            routes, route_schedules, lower_results
        )
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = [
            save_dir / "network_plot_full.png",
            save_dir / "schedule_plot_full.png",
            save_dir / "agent_schedule_plot_full.png",
        ]
        visualization.plot_full_solution_network(self, full_routes, full_schedules, paths[0])
        visualization.plot_solution_waypoint_schedules(
            full_routes, full_schedules, self.sd_mat, tolArray, paths[1]
        )
        visualization.plot_solution_schedules(
            full_routes, full_schedules, self.sd_mat, paths[2]
        )
        return full_routes, full_schedules, overlap, paths


def generate_partition(wp_locations, n_parts, seed=0):
    """Spatial partition of the nodes into n_parts subgraphs via k-means on node locations."""
    from scipy.cluster.vq import kmeans2
    _, labels = kmeans2(np.asarray(wp_locations, float), n_parts, minit='++', seed=seed)
    return labels
