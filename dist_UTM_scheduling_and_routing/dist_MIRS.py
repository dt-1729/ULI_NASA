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

    def build_coarse_graph(self):
        mask = self.mask > 0
        dist = self.dist_mat
        part = self.partition
        selected = {(min(i, j), max(i, j)) for i, j in self.selected_edges}

        self.coarse_nodes = np.unique(np.concatenate([self.sd_mat.ravel(), self.boundary_nodes()]))
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


def generate_partition(wp_locations, n_parts, seed=0):
    """Spatial partition of the nodes into n_parts subgraphs via k-means on node locations."""
    from scipy.cluster.vq import kmeans2
    _, labels = kmeans2(np.asarray(wp_locations, float), n_parts, minit='++', seed=seed)
    return labels
