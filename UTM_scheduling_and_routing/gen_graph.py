import numpy as np
from scipy.spatial import KDTree
import networkx as nx
import matplotlib.pyplot as plt
import random


def poisson_disk_sampling(width, height, min_dist, k=30):
    """
    Bridson's Poisson Disk Sampling.
    Returns points with minimum spacing min_dist.
    """
    cell_size = min_dist / np.sqrt(2)

    grid_w = int(np.ceil(width / cell_size))
    grid_h = int(np.ceil(height / cell_size))
    grid = -np.ones((grid_w, grid_h), dtype=int)

    points = []
    active = []

    # Initial point
    p = np.array([random.uniform(0, width),
                  random.uniform(0, height)])
    points.append(p)
    active.append(0)

    gx = int(p[0] // cell_size)
    gy = int(p[1] // cell_size)
    grid[gx, gy] = 0

    while active:
        idx = random.choice(active)
        center = points[idx]
        found = False

        for _ in range(k):
            r = random.uniform(min_dist, 2 * min_dist)
            theta = random.uniform(0, 2 * np.pi)

            candidate = center + r * np.array(
                [np.cos(theta), np.sin(theta)]
            )

            if not (0 <= candidate[0] < width and
                    0 <= candidate[1] < height):
                continue

            gx = int(candidate[0] // cell_size)
            gy = int(candidate[1] // cell_size)

            valid = True

            for i in range(max(0, gx - 2), min(grid_w, gx + 3)):
                for j in range(max(0, gy - 2), min(grid_h, gy + 3)):
                    neighbor = grid[i, j]
                    if neighbor != -1:
                        d = np.linalg.norm(candidate - points[neighbor])
                        if d < min_dist:
                            valid = False
                            break
                if not valid:
                    break

            if valid:
                points.append(candidate)
                active.append(len(points) - 1)
                grid[gx, gy] = len(points) - 1
                found = True
                break

        if not found:
            active.remove(idx)

    return np.array(points)

def generate_points_with_target_count(
    n_nodes,
    width,
    height,
    tol=2,
    max_iter=20,
):
    """
    Generate approximately n_nodes Poisson-distributed points.
    """

    area = width * height

    # Initial estimate
    r_low = 0.1
    r_high = np.sqrt(area)

    best_points = None
    best_error = np.inf

    for _ in range(max_iter):

        r = 0.5 * (r_low + r_high)

        pts = poisson_disk_sampling(
            width,
            height,
            min_dist=r,
        )

        error = abs(len(pts) - n_nodes)

        if error < best_error:
            best_error = error
            best_points = pts

        if error <= tol:
            break

        if len(pts) > n_nodes:
            # Too many points -> increase spacing
            r_low = r
        else:
            # Too few points -> decrease spacing
            r_high = r

    return best_points

def generate_graph(
    n_nodes=150,
    width=100,
    height=100,
    min_spacing=8,
    connection_radius=20,
    max_neighbors=5,
    long_edge_probability=0.02,
    long_edge_decay=30,
):
    """
    Parameters
    ----------
    min_spacing : minimum allowed point distance
    connection_radius : local edge radius
    max_neighbors : max degree per node
    long_edge_probability : controls frequency of long edges
    long_edge_decay : larger => more long edges
    """

    # points = poisson_disk_sampling(
    #     width, height, min_spacing
    # )

    points = generate_points_with_target_count(
        n_nodes=n_nodes,
        width=width,
        height=height,
    )

    tree = KDTree(points)
    G = nx.Graph()

    for i, p in enumerate(points):
        G.add_node(i, pos=p)

    degree = {i: 0 for i in range(len(points))}

    # Local edges
    for i, p in enumerate(points):
        neighbors = tree.query_ball_point(
            p, connection_radius
        )

        neighbors = [
            j for j in neighbors if j != i
        ]

        neighbors.sort(
            key=lambda j: np.linalg.norm(
                points[i] - points[j]
            )
        )

        for j in neighbors:
            if degree[i] >= max_neighbors:
                break
            if degree[j] >= max_neighbors:
                continue

            if not G.has_edge(i, j):
                G.add_edge(i, j)
                degree[i] += 1
                degree[j] += 1

    # Connect disconnected components
    while not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        c1 = comps[0]
        c2 = comps[1]

        best_pair = None
        best_dist = np.inf

        for u in c1:
            if degree[u] >= max_neighbors:
                continue
            for v in c2:
                if degree[v] >= max_neighbors:
                    continue

                d = np.linalg.norm(
                    points[u] - points[v]
                )

                if d < best_dist:
                    best_dist = d
                    best_pair = (u, v)

        if best_pair is None:
            break

        u, v = best_pair
        G.add_edge(u, v)
        degree[u] += 1
        degree[v] += 1

    # Add long-range edges
    n = len(points)

    for i in range(n):
        if degree[i] >= max_neighbors:
            continue

        for j in range(i + 1, n):
            if G.has_edge(i, j):
                continue
            if degree[j] >= max_neighbors:
                continue

            d = np.linalg.norm(
                points[i] - points[j]
            )

            if d <= connection_radius:
                continue

            # Exponentially decreasing probability
            p = long_edge_probability * np.exp(
                -d / long_edge_decay
            )

            if random.random() < p:
                G.add_edge(i, j)
                degree[i] += 1
                degree[j] += 1

                if degree[i] >= max_neighbors:
                    break

    return G, points


# -----------------------------
# Example
# -----------------------------

G, pts = generate_graph(
    n_nodes=25,
    width=10000,
    height=10000,
    min_spacing=60,
    connection_radius=15,
    max_neighbors=5,
    long_edge_probability=0.3,
    long_edge_decay=40,
)

pos = {i: pts[i] for i in range(len(pts))}

plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(
    G,
    pos,
    edge_color="gray",
    alpha=0.6,
    width=1,
)
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=25,
    node_color="red",
)

plt.axis("equal")
plt.axis("off")
plt.show()