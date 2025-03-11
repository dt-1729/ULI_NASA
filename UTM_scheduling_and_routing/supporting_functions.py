import numpy as np
import random
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


# function to pick a point at random from a square
def random_point_in_square(xlim:np.ndarray, ylim:np.ndarray):
    # Generate random x and y coordinates
    x = np.random.uniform(xlim[0], xlim[1])
    y = np.random.uniform(ylim[0], ylim[1])
    return np.array([x,y])


# function to generate a square matrix with minimum number of zeroes and ones
def random_symmetric_binary_matrix_with_constraints(n, min_ones, min_zeros):
    # Initialize an n x n matrix with zeros
    matrix = np.zeros((n, n), dtype=int)
    # Ensure all diagonal elements are 1
    np.fill_diagonal(matrix, 1)
    # Track remaining ones and zeros needed for each row (accounting for the diagonal 1)
    ones_needed = [max(0, min_ones - 1) for _ in range(n)]
    zeros_needed = [max(0, min_zeros) for _ in range(n)]
    # Fill the upper triangle of the matrix
    for i in range(n):
        for j in range(i + 1, n):
            # Determine the value based on remaining constraints
            if ones_needed[i] > 0 and ones_needed[j] > 0:
                # Set both entries to 1
                matrix[i][j] = matrix[j][i] = 1
                ones_needed[i] -= 1
                ones_needed[j] -= 1
            elif zeros_needed[i] > 0 and zeros_needed[j] > 0:
                # Set both entries to 0
                matrix[i][j] = matrix[j][i] = 0
                zeros_needed[i] -= 1
                zeros_needed[j] -= 1
            else:
                # Randomly choose 0 or 1 while keeping symmetry
                value = random.choice([0, 1])
                matrix[i][j] = matrix[j][i] = value
                if value == 1:
                    ones_needed[i] -= 1
                    ones_needed[j] -= 1
                else:
                    zeros_needed[i] -= 1
                    zeros_needed[j] -= 1
    # Adjust rows if necessary to meet exact constraints
    for i in range(n):
        row_ones = np.sum(matrix[i])
        row_zeros = n - row_ones
        # Add additional ones or zeros if the row does not meet the constraints
        if row_ones < min_ones:
            # Add ones to random zero entries in the row
            zero_indices = [j for j in range(n) if matrix[i][j] == 0 and i != j]
            additional_ones = random.sample(zero_indices, min_ones - row_ones)
            for j in additional_ones:
                matrix[i][j] = matrix[j][i] = 1
        elif row_zeros < min_zeros:
            # Add zeros to random one entries in the row
            one_indices = [j for j in range(n) if matrix[i][j] == 1 and i != j]
            additional_zeros = random.sample(one_indices, min_zeros - row_zeros)
            for j in additional_zeros:
                matrix[i][j] = matrix[j][i] = 0
    return matrix


def uniformly_sample_vec_from_a_ball(c_vec, rad, n_samples):
    sampled_vecs = []
    # np.random.seed(seed)
    while len(sampled_vecs) <= n_samples-1:
        rad_s = np.random.uniform(0,rad)
        dir_s = np.random.uniform(0,1,c_vec.shape)
        norm_dir_s = np.linalg.norm(dir_s)
        if norm_dir_s > 0.0:
            unit_dir_s = dir_s/norm_dir_s
            vec_s = c_vec + rad_s * unit_dir_s
            sampled_vecs.append(vec_s)
    # np.random.seed(None)

    return np.array(sampled_vecs)


def myPenaltyFunc(X, gamma, coeff):
    # penalty is applied if X-offset >= 0
    def my_exp(x):
        exp_inf = 700
        return np.exp(np.minimum(x,exp_inf))
    pX = coeff*gamma*np.log((
            np.log(my_exp(1/gamma)+my_exp(gamma*X))
            )/(np.log(my_exp(1/gamma)+my_exp(-gamma))))
    return pX


def generate_non_uniform_grid_points(n_points, square_size=1.0, noise_factor=0.05):
    grid_size = int(np.ceil(np.sqrt(n_points)))
    x = np.linspace(0, square_size, grid_size)
    y = np.linspace(0, square_size, grid_size)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])[:n_points]
    noise = np.random.uniform(-noise_factor, noise_factor, points.shape)
    points += noise
    return points


def generate_non_uniform_grid_graph_numpy(params):
    n_points = params['n_points']
    square_size = params['grid_size']
    noise_factor=params['noise_factor']
    points = generate_non_uniform_grid_points(n_points, square_size, noise_factor)
    adj = np.zeros((n_points, n_points), dtype=int)
    tolerance = 0.1 * square_size / np.sqrt(n_points)
    grid_size = int(np.ceil(np.sqrt(n_points)))
    
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            if index >= n_points:
                break
            
            if j + 1 < grid_size and index + 1 < n_points:
                right_neighbor = index + 1
                if abs(points[index][1] - points[right_neighbor][1]) < tolerance:
                    adj[index, right_neighbor] = 1
                    adj[right_neighbor, index] = 1
            
            if i + 1 < grid_size and index + grid_size < n_points:
                bottom_neighbor = index + grid_size
                if abs(points[index][0] - points[bottom_neighbor][0]) < tolerance:
                    adj[index, bottom_neighbor] = 1
                    adj[bottom_neighbor, index] = 1

    def find_connected_component(node, visited):
        stack = [node]
        component = []
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                component.append(current)
                stack.extend(np.where(adj[current] == 1)[0])
        return component
    
    visited = np.zeros(n_points, dtype=bool)
    components = []
    
    for node in range(n_points):
        if not visited[node]:
            component = find_connected_component(node, visited)
            components.append(component)
    
    if len(components) > 1:
        kd_tree = KDTree(points)
        for i in range(len(components) - 1):
            comp1, comp2 = components[i], components[i + 1]
            min_dist = float('inf')
            closest_pair = (None, None)
            for node1 in comp1:
                for node2 in comp2:
                    dist = np.linalg.norm(points[node1] - points[node2])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (node1, node2)
            adj[closest_pair[0], closest_pair[1]] = 1
            adj[closest_pair[1], closest_pair[0]] = 1

    return points, adj


def generate_ring_network(params):
    """
    Generate a network of waypoints forming deformed concentric rings with missing and noisy connections.

    Args:
    - num_rings (int): Number of concentric rings.
    - points_per_ring (int): Number of waypoints per ring.
    - center_distance (float): Base distance of each ring from the center.
    - deformation_level (float): Maximum random perturbation of the radius (deformation).
    - extra_connections (float): Probability of adding random extra connections.
    - missing_connections (float): Probability of removing an existing connection.

    Returns:
    - positions (np.ndarray): Array of shape (total_points, 2) with the (x, y) coordinates of waypoints.
    - adjacency_matrix (np.ndarray): Adjacency matrix of the network (total_points x total_points).
    """
    num_rings = params['num_rings']
    points_per_ring = params['points_per_ring']
    center_distance = params['center_distance']
    deformation_level = params['deformation_level']
    extra_connections = params['extra_connections']
    missing_connections = params['missing_connections']

    positions = []  # Store the (x, y) positions of the waypoints

    # Generate positions for deformed rings
    for ring in range(num_rings):
        base_radius = (ring + 1) * center_distance
        angles = np.linspace(0, 2 * np.pi, points_per_ring, endpoint=False)
        for angle in angles:
            # Add deformation to the radius
            radius = base_radius + np.random.uniform(-deformation_level, deformation_level)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))

    positions = np.array(positions)
    total_points = len(positions)

    # Initialize adjacency matrix
    adjacency_matrix = np.zeros((total_points, total_points), dtype=int)

    # Create connectivity within each deformed ring
    start_index = 0
    for ring in range(num_rings):
        for i in range(points_per_ring):
            current = start_index + i
            next_point = start_index + (i + 1) % points_per_ring
            adjacency_matrix[current, next_point] = 1
            adjacency_matrix[next_point, current] = 1
        start_index += points_per_ring

    # Create radial connectivity between rings
    for ring in range(num_rings - 1):
        ring_start = ring * points_per_ring
        next_ring_start = (ring + 1) * points_per_ring
        for i in range(points_per_ring):
            adjacency_matrix[ring_start + i, next_ring_start + i] = 1
            adjacency_matrix[next_ring_start + i, ring_start + i] = 1

    # Add random noisy connections
    num_possible_connections = total_points * (total_points - 1) // 2
    num_extra_connections = int(extra_connections * num_possible_connections)

    for _ in range(num_extra_connections):
        p1, p2 = np.random.choice(total_points, size=2, replace=False)
        adjacency_matrix[p1, p2] = 1
        adjacency_matrix[p2, p1] = 1

    # Remove random connections to create missing edges
    num_missing_connections = int(missing_connections * total_points)
    for _ in range(num_missing_connections):
        i, j = np.random.choice(total_points, size=2, replace=False)
        adjacency_matrix[i, j] = 0
        adjacency_matrix[j, i] = 0

    return positions, adjacency_matrix


