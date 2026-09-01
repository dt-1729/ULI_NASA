import numpy as np
import random
from scipy.spatial.distance import cdist
import utils
import flpoAgent

def init_waypoints(wp_params: dict, seed: int, INF:float):
    np.random.seed(seed)
    random.seed(seed)

    if wp_params['type'] == 'grid':
        wp_locations, mask = utils.generate_non_uniform_grid_graph_numpy(wp_params)
    elif wp_params['type'] == 'ring':
        wp_locations, mask = utils.generate_ring_network(wp_params)
    elif wp_params['type'] == 'multigraph':
        wp_locations, mask = utils.generate_multigraph(wp_params)
    elif wp_params['type'] == 'random':
        wp_locations, mask = utils.generate_random_graph(wp_params)

    dist_mat = cdist(wp_locations, wp_locations, 'euclidean')
    # dist_mat[mask == 0] = INF #float('inf')
    n_waypoints = len(wp_locations)
    wp_weights = np.ones(n_waypoints) / np.sum(np.ones(n_waypoints))

    np.random.seed(None)
    random.seed(None)

    return wp_locations, mask, dist_mat, n_waypoints, wp_weights


def init_agent_params(
    n_agents,
    wp_locations,
    seed,
    tolArray=None,
    mode=None):

    np.random.seed(seed)
    random.seed(seed)
    n_waypoints = wp_locations.shape[0]

    # agent_weights = np.ones(n_agents) / n_agents
    agent_weights = np.random.uniform(0.1, 1, (n_agents,))
    agent_weights = agent_weights/np.sum(agent_weights)
    sd_mat = np.array([np.random.choice(n_waypoints, size=2, replace=False) for _ in range(n_agents)])
    min_speeds = np.random.uniform(0.01, 1, n_agents).reshape(-1, 1)
    max_speeds = np.random.uniform(50, 80, n_agents).reshape(-1, 1)
    speed_lim_mat = np.concatenate((min_speeds, max_speeds), axis=1)
    speed_vec = speed_lim_mat.mean(axis=1)
    max_time = np.max(cdist(wp_locations, wp_locations, 'euclidean')) / np.min(min_speeds)

    if mode == 'lin_static':
        if tolArray is None:
            tolArray = np.ones(n_waypoints)
        max_tol = float(np.max(tolArray))
        min_spacing = max_tol * 1.5 if max_tol > 0 else 1.0
        start_times = np.arange(n_agents, dtype=float) * min_spacing
        waypoint_drift = np.linspace(0.0, max_tol * 2.25, n_waypoints)
        sched_mat = start_times[:, None] + waypoint_drift[None, :]
    else:
        sched_mat = np.random.uniform(0.0, 50.0, (n_agents, n_waypoints))
        start_times = np.random.uniform(0.0, 0.0, n_agents)
        sched_mat[np.arange(n_agents), sd_mat[:, 0]] = start_times

    T_upper_bound = 600
    process_T = np.random.uniform(3, 5, (n_agents, n_waypoints)) * 0
    process_T[np.arange(n_agents), sd_mat[:, 1]] = np.zeros(n_agents)

    np.random.seed(None)
    random.seed(None)

    return agent_weights, sd_mat, speed_lim_mat, speed_vec, sched_mat, start_times, T_upper_bound, process_T


def init_agents(n_agents, n_waypoints, sd_mat, sched_mat, start_times, speed_vec, speed_lim_mat, process_T, INF, offset_energy, selfHop, mask, stagewise_cost_coeffs):
    list_agents = []
    for i in range(n_agents):
        agent = flpoAgent.flpoAgent(
            n_wp=n_waypoints,
            sd=sd_mat[i, :],
            sched=sched_mat[i, :],
            start_time=start_times[i],
            speed=speed_vec[i],
            speedLim=speed_lim_mat[i, :],
            process_T=process_T[i, :],
            INF=INF,
            offset_energy=offset_energy,
            selfHop=selfHop,
            net_mask=mask,
            stagewise_cost_coeffs=stagewise_cost_coeffs
        )
        list_agents.append(agent)
    return list_agents


def print_initialization_data(n_waypoints, n_agents, tolArray, wp_locations, mask, dist_mat, wp_weights, agent_weights, sd_mat, process_T, speed_lim_mat, sched_mat, speed_vec, printFlag):
    if printFlag:
        print(f'n_waypoints: {n_waypoints} \nn_agents: {n_agents} \nCAT:\n{tolArray}')
        print('---------')
        print(f'wp_locations:\n{wp_locations} \nmask:\n{mask} \ndist_mat:\n{dist_mat} \nwp_weights:\n{wp_weights}')
        print('---------')
        print(f'agent_weights:\n{agent_weights} \nsd_mat:\n{sd_mat} \neta_arr: \nprocessing_time:\n{process_T} \nspeed_lim_mat:\n{speed_lim_mat} \nsched_mat:\n{sched_mat} \nspeed_vec:\n{speed_vec}')


def get_network_params(n_wp:int, tol_range:list, name:str):
    if name == "grid":
        # Grid network parameters
        net_params = {
            'type':'grid',
            'n_points':n_wp,
            'grid_size':5000,
            'noise_factor':100,
            'extra_connections':10
        }
    elif name == "ring":
        # Ring network parameters
        net_params = {
            'type':'ring',
            'num_rings':3,
            'points_per_ring':int(n_wp/3),
            'center_distance':1000,
            'deformation_level':50.0,
            'extra_connections':0.3,
            'missing_connections':0.05
        }
    elif name == "multi":
        net_params = {
            'type':'multigraph',
            'n_points':n_wp,
            'n_graphs':4,
            'grid_size':10000.0,
            'intra_graph_connectivity':0.1,
            'inter_graph_connectivity':0.01,
            'subgraph_type':'grid',
            'seed':None
        }
    elif name == "random":
        net_params = {
            'type':'random',
            'n_nodes':n_wp,
            'width':10000.0,
            'height':10000.0,
            'min_spacing':500,
            'connection_radius':2000,
            'max_neighbors':6,
            'long_edge_probability':0.2,
            'long_edge_decay':5000
        }

    tolArray = np.random.uniform(tol_range[0], tol_range[1], n_wp)

    return net_params, tolArray