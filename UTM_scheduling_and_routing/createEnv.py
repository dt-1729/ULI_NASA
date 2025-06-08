import numpy as np
import matplotlib.pyplot as plt
import flpoAgent
import importlib
import random
from scipy.spatial.distance import cdist
from scipy.optimize import minimize, Bounds
import flpoAgent
import time
import supporting_functions
from collections import defaultdict
from matplotlib.cm import get_cmap


# create a multiagent scheduling and routing class
class MARS():

    def __init__(
        self, 
        n_waypoints, 
        n_agents, 
        tolArray, 
        b1, 
        b2, 
        wp_params, 
        seed, 
        printFlag):

        self.n_waypoints = n_waypoints
        self.n_agents = n_agents
        self.tolArray = tolArray
        self.b1 = b1
        self.b2 = b2
        self.INF = 1e8
        self.initWaypoints(wp_params, seed=seed)
        self.initAgentParams(seed=seed)
        self.initFlpoAgents()
        self.printInitializationData(printFlag)


    # function to create waypoints and the corresponding adjacency matrix
    def initWaypoints(self, wp_params:dict, seed:int):
        np.random.seed(seed)
        random.seed(seed)
        nw = self.n_waypoints
        if wp_params['type']=='grid':
            self.wp_locations, self.mask = supporting_functions.generate_non_uniform_grid_graph_numpy(wp_params)
        elif wp_params['type']=='ring':
            self.wp_locations, self.mask = supporting_functions.generate_ring_network(wp_params)
        self.dist_mat = cdist(self.wp_locations, self.wp_locations, 'euclidean')
        self.dist_mat[self.mask==0] = self.INF
        # wp_weights = np.random.uniform(0,1,nw)
        wp_weights = np.ones(nw)
        self.wp_weights = wp_weights/np.sum(wp_weights)
        np.random.seed(None)
        random.seed(None)
        pass


    # function to create agents and assign their parameters
    def initAgentParams(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        na = self.n_agents
        nw = self.n_waypoints
        # agent_weights = np.random.uniform(0,1, na)
        agent_weights = np.ones(na)
        # agent_weights[0] = 10.0
        # agent_weights[4] = 10.0
        # agent_weights[5] = 10.0
        self.agent_weights = agent_weights#/np.sum(agent_weights)
        self.sd_mat = np.random.choice(range(nw), (na,2))
        # self.sd_mat[3,1] = 5
        # self.sd_mat[0,1] = 10
        self.eta_arr = np.ones(na)*1000.0
        self.theta_arr = np.ones(na)*100
        min_speeds = np.random.uniform(2,2.5,na).reshape(-1,1)
        max_speeds = np.random.uniform(30,45,na).reshape(-1,1)
        # max_speeds = np.array([[101],[280]])
        self.speed_lim_mat = np.concatenate((min_speeds,max_speeds), axis=1)
        self.t_start_min = 0.0
        max_time = np.max(cdist(self.wp_locations, self.wp_locations, 'euclidean'))/np.min(min_speeds)
        self.sched_mat = np.random.uniform(self.t_start_min, 100.0, (na, nw))
        # for i in range(1,na):
        #     self.sched_mat[i,:] = self.sched_mat[i-1,:] + np.ones(nw)*5
        self.sched_mat[np.arange(na),self.sd_mat[:,0]] = np.ones(na)*self.t_start_min
        # print(self.sched_mat[:,self.sd_mat[:,0]])
        self.process_T = np.random.uniform(3,5,(na, nw))
        self.process_T[np.arange(na),self.sd_mat[:,1]] = np.zeros(na)
        
        np.random.seed(None)
        random.seed(None)
        pass


    # function to initialize agents
    def initFlpoAgents(self):
        list_agents = []
        na = self.n_agents
        nw = self.n_waypoints
        for i in range(na):
            v = flpoAgent.flpoAgent(n_wp=nw, sd=self.sd_mat[i,:], sched=self.sched_mat[i,:],
                            speedLim=self.speed_lim_mat[i,:], process_T=self.process_T[i,:], eta=self.eta_arr[i], theta=self.theta_arr[i], INF=self.INF)
            list_agents.append(v)
        self.agents = list_agents
        pass


    # function to print initialization data
    def printInitializationData(self,printFlag):
        if printFlag:
            print(f'n_waypoints: {self.n_waypoints} \nn_agents: {self.n_agents} \nCAT:\n{self.tolArray} \nb1:\n{self.b1} \nb2:\n{self.b2}')
            print('---------')
            print(f'wp_locations:\n{self.wp_locations} \nmask:\n{self.mask} \ndist_mat:\n{self.dist_mat} \nwp_weights:\n{self.wp_weights}')
            print('---------')
            print(f'agent_weights:\n{self.agent_weights} \nsd_mat:\n{self.sd_mat} \neta_arr:\n{self.eta_arr} \nprocessing_time:\n{self.process_T} \nspeed_lim_mat:\n{self.speed_lim_mat} \nsched_mat:\n{self.sched_mat}')


    def calc_agent_reach_mat(self, sched_mat, beta, gamma, coeff):
        reach_mat = []
        for i,ag in enumerate(self.agents):
            ag_reach = ag.calc_probability_of_reach(sched_mat[i,:], self.dist_mat, beta, gamma, coeff)
            reach_mat.append(ag_reach)
        return np.array(reach_mat)


    # function to compute total vehicle cost of transportation
    def transportCost(self, sched_mat:np.ndarray, beta:float, gamma:float, coeff:float, allowPrint=False):
        C_arr = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            v = self.agents[i]
            v.getFreeEnergy_s(sched_mat[i,:], self.dist_mat, beta, gamma, coeff)     
            C_arr[i] = v.freeEnergy_s
        self.C_agents = C_arr
        
        if allowPrint:
            print('\nindividual agent costs:')
            for i,v in enumerate(self.agents):
                print(f'v{i+1}:\t{v.freeEnergy_s:.2f}')
            print(f'sum: {self.C_agents}')
        pass


    # function to compute total node conflict cost
    def conflictCost(self, sched_mat:np.ndarray, gamma:float, coeff:float, filter_wp:np.ndarray, allowPrint=False):
        nw = self.n_waypoints
        na = self.n_agents
        # conflictMat = np.zeros(shape=(nw, na, na))
        conflictCostArray = np.zeros(nw)
        # filter_wp_arr = filter_wp.max(axis=0)
        for j in range(nw):
            ids = np.where(filter_wp[:,j]==1)[0]
            if len(ids)>=2:
                tj = sched_mat[ids,j].reshape(-1,1)
                tauj = np.tile(tj,(1,len(ids))) - np.transpose(np.tile(tj,(1,len(ids))))
                # conflictMat = 2000*(1 + np.tanh(gamma*(self.tolArray[j]-np.abs(tauj))))
                conflictMat = supporting_functions.myPenaltyFunc(self.tolArray[j]**2-np.abs(tauj)**2, gamma, coeff)
                # print(conflictMat)
                np.fill_diagonal(conflictMat, 0.0)
                conflictCostArray[j] = self.wp_weights[j]*np.sum(np.abs(conflictMat))
        self.C_wp_conflict = conflictCostArray

        if allowPrint:
            print(f'\ninidividual conflict costs: \n {conflictCostArray}\nsum: {self.C_wp_conflict}')
        pass

    # function to compute total node conflict cost
    def conflictCost1(self, sched_mat:np.ndarray, gamma:float, coeff:float, allowPrint=False):
        nw = self.n_waypoints
        na = self.n_agents
        # conflictMat = np.zeros(shape=(nw, na, na))
        conflictCostArray = np.zeros(nw)
        # filter_wp_arr = filter_wp.max(axis=0)
        for j in range(nw):
            tj = sched_mat[:,j].reshape(-1,1)
            tauj = np.tile(tj,(1,self.n_agents)) - np.transpose(np.tile(tj,(1,self.n_agents)))
            # conflictMat = 2000*(1 + np.tanh(gamma*(self.tolArray[j]-np.abs(tauj))))
            conflictMat = supporting_functions.myPenaltyFunc(self.tolArray[j]**2-np.abs(tauj)**2, gamma, coeff)
            # print(conflictMat)
            np.fill_diagonal(conflictMat, 0.0)
            conflictCostArray[j] = self.wp_weights[j] * np.sum(np.abs(conflictMat))
        self.C_wp_conflict = conflictCostArray

        if allowPrint:
            print(f'\ninidividual conflict costs: \n {conflictCostArray}\nsum: {self.C_wp_conflict}')
        pass


    # function to compute total cost
    def totalCost(self, sched_vec:np.ndarray, beta:float, gamma_t:float, gamma_c:float, coeff_t:float, coeff_c:float, filter_wp:np.ndarray):
        assert(sched_vec.shape == (self.n_agents*self.n_waypoints,))
        sched_mat = sched_vec.reshape(self.n_agents,self.n_waypoints)
        self.transportCost(sched_mat=sched_mat, beta=beta, gamma=gamma_t, coeff=coeff_t, allowPrint=False)
        self.conflictCost(sched_mat=sched_mat, gamma=gamma_c, coeff=coeff_c, filter_wp=filter_wp)
        nw = self.n_waypoints
        na = self.n_agents
        C = np.sum(self.agent_weights*self.C_agents) + np.sum(self.C_wp_conflict) + 0.00*np.sum(np.abs(sched_vec))
        return C


    # function to perform optimization iterations at a given beta
    # def optimize_schedule_trust_constr1(self, init_sched_vec0, beta, gammaLim, filter_wp, bds, opts, allowPrintOptimize=False):
    #     t0 = time.time()
    #     i = 0
    #     gamma = gammaLim[0]
    #     gamma_max = gammaLim[1]
    #     gamma_grow = gammaLim[2]
    #     while gamma <= gamma_max:
    #         if i == 0:
    #             if allowPrintOptimize:
    #                 print('\tInside optimize_schedule:')
    #             init_sched_vec = init_sched_vec0
    #         else:
    #             init_sched_vec = sched
    #         res = minimize(self.totalCost, init_sched_vec, 
    #                         args=(beta,gamma,filter_wp), bounds=bds,
    #                         method='trust-constr', options=opts)
    #         sched = res.x
    #         cost_fun = res.fun
    #         i+=1
    #         if allowPrintOptimize:
    #             print(f'\tgamma: {gamma:.3e},\tt: {np.round(sched,2)},\tcost: {cost_fun:.3f}')
    #             # print(f'\tpb: {pb[0]}\tXi: {Xi[0]}')
    #         gamma=gamma*gamma_grow
    #     computeTime = time.time() - t0
    #     return cost_fun, sched, computeTime


    # function to perform optimization iterations at a given beta
    def optimize_schedule_trust_constr(self, init_sched_vec0, beta, gamma_t, gamma_c, coeff_t, coeff_c, filter_wp, bds, opts, allowPrintOptimize=False):
        t0 = time.time()
        res = minimize(self.totalCost, init_sched_vec0, 
                        args=(beta,gamma_t,gamma_c,coeff_t,coeff_c,filter_wp), bounds=bds,
                        method='trust-constr', options=opts)
        sched = res.x
        cost_fun = res.fun
        computeTime = time.time() - t0
        return res.fun, res.x, computeTime


    # function to perform annealing
    def annealing(self, beta_lims, beta_grow, sched_vec0, init_sched_bounds, optimize_opt, allowPrintAnneal=False, allowPrintOptimize=False):
        beta = beta_lims[0]
        beta_max = beta_lims[1]
        gamma_t = 5
        gamma_c = 7.5
        coeff_t = 50
        coeff_c = 5.0
        reach_mat = self.calc_agent_reach_mat(
                sched_vec0.reshape(self.n_agents, self.n_waypoints), 
                beta, gamma=gamma_t, coeff=coeff_t)
        sched_vec = sched_vec0
        filter_wp = np.ones(shape=reach_mat.shape)
        filter_wp[reach_mat <= 1.0e-10] = 0.0
        reach_mat_beta_data = np.array([reach_mat])
        filter_wp_beta_data = np.array([filter_wp])
        C_arr = [self.totalCost(sched_vec, beta, gamma_t, gamma_c, coeff_t, coeff_c, filter_wp)]
        conflict_C_arr = np.array([self.C_wp_conflict])
        rt_arr = [0]
        beta_arr = [beta]
        gamma_arr = [[gamma_t, gamma_c, coeff_t, coeff_c]]
        lb0, ub0 = init_sched_bounds; lb=lb0; ub=ub0
        while beta < beta_max:
            if allowPrintAnneal:
                print(f'\nbeta:{beta:.3e}\tgamma_t:{gamma_t:.3e}\tgamma_c:{gamma_c:.3e}\tcoeff_t:{coeff_t:.3e}\tcoeff_c:{coeff_c:.3e}')
                print(f'filter_wp:{filter_wp.flatten()},\n\treach_mat:{np.round(reach_mat.flatten(),3)}')
                # print(f'schedule: {np.round(sched_vec,2)}')
                print(f'C_agents:{np.round(self.C_agents,2)},\nC_conflict:{np.round(self.C_wp_conflict,2)},\n\tcost:{C_arr[-1]:.2f}')
                # print(f'\t:{np.round(sched_vec,2)},\n\treach_mat:{np.round(reach_mat.flatten(),3)},\n\tfilter_wp:{filter_wp.flatten()},\n\tlb:{lb}\n\tub:{ub}')
 
            C, sched_vec, rt = self.optimize_schedule_trust_constr(
                                    init_sched_vec0=sched_vec, 
                                    beta=beta, 
                                    gamma_t=gamma_t, gamma_c=gamma_c,
                                    coeff_t=coeff_t, coeff_c=coeff_c,
                                    filter_wp=filter_wp,
                                    bds=Bounds(lb,ub), opts=optimize_opt, allowPrintOptimize=allowPrintOptimize)
            self.sched_mat = sched_vec.reshape(self.n_agents,self.n_waypoints)
            sched_vec += np.random.uniform(0,0.01,len(sched_vec))
            beta = beta * beta_grow
            beta_arr.append(beta)
            C_arr.append(C)
            rt_arr.append(rt)
            reach_mat = self.calc_agent_reach_mat(self.sched_mat, beta, gamma=gamma_t, coeff=coeff_t)
            filter_wp = np.ones(shape=reach_mat.shape)
            # if beta >= 1.0:
            filter_wp[reach_mat <= 1.0e-10] = 0.0
            lb = lb0*filter_wp.flatten()
            ub = ub0*filter_wp.flatten()
            sched_vec=sched_vec*filter_wp.flatten()
            
            reach_mat_beta_data = np.concatenate((reach_mat_beta_data,np.array([reach_mat])),axis=0)
            filter_wp_beta_data = np.concatenate((filter_wp_beta_data,np.array([filter_wp])),axis=0)
            conflict_C_arr = np.concatenate((conflict_C_arr, np.array([self.C_wp_conflict])),axis=0)
            
            # penalty parameters
            gamma_t += 1.00
            gamma_c += 0.00
            coeff_t += 0.00
            coeff_c *= 1.00
            gamma_arr.append([gamma_t, gamma_c, coeff_t, coeff_c])
            # if abs(C_arr[-1]-C_arr[-2])/max(abs(C_arr[-1]),1) < 1e-5:
            #     print(f'beta:{beta:.3e},\tcost:{C_arr[-1]:.2f},\tt:{np.round(sched_vec,2)}')
            #     print(f'stopping condition reached: rel_cost:{abs(C_arr[-1]-C_arr[-2])/abs(C_arr[-1])}')
            #     break
        
        # compute final probability associations
        Pb_a = []
        for i,a in enumerate(self.agents):
            Pb = a.getPathAssociations(self.sched_mat[i,:], self.dist_mat, beta, gamma_t, coeff_t)
            Pb_a.append(Pb)

        return C_arr, sched_vec, rt_arr, reach_mat_beta_data, filter_wp_beta_data, beta_arr, gamma_arr, conflict_C_arr, Pb_a


def calc_agent_routes_and_schedules(mars:MARS, Pb_a:list, printRoutes=False):
    routes = []
    fin_schedules = []
    
    for i,a in enumerate(mars.agents):
        a.calc_route_and_schedule(sched=mars.sched_mat[i,:],dist_mat=mars.dist_mat, Pb=Pb_a[i])
        routes.append(a.route)
        fin_schedules.append(a.fin_sched)
        if printRoutes:
            print(f'route v{i}: {a.route}, schedule: {np.round(a.fin_sched,2)}')

    return routes, fin_schedules


# function to show the network
def plotNetwork(mars:MARS, figuresize, routes, agent_colors, showEdgeLength=True, plotPaths=False):

    nw = mars.n_waypoints
    na = mars.n_agents
    wp_xy = mars.wp_locations

    # Create a figure for the plot
    plt.figure(figsize=figuresize)

    # Plot the nodes
    for i, (x, y) in enumerate(wp_xy):
        plt.scatter(x, y, color='skyblue', s=500, alpha=0.3)  # Other nodes
        plt.text(x, y, rf'{i}', fontsize=16, color='grey', ha='center', va='center')  # Label the nodes
    
    # Plot the edges and annotate lengths
    for i in range(nw):
        for j in range(i + 1, nw):  # Only iterate over the upper triangle of the adjacency matrix
            if mars.mask[i, j] == 1:
                # Draw an edge (a line) between node i and node j
                plt.plot([wp_xy[i, 0], wp_xy[j, 0]], [wp_xy[i, 1], wp_xy[j, 1]], color='skyblue', alpha=0.2, linewidth=12)
                if showEdgeLength:
                    # Calculate the distance between node i and node j
                    distance = mars.dist_mat[i,j]
                    # Annotate the edge with length
                    mid_x = (wp_xy[i, 0] + wp_xy[j, 0]) / 2
                    mid_y = (wp_xy[i, 1] + wp_xy[j, 1]) / 2
                    plt.text(mid_x, mid_y, rf'${distance:.1f}$', fontsize=8, color='black', ha='center', va='center')
                    # plt.text(mid_x, mid_y, rf'$[{t_min:.2f},{t_max:.2f}]$', fontsize=10, color='red', ha='center', va='center')

    # plot agent start and destinations
    start_groups = defaultdict(list)
    destination_groups = defaultdict(list)
    # Enumerate agent indices and store them by waypoint index
    for agent_index, start_idx in enumerate(mars.sd_mat[:,0]):
        start_groups[start_idx].append(agent_index)
    for agent_index, dest_idx in enumerate(mars.sd_mat[:,1]):
        destination_groups[dest_idx].append(agent_index)
    # Annotate each unique start and destination point
    for start_idx, agents in start_groups.items():
        start_x, start_y = wp_xy[start_idx]
        if len(agents) > 1:
            label = ', '.join([rf'$a_{i}$' for i in agents])  # Combined label for multiple agents
        else:
            label = rf'$a_{agents[0]}$'
        plt.text(start_x+0.5, start_y+0.5, label, color='darkgreen', fontsize=18,
                ha='left', va='top', fontweight='bold')
    for dest_idx, agents in destination_groups.items():
        dest_x, dest_y = wp_xy[dest_idx]
        if len(agents) > 1:
            label = ', '.join([rf'$a_{i}$' for i in agents])  # Combined label for multiple agents
        else:
            label = rf'$a_{agents[0]}$'
        plt.text(dest_x + 0.5, dest_y + 0.5, label, color='red', fontsize=18,
                ha='left', va='bottom', fontweight='bold')

    # Draw each agent's path
    if plotPaths:
        offset_mag = 5
        for i, ai in enumerate(mars.agents):
            # ai.calc_route(sched=mars.sched_mat[0,:],dist_mat=mars.dist_mat, beta=1000, gamma=1000)
            path = routes[i]
            # Extract x and y coordinates of waypoints in this path
            path_coords = wp_xy[path]  # Select rows based on path indices
            x_coords, y_coords = path_coords[:, 0], path_coords[:, 1]
            x_offset = offset_mag*np.random.uniform(-1,-1) * (i+1)
            y_offset = offset_mag*np.random.uniform(-1,-1) * (i+1)
            x_coords_offset = x_coords + x_offset
            y_coords_offset = y_coords + y_offset
            # Plot the path line with a unique color for each agent
            plt.plot(x_coords_offset, y_coords_offset, label=f'Path a_{i}', linestyle='--', linewidth=1.5, color=agent_colors[i])
            for j in range(len(path) - 1):
                dx = x_coords_offset[j + 1] - x_coords_offset[j]
                dy = y_coords_offset[j + 1] - y_coords_offset[j]
                plt.quiver(
                    x_coords_offset[j], y_coords_offset[j], dx, dy, angles='xy', scale_units='xy', scale=1,
                    color=plt.gca().lines[-1].get_color(), width=0.003, headwidth=6, headlength=7, alpha=0.5
                )
                # plt.quiver(
                #     x_coords_offset[j], y_coords_offset[j], dx, dy, angles='xy', scale_units='xy', scale=1,
                #     color=agent_colors[i], width=0.003, headwidth=6, headlength=7, alpha=0.5
                # )

    # Create dummy handles for legend
    node_handle = plt.Line2D([], [], color='skyblue', marker='o', linestyle='None', markersize=8, label='Waypoints')
    pathway_handle = plt.Line2D([], [], color='skyblue', linestyle='-', linewidth=8, markersize=8, alpha=0.5, label='Air Corridors')
    start_handle = plt.Line2D([], [], color='green', marker='o', linestyle='None', markersize=8, label='Agent Start')
    dest_handle = plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='Agent Destination')
    path_handle = plt.Line2D([], [], color='orange', linestyle='--', markersize=8, label='Agent Paths')
    
    # Set axis labels and title
    # plt.xlabel(rf'$X$')
    # plt.ylabel(rf'$Y$')
    # plt.title('UAV Network of pathways')
    # Show legend for start and destination nodes
    # if plotPaths:
    #     plt.legend(handles=[node_handle, pathway_handle, start_handle, dest_handle, path_handle], loc='lower center', handletextpad=2.0, 
    #     bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    # else:
    #     plt.legend(handles=[node_handle, pathway_handle, start_handle, dest_handle], loc='lower center', handletextpad=2.0, 
    #         bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    # Show the plot
    plt.show()


def plot_vehicle_routes(routes, schedule_matrix, process_T, agent_colors):
    # Get the number of unique vertices
    unique_vertices = sorted(set(v for route in routes for v in route))
    num_vertices = len(unique_vertices)
    
    # Assign a color to each vertex using a colormap
    cmap = get_cmap('Dark2')  # You can change this to any other colormap
    vertex_colors = {vertex: cmap(i / num_vertices) for i, vertex in enumerate(unique_vertices)}
    
    plt.figure(figsize=(10, 4))
    num_vehicles = len(routes)
    plt.ylim(-0.5,num_vehicles-0.5)

    for vehicle_id in range(num_vehicles):
        route = routes[vehicle_id]
        schedule = schedule_matrix[vehicle_id]
        t_process = process_T[vehicle_id]
        # v_max = speedLim[vehicle_id,1]
        
        # Y-coordinate for this vehicle (vehicle ID)
        y = vehicle_id
        
        # Plot the horizontal line for the vehicle
        plt.hlines(y, schedule[0], schedule[-1], colors=agent_colors[vehicle_id], linestyles='dashed', linewidth=1)
        
        # Plot the schedule points with assigned colors
        # vertex_prev = []
        for time, vertex in zip(schedule, route):
            color = vertex_colors[vertex]
            # Vertical line
            # plt.vlines(time, 0, y, color=color, linestyles='dotted', linewidth=1, alpha=0.8)
            plt.plot(time, y, '|', color=color, markersize=5, markeredgewidth=3)
            plt.plot(time+t_process[vertex], y, '|', color=color, markersize=5, markeredgewidth=3)
            # plt.plot(time-t_process[vertex_prev]+dist_mat[vertex_prev,vertex]/v_max, y, '|', color=color, markersize=10, markeredgewidth=1)
            plt.text(time, y, rf'$w_{{{vertex}}}$', color='black', fontsize=12, ha='right', va='bottom')
            # vertex_prev = 
    # Add a legend to show the mapping of colors to vertices
    for vertex, color in vertex_colors.items():
        plt.plot([], [], 'o', color=color, label=f"Vertex {vertex}")
    
    # plt.title("Vehicle Routes and Schedules through Waypoint")
    plt.xlabel("Time (s)")
    plt.ylabel("Agent ID")
    plt.yticks(range(num_vehicles), [rf"$v_{{{i}}}$" for i in range(num_vehicles)])
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Waypoints")
    plt.tight_layout()
    plt.show()


def plot_waypoint_schedules(association_matrix, schedule_matrix):
    na, nwp = association_matrix.shape  # Number of agents and waypoints
    tick_wp = []
    count = 0
    plt.figure(figsize=(10, 4))
    
    for waypoint_idx in range(nwp):
        # plot only if at least one agent passes through that waypoint
        if association_matrix[:, waypoint_idx].sum() > 0.0:
            # Extract the schedule times and agent indices for the current waypoint
            agent_indices = np.where(association_matrix[:, waypoint_idx] == 1.0)[0]
            schedule_times = schedule_matrix[agent_indices, waypoint_idx]
            
            # Sort agents by their schedule times
            sorted_indices = np.argsort(schedule_times)
            sorted_agents = agent_indices[sorted_indices]
            sorted_times = schedule_times[sorted_indices]
            
            # Plot the horizontal line for the waypoint
            plt.hlines(count, sorted_times.min(), sorted_times.max(), colors='gray', linestyles='dashed', linewidth=1)
            
            # Plot the markers for each agent at their respective schedule time
            for time, agent in zip(sorted_times, sorted_agents):
                plt.plot(time, count, 'o', label=f'Agent {agent+1}' if waypoint_idx == 0 else "", markersize=5)
                plt.text(time, count, rf'$v_{agent}$', fontsize=12, ha='right', va='bottom', color='black')

            tick_wp.append(waypoint_idx)
            count += 1
            # plt.yticks(range(nwp), rf'$w_{i+1}$')

    # Customize the plot
    # plt.title("Agent Schedules at Waypoints")
    plt.ylim(-0.2,len(tick_wp)-0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Waypoint ID")
    plt.yticks(range(len(tick_wp)), [rf'$w_{{{i}}}$' for i in tick_wp])
    # plt.xticks([])
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Agents")
    plt.tight_layout()
    plt.show()


def plot_waypoint_agent_schedules(
    routes, schedules, schedule_matrix, association_matrix, 
    process_T, agent_colors, figuresize):
    # Get the number of unique vertices
    unique_vertices = sorted(set(v for route in routes for v in route))
    num_vertices = len(unique_vertices)
    
    # Assign a color to each vertex using a colormap
    cmap = get_cmap('Dark2')  # You can change this to any other colormap
    vertex_colors = {vertex: cmap(i / num_vertices) for i, vertex in enumerate(unique_vertices)}

    plt.figure(figsize=figuresize)

    plt.subplot(2,1,1)
    num_vehicles = len(routes)
    plt.ylim(-0.5,num_vehicles-0.1)

    for vehicle_id in range(num_vehicles):
        route = routes[vehicle_id]
        schedule = schedules[vehicle_id]
        t_process = process_T[vehicle_id]
        # v_max = speedLim[vehicle_id,1]
        
        # Y-coordinate for this vehicle (vehicle ID)
        y = vehicle_id
        
        # Plot the horizontal line for the vehicle
        plt.hlines(y, schedule[0], schedule[-1], colors=agent_colors[vehicle_id], linestyles='dashed', linewidth=2)
        
        # Plot the schedule points with assigned colors
        # vertex_prev = []
        for time, vertex in zip(schedule, route):
            color = vertex_colors[vertex]
            # Vertical line
            # plt.vlines(time, 0, y, color=color, linestyles='dotted', linewidth=1, alpha=0.8)
            plt.plot(time, y, '|', color=color, markersize=10, markeredgewidth=3)
            plt.plot(time+t_process[vertex], y, '|', color=color, markersize=5, markeredgewidth=3)
            # plt.plot(time-t_process[vertex_prev]+dist_mat[vertex_prev,vertex]/v_max, y, '|', color=color, markersize=10, markeredgewidth=1)
            plt.text(time, y, rf'${{{vertex}}}$', color='black', fontsize=22, ha='center', va='bottom')
            # vertex_prev = 
    # Add a legend to show the mapping of colors to vertices
    for vertex, color in vertex_colors.items():
        plt.plot([], [], 'o', color=color, label=f"Vertex {vertex}")
    
    # plt.title("Vehicle Routes and Schedules through Waypoint")
    # plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel(r"$a_j$", fontsize=18)
    plt.yticks(range(num_vehicles), [rf"${{{i}}}$" for i in range(num_vehicles)], fontsize=18)
    plt.xticks(fontsize=18)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Waypoints")
    plt.tight_layout()


    na, nwp = association_matrix.shape  # Number of agents and waypoints
    tick_wp = []
    count = 0
    plt.subplot(2,1,2)
    for waypoint_idx in range(nwp):
        # plot only if at least one agent passes through that waypoint
        if association_matrix[:, waypoint_idx].sum() > 0.0:
            # Extract the schedule times and agent indices for the current waypoint
            agent_indices = np.where(association_matrix[:, waypoint_idx] == 1.0)[0]
            schedule_times = schedule_matrix[agent_indices, waypoint_idx]
            
            # Sort agents by their schedule times
            sorted_indices = np.argsort(schedule_times)
            sorted_agents = agent_indices[sorted_indices]
            sorted_times = schedule_times[sorted_indices]
            
            # Plot the horizontal line for the waypoint
            plt.hlines(count, sorted_times.min(), sorted_times.max(), colors='gray', linestyles='dashed', linewidth=2)
            
            # Plot the markers for each agent at their respective schedule time
            for time, agent in zip(sorted_times, sorted_agents):
                plt.plot(time, count, 'o', label=f'Agent {agent+1}' if waypoint_idx == 0 else "", markersize=5)
                plt.text(time, count, rf'${agent}$', fontsize=22, ha='center', va='bottom', color='black')

            tick_wp.append(waypoint_idx)
            count += 1
            # plt.yticks(range(nwp), rf'$w_{i+1}$')

    # Customize the plot
    # plt.title("Agent Schedules at Waypoints")
    plt.ylim(-0.5,len(tick_wp)-0.1)
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel(r"$w_i$", fontsize=18)
    plt.yticks(range(len(tick_wp)), [rf'${{{i}}}$' for i in tick_wp], fontsize=18)
    plt.xticks(fontsize=18)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Agents")
    plt.tight_layout()
    plt.show()
    