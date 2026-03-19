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
from matplotlib import cm  # for colormap
import cvxpy as cp
from tabulate import tabulate
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib as mpl
import gurobipy as gp
from gurobipy import GRB


# create a multiagent routing and scheduling class
class MARS():

    def __init__(
        self, 
        n_waypoints         :int,
        n_agents            :int, 
        tolArray            :np.ndarray, 
        wp_params           :np.ndarray, 
        seed                :int, 
        offset_energy       :bool,
        stagewiseCostCoeffs :np.ndarray,
        selfHop             :bool,
        cost_mode           :str,
        lm                  :float,
        ca_cbf              :str,
        filter_wp_thresh    :float,
        prune_mode          :bool,
        printFlag           :bool
        ):

        # self.n_waypoints = n_waypoints
        self.n_agents = n_agents
        self.tolArray = tolArray
        self.INF = 1e8
        self.offset_energy = offset_energy
        self.selfHop = selfHop
        self.initWaypoints(wp_params, seed=seed)
        self.initAgentParams(seed=seed)
        self.stagewiseCostCoeffs = stagewiseCostCoeffs
        self.initFlpoAgents()
        self.printInitializationData(printFlag)
        self.active_waypoints = range(self.n_waypoints)
        self.ca_cbf = ca_cbf
        self.cost_mode = cost_mode
        self.lm = lm
        self.prune_mode = prune_mode
        self.filter_wp_thresh = filter_wp_thresh

    # function to create waypoints and the corresponding adjacency matrix
    def initWaypoints(self, wp_params:dict, seed:int):
        np.random.seed(seed)
        random.seed(seed)

        if wp_params['type']=='grid':
            self.wp_locations, self.mask = supporting_functions.generate_non_uniform_grid_graph_numpy(wp_params)
        elif wp_params['type']=='ring':
            self.wp_locations, self.mask = supporting_functions.generate_ring_network(wp_params)
        elif wp_params['type']=='multigraph':
            self.wp_locations, self.mask = supporting_functions.generate_multigraph(wp_params)
        self.dist_mat = cdist(self.wp_locations, self.wp_locations, 'euclidean')
        self.dist_mat[self.mask==0] = self.INF
        # wp_weights = np.random.uniform(0,1,nw)
        self.n_waypoints = len(self.wp_locations)
        wp_weights = np.ones(self.n_waypoints)
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
        agent_weights = np.ones(na)
        self.agent_weights = agent_weights/np.sum(agent_weights)
        # self.sd_mat = np.random.choice(range(nw), (na,2))[:, :2]
        # mat = np.zeros((na, 2), dtype=int)
        # for i in range(na):
        #     mat[i] = np.random.choice(nw, size=2, replace=False)
        self.sd_mat = np.array([np.random.choice(nw, size=2, replace=False) for i in range(na)])
        # self.sd_mat = np.random.rand(na, nw).argsort(axis=1)
        min_speeds = np.random.uniform(0.01,1,na).reshape(-1,1)
        max_speeds = np.random.uniform(70,80,na).reshape(-1,1)
        self.speed_lim_mat = np.concatenate((min_speeds,max_speeds), axis=1)
        self.speed_vec = self.speed_lim_mat.mean(axis=1)
        max_time = np.max(cdist(self.wp_locations, self.wp_locations, 'euclidean'))/np.min(min_speeds)
        self.sched_mat = np.random.uniform(0.0, 50.0, (na, nw))
        self.start_times = np.random.uniform(0.0, 0.0, na)
        self.sched_mat[np.arange(na),self.sd_mat[:,0]] = self.start_times
        self.T_upper_bound = 600
        self.process_T = np.random.uniform(3,5,(na, nw))*0 
        self.process_T[np.arange(na),self.sd_mat[:,1]] = np.zeros(na)
        np.random.seed(None)
        random.seed(None)
        pass

    def set_offset_energy(self,flag):
        for a in self.agents:
            a.offset_energy = flag

    # function to initialize agents
    def initFlpoAgents(self):
        list_agents = []
        na = self.n_agents
        nw = self.n_waypoints
        for i in range(na):
            v = flpoAgent.flpoAgent(
                n_wp=nw, 
                sd=self.sd_mat[i,:], 
                sched=self.sched_mat[i,:],
                start_time=self.start_times[i],
                speed=self.speed_vec[i],
                speedLim=self.speed_lim_mat[i,:], 
                process_T=self.process_T[i,:], 
                INF=self.INF,
                offset_energy=self.offset_energy,
                selfHop=self.selfHop,
                net_mask=self.mask,
                stagewise_cost_coeffs=self.stagewiseCostCoeffs)
            list_agents.append(v)
        self.agents = list_agents
        pass

    # function to print initialization data
    def printInitializationData(self,printFlag):
        if printFlag:
            print(f'n_waypoints: {self.n_waypoints} \nn_agents: {self.n_agents} \nCAT:\n{self.tolArray}')
            print('---------')
            print(f'wp_locations:\n{self.wp_locations} \nmask:\n{self.mask} \ndist_mat:\n{self.dist_mat} \nwp_weights:\n{self.wp_weights}')
            print('---------')
            print(f'agent_weights:\n{self.agent_weights} \nsd_mat:\n{self.sd_mat} \neta_arr: \nprocessing_time:\n{self.process_T} \nspeed_lim_mat:\n{self.speed_lim_mat} \nsched_mat:\n{self.sched_mat} \nspeed_vec:\n{self.speed_vec}')

    def calc_agent_reach_mat_v1(self, sched_mat, speed_vec, beta):
        reach_mat = []
        for i,ag in enumerate(self.agents):
            Pb = ag.getPathAssociations_v1(sched_mat[i,:], speed_vec[i], self.dist_mat, beta)
            ag_reach = ag.calc_agent_wp_association(Pb)[0][0]
            reach_mat.append(ag_reach)
        return np.array(reach_mat)

    # function to compute total vehicle cost of transportation based on mean speed
    def transportCost_v1(
        self, 
        sched_mat:np.ndarray, 
        speed_vec:np.ndarray, 
        beta:float
        ):

        C_arr = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            v = self.agents[i]
            C_arr[i] = v.getFreeEnergy_s_v1(sched_mat[i,:], speed_vec[i], self.dist_mat, beta)
        self.C_agents = C_arr

        if self.cost_mode == 'sum':
            transport_cost = np.sum(self.agent_weights * C_arr)
        elif self.cost_mode == 'slowest':
            C_max = np.max(C_arr)
            transport_cost = 1/self.lm * np.log(np.sum(np.exp(self.lm*(C_arr-C_max)))) + C_max
        
        return transport_cost


    def grad_transportCost_v1(self, sched_mat, speed_vec, beta):
        # compute gradient of free energy of all UAVs w.r.t sched_mat, speed
        Grad_F = np.zeros(shape=(self.n_agents, self.n_waypoints+1))
        # the following loop is parallelizable
        for i,a in enumerate(self.agents):
            GD_a = a.returnStagewiseGrad_v1(sched_mat[i,:], speed_vec[i], self.dist_mat)
            P_a = a.getPathAssociations_v1(sched_mat[i,:], speed_vec[i], self.dist_mat, beta)
            G_Fa, _ = a.backPropDP_grad(GD_a, P_a)
            Grad_F[i,:] = G_Fa

        return Grad_F


    def CBF_waypoints(self, sched_mat, waypoints, returnGrad=True):
        Nw = len(waypoints)
        Na = self.n_agents

        # define gradient as a 3D tensor
        H = np.zeros((Na * (Na-1)//2, Nw))
        Grad_H = np.zeros((Nw, Na * (Na-1)//2, Na))

        if self.ca_cbf['mode'] == 'linear':
            eps = self.ca_cbf['eps']
            # the following loop over waypoints is parallelizable
            for i, wp in enumerate(waypoints):
                # print(f'i:{i}\twp:{wp}')
                Ti = sched_mat[:,wp]
                KTi = Ti - Ti.reshape(-1,1)
                Hi_mat = KTi**2
                Hi_triu = np.triu_indices_from(Hi_mat, k=1)
                H[:,i] = Hi_mat[Hi_triu] - eps[wp]**2
                if returnGrad==True:
                    # Compute gradient
                    start_row = 0
                    n_rows = Na-1
                    for j in range(Na-1):
                        Grad_H[i, start_row : start_row + n_rows, j] = 2*KTi[j+1:, j]
                        Grad_H[i, start_row : start_row + n_rows, j+1:] = np.diag(2*KTi[j,j+1:])
                        start_row = start_row + n_rows
                        n_rows = n_rows-1
        
        elif self.ca_cbf['mode'] == 'ellipse':
            a, b, n = self.ca_cbf['major_axis'], self.ca_cbf['minor_axis'], self.ca_cbf['degree']

            # the following loop over waypoints is parallelizable
            for i, wp in enumerate(waypoints):

                Ti = sched_mat[:,wp]
                KTi1 = (Ti + Ti.reshape(-1,1))
                KTi2 = (Ti - Ti.reshape(-1,1))
                Hi_mat = (np.abs(KTi1)*b[wp])**n + (np.abs(KTi2)*a[wp])**n - (a[wp]*b[wp])**n
                Hi_triu = np.triu_indices_from(Hi_mat, k=1)
                H[:,i] = Hi_mat[Hi_triu]

                if returnGrad==True:
                    # Compute gradient 
                    start_row = 0 
                    n_rows = Na-1 
                    for j in range(Na-1): 
                        if n%2 == 0:
                            Grad_H[i, start_row : start_row + n_rows, j] = n*KTi1[j+1:, j]**(n-1)*b[wp]**n + n*KTi2[j+1:, j]**(n-1)*a[wp]**n 
                            Grad_H[i, start_row : start_row + n_rows, j+1:] = np.diag(n*KTi1[j,j+1:]**(n-1)*b[wp]**n) + np.diag(n*KTi2[j,j+1:]**(n-1)*a[wp]**n) 
                        else:
                            Grad_H[i, start_row : start_row + n_rows, j] = n*KTi1[j+1:, j]**(n-1)*np.sign(KTi1[j+1:, j])*b[wp]**n + n*KTi2[j+1:, j]**(n-1)*np.sign(KTi2[j+1:, j])*a[wp]**n 
                            Grad_H[i, start_row : start_row + n_rows, j+1:] = np.diag(n*KTi1[j,j+1:]**(n-1)*np.sign(KTi1[j,j+1:])*b[wp]**n) + np.diag(n*KTi2[j,j+1:]**(n-1)*np.sign(KTi2[j,j+1:])*a[wp]**n) 
                        start_row = start_row + n_rows 
                        n_rows = n_rows-1

        return H, Grad_H 

    def CBF_waypoints_v1(self, sched_mat, waypoints, weight_mat, returnGrad=True):
        Nw = len(waypoints)
        Na = self.n_agents
        # calculate filter matrix
        filter_wp = np.ones(weight_mat.shape)
        if self.prune_mode == True:
            filter_wp[weight_mat <= self.filter_wp_thresh] = 0.0

        # define gradient as a 3D tensor
        H = np.array([])
        Grad_H = np.empty((0,Na))

        # if self.ca_cbf['mode'] == 'linear':
        #     eps = self.ca_cbf['eps']
        #     # the following loop over waypoints is parallelizable
        #     for i, wp in enumerate(waypoints):
        #         n_active_agents = sum(filter_wp[:,wp])
        #         if n_active_agents > 1:
        #             Ti = sched_mat[:,wp] 
        #             KTi = Ti - Ti.reshape(-1,1)
        #             Hi_mat = KTi**2
        #             if n_active_agents > 2:
        #                 Hi_triu = np.triu_indices_from(Hi_mat, k=1)
        #                 H = np.concatenate((H,Hi_mat[Hi_triu]))
        #             else:
        #                 H = np.concatenate((H,Hi_mat))

        #             if returnGrad==True:
        #                 # Compute gradient
        #                 start_row = 0
        #                 n_rows = n_active_agents
        #                 Grad_Hi = np.zeros((n_active_agents * (n_active_agents-1)//2, n_active_agents))
        #                 for j in range(n_active_agents-1):
        #                     Grad_Hi[start_row : start_row + n_rows, j] = 2*KTi[j+1:, j]
        #                     Grad_Hi[start_row : start_row + n_rows, j+1:] = np.diag(2*KTi[j,j+1:])
        #                     start_row = start_row + n_rows
        #                     n_rows = n_rows-1
        #             Grad_H = np.concatenate((Grad_H, Grad_Hi),axis=0)    
        
        if self.ca_cbf['mode'] == 'ellipse':
            a, b, n = self.ca_cbf['major_axis'], self.ca_cbf['minor_axis'], self.ca_cbf['degree']

            # the following loop over waypoints is parallelizable
            for i, wp in enumerate(waypoints):
                n_active_agents = int(sum(filter_wp[:,wp]))
                if n_active_agents > 1:
                    Ti = sched_mat[:,wp][filter_wp[:,wp]==1.0]
                    KTi1 = (Ti + Ti.reshape(-1,1))
                    KTi2 = (Ti - Ti.reshape(-1,1))
                    Hi_mat = (np.abs(KTi1)*b[wp])**n + (np.abs(KTi2)*a[wp])**n - (a[wp]*b[wp])**n
                    Hi_triu = np.triu_indices_from(Hi_mat, k=1)
                    H = np.concatenate((H,Hi_mat[Hi_triu]))
                    
                    if returnGrad==True:
                        start_row = 0
                        n_rows = n_active_agents-1
                        Grad_Hi = np.zeros((n_active_agents * (n_active_agents-1)//2, Na))
                        temp_grad = np.zeros((n_active_agents * (n_active_agents-1)//2, n_active_agents))
                        for j in range(n_active_agents-1):
                            if n%2 == 0:
                                temp_grad[start_row : start_row + n_rows, j] = n*KTi1[j+1:, j]**(n-1)*b[wp]**n + n*KTi2[j+1:, j]**(n-1)*a[wp]**n 
                                temp_grad[start_row : start_row + n_rows, j+1:] = np.diag(n*KTi1[j,j+1:]**(n-1)*b[wp]**n) + np.diag(n*KTi2[j,j+1:]**(n-1)*a[wp]**n) 
                            else:
                                temp_grad[start_row : start_row + n_rows, j] = n*KTi1[j+1:, j]**(n-1)*np.sign(KTi1[j+1:, j])*b[wp]**n + n*KTi2[j+1:, j]**(n-1)*np.sign(KTi2[j+1:, j])*a[wp]**n 
                                temp_grad[start_row : start_row + n_rows, j+1:] = np.diag(n*KTi1[j,j+1:]**(n-1)*np.sign(KTi1[j,j+1:])*b[wp]**n) + np.diag(n*KTi2[j,j+1:]**(n-1)*np.sign(KTi2[j,j+1:])*a[wp]**n) 
                            start_row = start_row + n_rows 
                            n_rows = n_rows-1    
                        Grad_Hi[:, filter_wp[:,wp]==1.0] = temp_grad
                        Grad_H = np.concatenate((Grad_H, Grad_Hi),axis=0)

        elif self.ca_cbf['mode'] == 'rect':
            w, h, gamma = self.ca_cbf['width'], self.ca_cbf['height'], self.ca_cbf['gamma']
            ew, eh = self.ca_cbf['width_correction_fac'], self.ca_cbf['height_correction_fac']
            # print(f'inside_rect')
            for i, wp in enumerate(waypoints):
                n_active_agents = int(sum(filter_wp[:,wp]))
                if n_active_agents > 1:
                    Ti = sched_mat[:,wp][filter_wp[:,wp]==1.0]
                    KTi1 = (Ti + Ti.reshape(-1,1))
                    KTi2 = (Ti - Ti.reshape(-1,1))
                    a = KTi1**2 - (w[i]*(1+ew))**2/2
                    b = KTi2**2 - (h[i]*(1+eh))**2/2
                    ab = np.concatenate((np.array([a]),np.array([b])))
                    ab_max = np.max(ab, axis=0)
                    ab_shift = ab - ab_max
                    exp_ab = np.exp(gamma * ab_shift)
                    sum_exp_ab = np.sum(exp_ab, axis=0)
                    p_ab = exp_ab / (sum_exp_ab)
                    Hi_mat = 1/gamma * np.log(sum_exp_ab) + ab_max
                    Hi_triu = np.triu_indices_from(Hi_mat, k=1)
                    H = np.concatenate((H,Hi_mat[Hi_triu]))

                    if returnGrad==True:
                        start_row = 0
                        n_rows = n_active_agents-1
                        Grad_Hi = np.zeros((n_active_agents * (n_active_agents-1)//2, Na))
                        temp_grad = np.zeros((n_active_agents * (n_active_agents-1)//2, n_active_agents))
                        for j in range(n_active_agents-1):
                            temp_grad[start_row : start_row + n_rows, j] =  2*(p_ab[0]*KTi1)[j+1:, j] + 2*(p_ab[1]*KTi2)[j+1:, j]
                            temp_grad[start_row : start_row + n_rows, j+1:] = np.diag(2*(p_ab[0]*KTi1)[j,j+1:]) + np.diag(2*(p_ab[1]*KTi2)[j,j+1:]) 
                            start_row = start_row + n_rows 
                            n_rows = n_rows-1    
                        Grad_Hi[:, filter_wp[:,wp]==1.0] = temp_grad
                        Grad_H = np.concatenate((Grad_H, Grad_Hi),axis=0)

        return H, Grad_H, filter_wp


    def CBF_agents(self, speed_vec):
        H = (speed_vec - self.speed_lim_mat[:,0])*(self.speed_lim_mat[:,1]-speed_vec)
        gradH = np.diag(self.speed_lim_mat.sum(axis=1) - 2 * speed_vec)
        return H, gradH


    def get_CBF_control(
        self, 
        sched_mat, 
        speed_vec, 
        weight_mat, 
        beta, 
        gamma, 
        alpha_s, 
        alpha_c, 
        alpha_r, 
        alpha_q, 
        p):

        Nw = self.n_waypoints
        Na = self.n_agents
        # control decision variables
        U = cp.Variable((Na, Nw+1))
        delta = cp.Variable(1)

        # get free energy and its dot
        F = self.transportCost_v1(sched_mat, speed_vec, beta)
        # F = np.sum(self.agent_weights * self.C_agents)
        Grad_F = self.grad_transportCost_v1(sched_mat, speed_vec, beta)
        if self.cost_mode == 'sum':
            F_dot = cp.sum(cp.multiply(self.agent_weights, cp.sum(cp.multiply(Grad_F, U), axis=1)))
        elif self.cost_mode == 'slowest':
            C_arr_lm = self.lm * self.C_agents
            C_arr_max = np.max(C_arr_lm)
            weights1 = np.exp(C_arr_lm - C_arr_max) / np.sum(np.exp(C_arr_lm - C_arr_max))
            F_dot = cp.sum(cp.multiply(weights1, cp.sum(cp.multiply(Grad_F, U), axis=1)))

        # speed speed cbf and its dot
        H_speed, Grad_H_speed = self.CBF_agents(speed_vec)
        Grad_H_diag = Grad_H_speed.diagonal().copy()
        H_speed_dot = cp.multiply(Grad_H_diag, U[:,-1])

        # define objective, constraints and problem
        objective = cp.Minimize(cp.sum_squares(U) + p*delta**2)
        
        # pick agent start schedules and its dot
        start_indices = np.array([[i, a.s] for i, a in enumerate(self.agents)])
        tup_start_indices = (start_indices[:,0], start_indices[:,1])

        # compute conflict barrier functions
        H, Grad_H, filter_wp = self.CBF_waypoints_v1(sched_mat, self.active_waypoints, weight_mat, returnGrad=True)

        # bypass CBF constraint if no conflicts recorded in H
        # if self.active_waypoints == [] or self.active_waypoints == None:
        if len(H) == 0:
            constraints = [
                F_dot <= -gamma * F + delta, # to decrease free energy
                H_speed_dot >= -alpha_s * H_speed, # speed limit constraint
                U[tup_start_indices] >= -alpha_r * (sched_mat[tup_start_indices]-self.start_times), # to maintain time-positivity
                U[:,:-1] <= alpha_q * (self.T_upper_bound - sched_mat) # to maintain the schedule bounded
                # U[dropped_indices] == 0.0 # no control for unvisited nodes
            ]
        else:
            # H, Grad_H, filter_wp = self.CBF_waypoints_v1(sched_mat, self.active_waypoints, weight_mat, returnGrad=True)
            H_dot_list = []
            cnt = 0
            for wp in self.active_waypoints:
                n_active_agents = int(np.sum(filter_wp[:,wp]))
                if n_active_agents > 1:
                    n_conflicts = n_active_agents * (n_active_agents-1) // 2
                    H_dot_list.append(
                        cp.sum(cp.multiply(Grad_H[cnt:cnt+n_conflicts], cp.reshape(U[:,wp], (1,Na), order='C')), axis=1, keepdims=True)
                    )
                    cnt += n_conflicts
            # print(H, Grad_H, H_dot_list)
            H_dot = cp.vstack(H_dot_list) 
            H_dot = cp.reshape(H_dot, (-1,), order='C')

            constraints = [
                F_dot <= -gamma * F + delta, # decrease free energy
                H_speed_dot >= -alpha_s * H_speed, # speed limit constraint
                H_dot >= -alpha_c * H, # collision avoidance constraint
                U[tup_start_indices] >= -alpha_r * (sched_mat[tup_start_indices]-self.start_times), # to maintain time-positivity
                U[:,:-1] <= alpha_q * (self.T_upper_bound - sched_mat) # to maintain the schedule bounded
                # U[dropped_indices] == 0.0 # no control for unvisited nodes
            ]

        problem = cp.Problem(objective, constraints)

        # Solver Options for OSQP
        solver_options = {
            'max_iter': 100000,         # Increase max iterations to 20000
            'eps_abs': 1e-3,           # Adjust absolute tolerance
            'eps_rel': 1e-3,           # Adjust relative tolerance
            'eps_prim_inf': 1e-2,      # Adjust primal infeasibility tolerance
            'eps_dual_inf': 1e-2,      # Adjust dual infeasibility tolerance
            'verbose': False           # Enable verbose output to track solver progress
        }

        # Solve the problem using OSQP with customized options
        result = problem.solve(solver = 'OSQP', **solver_options)

        # Check the results
        if np.isnan(problem.value).any() == True:
            print("Nan encountered inside get_CBF_control!")
            return None, None, None, None
        elif U.value is None or F_dot.value is None:
            print("None type returned inside get_CBF_control!")
            return None, None, None, None
        else:
            return U.value, F, F_dot.value, delta.value

    # optimize at given beta using CBF CLF
    def CBF_CLF_at_beta(
        self,
        beta, 
        Tb, 
        Vb,
        dt_init, 
        dt_min, 
        dt_max, 
        Tf, 
        gamma, 
        alpha_s,
        alpha_c,
        alpha_r, 
        alpha_q,
        p, 
        weight_mat,
        stop_tol,
        stop_tol_weight,
        verbose=0
        ):
    
        T_prev = Tb
        V_prev = Vb
        dt_prev = dt_init
        theta_prev = np.inf
        t = 0.0
        iter_count = 0
        F_prev = 1.0

        while t < Tf:
            # get control
            U, F, Fdot, delta = self.get_CBF_control(
                T_prev, V_prev, 
                weight_mat, beta, 
                gamma, alpha_s, alpha_c, 
                alpha_r, alpha_q, p)

            if None in (F, Fdot, delta):
                print('None encountered from get_CBF_control... returning previous iterate!')
                return T_prev, V_prev, np.zeros(shape=(self.n_agents,self.n_waypoints+1)), 0.0, 0.0, 0.0, 0.0

            # compute new stepsize
            if iter_count > 0:
                step_size_1 = np.sqrt(1 + theta_prev) * dt_prev
                grad_diff = np.linalg.norm(U - U_old) + 1e-6  # Regularization term to prevent division by zero
                step_size_2 = (np.linalg.norm(T_prev - T_old) + np.linalg.norm(V_prev - V_old)) / (2 * grad_diff)  
                dt = min(max(step_size_2, dt_min), dt_max)  # Keep dt in range [dt_min, dt_max]
            else:
                dt = dt_init

            # Euler update
            T_next = T_prev + dt * U[:,:-1]
            V_next = V_prev + dt * U[:,-1]

            # compute new theta_k
            if iter_count > 0:
                theta = dt/dt_prev
            else:
                theta = 1.0

            tol_T = np.max(np.abs(T_next-T_prev))/np.max(np.abs(T_prev))
            tol_V = np.max(np.abs(V_next-V_prev))/np.max(np.abs(V_prev))
            tol_F = np.abs(F - F_prev)/np.abs(F_prev)
            tol_Fdot = abs(Fdot * dt)/np.abs(F_prev)
            stop_tol_weight = stop_tol_weight/np.sum(stop_tol_weight)
            tol = np.sum(stop_tol_weight * np.array([tol_T, tol_V, tol_F, tol_Fdot]))
            if tol < stop_tol:
                if verbose == 1 or verbose == 2:
                    print(f'\tt:{t:.3e}\tdt:{dt:.4e}\tF:{F:.4f}\tFdot:{Fdot:.4f}\tdelta:{delta[0]:.4f}\ttol:{tol:.3e}')
                break
            if verbose == 2:
                print(f'\tt:{t:.3e}\tdt:{dt:.4e}\tF:{F:.4f}\tFdot:{Fdot:.4f}\tdelta:{delta[0]:.4f}\ttol:{tol:.3e}')
            
            # Update variables for next iteration
            T_old = T_prev
            V_old = V_prev
            T_prev = T_next
            V_prev = V_next
            U_old = U
            dt_prev = dt
            theta_prev = theta
            t += dt
            iter_count += 1
            F_prev = F

        return T_prev, V_prev, U, F, Fdot, tol, delta[0]


    # function to perform optimization iterations at a given beta
    def optimize_slsqp_v1(
        self, 
        beta,
        Tb, 
        Vb, 
        weight_mat,
        stop_tol,
        disp):
        
        t0 = time.time()
        Na = self.n_agents
        Nw = self.n_waypoints
        xb = np.concatenate((Tb.flatten(),Vb.flatten()))

        # cost function
        F = lambda x : self.transportCost_v1(x[0:Na*Nw].reshape(-1,Nw), x[Na*Nw:], beta)
        gradF = lambda x : self.grad_transportCost_v1(x[0:Na*Nw].reshape(-1,Nw), x[Na*Nw:], beta)

        # collision avoidance constraints
        # H = lambda x : self.CBF_waypoints(x[0:Na*Nw].reshape(-1,Nw), self.active_waypoints, returnGrad=False)[0].flatten()
        H = lambda x : self.CBF_waypoints_v1(x[0:Na*Nw].reshape(-1,Nw), self.active_waypoints, weight_mat, returnGrad=False)[0].flatten()
        # gradH = lambda x : self.CBF_waypoints(x[0:Na*Nw].reshape(-1,Nw), self.active_waypoints, returnGrad=True)[1].reshape(1,-1)

        # collision avoidance inequality
        # collision_avoid_ineq = {'type':'ineq', 'fun':H, 'jac':gradH}
        collision_avoid_ineq = {'type':'ineq', 'fun':H}

        # pick agent start schedules and its dot
        start_indices = np.array([[i, a.s] for i, a in enumerate(self.agents)])
        tup_start_indices = (start_indices[:,0], start_indices[:,1])

        # schedule bounds
        lbT, ubT = -np.inf*np.ones(Tb.shape), np.ones(Tb.shape)*self.T_upper_bound
        lbT[tup_start_indices] = self.start_times

        # bounds
        lb = np.concatenate((lbT.flatten(), self.speed_lim_mat[:,0]))
        ub = np.concatenate((ubT.flatten(), self.speed_lim_mat[:,1]))
        # print(F(xb), gradF(xb).shape, xb.shape, lb.shape, ub.shape, H(xb).shape, gradH(xb).shape)
        res = minimize(
            F, xb,
            # jac = gradF,
            method='SLSQP',
            constraints=[collision_avoid_ineq],
            bounds = Bounds(lb, ub),
            options={'disp':disp, 'ftol':stop_tol})

        xb1 = res.x
        Tb1 = xb1[0:Na*Nw].reshape(-1,Nw)
        Vb1 = xb1[Na*Nw:]
        cost_fun = res.fun
        computeTime = time.time() - t0
        return Tb1, Vb1, cost_fun, computeTime

    def anneal(
        self,
        beta_arr,
        T0,
        V0,
        active_waypoints,
        optimizer,
        annealPrint=False,
        ):

        Tb = T0
        Vb = V0
        ra = 0.01
        rb = 0.01
        rw = 0.001
        rh = 0.01

        if optimizer['name'] == 'cbf_clf':
            dt_init         = optimizer['dt_init']
            dt_min          = optimizer['dt_min']
            dt_max          = optimizer['dt_max']
            Tf              = optimizer['Tf']
            gamma           = optimizer['gamma']
            alpha_s         = optimizer['alpha_s']
            alpha_c         = optimizer['alpha_c']
            alpha_r         = optimizer['alpha_r']
            alpha_q         = optimizer['alpha_q']
            p               = optimizer['p']
            stop_tol        = optimizer['stop_tol']
            stop_tol_weight = optimizer['stop_tol_weight']
            verbose         = optimizer['verbose']

        elif optimizer['name'] == 'SLSQP':
            stop_tol        = optimizer['stop_tol']
            disp            = optimizer['disp']

        for i, beta in enumerate(beta_arr):
            t0 = time.time()
            weight_mat = self.calc_agent_reach_mat_v1(Tb, Vb, beta)
            filter_wp = np.ones(weight_mat.shape)
            filter_wp[weight_mat <= self.filter_wp_thresh] = 0.0
            if active_waypoints is None:
                self.active_waypoints = list(np.where(np.sum(filter_wp, axis=0)>=2)[0])
            else:
                self.active_waypoints = active_waypoints

            if self.ca_cbf['mode'] == 'ellipse':
                if beta <= 1:
                    self.ca_cbf['major_axis'] = ra**2 / (ra**2 + 1) * np.ones(self.tolArray.shape) * 200
                    self.ca_cbf['minor_axis'] = rb**2 / (rb**2 + 1) * self.tolArray
                else:
                    self.ca_cbf['major_axis'] = ra**3 / (ra**3 + 1) * np.ones(self.tolArray.shape) * 200
                    self.ca_cbf['minor_axis'] = rb**3 / (rb**3 + 1) * self.tolArray
                ra = ra*2
                rb = rb*2
                if annealPrint:
                    a, b = self.ca_cbf['major_axis'][0], self.ca_cbf['minor_axis'][0]
                    print(f'a:{a:.3f}\tb:{b:.3f}')
            elif self.ca_cbf['mode'] == 'rect':
                self.ca_cbf['width'] = rw**2 / (rw**2 + 1) * np.ones(self.tolArray.shape) * self.T_upper_bound
                self.ca_cbf['height'] = rh/ (rh + 1) * self.tolArray
                rw = rw*2
                rh = rh*2
                if annealPrint:
                    w, h = self.ca_cbf['width'][0], self.ca_cbf['height'][0]
                    print(f'w:{w:.3f}\th:{h:.3f}')

            if optimizer['name'] == 'cbf_clf':
                Tb, Vb, Ub, Fb, Fdot_b, tolb, delta_b = self.CBF_CLF_at_beta(
                    beta, Tb, Vb, dt_init, dt_min, dt_max, 
                    Tf, gamma, alpha_s, alpha_c, alpha_r, alpha_q, p,
                    weight_mat, stop_tol=stop_tol, 
                    stop_tol_weight=stop_tol_weight/np.sum(stop_tol_weight), 
                    verbose=verbose)
                Hb, GradHb, filter_wp = self.CBF_waypoints_v1(Tb, self.active_waypoints, weight_mat, returnGrad=True)
                
                if annealPrint:
                    print(f'\nbeta: {beta:.4e}\tcost: {Fb:.3f}\ttolb: {tolb:.3e}\tn_active_waypoints:{len(self.active_waypoints)}\ttol_mag:{self.tolArray[0]:.2f}\tHbshape:{Hb.shape}')

                print(Hb, GradHb)

            elif optimizer['name'] == 'SLSQP':
                Tb, Vb, Fb, compute_time = self.optimize_slsqp_v1(
                    beta, Tb, Vb, weight_mat, stop_tol, disp
                )
                if annealPrint:
                    print(f'\nbeta: {beta:.4e}\tcost: {Fb:.3f}\ttol_mag:{self.tolArray[0]:.2f}')
            t1 = time.time()

            if i == 0:
                Tb_array = np.array([Tb])
                Vb_array = np.array([Vb])
                chi_array = np.array([weight_mat])
                t_compute_array = np.array([t1-t0])
            else:
                Tb_array = np.concatenate((Tb_array, np.array([Tb])))
                Vb_array = np.concatenate((Vb_array, np.array([Vb])))
                chi_array = np.concatenate((chi_array, np.array([weight_mat])))
                t_compute_array = np.concatenate((t_compute_array, np.array([t1-t0])))

        # compute final probability associations
        Pb_a = []
        for i,a in enumerate(self.agents):
            Pb = a.getPathAssociations_v1(Tb[i,:], Vb[i], self.dist_mat, beta_arr[-1])
            Pb_a.append(Pb)

        return Tb_array, Vb_array, Fb, Pb_a, chi_array, t_compute_array


def calc_agent_routes_and_schedules(mars:MARS, Pb_a:list, printRoutes=False):
    routes = []
    fin_schedules = []
    fin_speeds = []

    table_data = []
    headers = ["A", "R", "T", "V", "V Mean", "V Max", "V Lim", "Cost"]
    
    for i, a in enumerate(mars.agents):
        a.calc_route_and_schedule(sched=mars.sched_mat[i,:], dist_mat=mars.dist_mat, Pb=Pb_a[i])
        routes.append(a.route)
        fin_schedules.append(a.fin_sched)
        fin_speeds.append(a.fin_avg_speed)
        
        if printRoutes:
            row = [
                f"v{i}",
                str(a.route),
                np.round(a.fin_sched, 2),
                f"{mars.speed_vec[i]:.2f}",
                f"{a.fin_avg_speed:.2f}",
                f"{np.max(a.route_speed):.2f}",
                np.round(mars.speed_lim_mat[i], 2)
            ]
            table_data.append(row)

    if printRoutes:
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))

    return routes, fin_schedules


def show_solution_table(mars:MARS, Pb_a:list, beta:float, printRoutes=False):
    routes = []
    fin_schedules = []
    fin_speeds = []

    table_data = []
    headers = ["A", "R", "T", "V", "V Mean", "V Max", "V Lim", "Cost"]

    mars.transportCost_v1(
        mars.sched_mat, 
        mars.speed_vec, 
        beta=beta) 

    for i, a in enumerate(mars.agents):
        a.calc_route_and_schedule(sched=mars.sched_mat[i,:], dist_mat=mars.dist_mat, Pb=Pb_a[i])
        routes.append(a.route)
        fin_schedules.append(a.fin_sched)
        fin_speeds.append(a.fin_avg_speed)
        
        if printRoutes:
            row = [
                f"v{i}",
                str(a.route),
                np.round(a.fin_sched, 2),
                f"{mars.speed_vec[i]:.2f}",
                f"{a.fin_avg_speed:.2f}",
                f"{np.max(a.route_speed):.2f}",
                np.round(mars.speed_lim_mat[i], 2),
                f"{mars.C_agents[i]:.2f}"
            ]
            table_data.append(row)

    if printRoutes:
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))

    return routes, fin_schedules


# function to show the network
def plotNetwork(mars:MARS, figuresize, routes, agent_colors, showEdgeLength=True):

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
            label = ', '.join([rf'$a_{{{i}}}$' for i in agents])  # Combined label for multiple agents
        else:
            label = rf'$a_{{{agents[0]}}}$'
        plt.text(start_x+0.0, start_y+0.0, label, color='darkgreen', fontsize=18,
                ha='left', va='top', fontweight='bold')
    for dest_idx, agents in destination_groups.items():
        dest_x, dest_y = wp_xy[dest_idx]
        if len(agents) > 1:
            label = ', '.join([rf'$a_{{{i}}}$' for i in agents])  # Combined label for multiple agents
        else:
            label = rf'$a_{{{agents[0]}}}$'
        plt.text(dest_x + 0.0, dest_y + 0.0, label, color='red', fontsize=18,
                ha='left', va='bottom', fontweight='bold')

    # Draw each agent's path
    if routes != []:
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
    # if routes != []:
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


def plot_waypoint_agent_schedules(
    routes, schedules, schedule_matrix, association_matrix,
    process_T, tolArray, agent_colors, figuresize,
    bar_thickness=0.1, marker_size=8, start_marker_size=15, index_size=24,
    x_tick_size=20, y_tick_size=20, x_label_size=20, y_label_size=20
):
    """
    Visualizes:
      1. Vehicle routes and their schedules over time (top subplot)
      2. Waypoint occupancy timelines and conflicts (bottom subplot)
    """

    # ------------------------------------------------------------
    # Helper: Assign distinct colors to each vertex
    # ------------------------------------------------------------
    unique_vertices = sorted(set(v for route in routes for v in route))
    num_vertices = len(unique_vertices)
    cmap_vertices = get_cmap('Dark2')
    vertex_colors = {v: cmap_vertices(i / num_vertices) for i, v in enumerate(unique_vertices)}

    plt.figure(figsize=figuresize)

    # ------------------------------------------------------------
    # Subplot 1: Vehicle timelines
    # ------------------------------------------------------------
    plt.subplot(1, 2, 1)
    num_vehicles = len(routes)
    plt.ylim(-0.5, num_vehicles - 0.1)

    for vehicle_id in range(num_vehicles):
        route = routes[vehicle_id]
        schedule = schedules[vehicle_id]
        t_process = process_T[vehicle_id]
        y = vehicle_id

        # --- Start Marker (gray triangle pointing right) ---
        start_time = schedule[0]
        departure_time_first = schedule[1]
        plt.plot(start_time, y, '>', color='grey', markersize=start_marker_size,
                 markeredgewidth=1.5, alpha=0.3, zorder=5)

        # --- Bars between consecutive events (light gray background) ---
        for i in range(len(schedule) - 1):
            plt.fill_betweenx(
                [y - bar_thickness, y + bar_thickness],
                schedule[i], schedule[i + 1],
                color='gray', alpha=0.1, zorder=1
            )

        # --- Route Vertices and Processing Durations ---
        for i, vertex in enumerate(route):
            color = vertex_colors[vertex]

            if i == 0:
                # First vertex (special case: start)
                plt.plot(departure_time_first, y, 'o',
                         color=color, markersize=marker_size,
                         markeredgewidth=1.5, markerfacecolor=color, zorder=10)
                plt.plot(departure_time_first + t_process[vertex], y, 'o',
                         color=color, markersize=marker_size, markerfacecolor=color, zorder=10)
                plt.text(departure_time_first, y + 0.1, rf'${{{vertex}}}$',
                         color='black', fontsize=index_size, ha='center', va='bottom')
            else:
                # Other vertices
                time = schedule[i + 1]  # offset because schedule[0] is start time
                plt.plot(time, y, 'o', color=color, markersize=marker_size,
                         markeredgewidth=1.5, markerfacecolor=color, zorder=10)
                plt.fill_betweenx([y - bar_thickness, y + bar_thickness],
                                  time, time + t_process[vertex],
                                  color='violet', alpha=0.1, zorder=1)
                plt.plot(time + t_process[vertex], y, 'o', color=color,
                         markersize=marker_size, markerfacecolor=color, zorder=10)
                plt.text(time, y + 0.1, rf'${{{vertex}}}$',
                         color='black', fontsize=index_size, ha='center', va='bottom')

    # --- Vertex Color Legend ---
    for vertex, color in vertex_colors.items():
        plt.plot([], [], 'o', color=color, label=f"Vertex {vertex}")

    plt.ylabel(r"$a_j$", fontsize=y_label_size)
    plt.yticks(range(num_vehicles), [rf"${{{i}}}$" for i in range(num_vehicles)], fontsize=y_tick_size)
    plt.xticks(fontsize=x_tick_size)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    # ------------------------------------------------------------
    # Subplot 2: Waypoint timelines (conflict visualization)
    # ------------------------------------------------------------
    plt.subplot(1, 2, 2)

    cmap = cm.get_cmap('RdYlGn')  # green (safe) → yellow → red (conflict)
    na, nwp = association_matrix.shape
    tick_wp = []
    count = 0

    for waypoint_idx in range(nwp):
        # Skip waypoints not visited by any agent
        if association_matrix[:, waypoint_idx].sum() <= 0.0:
            continue

        # Agents visiting this waypoint
        agent_indices = np.where(association_matrix[:, waypoint_idx] == 1.0)[0]
        schedule_times = schedule_matrix[agent_indices, waypoint_idx]

        # Sort by arrival time
        sorted_indices = np.argsort(schedule_times)
        sorted_agents = agent_indices[sorted_indices]
        sorted_times = schedule_times[sorted_indices]
        tol = tolArray[waypoint_idx]

        # --- Draw bars between consecutive arrivals, color-coded by gap ---
        for i in range(len(sorted_times) - 1):
            t1, t2 = sorted_times[i], sorted_times[i + 1]
            gap = t2 - t1

            # Nonlinear normalized ratio: 0=red (conflict), 1=green (safe)
            ratio = np.clip((gap - tol + 0.1 * tol) / tol, 0, 1) ** 0.3
            color = cmap(ratio)

            plt.fill_betweenx(
                [count - bar_thickness, count + bar_thickness],
                t1, t2,
                color=color, alpha=0.8, zorder=1
            )

        # --- Mark each agent's arrival ---
        for time, agent in zip(sorted_times, sorted_agents):
            plt.plot(time, count, 'o', markersize=marker_size,
                     color='black', markerfacecolor='white', zorder=3)
            plt.text(time, count + 0.1, rf'${agent}$',
                     fontsize=index_size, ha='center', va='bottom', color='black')

        tick_wp.append(waypoint_idx)
        count += 1

    # --- Axes Formatting ---
    plt.ylim(-0.5, len(tick_wp) - 0.1)
    plt.xlabel("Time (s)", fontsize=x_label_size)
    plt.ylabel(r"$w_i$", fontsize=y_label_size)
    plt.yticks(range(len(tick_wp)), [rf'${{{i}}}$' for i in tick_wp], fontsize=y_tick_size)
    plt.xticks(fontsize=x_tick_size)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    # ------------------------------------------------------------
    # Colorbar (continuous legend inside the bottom subplot)
    # ------------------------------------------------------------
    ax = plt.gca()  # current (bottom) axes
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create a thin inset axis inside the current subplot (no change to layout)
    # [x0, y0, width, height] in axis coordinates (0–1)
    cax = ax.inset_axes([0.80, 0.10, 0.02, 0.7])  # adjust height/position as needed

    # Add a horizontal colorbar in that inset
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    # cbar.set_label('Gap between consecutive agent arrivals', fontsize=12, labelpad=4)

    # Tick positions matching nonlinear mapping
    ticks = [0.0, (0.1)**0.3, (1.0)**0.3, 1.0]
    tick_labels = [
        'Conflict (gap ≪ tol)',
        '≈ 0.9×tol',
        'Tolerance (gap = tol)',
        'Safe (gap > tol)'
    ]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=x_tick_size)

    # Optional: draw a subtle border around the colorbar region
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(0.3)



    plt.show()
