import numpy as np
import matplotlib.pyplot as plt
import flpoAgent
import importlib
import random
from scipy.spatial.distance import cdist
from scipy.optimize import minimize, Bounds
import flpoAgent
import time
import utils
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
from problem_generator import init_waypoints, init_agent_params, init_agents, print_initialization_data


# create a multiagent routing and scheduling class
class MIRS():

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
        self.initProblem(wp_params, seed)
        self.stagewiseCostCoeffs = stagewiseCostCoeffs
        self.initAgents()
        self.printInitializationData(printFlag)
        self.active_waypoints = range(self.n_waypoints)
        self.ca_cbf = ca_cbf
        self.cost_mode = cost_mode
        self.lm = lm
        self.prune_mode = prune_mode
        self.filter_wp_thresh = filter_wp_thresh

    def initProblem(self, wp_params:dict, seed:int):
        self.wp_locations, self.mask, self.dist_mat, self.n_waypoints, self.wp_weights = init_waypoints(wp_params, seed, self.INF)
        self.agent_weights, self.sd_mat, self.speed_lim_mat, self.speed_vec, self.sched_mat, self.start_times, self.T_upper_bound, self.process_T = init_agent_params(self.n_agents, self.wp_locations, seed)

    def initAgents(self):
        self.agents = init_agents(
            self.n_agents, 
            self.n_waypoints, 
            self.sd_mat, 
            self.sched_mat, 
            self.start_times, 
            self.speed_vec, 
            self.speed_lim_mat, 
            self.process_T,
            self.INF, 
            self.offset_energy, 
            self.selfHop, 
            self.mask, 
            self.stagewiseCostCoeffs
        )

    def printInitializationData(self, printFlag):
        print_initialization_data(
            self.n_waypoints, 
            self.n_agents, 
            self.tolArray, 
            self.wp_locations, 
            self.mask, 
            self.dist_mat, 
            self.wp_weights,
            self.agent_weights, 
            self.sd_mat, 
            self.process_T, 
            self.speed_lim_mat, 
            self.sched_mat, 
            self.speed_vec, 
            printFlag
        )

    def set_offset_energy(self,flag):
        for a in self.agents:
            a.offset_energy = flag

    def calc_agent_reach_mat_v1(self, sched_mat, speed_vec, beta):
        reach_mat = []
        for i,ag in enumerate(self.agents):
            Pb = ag.getPathAssociations_v1(sched_mat[i,:], speed_vec[i], self.dist_mat, beta)
            ag_reach = ag.calc_agent_wp_association(Pb)[0][0]
            reach_mat.append(ag_reach)

        reach_mat = np.array(reach_mat)
        association_mat = np.ones(shape=reach_mat.shape)
        association_mat[reach_mat <= 1e-10] = 0.0
    
        return reach_mat, association_mat

    # function to compute total vehicle cost of transportation based on mean speed
    def transportCost_v1(
        self, 
        sched_mat:np.ndarray, 
        speed_vec:np.ndarray, 
        beta:float,
        returnGrad:bool
        ):

        C_arr = np.zeros(self.n_agents)
        Grad_F = np.zeros(shape=(self.n_agents, self.n_waypoints+1))

        for i,a in enumerate(self.agents):
            C_arr[i] = a.getFreeEnergy_s_v1(sched_mat[i,:], speed_vec[i], self.dist_mat, beta)
            if returnGrad:
                GD_a = a.returnStagewiseGrad_v1(sched_mat[i,:], speed_vec[i], self.dist_mat)
                P_a = a.getPathAssociations_v1(sched_mat[i,:], speed_vec[i], self.dist_mat, beta)
                G_Fa, _ = a.backPropDP_grad(GD_a, P_a)
                Grad_F[i,:] = G_Fa

        self.C_agents = C_arr

        if self.cost_mode == 'sum':
            transport_cost = np.sum(self.agent_weights * C_arr)
        elif self.cost_mode == 'slowest':
            C_max = np.max(C_arr)
            transport_cost = 1/self.lm * np.log(np.sum(np.exp(self.lm*(C_arr-C_max)))) + C_max
        
        return transport_cost, Grad_F
        

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

        elif self.ca_cbf['mode'] == 'lin_static':
            # Keep the original linear pairwise barrier definition and ordering.
            for i, wp in enumerate(waypoints):
                n_active_agents = int(sum(filter_wp[:,wp]))
                if n_active_agents > 1:
                    Ti = sched_mat[:,wp][filter_wp[:,wp]==1.0]
                    KTi = (Ti - Ti.reshape(-1,1))
                    Hi_mat = np.abs(KTi)**2 - self.tolArray[wp]**2
                    Hi_triu = np.triu_indices_from(Hi_mat, k=1)
                    H = np.concatenate((H,Hi_mat[Hi_triu]))

                    if returnGrad==True:
                        start_row = 0
                        n_rows = n_active_agents-1
                        Grad_Hi = np.zeros((n_active_agents * (n_active_agents-1)//2, Na))
                        temp_grad = np.zeros((n_active_agents * (n_active_agents-1)//2, n_active_agents))
                        for j in range(n_active_agents-1):
                            # For h_jk = (T_j - T_k)^2 - tol^2:
                            # dh/dT_j = 2*(T_j - T_k), dh/dT_k = -2*(T_j - T_k)
                            temp_grad[start_row : start_row + n_rows, j] = 2 * KTi[j, j+1:]
                            temp_grad[start_row : start_row + n_rows, j+1:] = np.diag(-2 * KTi[j, j+1:])
                            start_row = start_row + n_rows
                            n_rows = n_rows-1
                        Grad_Hi[:, filter_wp[:,wp]==1.0] = temp_grad
                        Grad_H = np.concatenate((Grad_H, Grad_Hi),axis=0)

        return H, Grad_H, filter_wp


    def CBF_agents(self, speed_vec):
        H = (speed_vec - self.speed_lim_mat[:,0])*(self.speed_lim_mat[:,1]-speed_vec)
        gradH = np.diag(self.speed_lim_mat.sum(axis=1) - 2 * speed_vec)
        return H, gradH

    def count_conflicts(self, sched_mat, reach_mat):
        n_conflicts = 0
        no_conflicts = True
        pct_conflict = []
        max_pct_conflict = 0
        for j in range(self.n_waypoints):
            visited_agents = np.where(reach_mat[:,j]==1.0)[0]
            for a1 in range(len(visited_agents)):
                for a2 in range(a1+1,len(visited_agents)):
                    check_conflict = np.abs(sched_mat[visited_agents[a1],j] - sched_mat[visited_agents[a2],j]) - 0.9*self.tolArray[j]
                    pct_conflict.append(check_conflict/(0.9*self.tolArray[j])*100)
                    if check_conflict < 0:
                        n_conflicts += 1
                        
        if n_conflicts > 0:
            no_conflicts = False
        if len(pct_conflict) > 0:
            max_pct_conflict = min(pct_conflict)
        else:
            max_pct_conflict = 100
        
        return no_conflicts, no_conflicts, max_pct_conflict

    def solution_table(
        self, 
        Pb_a:list, 
        sched_mat:np.ndarray, 
        speed_vec:np.ndarray, 
        beta:float, 
        printRoutes=False):
        
        routes = []
        fin_schedules = []
        fin_speeds = []

        table_data = [] 
        headers = ["A", "R", "T", "V", "V Mean", "V Max", "V Lim", "Cost"] 

        self.transportCost_v1( 
            sched_mat, 
            speed_vec, 
            beta=beta, 
            returnGrad=True) 

        for i, a in enumerate(self.agents):
            a.calc_route_and_schedule(sched=sched_mat[i,:], dist_mat=self.dist_mat, Pb=Pb_a[i])
            routes.append(a.route)
            fin_schedules.append(a.fin_sched)
            fin_speeds.append(a.fin_avg_speed)
            
            if printRoutes:
                row = [
                    f"v{i}",
                    str(a.route),
                    np.round(a.fin_sched, 2),
                    f"{speed_vec[i]:.2f}",
                    f"{a.fin_avg_speed:.2f}",
                    f"{np.max(a.route_speed):.2f}",
                    np.round(self.speed_lim_mat[i], 2),
                    f"{self.C_agents[i]:.2f}"
                ]
                table_data.append(row)

        if printRoutes:
            print(tabulate(table_data, headers=headers, tablefmt="pretty"))

        return routes, fin_schedules
