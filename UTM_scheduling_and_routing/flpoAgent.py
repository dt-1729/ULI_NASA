# import all the packages
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams.update(plt.rcParamsDefault)
import itertools
import math
import scipy.io as scio
from scipy.spatial.distance import cdist
import copy
import time
import pickle
from scipy.optimize import minimize
from matplotlib import patches
import warnings
from scipy.optimize import LinearConstraint
from scipy.special import *
import supporting_functions


class flpoAgent():

    def __init__(self, n_wp:int, sd:list, sched:np.ndarray, 
                speedLim:np.ndarray, process_T:np.ndarray, INF:float):

        assert(sd[0] != sd[1])
        self.n_wp = n_wp # number of waypoints
        self.s = sd[0] # starting waypoint index
        self.d = sd[1] # destination waypoint index
        self.sched = sched # schedule at all the waypoints
        self.stageHorizon = self.n_wp+1 # number of FLPO stages
        self.speedLim = speedLim # minimum and maximum speeds allowed
        self.mean_speed = np.mean(self.speedLim)
        self.INF = INF
        self.freeEnergy_s = -INF # initialize free energy of the vehicle (assuming beta = 0)
        self.route = []
        self.fin_sched = []
        self.t_process = process_T # processing time of the vehicle at the waypoints


    def speedConsPenalty(self, transitSchedMat, distMat, gamma, coeff):
        assert(transitSchedMat.shape==distMat.shape)
        vmin = self.speedLim[0]
        vmax = self.speedLim[1]
        # P_up = supporting_functions.myPenaltyFunc(transitSchedMat - distMat/self.speedLim[0], gamma, coeff)
        # P_low = supporting_functions.myPenaltyFunc(distMat/self.speedLim[1] - transitSchedMat, gamma, coeff)
        # assert(P_up.shape == P_low.shape)
        # return P_up + P_low
        X = vmin/(vmax-vmin) * (transitSchedMat * vmax - distMat)/distMat
        P = supporting_functions.myPenaltyFunc1(X, coeff)
        assert(P.shape == X.shape)
        return P


    def returnStageWiseCost(self, sched, distMat, gamma, coeff):
        K = self.stageHorizon
        Xi_flip = [0]*K
        dt_w2w = np.tile(sched, (self.n_wp, 1)) - np.tile(sched.reshape(-1,1), (1, self.n_wp))
        dt_w2d = (np.array([sched[self.d]]) - sched).reshape(-1,1)
        dt_n2w = sched - np.array([sched[self.s]])

        dt_w2w[distMat==self.INF] = self.INF
        dt_w2d[distMat[:,self.d]==self.INF] = self.INF
        dt_n2w[distMat[self.s,:]==self.INF] = self.INF

        for i in range(K):
            if i == 0: # penultimate stage to destination
                Xi_flip[i] = dt_w2d + self.speedConsPenalty(
                    dt_w2d-self.t_process.reshape(-1,1), distMat[:,self.d].reshape(-1,1), gamma, coeff)
                Xi_flip[i][self.d,:] = 0.0
            elif i>0 and i<self.stageHorizon-1: # internal stages
                Xi_flip[i] = dt_w2w + self.speedConsPenalty(
                    dt_w2w-self.t_process.reshape(-1,1), distMat, gamma, coeff)
                Xi_flip[i][self.d,self.d] = 0.0
            elif i == self.stageHorizon-1: # starting stage to 1st stage
                Xi_flip[i] = np.expand_dims(
                    sched[self.s] + dt_n2w + self.speedConsPenalty(
                        dt_n2w-self.t_process[self.s], distMat[self.s,:], gamma, coeff), axis=0)
        return Xi_flip[::-1]


    def returnStageWiseCost_v1(self, sched, distMat):
        K = self.stageHorizon
        Xi_flip = [0]*K
        dt_w2w = np.tile(sched, (self.n_wp, 1)) - np.tile(sched.reshape(-1,1), (1, self.n_wp))
        dt_w2d = (np.array([sched[self.d]]) - sched).reshape(-1,1)
        dt_n2w = sched - np.array([sched[self.s]])

        dt_w2w[distMat==self.INF] = self.INF
        dt_w2d[distMat[:,self.d]==self.INF] = self.INF
        dt_n2w[distMat[self.s,:]==self.INF] = self.INF

        for i in range(K):
            if i == 0: # penultimate stage to destination
                Xi_flip[i] = (dt_w2d - distMat[:,self.d].reshape(-1,1)/self.mean_speed)**2
                Xi_flip[i][self.d,:] = 0.0
            elif i>0 and i<K-1: # internal stages
                Xi_flip[i] = (dt_w2w - distMat/self.mean_speed)**2
                Xi_flip[i][self.d,self.d] = 0.0
            elif i == K-1: # starting stage to 1st stage
                Xi_flip[i] = np.expand_dims(
                    sched[self.s]**2 + (dt_n2w - distMat[self.s,:]/self.mean_speed)**2, axis=0)
        return Xi_flip[::-1]


    def returnStagewiseGrad_v1(self, sched, distMat):
        N = self.n_wp
        G_w2w = np.zeros((N,N,N))
        K = self.stageHorizon
        G_flip = [0]*K

        # compute gradient for w2w transitions
        for k in range(N):
            G_w2w[k,k,:] = sched[k] - sched + self.mean_speed/distMat[k,:]
            G_w2w[k,:,k] = sched[k] - sched - self.mean_speed/distMat[:,k]

        for i in range(K):
            if i == 0: # penultimate stage to destination
                G_flip[i] = G_w2w[:,:,self.d].reshape(-1,N,1)
            elif i == K-1: # start to first stage
                G_flip[i] = G_w2w[:,self.s,:].reshape(-1,1,N)
            else:
                G_flip[i] = G_w2w

        return G_flip[::-1]
    

    def backPropDP(self, Xi_s, beta, returnPb=True):
        t0 = time.time()
        K = self.stageHorizon
        Lambda_flip = [0]*K
        freeEnergy_flip = [0]*K
        Xi_flip = Xi_s[::-1]
        p_flip = [0]*K
        for i in range(K):
            if i == 0:
                Lambda_flip[i] = Xi_flip[i]
                freeEnergy_flip[i] = Xi_flip[i]
            else:
                Lambda_flip[i] = Xi_flip[i] + np.tile(np.transpose(freeEnergy_flip[i-1]), (Xi_flip[i].shape[0],1))
                minLambda = Lambda_flip[i].min(axis=1,keepdims=True)
                freeEnergy_flip[i] = -1/beta*np.log(
                    np.exp(-beta*(Lambda_flip[i] - minLambda)).sum(axis=1,keepdims=True)) + minLambda
            if returnPb:
                p_flip[i] = np.exp(-beta*(Lambda_flip[i]-freeEnergy_flip[i]))
        tf = time.time()
        finalCost = np.sum(freeEnergy_flip[K-1])
        return Lambda_flip[::-1], freeEnergy_flip[::-1], finalCost, tf-t0, p_flip[::-1]

    def getFreeEnergy_s(self, sched, distMat, beta, gamma, coeff):
        Xi = self.returnStageWiseCost(sched, distMat, gamma, coeff)
        _, _, freeEnergy_s, _, _ = self.backPropDP(Xi, beta, returnPb=False)
        self.freeEnergy_s = freeEnergy_s
        return freeEnergy_s

    def getPathAssociations(self, sched, dist_mat, beta, gamma, coeff):
        Xi = self.returnStageWiseCost(sched=sched, distMat=dist_mat, gamma=gamma, coeff=coeff)
        _, _, _, _, Pb = self.backPropDP(Xi_s=Xi, beta=beta, returnPb=True)
        return Pb

    def getFreeEnergy_s_v1(self, sched, distMat, beta):
        Xi = self.returnStageWiseCost_v1(sched, distMat)
        _, _, freeEnergy_s, _, _ = self.backPropDP(Xi, beta, returnPb=False)
        self.freeEnergy_s = freeEnergy_s
        return freeEnergy_s

    def getPathAssociations_v1(self, sched, dist_mat, beta):
        Xi = self.returnStageWiseCost_v1(sched=sched, distMat=dist_mat)
        _, _, _, _, Pb = self.backPropDP(Xi_s=Xi, beta=beta, returnPb=True)
        return Pb

    def backPropDP_grad(self, GD_s, P):
        K = self.stageHorizon
        GV_flip = [0]*K
        GD_flip = GD_s[::-1]
        P_flip = P[::-1]
        for i in range(K):
            if i == 0:
                GV_flip[i] = GD_flip[i]
            else:
                GV_flip[i] = np.sum(P_flip[i][:,:,None,None] * (GD_flip[i] + GV_flip[i-1].squeeze()),axis=1,keepdims=True)
        # G_freeEnergy = GV_flip[i].squeeze().sum(axis=0)
        return GV_flip[::-1] #, G_freeEnergy

    def calc_probability_of_reach(self, Pb):
        Pb_reach = [0]*(self.stageHorizon+1) # equal to the number of intermediate stages (=#waypoints currently)
        reach_mat = []
        for i in range(self.stageHorizon+1):
            if i==0:
                Pb_reach[i] = np.array([1.0]) # probability of reaching start position is 1
            else:
                Pb_reach[i] = np.dot(np.transpose(Pb[i-1]),Pb_reach[i-1])
                if i < self.stageHorizon:
                    reach_mat.append(Pb_reach[i])
        reach_mat = np.array(reach_mat)
        reach_array = np.max(reach_mat,axis=0)
        reach_array[self.s] = 1.0
        return reach_array


    def calc_route_and_schedule(self, sched, dist_mat, Pb):
        O = [self.s]
        T = [sched[self.s]]
        # Pb = self.getPathAssociations(sched=sched, dist_mat=dist_mat, beta=beta, gamma=gamma, coeff=coeff)
        for i,p in enumerate(Pb):
            if i==0:
                m = np.argmax(p[0,:])
            else:
                m = np.argmax(p[m,:])
            O.append(m)
            T.append(sched[m])
            if m == self.d:
                break
        self.route = O
        self.fin_sched = T


    def showGraph(self, wpLocations, distMat, mask, sched, figuresize, showEdgeTimeLim=True):

        vmin = self.speedLim[0]
        vmax = self.speedLim[1]
        start_node = self.s
        destination_node = self.d

        # Create a figure for the plot
        plt.figure(figsize=figuresize)

        # Plot the nodes
        for i, (x, y) in enumerate(wpLocations):
            if i == start_node:
                plt.scatter(x, y, color='green', marker='v', s=100, label="Start Node")  # Start node in green
            elif i == destination_node:
                plt.scatter(x, y, color='red', s=100, marker='s', label="Destination Node")  # Destination node in red
            else:
                plt.scatter(x, y, color='skyblue', s=100)  # Other nodes
            # plt.text(x, y, f'{i}', fontsize=10, ha='right', va='bottom')  # Label the nodes
            plt.text(x, y, f't[{i}]={sched[i]:.2f}', fontsize=10, ha='left', va='top')  # Label the nodes

        # Plot the edges and annotate times
        for i in range(len(wpLocations)):
            for j in range(i + 1, len(wpLocations)):  # Only iterate over the upper triangle of the adjacency matrix
                if mask[i, j] == 1:
                    # Draw an edge (a line) between node i and node j
                    plt.plot([wpLocations[i, 0], wpLocations[j, 0]], [wpLocations[i, 1], wpLocations[j, 1]], color='gray')

                    if showEdgeTimeLim:
                        # Calculate the distance between node i and node j
                        distance = distMat[i,j]
                        
                        # Calculate minimum and maximum time to travel
                        t_min = distance / vmax
                        t_max = distance / vmin
                        
                        # Annotate the edge with the min/max times
                        mid_x = (wpLocations[i, 0] + wpLocations[j, 0]) / 2
                        mid_y = (wpLocations[i, 1] + wpLocations[j, 1]) / 2
                        plt.text(mid_x, mid_y, rf'$[{t_min:.2f},{t_max:.2f}]$', fontsize=10, color='red', ha='center', va='center')

        # Set axis labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Graph with Travel Times and Vehicle Path')
        # Show legend for start and destination nodes
        plt.legend()
        # Show the plot
        plt.show()
