import time

import numpy as np


class flpoAgent_fine:
    """FLPO agent on a subgraph (fine level).

    Same stage-wise free-energy formulation as flpoAgent, on the subgraph's own waypoints.
    The agent enters the subgraph at waypoint ``sd[0]`` at ``start_time`` and must leave it at
    waypoint ``sd[1]``. In addition, every schedule parameter of this agent is bounded above by
    ``T_upper_bound`` (a per-agent bound inherited from the higher-level solution).
    """

    def __init__(
        self,
        n_wp: int,
        sd: list,
        sched: np.ndarray,
        start_time: float,
        T_upper_bound: float,
        speed: float,
        speedLim: np.ndarray,
        process_T: np.ndarray,
        INF: float,
        offset_energy: bool,
        selfHop: bool,
        net_mask: np.ndarray,
        stagewise_cost_coeffs: np.ndarray,
    ):
        assert sd[0] != sd[1]
        self.n_wp = n_wp  # number of waypoints in the subgraph
        self.s = int(sd[0])  # entry waypoint index
        self.d = int(sd[1])  # exit waypoint index
        self.sched = sched  # schedule at all the subgraph waypoints
        self.start_time = start_time  # entry time of the agent
        self.T_upper_bound = T_upper_bound  # upper bound on this agent's schedule parameters
        self.speed = speed
        self.stageHorizon = self.n_wp + 1  # number of FLPO stages
        self.speedLim = speedLim  # minimum and maximum speeds allowed
        self.mean_speed = np.mean(self.speedLim)
        self.INF = INF
        self.freeEnergy_s = -INF
        self.route = []
        self.fin_sched = []
        self.t_process = process_T
        self.offset_energy = offset_energy
        self.selfHop = selfHop
        self.net_mask = net_mask.copy()
        self.net_mask[self.d, :] = 0
        self.net_mask[self.d, self.d] = 1
        self.net_mask = self.net_mask + self.selfHop * np.eye(self.net_mask.shape[0])
        self.n_stagewise_paths()
        self.stagewise_cost_coeffs = stagewise_cost_coeffs

    def n_stagewise_paths(self):
        K = self.stageHorizon
        n_paths = [0] * K
        mask = self.net_mask
        for i in range(K):
            if i == 0:  # start to first intermediate stage
                n_paths[i] = np.sum(mask[self.s, :], keepdims=True)
            elif i == K - 1:  # penultimate to final stage
                n_paths[i] = mask[:, self.d].reshape(-1, 1)
            else:
                n_paths[i] = np.sum(mask, axis=1, keepdims=True)
        self.n_paths = n_paths

    def returnStageWiseCost_v1(self, sched, speed, distMat):
        K = self.stageHorizon
        Xi_flip = [0] * K
        c0, c1, c2 = self.stagewise_cost_coeffs
        dt_w2w = np.tile(sched, (self.n_wp, 1)) - np.tile(sched.reshape(-1, 1), (1, self.n_wp))
        dt_w2d = (np.array([sched[self.d]]) - sched).reshape(-1, 1)
        dt_n2w = sched - np.array([sched[self.s]])

        for i in range(K):
            if i == 0:  # penultimate stage to destination
                Xi_flip[i] = c1 * (dt_w2d - distMat[:, self.d].reshape(-1, 1) / speed) ** 2 + c2 * (distMat[:, self.d].reshape(-1, 1) / speed)
                Xi_flip[i][self.net_mask[:, self.d].reshape(-1, 1) == 0] = self.INF
                Xi_flip[i][self.d, :] = 0
            elif i > 0 and i < K - 1:  # internal stages
                Xi_flip[i] = c1 * (dt_w2w - distMat / speed) ** 2 + c2 * (distMat / speed)
                Xi_flip[i][self.net_mask == 0] = self.INF
                Xi_flip[i][self.d, self.d] = 0.0
            elif i == K - 1:  # starting stage to 1st stage
                Xi_flip[i] = np.expand_dims(
                    c0 * (sched[self.s] - self.start_time) ** 2 + c1 * (dt_n2w - distMat[self.s, :] / speed) ** 2 + c2 * (distMat[self.s, :] / speed), axis=0)
                Xi_flip[i][self.net_mask[self.s, None] == 0] = self.INF
        return Xi_flip[::-1]

    def returnStagewiseGrad_v1(self, sched, speed, distMat):
        Nw = self.n_wp
        N_params = len(sched) + 1
        K = self.stageHorizon
        c0, c1, c2 = self.stagewise_cost_coeffs
        G_w2w = np.zeros((N_params, Nw, Nw))
        G_flip = [0] * K
        mask = self.net_mask

        # gradient for w2w transitions w.r.t. schedules
        for k in range(N_params - 1):
            G_w2w[k, k, :] = 2 * c1 * (sched[k] - sched + distMat[k, :] / speed)
            G_w2w[k, :, k] = 2 * c1 * (sched[k] - sched - distMat[:, k] / speed)

        # gradient for w2w transitions w.r.t. speed
        G_w2w[N_params - 1] = distMat / speed**2 * (2 * c1 * (sched - sched.reshape(-1, 1)) - (2 * c1 + c2) * distMat / speed)

        for i in range(K):
            if i == 0:  # penultimate stage to destination
                G_flip[i] = G_w2w[:, :, self.d].reshape(-1, Nw, 1) * mask[:, self.d].reshape(-1, 1)
                G_flip[i][:, self.d, :] = 0
            elif i == K - 1:  # start to first stage
                G_start = (G_w2w[:, self.s, :]).reshape(-1, 1, Nw) * mask[self.s, :]
                G_off = np.zeros(G_start.shape)
                G_off[self.s, :, :] = 2 * c0 * (sched[self.s] - self.start_time) * np.ones(Nw) * mask[self.s, :]
                G_flip[i] = G_start + G_off
            else:
                G_flip[i] = G_w2w * mask
                G_flip[i][:, self.d, :] = 0

        return G_flip[::-1]

    def returnStagewiseGrad_l(self, sched, speed, distMat, i, j):
        # gradient of each stage-transition cost matrix w.r.t. the single edge length l_ij=distMat[i,j]
        Nw = self.n_wp
        K = self.stageHorizon
        c0, c1, c2 = self.stagewise_cost_coeffs
        edge_exists = self.net_mask[i, j] != 0

        if edge_exists:
            dt_ij = sched[j] - sched[i]
            deriv = -2 * c1 / speed * (dt_ij - distMat[i, j] / speed) + c2 / speed
        else:
            deriv = 0.0

        G_flip = [0] * K
        for stage in range(K):
            if stage == 0:  # penultimate stage to destination
                G_flip[stage] = np.zeros((1, Nw, 1))
                if edge_exists and j == self.d and i != self.d:
                    G_flip[stage][0, i, 0] = deriv
            elif stage == K - 1:  # start stage to first stage
                G_flip[stage] = np.zeros((1, 1, Nw))
                if edge_exists and i == self.s:
                    G_flip[stage][0, 0, j] = deriv
            else:  # internal stages
                G_flip[stage] = np.zeros((1, Nw, Nw))
                if edge_exists and i != self.d:
                    G_flip[stage][0, i, j] = deriv

        return G_flip[::-1]

    def backPropDP(self, Xi_s, beta, returnPb=True):
        t0 = time.time()
        K = self.stageHorizon
        Lambda_flip = [0] * K
        Value_flip = [0] * K
        Xi_flip = Xi_s[::-1]
        p_flip = [0] * K
        n_paths_flip = self.n_paths[::-1]
        for i in range(K):
            if i == 0:
                Lambda_flip[i] = Xi_flip[i]
                Value_flip[i] = Xi_flip[i]
                minLambda = Lambda_flip[i].min(axis=1, keepdims=True)
                exp_beta = np.exp(-beta * (Lambda_flip[i] - minLambda))
            else:
                Lambda_flip[i] = Xi_flip[i] + np.tile(np.transpose(Value_flip[i - 1]), (Xi_flip[i].shape[0], 1))
                m = n_paths_flip[i]
                minLambda = Lambda_flip[i].min(axis=1, keepdims=True)
                exp_beta = np.exp(-beta * (Lambda_flip[i] - minLambda))
                log_exp_beta = np.log(exp_beta.sum(axis=1, keepdims=True))
                Value_flip[i] = -1 / beta * log_exp_beta + minLambda + self.offset_energy * 1 / beta * np.log(m)
            if returnPb:
                p_flip[i] = exp_beta / exp_beta.sum(axis=1, keepdims=True)
        tf = time.time()
        finalCost = np.sum(Value_flip[K - 1])
        return Lambda_flip[::-1], Value_flip[::-1], finalCost, tf - t0, p_flip[::-1]

    def getFreeEnergy_s_v1(self, sched, speed, distMat, beta):
        Xi = self.returnStageWiseCost_v1(sched=sched, speed=speed, distMat=distMat)
        _, _, freeEnergy_s, _, _ = self.backPropDP(Xi_s=Xi, beta=beta, returnPb=False)
        self.freeEnergy_s = freeEnergy_s
        return freeEnergy_s

    def getPathAssociations_v1(self, sched, speed, dist_mat, beta):
        Xi = self.returnStageWiseCost_v1(sched=sched, speed=speed, distMat=dist_mat)
        _, _, _, _, Pb = self.backPropDP(Xi_s=Xi, beta=beta, returnPb=True)
        return Pb

    def backPropDP_grad(self, GD_s, P):
        K = self.stageHorizon
        GV_flip = [0] * K
        GD_flip = GD_s[::-1]
        P_flip = P[::-1]
        for i in range(K):
            if i == 0:  # from penultimate stage to the destination
                GV_flip[i] = GD_flip[i]
            else:  # intermediate stages
                GV_flip[i] = np.sum(P_flip[i] * GD_flip[i], axis=2, keepdims=True) + P_flip[i] @ GV_flip[i - 1]
        G_freeEnergy = GV_flip[-1].squeeze()
        return G_freeEnergy, GV_flip[::-1]

    def calc_agent_wp_association(self, Pb):
        Nw = self.n_wp
        K = self.stageHorizon
        Pb_nogo_flip = [0] * K
        Pb_reach_flip = [0] * K
        Pb_flip = Pb[::-1]
        for i in range(K):
            if i == 0:
                Pb_nogo_flip[i] = np.ones((Nw, Nw))
                Pb_nogo_flip[i][:, self.d] = np.zeros(Nw)
            else:
                Pb_nogo_flip[i] = np.dot(Pb_flip[i], Pb_nogo_flip[i - 1]) - np.dot(Pb_flip[i], (Pb_nogo_flip[i - 1] * np.eye(Nw)))
            Pb_reach_flip[i] = 1.0 - Pb_nogo_flip[i]

        Pb_reach_flip[-1][0, self.s] = 1.0

        return Pb_reach_flip[::-1]

    def schedule_upper_bound_slack(self, sched):
        """Per-waypoint slack T_upper_bound - sched; negative entries violate the bound."""
        return self.T_upper_bound - np.asarray(sched, dtype=float)

    def calc_route_and_schedule(self, sched, dist_mat, Pb):
        O = [self.s]
        T = [self.start_time, sched[self.s]]
        speeds = []
        m_prev = self.s
        dist = 0
        for i, p in enumerate(Pb):
            if i == 0:
                m = int(np.argmax(p[0, :]))
            else:
                m = int(np.argmax(p[m, :]))
            O.append(m)
            T.append(sched[m])
            dist = dist + dist_mat[m_prev, m]
            speed = dist_mat[m_prev, m] / (sched[m] - sched[m_prev])
            speeds.append(speed)
            if m == self.d:
                break
            else:
                m_prev = m
        self.route = O
        self.fin_sched = T
        self.fin_avg_speed = np.mean(speeds)
        self.route_speed = speeds
