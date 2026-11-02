import numpy as np
from scipy.special import logsumexp


class flpoAgent_coarse:
    """FLPO agent on the coarse graph.

    Stage nodes (same order for all agents): the shared nodes (boundary nodes common to the
    subgraphs) followed by the agent's own destination as the last node, so n = n_shared + 1.
    The start is outside the stage graph:

        start -> stage 1 -> ... -> stage n -> exit        (K = n + 1 transitions)

    The destination node is absorbing: once reached, the agent stays there. The final transition
    to the exit is free and only allowed from the destination node.

    sched has length n + 1: [schedule at start, schedule at shared nodes, schedule at destination].
    Hence sched[1:] is the schedule of the stage nodes.

    mask_start (n,), mask_int (n, n): 1 where an edge exists; dist_* are the edge lengths.
    selfHop lets the agent stay at a non-destination node at zero cost.
    """

    def __init__(self, dist_start, mask_start, dist_int, mask_int,
                 sched, start_time, speed, speedLim, stagewise_cost_coeffs,
                 INF=1e8, offset_energy=True, selfHop=False):
        self.n = len(dist_int)
        self.dest = self.n - 1
        self.stageHorizon = self.n + 1
        assert len(sched) == self.n + 1
        self.dist_start, self.dist_int = dist_start, dist_int
        self.sched = sched
        self.start_time = start_time
        self.speed = speed
        self.speedLim = speedLim
        self.coeffs = stagewise_cost_coeffs
        self.INF = INF
        self.offset_energy = offset_energy
        self.mask_start = np.asarray(mask_start) > 0
        mask = (np.asarray(mask_int) > 0) | (bool(selfHop) * np.eye(self.n, dtype=bool))
        mask[self.dest, :] = False
        mask[self.dest, self.dest] = True
        self.mask_int = mask
        self.freeEnergy_s = None
        self.route, self.fin_sched = [], []

    def stagewise_cost(self, sched, speed):
        """Cost matrices in forward order: (1, n), (n, n) x (n-1), (n, 1)."""
        c0, c1, c2 = self.coeffs
        s_start, s_node = sched[0], sched[1:]

        t = self.dist_start / speed
        first = c0 * (s_start - self.start_time) ** 2 + c1 * (s_node - s_start - t) ** 2 + c2 * t
        first = np.where(self.mask_start, first, self.INF)[None, :]

        t = self.dist_int / speed
        mid = c1 * (s_node[None, :] - s_node[:, None] - t) ** 2 + c2 * t
        mid = np.where(self.mask_int, mid, self.INF)

        last = np.full((self.n, 1), self.INF)
        last[self.dest] = 0.0

        return [first] + [mid] * (self.n - 1) + [last]

    def stagewise_cost_grad(self, sched, speed):
        """Cost gradients in parameter order: sched entries, then speed."""
        c0, c1, c2 = self.coeffs
        s_start, s_node = sched[0], sched[1:]
        n_params = len(sched) + 1

        t = self.dist_start / speed
        residual = s_node - s_start - t
        first_grad = np.zeros((n_params, 1, self.n))
        first_grad[0, 0] = 2 * c0 * (s_start - self.start_time)
        first_grad[0, 0] -= 2 * c1 * residual
        first_grad[1:self.n + 1, 0, :] = np.diag(2 * c1 * residual)
        first_grad[-1, 0] = self.dist_start / speed**2 * (2 * c1 * residual - c2)
        first_grad = np.where(self.mask_start[None, None, :], first_grad, 0.0)

        t = self.dist_int / speed
        residual = s_node[None, :] - s_node[:, None] - t
        mid_grad = np.zeros((n_params, self.n, self.n))
        for i in range(self.n):
            mid_grad[i + 1, i, :] -= 2 * c1 * residual[i, :]
            mid_grad[i + 1, :, i] += 2 * c1 * residual[:, i]
        mid_grad[-1] = self.dist_int / speed**2 * (2 * c1 * residual - c2)
        mid_grad = np.where(self.mask_int[None, :, :], mid_grad, 0.0)

        last_grad = np.zeros((n_params, self.n, 1))
        return [first_grad] + [mid_grad] * (self.n - 1) + [last_grad]

    def n_paths(self):
        return [self.mask_start.sum(keepdims=True)] + [self.mask_int.sum(axis=1, keepdims=True)] * (self.n - 1)

    def backPropDP(self, Xi, beta):
        """Backward soft-min DP. Returns free energy and transition probabilities P (forward order)."""
        m = self.n_paths()
        P = [None] * self.stageHorizon
        V = Xi[-1]
        P[-1] = np.ones_like(V)
        for k in range(self.stageHorizon - 2, -1, -1):
            Lam = Xi[k] + V.T
            lse = logsumexp(-beta * Lam, axis=1, keepdims=True)
            P[k] = np.exp(-beta * Lam - lse)
            V = -lse / beta + self.offset_energy * np.log(np.maximum(m[k], 1)) / beta
        self.freeEnergy_s = float(V.sum())
        return self.freeEnergy_s, P

    def backPropDP_grad(self, stagewise_grad, P):
        """Propagate transition-cost gradients to free energy using forward reach probabilities."""
        gradient = np.zeros(stagewise_grad[0].shape[0])
        reach = np.ones(1)
        for grad, transition_prob in zip(stagewise_grad, P):
            occupancy = reach[:, None] * transition_prob
            gradient += np.sum(grad * occupancy[None, :, :], axis=(1, 2))
            reach = occupancy.sum(axis=0)
        return gradient

    def getFreeEnergyGradients(self, beta, sched=None, speed=None):
        """Return free energy, schedule gradient, and speed gradient."""
        Xi = self.stagewise_cost(sched, speed)
        free_energy, P = self.backPropDP(Xi, beta)
        grad = self.backPropDP_grad(self.stagewise_cost_grad(sched, speed), P)
        return free_energy, grad[:-1], grad[-1]

    def getFreeEnergy_s(self, beta, sched=None, speed=None):
        return self.backPropDP(self.stagewise_cost(sched, speed), beta)[0]

    def getPathAssociations(self, beta, sched=None, speed=None):
        return self.backPropDP(self.stagewise_cost(sched, speed), beta)[1]

    def calc_route_and_schedule(self, P):
        """Most likely route over stage-node indices (repeats collapsed, ends at the destination)."""
        m = int(np.argmax(P[0][0]))
        nodes = [m]
        for k in range(1, self.n):
            m = int(np.argmax(P[k][m]))
            nodes.append(m)
        self.route = [v for i, v in enumerate(nodes) if i == 0 or v != nodes[i - 1]]
        self.fin_sched = [self.start_time, self.sched[0]] + [self.sched[1 + v] for v in self.route]
