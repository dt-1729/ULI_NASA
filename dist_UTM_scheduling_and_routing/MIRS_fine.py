import numpy as np
from scipy.spatial.distance import cdist
from tabulate import tabulate

from flpoAgent_fine import flpoAgent_fine


class MIRS_fine:
    """Multi-agent routing and scheduling on a single subgraph (fine level).

    Same framework as MIRS, but the problem is given instead of generated. ``subgraph`` is a dict:

        wp_locations   (n, 2)  subgraph waypoint coordinates
        mask           (n, n)  subgraph adjacency (edges inside the subgraph only)
        dist_mat       (n, n)  subgraph edge lengths
        tolArray       (n,)    separation tolerance at each waypoint
        sd_mat         (m, 2)  per-agent entry and exit waypoint (local subgraph indices)
        start_times    (m,)    per-agent entry time
        T_upper_bound  (m,) or scalar   per-agent upper bound on that agent's schedule parameters
    optional:
        speed_lim_mat  (m, 2)  per-agent [min, max] speed (random if absent, as in MIRS)
        agent_weights  (m,)    cost weights (uniform if absent)
        seed           int     seed for the random defaults
        node_ids       (n,)    global waypoint ids of the subgraph nodes (kept for mapping back)
        agent_ids      (m,)    global agent ids (kept for mapping back)

    ``T_upper_bound`` is stored as a column (m, 1) so ``T_upper_bound - sched_mat`` is the
    per-agent slack matrix.
    """

    def __init__(
        self,
        subgraph,
        offset_energy=True,
        stagewiseCostCoeffs=(0.1, 10.0, 0.1),
        selfHop=False,
        cost_mode="sum",
        lm=1.0,
        ca_cbf=None,
        filter_wp_thresh=0.0,
        prune_mode=False,
        printFlag=False,
    ):
        self.INF = 1e8
        self.subgraph = subgraph
        self.wp_locations = np.asarray(subgraph["wp_locations"], dtype=float)
        self.mask = np.asarray(subgraph["mask"])
        self.dist_mat = np.asarray(subgraph["dist_mat"], dtype=float)
        self.n_waypoints = len(self.wp_locations)
        self.sd_mat = np.asarray(subgraph["sd_mat"], dtype=int).reshape(-1, 2)
        self.n_agents = len(self.sd_mat)
        self.tolArray = np.asarray(subgraph["tolArray"], dtype=float)
        self.node_ids = np.asarray(subgraph.get("node_ids", np.arange(self.n_waypoints)))
        self.agent_ids = np.asarray(subgraph.get("agent_ids", np.arange(self.n_agents)))

        self.offset_energy = offset_energy
        self.selfHop = selfHop
        self.cost_mode = cost_mode
        self.lm = lm
        self.ca_cbf = ca_cbf or {"mode": "lin_static"}
        self.filter_wp_thresh = filter_wp_thresh
        self.prune_mode = prune_mode
        self.stagewiseCostCoeffs = np.asarray(stagewiseCostCoeffs, dtype=float)

        self._validate(subgraph)
        self.initProblem(subgraph)
        self.initAgents()
        self.printInitializationData(printFlag)
        self.active_waypoints = range(self.n_waypoints)

    def _validate(self, subgraph):
        n, m = self.n_waypoints, self.n_agents
        if m == 0:
            raise ValueError("subgraph has no agents")
        if self.mask.shape != (n, n) or self.dist_mat.shape != (n, n):
            raise ValueError("mask and dist_mat must be (n_waypoints, n_waypoints)")
        if len(self.tolArray) != n:
            raise ValueError("tolArray must have one entry per subgraph waypoint")
        if self.sd_mat.min() < 0 or self.sd_mat.max() >= n:
            raise ValueError("sd_mat must hold local subgraph waypoint indices")
        if np.any(self.sd_mat[:, 0] == self.sd_mat[:, 1]):
            raise ValueError("each agent needs distinct entry and exit waypoints")
        if self.stagewiseCostCoeffs.shape != (3,):
            raise ValueError("stagewiseCostCoeffs must contain exactly three coefficients")
        if self.cost_mode not in {"sum", "slowest"}:
            raise ValueError("cost_mode must be 'sum' or 'slowest'")
        if self.ca_cbf.get("mode") not in {"ellipse", "rect", "lin_static"}:
            raise ValueError("ca_cbf mode must be 'ellipse', 'rect', or 'lin_static'")

        self.start_times = np.asarray(subgraph["start_times"], dtype=float).reshape(-1)
        if len(self.start_times) != m:
            raise ValueError("start_times must have one entry per agent")
        upper = np.broadcast_to(
            np.asarray(subgraph["T_upper_bound"], dtype=float).reshape(-1), (m,)
        ).copy()
        if np.any(upper <= self.start_times):
            raise ValueError("T_upper_bound must exceed the entry time of every agent")
        self.agent_T_upper_bound = upper
        self.T_upper_bound = upper[:, None]

    def initProblem(self, subgraph):
        m, n = self.n_agents, self.n_waypoints
        rng = np.random.default_rng(subgraph.get("seed", 1))

        weights = subgraph.get("agent_weights")
        if weights is None:
            weights = np.ones(m)
        weights = np.asarray(weights, dtype=float)
        self.agent_weights = weights / np.sum(weights)

        speed_lim = subgraph.get("speed_lim_mat")
        if speed_lim is None:
            speed_lim = np.column_stack((rng.uniform(0.01, 1, m), rng.uniform(50, 80, m)))
        self.speed_lim_mat = np.asarray(speed_lim, dtype=float).reshape(m, 2)
        self.speed_vec = self.speed_lim_mat.mean(axis=1)

        # schedule parameters start inside each agent's own [entry time, upper bound] window
        if self.ca_cbf.get("mode") == "lin_static":
            max_tol = float(np.max(self.tolArray))
            agent_offset = np.arange(m, dtype=float) * (max_tol + float(np.max(self.start_times)))
            waypoint_drift = np.linspace(0.0, max_tol * 2.25, n)
            self.sched_mat = (
                self.start_times[:, None] + agent_offset[:, None] + waypoint_drift[None, :]
            )
        else:
            self.sched_mat = rng.uniform(self.start_times[:, None], self.T_upper_bound, (m, n))
        self.sched_mat[np.arange(m), self.sd_mat[:, 0]] = self.start_times

        self.wp_weights = np.ones(n) / n
        self.process_T = np.zeros((m, n))
        self.max_time = np.max(cdist(self.wp_locations, self.wp_locations)) / np.min(
            self.speed_lim_mat[:, 0]
        )

    def initAgents(self):
        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(
                flpoAgent_fine(
                    n_wp=self.n_waypoints,
                    sd=self.sd_mat[i, :],
                    sched=self.sched_mat[i, :],
                    start_time=self.start_times[i],
                    T_upper_bound=self.agent_T_upper_bound[i],
                    speed=self.speed_vec[i],
                    speedLim=self.speed_lim_mat[i, :],
                    process_T=self.process_T[i, :],
                    INF=self.INF,
                    offset_energy=self.offset_energy,
                    selfHop=self.selfHop,
                    net_mask=self.mask,
                    stagewise_cost_coeffs=self.stagewiseCostCoeffs,
                )
            )

    def printInitializationData(self, printFlag):
        if printFlag:
            print(f"n_waypoints: {self.n_waypoints} \nn_agents: {self.n_agents} \nCAT:\n{self.tolArray}")
            print("---------")
            print(f"wp_locations:\n{self.wp_locations} \nmask:\n{self.mask} \ndist_mat:\n{self.dist_mat}")
            print("---------")
            print(
                f"agent_weights:\n{self.agent_weights} \nsd_mat:\n{self.sd_mat} "
                f"\nstart_times:\n{self.start_times} \nT_upper_bound:\n{self.agent_T_upper_bound} "
                f"\nspeed_lim_mat:\n{self.speed_lim_mat} \nsched_mat:\n{self.sched_mat} "
                f"\nspeed_vec:\n{self.speed_vec}"
            )

    def set_offset_energy(self, flag):
        self.offset_energy = flag
        for a in self.agents:
            a.offset_energy = flag

    def calc_agent_reach_mat_v1(self, sched_mat, speed_vec, beta):
        reach_mat = []
        for i, ag in enumerate(self.agents):
            Pb = ag.getPathAssociations_v1(sched_mat[i, :], speed_vec[i], self.dist_mat, beta)
            ag_reach = ag.calc_agent_wp_association(Pb)[0][0]
            reach_mat.append(ag_reach)

        reach_mat = np.array(reach_mat)
        association_mat = np.ones(shape=reach_mat.shape)
        association_mat[reach_mat <= 1e-10] = 0.0

        return reach_mat, association_mat

    # total vehicle cost of transportation based on mean speed
    def transportCost_v1(self, sched_mat, speed_vec, beta, returnGrad):
        C_arr = np.zeros(self.n_agents)
        Grad_F = np.zeros(shape=(self.n_agents, self.n_waypoints + 1))

        for i, a in enumerate(self.agents):
            C_arr[i] = a.getFreeEnergy_s_v1(sched_mat[i, :], speed_vec[i], self.dist_mat, beta)
            if returnGrad:
                GD_a = a.returnStagewiseGrad_v1(sched_mat[i, :], speed_vec[i], self.dist_mat)
                P_a = a.getPathAssociations_v1(sched_mat[i, :], speed_vec[i], self.dist_mat, beta)
                G_Fa, _ = a.backPropDP_grad(GD_a, P_a)
                Grad_F[i, :] = G_Fa

        self.C_agents = C_arr

        if self.cost_mode == "sum":
            transport_cost = np.sum(self.agent_weights * C_arr)
        elif self.cost_mode == "slowest":
            C_max = np.max(C_arr)
            transport_cost = 1 / self.lm * np.log(np.sum(np.exp(self.lm * (C_arr - C_max)))) + C_max

        return transport_cost, Grad_F

    # gradient of each agent's free energy, and of the total transport cost, w.r.t. a single edge length l_ij
    def transportCostGrad_l(self, sched_mat, speed_vec, beta, i, j):
        C_arr = np.zeros(self.n_agents)
        Grad_l = np.zeros(self.n_agents)

        for k, a in enumerate(self.agents):
            C_arr[k] = a.getFreeEnergy_s_v1(sched_mat[k, :], speed_vec[k], self.dist_mat, beta)
            if a.net_mask[i, j] == 0:
                continue  # edge (i,j) does not exist in this agent's graph -> zero gradient
            GD_a = a.returnStagewiseGrad_l(sched_mat[k, :], speed_vec[k], self.dist_mat, i, j)
            P_a = a.getPathAssociations_v1(sched_mat[k, :], speed_vec[k], self.dist_mat, beta)
            G_Fa, _ = a.backPropDP_grad(GD_a, P_a)
            Grad_l[k] = G_Fa

        if self.cost_mode == "sum":
            total_grad = np.sum(self.agent_weights * Grad_l)
        elif self.cost_mode == "slowest":
            C_max = np.max(C_arr)
            w = np.exp(self.lm * (C_arr - C_max))
            w = w / np.sum(w)
            total_grad = np.sum(w * Grad_l)

        return Grad_l, total_grad

    # gradient of the total transport cost w.r.t. every physical edge length l_ij (i<j)
    def edgeGradientMatrix_l(self, sched_mat, speed_vec, beta):
        Nw = self.n_waypoints
        grad_mat = np.zeros((Nw, Nw))
        for i in range(Nw):
            for j in range(i + 1, Nw):
                if self.mask[i, j] == 0:
                    continue
                # dist_mat is symmetric, so l_ij's true gradient sums the partials from both directions
                _, g_fwd = self.transportCostGrad_l(sched_mat, speed_vec, beta, i, j)
                _, g_bwd = self.transportCostGrad_l(sched_mat, speed_vec, beta, j, i)
                grad_mat[i, j] = g_fwd + g_bwd
                grad_mat[j, i] = grad_mat[i, j]
        return grad_mat

    def CBF_waypoints_v1(self, sched_mat, waypoints, weight_mat, returnGrad=True):
        Na = self.n_agents
        filter_wp = np.ones(weight_mat.shape)
        if self.prune_mode:
            filter_wp[weight_mat <= self.filter_wp_thresh] = 0.0

        H = np.array([])
        Grad_H = np.empty((0, Na))

        if self.ca_cbf["mode"] == "ellipse":
            a, b, n = self.ca_cbf["major_axis"], self.ca_cbf["minor_axis"], self.ca_cbf["degree"]

            for i, wp in enumerate(waypoints):
                n_active_agents = int(sum(filter_wp[:, wp]))
                if n_active_agents > 1:
                    Ti = sched_mat[:, wp][filter_wp[:, wp] == 1.0]
                    KTi1 = (Ti + Ti.reshape(-1, 1))
                    KTi2 = (Ti - Ti.reshape(-1, 1))
                    Hi_mat = (np.abs(KTi1) * b[wp]) ** n + (np.abs(KTi2) * a[wp]) ** n - (a[wp] * b[wp]) ** n
                    Hi_triu = np.triu_indices_from(Hi_mat, k=1)
                    H = np.concatenate((H, Hi_mat[Hi_triu]))

                    if returnGrad:
                        start_row = 0
                        n_rows = n_active_agents - 1
                        Grad_Hi = np.zeros((n_active_agents * (n_active_agents - 1) // 2, Na))
                        temp_grad = np.zeros((n_active_agents * (n_active_agents - 1) // 2, n_active_agents))
                        for j in range(n_active_agents - 1):
                            if n % 2 == 0:
                                temp_grad[start_row: start_row + n_rows, j] = n * KTi1[j + 1:, j] ** (n - 1) * b[wp] ** n + n * KTi2[j + 1:, j] ** (n - 1) * a[wp] ** n
                                temp_grad[start_row: start_row + n_rows, j + 1:] = np.diag(n * KTi1[j, j + 1:] ** (n - 1) * b[wp] ** n) + np.diag(n * KTi2[j, j + 1:] ** (n - 1) * a[wp] ** n)
                            else:
                                temp_grad[start_row: start_row + n_rows, j] = n * KTi1[j + 1:, j] ** (n - 1) * np.sign(KTi1[j + 1:, j]) * b[wp] ** n + n * KTi2[j + 1:, j] ** (n - 1) * np.sign(KTi2[j + 1:, j]) * a[wp] ** n
                                temp_grad[start_row: start_row + n_rows, j + 1:] = np.diag(n * KTi1[j, j + 1:] ** (n - 1) * np.sign(KTi1[j, j + 1:]) * b[wp] ** n) + np.diag(n * KTi2[j, j + 1:] ** (n - 1) * np.sign(KTi2[j, j + 1:]) * a[wp] ** n)
                            start_row = start_row + n_rows
                            n_rows = n_rows - 1
                        Grad_Hi[:, filter_wp[:, wp] == 1.0] = temp_grad
                        Grad_H = np.concatenate((Grad_H, Grad_Hi), axis=0)

        elif self.ca_cbf["mode"] == "rect":
            w, h, gamma = self.ca_cbf["width"], self.ca_cbf["height"], self.ca_cbf["gamma"]
            ew, eh = self.ca_cbf["width_correction_fac"], self.ca_cbf["height_correction_fac"]

            for i, wp in enumerate(waypoints):
                n_active_agents = int(sum(filter_wp[:, wp]))
                if n_active_agents > 1:
                    Ti = sched_mat[:, wp][filter_wp[:, wp] == 1.0]
                    KTi1 = (Ti + Ti.reshape(-1, 1))
                    KTi2 = (Ti - Ti.reshape(-1, 1))
                    a = KTi1 ** 2 - (w[i] * (1 + ew)) ** 2 / 2
                    b = KTi2 ** 2 - (h[i] * (1 + eh)) ** 2 / 2
                    ab = np.concatenate((np.array([a]), np.array([b])))
                    ab_max = np.max(ab, axis=0)
                    ab_shift = ab - ab_max
                    exp_ab = np.exp(gamma * ab_shift)
                    sum_exp_ab = np.sum(exp_ab, axis=0)
                    p_ab = exp_ab / sum_exp_ab
                    Hi_mat = 1 / gamma * np.log(sum_exp_ab) + ab_max
                    Hi_triu = np.triu_indices_from(Hi_mat, k=1)
                    H = np.concatenate((H, Hi_mat[Hi_triu]))

                    if returnGrad:
                        start_row = 0
                        n_rows = n_active_agents - 1
                        Grad_Hi = np.zeros((n_active_agents * (n_active_agents - 1) // 2, Na))
                        temp_grad = np.zeros((n_active_agents * (n_active_agents - 1) // 2, n_active_agents))
                        for j in range(n_active_agents - 1):
                            temp_grad[start_row: start_row + n_rows, j] = 2 * (p_ab[0] * KTi1)[j + 1:, j] + 2 * (p_ab[1] * KTi2)[j + 1:, j]
                            temp_grad[start_row: start_row + n_rows, j + 1:] = np.diag(2 * (p_ab[0] * KTi1)[j, j + 1:]) + np.diag(2 * (p_ab[1] * KTi2)[j, j + 1:])
                            start_row = start_row + n_rows
                            n_rows = n_rows - 1
                        Grad_Hi[:, filter_wp[:, wp] == 1.0] = temp_grad
                        Grad_H = np.concatenate((Grad_H, Grad_Hi), axis=0)

        elif self.ca_cbf["mode"] == "lin_static":
            for i, wp in enumerate(waypoints):
                n_active_agents = int(sum(filter_wp[:, wp]))
                if n_active_agents > 1:
                    Ti = sched_mat[:, wp][filter_wp[:, wp] == 1.0]
                    KTi = (Ti - Ti.reshape(-1, 1))
                    Hi_mat = np.abs(KTi) ** 2 - self.tolArray[wp] ** 2
                    Hi_triu = np.triu_indices_from(Hi_mat, k=1)
                    H = np.concatenate((H, Hi_mat[Hi_triu]))

                    if returnGrad:
                        start_row = 0
                        n_rows = n_active_agents - 1
                        Grad_Hi = np.zeros((n_active_agents * (n_active_agents - 1) // 2, Na))
                        temp_grad = np.zeros((n_active_agents * (n_active_agents - 1) // 2, n_active_agents))
                        for j in range(n_active_agents - 1):
                            # KTi[r, c] = T_c - T_r, so d/dT_j (T_j - T_k)^2 = -2*KTi[j, k]
                            temp_grad[start_row: start_row + n_rows, j] = -2 * KTi[j, j + 1:]
                            temp_grad[start_row: start_row + n_rows, j + 1:] = np.diag(2 * KTi[j, j + 1:])
                            start_row = start_row + n_rows
                            n_rows = n_rows - 1
                        Grad_Hi[:, filter_wp[:, wp] == 1.0] = temp_grad
                        Grad_H = np.concatenate((Grad_H, Grad_Hi), axis=0)

        return H, Grad_H, filter_wp

    def CBF_agents(self, speed_vec):
        H = (speed_vec - self.speed_lim_mat[:, 0]) * (self.speed_lim_mat[:, 1] - speed_vec)
        gradH = np.diag(self.speed_lim_mat.sum(axis=1) - 2 * speed_vec)
        return H, gradH

    def CBF_schedule_upper_bound(self, sched_mat):
        """Per-agent schedule bound barrier h = T_upper_bound - sched (>= 0 when satisfied).

        Returns h with shape (n_agents, n_waypoints); dh/dsched is -1 elementwise.
        """
        return self.T_upper_bound - sched_mat

    def count_conflicts(self, sched_mat, reach_mat):
        n_conflicts = 0
        pct_conflict = []
        for j in range(self.n_waypoints):
            visited_agents = np.where(reach_mat[:, j] == 1.0)[0]
            for a1 in range(len(visited_agents)):
                for a2 in range(a1 + 1, len(visited_agents)):
                    check_conflict = np.abs(sched_mat[visited_agents[a1], j] - sched_mat[visited_agents[a2], j]) - 0.9 * self.tolArray[j]
                    pct_conflict.append(check_conflict / (0.9 * self.tolArray[j]) * 100)
                    if check_conflict < 0:
                        n_conflicts += 1

        no_conflicts = n_conflicts == 0
        max_pct_conflict = min(pct_conflict) if len(pct_conflict) > 0 else 100

        return no_conflicts, no_conflicts, max_pct_conflict

    def solution_table(self, Pb_a, sched_mat, speed_vec, beta, printRoutes=False):
        routes = []
        fin_schedules = []
        fin_speeds = []

        table_data = []
        headers = ["A", "R", "T", "V", "V Mean", "V Max", "V Lim", "Cost"]

        self.transportCost_v1(sched_mat, speed_vec, beta=beta, returnGrad=True)

        for i, a in enumerate(self.agents):
            a.calc_route_and_schedule(sched=sched_mat[i, :], dist_mat=self.dist_mat, Pb=Pb_a[i])
            routes.append(a.route)
            fin_schedules.append(a.fin_sched)
            fin_speeds.append(a.fin_avg_speed)

            if printRoutes:
                table_data.append([
                    f"v{i}",
                    str(a.route),
                    np.round(a.fin_sched, 2),
                    f"{speed_vec[i]:.2f}",
                    f"{a.fin_avg_speed:.2f}",
                    f"{np.max(a.route_speed):.2f}",
                    np.round(self.speed_lim_mat[i], 2),
                    f"{self.C_agents[i]:.2f}",
                ])

        if printRoutes:
            print(tabulate(table_data, headers=headers, tablefmt="pretty"))

        return routes, fin_schedules
