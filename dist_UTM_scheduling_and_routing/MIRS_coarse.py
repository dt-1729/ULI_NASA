import numpy as np
from scipy.special import logsumexp
from tabulate import tabulate

from dist_MIRS import heir_MIRS
from flpoAgent_coarse import flpoAgent_coarse
from problem_generator import init_agent_params, space_agent_start_times


class MIRS_coarse:
    """Multi-agent routing and scheduling on a hierarchical coarse graph."""

    def __init__(
        self,
        scenario,
        boundary_fraction=0.2,
        offset_energy=True,
        stagewiseCostCoeffs=(0.1, 10.0, 0.1),
        selfHop=False,
        cost_mode="sum",
        lm=1.0,
        ca_cbf=None,
        filter_wp_thresh=0.0,
        prune_mode=False,
        printFlag=True,
        T_upper_bound=10000,
    ):
        self.INF = 1e8
        self.graph = scenario
        self.wp_locations = np.asarray(scenario["wp_locations"])
        self.mask = np.asarray(scenario["mask"])
        self.dist_mat = np.asarray(scenario["dist_mat"])
        self.n_waypoints = len(self.wp_locations)
        self.sd_mat = np.asarray(scenario["sd_mat"], dtype=int)
        self.n_agents = len(self.sd_mat)
        self.tolArray = np.asarray(
            scenario.get("tolArray", np.ones(self.n_waypoints)), dtype=float
        )
        self.offset_energy = offset_energy
        self.selfHop = selfHop
        self.cost_mode = cost_mode
        self.lm = lm
        self.ca_cbf = ca_cbf or {"mode": "lin_static"}
        self.filter_wp_thresh = filter_wp_thresh
        self.prune_mode = prune_mode
        self.T_upper_bound = T_upper_bound
        self.stagewiseCostCoeffs = np.asarray(stagewiseCostCoeffs, dtype=float)
        if self.stagewiseCostCoeffs.shape != (3,):
            raise ValueError("stagewiseCostCoeffs must contain exactly three coefficients")
        if cost_mode not in {"sum", "slowest"}:
            raise ValueError("cost_mode must be 'sum' or 'slowest'")
        if self.ca_cbf.get("mode") not in {"ellipse", "rect", "lin_static"}:
            raise ValueError("ca_cbf mode must be 'ellipse', 'rect', or 'lin_static'")
        if len(self.tolArray) != self.n_waypoints:
            raise ValueError("tolArray must have one entry per waypoint")

        self.coarse_graph = heir_MIRS(scenario, boundary_fraction)
        self.active_waypoints = range(self.n_waypoints)
        self._init_agent_parameters(scenario)
        self.initAgents()
        if printFlag:
            print(
                f"n_waypoints: {self.n_waypoints}, n_agents: {self.n_agents}, "
                f"coarse_waypoints: {len(self.coarse_graph.coarse_nodes)}"
            )

    def _init_agent_parameters(self, scenario):
        seed = int(scenario.get("seed", 1))
        mode = self.ca_cbf.get("mode")
        (
            self.agent_weights,
            _,
            self.speed_lim_mat,
            self.speed_vec,
            self.sched_mat,
            _,
            self.T_upper_bound,
            self.process_T,
        ) = init_agent_params(
            self.n_agents,
            self.wp_locations,
            seed,
            tolArray=self.tolArray,
            mode=mode,
            T_upper_bound=self.T_upper_bound,
        )

        self.start_times = space_agent_start_times(self.sd_mat, self.tolArray)
        if mode == "lin_static":
            max_tol = float(np.max(self.tolArray))
            agent_offset = np.arange(self.n_agents, dtype=float) * (
                max_tol + float(np.max(self.start_times))
            )
            waypoint_drift = np.linspace(0.0, max_tol * 2.25, self.n_waypoints)
            self.sched_mat = (
                self.start_times[:, None]
                + agent_offset[:, None]
                + waypoint_drift[None, :]
            )
        self.sched_mat[np.arange(self.n_agents), self.sd_mat[:, 0]] = self.start_times

    def initAgents(self):
        self.agents = []
        self.agent_stage_waypoints = []
        self.agent_sched_indices = []
        for i in range(self.n_agents):
            graph = self.coarse_graph.agent_graph(i)
            start_wp = int(self.sd_mat[i, 0])
            print(f"agent {i}: shared_nodes: {len(self.coarse_graph.shared_nodes)}")
            stage_nodes = np.append(self.coarse_graph.shared_nodes, self.coarse_graph.coarse_sd_mat[i, 1])
            stage_waypoints = self.coarse_graph.coarse_nodes[stage_nodes]
            # print(f"agent {i}: stage_nodes={len(stage_nodes)}, stage_waypoints={len(stage_waypoints)}")
            sched_indices = np.append(start_wp, stage_waypoints)
            self.agent_stage_waypoints.append(stage_waypoints)
            self.agent_sched_indices.append(sched_indices)
            self.agents.append(
                flpoAgent_coarse(
                    dist_start=graph["dist_start"],
                    mask_start=graph["mask_start"],
                    dist_int=graph["dist_int"],
                    mask_int=graph["mask_int"],
                    sched=self.sched_mat[i, sched_indices],
                    start_time=self.start_times[i],
                    speed=self.speed_vec[i],
                    speedLim=self.speed_lim_mat[i],
                    stagewise_cost_coeffs=self.stagewiseCostCoeffs,
                    INF=self.INF,
                    offset_energy=self.offset_energy,
                    selfHop=self.selfHop,
                )
            )

    def set_offset_energy(self, flag):
        self.offset_energy = flag
        for agent in self.agents:
            agent.offset_energy = flag

    def _local_sched(self, agent_index, sched_mat):
        return sched_mat[agent_index, self.agent_sched_indices[agent_index]]

    def calc_agent_reach_mat_v1(self, sched_mat, speed_vec, beta):
        reach_mat = np.zeros((self.n_agents, self.n_waypoints))
        for i, agent in enumerate(self.agents):
            sched = self._local_sched(i, sched_mat)
            probabilities = agent.getPathAssociations(beta, sched, speed_vec[i])
            reach = np.asarray(probabilities[0][0]).copy()
            unvisited = reach.copy()
            for node in range(agent.n):
                unvisited = reach.copy()
                unvisited[node] = 0.0
                for transition in probabilities[1:agent.n]:
                    unvisited = unvisited @ transition
                    unvisited[node] = 0.0
                waypoint = self.agent_stage_waypoints[i][node]
                reach_mat[i, waypoint] = max(
                    reach_mat[i, waypoint], 1.0 - float(unvisited.sum())
                )
            reach_mat[i, self.sd_mat[i, 0]] = 1.0

        association_mat = (reach_mat > 1e-10).astype(float)
        return reach_mat, association_mat

    def transportCost_v1(self, sched_mat, speed_vec, beta, returnGrad):
        costs = np.zeros(self.n_agents)
        gradients = np.zeros((self.n_agents, self.n_waypoints + 1))
        for i, agent in enumerate(self.agents):
            sched = self._local_sched(i, sched_mat)
            if returnGrad:
                costs[i], schedule_grad, speed_grad = agent.getFreeEnergyGradients(
                    beta, sched, speed_vec[i]
                )
                np.add.at(gradients[i, :-1], self.agent_sched_indices[i], schedule_grad)
                gradients[i, -1] = speed_grad
            else:
                costs[i] = agent.getFreeEnergy_s(beta, sched, speed_vec[i])

        self.C_agents = costs
        if self.cost_mode == "sum":
            transport_cost = float(np.sum(self.agent_weights * costs))
        else:
            transport_cost = float(logsumexp(self.lm * costs) / self.lm)
        return transport_cost, gradients

    def CBF_waypoints_v1(self, sched_mat, waypoints, weight_mat, returnGrad=True):
        filter_wp = np.ones(weight_mat.shape)
        if self.prune_mode:
            filter_wp[weight_mat <= self.filter_wp_thresh] = 0.0

        constraints = []
        gradients = []
        mode = self.ca_cbf["mode"]
        for wp in waypoints:
            active = np.flatnonzero(filter_wp[:, wp] == 1.0)
            if len(active) < 2:
                continue
            times = sched_mat[active, wp]
            pair_indices = np.triu_indices(len(active), k=1)
            first, second = pair_indices
            delta_sum = times[first] + times[second]
            delta = times[second] - times[first]

            if mode == "lin_static":
                h = delta**2 - self.tolArray[wp] ** 2
                grad_first = -2.0 * delta
                grad_second = 2.0 * delta
            elif mode == "ellipse":
                major = self.ca_cbf["major_axis"][wp]
                minor = self.ca_cbf["minor_axis"][wp]
                degree = self.ca_cbf["degree"]
                h = (
                    (minor * np.abs(delta_sum)) ** degree
                    + (major * np.abs(delta)) ** degree
                    - (major * minor) ** degree
                )
                grad_sum = (
                    degree
                    * minor**degree
                    * np.abs(delta_sum) ** (degree - 1)
                    * np.sign(delta_sum)
                )
                grad_delta = (
                    degree
                    * major**degree
                    * np.abs(delta) ** (degree - 1)
                    * np.sign(delta)
                )
                grad_first = grad_sum - grad_delta
                grad_second = grad_sum + grad_delta
            else:
                width = self.ca_cbf["width"][wp] * (
                    1 + self.ca_cbf["width_correction_fac"]
                )
                height = self.ca_cbf["height"][wp] * (
                    1 + self.ca_cbf["height_correction_fac"]
                )
                gamma = self.ca_cbf["gamma"]
                a = delta_sum**2 - width**2 / 2
                b = delta**2 - height**2 / 2
                max_ab = np.maximum(a, b)
                exp_a = np.exp(gamma * (a - max_ab))
                exp_b = np.exp(gamma * (b - max_ab))
                prob_a = exp_a / (exp_a + exp_b)
                prob_b = exp_b / (exp_a + exp_b)
                h = max_ab + np.log(exp_a + exp_b) / gamma
                grad_first = 2 * prob_a * delta_sum - 2 * prob_b * delta
                grad_second = 2 * prob_a * delta_sum + 2 * prob_b * delta

            constraints.extend(h.tolist())
            if returnGrad:
                grad_h = np.zeros((len(h), self.n_agents))
                grad_h[np.arange(len(h)), active[first]] = grad_first
                grad_h[np.arange(len(h)), active[second]] = grad_second
                gradients.append(grad_h)

        h_values = np.asarray(constraints)
        grad_values = (
            np.concatenate(gradients, axis=0)
            if returnGrad and gradients
            else np.empty((0, self.n_agents))
        )
        return h_values, grad_values, filter_wp

    def CBF_agents(self, speed_vec):
        h = (speed_vec - self.speed_lim_mat[:, 0]) * (
            self.speed_lim_mat[:, 1] - speed_vec
        )
        grad_h = np.diag(self.speed_lim_mat.sum(axis=1) - 2 * speed_vec)
        return h, grad_h

    def count_conflicts(self, sched_mat, reach_mat):
        conflicts = 0
        margins = []
        for wp in range(self.n_waypoints):
            visited = np.flatnonzero(reach_mat[:, wp] == 1.0)
            for i, agent_a in enumerate(visited):
                for agent_b in visited[i + 1 :]:
                    margin = (
                        abs(sched_mat[agent_a, wp] - sched_mat[agent_b, wp])
                        - 0.9 * self.tolArray[wp]
                    )
                    margins.append(margin / (0.9 * self.tolArray[wp]) * 100)
                    conflicts += margin < 0
        no_conflicts = conflicts == 0
        return no_conflicts, no_conflicts, min(margins) if margins else 100

    def solution_table(self, Pb_a, sched_mat, speed_vec, beta, printRoutes=False):
        routes, schedules = [], []
        table = []
        self.transportCost_v1(sched_mat, speed_vec, beta, returnGrad=False)
        for i, agent in enumerate(self.agents):
            local_sched = self._local_sched(i, sched_mat)
            agent.sched = local_sched
            agent.calc_route_and_schedule(Pb_a[i])
            agent.route = self.agent_stage_waypoints[i][np.asarray(agent.route, dtype=int)].tolist()
            routes.append(agent.route)
            schedules.append(agent.fin_sched)
            if printRoutes:
                table.append(
                    [
                        f"v{i}",
                        agent.route,
                        np.round(agent.fin_sched, 2),
                        f"{speed_vec[i]:.2f}",
                        np.round(self.speed_lim_mat[i], 2),
                        f"{self.C_agents[i]:.2f}",
                    ]
                )
        if printRoutes:
            print(
                tabulate(
                    table,
                    headers=["A", "R", "T", "V", "V Lim", "Cost"],
                    tablefmt="pretty",
                )
            )
        return routes, schedules
