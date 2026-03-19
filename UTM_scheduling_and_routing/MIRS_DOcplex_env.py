from docplex.mp.model import Model

class MIRS_DOcplex:
    def __init__(
        self,
        num_agents,
        num_waypoints,
        start_goal,
        theta_bounds,
        inv_speed_bounds,
        INF,
        stage_sizes,
        mask,
        start_times,
        distMat,
        tolerance,
        stagewiseCostCoeffs
    ):
        # Store inputs
        self.N = num_agents
        self.M = num_waypoints
        self.start_goal = start_goal
        self.theta_lb, self.theta_ub = theta_bounds
        self.inv_lb, self.inv_ub = inv_speed_bounds
        self.INF = INF
        self.stage_sizes = stage_sizes
        self.K = len(stage_sizes) - 1
        self.c0, self.c1, self.c2 = stagewiseCostCoeffs
        self.mask = mask
        self.start_times = start_times
        self.distMat = distMat
        self.tolerance = tolerance

        # Model
        self.model = Model(name="single_agent_MIRS")

        # Variable containers
        self.theta = {}
        self.inv_speed = {}
        self.Z = {}
        self.eta = {}
        self.Xi = {}

        self._build_variables()
        self._build_constraints()

    def _build_variables(self):
        for n in range(self.N):

            for i in range(self.M):
                self.theta[n, i] = self.model.continuous_var(
                    lb=self.theta_lb,
                    ub=self.theta_ub,
                    name=f"T_{n}_{i}"
                )

            self.inv_speed[n] = self.model.continuous_var(
                lb=self.inv_lb,
                ub=self.inv_ub,
                name=f"inv_speed_{n}"
            )

            for k in range(self.K):
                for i in range(self.stage_sizes[k]):
                    for j in range(self.stage_sizes[k+1]):
                        self.Z[n,k,i,j] = self.model.continuous_var(name=f"Z_{n}_{k}_{i}_{j}")
                        self.eta[n,k,i,j] = self.model.binary_var(name=f"eta_{n}_{k}_{i}_{j}")
                        self.Xi[n,k,i,j] = self.model.continuous_var(name=f"Xi_{n}_{k}_{i}_{j}")

    def _build_constraints(self):
        for n in range(self.N):
            s, d = self.start_goal[n]

            net_mask = self.mask.copy()
            net_mask[d, :] = 0
            net_mask[d, d] = 1

            distMat_temp = self.distMat.copy()
            distMat_temp[net_mask == 0] = self.INF

            for k in range(self.K):
                for i in range(self.stage_sizes[k]):
                    for j in range(self.stage_sizes[k+1]):

                        # Product constraint (nonconvex)
                        self.model.add_constraint(
                            self.Z[n,k,i,j] == self.eta[n,k,i,j] * self.Xi[n,k,i,j]
                        )

                        # Cost expression
                        if k == 0:
                            if net_mask[s, j] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[s, j] * self.inv_speed[n]

                                expr = (
                                    self.c0 * (self.theta[n,s] - self.start_times[n])**2 +
                                    self.c1 * (self.theta[n,j] - self.theta[n,s] - travel_time)**2 +
                                    self.c2 * travel_time**2
                                )

                        elif k < self.K - 1:
                            if net_mask[i, j] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[i, j] * self.inv_speed[n]

                                expr = (
                                    self.c1 * (self.theta[n,j] - self.theta[n,i] - travel_time)**2 +
                                    self.c2 * travel_time**2
                                )

                        else:
                            if net_mask[i, d] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[i, d] * self.inv_speed[n]

                                expr = (
                                    self.c1 * (self.theta[n,d] - self.theta[n,i] - travel_time)**2 +
                                    self.c2 * travel_time**2
                                )

                        self.model.add_constraint(self.Xi[n,k,i,j] == expr)

                    if k > 0 and k < self.K:

                        if k < self.K-1:
                            self.model.add_constraint(
                                self.model.sum(
                                    self.eta[n,k,i,j]*net_mask[i,j]
                                    for j in range(self.stage_sizes[k+1])
                                ) <= 1
                            )

                        if k == 1:
                            self.model.add_constraint(
                                self.eta[n,k-1,0,i]*net_mask[s,i]
                                ==
                                self.model.sum(
                                    self.eta[n,k,i,i_]*net_mask[i,i_]
                                    for i_ in range(self.stage_sizes[k+1])
                                )
                            )

                        elif k == self.K-1:
                            self.model.add_constraint(
                                self.model.sum(
                                    self.eta[n,k-1,_i,i]*net_mask[_i,i]
                                    for _i in range(self.stage_sizes[k-1])
                                )
                                ==
                                self.eta[n,k,i,0]*net_mask[i,d]
                            )

                        else:
                            self.model.add_constraint(
                                self.model.sum(
                                    self.eta[n,k-1,_i,i]*net_mask[_i,i]
                                    for _i in range(self.stage_sizes[k-1])
                                )
                                ==
                                self.model.sum(
                                    self.eta[n,k,i,i_]*net_mask[i,i_]
                                    for i_ in range(self.stage_sizes[k+1])
                                )
                            )

                if k == 0:
                    self.model.add_constraint(
                        self.model.sum(
                            self.eta[n,k,0,j]*net_mask[s,j]
                            for j in range(self.stage_sizes[k+1])
                        ) == 1
                    )

                elif k == self.K-1:
                    self.model.add_constraint(
                        self.model.sum(
                            self.eta[n,k,i,0]*net_mask[i,d]
                            for i in range(self.stage_sizes[k])
                        ) == 1
                    )

            # Collision constraints
            if self.N > 1:
                for n1 in range(n+1, self.N):
                    for j in range(self.M):
                        self.model.add_constraint(
                            (self.theta[n,j] - self.theta[n1,j])**2
                            >= self.tolerance[j]**2
                        )

    def compute_total_cost(self):
        V = []

        for n in range(self.N):
            U = [None] * self.K

            for k in range(self.K):
                stage_idx = self.K - 1 - k

                if k == 0:
                    U[stage_idx] = {
                        i: self.Z[n, stage_idx, i, 0]
                        for i in range(self.stage_sizes[stage_idx])
                    }

                elif k < self.K - 1:
                    U[stage_idx] = {}
                    for i in range(self.stage_sizes[stage_idx]):
                        U[stage_idx][i] = self.model.sum(
                            self.Z[n, stage_idx, i, j] + U[stage_idx + 1][j]
                            for j in range(self.stage_sizes[stage_idx + 1])
                        )

                else:
                    U[stage_idx] = self.model.sum(
                        self.Z[n, stage_idx, 0, j] + U[stage_idx + 1][j]
                        for j in range(self.stage_sizes[stage_idx + 1])
                    )

            V.append(U[0])

        return self.model.sum(V)

    def optimize(self, time_limit=600, mip_gap=0.05):

        self.model.parameters.timelimit = time_limit
        self.model.parameters.mip.tolerances.mipgap = mip_gap

        obj = self.compute_total_cost()
        self.model.minimize(obj)

        self.model.solve(log_output=True)

    def get_model(self):
        return self.model