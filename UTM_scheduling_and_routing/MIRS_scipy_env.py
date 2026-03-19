from pyscipopt import Model, quicksum

class MIRS_SCIP:
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

        self.model = Model("MIRS_SCIP")

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
                self.theta[n,i] = self.model.addVar(
                    name=f"T_{n}_{i}",
                    lb=self.theta_lb,
                    ub=self.theta_ub,
                    vtype="C"
                )

            self.inv_speed[n] = self.model.addVar(
                name=f"inv_speed_{n}",
                lb=self.inv_lb,
                ub=self.inv_ub,
                vtype="C"
            )

            for k in range(self.K):
                for i in range(self.stage_sizes[k]):
                    for j in range(self.stage_sizes[k+1]):
                        self.Z[n,k,i,j] = self.model.addVar(name=f"Z_{n}_{k}_{i}_{j}", vtype="C")
                        self.eta[n,k,i,j] = self.model.addVar(name=f"eta_{n}_{k}_{i}_{j}", vtype="B")
                        self.Xi[n,k,i,j] = self.model.addVar(name=f"Xi_{n}_{k}_{i}_{j}", vtype="C")

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

                        # Bilinear constraint
                        self.model.addCons(
                            self.Z[n,k,i,j] == self.eta[n,k,i,j] * self.Xi[n,k,i,j]
                        )

                        # Cost expression
                        if k == 0:
                            if net_mask[s, j] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[s,j] * self.inv_speed[n]

                                expr = (
                                    self.c0 * (self.theta[n,s] - self.start_times[n])**2 +
                                    self.c1 * (self.theta[n,j] - self.theta[n,s] - travel_time)**2 +
                                    self.c2 * travel_time**2
                                )

                        elif k < self.K - 1:
                            if net_mask[i,j] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[i,j] * self.inv_speed[n]

                                expr = (
                                    self.c1 * (self.theta[n,j] - self.theta[n,i] - travel_time)**2 +
                                    self.c2 * travel_time**2
                                )

                        else:
                            if net_mask[i,d] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[i,d] * self.inv_speed[n]

                                expr = (
                                    self.c1 * (self.theta[n,d] - self.theta[n,i] - travel_time)**2 +
                                    self.c2 * travel_time**2
                                )

                        self.model.addCons(self.Xi[n,k,i,j] == expr)

                    if k > 0 and k < self.K:

                        if k < self.K-1:
                            self.model.addCons(
                                quicksum(
                                    self.eta[n,k,i,j]*net_mask[i,j]
                                    for j in range(self.stage_sizes[k+1])
                                ) <= 1
                            )

                        if k == 1:
                            self.model.addCons(
                                self.eta[n,k-1,0,i]*net_mask[s,i]
                                ==
                                quicksum(
                                    self.eta[n,k,i,i_]*net_mask[i,i_]
                                    for i_ in range(self.stage_sizes[k+1])
                                )
                            )

                        elif k == self.K-1:
                            self.model.addCons(
                                quicksum(
                                    self.eta[n,k-1,_i,i]*net_mask[_i,i]
                                    for _i in range(self.stage_sizes[k-1])
                                )
                                ==
                                self.eta[n,k,i,0]*net_mask[i,d]
                            )

                        else:
                            self.model.addCons(
                                quicksum(
                                    self.eta[n,k-1,_i,i]*net_mask[_i,i]
                                    for _i in range(self.stage_sizes[k-1])
                                )
                                ==
                                quicksum(
                                    self.eta[n,k,i,i_]*net_mask[i,i_]
                                    for i_ in range(self.stage_sizes[k+1])
                                )
                            )

                if k == 0:
                    self.model.addCons(
                        quicksum(
                            self.eta[n,k,0,j]*net_mask[s,j]
                            for j in range(self.stage_sizes[k+1])
                        ) == 1
                    )

                elif k == self.K-1:
                    self.model.addCons(
                        quicksum(
                            self.eta[n,k,i,0]*net_mask[i,d]
                            for i in range(self.stage_sizes[k])
                        ) == 1
                    )

            # Collision constraints
            if self.N > 1:
                for n1 in range(n+1, self.N):
                    for j in range(self.M):
                        self.model.addCons(
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
                        U[stage_idx][i] = quicksum(
                            self.Z[n, stage_idx, i, j] + U[stage_idx + 1][j]
                            for j in range(self.stage_sizes[stage_idx + 1])
                        )

                else:
                    U[stage_idx] = quicksum(
                        self.Z[n, stage_idx, 0, j] + U[stage_idx + 1][j]
                        for j in range(self.stage_sizes[stage_idx + 1])
                    )

            V.append(U[0])

        return quicksum(V)

    def optimize(self, time_limit=600):
        self.model.setParam("limits/time", time_limit)

        obj = self.compute_total_cost()
        self.model.setObjective(obj, "minimize")

        self.model.optimize()

    def print_solution(self):
        """
        Prints the optimized solution in a structured format (SCIP version).
        """

        status = self.model.getStatus()

        if status not in ["optimal", "timelimit", "gaplimit"]:
            print("No solution available.")
            print(f"SCIP Status: {status}")
            return

        print("\n=== SOLUTION SUMMARY ===")

        # Distance matrix
        print(f"\n--- Distance matrix ---")
        print(self.distMat)

        for n in range(self.N):
            print(f"\n==============================")
            print(f"Agent: {n}")
            print(f"==============================")

            s, d = self.start_goal[n]

            print(f"\n--- Start / Goal ---")
            print(f"Start: {s}\tTheta: {self.model.getVal(self.theta[n,s]):.3f}")
            print(f"Goal : {d}\tTheta: {self.model.getVal(self.theta[n,d]):.3f}")

            print("\n--- Theta values ---")
            for i in range(self.M):
                val = self.model.getVal(self.theta[n,i])
                print(f"theta[{n},{i}] = {val:.6f}")

            print("\n--- Inverse Speed ---")
            inv_val = self.model.getVal(self.inv_speed[n])
            print(f"inv_speed[{n}] = {inv_val:.6f}")

            print("\n--- Stagewise variables ---")
            for k in range(self.K):
                print(f"\n  Stage {k} -> {k+1}")
                for i in range(self.stage_sizes[k]):
                    for j in range(self.stage_sizes[k+1]):

                        z_val = self.model.getVal(self.Z[n,k,i,j])
                        eta_val = self.model.getVal(self.eta[n,k,i,j])
                        xi_val = self.model.getVal(self.Xi[n,k,i,j])

                        print(
                            f"Z[{n},{k},{i},{j}] = {z_val:.3e}, "
                            f"eta[{n},{k},{i},{j}] = {round(eta_val)}, "
                            f"Xi[{n},{k},{i},{j}] = {xi_val:.3e}"
                        )