import gurobipy as gp
from gurobipy import GRB

class MIRS_Gurobi:
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
        self.model = gp.Model("single_agent_MIRS")

        # Variable containers
        self.theta = {}
        self.inv_speed = {}
        self.Z = {}
        self.eta = {}
        self.Xi = {}

        # Build model
        self._build_variables()
        self._build_constraints()

        self.model.Params.NonConvex = 2
        self.model.update()

    def _build_variables(self):

        for n in range(self.N):
            # Continuous variables
            for i in range(self.M):
                self.theta[n, i] = self.model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=self.theta_lb,
                    ub=self.theta_ub,
                    name=f"T_{n}_{i}"
                )

            self.inv_speed[n] = self.model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=self.inv_lb,
                ub=self.inv_ub,
                name=f"inv_speed_{n}"
            )

            # stage variables
            for k in range(self.K):
                for i in range(self.stage_sizes[k]):
                    for j in range(self.stage_sizes[k+1]):
                        self.Z[n,k,i,j] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"Z_{n}_{k}_{i}_{j}")
                        self.eta[n,k,i,j] = self.model.addVar(vtype=GRB.BINARY, name=f"eta_{n}_{k}_{i}_{j}")
                        self.Xi[n,k,i,j] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"Xi_{n}_{k}_{i}_{j}")

    def _build_constraints(self):
        for n in range(self.N):
            s, d = self.start_goal[n]

            # Build masked network
            net_mask = self.mask.copy()
            net_mask[d, :] = 0
            net_mask[d, d] = 1

            distMat_temp = self.distMat.copy()
            distMat_temp[net_mask == 0] = self.INF

            for k in range(self.K):
                for i in range(self.stage_sizes[k]):
                    for j in range(self.stage_sizes[k+1]):

                        # Product linearization (nonconvex)
                        self.model.addQConstr(
                            self.Z[n,k,i,j] == self.eta[n,k,i,j] * self.Xi[n,k,i,j],
                            name=f"prod_{n}_{k}_{i}_{j}"
                        )

                        # Build cost expression
                        expr = None

                        if k == 0:
                            if net_mask[s, j] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[s, j] * self.inv_speed[n]

                                expr0 = self.c0 * (self.theta[n,s] - self.start_times[n])*(self.theta[n,s] - self.start_times[n])
                                expr1 = self.c1 * (self.theta[n,j] - self.theta[n,s] - travel_time)*(self.theta[n,j] - self.theta[n,s] - travel_time)
                                expr2 = self.c2 * travel_time*travel_time
                                expr = expr0 + expr1 + expr2

                        elif k < self.K - 1:
                            if net_mask[i, j] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[i, j] * self.inv_speed[n]

                                expr1 = self.c1 * (self.theta[n,j] - self.theta[n,i] - travel_time)*(self.theta[n,j] - self.theta[n,i] - travel_time)
                                expr2 = self.c2 * travel_time*travel_time
                                expr = expr1 + expr2

                        else:  # k == K-1
                            if net_mask[i, d] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[i, d] * self.inv_speed[n]

                                expr1 = self.c1 * (self.theta[n,d] - self.theta[n,i] - travel_time)*(self.theta[n,d] - self.theta[n,i] - travel_time)
                                expr2 = self.c2 * travel_time*travel_time
                                expr = expr1 + expr2

                        # Cost constraint
                        self.model.addConstr(
                            self.Xi[n,k,i,j] == expr,
                            name=f"cost_{n}_{k}_{i}_{j}"
                        )

                    if k > 0 and k < self.K:
                        if k < self.K-1:
                            # at most one path from each node
                            self.model.addConstr(
                                gp.quicksum(self.eta[n,k,i,j]*net_mask[i,j] for j in range(self.stage_sizes[k+1])) <= 1,
                                name=f"assign_{n}_{k}_{i}"
                            )

                        # number of paths in must be equal to number of paths out
                        if k == 1:
                            self.model.addConstr(
                                self.eta[n,k-1,0,i]*net_mask[s,i]
                                     == gp.quicksum(
                                        self.eta[n,k,i,i_]*net_mask[i,i_] for i_ in range(self.stage_sizes[k+1])
                                        ),
                                name = f"flow_{n}_{k}_{i}"
                            )
                        elif k == self.K-1:
                            self.model.addConstr(
                                gp.quicksum(
                                    self.eta[n,k-1,_i,i]*net_mask[_i,i] for _i in range(self.stage_sizes[k-1])
                                    ) == self.eta[n,k,i,0]*net_mask[i,d],
                                name = f"flow_{n}_{k}_{i}"
                            )
                        else:
                            self.model.addConstr(
                                gp.quicksum(
                                    self.eta[n,k-1,_i,i]*net_mask[_i,i] for _i in range(self.stage_sizes[k-1])
                                    ) == gp.quicksum(
                                        self.eta[n,k,i,i_]*net_mask[i,i_] for i_ in range(self.stage_sizes[k+1])
                                        ),
                                name = f"flow_{n}_{k}_{i}"
                            )

                # exactly one path from start
                if k == 0:
                    self.model.addConstr(
                        gp.quicksum(self.eta[n,k,0,j]*net_mask[s,j] for j in range(self.stage_sizes[k+1])) == 1,
                    )
                # exactly one path to goal
                elif k == self.K-1:
                    self.model.addConstr(
                        gp.quicksum(self.eta[n,k,i,0]*net_mask[i,d] for i in range(self.stage_sizes[k])) == 1,
                    )

            # Collision constraints
            if self.N > 1:
                for n1 in range(n+1, self.N):
                    for j in range(self.M):
                        self.model.addQConstr(
                            (self.theta[n,j] - self.theta[n1,j])**2 >= self.tolerance[j]*self.tolerance[j],
                            name=f"collision_{n}_{n1}_{j}"
                        )

    def compute_total_cost(self):
        """
        Computes the dynamic programming cost expression from Z variables.
        Returns a Gurobi expression.
        """
        V = [None] * self.N

        for n in range(self.N):
            U = [None] * self.K

            for k in range(self.K):
                stage_idx = self.K - 1 - k

                # Final stage to destination
                if k == 0:
                    U[stage_idx] = {}
                    j = 0
                    for i in range(self.stage_sizes[stage_idx]):
                        U[stage_idx][i] = self.Z[n, stage_idx, i, j]

                # Intermediate stages
                elif k < self.K - 1:
                    U[stage_idx] = {}
                    for i in range(self.stage_sizes[stage_idx]):
                        expr = []
                        for j in range(self.stage_sizes[stage_idx + 1]):
                            expr.append(self.Z[n, stage_idx, i, j] + U[stage_idx + 1][j])
                        U[stage_idx][i] = gp.quicksum(expr)

                # First stage
                else:
                    i = 0
                    expr = []
                    for j in range(self.stage_sizes[stage_idx + 1]):
                        expr.append(self.Z[n, stage_idx, i, j] + U[stage_idx + 1][j])
                    U[stage_idx] = gp.quicksum(expr)

            V[n] = U[0]

        return gp.quicksum(V)

    def optimize(self, time_limit=600, mip_gap=0.05):
        """
        Set solver parameters, define objective, and optimize the model.
        """

        # --- Solver parameters ---
        self.model.setParam("NonConvex", 2)

        self.model.setParam("TimeLimit", time_limit)
        self.model.setParam("MIPGap", mip_gap)

        self.model.setParam("MIPFocus", 1)
        self.model.setParam("Heuristics", 0.8)

        self.model.setParam("Cuts", 2)
        self.model.setParam("OBBT", 2)

        self.model.setParam("NumericFocus", 2)
        self.model.setParam("Presolve", 2)

        # --- Objective ---
        obj = self.compute_total_cost()
        self.model.setObjective(obj, GRB.MINIMIZE)

        # --- Optimize ---
        self.model.optimize()


    def print_solution(self):
        """
        Prints the optimized solution in a structured format.
        """

        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            print("No solution available.")
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
            print(f"Start: {s}\tTheta: {self.theta[n,s].X:.3f}")
            print(f"Goal : {d}\tTheta: {self.theta[n,d].X:.3f}")

            print("\n--- Theta values ---")
            for i in range(self.M):
                print(f"theta[{n},{i}] = {self.theta[n,i].X:.6f}")

            print("\n--- Inverse Speed ---")
            print(f"inv_speed[{n}] = {self.inv_speed[n].X:.6f}")

            print("\n--- Stagewise variables ---")
            for k in range(self.K):
                print(f"\n  Stage {k} -> {k+1}")
                for i in range(self.stage_sizes[k]):
                    for j in range(self.stage_sizes[k+1]):

                        z_val = self.Z[n,k,i,j].X
                        eta_val = self.eta[n,k,i,j].X
                        xi_val = self.Xi[n,k,i,j].X

                        # Optional: only print active edges
                        # if abs(z_val) > 1e-6 or eta_val > 0.5:
                        print(
                            f"Z[{n},{k},{i},{j}] = {z_val:.3e}, "
                            f"eta[{n},{k},{i},{j}] = {eta_val:.0f}, "
                            f"Xi[{n},{k},{i},{j}] = {xi_val:.3e}"
                        )


    def get_model(self):
        return self.model