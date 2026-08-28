import gurobipy as gp
from gurobipy import GRB
import numpy as np
from matplotlib.cm import get_cmap 
from matplotlib import cm  # for colormap
import matplotlib as mpl
import matplotlib.pyplot as plt 
from MIRS import MIRS


class MIRSGurobiOptimizer:
    def __init__(
        self,
        mirs:MIRS,
        model_name:str
    ):
        # Store inputs
        self.mirs = mirs
        self.N = mirs.n_agents
        self.M = mirs.n_waypoints 
        self.K = self.M + 1 
        self.init_stages() 
        self.start_goal = mirs.sd_mat
        self.theta_lb, self.theta_ub = (0,mirs.T_upper_bound) 
        self.inv_speed_lb, self.inv_speed_ub = (1/np.max(mirs.speed_lim_mat) , 1/np.min(mirs.speed_lim_mat) ) 
        self.INF = mirs.INF
        self.c0, self.c1, self.c2 = mirs.stagewiseCostCoeffs 
        self.mask = mirs.mask
        self.start_times = mirs.start_times
        self.distMat = mirs.dist_mat
        self.tolerance = mirs.tolArray

        # Model
        self.model = gp.Model(model_name)

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

    def init_stages(self):
        # initialize stage sizes 
        self.stage_sizes = [] 
        for k in range(self.K+1):
            if k == 0 or k == self.K:
                self.stage_sizes.append(1)
            else:
                self.stage_sizes.append(self.M)


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
                lb=self.inv_speed_lb,
                ub=self.inv_speed_ub,
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
                                expr2 = self.c2 * travel_time
                                expr = expr0 + expr1 + expr2

                        elif k < self.K - 1:
                            if net_mask[i, j] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[i, j] * self.inv_speed[n]

                                expr1 = self.c1 * (self.theta[n,j] - self.theta[n,i] - travel_time)*(self.theta[n,j] - self.theta[n,i] - travel_time)
                                expr2 = self.c2 * travel_time
                                expr = expr1 + expr2

                        else:  # k == K-1
                            if net_mask[i, d] == 0:
                                expr = self.INF
                            else:
                                travel_time = distMat_temp[i, d] * self.inv_speed[n]

                                expr1 = self.c1 * (self.theta[n,d] - self.theta[n,i] - travel_time)*(self.theta[n,d] - self.theta[n,i] - travel_time)
                                expr2 = self.c2 * travel_time
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
        # V = [None] * self.N

        # for n in range(self.N):
        #     U = [None] * self.K

        #     for k in range(self.K):
        #         stage_idx = self.K - 1 - k

        #         # Final stage to destination
        #         if k == 0:
        #             U[stage_idx] = {}
        #             j = 0
        #             for i in range(self.stage_sizes[stage_idx]):
        #                 U[stage_idx][i] = self.Z[n, stage_idx, i, j]

        #         # Intermediate stages
        #         elif k < self.K - 1:
        #             U[stage_idx] = {}
        #             for i in range(self.stage_sizes[stage_idx]):
        #                 expr = []
        #                 for j in range(self.stage_sizes[stage_idx + 1]):
        #                     expr.append(self.Z[n, stage_idx, i, j] + U[stage_idx + 1][j])
        #                 U[stage_idx][i] = gp.quicksum(expr)

        #         # First stage
        #         else:
        #             i = 0
        #             expr = []
        #             for j in range(self.stage_sizes[stage_idx + 1]):
        #                 expr.append(self.Z[n, stage_idx, i, j] + U[stage_idx + 1][j])
        #             U[stage_idx] = gp.quicksum(expr)

        #     V[n] = U[0]

        return gp.quicksum(var for var in self.Z.values())


    # -------------------------
    # Stagnation callback
    # -------------------------
    def _stagnation_callback(self, model, where):
        """
        Terminates the optimization if incumbent cost hasn't improved much.
        """
        if where == GRB.Callback.MIP:
            new_obj = model.cbGet(GRB.Callback.MIP_OBJBST)

            if new_obj is None:
                # No feasible solution yet
                return

            # Initialize if first valid incumbent
            if self._best_incumbent is None:
                self._best_incumbent = new_obj
                self._stagnation_counter = 0
            else:
                # Check improvement
                if abs(new_obj - self._best_incumbent) > 1e-6:
                    self._best_incumbent = new_obj
                    self._stagnation_counter = 0
                else:
                    self._stagnation_counter += 1

            # Terminate if stagnated for too long
            if self._stagnation_counter >= getattr(self, "_stagnation_limit", 50):
                model.terminate()

    # -------------------------
    # Optimize method
    # -------------------------
    def optimize(self, time_limit=600, mip_gap=0.05, stagnation_limit=50):
        """
        Optimize the model with early stopping based on stagnation.
        """

        # Solver parameters
        self.model.Params.NonConvex = 2
        self.model.Params.TimeLimit = time_limit
        self.model.Params.MIPGap = mip_gap

        self.model.Params.MIPFocus = 1
        self.model.Params.Heuristics = 0.8
        self.model.Params.Cuts = 2
        self.model.Params.OBBT = 2
        self.model.Params.NumericFocus = 2
        self.model.Params.Presolve = 2

        # Stagnation limit
        self._stagnation_limit = stagnation_limit
        self._best_incumbent = None
        self._stagnation_counter = 0

        # Objective
        obj = self.compute_total_cost()
        self.model.setObjective(obj, GRB.MINIMIZE)

        # Optimize with callback
        self.model.optimize(self._stagnation_callback)

    def get_path_associations(self):
        Pb = [None]*self.N
        for n in range(self.N):
            Pb[n] = [None]*self.K
            for k in range(self.K):
                Pb[n][k] = np.ones((self.stage_sizes[k], self.stage_sizes[k+1]))
                for i in range(self.stage_sizes[k]):
                    for j in range(self.stage_sizes[k+1]):
                        # print(n,k,i,j,Pb[n][k])
                        print(n,k,i,j,"\t",self.eta[n,k,i,j].X)
                        Pb[n][k][i,j] = self.eta[n,k,i,j].X


    def print_solution(self):
        """
        Prints the optimized solution in a structured format.
        """

        if self.model.SolCount == 0:
            print("No feasible solution found yet.")
            return

        # if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        #     print("No solution available.")
        #     return

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


    def extract_routes_and_schedules(self):
        """
        Extract agent routes, schedules, schedule matrix, and association matrix
        from the Gurobi solution, ensuring no repeated waypoints and integer indices.

        Returns:
            agent_routes      : list of lists of waypoint IDs per agent
            agent_schedules   : list of lists of theta values per agent
            schedule_matrix   : n_agents x n_waypoints matrix of theta
            association_mat   : n_agents x n_waypoints matrix of 0/1
        """
        if self.model.SolCount == 0:
            print("No feasible solution to extract.")
            return None, None, None, None

        n_agents = self.N
        n_waypoints = self.M

        agent_routes = []
        agent_schedules = []
        agent_speeds = []
        schedule_matrix = [[0.0 for _ in range(n_waypoints)] for _ in range(n_agents)]
        association_mat = [[0 for _ in range(n_waypoints)] for _ in range(n_agents)]

        for n in range(n_agents):
            route = []
            schedule = [0.0]
            visited = set()  # Track visited waypoints to avoid repetition

            s, d = self.start_goal[n]
            current_wp = int(s)
            route.append(current_wp)
            schedule.append(self.theta[n,current_wp].X)
            visited.add(current_wp)
            association_mat[n][current_wp] = 1
            schedule_matrix[n][current_wp] = self.theta[n,current_wp].X

            # Traverse stages
            for k in range(self.K):
                next_wp = None
                for i in range(self.stage_sizes[k]):
                    for j in range(self.stage_sizes[k+1]):
                        eta_val = self.eta[n,k,i,j].X
                        if eta_val > 0.5:
                            # Determine the index corresponding to current_wp
                            if k == 0:
                                idx_i = int(s)
                            else:
                                idx_i = int(i)
                            if idx_i == current_wp:
                                next_wp = int(j)
                                break
                    if next_wp is not None:
                        break

                if next_wp is None or next_wp in visited:
                    # Stop if no next stage or waypoint already visited
                    break

                current_wp = next_wp
                route.append(current_wp)
                schedule.append(self.theta[n,current_wp].X)
                visited.add(current_wp)
                association_mat[n][current_wp] = 1
                schedule_matrix[n][current_wp] = self.theta[n,current_wp].X

            agent_routes.append(route)
            agent_schedules.append(schedule)
            agent_speeds.append(self.inv_speed[n].X)

        return agent_routes, agent_schedules, agent_speeds, np.array(schedule_matrix), np.array(association_mat)


    def plot_waypoint_agent_schedules(
        self,routes, schedules, schedule_matrix, association_matrix,
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
                    print(schedule, i+1)
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
