import numpy as np
import cvxpy as cp
from MIRS import MIRS
import time
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds

class MIRSOptimizer:

    def __init__(
        self,
        mirs:MIRS, 
        optim_config:dict,
        anneal_config:dict
        ):

        self.mirs = mirs
        self.optim_config = optim_config
        self.init_b_arr(anneal_config)
        self.ra = anneal_config['ra']
        self.rb = anneal_config['rb']
        self.rw = anneal_config['rw']
        self.rh = anneal_config['rh']
        self.ra_rate = anneal_config['ra_rate']
        self.rb_rate = anneal_config['rb_rate']
        self.rw_rate = anneal_config['rw_rate']
        self.rh_rate = anneal_config['rh_rate']

    def init_b_arr(self, anneal_config):
        r = 10**((anneal_config['log_bmax'] - anneal_config['log_bmin'])/(anneal_config['nb']-1))
        b_arr = []
        for i in range(anneal_config['nb']):
            bi = (10**anneal_config['log_bmin'])*(r**i)
            b_arr.append(bi)

        self.b_arr = np.array(b_arr)
        

    def get_CBF_control(
        self, 
        sched_mat, 
        speed_vec, 
        weight_mat, 
        beta
        ):

        if self.optim_config['name'] == 'cbf_clf':
            gamma = self.optim_config['gamma']
            alpha_s =  self.optim_config['alpha_s']
            alpha_c = self.optim_config['alpha_c']
            alpha_r = self.optim_config['alpha_r']
            alpha_q = self.optim_config['alpha_q']
            p = self.optim_config['p']
            qp_solver_name = self.optim_config['QP_solver_name']
            qp_solver_opts = self.optim_config['qp_solver_opts']
        else:
            raise Exception("--Inside MIRSOptimizer.get_CBF_control--\n--Wrong solver configuration, check solver name--")

        mirs = self.mirs
        Nw = mirs.n_waypoints
        Na = mirs.n_agents

        # control decision variables
        U = cp.Variable((Na, Nw+1))
        delta = cp.Variable(1)
        
        # define objective, constraints and problem
        objective = cp.Minimize(cp.sum_squares(U) + p*delta**2)

        # get free energy and its time-derivative
        F, Grad_F = mirs.transportCost_v1(sched_mat, speed_vec, beta, returnGrad=True)
        if mirs.cost_mode == 'sum':
            F_dot = cp.sum(cp.multiply(mirs.agent_weights, cp.sum(cp.multiply(Grad_F, U), axis=1)))
        elif mirs.cost_mode == 'slowest':
            C_arr_lm = mirs.lm * mirs.C_agents
            C_arr_max = np.max(C_arr_lm)
            weights1 = np.exp(C_arr_lm - C_arr_max) / np.sum(np.exp(C_arr_lm - C_arr_max))
            F_dot = cp.sum(cp.multiply(weights1, cp.sum(cp.multiply(Grad_F, U), axis=1)))

        # speed speed cbf and its dot
        H_speed, Grad_H_speed = mirs.CBF_agents(speed_vec)
        Grad_H_diag = Grad_H_speed.diagonal().copy()
        H_speed_dot = cp.multiply(Grad_H_diag, U[:,-1])
        
        # pick agent start schedules and its dot
        start_indices = np.array([[i, a.s] for i, a in enumerate(mirs.agents)])
        tup_start_indices = (start_indices[:,0], start_indices[:,1])

        # compute conflict barrier functions
        H, Grad_H, filter_wp = mirs.CBF_waypoints_v1(sched_mat, mirs.active_waypoints, weight_mat, returnGrad=True)

        # bypass CBF constraint if no conflicts recorded in H
        if len(H) == 0:
            constraints = [
                F_dot <= -gamma * F + delta, # to decrease free energy
                H_speed_dot >= -alpha_s * H_speed, # speed limit constraint
                U[tup_start_indices] >= -alpha_r * (sched_mat[tup_start_indices]-mirs.start_times), # to maintain time-positivity
                U[:,:-1] <= alpha_q * (mirs.T_upper_bound - sched_mat) # to maintain the schedule bounded
                # U[dropped_indices] == 0.0 # no control for unvisited nodes
            ]
        else:
            H_dot_list = []
            cnt = 0
            for wp in mirs.active_waypoints:
                n_active_agents = int(np.sum(filter_wp[:,wp]))
                if n_active_agents > 1:
                    n_conflicts = n_active_agents * (n_active_agents-1) // 2
                    H_dot_list.append(
                        cp.sum(cp.multiply(Grad_H[cnt:cnt+n_conflicts], cp.reshape(U[:,wp], (1,Na), order='C')), axis=1, keepdims=True)
                    )
                    cnt += n_conflicts
            H_dot = cp.vstack(H_dot_list) 
            H_dot = cp.reshape(H_dot, (-1,), order='C')

            constraints = [
                F_dot <= -gamma * F + delta, # decrease free energy
                H_speed_dot >= -alpha_s * H_speed, # speed limit constraint
                H_dot >= -alpha_c * H, # collision avoidance constraint
                U[tup_start_indices] >= -alpha_r * (sched_mat[tup_start_indices]-mirs.start_times), # to maintain time-positivity
                U[:,:-1] <= alpha_q * (mirs.T_upper_bound - sched_mat) # to maintain the schedule bounded
                # U[dropped_indices] == 0.0 # no control for unvisited nodes
            ]

        problem = cp.Problem(objective, constraints)

        # Solve the problem using OSQP with customized options
        result = problem.solve(solver = qp_solver_name, **qp_solver_opts)

        # Check the results
        if np.isnan(problem.value).any() == True:
            print("Nan encountered inside get_CBF_control!")
            return None, None, None, None
        elif U.value is None or F_dot.value is None:
            print("None type returned inside get_CBF_control!")
            return None, None, None, None
        else:
            return U.value, F, F_dot.value, delta.value    

    # optimize at given beta using CBF CLF
    def CBF_CLF_at_beta(
        self,
        beta, 
        Tb, 
        Vb,
        weight_mat
        ):

        if self.optim_config['name'] == 'cbf_clf':
            dt_init = self.optim_config['dt_init']
            dt_min = self.optim_config['dt_min']
            dt_max = self.optim_config['dt_max']
            Tf = self.optim_config['Tf']
            stop_tol = self.optim_config['stop_tol']
            stop_tol_weight = self.optim_config['stop_tol_weight']
            verbose = self.optim_config['verbose']
        else:
            raise Exception("--Inside MIRSOptimizer.get_CBF_control--\n--Wrong solver configuration, check solver name--")

        Na = self.mirs.n_agents
        Nw = self.mirs.n_waypoints
    
        T_prev = Tb
        V_prev = Vb
        dt_prev = dt_init
        theta_prev = np.inf
        t = 0.0
        iter_count = 0
        F_prev = 1.0

        while t < Tf:
            # get control
            U, F, Fdot, delta = self.get_CBF_control(
                T_prev, 
                V_prev, 
                weight_mat, 
                beta)

            if None in (F, Fdot, delta):
                print('None encountered from get_CBF_control... returning previous iterate!')
                return T_prev, V_prev, np.zeros(shape=(Na,Nw+1)), 0.0, 0.0, 0.0, 0.0

            # compute new stepsize
            if iter_count > 0:
                step_size_1 = np.sqrt(1 + theta_prev) * dt_prev
                grad_diff = np.linalg.norm(U - U_old) + 1e-6  # Regularization term to prevent division by zero
                step_size_2 = (np.linalg.norm(T_prev - T_old) + np.linalg.norm(V_prev - V_old)) / (2 * grad_diff)  
                dt = min(max(step_size_2, dt_min), dt_max)  # Keep dt in range [dt_min, dt_max]
            else:
                dt = dt_init

            # Euler update
            T_next = T_prev + dt * U[:,:-1]
            V_next = V_prev + dt * U[:,-1]

            # compute new theta_k
            if iter_count > 0:
                theta = dt/dt_prev
            else:
                theta = 1.0

            tol_T = np.max(np.abs(T_next-T_prev))/np.max(np.abs(T_prev))
            tol_V = np.max(np.abs(V_next-V_prev))/np.max(np.abs(V_prev))
            tol_F = np.abs(F - F_prev)/np.abs(F_prev)
            tol_Fdot = abs(Fdot * dt)/np.abs(F_prev)
            stop_tol_weight = stop_tol_weight/np.sum(stop_tol_weight)
            tol = np.sum(stop_tol_weight * np.array([tol_T, tol_V, tol_F, tol_Fdot]))
            if tol < stop_tol:
                if verbose == 1 or verbose == 2:
                    print(f'\tt:{t:.3e}\tdt:{dt:.4e}\tF:{F:.4f}\tFdot:{Fdot:.4f}\tdelta:{delta[0]:.4f}\ttol:{tol:.3e}')
                break
            if verbose == 2:
                print(f'\tt:{t:.3e}\tdt:{dt:.4e}\tF:{F:.4f}\tFdot:{Fdot:.4f}\tdelta:{delta[0]:.4f}\ttol:{tol:.3e}')
            
            # Update variables for next iteration
            T_old = T_prev
            V_old = V_prev
            T_prev = T_next
            V_prev = V_next
            U_old = U
            dt_prev = dt
            theta_prev = theta
            t += dt
            iter_count += 1
            F_prev = F

        return T_prev, V_prev, U, F, Fdot, tol, delta[0]


    def slsqp_at_beta(
        self, 
        beta,
        T0, 
        V0,
        weight_mat
        ):
        t0 = time.time()

        if self.optim_config['name'] == 'slsqp':
            stop_tol = self.optim_config['stop_tol']
            disp =  self.optim_config['disp']
            maxiter = self.optim_config['maxiter']
        else:
            raise Exception("--Inside MIRSOptimizer.slsqp_at_beta--\n--Wrong solver configuration, check solver name--")

        N = self.mirs.n_agents
        M = self.mirs.n_waypoints

        # define objective
        Fx = lambda x : self.mirs.transportCost_v1(x[0:N*M].reshape(-1,M), x[N*M:], beta, returnGrad=False)[0]
        Grad_Fx = lambda x : self.mirs.transportCost_v1(x[0:N*M].reshape(-1,M), x[N*M:], beta, returnGrad=True)[1]

        # define constraint functions
        H1x = lambda x : self.mirs.CBF_waypoints_v1(x[0:N*M].reshape(-1,M), self.mirs.active_waypoints, weight_mat, returnGrad=False)[0]
        Grad_H1x = lambda x : self.mirs.CBF_waypoints_v1(x[0:N*M].reshape(-1,M), self.mirs.active_waypoints, weight_mat, returnGrad=True)[1]

        # H2x = lambda x : self.mirs.CBF_agents(x[N*M:])[0]
        # Grad_H2x = lambda x : self.mirs.CBF_agents(x[N*M:])[1]

        constraints = {
            'type':'ineq',
            'fun':H1x
        }

        # define bounds
        lb = np.concatenate(
            (np.zeros(N*M),
            self.mirs.speed_lim_mat[:,0])
        )
        ub = np.concatenate(
            (np.ones(N*M)*self.mirs.T_upper_bound,
            self.mirs.speed_lim_mat[:,1])
        )

        bounds = Bounds(lb, ub)

        res = minimize(
            Fx, 
            np.concatenate((T0.flatten(), V0)), 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'ftol': stop_tol,
                'disp': disp,
                'maxiter':maxiter
            }
        )

        x = res.x
        f = res.fun
        Tf = x[0:N*M].reshape((-1,M))
        Vf = x[N*M:]
        t1 = time.time()

        return Tf, Vf, f, t1-t0


    def anneal(
        self,
        T0,
        V0,
        active_waypoints,
        annealPrint=False,
        ):

        Tb = T0
        Vb = V0
        ra = self.ra
        rb = self.rb
        rw = self.rw
        rh = self.rh
        mirs = self.mirs

        # elif optimizer['name'] == 'SLSQP':
        #     stop_tol        = optimizer['stop_tol']
        #     disp            = optimizer['disp']

        for i, beta in enumerate(self.b_arr):
            t0 = time.time()
            weight_mat, _ = mirs.calc_agent_reach_mat_v1(Tb, Vb, beta)
            filter_wp = np.ones(weight_mat.shape)
            filter_wp[weight_mat <= mirs.filter_wp_thresh] = 0.0
            if active_waypoints is None:
                mirs.active_waypoints = list(np.where(np.sum(filter_wp, axis=0)>=2)[0])
            else:
                mirs.active_waypoints = active_waypoints

            if mirs.ca_cbf['mode'] == 'ellipse':
                if beta <= 1:
                    mirs.ca_cbf['major_axis'] = ra**2 / (ra**2 + 1) * np.ones(mirs.tolArray.shape) * 200
                    mirs.ca_cbf['minor_axis'] = rb**2 / (rb**2 + 1) * mirs.tolArray
                else:
                    mirs.ca_cbf['major_axis'] = ra**3 / (ra**3 + 1) * np.ones(mirs.tolArray.shape) * 200
                    mirs.ca_cbf['minor_axis'] = rb**3 / (rb**3 + 1) * mirs.tolArray
                ra = ra*self.ra_rate
                rb = rb*self.rb_rate
                if annealPrint:
                    a, b = mirs.ca_cbf['major_axis'][0], mirs.ca_cbf['minor_axis'][0]
                    print(f'a:{a:.3f}\tb:{b:.3f}')
            elif mirs.ca_cbf['mode'] == 'rect':
                mirs.ca_cbf['width'] = rw**2 / (rw**2 + 1) * np.ones(mirs.tolArray.shape) * mirs.T_upper_bound
                mirs.ca_cbf['height'] = rh/ (rh + 1) * mirs.tolArray
                rw = rw*self.rw_rate
                rh = rh*self.rh_rate
                if annealPrint:
                    w, h = mirs.ca_cbf['width'][0], mirs.ca_cbf['height'][0]
                    print(f'w:{w:.3f}\th:{h:.3f}')

            if self.optim_config['name'] == 'cbf_clf':
                Tb, Vb, Ub, Fb, Fdot_b, tolb, delta_b = self.CBF_CLF_at_beta(
                    beta, Tb, Vb, weight_mat)
                Hb, GradHb, filter_wp = mirs.CBF_waypoints_v1(Tb, mirs.active_waypoints, weight_mat, returnGrad=True)
                
                if annealPrint:
                    print(f'\nbeta: {beta:.4e}\tcost: {Fb:.3f}\ttolb: {tolb:.3e}\tn_active_waypoints:{len(mirs.active_waypoints)}\ttol_mag:{mirs.tolArray[0]:.2f}\tHbshape:{Hb.shape}')

            elif self.optim_config['name'] == 'slsqp':
                Tb, Vb, Fb, compute_time = self.slsqp_at_beta(
                    beta, Tb, Vb, weight_mat
                )
                if annealPrint:
                    print(f'\nbeta: {beta:.4e}\tcost: {Fb:.3f}\ttol_mag:{mirs.tolArray[0]:.2f}')
            t1 = time.time()

            if i == 0:
                Tb_array = np.array([Tb])
                Vb_array = np.array([Vb])
                Fb_array = np.array([Fb])
                chi_array = np.array([weight_mat])
                t_compute_array = np.array([t1-t0])
            else:
                Tb_array = np.concatenate((Tb_array, np.array([Tb])))
                Vb_array = np.concatenate((Vb_array, np.array([Vb])))
                Fb_array = np.concatenate((Fb_array, np.array([Fb])))
                chi_array = np.concatenate((chi_array, np.array([weight_mat])))
                t_compute_array = np.concatenate((t_compute_array, np.array([t1-t0])))

        # compute final probability associations
        Pb_a = []
        for i,a in enumerate(mirs.agents):
            Pb = a.getPathAssociations_v1(Tb[i,:], Vb[i], mirs.dist_mat, self.b_arr[-1])
            Pb_a.append(Pb)

        return Tb_array, Vb_array, Fb_array, Pb_a, chi_array, t_compute_array
