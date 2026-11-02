import time

import cvxpy as cp
import numpy as np


class MIRSOptimizer:
    """CBF-CLF annealing optimizer for a coarse-network MIRS instance."""

    def __init__(self, mirs, optim_config, anneal_config):
        if optim_config.get("name") != "cbf_clf":
            raise ValueError("higher_mep_opt supports only the 'cbf_clf' optimizer")
        self.mirs = mirs
        self.optim_config = optim_config
        self.b_arr = self._make_beta_array(anneal_config)
        for name in ("ra", "rb", "rw", "rh", "ra_rate", "rb_rate", "rw_rate", "rh_rate"):
            if name not in anneal_config:
                raise ValueError(f"Missing annealing parameter: {name}")
            setattr(self, name, anneal_config[name])

    @staticmethod
    def _make_beta_array(anneal_config):
        n_beta = int(anneal_config["nb"])
        if n_beta < 1:
            raise ValueError("anneal_config['nb'] must be positive")
        beta_min = float(anneal_config["log_bmin"])
        beta_max = float(anneal_config["log_bmax"])
        return np.logspace(beta_min, beta_max, n_beta)

    def get_CBF_control(self, sched_mat, speed_vec, weight_mat, beta):
        config = self.optim_config
        mirs = self.mirs
        n_agents, n_waypoints = mirs.n_agents, mirs.n_waypoints

        control = cp.Variable((n_agents, n_waypoints + 1))
        slack = cp.Variable()
        free_energy, grad_free_energy = mirs.transportCost_v1(
            sched_mat, speed_vec, beta, returnGrad=True
        )
        agent_energy_grad = cp.sum(
            cp.multiply(grad_free_energy, control), axis=1
        )
        if mirs.cost_mode == "sum":
            free_energy_dot = cp.sum(
                cp.multiply(mirs.agent_weights, agent_energy_grad)
            )
        else:
            scaled_costs = mirs.lm * mirs.C_agents
            weights = np.exp(scaled_costs - np.max(scaled_costs))
            weights /= weights.sum()
            free_energy_dot = cp.sum(cp.multiply(weights, agent_energy_grad))

        speed_barrier, speed_barrier_grad = mirs.CBF_agents(speed_vec)
        speed_barrier_dot = cp.multiply(
            speed_barrier_grad.diagonal(), control[:, -1]
        )
        waypoint_barrier, waypoint_grad, filter_wp = mirs.CBF_waypoints_v1(
            sched_mat,
            mirs.active_waypoints,
            weight_mat,
            returnGrad=True,
        )

        constraints = [
            free_energy_dot <= -config["gamma"] * free_energy + slack,
            speed_barrier_dot >= -config["alpha_s"] * speed_barrier,
        ]
        if len(waypoint_barrier):
            barrier_dots = []
            offset = 0
            for waypoint in mirs.active_waypoints:
                n_active = int(np.sum(filter_wp[:, waypoint]))
                if n_active < 2:
                    continue
                n_pairs = n_active * (n_active - 1) // 2
                barrier_dots.append(
                    waypoint_grad[offset : offset + n_pairs]
                    @ control[:, waypoint]
                )
                offset += n_pairs
            if offset != len(waypoint_barrier):
                raise ValueError("Waypoint CBF values and gradients are misaligned")
            constraints.append(
                cp.hstack(barrier_dots) >= -config["alpha_c"] * waypoint_barrier
            )

        start_indices = np.arange(n_agents)
        start_waypoints = mirs.sd_mat[:, 0]
        constraints.extend(
            [
                control[start_indices, start_waypoints]
                >= -config["alpha_r"]
                * (sched_mat[start_indices, start_waypoints] - mirs.start_times),
                control[:, :-1]
                <= config["alpha_q"] * (mirs.T_upper_bound - sched_mat),
            ]
        )

        objective = cp.Minimize(
            cp.sum_squares(control) + config["p"] * cp.square(slack)
        )
        problem = cp.Problem(objective, constraints)
        problem.solve(
            solver=config["QP_solver_name"],
            **config["qp_solver_opts"],
        )
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(
                f"CBF QP failed at beta={beta}: status={problem.status}, "
                f"solver={config['QP_solver_name']}"
            )
        if control.value is None or slack.value is None or free_energy_dot.value is None:
            raise RuntimeError(f"CBF QP returned incomplete values at beta={beta}")
        return control.value, free_energy, float(free_energy_dot.value), float(slack.value)

    def CBF_CLF_at_beta(self, beta, sched_mat, speed_vec, weight_mat):
        config = self.optim_config
        dt_min = float(config["dt_min"])
        dt_max = float(config["dt_max"])
        dt_init = float(config["dt_init"])
        final_time = float(config["Tf"])
        if not (0 < dt_min <= dt_init <= dt_max) or final_time < 0:
            raise ValueError("Invalid CBF integration step or final time")

        schedule = np.asarray(sched_mat, dtype=float).copy()
        speed = np.asarray(speed_vec, dtype=float).copy()
        elapsed = 0.0
        previous_control = None
        previous_schedule = None
        previous_speed = None
        previous_energy = None
        tolerance = np.inf
        control = np.zeros((self.mirs.n_agents, self.mirs.n_waypoints + 1))
        free_energy = 0.0
        free_energy_dot = 0.0
        slack = 0.0
        stopped_early = False
        stop_weights = np.asarray(config["stop_tol_weight"], dtype=float)
        if stop_weights.shape != (4,) or np.any(stop_weights < 0) or stop_weights.sum() <= 0:
            raise ValueError("stop_tol_weight must contain four nonnegative, nonzero weights")
        stop_weights /= stop_weights.sum()

        max_steps = int(config.get("max_steps", 10000))
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        for iteration in range(max_steps):
            if elapsed >= final_time:
                break
            control, free_energy, free_energy_dot, slack = self.get_CBF_control(
                schedule, speed, weight_mat, beta
            )
            if iteration == 0:
                dt = dt_init
            else:
                delta_state = np.linalg.norm(schedule - previous_schedule) + np.linalg.norm(
                    speed - previous_speed
                )
                delta_control = np.linalg.norm(control - previous_control) + 1e-6
                dt = np.clip(delta_state / (2 * delta_control), dt_min, dt_max)
            dt = min(float(dt), final_time - elapsed)

            next_schedule = schedule + dt * control[:, :-1]
            next_speed = speed + dt * control[:, -1]
            if previous_energy is None:
                energy_change = abs(free_energy_dot * dt) / max(abs(free_energy), 1.0)
            else:
                energy_change = abs(free_energy - previous_energy) / max(
                    abs(previous_energy), 1.0
                )
            metrics = np.array(
                [
                    np.max(np.abs(next_schedule - schedule))
                    / max(np.max(np.abs(schedule)), 1.0),
                    np.max(np.abs(next_speed - speed))
                    / max(np.max(np.abs(speed)), 1.0),
                    energy_change,
                    abs(free_energy_dot * dt) / max(abs(free_energy), 1.0),
                ]
            )
            tolerance = float(stop_weights @ metrics)

            previous_schedule, previous_speed = schedule, speed
            previous_control = control
            previous_energy = free_energy
            schedule, speed = next_schedule, next_speed
            elapsed += dt
            if tolerance < config["stop_tol"]:
                stopped_early = True
                break
        if elapsed < final_time and not stopped_early and iteration + 1 >= max_steps:
            raise RuntimeError(
                f"CBF integration exceeded max_steps={max_steps} at beta={beta}"
            )

        final_energy, _ = self.mirs.transportCost_v1(
            schedule, speed, beta, returnGrad=False
        )
        return schedule, speed, control, final_energy, free_energy_dot, tolerance, slack

    def _update_barrier_geometry(
        self, beta_index, beta, n_beta, ra, rb, rw, rh, anneal_print=False
    ):
        mirs = self.mirs
        if mirs.ca_cbf["mode"] == "ellipse":
            if beta <= 1:
                power = 2
            else:
                power = 3
            mirs.ca_cbf["major_axis"] = (
                ra**power / (ra**power + 1)
                * np.ones(mirs.tolArray.shape)
                * 200
            )
            mirs.ca_cbf["minor_axis"] = (
                rb**power / (rb**power + 1) * mirs.tolArray
            )
            ra *= self.ra_rate
            rb *= self.rb_rate
            if anneal_print:
                print(
                    f"a:{mirs.ca_cbf['major_axis'][0]:.3f}\t"
                    f"b:{mirs.ca_cbf['minor_axis'][0]:.3f}"
                )
        elif mirs.ca_cbf["mode"] == "rect":
            initial_width = (
                rw**2 / (rw**2 + 1) * mirs.T_upper_bound
            )
            initial_height = rh / (rh + 1) * mirs.tolArray
            progress = beta_index / max(n_beta - 1, 1)
            mirs.ca_cbf["width"] = np.ones(mirs.tolArray.shape) * initial_width * (
                mirs.T_upper_bound / initial_width
            ) ** progress
            mirs.ca_cbf["height"] = initial_height * (
                mirs.tolArray / initial_height
            ) ** progress
            if anneal_print:
                print(
                    f"w:{mirs.ca_cbf['width'][0]:.3f}\t"
                    f"h:{mirs.ca_cbf['height'][0]:.3f}"
                )
        return ra, rb, rw, rh

    def anneal(
        self,
        T0,
        V0,
        active_waypoints=None,
        time_limit=np.inf,
        annealPrint=False,
    ):
        mirs = self.mirs
        schedule = np.asarray(T0, dtype=float).copy()
        speed = np.asarray(V0, dtype=float).copy()
        schedule_history, speed_history = [], []
        cost_history, reach_history, compute_times = [], [], []
        start_time = time.time()
        ra, rb, rw, rh = self.ra, self.rb, self.rw, self.rh

        for index, beta in enumerate(self.b_arr):
            beta_start = time.time()
            reach_mat, _ = mirs.calc_agent_reach_mat_v1(schedule, speed, beta)
            filter_wp = np.ones(reach_mat.shape)
            if mirs.prune_mode:
                filter_wp[reach_mat <= mirs.filter_wp_thresh] = 0.0
            mirs.active_waypoints = (
                list(np.flatnonzero(np.sum(filter_wp, axis=0) >= 2))
                if active_waypoints is None
                else list(active_waypoints)
            )
            ra, rb, rw, rh = self._update_barrier_geometry(
                index, beta, len(self.b_arr), ra, rb, rw, rh, annealPrint
            )
            (
                schedule,
                speed,
                _,
                cost,
                _,
                tolerance,
                _,
            ) = self.CBF_CLF_at_beta(beta, schedule, speed, reach_mat)

            schedule_history.append(schedule.copy())
            speed_history.append(speed.copy())
            cost_history.append(cost)
            reach_history.append(reach_mat.copy())
            compute_times.append(time.time() - beta_start)
            if annealPrint:
                print(
                    f"\nbeta:{beta:.4e}\tcost:{cost:.3f}\t"
                    f"tol:{tolerance:.3e}\t"
                    f"active_waypoints:{len(mirs.active_waypoints)}"
                )
            if time.time() - start_time > time_limit:
                print(f"Warning: computation exceeded time limit of {time_limit} seconds.")
                break

        final_beta = self.b_arr[len(cost_history) - 1]
        path_probabilities = []
        for i, agent in enumerate(mirs.agents):
            path_probabilities.append(
                agent.getPathAssociations(
                    final_beta,
                    mirs._local_sched(i, schedule),
                    speed[i],
                )
            )
        return (
            np.asarray(schedule_history),
            np.asarray(speed_history),
            np.asarray(cost_history),
            path_probabilities,
            np.asarray(reach_history),
            np.asarray(compute_times),
        )
