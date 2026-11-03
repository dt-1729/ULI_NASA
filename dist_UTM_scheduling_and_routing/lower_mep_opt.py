import time

import numpy as np

from higher_mep_opt import MIRSOptimizer


def default_optim_config():
    """CBF-CLF settings (same defaults as the higher-level CLI)."""
    return {
        "name": "cbf_clf",
        "dt_init": 0.01,
        "dt_min": 1e-6,
        "dt_max": 0.01,
        "Tf": 10.0,
        "gamma": 1.0,
        "alpha_s": 100.0,
        "alpha_c": 10.0,
        "alpha_r": 100.0,
        "alpha_q": 100.0,
        "p": 1.0,
        "stop_tol": 5e-6,
        "stop_tol_weight": np.array([0.0, 0.0, 1.0, 0.0]),
        "verbose": 0,
        "QP_solver_name": "OSQP",
        "qp_solver_opts": {
            "verbose": False,
            "max_iter": 100000,
            "eps_abs": 1e-3,
            "eps_rel": 1e-3,
        },
    }


def default_anneal_config():
    return {
        "log_bmin": -6.0,
        "log_bmax": 0.0,
        "nb": 15,
        "ra": 0.01,
        "rb": 0.001,
        "rw": 0.001,
        "rh": 0.01,
        "ra_rate": 2.0,
        "rb_rate": 2.0,
        "rw_rate": 2.0,
        "rh_rate": 2.0,
    }


def default_rect_cbf(tol_array):
    tol_array = np.asarray(tol_array, dtype=float)
    return {
        "mode": "rect",
        "width": 1e-3 * tol_array,
        "height": 1e-4 * tol_array,
        "gamma": 1.0,
        "width_correction_fac": 0.5,
        "height_correction_fac": 0.5,
    }


class MIRSOptimizer_fine(MIRSOptimizer):
    """CBF-CLF annealing optimizer for a subgraph-level MIRS_fine instance.

    The QP, integration and CBFs are inherited. The per-agent schedule upper bound enters the
    QP through mirs.T_upper_bound, an (n_agents, 1) column.
    """

    def _update_barrier_geometry(self, beta_index, beta, n_beta, ra, rb, rw, rh, anneal_print=False):
        mirs = self.mirs
        if mirs.ca_cbf["mode"] != "rect":
            return super()._update_barrier_geometry(
                beta_index, beta, n_beta, ra, rb, rw, rh, anneal_print
            )
        # the rect width is one number per waypoint, so anneal it up to the largest agent bound
        bound = float(np.max(mirs.agent_T_upper_bound))
        initial_width = rw**2 / (rw**2 + 1) * bound
        initial_height = rh / (rh + 1) * mirs.tolArray
        progress = beta_index / max(n_beta - 1, 1)
        mirs.ca_cbf["width"] = np.ones(mirs.tolArray.shape) * initial_width * (
            bound / initial_width
        ) ** progress
        mirs.ca_cbf["height"] = initial_height * (mirs.tolArray / initial_height) ** progress
        if anneal_print:
            print(f"w:{mirs.ca_cbf['width'][0]:.3f}\th:{mirs.ca_cbf['height'][0]:.3f}")
        return ra, rb, rw, rh

    def anneal(self, T0, V0, active_waypoints=None, time_limit=np.inf, annealPrint=False):
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
            schedule, speed, _, cost, _, tolerance, _ = self.CBF_CLF_at_beta(
                beta, schedule, speed, reach_mat
            )

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
        path_probabilities = [
            agent.getPathAssociations_v1(schedule[i], speed[i], mirs.dist_mat, final_beta)
            for i, agent in enumerate(mirs.agents)
        ]
        return (
            np.asarray(schedule_history),
            np.asarray(speed_history),
            np.asarray(cost_history),
            path_probabilities,
            np.asarray(reach_history),
            np.asarray(compute_times),
        )
