my_inf = 1e6
from numpy import *
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.special import *
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from matplotlib import patches
import warnings
import math
from utils.my_log import my_log
from utils.penalty import penalty
from utils.return_arc_length import return_arc_length
from utils.add_block_dist import add_block_dist
from utils.line_circle_intersection import line_circle_intersection
from utils.calculate_shorter_arc import calculate_shorter_arc
from utils.get_drone_path_segments import get_drone_path_segments
from utils.normalize_drones import normalize_drones
from utils.denormalize_drones import denormalize_drones
from utils.normalize_blocks import normalize_drones
from utils.denormalize_blocks import denormalize_drones
from uav_flpo import uav_flpo


# a list of drones, each element containing two tuples,
# representing the coordinates of the initial deployment position
# the destination coordinates, and initial charge respectively

drones = [
    ((10.0, 5.0), (45.0, 50.0), 0.7),  # Long distance, high charge
    ((3.0, 40.0), (50.0, 10.0), 0.5),   # Long distance, medium charge
    ((20.0, 15.0), (35.0, 35.0), 0.6),  # Moderate distance, medium charge
    ((5.0, 30.0), (25.0, 5.0), 0.4),    # Moderate distance, low charge
    ((40.0, 45.0), (10.0, 10.0), 0.8),  # Long distance, high charge
    ((30.0, 20.0), (5.0, 35.0), 0.6),   # Moderate distance, medium charge
    ((15.0, 10.0), (40.0, 40.0), 0.4),  # Moderate distance, low charge
    ((35.0, 5.0), (10.0, 45.0), 0.5),   # Long distance, medium charge
    ((25.0, 40.0), (20.0, 10.0), 0.7),  # Moderate distance, high charge
    ((45.0, 15.0), (5.0, 20.0), 0.3)    # Long distance, low charge
]



blocks = [
    ((30.0, 30.0), 3.0),  # Large obstacle, centrally located
    ((15.0, 25.0), 1.50),    # Smaller obstacle, near a cluster of start/destination points
    ((30.0, 10.0), 2.00),
]

fcr = 25# Full Charge Range
ugv_factor = 0 # the cost factor for UGV transportation
distance = 'euclidean' # distance measure in the environment

beta_init = 1e-8 # initial beta value for the optimization.
beta_f = 100 # final beta value for the optimization
alpha = 3 # beta growth rate
purturb = 0.001 # random purturbation in optimization

env=uav_flpo(drones,4,blocks=blocks,
             ugv_factor=ugv_factor,fcr=fcr,distance=distance)
env.train(beta_init=beta_init,beta_f=beta_f,alpha=alpha,
          purturb=purturb,method='powell',verbos=1)
env.print_routs()
env.plot_routs(show_info=1,show_nums=0,save=1,show_ugv=0)
print('total cost: ',env.return_total_cost())
print('If all drones went directly to destintaion the cost would be: ',env.return_direct_cost())
