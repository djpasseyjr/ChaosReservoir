from res_experiment import *
from rescomp.lorenz_sol import *
from parameter_experiments import *
from math import floor
from scipy import integrate

""" there should only be one file that the user will have to edit,
so in main.py the user can input parameter ranges, and then parameter_experiments.py
will create individual experiment files, and bash scripts, to be run.

See the README.md for the file naming system for FNAME input
    in preparation for when we need to systematically
    read in each result pkl file exactly once (no more, no less) to one data source
    file per batch
"""
#edit USER_ID, and BATCH_NUMBER to follow file naming style guide in README.md
#USER_ID should be one of the following: ['JW','DJ','BW','JJ','IB']
USER_ID = 'JW'
BATCH_NUMBER = 1

#edit directly into function for parameters,
generate_experiments(
    FNAME = USER_ID + str(BATCH_NUMBER),
    nets_per_experiment = 1,
    orbits_per_experiment = 1,
    topology = 'barab1',
    # parameters below should be a list
    gamma_vals = [1],
    sigma_vals = [1],
    spectr_vals = [0.9],
    topo_p_vals = [None],
    ridge_alphas = [0.001],
    remove_p_list = [0.1,0.2]
)