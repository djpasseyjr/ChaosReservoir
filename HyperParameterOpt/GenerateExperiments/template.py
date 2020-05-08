from res_experiment import *
from rescomp.lorenz_sol import *
from parameter_experiments import *
from math import floor
from scipy import integrate

# i decided to change the template.py file because I took that ino and put it into the generate_experiments file
# that way I can return the results, and hopefully just store one file
# this should be the only file that need be edited to change the experiments
# these comments above can be deleted if this file is to be kept

#edit directly into function
generate_experiments(
    condensed_output_filename = 'filename',
    nets_per_experiment = 2,
    orbits_per_experiment = 2,
    topology = 'barab1',
    # parameters below should be a list
    gamma_vals = [1],
    sigma_vals = [1],
    spectr_vals = [0.9],
    topo_p_vals = [None],
    ridge_alphas = [0.001],
    remove_p_list = [0]
)
